//! Source-file ownership and exact source-location handling.
//!
//! All ranges in this module are half-open UTF-8 byte ranges.  Converting a
//! byte offset to a human-readable position therefore requires the offset to
//! fall on a UTF-8 character boundary.  Human-readable lines and columns are
//! one-based and columns count Unicode scalar values; LSP positions are
//! zero-based and their character component counts UTF-16 code units.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::string::FromUtf8Error;

/// Maximum number of bytes accepted from one disk-backed source file.
///
/// The loader enforces this limit both from file metadata and while reading,
/// because metadata is not authoritative for every kind of filesystem object.
pub const MAX_SOURCE_FILE_BYTES: usize = 8 * 1024 * 1024;

/// Stable identity of a source stored in a [`SourceManager`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceId(usize);

impl SourceId {
    /// Constructs a source identity from its raw manager index.
    ///
    /// A constructed ID is not necessarily present in a particular manager;
    /// manager operations validate it and return [`SourceError::UnknownSource`]
    /// when it is not present.
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the manager index represented by this identity.
    pub const fn index(self) -> usize {
        self.0
    }
}

impl fmt::Display for SourceId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "source {}", self.0)
    }
}

/// A half-open UTF-8 byte range: `start..end`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct TextRange {
    /// Inclusive starting byte offset.
    pub start: usize,
    /// Exclusive ending byte offset.
    pub end: usize,
}

impl TextRange {
    /// Constructs a range.  Use [`SourceManager::checked_span`] before using a
    /// range to index source text.
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Returns the byte length, or zero for a reversed (invalid) range.
    pub const fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Returns whether this range contains no bytes.
    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }

    /// Returns whether `offset` is contained in this half-open range.
    pub const fn contains(self, offset: usize) -> bool {
        self.start <= offset && offset < self.end
    }
}

impl From<std::ops::Range<usize>> for TextRange {
    fn from(range: std::ops::Range<usize>) -> Self {
        Self::new(range.start, range.end)
    }
}

impl From<TextRange> for std::ops::Range<usize> {
    fn from(range: TextRange) -> Self {
        range.start..range.end
    }
}

/// A byte range tied to one retained source file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceSpan {
    /// Source containing the range.
    pub source: SourceId,
    /// Half-open UTF-8 byte range within the source.
    pub range: TextRange,
}

impl SourceSpan {
    /// Constructs a span without validating it against a manager.
    ///
    /// Use [`SourceManager::checked_span`] when accepting offsets that have not
    /// already been validated.
    pub const fn new(source: SourceId, range: TextRange) -> Self {
        Self { source, range }
    }
}

/// A one-based position intended for diagnostics and command-line output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LineColumn {
    /// One-based line number.
    pub line: usize,
    /// One-based column measured in Unicode scalar values.
    pub column: usize,
}

/// A zero-based Language Server Protocol position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LspPosition {
    /// Zero-based line number.
    pub line: u32,
    /// Zero-based UTF-16 code-unit offset within the line.
    pub character: u32,
}

/// Exact UTF-8 source text and its precomputed line index.
#[derive(Debug, Clone)]
pub struct SourceFile {
    id: SourceId,
    canonical_path: Option<PathBuf>,
    display_name: String,
    text: String,
    line_starts: Vec<usize>,
}

impl SourceFile {
    fn new(
        id: SourceId,
        canonical_path: Option<PathBuf>,
        display_name: String,
        text: String,
    ) -> Self {
        let mut line_starts = vec![0];
        line_starts.extend(
            text.bytes()
                .enumerate()
                .filter_map(|(offset, byte)| (byte == b'\n').then_some(offset + 1)),
        );

        Self {
            id,
            canonical_path,
            display_name,
            text,
            line_starts,
        }
    }

    /// Returns this file's stable manager identity.
    pub const fn id(&self) -> SourceId {
        self.id
    }

    /// Returns the canonical path for disk-backed sources.
    pub fn canonical_path(&self) -> Option<&Path> {
        self.canonical_path.as_deref()
    }

    /// Returns the name suitable for diagnostics and user interfaces.
    pub fn display_name(&self) -> &str {
        &self.display_name
    }

    /// Returns the exact source text, with no newline, BOM, or whitespace
    /// normalization.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Returns the byte offset at which every line begins.
    ///
    /// The first element is always zero.  A source ending in `\n` has a final
    /// line start equal to `text().len()`.
    pub fn line_starts(&self) -> &[usize] {
        &self.line_starts
    }

    /// Converts a UTF-8 byte offset to a one-based line and scalar-value
    /// column.
    pub fn line_column(&self, offset: usize) -> Result<LineColumn, SourceError> {
        self.validate_offset(offset)?;
        let line_index = self.line_index(offset);
        let line_start = self.line_starts[line_index];
        let column = self.text[line_start..offset].chars().count() + 1;

        Ok(LineColumn {
            line: line_index + 1,
            column,
        })
    }

    /// Alias for [`SourceFile::line_column`] that makes the input unit
    /// explicit at call sites.
    pub fn offset_to_line_column(&self, offset: usize) -> Result<LineColumn, SourceError> {
        self.line_column(offset)
    }

    /// Converts a UTF-8 byte offset to a zero-based LSP position whose
    /// character component is measured in UTF-16 code units.
    pub fn lsp_position(&self, offset: usize) -> Result<LspPosition, SourceError> {
        self.validate_offset(offset)?;
        let line_index = self.line_index(offset);
        let line_start = self.line_starts[line_index];
        let utf16_column = self.text[line_start..offset].encode_utf16().count();
        let line = u32::try_from(line_index).map_err(|_| SourceError::PositionOverflow {
            source: self.id,
            offset,
        })?;
        let character = u32::try_from(utf16_column).map_err(|_| SourceError::PositionOverflow {
            source: self.id,
            offset,
        })?;

        Ok(LspPosition { line, character })
    }

    /// Alias for [`SourceFile::lsp_position`] that makes the input unit
    /// explicit at call sites.
    pub fn offset_to_lsp_position(&self, offset: usize) -> Result<LspPosition, SourceError> {
        self.lsp_position(offset)
    }

    /// Validates and slices a byte range from this file.
    pub fn slice(&self, range: TextRange) -> Result<&str, SourceError> {
        self.validate_range(range)?;
        Ok(&self.text[range.start..range.end])
    }

    fn line_index(&self, offset: usize) -> usize {
        match self.line_starts.binary_search(&offset) {
            Ok(line) => line,
            Err(next_line) => next_line - 1,
        }
    }

    fn validate_offset(&self, offset: usize) -> Result<(), SourceError> {
        if offset > self.text.len() {
            return Err(SourceError::OffsetOutOfBounds {
                source: self.id,
                offset,
                text_len: self.text.len(),
            });
        }
        if !self.text.is_char_boundary(offset) {
            return Err(SourceError::NotCharBoundary {
                source: self.id,
                offset,
            });
        }
        Ok(())
    }

    fn validate_range(&self, range: TextRange) -> Result<(), SourceError> {
        if range.start > range.end || range.end > self.text.len() {
            return Err(SourceError::InvalidRange {
                source: self.id,
                range,
                text_len: self.text.len(),
            });
        }

        for offset in [range.start, range.end] {
            if !self.text.is_char_boundary(offset) {
                return Err(SourceError::NotCharBoundary {
                    source: self.id,
                    offset,
                });
            }
        }
        Ok(())
    }
}

/// Owns source files and guarantees canonical-path deduplication.
#[derive(Debug, Default)]
pub struct SourceManager {
    sources: Vec<SourceFile>,
    paths: HashMap<PathBuf, SourceId>,
}

impl SourceManager {
    /// Constructs an empty manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of retained source files.
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// Returns whether this manager contains no source files.
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// Inserts an in-memory source.  Virtual sources deliberately have no
    /// canonical path and are not deduplicated by display name.
    pub fn add_virtual(
        &mut self,
        display_name: impl Into<String>,
        text: impl Into<String>,
    ) -> SourceId {
        let id = SourceId::new(self.sources.len());
        self.sources
            .push(SourceFile::new(id, None, display_name.into(), text.into()));
        id
    }

    /// Alias for [`SourceManager::add_virtual`].
    pub fn add_virtual_source(
        &mut self,
        display_name: impl Into<String>,
        text: impl Into<String>,
    ) -> SourceId {
        self.add_virtual(display_name, text)
    }

    /// Canonicalizes and loads a UTF-8 file.
    ///
    /// Loading the same filesystem object through different relative paths or
    /// symlinks returns the already assigned ID and does not reread the file.
    pub fn load_file(&mut self, path: impl AsRef<Path>) -> Result<SourceId, SourceError> {
        let requested_path = path.as_ref();
        let canonical_path = fs::canonicalize(requested_path).map_err(|error| SourceError::Io {
            path: requested_path.to_path_buf(),
            operation: IoOperation::Canonicalize,
            error,
        })?;
        self.load_canonical_file(canonical_path)
    }

    /// Canonicalizes a disk-backed source identity while retaining caller-
    /// supplied text for that file. Imports still resolve relative to the
    /// canonical path, so editor overlays do not acquire virtual-source/CWD
    /// semantics.
    pub(crate) fn load_file_overlay(
        &mut self,
        path: impl AsRef<Path>,
        text: impl Into<String>,
    ) -> Result<SourceId, SourceError> {
        let requested_path = path.as_ref();
        let canonical_path = fs::canonicalize(requested_path).map_err(|error| SourceError::Io {
            path: requested_path.to_path_buf(),
            operation: IoOperation::Canonicalize,
            error,
        })?;
        if let Some(id) = self.paths.get(&canonical_path) {
            return Ok(*id);
        }
        let text = text.into();
        if text.len() > MAX_SOURCE_FILE_BYTES {
            return Err(SourceError::SourceTooLarge {
                path: canonical_path,
                limit_bytes: MAX_SOURCE_FILE_BYTES,
                observed_bytes: text.len() as u64,
            });
        }
        let id = SourceId::new(self.sources.len());
        let display_name = canonical_path.display().to_string();
        self.sources.push(SourceFile::new(
            id,
            Some(canonical_path.clone()),
            display_name,
            text,
        ));
        self.paths.insert(canonical_path, id);
        Ok(id)
    }

    /// Resolves an import relative to its importing disk-backed source and
    /// returns the target's canonical path without loading it.
    pub fn resolve_import_path(
        &self,
        importer: SourceId,
        import: impl AsRef<Path>,
    ) -> Result<PathBuf, SourceError> {
        let importing_file = self.get(importer)?;
        if import.as_ref().is_absolute() {
            return fs::canonicalize(import.as_ref()).map_err(|error| SourceError::Io {
                path: import.as_ref().to_path_buf(),
                operation: IoOperation::Canonicalize,
                error,
            });
        }

        let importer_path =
            importing_file
                .canonical_path()
                .ok_or(SourceError::RelativeImportFromVirtual {
                    source: importer,
                    import: import.as_ref().to_path_buf(),
                })?;
        let parent = importer_path
            .parent()
            .ok_or_else(|| SourceError::MissingParentDirectory {
                source: importer,
                path: importer_path.to_path_buf(),
            })?;
        let requested_path = parent.join(import.as_ref());

        fs::canonicalize(&requested_path).map_err(|error| SourceError::Io {
            path: requested_path,
            operation: IoOperation::Canonicalize,
            error,
        })
    }

    /// Resolves and loads an import relative to its importing source.
    pub fn load_import(
        &mut self,
        importer: SourceId,
        import: impl AsRef<Path>,
    ) -> Result<SourceId, SourceError> {
        let canonical_path = self.resolve_import_path(importer, import)?;
        self.load_canonical_file(canonical_path)
    }

    /// Alias for [`SourceManager::load_import`].
    pub fn load_relative(
        &mut self,
        importer: SourceId,
        import: impl AsRef<Path>,
    ) -> Result<SourceId, SourceError> {
        self.load_import(importer, import)
    }

    /// Looks up a retained source by identity.
    pub fn get(&self, source: SourceId) -> Result<&SourceFile, SourceError> {
        self.sources
            .get(source.index())
            .ok_or(SourceError::UnknownSource { source })
    }

    /// Alias for [`SourceManager::get`].
    pub fn source(&self, source: SourceId) -> Result<&SourceFile, SourceError> {
        self.get(source)
    }

    /// Validates a range and ties it to a retained source.
    pub fn checked_span(
        &self,
        source: SourceId,
        range: impl Into<TextRange>,
    ) -> Result<SourceSpan, SourceError> {
        let range = range.into();
        self.get(source)?.validate_range(range)?;
        Ok(SourceSpan::new(source, range))
    }

    /// Constructs and validates a source span from explicit byte offsets.
    pub fn span(
        &self,
        source: SourceId,
        start: usize,
        end: usize,
    ) -> Result<SourceSpan, SourceError> {
        self.checked_span(source, TextRange::new(start, end))
    }

    /// Validates and returns the exact source text covered by a span.
    pub fn slice(&self, span: SourceSpan) -> Result<&str, SourceError> {
        self.get(span.source)?.slice(span.range)
    }

    /// Alias for [`SourceManager::slice`].
    pub fn span_text(&self, span: SourceSpan) -> Result<&str, SourceError> {
        self.slice(span)
    }

    pub(crate) fn canonical_source_id(&self, canonical_path: &Path) -> Option<SourceId> {
        self.paths.get(canonical_path).copied()
    }

    pub(crate) fn load_canonical_file(
        &mut self,
        canonical_path: PathBuf,
    ) -> Result<SourceId, SourceError> {
        if let Some(id) = self.paths.get(&canonical_path) {
            return Ok(*id);
        }

        let mut file = fs::File::open(&canonical_path).map_err(|error| SourceError::Io {
            path: canonical_path.clone(),
            operation: IoOperation::Read,
            error,
        })?;
        let metadata = file.metadata().map_err(|error| SourceError::Io {
            path: canonical_path.clone(),
            operation: IoOperation::Read,
            error,
        })?;
        if metadata.len() > MAX_SOURCE_FILE_BYTES as u64 {
            return Err(SourceError::SourceTooLarge {
                path: canonical_path,
                limit_bytes: MAX_SOURCE_FILE_BYTES,
                observed_bytes: metadata.len(),
            });
        }

        let bounded_capacity = usize::try_from(metadata.len())
            .unwrap_or(MAX_SOURCE_FILE_BYTES)
            .min(MAX_SOURCE_FILE_BYTES)
            .saturating_add(1);
        let mut bytes = Vec::with_capacity(bounded_capacity);
        file.by_ref()
            .take((MAX_SOURCE_FILE_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|error| SourceError::Io {
                path: canonical_path.clone(),
                operation: IoOperation::Read,
                error,
            })?;
        if bytes.len() > MAX_SOURCE_FILE_BYTES {
            return Err(SourceError::SourceTooLarge {
                path: canonical_path,
                limit_bytes: MAX_SOURCE_FILE_BYTES,
                observed_bytes: bytes.len() as u64,
            });
        }
        let text = String::from_utf8(bytes).map_err(|error| SourceError::InvalidUtf8 {
            path: canonical_path.clone(),
            error,
        })?;
        let id = SourceId::new(self.sources.len());
        let display_name = canonical_path.display().to_string();

        self.sources.push(SourceFile::new(
            id,
            Some(canonical_path.clone()),
            display_name,
            text,
        ));
        self.paths.insert(canonical_path, id);
        Ok(id)
    }
}

/// Filesystem operation associated with an I/O failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IoOperation {
    /// Canonicalizing a requested source path.
    Canonicalize,
    /// Reading a canonical source file.
    Read,
}

impl fmt::Display for IoOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Canonicalize => formatter.write_str("canonicalize"),
            Self::Read => formatter.write_str("read"),
        }
    }
}

/// Structured failures produced while retaining or locating source text.
#[derive(Debug)]
pub enum SourceError {
    /// A manager operation received an identity it does not own.
    UnknownSource { source: SourceId },
    /// A single byte position exceeds the source length.
    OffsetOutOfBounds {
        source: SourceId,
        offset: usize,
        text_len: usize,
    },
    /// A range is reversed or extends past the source length.
    InvalidRange {
        source: SourceId,
        range: TextRange,
        text_len: usize,
    },
    /// A byte position splits a UTF-8 code point.
    NotCharBoundary { source: SourceId, offset: usize },
    /// A position cannot be represented by the LSP's 32-bit fields.
    PositionOverflow { source: SourceId, offset: usize },
    /// A relative import was requested from an in-memory source with no base
    /// directory.
    RelativeImportFromVirtual { source: SourceId, import: PathBuf },
    /// The importing file's canonical path has no parent directory.
    MissingParentDirectory { source: SourceId, path: PathBuf },
    /// A filesystem operation failed.
    Io {
        path: PathBuf,
        operation: IoOperation,
        error: io::Error,
    },
    /// A disk-backed source was not valid UTF-8.
    InvalidUtf8 { path: PathBuf, error: FromUtf8Error },
    /// A disk-backed source exceeded the bounded input size.
    SourceTooLarge {
        path: PathBuf,
        limit_bytes: usize,
        observed_bytes: u64,
    },
}

impl fmt::Display for SourceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownSource { source } => write!(formatter, "unknown {source}"),
            Self::OffsetOutOfBounds {
                source,
                offset,
                text_len,
            } => write!(
                formatter,
                "byte offset {offset} is outside {source} (length {text_len})"
            ),
            Self::InvalidRange {
                source,
                range,
                text_len,
            } => write!(
                formatter,
                "byte range {}..{} is invalid for {source} (length {text_len})",
                range.start, range.end
            ),
            Self::NotCharBoundary { source, offset } => {
                write!(
                    formatter,
                    "byte offset {offset} splits a UTF-8 character in {source}"
                )
            }
            Self::PositionOverflow { source, offset } => write!(
                formatter,
                "position at byte offset {offset} in {source} exceeds the LSP limit"
            ),
            Self::RelativeImportFromVirtual { source, import } => write!(
                formatter,
                "cannot resolve import {:?} relative to virtual {source}",
                import
            ),
            Self::MissingParentDirectory { source, path } => write!(
                formatter,
                "cannot determine an import directory for {source} at {:?}",
                path
            ),
            Self::Io {
                path,
                operation,
                error,
            } => write!(formatter, "failed to {operation} {:?}: {error}", path),
            Self::InvalidUtf8 { path, error } => {
                write!(
                    formatter,
                    "source file {:?} is not valid UTF-8: {error}",
                    path
                )
            }
            Self::SourceTooLarge {
                path,
                limit_bytes,
                observed_bytes,
            } => write!(
                formatter,
                "source file {:?} exceeds the {limit_bytes}-byte limit (observed at least {observed_bytes} bytes)",
                path
            ),
        }
    }
}

impl Error for SourceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { error, .. } => Some(error),
            Self::InvalidUtf8 { error, .. } => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEMP_DIR: AtomicUsize = AtomicUsize::new(0);

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let sequence = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "ourochronos-source-tests-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir_all(&path).expect("create temporary source directory");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn ranges_are_half_open() {
        let range = TextRange::new(2, 5);
        assert_eq!(range.len(), 3);
        assert!(!range.is_empty());
        assert!(range.contains(2));
        assert!(range.contains(4));
        assert!(!range.contains(5));
    }

    #[test]
    fn virtual_source_retains_exact_text_and_line_starts() {
        let exact = "\u{feff}first\r\nβeta\n";
        let mut manager = SourceManager::new();
        let id = manager.add_virtual("memory://exact", exact);
        let source = manager.get(id).unwrap();

        assert_eq!(source.id(), id);
        assert_eq!(source.canonical_path(), None);
        assert_eq!(source.display_name(), "memory://exact");
        assert_eq!(source.text(), exact);
        assert_eq!(source.line_starts(), &[0, 10, 16]);
    }

    #[test]
    fn diagnostic_columns_count_unicode_scalars_and_reject_split_code_points() {
        let mut manager = SourceManager::new();
        let id = manager.add_virtual("unicode", "aé\n😀z");
        let source = manager.get(id).unwrap();

        assert_eq!(
            source.line_column(0).unwrap(),
            LineColumn { line: 1, column: 1 }
        );
        assert_eq!(
            source.line_column(3).unwrap(),
            LineColumn { line: 1, column: 3 }
        );
        assert_eq!(
            source.line_column(4).unwrap(),
            LineColumn { line: 2, column: 1 }
        );
        assert_eq!(
            source.line_column(8).unwrap(),
            LineColumn { line: 2, column: 2 }
        );
        assert_eq!(
            source.line_column(9).unwrap(),
            LineColumn { line: 2, column: 3 }
        );
        assert!(matches!(
            source.line_column(2),
            Err(SourceError::NotCharBoundary { source, offset: 2 }) if source == id
        ));
        assert!(matches!(
            source.line_column(10),
            Err(SourceError::OffsetOutOfBounds { source, offset: 10, text_len: 9 })
                if source == id
        ));
    }

    #[test]
    fn lsp_columns_count_utf16_code_units() {
        let mut manager = SourceManager::new();
        let id = manager.add_virtual("unicode", "aé\n😀z");
        let source = manager.get(id).unwrap();

        assert_eq!(
            source.lsp_position(0).unwrap(),
            LspPosition {
                line: 0,
                character: 0
            }
        );
        assert_eq!(
            source.lsp_position(3).unwrap(),
            LspPosition {
                line: 0,
                character: 2
            }
        );
        assert_eq!(
            source.lsp_position(4).unwrap(),
            LspPosition {
                line: 1,
                character: 0
            }
        );
        assert_eq!(
            source.lsp_position(8).unwrap(),
            LspPosition {
                line: 1,
                character: 2
            }
        );
        assert_eq!(
            source.lsp_position(9).unwrap(),
            LspPosition {
                line: 1,
                character: 3
            }
        );
    }

    #[test]
    fn canonical_file_loading_deduplicates_equivalent_paths() {
        let temp = TempDir::new();
        let source_path = temp.path().join("module.ouro");
        fs::write(&source_path, "1 2 +\n").unwrap();

        let mut manager = SourceManager::new();
        let first = manager.load_file(&source_path).unwrap();
        let second = manager
            .load_file(temp.path().join(".").join("module.ouro"))
            .unwrap();

        assert_eq!(first, second);
        assert_eq!(manager.len(), 1);
        assert_eq!(manager.get(first).unwrap().text(), "1 2 +\n");
        assert_eq!(
            manager.get(first).unwrap().canonical_path(),
            Some(fs::canonicalize(source_path).unwrap().as_path())
        );
    }

    #[test]
    fn imports_resolve_relative_to_the_importing_file() {
        let temp = TempDir::new();
        let package = temp.path().join("package");
        fs::create_dir(&package).unwrap();
        let main_path = package.join("main.ouro");
        let library_path = temp.path().join("library.ouro");
        fs::write(&main_path, "IMPORT ../library.ouro\n").unwrap();
        fs::write(&library_path, "41 1 +\n").unwrap();

        let mut manager = SourceManager::new();
        let main = manager.load_file(&main_path).unwrap();
        let resolved = manager
            .resolve_import_path(main, "../library.ouro")
            .unwrap();
        let library = manager.load_import(main, "../library.ouro").unwrap();
        let direct = manager.load_file(&library_path).unwrap();

        assert_eq!(resolved, fs::canonicalize(&library_path).unwrap());
        assert_eq!(library, direct);
        assert_eq!(manager.len(), 2);
        assert_eq!(manager.get(library).unwrap().text(), "41 1 +\n");
    }

    #[test]
    fn virtual_sources_cannot_supply_a_relative_import_base() {
        let mut manager = SourceManager::new();
        let virtual_id = manager.add_virtual("repl", "");

        assert!(matches!(
            manager.resolve_import_path(virtual_id, "module.ouro"),
            Err(SourceError::RelativeImportFromVirtual { source, import })
                if source == virtual_id && import == Path::new("module.ouro")
        ));
    }

    #[test]
    fn checked_spans_reject_unknown_sources_invalid_ranges_and_utf8_splits() {
        let mut manager = SourceManager::new();
        let id = manager.add_virtual("span", "aéb");

        assert!(matches!(
            manager.checked_span(SourceId::new(999), 0..0),
            Err(SourceError::UnknownSource { source }) if source == SourceId::new(999)
        ));
        assert!(matches!(
            manager.checked_span(id, TextRange::new(3, 2)),
            Err(SourceError::InvalidRange { source, range, text_len: 4 })
                if source == id && range == TextRange::new(3, 2)
        ));
        assert!(matches!(
            manager.checked_span(id, 0..5),
            Err(SourceError::InvalidRange { source, range, text_len: 4 })
                if source == id && range == TextRange::new(0, 5)
        ));
        assert!(matches!(
            manager.checked_span(id, 1..2),
            Err(SourceError::NotCharBoundary { source, offset: 2 }) if source == id
        ));

        let valid = manager.checked_span(id, 1..3).unwrap();
        assert_eq!(manager.slice(valid).unwrap(), "é");

        let fabricated = SourceSpan::new(id, TextRange::new(2, 3));
        assert!(matches!(
            manager.slice(fabricated),
            Err(SourceError::NotCharBoundary { source, offset: 2 }) if source == id
        ));
    }

    #[test]
    fn invalid_utf8_is_reported_without_registering_a_source() {
        let temp = TempDir::new();
        let path = temp.path().join("invalid.ouro");
        fs::write(&path, [0xff, 0xfe]).unwrap();
        let mut manager = SourceManager::new();

        assert!(matches!(
            manager.load_file(&path),
            Err(SourceError::InvalidUtf8 { path: error_path, .. })
                if error_path == fs::canonicalize(&path).unwrap()
        ));
        assert!(manager.is_empty());
    }

    #[test]
    fn oversized_disk_source_is_rejected_from_metadata_before_reading() {
        let temp = TempDir::new();
        let path = temp.path().join("oversized.ouro");
        let file = fs::File::create(&path).unwrap();
        file.set_len((MAX_SOURCE_FILE_BYTES + 1) as u64).unwrap();
        drop(file);
        let canonical = fs::canonicalize(&path).unwrap();
        let mut manager = SourceManager::new();

        let error = manager.load_file(&path).unwrap_err();
        assert!(matches!(
            &error,
            SourceError::SourceTooLarge {
                path: error_path,
                limit_bytes: MAX_SOURCE_FILE_BYTES,
                observed_bytes,
            } if error_path == &canonical && *observed_bytes == (MAX_SOURCE_FILE_BYTES + 1) as u64
        ));
        assert_eq!(
            error.to_string(),
            format!(
                "source file {:?} exceeds the {}-byte limit (observed at least {} bytes)",
                canonical,
                MAX_SOURCE_FILE_BYTES,
                MAX_SOURCE_FILE_BYTES + 1
            )
        );
        assert!(manager.is_empty());
    }
}
