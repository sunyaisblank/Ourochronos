//! Capability-scoped host adapter for selected temporal effect batches.
//!
//! Nothing here is reachable from candidate evaluation. A caller must grant
//! exact file, endpoint, process-invocation, sleep, or custom capabilities and then pass this
//! adapter to `TemporalTransaction::commit_selected_with_adapter` after an
//! explicit timeline selection. Application is preflighted and at-most-once,
//! but it is not atomic across heterogeneous host resources: a runtime failure
//! can leave an already-applied prefix, and replay returns that exact failure
//! without applying the batch again.

use crate::temporal::transaction::{CommitReceipt, CommitToken, EffectCommitAdapter, EffectIntent};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::Command;

type CustomHandler = Box<dyn FnMut(&[u8]) -> Result<(), String> + Send>;

enum BatchApplicationOutcome {
    Applying,
    Applied,
    Failed(String),
}

struct AppliedBatchRecord {
    receipt: CommitReceipt,
    effects: Vec<EffectIntent>,
    outcome: BatchApplicationOutcome,
}

struct PreparedFiles {
    handles: BTreeMap<String, File>,
    #[cfg(unix)]
    missing: BTreeMap<String, PreparedMissingFile>,
}

#[cfg(unix)]
struct PreparedMissingFile {
    parent: File,
    leaf: std::ffi::CString,
}

#[cfg(unix)]
fn same_file_identity(left: &std::fs::Metadata, right: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    left.dev() == right.dev() && left.ino() == right.ino()
}

#[cfg(not(unix))]
fn same_file_identity(_left: &std::fs::Metadata, _right: &std::fs::Metadata) -> bool {
    // Portable std exposes no stable cross-platform file ID. Exact native
    // file commits therefore fail closed on these platforms instead of
    // treating equal length/content as identity authority.
    false
}

/// Opt-in process host for a ledgered, selected batch.
///
/// Capabilities are exact identities, not prefixes: granting one file does
/// not grant its directory, and granting one endpoint or process invocation
/// does not grant another. The adapter is intentionally deny-by-default.
#[derive(Default)]
pub struct NativeEffectAdapter {
    files: BTreeSet<PathBuf>,
    endpoints: BTreeSet<String>,
    processes: BTreeSet<(String, Vec<String>)>,
    max_sleep_milliseconds: Option<u64>,
    custom: BTreeMap<String, CustomHandler>,
    applied_batches: BTreeMap<CommitToken, AppliedBatchRecord>,
}

impl NativeEffectAdapter {
    /// Construct a host adapter with no capabilities.
    pub fn new() -> Self {
        Self::default()
    }

    /// Permit writes to exactly one path.
    pub fn allow_file(&mut self, path: impl Into<PathBuf>) -> &mut Self {
        self.files.insert(path.into());
        self
    }

    /// Permit network sends to exactly one `host:port` endpoint.
    pub fn allow_endpoint(&mut self, endpoint: impl Into<String>) -> &mut Self {
        self.endpoints.insert(endpoint.into());
        self
    }

    /// Permit invocation of exactly one zero-argument process identity.
    pub fn allow_process(&mut self, program: impl Into<String>) -> &mut Self {
        self.processes.insert((program.into(), Vec::new()));
        self
    }

    /// Permit exactly one program and argument-vector identity.
    pub fn allow_process_with_arguments<I, S>(
        &mut self,
        program: impl Into<String>,
        arguments: I,
    ) -> &mut Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.processes.insert((
            program.into(),
            arguments.into_iter().map(Into::into).collect(),
        ));
        self
    }

    /// Permit selected sleeps up to and including one duration bound.
    pub fn allow_sleep(&mut self, max_milliseconds: u64) -> &mut Self {
        self.max_sleep_milliseconds = Some(max_milliseconds);
        self
    }

    /// Register one application-defined namespace.
    pub fn register_custom(
        &mut self,
        namespace: impl Into<String>,
        handler: impl FnMut(&[u8]) -> Result<(), String> + Send + 'static,
    ) -> &mut Self {
        self.custom.insert(namespace.into(), Box::new(handler));
        self
    }

    fn authorize(&self, effect: &EffectIntent) -> Result<(), String> {
        match effect {
            EffectIntent::FileWrite { path, .. } | EffectIntent::FileSetLength { path, .. } => {
                let path = Path::new(path);
                if self.files.contains(path) {
                    Ok(())
                } else {
                    Err(format!("file capability denied for '{}'", path.display()))
                }
            }
            EffectIntent::NetworkSend { endpoint, .. } => {
                if self.endpoints.contains(endpoint) {
                    Ok(())
                } else {
                    Err(format!("network capability denied for '{endpoint}'"))
                }
            }
            EffectIntent::ProcessSpawn { program, arguments } => {
                if self
                    .processes
                    .contains(&(program.clone(), arguments.clone()))
                {
                    Ok(())
                } else {
                    Err(format!(
                        "process capability denied for exact invocation {program:?} {arguments:?}"
                    ))
                }
            }
            EffectIntent::Sleep { milliseconds } => {
                let Some(maximum) = self.max_sleep_milliseconds else {
                    return Err("sleep capability denied".to_string());
                };
                if *milliseconds <= maximum {
                    Ok(())
                } else {
                    Err(format!(
                        "sleep duration {milliseconds}ms exceeds capability limit {maximum}ms"
                    ))
                }
            }
            EffectIntent::Custom { namespace, .. } => {
                if self.custom.contains_key(namespace) {
                    Ok(())
                } else {
                    Err(format!("custom capability denied for '{namespace}'"))
                }
            }
        }
    }

    fn prepare_files(&self, effects: &[EffectIntent]) -> Result<PreparedFiles, String> {
        let mut expected =
            BTreeMap::<String, crate::temporal::transaction::FrozenFileSnapshot>::new();
        for effect in effects {
            let (path, initial) = match effect {
                EffectIntent::FileWrite { path, initial, .. }
                | EffectIntent::FileSetLength { path, initial, .. } => (path, initial),
                _ => continue,
            };
            if initial.path != *path {
                return Err(format!(
                    "file precondition path {:?} does not match effect path {path:?}",
                    initial.path
                ));
            }
            if let Some(previous) = expected.insert(path.clone(), initial.clone()) {
                if previous != *initial {
                    return Err(format!(
                        "file effects disagree on initial frozen state for '{path}'"
                    ));
                }
            }
        }

        // Check every lexical path before opening any existing retained
        // handle, including missing-path preconditions. Missing targets are
        // atomically create_new'd only when their first intent is reached.
        let mut lexical_metadata = BTreeMap::new();
        for (path, initial) in &expected {
            match std::fs::symlink_metadata(path) {
                Ok(metadata) => {
                    if metadata.file_type().is_symlink() {
                        return Err(format!("file precondition rejects symlink '{path}'"));
                    }
                    if !metadata.is_file() {
                        return Err(format!("file precondition rejects non-regular '{path}'"));
                    }
                    if initial.contents.is_none() {
                        return Err(format!("file precondition expected '{path}' to be missing"));
                    }
                    lexical_metadata.insert(path.clone(), metadata);
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    if initial.contents.is_some() {
                        return Err(format!(
                            "file precondition expected regular file '{path}', but it is missing"
                        ));
                    }
                }
                Err(error) => {
                    return Err(format!(
                        "cannot inspect file precondition '{path}': {error}"
                    ));
                }
            }
        }

        let mut handles = BTreeMap::new();
        #[cfg(unix)]
        let mut missing = BTreeMap::new();
        for (path, initial) in &expected {
            let Some(expected_contents) = &initial.contents else {
                #[cfg(unix)]
                {
                    missing.insert(path.clone(), Self::prepare_missing_file(path)?);
                    continue;
                }
                #[cfg(not(unix))]
                {
                    return Err(format!(
                        "secure missing-file creation is unsupported on this platform for '{path}'"
                    ));
                }
            };
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(path)
                .map_err(|error| format!("cannot open verified file '{path}': {error}"))?;
            let opened_metadata = file
                .metadata()
                .map_err(|error| format!("cannot inspect opened file '{path}': {error}"))?;
            let current_lexical = std::fs::symlink_metadata(path)
                .map_err(|error| format!("cannot recheck file '{path}': {error}"))?;
            if current_lexical.file_type().is_symlink()
                || !current_lexical.is_file()
                || !same_file_identity(
                    lexical_metadata
                        .get(path)
                        .expect("existing path metadata was retained"),
                    &opened_metadata,
                )
                || !same_file_identity(&current_lexical, &opened_metadata)
            {
                return Err(format!("file identity changed while verifying '{path}'"));
            }
            let mut actual = Vec::new();
            let read_limit = u64::try_from(expected_contents.len())
                .unwrap_or(u64::MAX)
                .saturating_add(1);
            Read::by_ref(&mut file)
                .take(read_limit)
                .read_to_end(&mut actual)
                .map_err(|error| format!("cannot read verified file '{path}': {error}"))?;
            if actual != *expected_contents {
                return Err(format!("file precondition contents changed for '{path}'"));
            }
            file.seek(SeekFrom::Start(0))
                .map_err(|error| format!("cannot rewind verified file '{path}': {error}"))?;
            handles.insert(path.clone(), file);
        }
        Ok(PreparedFiles {
            handles,
            #[cfg(unix)]
            missing,
        })
    }

    #[cfg(unix)]
    fn prepare_missing_file(path: &str) -> Result<PreparedMissingFile, String> {
        use std::os::fd::AsRawFd;
        use std::os::unix::ffi::OsStrExt;

        let target = Path::new(path);
        let leaf = target
            .file_name()
            .ok_or_else(|| format!("missing file target has no leaf name: '{path}'"))?;
        let leaf = std::ffi::CString::new(leaf.as_bytes())
            .map_err(|_| format!("file target contains a NUL byte: '{path}'"))?;
        let parent_path = target
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let before = std::fs::symlink_metadata(parent_path).map_err(|error| {
            format!(
                "cannot inspect parent directory '{}' for missing target '{path}': {error}",
                parent_path.display()
            )
        })?;
        if before.file_type().is_symlink() || !before.is_dir() {
            return Err(format!(
                "file precondition rejects non-directory or symlink parent '{}' for '{path}'",
                parent_path.display()
            ));
        }
        let parent = File::open(parent_path).map_err(|error| {
            format!(
                "cannot retain parent directory '{}' for missing target '{path}': {error}",
                parent_path.display()
            )
        })?;
        let opened = parent.metadata().map_err(|error| {
            format!(
                "cannot inspect retained parent directory '{}' for '{path}': {error}",
                parent_path.display()
            )
        })?;
        let current = std::fs::symlink_metadata(parent_path).map_err(|error| {
            format!(
                "cannot recheck parent directory '{}' for '{path}': {error}",
                parent_path.display()
            )
        })?;
        if current.file_type().is_symlink()
            || !current.is_dir()
            || !same_file_identity(&before, &opened)
            || !same_file_identity(&current, &opened)
        {
            return Err(format!(
                "parent directory identity changed while verifying missing target '{path}'"
            ));
        }

        let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
        // SAFETY: `parent` is a live directory descriptor, `leaf` is NUL
        // terminated, and `stat` points to writable storage. `fstatat` does
        // not retain any pointer after returning.
        let status = unsafe {
            libc::fstatat(
                parent.as_raw_fd(),
                leaf.as_ptr(),
                stat.as_mut_ptr(),
                libc::AT_SYMLINK_NOFOLLOW,
            )
        };
        if status == 0 {
            return Err(format!("file precondition expected '{path}' to be missing"));
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::NotFound {
            return Err(format!(
                "cannot inspect missing target '{path}' relative to its retained parent: {error}"
            ));
        }
        Ok(PreparedMissingFile { parent, leaf })
    }

    fn retained_file<'a>(path: &str, files: &'a mut PreparedFiles) -> Result<&'a mut File, String> {
        if !files.handles.contains_key(path) {
            #[cfg(unix)]
            let file = {
                use std::os::fd::{AsRawFd, FromRawFd};

                let missing = files
                    .missing
                    .remove(path)
                    .ok_or_else(|| format!("missing retained file authority for '{path}'"))?;
                // SAFETY: `missing.parent` remains live for the call, the leaf
                // is a checked single component, and a successful descriptor
                // is transferred exactly once into `File`.
                let descriptor = unsafe {
                    libc::openat(
                        missing.parent.as_raw_fd(),
                        missing.leaf.as_ptr(),
                        libc::O_CLOEXEC
                            | libc::O_CREAT
                            | libc::O_EXCL
                            | libc::O_NOFOLLOW
                            | libc::O_RDWR,
                        0o666,
                    )
                };
                if descriptor < 0 {
                    return Err(format!(
                        "cannot create verified file '{path}': {}",
                        std::io::Error::last_os_error()
                    ));
                }
                // SAFETY: `openat` returned a new owned descriptor above.
                unsafe { File::from_raw_fd(descriptor) }
            };
            #[cfg(not(unix))]
            return Err(format!(
                "secure missing-file creation is unsupported on this platform for '{path}'"
            ));
            files.handles.insert(path.to_string(), file);
        }
        files
            .handles
            .get_mut(path)
            .ok_or_else(|| format!("missing retained file handle for '{path}'"))
    }

    fn apply_file(effect: &EffectIntent, files: &mut PreparedFiles) -> Result<(), String> {
        match effect {
            EffectIntent::FileWrite {
                path,
                offset,
                bytes,
                ..
            } => {
                let file = Self::retained_file(path, files)?;
                file.seek(SeekFrom::Start(*offset))
                    .map_err(|error| format!("cannot seek '{path}': {error}"))?;
                file.write_all(bytes)
                    .map_err(|error| format!("cannot write '{path}': {error}"))
            }
            EffectIntent::FileSetLength { path, length, .. } => Self::retained_file(path, files)?
                .set_len(*length)
                .map_err(|error| format!("cannot resize '{path}': {error}")),
            _ => Err("non-file effect reached file application".to_string()),
        }
    }

    fn apply(&mut self, effect: &EffectIntent) -> Result<(), String> {
        match effect {
            EffectIntent::FileWrite { .. } | EffectIntent::FileSetLength { .. } => {
                Err("file effect bypassed retained-handle batch application".to_string())
            }
            EffectIntent::NetworkSend { endpoint, bytes } => {
                if !self.endpoints.contains(endpoint) {
                    return Err(format!("network capability denied for '{endpoint}'"));
                }
                let mut stream = TcpStream::connect(endpoint)
                    .map_err(|error| format!("cannot connect to '{endpoint}': {error}"))?;
                stream
                    .write_all(bytes)
                    .map_err(|error| format!("cannot send to '{endpoint}': {error}"))?;
                stream
                    .flush()
                    .map_err(|error| format!("cannot flush '{endpoint}': {error}"))
            }
            EffectIntent::ProcessSpawn { program, arguments } => {
                if !self
                    .processes
                    .contains(&(program.clone(), arguments.clone()))
                {
                    return Err(format!(
                        "process capability denied for exact invocation {program:?} {arguments:?}"
                    ));
                }
                Command::new(program)
                    .args(arguments)
                    .status()
                    .map_err(|error| format!("cannot spawn '{program}': {error}"))?;
                // PROC_EXEC exposes arbitrary exit status as language data.
                // Successfully spawning and waiting applies the selected
                // effect even when that process reports a nonzero status.
                Ok(())
            }
            EffectIntent::Sleep { milliseconds } => {
                let Some(maximum) = self.max_sleep_milliseconds else {
                    return Err("sleep capability denied".to_string());
                };
                if *milliseconds > maximum {
                    return Err(format!(
                        "sleep duration {milliseconds}ms exceeds capability limit {maximum}ms"
                    ));
                }
                std::thread::sleep(std::time::Duration::from_millis(*milliseconds));
                Ok(())
            }
            EffectIntent::Custom { namespace, payload } => self
                .custom
                .get_mut(namespace)
                .ok_or_else(|| format!("custom capability denied for '{namespace}'"))?(
                payload
            ),
        }
    }
}

impl EffectCommitAdapter for NativeEffectAdapter {
    fn apply_selected(
        &mut self,
        receipt: &CommitReceipt,
        effects: &[EffectIntent],
    ) -> Result<(), String> {
        if let Some(previous) = self.applied_batches.get(&receipt.token) {
            if previous.receipt != *receipt || previous.effects != effects {
                return Err(format!(
                    "commit token {:?} was already used for a different exact receipt or effect batch",
                    receipt.token
                ));
            }
            return match &previous.outcome {
                BatchApplicationOutcome::Applied => Ok(()),
                BatchApplicationOutcome::Failed(message) => Err(message.clone()),
                BatchApplicationOutcome::Applying => {
                    Err("selected batch application is already in progress".to_string())
                }
            };
        }

        // Reject a completely unauthorized batch before its first host call.
        // The failure is still terminal for this token and cannot be retried
        // by mutating the adapter's capabilities afterward.
        if let Err(message) = effects.iter().try_for_each(|effect| self.authorize(effect)) {
            self.applied_batches.insert(
                receipt.token,
                AppliedBatchRecord {
                    receipt: receipt.clone(),
                    effects: effects.to_vec(),
                    outcome: BatchApplicationOutcome::Failed(message.clone()),
                },
            );
            return Err(message);
        }

        // Claim the idempotency identity before the first irreversible call.
        // A host failure therefore remains at-most-once on an accidental
        // replay of the same selected batch.
        self.applied_batches.insert(
            receipt.token,
            AppliedBatchRecord {
                receipt: receipt.clone(),
                effects: effects.to_vec(),
                outcome: BatchApplicationOutcome::Applying,
            },
        );
        let mut files = match self.prepare_files(effects) {
            Ok(files) => files,
            Err(message) => {
                self.applied_batches
                    .get_mut(&receipt.token)
                    .expect("batch token was claimed above")
                    .outcome = BatchApplicationOutcome::Failed(message.clone());
                return Err(message);
            }
        };
        for effect in effects {
            let result = match effect {
                EffectIntent::FileWrite { .. } | EffectIntent::FileSetLength { .. } => {
                    Self::apply_file(effect, &mut files)
                }
                _ => self.apply(effect),
            };
            if let Err(message) = result {
                self.applied_batches
                    .get_mut(&receipt.token)
                    .expect("batch token was claimed above")
                    .outcome = BatchApplicationOutcome::Failed(message.clone());
                return Err(message);
            }
        }
        for (path, file) in &mut files.handles {
            if let Err(error) = file.flush() {
                let message = format!("cannot flush '{path}': {error}");
                self.applied_batches
                    .get_mut(&receipt.token)
                    .expect("batch token was claimed above")
                    .outcome = BatchApplicationOutcome::Failed(message.clone());
                return Err(message);
            }
        }
        self.applied_batches
            .get_mut(&receipt.token)
            .expect("batch token was claimed above")
            .outcome = BatchApplicationOutcome::Applied;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Value;
    use crate::temporal::transaction::{
        CommitLog, CommitOutcome, CommitToken, TemporalTransaction, TransactionLimits,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn unique_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "ouro-native-effect-{}-{}-{name}",
            std::process::id(),
            std::thread::current().name().unwrap_or("thread")
        ))
    }

    #[test]
    fn selected_file_and_custom_effects_commit_once() {
        let path = unique_path("commit-once");
        let _ = std::fs::remove_file(&path);
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&calls);
        let mut adapter = NativeEffectAdapter::new();
        adapter
            .allow_file(path.clone())
            .register_custom("count", move |payload| {
                if payload != b"chosen" {
                    return Err("unexpected payload".to_string());
                }
                observed.fetch_add(1, Ordering::SeqCst);
                Ok(())
            });

        let mut transaction =
            TemporalTransaction::new(Vec::new(), TransactionLimits::default()).unwrap();
        let mut context = transaction.begin_candidate().unwrap();
        context
            .stage_effect(EffectIntent::FileWrite {
                path: path.display().to_string(),
                offset: 2,
                bytes: b"selected".to_vec(),
                initial: crate::temporal::transaction::FrozenFileSnapshot {
                    path: path.display().to_string(),
                    contents: None,
                },
            })
            .unwrap();
        context
            .stage_effect(EffectIntent::Custom {
                namespace: "count".to_string(),
                payload: b"chosen".to_vec(),
            })
            .unwrap();
        let candidate = context.finish(vec![(0, Value::new(1))]).unwrap();
        let timeline = transaction.stage_candidate(candidate).unwrap();
        transaction.select(timeline).unwrap();
        let mut log = CommitLog::default();

        assert!(matches!(
            transaction
                .commit_selected_with_adapter(CommitToken(91), &mut log, &mut adapter)
                .unwrap(),
            CommitOutcome::Committed(_)
        ));
        assert!(matches!(
            transaction
                .commit_selected_with_adapter(CommitToken(91), &mut log, &mut adapter)
                .unwrap(),
            CommitOutcome::AlreadyCommitted(_)
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(std::fs::read(&path).unwrap(), b"\0\0selected");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn native_adapter_denies_ungranted_capabilities() {
        let mut adapter = NativeEffectAdapter::new();
        let receipt = CommitReceipt {
            token: CommitToken(1),
            timeline: crate::temporal::transaction::TimelineId(2),
            batch_digest: 3,
            sequence: 0,
        };
        let error = adapter
            .apply_selected(
                &receipt,
                &[EffectIntent::FileWrite {
                    path: unique_path("denied").display().to_string(),
                    offset: 0,
                    bytes: vec![1],
                    initial: crate::temporal::transaction::FrozenFileSnapshot {
                        path: unique_path("denied").display().to_string(),
                        contents: None,
                    },
                }],
            )
            .unwrap_err();
        assert!(error.contains("capability denied"));
    }

    #[test]
    fn native_adapter_deduplicates_receipts_and_rejects_token_conflicts() {
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&calls);
        let mut adapter = NativeEffectAdapter::new();
        adapter.register_custom("once", move |_| {
            observed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });
        let receipt = CommitReceipt {
            token: CommitToken(44),
            timeline: crate::temporal::transaction::TimelineId(1),
            batch_digest: 0xAA,
            sequence: 0,
        };
        let effects = [EffectIntent::Custom {
            namespace: "once".to_string(),
            payload: Vec::new(),
        }];

        adapter.apply_selected(&receipt, &effects).unwrap();
        adapter.apply_selected(&receipt, &effects).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        let conflict = CommitReceipt {
            batch_digest: 0xBB,
            ..receipt
        };
        assert!(adapter
            .apply_selected(&conflict, &effects)
            .unwrap_err()
            .contains("already used"));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn adapter_digest_is_never_equality_authority() {
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&calls);
        let mut adapter = NativeEffectAdapter::new();
        adapter.register_custom("exact", move |_| {
            observed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });
        let receipt = CommitReceipt {
            token: CommitToken(441),
            timeline: crate::temporal::transaction::TimelineId(1),
            batch_digest: 0xAA,
            sequence: 0,
        };
        adapter
            .apply_selected(
                &receipt,
                &[EffectIntent::Custom {
                    namespace: "exact".to_string(),
                    payload: b"first".to_vec(),
                }],
            )
            .unwrap();

        let error = adapter
            .apply_selected(
                &receipt,
                &[EffectIntent::Custom {
                    namespace: "exact".to_string(),
                    payload: b"different despite the same supplied digest".to_vec(),
                }],
            )
            .unwrap_err();
        assert!(
            error.contains("different exact receipt or effect batch"),
            "{error}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn selected_process_nonzero_exit_is_still_an_applied_effect() {
        #[cfg(not(windows))]
        let (program, arguments) = (
            "sh".to_string(),
            vec!["-c".to_string(), "exit 7".to_string()],
        );
        #[cfg(windows)]
        let (program, arguments) = (
            "cmd".to_string(),
            vec!["/C".to_string(), "exit 7".to_string()],
        );
        let mut adapter = NativeEffectAdapter::new();
        adapter.allow_process_with_arguments(program.clone(), arguments.clone());
        let receipt = CommitReceipt {
            token: CommitToken(45),
            timeline: crate::temporal::transaction::TimelineId(1),
            batch_digest: 0xCC,
            sequence: 0,
        };
        adapter
            .apply_selected(
                &receipt,
                &[EffectIntent::ProcessSpawn { program, arguments }],
            )
            .unwrap();
    }

    #[test]
    fn failed_batch_replay_returns_the_same_failure_without_partial_reexecution() {
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&calls);
        let mut adapter = NativeEffectAdapter::new();
        adapter
            .register_custom("first", move |_| {
                observed.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
            .register_custom("fail", |_| Err("deliberate failure".to_string()));
        let receipt = CommitReceipt {
            token: CommitToken(46),
            timeline: crate::temporal::transaction::TimelineId(1),
            batch_digest: 0xDD,
            sequence: 0,
        };
        let effects = [
            EffectIntent::Custom {
                namespace: "first".to_string(),
                payload: Vec::new(),
            },
            EffectIntent::Custom {
                namespace: "fail".to_string(),
                payload: Vec::new(),
            },
        ];

        let first = adapter.apply_selected(&receipt, &effects).unwrap_err();
        let replay = adapter.apply_selected(&receipt, &effects).unwrap_err();
        assert_eq!(first, "deliberate failure");
        assert_eq!(replay, first);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn changed_file_precondition_is_rejected_before_write() {
        let path = unique_path("changed-precondition");
        std::fs::write(&path, b"frozen").unwrap();
        let effect = EffectIntent::FileWrite {
            path: path.display().to_string(),
            offset: 0,
            bytes: b"selected".to_vec(),
            initial: crate::temporal::transaction::FrozenFileSnapshot {
                path: path.display().to_string(),
                contents: Some(b"frozen".to_vec()),
            },
        };
        std::fs::write(&path, b"concurrent").unwrap();
        let mut adapter = NativeEffectAdapter::new();
        adapter.allow_file(path.clone());
        let receipt = CommitReceipt {
            token: CommitToken(47),
            timeline: crate::temporal::transaction::TimelineId(1),
            batch_digest: 0xEE,
            sequence: 0,
        };
        assert!(adapter
            .apply_selected(&receipt, &[effect])
            .unwrap_err()
            .contains("contents changed"));
        assert_eq!(std::fs::read(&path).unwrap(), b"concurrent");
        let _ = std::fs::remove_file(path);
    }

    #[cfg(unix)]
    #[test]
    fn symlink_file_precondition_is_rejected() {
        use std::os::unix::fs::symlink;
        let target = unique_path("symlink-target");
        let link = unique_path("symlink-link");
        let _ = std::fs::remove_file(&link);
        std::fs::write(&target, b"frozen").unwrap();
        symlink(&target, &link).unwrap();
        let mut adapter = NativeEffectAdapter::new();
        adapter.allow_file(link.clone());
        let receipt = CommitReceipt {
            token: CommitToken(48),
            timeline: crate::temporal::transaction::TimelineId(1),
            batch_digest: 0xEF,
            sequence: 0,
        };
        let error = adapter
            .apply_selected(
                &receipt,
                &[EffectIntent::FileWrite {
                    path: link.display().to_string(),
                    offset: 0,
                    bytes: b"selected".to_vec(),
                    initial: crate::temporal::transaction::FrozenFileSnapshot {
                        path: link.display().to_string(),
                        contents: Some(b"frozen".to_vec()),
                    },
                }],
            )
            .unwrap_err();
        assert!(error.contains("rejects symlink"), "{error}");
        assert_eq!(std::fs::read(&target).unwrap(), b"frozen");
        let _ = std::fs::remove_file(link);
        let _ = std::fs::remove_file(target);
    }

    #[test]
    fn whole_batch_capabilities_are_preflighted_before_file_write() {
        let path = unique_path("partial-capability");
        std::fs::write(&path, b"frozen").unwrap();
        let mut adapter = NativeEffectAdapter::new();
        adapter.allow_file(path.clone());
        let receipt = CommitReceipt {
            token: CommitToken(49),
            timeline: crate::temporal::transaction::TimelineId(1),
            batch_digest: 0xF0,
            sequence: 0,
        };
        let effects = [
            EffectIntent::FileWrite {
                path: path.display().to_string(),
                offset: 0,
                bytes: b"selected".to_vec(),
                initial: crate::temporal::transaction::FrozenFileSnapshot {
                    path: path.display().to_string(),
                    contents: Some(b"frozen".to_vec()),
                },
            },
            EffectIntent::NetworkSend {
                endpoint: "denied.invalid:1".to_string(),
                bytes: vec![1],
            },
        ];
        assert!(adapter
            .apply_selected(&receipt, &effects)
            .unwrap_err()
            .contains("network capability denied"));
        assert_eq!(std::fs::read(&path).unwrap(), b"frozen");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn missing_file_preparation_failure_is_claimed_and_not_replayed() {
        let parent = unique_path("absent-parent");
        let _ = std::fs::remove_dir_all(&parent);
        let path = parent.join("file");
        let mut adapter = NativeEffectAdapter::new();
        adapter.allow_file(path.clone());
        let receipt = CommitReceipt {
            token: CommitToken(50),
            timeline: crate::temporal::transaction::TimelineId(1),
            batch_digest: 0xF1,
            sequence: 0,
        };
        let effects = [EffectIntent::FileSetLength {
            path: path.display().to_string(),
            length: 0,
            initial: crate::temporal::transaction::FrozenFileSnapshot {
                path: path.display().to_string(),
                contents: None,
            },
        }];
        let first = adapter.apply_selected(&receipt, &effects).unwrap_err();
        assert!(
            first.contains("parent directory") || first.contains("cannot create verified file"),
            "{first}"
        );
        assert_eq!(
            adapter.apply_selected(&receipt, &effects).unwrap_err(),
            first
        );
        assert!(!path.exists());
    }

    #[test]
    fn missing_file_can_remain_after_later_failure_but_batch_is_not_replayed() {
        let path = unique_path("created-before-failure");
        let _ = std::fs::remove_file(&path);
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&calls);
        let mut adapter = NativeEffectAdapter::new();
        adapter
            .allow_file(path.clone())
            .register_custom("fail-after-file", move |_| {
                observed.fetch_add(1, Ordering::SeqCst);
                Err("failure after file effect".to_string())
            });
        let receipt = CommitReceipt {
            token: CommitToken(51),
            timeline: crate::temporal::transaction::TimelineId(1),
            batch_digest: 0xF2,
            sequence: 0,
        };
        let effects = [
            EffectIntent::FileSetLength {
                path: path.display().to_string(),
                length: 3,
                initial: crate::temporal::transaction::FrozenFileSnapshot {
                    path: path.display().to_string(),
                    contents: None,
                },
            },
            EffectIntent::Custom {
                namespace: "fail-after-file".to_string(),
                payload: Vec::new(),
            },
        ];

        let first = adapter.apply_selected(&receipt, &effects).unwrap_err();
        assert_eq!(first, "failure after file effect");
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 3);

        let replay = adapter.apply_selected(&receipt, &effects).unwrap_err();
        assert_eq!(replay, first);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 3);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn earlier_network_failure_does_not_create_a_later_missing_file() {
        let path = unique_path("network-fails-before-file");
        let _ = std::fs::remove_file(&path);
        let endpoint = "127.0.0.1:0";
        let mut adapter = NativeEffectAdapter::new();
        adapter.allow_endpoint(endpoint).allow_file(path.clone());
        let receipt = CommitReceipt {
            token: CommitToken(52),
            timeline: crate::temporal::transaction::TimelineId(1),
            batch_digest: 0xF3,
            sequence: 0,
        };
        let effects = [
            EffectIntent::NetworkSend {
                endpoint: endpoint.to_string(),
                bytes: vec![1],
            },
            EffectIntent::FileSetLength {
                path: path.display().to_string(),
                length: 3,
                initial: crate::temporal::transaction::FrozenFileSnapshot {
                    path: path.display().to_string(),
                    contents: None,
                },
            },
        ];

        let first = adapter.apply_selected(&receipt, &effects).unwrap_err();
        assert!(first.contains("cannot connect"), "{first}");
        assert!(!path.exists());
        assert_eq!(
            adapter.apply_selected(&receipt, &effects).unwrap_err(),
            first
        );
        assert!(!path.exists());
    }

    #[cfg(unix)]
    #[test]
    fn missing_target_creation_uses_the_preflighted_parent_handle() {
        use std::os::unix::fs::symlink;

        let parent = unique_path("pinned-parent");
        let moved = unique_path("pinned-parent-moved");
        let victim = unique_path("pinned-parent-victim");
        let _ = std::fs::remove_file(&parent);
        let _ = std::fs::remove_dir_all(&parent);
        let _ = std::fs::remove_dir_all(&moved);
        let _ = std::fs::remove_dir_all(&victim);
        std::fs::create_dir(&parent).unwrap();
        std::fs::create_dir(&victim).unwrap();
        let path = parent.join("target");

        let swap_parent = parent.clone();
        let swap_moved = moved.clone();
        let swap_victim = victim.clone();
        let mut adapter = NativeEffectAdapter::new();
        adapter
            .allow_file(path.clone())
            .register_custom("swap-parent", move |_| {
                std::fs::rename(&swap_parent, &swap_moved).map_err(|error| error.to_string())?;
                symlink(&swap_victim, &swap_parent).map_err(|error| error.to_string())?;
                Ok(())
            });
        let receipt = CommitReceipt {
            token: CommitToken(53),
            timeline: crate::temporal::transaction::TimelineId(1),
            batch_digest: 0xF4,
            sequence: 0,
        };
        adapter
            .apply_selected(
                &receipt,
                &[
                    EffectIntent::Custom {
                        namespace: "swap-parent".to_string(),
                        payload: Vec::new(),
                    },
                    EffectIntent::FileWrite {
                        path: path.display().to_string(),
                        offset: 0,
                        bytes: b"selected".to_vec(),
                        initial: crate::temporal::transaction::FrozenFileSnapshot {
                            path: path.display().to_string(),
                            contents: None,
                        },
                    },
                ],
            )
            .unwrap();

        assert_eq!(std::fs::read(moved.join("target")).unwrap(), b"selected");
        assert!(!victim.join("target").exists());
        std::fs::remove_file(&parent).unwrap();
        let _ = std::fs::remove_dir_all(moved);
        let _ = std::fs::remove_dir_all(victim);
    }
}
