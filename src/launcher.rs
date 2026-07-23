//! Native self-contained launcher envelope.
//!
//! ELF, PE, and Mach-O loaders ignore ordinary trailing data. Ourochronos
//! therefore builds a directly runnable launcher by copying the current
//! runtime executable and appending one validated portable package plus a
//! fixed footer. The executable runtime remains the authority: startup finds
//! and validates the embedded `OUROPK` bytes before dispatch.

use crate::package::{PackageError, PortablePackage, MAX_PACKAGE_BYTES};
use std::error::Error;
use std::fmt;

const FOOTER_MAGIC: &[u8; 8] = b"OUROLNCH";
const FOOTER_BYTES: usize = 16;
/// Maximum runtime-plus-package launcher accepted by the envelope helpers.
pub const MAX_LAUNCHER_BYTES: usize = 512 * 1024 * 1024;

/// Native launcher construction or decoding failure.
#[derive(Debug)]
pub enum LauncherError {
    /// Runtime or complete launcher exceeded its deterministic bound.
    TooLarge { size: usize, limit: usize },
    /// Footer length did not identify a payload within the executable.
    InvalidFooter { payload_bytes: u64 },
    /// Embedded package failed its own bounded validation.
    InvalidPackage(PackageError),
    /// An encoded length could not fit the footer.
    LengthOverflow,
}

impl fmt::Display for LauncherError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge { size, limit } => {
                write!(formatter, "launcher size {size} exceeds {limit}")
            }
            Self::InvalidFooter { payload_bytes } => write!(
                formatter,
                "launcher footer declares invalid {payload_bytes}-byte payload"
            ),
            Self::InvalidPackage(error) => write!(formatter, "invalid embedded package: {error}"),
            Self::LengthOverflow => formatter.write_str("launcher payload length does not fit u64"),
        }
    }
}

impl Error for LauncherError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidPackage(error) => Some(error),
            _ => None,
        }
    }
}

impl From<PackageError> for LauncherError {
    fn from(error: PackageError) -> Self {
        Self::InvalidPackage(error)
    }
}

/// Append a validated package to runtime executable bytes.
///
/// If `runtime` is itself an Ourochronos launcher, its old payload is removed
/// first. This makes launcher rebuilds idempotent instead of nesting packages.
pub fn build_native_launcher(
    runtime: &[u8],
    package: &PortablePackage,
) -> Result<Vec<u8>, LauncherError> {
    if runtime.len() > MAX_LAUNCHER_BYTES {
        return Err(LauncherError::TooLarge {
            size: runtime.len(),
            limit: MAX_LAUNCHER_BYTES,
        });
    }
    let package_bytes = package.to_bytes()?;
    let base_end = embedded_range(runtime)?.map_or(runtime.len(), |(start, _)| start);
    let payload_len =
        u64::try_from(package_bytes.len()).map_err(|_| LauncherError::LengthOverflow)?;
    let complete_len = base_end
        .checked_add(package_bytes.len())
        .and_then(|size| size.checked_add(FOOTER_BYTES))
        .ok_or(LauncherError::TooLarge {
            size: usize::MAX,
            limit: MAX_LAUNCHER_BYTES,
        })?;
    if complete_len > MAX_LAUNCHER_BYTES {
        return Err(LauncherError::TooLarge {
            size: complete_len,
            limit: MAX_LAUNCHER_BYTES,
        });
    }

    let mut launcher = Vec::with_capacity(complete_len);
    launcher.extend_from_slice(&runtime[..base_end]);
    launcher.extend_from_slice(&package_bytes);
    launcher.extend_from_slice(&payload_len.to_le_bytes());
    launcher.extend_from_slice(FOOTER_MAGIC);
    Ok(launcher)
}

/// Decode a package appended to executable bytes, or return `None` for an
/// ordinary runtime executable.
pub fn embedded_package(bytes: &[u8]) -> Result<Option<PortablePackage>, LauncherError> {
    let Some((start, end)) = embedded_range(bytes)? else {
        return Ok(None);
    };
    PortablePackage::from_bytes(&bytes[start..end])
        .map(Some)
        .map_err(LauncherError::InvalidPackage)
}

fn embedded_range(bytes: &[u8]) -> Result<Option<(usize, usize)>, LauncherError> {
    if bytes.len() < FOOTER_BYTES || &bytes[bytes.len() - FOOTER_MAGIC.len()..] != FOOTER_MAGIC {
        return Ok(None);
    }
    let length_start = bytes.len() - FOOTER_BYTES;
    let payload_bytes = u64::from_le_bytes(
        bytes[length_start..length_start + 8]
            .try_into()
            .expect("exact launcher footer width"),
    );
    if payload_bytes as u128 > MAX_PACKAGE_BYTES as u128 {
        return Err(LauncherError::InvalidFooter { payload_bytes });
    }
    let payload_len = usize::try_from(payload_bytes)
        .map_err(|_| LauncherError::InvalidFooter { payload_bytes })?;
    let start = length_start
        .checked_sub(payload_len)
        .ok_or(LauncherError::InvalidFooter { payload_bytes })?;
    Ok(Some((start, length_start)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Program, Stmt};
    use crate::bytecode::BytecodeProgram;
    use crate::core::Value;
    use crate::hir::HirProgram;
    use crate::package::PackageManifest;

    fn package(word: u64) -> PortablePackage {
        let mut source = Program::new();
        source.body = vec![Stmt::Push(Value::new(word))];
        let bytecode = BytecodeProgram::compile(&HirProgram::resolve(&source).unwrap()).unwrap();
        PortablePackage::new(PackageManifest::with_memory("launcher-test", 4), bytecode).unwrap()
    }

    #[test]
    fn deterministic_envelope_round_trips_and_replaces_an_old_payload() {
        let runtime = b"not-a-real-executable";
        let first = build_native_launcher(runtime, &package(7)).unwrap();
        assert_eq!(embedded_package(&first).unwrap(), Some(package(7)));
        assert_eq!(build_native_launcher(runtime, &package(7)).unwrap(), first);

        let replaced = build_native_launcher(&first, &package(11)).unwrap();
        assert_eq!(embedded_package(&replaced).unwrap(), Some(package(11)));
        assert_eq!(&replaced[..runtime.len()], runtime);
    }

    #[test]
    fn ordinary_runtime_and_malformed_footer_are_distinct() {
        assert!(embedded_package(b"ordinary executable").unwrap().is_none());

        let mut malformed = b"runtime".to_vec();
        malformed.extend_from_slice(&(MAX_PACKAGE_BYTES as u64 + 1).to_le_bytes());
        malformed.extend_from_slice(FOOTER_MAGIC);
        assert!(matches!(
            embedded_package(&malformed),
            Err(LauncherError::InvalidFooter { .. })
        ));
    }

    #[test]
    fn corrupt_embedded_package_is_rejected() {
        let mut launcher = build_native_launcher(b"runtime", &package(9)).unwrap();
        launcher[7] ^= 0xff;
        assert!(matches!(
            embedded_package(&launcher),
            Err(LauncherError::InvalidPackage(_))
        ));
    }
}
