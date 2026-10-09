//! Native self-contained launcher envelope.
//!
//! ELF, PE, and Mach-O loaders ignore ordinary trailing data. Ourochronos
//! therefore builds a directly runnable launcher by copying the current
//! runtime executable and appending one validated portable package plus a
//! fixed footer. The executable runtime remains the authority: startup finds
//! and validates the embedded `OUROPK` bytes before dispatch.

use crate::package::{PackageError, PortablePackage, MAX_PACKAGE_BYTES};
use crate::portable_artifact::{
    PortableArtifact, PortableArtifactError, MAX_PORTABLE_ARTIFACT_BYTES,
};
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
    /// Embedded portable provenance or executable failed validation.
    InvalidArtifact(PortableArtifactError),
    /// Launchers require an explicit package runtime/resolution contract.
    ExpectedPackage,
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
            Self::InvalidArtifact(error) => write!(formatter, "invalid embedded artifact: {error}"),
            Self::ExpectedPackage => {
                formatter.write_str("launcher requires a portable package payload")
            }
            Self::LengthOverflow => formatter.write_str("launcher payload length does not fit u64"),
        }
    }
}

impl Error for LauncherError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidPackage(error) => Some(error),
            Self::InvalidArtifact(error) => Some(error),
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
    append_payload(runtime, &package_bytes)
}

/// Build a launcher retaining the package's portable source/evidence envelope.
/// Legacy package launchers remain readable by [`embedded_artifact`].
pub fn build_native_artifact_launcher(
    runtime: &[u8],
    artifact: &PortableArtifact,
) -> Result<Vec<u8>, LauncherError> {
    if artifact.package().is_none() {
        return Err(LauncherError::ExpectedPackage);
    }
    let payload = artifact
        .to_bytes()
        .map_err(LauncherError::InvalidArtifact)?;
    append_payload(runtime, &payload)
}

fn append_payload(runtime: &[u8], package_bytes: &[u8]) -> Result<Vec<u8>, LauncherError> {
    if runtime.len() > MAX_LAUNCHER_BYTES {
        return Err(LauncherError::TooLarge {
            size: runtime.len(),
            limit: MAX_LAUNCHER_BYTES,
        });
    }
    let base_end = embedded_range(runtime, MAX_PORTABLE_ARTIFACT_BYTES)?
        .map_or(runtime.len(), |(start, _)| start);
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
    launcher.extend_from_slice(package_bytes);
    launcher.extend_from_slice(&payload_len.to_le_bytes());
    launcher.extend_from_slice(FOOTER_MAGIC);
    Ok(launcher)
}

/// Decode a package appended to executable bytes, or return `None` for an
/// ordinary runtime executable.
pub fn embedded_package(bytes: &[u8]) -> Result<Option<PortablePackage>, LauncherError> {
    let Some((start, end)) = embedded_range(bytes, MAX_PACKAGE_BYTES)? else {
        return Ok(None);
    };
    PortablePackage::from_bytes(&bytes[start..end])
        .map(Some)
        .map_err(LauncherError::InvalidPackage)
}

/// Load either a legacy package or a provenance envelope from a launcher.
/// A bare bytecode payload is rejected: it has no package runtime contract.
pub fn embedded_artifact(bytes: &[u8]) -> Result<Option<PortableArtifact>, LauncherError> {
    let Some((start, end)) = embedded_range(bytes, MAX_PORTABLE_ARTIFACT_BYTES)? else {
        return Ok(None);
    };
    let artifact =
        PortableArtifact::from_bytes(&bytes[start..end]).map_err(LauncherError::InvalidArtifact)?;
    if artifact.package().is_none() {
        return Err(LauncherError::ExpectedPackage);
    }
    Ok(Some(artifact))
}

fn embedded_range(
    bytes: &[u8],
    payload_limit: usize,
) -> Result<Option<(usize, usize)>, LauncherError> {
    embedded_range_with_limit(bytes, payload_limit, MAX_LAUNCHER_BYTES)
}

fn embedded_range_with_limit(
    bytes: &[u8],
    payload_limit: usize,
    launcher_limit: usize,
) -> Result<Option<(usize, usize)>, LauncherError> {
    if bytes.len() > launcher_limit {
        return Err(LauncherError::TooLarge {
            size: bytes.len(),
            limit: launcher_limit,
        });
    }
    if bytes.len() < FOOTER_BYTES || &bytes[bytes.len() - FOOTER_MAGIC.len()..] != FOOTER_MAGIC {
        return Ok(None);
    }
    let length_start = bytes.len() - FOOTER_BYTES;
    let payload_bytes = u64::from_le_bytes(
        bytes[length_start..length_start + 8]
            .try_into()
            .expect("exact launcher footer width"),
    );
    if payload_bytes as u128 > payload_limit as u128 {
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
    fn total_decoder_limit_precedes_small_valid_payload() {
        let bytes = build_native_launcher(b"runtime", &package(7)).unwrap();
        assert_eq!(
            embedded_range_with_limit(&bytes, MAX_PACKAGE_BYTES, bytes.len()).unwrap(),
            embedded_range(&bytes, MAX_PACKAGE_BYTES).unwrap()
        );
        assert!(matches!(
            embedded_range_with_limit(&bytes, MAX_PACKAGE_BYTES, bytes.len() - 1),
            Err(LauncherError::TooLarge { size, limit })
                if size == bytes.len() && limit == bytes.len() - 1
        ));
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

    #[test]
    fn artifact_launchers_preserve_metadata_and_keep_package_policy() {
        let mut artifact = PortableArtifact::from_package(package(9)).unwrap();
        artifact
            .attach_evidence(
                crate::linker::VerificationArtifactKind::BytecodeReport,
                1,
                b"producer claim".to_vec(),
            )
            .unwrap();
        let legacy = build_native_launcher(b"runtime", &package(7)).unwrap();
        let launcher = build_native_artifact_launcher(&legacy, &artifact).unwrap();
        assert_eq!(
            embedded_artifact(&launcher).unwrap(),
            Some(artifact.clone())
        );
        assert_eq!(
            build_native_artifact_launcher(&launcher, &artifact).unwrap(),
            launcher
        );
        assert_eq!(
            embedded_artifact(&legacy).unwrap().unwrap().package(),
            Some(&package(7))
        );
        assert!(matches!(
            embedded_package(&launcher),
            Err(LauncherError::InvalidPackage(_))
        ));
        let bare = PortableArtifact::from_bytecode(package(9).program).unwrap();
        assert!(matches!(
            build_native_artifact_launcher(b"runtime", &bare),
            Err(LauncherError::ExpectedPackage)
        ));
        let injected = append_payload(b"runtime", &bare.to_bytes().unwrap()).unwrap();
        assert!(matches!(
            embedded_artifact(&injected),
            Err(LauncherError::ExpectedPackage)
        ));
    }
}
