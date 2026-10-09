//! Portable source provenance around the unchanged bytecode/package formats.
//!
//! SHA-256 binds this envelope's bytes and the original executable bytes. It
//! detects corruption and stale associations; it does not authenticate the
//! producer, prove compiler correctness, or validate an opaque proof payload.
//! Source names are display labels and never authorize filesystem access.

use crate::bytecode::{BytecodeError, BytecodeProgram, MAX_ARTIFACT_BYTES};
use crate::bytecode_verifier::verify_default;
use crate::linker::{LinkedProgram, ObjectSourceFile, VerificationArtifactKind};
use crate::package::{
    PackageError, PortablePackage, CURRENT_RUNTIME_ABI, MAX_PACKAGE_BYTES, MAX_PACKAGE_NAME_BYTES,
    MAX_PACKAGE_WITNESS_CELLS,
};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fmt;

const MAGIC: &[u8; 8] = b"OUROPA\0\0";
const BYTECODE_MAGIC: &[u8; 8] = b"OUROBC\0\0";
const PACKAGE_MAGIC: &[u8; 8] = b"OUROPK\0\0";
const FLAG_PROVENANCE: u16 = 1;
const HEADER_BYTES: usize = 24;
const CHECKSUM_BYTES: usize = 32;

pub const PORTABLE_ARTIFACT_VERSION: u16 = 1;
pub const MAX_PORTABLE_PROVENANCE_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_PORTABLE_SOURCE_FILES: usize = 100_000;
pub const MAX_PORTABLE_SOURCE_NAME_BYTES: usize = 16 * 1024;
pub const MAX_PORTABLE_EVIDENCE_BYTES: usize = 1024 * 1024;
pub const MAX_PORTABLE_ARTIFACT_BYTES: usize =
    MAX_PACKAGE_BYTES + MAX_PORTABLE_PROVENANCE_BYTES + HEADER_BYTES + CHECKSUM_BYTES;

/// The sole executable authority inside an envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PortableArtifactPayload {
    Bytecode(BytecodeProgram),
    Package(PortablePackage),
}

/// A producer's opaque evidence claim, bound to the final executable bytes.
/// Its interpretation and semantic verification belong to the evidence kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortableEvidence {
    pub program_digest: [u8; 32],
    pub kind: VerificationArtifactKind,
    pub format_version: u16,
    pub payload: Vec<u8>,
}

/// The original executable binding and linked source identities.
///
/// Source digests retain the legacy object's FNV-1a producer claims. They do
/// not become collision-resistant source attestations merely by being inside
/// this SHA-256-protected envelope. Exact source text is not embedded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortableProvenance {
    pub program_digest: [u8; 32],
    pub source_files: Vec<ObjectSourceFile>,
    pub evidence: Option<PortableEvidence>,
}

/// Availability of nonsynthetic linked source-manifest claims, not a claim
/// that source text is available or that the producer has been authenticated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceProvenanceStatus {
    Unavailable,
    ManifestAvailable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortableArtifact {
    pub payload: PortableArtifactPayload,
    pub provenance: Option<PortableProvenance>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PortableArtifactError {
    BadMagic,
    UnsupportedVersion(u16),
    UnsupportedFlags(u16),
    UnsupportedKind(u8),
    InvalidReserved,
    LimitExceeded {
        what: &'static str,
        count: usize,
        limit: usize,
    },
    Truncated,
    TrailingBytes,
    InvalidUtf8,
    InvalidProvenance(String),
    ChecksumMismatch,
    ProgramBindingMismatch,
    EvidenceBindingMismatch,
    InvalidBytecode(BytecodeError),
    BytecodeVerification(String),
    InvalidPackage(PackageError),
}

impl fmt::Display for PortableArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => f.write_str("unrecognized portable artifact magic"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported portable artifact version {v}"),
            Self::UnsupportedFlags(v) => write!(f, "unsupported portable artifact flags {v}"),
            Self::UnsupportedKind(v) => write!(f, "unsupported portable artifact payload kind {v}"),
            Self::InvalidReserved => f.write_str("portable artifact reserved fields are nonzero"),
            Self::LimitExceeded { what, count, limit } => {
                write!(f, "portable {what} count {count} exceeds {limit}")
            }
            Self::Truncated => f.write_str("truncated portable artifact"),
            Self::TrailingBytes => f.write_str("portable artifact has trailing bytes"),
            Self::InvalidUtf8 => f.write_str("portable source name is not UTF-8"),
            Self::InvalidProvenance(message) => write!(f, "invalid portable provenance: {message}"),
            Self::ChecksumMismatch => f.write_str("portable artifact SHA-256 checksum mismatch"),
            Self::ProgramBindingMismatch => {
                f.write_str("portable provenance belongs to different executable bytes")
            }
            Self::EvidenceBindingMismatch => {
                f.write_str("portable evidence belongs to different executable bytes")
            }
            Self::InvalidBytecode(error) => write!(f, "invalid portable bytecode: {error}"),
            Self::BytecodeVerification(error) => {
                write!(f, "portable bytecode verification failed: {error}")
            }
            Self::InvalidPackage(error) => write!(f, "invalid portable package: {error}"),
        }
    }
}

impl Error for PortableArtifactError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidBytecode(error) => Some(error),
            Self::InvalidPackage(error) => Some(error),
            _ => None,
        }
    }
}

impl From<BytecodeError> for PortableArtifactError {
    fn from(error: BytecodeError) -> Self {
        Self::InvalidBytecode(error)
    }
}

impl From<PackageError> for PortableArtifactError {
    fn from(error: PackageError) -> Self {
        Self::InvalidPackage(error)
    }
}

impl PortableArtifact {
    pub fn from_bytecode(program: BytecodeProgram) -> Result<Self, PortableArtifactError> {
        let artifact = Self {
            payload: PortableArtifactPayload::Bytecode(program),
            provenance: None,
        };
        artifact.validate_provenance()?;
        Ok(artifact)
    }

    pub fn from_package(package: PortablePackage) -> Result<Self, PortableArtifactError> {
        let artifact = Self {
            payload: PortableArtifactPayload::Package(package),
            provenance: None,
        };
        artifact.validate_provenance()?;
        // Retain the package's complete policy/capability/witness admission.
        artifact.legacy_bytes()?;
        Ok(artifact)
    }

    /// Retain final linked source identities. Pre-link verification payloads
    /// are deliberately discarded even if a caller supplied one manually.
    pub fn from_linked(linked: LinkedProgram) -> Result<Self, PortableArtifactError> {
        if linked.metadata.runtime_abi != CURRENT_RUNTIME_ABI {
            return Err(PortableArtifactError::InvalidProvenance(
                "linked runtime ABI is incompatible".into(),
            ));
        }
        let provenance = PortableProvenance {
            program_digest: program_digest(&linked.code)?,
            source_files: linked.metadata.source_files,
            evidence: None,
        };
        let artifact = Self {
            payload: PortableArtifactPayload::Bytecode(linked.code),
            provenance: Some(provenance),
        };
        artifact.validate_provenance()?;
        Ok(artifact)
    }

    /// Associate an already constructed legacy package with the exact linked
    /// executable that produced its source identities.
    pub fn from_linked_package(
        linked: LinkedProgram,
        package: PortablePackage,
    ) -> Result<Self, PortableArtifactError> {
        if linked.code != package.program {
            return Err(PortableArtifactError::ProgramBindingMismatch);
        }
        let mut artifact = Self::from_linked(linked)?;
        artifact.payload = PortableArtifactPayload::Package(package);
        artifact.legacy_bytes()?;
        Ok(artifact)
    }

    pub fn program(&self) -> &BytecodeProgram {
        match &self.payload {
            PortableArtifactPayload::Bytecode(program) => program,
            PortableArtifactPayload::Package(package) => &package.program,
        }
    }

    pub fn package(&self) -> Option<&PortablePackage> {
        match &self.payload {
            PortableArtifactPayload::Bytecode(_) => None,
            PortableArtifactPayload::Package(package) => Some(package),
        }
    }

    pub fn provenance_status(&self) -> SourceProvenanceStatus {
        // Public payload/provenance fields may have been changed since their
        // construction. Stale or malformed associations cannot advertise an
        // available manifest before serialization rejects them.
        if self.validate_provenance().is_err() {
            return SourceProvenanceStatus::Unavailable;
        }
        let available = self.provenance.as_ref().is_some_and(|provenance| {
            !provenance.source_files.is_empty()
                && provenance.source_files.iter().all(|source| {
                    source.content_digest != 0 && !source.name.starts_with("<source:")
                })
                && self.program().source_map.iter().all(|entry| {
                    provenance
                        .source_files
                        .get(entry.span.source.index())
                        .is_some()
                })
        });
        if available {
            SourceProvenanceStatus::ManifestAvailable
        } else {
            SourceProvenanceStatus::Unavailable
        }
    }

    /// Attach a new producer claim to these final executable bytes. This
    /// records binding only; it does not certify the evidence's semantics.
    pub fn attach_evidence(
        &mut self,
        kind: VerificationArtifactKind,
        format_version: u16,
        payload: Vec<u8>,
    ) -> Result<(), PortableArtifactError> {
        self.validate_provenance()?;
        limit("evidence byte", payload.len(), MAX_PORTABLE_EVIDENCE_BYTES)?;
        if format_version == 0 {
            return Err(PortableArtifactError::InvalidProvenance(
                "evidence version must be nonzero".into(),
            ));
        }
        let digest = program_digest(self.program())?;
        let mut provenance = self.provenance.clone().unwrap_or(PortableProvenance {
            program_digest: digest,
            source_files: Vec::new(),
            evidence: None,
        });
        provenance.evidence = Some(PortableEvidence {
            program_digest: digest,
            kind,
            format_version,
            payload,
        });
        encode_provenance(&provenance)?;
        self.provenance = Some(provenance);
        Ok(())
    }

    /// Explicitly obtain the unchanged legacy representation, dropping the
    /// envelope's provenance and checksum.
    pub fn legacy_bytes(&self) -> Result<Vec<u8>, PortableArtifactError> {
        match &self.payload {
            PortableArtifactPayload::Bytecode(program) => Ok(program.to_bytes()?),
            PortableArtifactPayload::Package(package) => Ok(package.to_bytes()?),
        }
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, PortableArtifactError> {
        self.validate_provenance()?;
        let payload = self.legacy_bytes()?;
        let provenance = self
            .provenance
            .as_ref()
            .map(encode_provenance)
            .transpose()?
            .unwrap_or_default();
        let size = HEADER_BYTES
            .checked_add(payload.len())
            .and_then(|v| v.checked_add(provenance.len()))
            .and_then(|v| v.checked_add(CHECKSUM_BYTES))
            .ok_or(PortableArtifactError::Truncated)?;
        limit("artifact byte", size, MAX_PORTABLE_ARTIFACT_BYTES)?;
        let mut bytes = Vec::with_capacity(size);
        bytes.extend_from_slice(MAGIC);
        put_u16(&mut bytes, PORTABLE_ARTIFACT_VERSION);
        put_u16(
            &mut bytes,
            if self.provenance.is_some() {
                FLAG_PROVENANCE
            } else {
                0
            },
        );
        bytes.push(match self.payload {
            PortableArtifactPayload::Bytecode(_) => 1,
            PortableArtifactPayload::Package(_) => 2,
        });
        bytes.extend_from_slice(&[0; 3]);
        put_u32(&mut bytes, payload.len())?;
        put_u32(&mut bytes, provenance.len())?;
        bytes.extend_from_slice(&payload);
        bytes.extend_from_slice(&provenance);
        let checksum = envelope_digest(&bytes);
        bytes.extend_from_slice(&checksum);
        Ok(bytes)
    }

    /// Accept legacy bytes as explicitly unavailable provenance, or validate
    /// an envelope before any legacy package witness replay is attempted.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PortableArtifactError> {
        limit("artifact byte", bytes.len(), MAX_PORTABLE_ARTIFACT_BYTES)?;
        if bytes.starts_with(BYTECODE_MAGIC) {
            return Self::from_bytecode(BytecodeProgram::from_bytes(bytes)?);
        }
        if bytes.starts_with(PACKAGE_MAGIC) {
            return Self::from_package(PortablePackage::from_bytes(bytes)?);
        }
        let mut reader = Reader::new(bytes);
        if reader.take(8)? != MAGIC {
            return Err(PortableArtifactError::BadMagic);
        }
        let version = reader.u16()?;
        if version != PORTABLE_ARTIFACT_VERSION {
            return Err(PortableArtifactError::UnsupportedVersion(version));
        }
        let flags = reader.u16()?;
        if flags & !FLAG_PROVENANCE != 0 {
            return Err(PortableArtifactError::UnsupportedFlags(flags));
        }
        let kind = reader.u8()?;
        let payload_limit = match kind {
            1 => MAX_ARTIFACT_BYTES,
            2 => MAX_PACKAGE_BYTES,
            v => return Err(PortableArtifactError::UnsupportedKind(v)),
        };
        if reader.take(3)? != [0; 3] {
            return Err(PortableArtifactError::InvalidReserved);
        }
        let payload_len = reader.u32()? as usize;
        limit("payload byte", payload_len, payload_limit)?;
        let provenance_len = reader.u32()? as usize;
        limit(
            "provenance byte",
            provenance_len,
            MAX_PORTABLE_PROVENANCE_BYTES,
        )?;
        if (flags & FLAG_PROVENANCE != 0) != (provenance_len != 0) {
            return Err(PortableArtifactError::InvalidProvenance(
                "provenance flag and length disagree".into(),
            ));
        }
        let payload_bytes = reader.take(payload_len)?;
        let provenance_bytes = reader.take(provenance_len)?;
        let checksum_start = reader.position;
        let checksum = reader.take(CHECKSUM_BYTES)?;
        if reader.remaining() != 0 {
            return Err(PortableArtifactError::TrailingBytes);
        }
        if envelope_digest(&bytes[..checksum_start]).as_slice() != checksum {
            return Err(PortableArtifactError::ChecksumMismatch);
        }
        let provenance = if provenance_bytes.is_empty() {
            None
        } else {
            Some(decode_provenance(provenance_bytes)?)
        };
        let artifact = match kind {
            1 => Self {
                payload: PortableArtifactPayload::Bytecode(BytecodeProgram::from_bytes(
                    payload_bytes,
                )?),
                provenance,
            },
            2 => {
                // Package v3's fixed header is used only to locate its bounded
                // bytecode slice. The existing decoder still owns all policy,
                // capability and witness validation. Provenance is checked
                // first, so stale maps never initiate a costly witness replay.
                let program = preflight_package_program(payload_bytes)?;
                validate_program_provenance(&program, provenance.as_ref())?;
                let package = PortablePackage::from_bytes(payload_bytes)?;
                Self {
                    payload: PortableArtifactPayload::Package(package),
                    provenance,
                }
            }
            _ => unreachable!("payload kind checked above"),
        };
        artifact.validate_provenance()?;
        Ok(artifact)
    }

    pub fn validate_provenance(&self) -> Result<(), PortableArtifactError> {
        validate_program_provenance(self.program(), self.provenance.as_ref())
    }
}

/// SHA-256 of the canonical legacy executable, with a format-specific domain.
pub fn program_digest(program: &BytecodeProgram) -> Result<[u8; 32], PortableArtifactError> {
    let mut hash = Sha256::new();
    hash.update(b"ourochronos.executable/v1\0");
    hash.update(program.to_bytes()?);
    Ok(hash.finalize().into())
}

fn envelope_digest(bytes: &[u8]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"ourochronos.portable-artifact/v1\0");
    hash.update(bytes);
    hash.finalize().into()
}

fn validate_program_provenance(
    program: &BytecodeProgram,
    provenance: Option<&PortableProvenance>,
) -> Result<(), PortableArtifactError> {
    program.validate()?;
    if let Some(provenance) = provenance {
        if program_digest(program)? != provenance.program_digest {
            return Err(PortableArtifactError::ProgramBindingMismatch);
        }
        validate_descriptor_table(provenance)?;
        // Empty descriptors represent missing source identities, never an
        // invented file table. Nonempty tables must cover every retained span.
        if !provenance.source_files.is_empty() {
            for entry in &program.source_map {
                let source = provenance
                    .source_files
                    .get(entry.span.source.index())
                    .ok_or_else(|| {
                        PortableArtifactError::InvalidProvenance(
                            "source map references an absent source identity".into(),
                        )
                    })?;
                if entry.span.range.end as u128 > u128::from(source.byte_len) {
                    return Err(PortableArtifactError::InvalidProvenance(
                        "source map range exceeds retained source length".into(),
                    ));
                }
            }
        }
        if let Some(evidence) = &provenance.evidence {
            if evidence.program_digest != provenance.program_digest {
                return Err(PortableArtifactError::EvidenceBindingMismatch);
            }
        }
    }
    verify_default(program)
        .map_err(|error| PortableArtifactError::BytecodeVerification(error.to_string()))?;
    Ok(())
}

fn validate_descriptor_table(provenance: &PortableProvenance) -> Result<(), PortableArtifactError> {
    limit(
        "source file",
        provenance.source_files.len(),
        MAX_PORTABLE_SOURCE_FILES,
    )?;
    let mut size = 37usize;
    for (index, source) in provenance.source_files.iter().enumerate() {
        if source.id as usize != index {
            return Err(PortableArtifactError::InvalidProvenance(
                "source identities are not canonical and contiguous".into(),
            ));
        }
        limit(
            "source name byte",
            source.name.len(),
            MAX_PORTABLE_SOURCE_NAME_BYTES,
        )?;
        if source.name.trim().is_empty() || source.name.contains('\0') {
            return Err(PortableArtifactError::InvalidProvenance(
                "source name is empty or contains NUL".into(),
            ));
        }
        size = size
            .checked_add(24)
            .and_then(|v| v.checked_add(source.name.len()))
            .ok_or(PortableArtifactError::Truncated)?;
    }
    if let Some(evidence) = &provenance.evidence {
        limit(
            "evidence byte",
            evidence.payload.len(),
            MAX_PORTABLE_EVIDENCE_BYTES,
        )?;
        if evidence.format_version == 0 {
            return Err(PortableArtifactError::InvalidProvenance(
                "evidence version must be nonzero".into(),
            ));
        }
        size = size
            .checked_add(39)
            .and_then(|v| v.checked_add(evidence.payload.len()))
            .ok_or(PortableArtifactError::Truncated)?;
    }
    limit("provenance byte", size, MAX_PORTABLE_PROVENANCE_BYTES)
}

fn encode_provenance(provenance: &PortableProvenance) -> Result<Vec<u8>, PortableArtifactError> {
    validate_descriptor_table(provenance)?;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&provenance.program_digest);
    put_u32(&mut bytes, provenance.source_files.len())?;
    for source in &provenance.source_files {
        bytes.extend_from_slice(&source.id.to_le_bytes());
        put_u32(&mut bytes, source.name.len())?;
        bytes.extend_from_slice(source.name.as_bytes());
        bytes.extend_from_slice(&source.byte_len.to_le_bytes());
        bytes.extend_from_slice(&source.content_digest.to_le_bytes());
    }
    bytes.push(u8::from(provenance.evidence.is_some()));
    if let Some(evidence) = &provenance.evidence {
        bytes.extend_from_slice(&evidence.program_digest);
        bytes.push(match evidence.kind {
            VerificationArtifactKind::BytecodeReport => 1,
            VerificationArtifactKind::SolverCertificate => 2,
        });
        put_u16(&mut bytes, evidence.format_version);
        put_u32(&mut bytes, evidence.payload.len())?;
        bytes.extend_from_slice(&evidence.payload);
    }
    Ok(bytes)
}

fn decode_provenance(bytes: &[u8]) -> Result<PortableProvenance, PortableArtifactError> {
    let mut reader = Reader::new(bytes);
    let program_digest = reader.take(32)?.try_into().expect("fixed SHA-256 width");
    let count = reader.u32()? as usize;
    limit("source file", count, MAX_PORTABLE_SOURCE_FILES)?;
    let minimum_bytes = count
        .checked_mul(24)
        .and_then(|v| v.checked_add(1))
        .ok_or(PortableArtifactError::Truncated)?;
    if minimum_bytes > reader.remaining() {
        return Err(PortableArtifactError::Truncated);
    }
    let mut source_files = Vec::with_capacity(count);
    for _ in 0..count {
        let id = reader.u32()?;
        let name_len = reader.u32()? as usize;
        limit("source name byte", name_len, MAX_PORTABLE_SOURCE_NAME_BYTES)?;
        let name = std::str::from_utf8(reader.take(name_len)?)
            .map_err(|_| PortableArtifactError::InvalidUtf8)?
            .to_string();
        source_files.push(ObjectSourceFile {
            id,
            name,
            byte_len: reader.u64()?,
            content_digest: reader.u64()?,
        });
    }
    let evidence = match reader.u8()? {
        0 => None,
        1 => {
            let program_digest = reader.take(32)?.try_into().expect("fixed SHA-256 width");
            let kind = match reader.u8()? {
                1 => VerificationArtifactKind::BytecodeReport,
                2 => VerificationArtifactKind::SolverCertificate,
                _ => {
                    return Err(PortableArtifactError::InvalidProvenance(
                        "unknown evidence kind".into(),
                    ))
                }
            };
            let format_version = reader.u16()?;
            let payload_len = reader.u32()? as usize;
            limit("evidence byte", payload_len, MAX_PORTABLE_EVIDENCE_BYTES)?;
            Some(PortableEvidence {
                program_digest,
                kind,
                format_version,
                payload: reader.take(payload_len)?.to_vec(),
            })
        }
        _ => {
            return Err(PortableArtifactError::InvalidProvenance(
                "unknown evidence option".into(),
            ))
        }
    };
    if reader.remaining() != 0 {
        return Err(PortableArtifactError::TrailingBytes);
    }
    let provenance = PortableProvenance {
        program_digest,
        source_files,
        evidence,
    };
    validate_descriptor_table(&provenance)?;
    Ok(provenance)
}

fn preflight_package_program(bytes: &[u8]) -> Result<BytecodeProgram, PortableArtifactError> {
    limit("package byte", bytes.len(), MAX_PACKAGE_BYTES)?;
    let mut reader = Reader::new(bytes);
    if reader.take(8)? != PACKAGE_MAGIC {
        return Err(PortableArtifactError::BadMagic);
    }
    let version = reader.u16()?;
    if version != 3 {
        return Err(PackageError::UnsupportedVersion(version).into());
    }
    reader.take(12)?;
    let name_len = reader.u16()? as usize;
    limit("package name byte", name_len, MAX_PACKAGE_NAME_BYTES)?;
    let bytecode_len = reader.u32()? as usize;
    limit("bytecode byte", bytecode_len, MAX_ARTIFACT_BYTES)?;
    let witness_count = reader.u32()? as usize;
    limit("witness cell", witness_count, MAX_PACKAGE_WITNESS_CELLS)?;
    reader.take(32)?;
    let prefix = name_len
        .checked_add(
            witness_count
                .checked_mul(16)
                .ok_or(PortableArtifactError::Truncated)?,
        )
        .ok_or(PortableArtifactError::Truncated)?;
    reader.take(prefix)?;
    let bytecode = reader.take(bytecode_len)?;
    if reader.remaining() != 0 {
        return Err(PortableArtifactError::TrailingBytes);
    }
    Ok(BytecodeProgram::from_bytes(bytecode)?)
}

fn limit(what: &'static str, count: usize, limit: usize) -> Result<(), PortableArtifactError> {
    if count > limit {
        Err(PortableArtifactError::LimitExceeded { what, count, limit })
    } else {
        Ok(())
    }
}

fn put_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
fn put_u32(bytes: &mut Vec<u8>, value: usize) -> Result<(), PortableArtifactError> {
    let value = u32::try_from(value).map_err(|_| PortableArtifactError::Truncated)?;
    bytes.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}
impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }
    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.position)
    }
    fn take(&mut self, count: usize) -> Result<&'a [u8], PortableArtifactError> {
        let end = self
            .position
            .checked_add(count)
            .ok_or(PortableArtifactError::Truncated)?;
        let bytes = self
            .bytes
            .get(self.position..end)
            .ok_or(PortableArtifactError::Truncated)?;
        self.position = end;
        Ok(bytes)
    }
    fn u8(&mut self) -> Result<u8, PortableArtifactError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, PortableArtifactError> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("fixed u16 width"),
        ))
    }
    fn u32(&mut self) -> Result<u32, PortableArtifactError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("fixed u32 width"),
        ))
    }
    fn u64(&mut self) -> Result<u64, PortableArtifactError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("fixed u64 width"),
        ))
    }
}
