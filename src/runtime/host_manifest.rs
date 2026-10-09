//! Passive, bounded scalar host dependencies and approved frozen snapshots.
//!
//! A manifest binds exact canonical bytecode, EVERY foreign descriptor, ordered
//! INPUT words and a finite deterministic map of scalar calls. Namespace and
//! symbol strings are passive artifact identities, never executable paths. No
//! credential fields, callback addresses, loader configuration or capabilities
//! are represented. Decoding does not register a callback or authorize execution.
//!
//! `check` requires both expected bytecode and an independently approved digest
//! of the manifest bytes. SHA256 checks integrity/identity, not authenticity;
//! the embedder owns approval and snapshot provenance. Checked data is immutable.
//! Explicit `frozen_host_table` creates only PURE lookup callbacks; all descriptor
//! identities remain exact and missing finite observations fail. INPUT is exposed
//! separately for explicit VM configuration. Packages/CLI gain no new authority.
//! The embedder selects the checked program and supplies its frozen INPUT and
//! resource configuration; the general host table enforces descriptor/call identities.
//! Checking establishes bounded data and identity consistency. Source admission
//! and runtime behavior retain their existing contracts.
//! Trusted general callbacks remain uninterruptible inside one VM instruction.
//! This bounded lookup implementation performs no native/host effects.

use crate::bytecode::{BytecodeProgram, ForeignEffects, ForeignEntry, ForeignScalarType};
use crate::hir::ForeignId;
use crate::runtime::ffi::{ForeignHostError, ForeignHostTable};
use sha2::{Digest, Sha256};
use std::fmt;
use std::sync::Arc;

pub const HOST_MANIFEST_VERSION: u16 = 1;
pub const HOST_SNAPSHOT_SEMANTICS_VERSION: u16 = 1;
pub const MAX_HOST_MANIFEST_BYTES: usize = 1024 * 1024;
pub const MAX_HOST_DEPENDENCIES: usize = 64;
pub const MAX_HOST_ARGUMENTS: usize = 16;
pub const MAX_HOST_INPUT_WORDS: usize = 4096;
pub const MAX_HOST_OBSERVATIONS: usize = 4096;
pub const MAX_HOST_NAME_BYTES: usize = 256;
const MAX_CODE_RECORDS: usize = 65_536;
const MAX_CODE_UNITS: usize = 4096;
const MAX_CODE_BYTES: usize = 8 * 1024 * 1024;
const MAGIC: &[u8; 8] = b"OUROHM\0\0";
const HEADER: usize = 16;
const HASH: usize = 32;

/// Words retain their u64/i64 bit representations; void is `None`.
#[derive(Clone, PartialEq, Eq)]
pub struct FrozenScalarObservation {
    pub foreign: ForeignId,
    pub arguments: Vec<u64>,
    pub result: Option<u64>,
}

impl fmt::Debug for FrozenScalarObservation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FrozenScalarObservation")
            .field("foreign", &self.foreign)
            .field("argument_count", &self.arguments.len())
            .field("has_result", &self.result.is_some())
            .finish()
    }
}

/// Mutable decoded data is untrusted until `check` compares expected identities.
#[derive(Clone, PartialEq, Eq)]
pub struct HostManifest {
    pub version: u16,
    pub semantics_version: u16,
    pub bytecode_sha256: [u8; 32],
    pub foreigns: Vec<ForeignEntry>,
    pub input: Vec<u64>,
    pub observations: Vec<FrozenScalarObservation>,
}

impl fmt::Debug for HostManifest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostManifest")
            .field("version", &self.version)
            .field("semantics_version", &self.semantics_version)
            .field("foreign_count", &self.foreigns.len())
            .field("input_word_count", &self.input.len())
            .field("observation_count", &self.observations.len())
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostManifestError {
    ResourceLimit(&'static str),
    InvalidEncoding(&'static str),
    InvalidDescriptor {
        index: usize,
    },
    UnsupportedVersion,
    CorruptEncoding,
    ProgramMismatch,
    DescriptorMismatch,
    ApprovalMismatch,
    UnknownForeign {
        foreign: ForeignId,
    },
    ArgumentMismatch {
        foreign: ForeignId,
        expected: usize,
        got: usize,
    },
    ResultMismatch {
        foreign: ForeignId,
    },
    DuplicateObservation,
    NoncanonicalObservations,
    UnsupportedEffect {
        foreign: ForeignId,
    },
    MissingObservation {
        foreign: ForeignId,
    },
    InvalidProgram(String),
    HostBinding(ForeignHostError),
}

impl fmt::Display for HostManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResourceLimit(kind) => write!(f, "host manifest {kind} exceeds its bound"),
            Self::InvalidEncoding(kind) => write!(f, "invalid host manifest {kind}"),
            Self::InvalidDescriptor { index } => write!(f, "invalid host descriptor {index}"),
            Self::UnsupportedVersion => {
                f.write_str("unsupported host manifest/schema semantics version")
            }
            Self::CorruptEncoding => f.write_str("host manifest checksum mismatch"),
            Self::ProgramMismatch => {
                f.write_str("host manifest belongs to different executable bytes")
            }
            Self::DescriptorMismatch => {
                f.write_str("host manifest does not retain every exact foreign descriptor")
            }
            Self::ApprovalMismatch => {
                f.write_str("host snapshot differs from independently approved bytes")
            }
            Self::UnknownForeign { foreign } => {
                write!(f, "host snapshot has no foreign target {foreign}")
            }
            Self::ArgumentMismatch {
                foreign,
                expected,
                got,
            } => write!(
                f,
                "host target {foreign} expects {expected} scalar arguments, got {got}"
            ),
            Self::ResultMismatch { foreign } => write!(
                f,
                "host target {foreign} has a different scalar result shape"
            ),
            Self::DuplicateObservation => f.write_str("duplicate finite scalar observation key"),
            Self::NoncanonicalObservations => {
                f.write_str("finite scalar observations are not canonically ordered")
            }
            Self::UnsupportedEffect { foreign } => {
                write!(f, "foreign target {foreign} is outside PURE frozen replay")
            }
            Self::MissingObservation { foreign } => write!(
                f,
                "foreign target {foreign} has no approved observation for these arguments"
            ),
            Self::InvalidProgram(reason) => write!(f, "invalid host-manifest executable: {reason}"),
            Self::HostBinding(error) => error.fmt(f),
        }
    }
}
impl std::error::Error for HostManifestError {}

fn descriptor_valid(entry: &ForeignEntry, index: usize) -> bool {
    let name = |name: &str| {
        !name.trim().is_empty() && name.len() <= MAX_HOST_NAME_BYTES && !name.contains('\0')
    };
    entry.id.index() == index
        && name(&entry.library)
        && name(&entry.symbol)
        && entry.parameters.len() <= MAX_HOST_ARGUMENTS
        && ForeignEffects::from_bits(entry.effects.bits()).is_some()
        && (entry.effects.bits() & ForeignEffects::PURE == 0 || entry.effects.is_pure())
}

fn program_identity(program: &BytecodeProgram) -> Result<[u8; 32], HostManifestError> {
    if program.instructions.len() > MAX_CODE_RECORDS
        || program.source_map.len() > MAX_CODE_RECORDS
        || program.procedures.len() > MAX_CODE_UNITS
        || program.quotations.len() > MAX_CODE_UNITS
        || program.foreigns.len() > MAX_HOST_DEPENDENCIES
    {
        return Err(HostManifestError::ResourceLimit("executable shape"));
    }
    for (index, entry) in program.foreigns.iter().enumerate() {
        if !descriptor_valid(entry, index) {
            return Err(HostManifestError::InvalidDescriptor { index });
        }
    }
    let bytes = program
        .to_bytes()
        .map_err(|error| HostManifestError::InvalidProgram(error.to_string()))?;
    if bytes.len() > MAX_CODE_BYTES {
        return Err(HostManifestError::ResourceLimit("executable bytes"));
    }
    Ok(Sha256::digest(bytes).into())
}

fn observation_key(row: &FrozenScalarObservation) -> (u32, &[u64]) {
    (row.foreign.as_u32(), &row.arguments)
}

impl HostManifest {
    /// Construct from locally selected data. Sorting gives one encoding; input
    /// consumption order is retained. Construction is not external approval.
    pub fn new(
        program: &BytecodeProgram,
        input: Vec<u64>,
        mut observations: Vec<FrozenScalarObservation>,
    ) -> Result<Self, HostManifestError> {
        if input.len() > MAX_HOST_INPUT_WORDS || observations.len() > MAX_HOST_OBSERVATIONS {
            return Err(HostManifestError::ResourceLimit("snapshot items"));
        }
        let identity = program_identity(program)?;
        for row in &observations {
            validate_row(row, &program.foreigns)?;
        }
        observations.sort_by(|a, b| observation_key(a).cmp(&observation_key(b)));
        let manifest = Self {
            version: HOST_MANIFEST_VERSION,
            semantics_version: HOST_SNAPSHOT_SEMANTICS_VERSION,
            bytecode_sha256: identity,
            foreigns: program.foreigns.clone(),
            input,
            observations,
        };
        manifest.validate_data()?;
        Ok(manifest)
    }
    fn validate_data(&self) -> Result<(), HostManifestError> {
        if self.version != HOST_MANIFEST_VERSION
            || self.semantics_version != HOST_SNAPSHOT_SEMANTICS_VERSION
        {
            return Err(HostManifestError::UnsupportedVersion);
        }
        if self.foreigns.len() > MAX_HOST_DEPENDENCIES
            || self.input.len() > MAX_HOST_INPUT_WORDS
            || self.observations.len() > MAX_HOST_OBSERVATIONS
        {
            return Err(HostManifestError::ResourceLimit("snapshot items"));
        }
        for (index, entry) in self.foreigns.iter().enumerate() {
            if !descriptor_valid(entry, index) {
                return Err(HostManifestError::InvalidDescriptor { index });
            }
        }
        let mut previous = None;
        for row in &self.observations {
            validate_row(row, &self.foreigns)?;
            let key = observation_key(row);
            if let Some(old) = previous {
                if old == key {
                    return Err(HostManifestError::DuplicateObservation);
                }
                if old > key {
                    return Err(HostManifestError::NoncanonicalObservations);
                }
            }
            previous = Some(key);
        }
        // The maximum element counts also imply a byte ceiling before buffer
        // allocation; measure exact size for public mutable caller data.
        self.encoded_size()?;
        Ok(())
    }
    fn encoded_size(&self) -> Result<usize, HostManifestError> {
        let mut size = HEADER + HASH + 2 + 32 + 4 + 4 + self.input.len() * 8 + 4;
        for entry in &self.foreigns {
            size = size
                .checked_add(15 + entry.library.len() + entry.symbol.len() + entry.parameters.len())
                .ok_or(HostManifestError::ResourceLimit("encoded length"))?;
        }
        for row in &self.observations {
            size = size
                .checked_add(6 + row.arguments.len() * 8 + usize::from(row.result.is_some()) * 8)
                .ok_or(HostManifestError::ResourceLimit("encoded length"))?;
        }
        if size > MAX_HOST_MANIFEST_BYTES {
            return Err(HostManifestError::ResourceLimit("encoded bytes"));
        }
        Ok(size)
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>, HostManifestError> {
        self.validate_data()?;
        let size = self.encoded_size()?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|_| HostManifestError::ResourceLimit("encoding allocation"))?;
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&self.version.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&((size - HEADER - HASH) as u32).to_le_bytes());
        bytes.extend_from_slice(&self.semantics_version.to_le_bytes());
        bytes.extend_from_slice(&self.bytecode_sha256);
        count(&mut bytes, self.foreigns.len());
        for entry in &self.foreigns {
            bytes.extend_from_slice(&entry.id.as_u32().to_le_bytes());
            blob(&mut bytes, entry.library.as_bytes());
            blob(&mut bytes, entry.symbol.as_bytes());
            bytes.push(entry.parameters.len() as u8);
            for typ in &entry.parameters {
                bytes.push(type_tag(*typ));
            }
            bytes.push(entry.result.map_or(0, |typ| type_tag(typ) + 1));
            bytes.push(entry.effects.bits());
        }
        count(&mut bytes, self.input.len());
        for word in &self.input {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        count(&mut bytes, self.observations.len());
        for row in &self.observations {
            bytes.extend_from_slice(&row.foreign.as_u32().to_le_bytes());
            bytes.push(row.arguments.len() as u8);
            for word in &row.arguments {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
            bytes.push(u8::from(row.result.is_some()));
            if let Some(word) = row.result {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
        }
        let checksum = checksum(&bytes);
        bytes.extend_from_slice(&checksum);
        debug_assert_eq!(bytes.len(), size);
        Ok(bytes)
    }
    /// SHA of all canonical bytes is an approval identity, not a signature.
    pub fn digest(&self) -> Result<[u8; 32], HostManifestError> {
        Ok(Sha256::digest(self.to_bytes()?).into())
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, HostManifestError> {
        if bytes.len() > MAX_HOST_MANIFEST_BYTES {
            return Err(HostManifestError::ResourceLimit("decoder bytes"));
        }
        if bytes.len() < HEADER + HASH || &bytes[..8] != MAGIC {
            return Err(HostManifestError::InvalidEncoding("magic/length"));
        }
        let mut header = Reader::new(&bytes[8..HEADER]);
        let version = header.u16()?;
        if version != HOST_MANIFEST_VERSION {
            return Err(HostManifestError::UnsupportedVersion);
        }
        if header.u16()? != 0 {
            return Err(HostManifestError::InvalidEncoding("flags"));
        }
        let end = HEADER
            .checked_add(header.u32()? as usize)
            .ok_or(HostManifestError::InvalidEncoding("body length"))?;
        if end.checked_add(HASH) != Some(bytes.len()) {
            return Err(HostManifestError::InvalidEncoding("body/trailing length"));
        }
        if checksum(&bytes[..end]).as_slice() != &bytes[end..] {
            return Err(HostManifestError::CorruptEncoding);
        }
        let mut reader = Reader::new(&bytes[HEADER..end]);
        let semantics_version = reader.u16()?;
        if semantics_version != HOST_SNAPSHOT_SEMANTICS_VERSION {
            return Err(HostManifestError::UnsupportedVersion);
        }
        let bytecode_sha256 = reader
            .take(32)?
            .try_into()
            .map_err(|_| HostManifestError::InvalidEncoding("program digest"))?;
        let n = reader.count(MAX_HOST_DEPENDENCIES, 15)?;
        let mut foreigns = reserve(n)?;
        for index in 0..n {
            let id = foreign_id(reader.u32()?)?;
            let library = reader.text(MAX_HOST_NAME_BYTES)?;
            let symbol = reader.text(MAX_HOST_NAME_BYTES)?;
            let argc = reader.u8()? as usize;
            if argc > MAX_HOST_ARGUMENTS {
                return Err(HostManifestError::ResourceLimit("descriptor arguments"));
            }
            let mut parameters = reserve(argc)?;
            for _ in 0..argc {
                parameters.push(decode_type(reader.u8()?)?);
            }
            let result = match reader.u8()? {
                0 => None,
                1 => Some(ForeignScalarType::U64),
                2 => Some(ForeignScalarType::I64),
                _ => return Err(HostManifestError::InvalidEncoding("scalar result type")),
            };
            let effects = ForeignEffects::from_bits(reader.u8()?)
                .ok_or(HostManifestError::InvalidDescriptor { index })?;
            let entry = ForeignEntry {
                id,
                library,
                symbol,
                parameters,
                result,
                effects,
            };
            if !descriptor_valid(&entry, index) {
                return Err(HostManifestError::InvalidDescriptor { index });
            }
            foreigns.push(entry);
        }
        let n = reader.count(MAX_HOST_INPUT_WORDS, 8)?;
        let mut input = reserve(n)?;
        for _ in 0..n {
            input.push(reader.u64()?);
        }
        let n = reader.count(MAX_HOST_OBSERVATIONS, 6)?;
        let mut observations = reserve(n)?;
        for _ in 0..n {
            let foreign = foreign_id(reader.u32()?)?;
            let argc = reader.u8()? as usize;
            if argc > MAX_HOST_ARGUMENTS {
                return Err(HostManifestError::ResourceLimit("observed arguments"));
            }
            let mut arguments = reserve(argc)?;
            for _ in 0..argc {
                arguments.push(reader.u64()?);
            }
            let result = match reader.u8()? {
                0 => None,
                1 => Some(reader.u64()?),
                _ => return Err(HostManifestError::InvalidEncoding("observed result tag")),
            };
            observations.push(FrozenScalarObservation {
                foreign,
                arguments,
                result,
            });
        }
        if !reader.remaining().is_empty() {
            return Err(HostManifestError::InvalidEncoding("trailing fields"));
        }
        let manifest = Self {
            version,
            semantics_version,
            bytecode_sha256,
            foreigns,
            input,
            observations,
        };
        manifest.validate_data()?;
        Ok(manifest)
    }
    /// Approval must come from trusted capture/configuration, never merely from
    /// recalculating an untrusted blob's own digest and approving that value.
    pub fn check(
        self,
        expected_program: &BytecodeProgram,
        approved_manifest_digest: [u8; 32],
    ) -> Result<CheckedHostManifest, HostManifestError> {
        self.validate_data()?;
        if self.bytecode_sha256 != program_identity(expected_program)? {
            return Err(HostManifestError::ProgramMismatch);
        }
        if self.foreigns != expected_program.foreigns {
            return Err(HostManifestError::DescriptorMismatch);
        }
        if self.digest()? != approved_manifest_digest {
            return Err(HostManifestError::ApprovalMismatch);
        }
        Ok(CheckedHostManifest {
            manifest: Arc::new(self),
            approved_digest: approved_manifest_digest,
        })
    }
}

fn validate_row(
    row: &FrozenScalarObservation,
    foreigns: &[ForeignEntry],
) -> Result<(), HostManifestError> {
    let entry = foreigns
        .get(row.foreign.index())
        .ok_or(HostManifestError::UnknownForeign {
            foreign: row.foreign,
        })?;
    if row.arguments.len() != entry.parameters.len() {
        return Err(HostManifestError::ArgumentMismatch {
            foreign: row.foreign,
            expected: entry.parameters.len(),
            got: row.arguments.len(),
        });
    }
    if row.result.is_some() != entry.result.is_some() {
        return Err(HostManifestError::ResultMismatch {
            foreign: row.foreign,
        });
    }
    if !entry.effects.is_pure() {
        return Err(HostManifestError::UnsupportedEffect {
            foreign: row.foreign,
        });
    }
    Ok(())
}

/// Owned immutable snapshot; clones share data without retaining caller borrows.
#[derive(Clone)]
pub struct CheckedHostManifest {
    manifest: Arc<HostManifest>,
    approved_digest: [u8; 32],
}
impl fmt::Debug for CheckedHostManifest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.manifest.fmt(f)
    }
}
impl CheckedHostManifest {
    pub fn input(&self) -> &[u64] {
        &self.manifest.input
    }
    pub fn descriptors(&self) -> &[ForeignEntry] {
        &self.manifest.foreigns
    }
    pub fn approved_digest(&self) -> [u8; 32] {
        self.approved_digest
    }
    pub fn lookup(
        &self,
        foreign: ForeignId,
        arguments: &[u64],
    ) -> Result<Option<u64>, HostManifestError> {
        lookup(&self.manifest, foreign, arguments)
    }
    /// Explicitly prepare a safe table. This does not attach it to any VM and
    /// never loads native code. General/effectful dependencies remain passive.
    /// The embedder must execute the same expected program supplied to `check`
    /// and explicitly configure INPUT/resources. A general `ForeignHostTable`
    /// carries descriptor bindings rather than a VM program-selection guard.
    pub fn frozen_host_table(&self) -> Result<ForeignHostTable, HostManifestError> {
        if let Some(entry) = self
            .manifest
            .foreigns
            .iter()
            .find(|entry| !entry.effects.is_pure())
        {
            return Err(HostManifestError::UnsupportedEffect { foreign: entry.id });
        }
        let mut table = ForeignHostTable::new();
        for entry in &self.manifest.foreigns {
            let snapshot = self.manifest.clone();
            let foreign = entry.id;
            table
                .bind(entry.clone(), move |arguments| {
                    lookup(&snapshot, foreign, arguments).map_err(|error| error.to_string())
                })
                .map_err(HostManifestError::HostBinding)?;
        }
        Ok(table)
    }
}
fn lookup(
    manifest: &HostManifest,
    foreign: ForeignId,
    arguments: &[u64],
) -> Result<Option<u64>, HostManifestError> {
    let entry = manifest
        .foreigns
        .get(foreign.index())
        .ok_or(HostManifestError::UnknownForeign { foreign })?;
    if arguments.len() != entry.parameters.len() {
        return Err(HostManifestError::ArgumentMismatch {
            foreign,
            expected: entry.parameters.len(),
            got: arguments.len(),
        });
    }
    if !entry.effects.is_pure() {
        return Err(HostManifestError::UnsupportedEffect { foreign });
    }
    let index = manifest
        .observations
        .binary_search_by(|row| observation_key(row).cmp(&(foreign.as_u32(), arguments)))
        .map_err(|_| HostManifestError::MissingObservation { foreign })?;
    Ok(manifest.observations[index].result)
}

fn type_tag(typ: ForeignScalarType) -> u8 {
    match typ {
        ForeignScalarType::U64 => 0,
        ForeignScalarType::I64 => 1,
    }
}
fn decode_type(tag: u8) -> Result<ForeignScalarType, HostManifestError> {
    match tag {
        0 => Ok(ForeignScalarType::U64),
        1 => Ok(ForeignScalarType::I64),
        _ => Err(HostManifestError::InvalidEncoding("scalar argument type")),
    }
}
fn foreign_id(raw: u32) -> Result<ForeignId, HostManifestError> {
    ForeignId::try_from_index(raw as usize)
        .ok_or(HostManifestError::InvalidEncoding("foreign identity"))
}
fn checksum(bytes: &[u8]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"ourochronos.host-manifest/v1\0");
    digest.update(bytes);
    digest.finalize().into()
}
fn count(bytes: &mut Vec<u8>, n: usize) {
    bytes.extend_from_slice(&(n as u32).to_le_bytes());
}
fn blob(bytes: &mut Vec<u8>, data: &[u8]) {
    count(bytes, data.len());
    bytes.extend_from_slice(data);
}
fn reserve<T>(n: usize) -> Result<Vec<T>, HostManifestError> {
    let mut data = Vec::new();
    data.try_reserve_exact(n)
        .map_err(|_| HostManifestError::ResourceLimit("decoder allocation"))?;
    Ok(data)
}
struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}
impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }
    fn remaining(&self) -> &'a [u8] {
        &self.bytes[self.position..]
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], HostManifestError> {
        let end = self
            .position
            .checked_add(n)
            .ok_or(HostManifestError::InvalidEncoding("length overflow"))?;
        let data = self
            .bytes
            .get(self.position..end)
            .ok_or(HostManifestError::InvalidEncoding("truncation"))?;
        self.position = end;
        Ok(data)
    }
    fn u8(&mut self) -> Result<u8, HostManifestError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, HostManifestError> {
        Ok(u16::from_le_bytes(
            self.take(2)?
                .try_into()
                .map_err(|_| HostManifestError::InvalidEncoding("u16"))?,
        ))
    }
    fn u32(&mut self) -> Result<u32, HostManifestError> {
        Ok(u32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| HostManifestError::InvalidEncoding("u32"))?,
        ))
    }
    fn u64(&mut self) -> Result<u64, HostManifestError> {
        Ok(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| HostManifestError::InvalidEncoding("u64"))?,
        ))
    }
    fn count(&mut self, bound: usize, minimum: usize) -> Result<usize, HostManifestError> {
        let n = self.u32()? as usize;
        if n > bound {
            return Err(HostManifestError::ResourceLimit("decoder count"));
        }
        if n.checked_mul(minimum)
            .is_none_or(|size| size > self.remaining().len())
        {
            return Err(HostManifestError::InvalidEncoding("count/remaining bytes"));
        }
        Ok(n)
    }
    fn text(&mut self, bound: usize) -> Result<String, HostManifestError> {
        let n = self.count(bound, 1)?;
        Ok(std::str::from_utf8(self.take(n)?)
            .map_err(|_| HostManifestError::InvalidEncoding("UTF8 namespace"))?
            .to_string())
    }
}
