//! Deterministic typed relocation and linking for bytecode objects.
//!
//! A relocatable object may contain deliberately unresolved call/quotation
//! operands at declared relocation sites, so it is not itself an executable
//! [`BytecodeProgram`]. The linker resolves every symbolic import, relocates
//! all local typed identities and control targets, and only returns a program
//! after the ordinary bytecode structural validator accepts the result.
//!
//! [`ObjectModule::to_bytes`] and [`ObjectModule::from_bytes`] define the
//! bounded `OUROOBJ` version-4 artifact. It embeds the ordinary versioned
//! bytecode encoding, followed by typed exports, imports, exact signatures,
//! relocation records, and stable compiler metadata. The bytecode is decoded without prematurely
//! requiring unresolved operands to be executable; object validation then
//! permits those operands at exactly the declared, kind-matched sites and
//! applies every other bytecode structural check unchanged.

use crate::ast::{
    Effect, EffectClass, OpCode, PropertyCell, PropertyComparison, PropertyPredicate, QuoteId,
};
use crate::bytecode::{
    BytecodeError, BytecodeProgram, CodeRange, ForeignEntry, Instruction, ProcedureEntry,
    QuoteEntry, SourceMapEntry, MAX_ARTIFACT_BYTES, MAX_FOREIGNS, MAX_INSTRUCTIONS, MAX_PROCEDURES,
    MAX_QUOTES, MAX_SOURCE_MAP_ENTRIES,
};
use crate::hir::{ForeignId, HirProgram, ProcedureId};
use crate::package::CURRENT_RUNTIME_ABI;
use crate::source::{SourceId, SourceManager, SourceSpan};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::fmt;

/// Link-visible symbol category. Cross-category name reuse is rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SymbolKind {
    /// Retained MANIFEST word.
    Constant,
    /// Ourochronos procedure entry.
    Procedure,
    /// Stored quotation entry.
    Quote,
    /// Narrow foreign ABI entry.
    Foreign,
}

/// Exact stack/effect shape used when resolving an import.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymbolSignature {
    /// Words consumed by the callable.
    pub inputs: u32,
    /// Words returned by the callable.
    pub outputs: u32,
    /// Stable effect-category bit set owned by the object producer.
    pub effects: u64,
    /// Exact category-specific ABI fingerprint. Language procedures/quotes use
    /// zero; foreign entries encode every scalar parameter and result type.
    pub abi: u64,
}

impl SymbolSignature {
    /// Construct a pure stack signature.
    pub const fn pure(inputs: u32, outputs: u32) -> Self {
        Self {
            inputs,
            outputs,
            effects: 0,
            abi: 0,
        }
    }

    /// Construct the exact link signature retained by a foreign descriptor.
    pub fn foreign(entry: &ForeignEntry) -> Self {
        const FOREIGN_MARKER: u64 = 1 << 63;
        let mut abi = FOREIGN_MARKER | u64::from(entry.result.is_some());
        if matches!(entry.result, Some(crate::bytecode::ForeignScalarType::I64)) {
            abi |= 1 << 1;
        }
        for (index, parameter) in entry.parameters.iter().enumerate() {
            if matches!(parameter, crate::bytecode::ForeignScalarType::I64) {
                abi |= 1 << (index + 2);
            }
        }
        Self {
            inputs: entry.parameters.len() as u32,
            outputs: u32::from(entry.result.is_some()),
            effects: if entry.effects.is_pure() {
                0
            } else {
                u64::from(entry.effects.bits() & !crate::bytecode::ForeignEffects::PURE)
            },
            abi,
        }
    }
}

/// A local typed definition exported under a link-visible name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectExport {
    /// Globally resolved symbol name.
    pub name: String,
    /// Local object target.
    pub target: SymbolTarget,
    /// Exact callable/data shape.
    pub signature: SymbolSignature,
}

/// An external definition required by an object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectImport {
    /// Globally resolved symbol name.
    pub name: String,
    /// Required category.
    pub kind: SymbolKind,
    /// Required exact shape.
    pub signature: SymbolSignature,
}

/// A typed local or linked target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolTarget {
    /// Link-time MANIFEST word value.
    Constant(u64),
    /// Procedure table identity.
    Procedure(ProcedureId),
    /// Quotation table identity.
    Quote(QuoteId),
    /// Foreign table identity.
    Foreign(ForeignId),
}

impl SymbolTarget {
    /// Category of this target.
    pub const fn kind(self) -> SymbolKind {
        match self {
            Self::Constant(_) => SymbolKind::Constant,
            Self::Procedure(_) => SymbolKind::Procedure,
            Self::Quote(_) => SymbolKind::Quote,
            Self::Foreign(_) => SymbolKind::Foreign,
        }
    }
}

/// The operand form replaced at a symbolic relocation site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelocationKind {
    /// Replace a `PushWord` operand with an exported MANIFEST value.
    ConstantLiteral,
    /// Replace a `CallProcedure` operand.
    ProcedureCall,
    /// Replace a `PushQuote` operand.
    QuoteLiteral,
    /// Replace a `CallForeign` operand.
    ForeignCall,
}

impl RelocationKind {
    const fn symbol_kind(self) -> SymbolKind {
        match self {
            Self::ConstantLiteral => SymbolKind::Constant,
            Self::ProcedureCall => SymbolKind::Procedure,
            Self::QuoteLiteral => SymbolKind::Quote,
            Self::ForeignCall => SymbolKind::Foreign,
        }
    }
}

/// One absolute object instruction whose typed operand names an import.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectRelocation {
    /// Absolute instruction index in [`ObjectModule::code`].
    pub instruction: u32,
    /// Index into [`ObjectModule::imports`].
    pub import: u32,
    /// Typed operand to replace.
    pub kind: RelocationKind,
}

/// Executable target described by an object artifact.
///
/// Target tags are closed: a decoder never turns an unknown target into a
/// best-effort host assumption.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ObjectTarget {
    /// Ourochronos' portable 64-bit little-endian bytecode machine.
    PortableVm64Le,
}

/// Stable effect categories summarized over a complete compiled module.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ObjectEffects(u64);

impl ObjectEffects {
    /// Oracle, prophecy, present-state, or paradox operations.
    pub const TEMPORAL: u64 = 1 << 0;
    /// Transaction-buffered language output.
    pub const BUFFERED_OUTPUT: u64 = 1 << 1;
    /// Runtime input.
    pub const INPUT: u64 = 1 << 2;
    /// File, network, process, system, or other externally visible I/O.
    pub const EXTERNAL_IO: u64 = 1 << 3;
    /// Clock, random, or another non-deterministic source.
    pub const NONDETERMINISTIC: u64 = 1 << 4;
    /// Dynamic collection or buffer allocation/manipulation.
    pub const ALLOCATION: u64 = 1 << 5;
    /// A linked narrow foreign call.
    pub const FOREIGN: u64 = 1 << 6;
    const ALL: u64 = Self::TEMPORAL
        | Self::BUFFERED_OUTPUT
        | Self::INPUT
        | Self::EXTERNAL_IO
        | Self::NONDETERMINISTIC
        | Self::ALLOCATION
        | Self::FOREIGN;

    /// Construct a checked stable bit set.
    pub const fn from_bits(bits: u64) -> Option<Self> {
        if bits & !Self::ALL == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }

    /// Raw stable artifact bits.
    pub const fn bits(self) -> u64 {
        self.0
    }

    const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// Whole-module declared and bytecode-observed effect summary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ObjectEffectSummary {
    /// Effects declared by source procedures and foreign declarations.
    pub declared: ObjectEffects,
    /// Effects exhaustively observed in the retained bytecode.
    pub observed: ObjectEffects,
}

/// One finite temporal region in instruction order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectTemporalRegion {
    /// `TemporalEnter` instruction carrying this schema.
    pub instruction: u32,
    /// First temporal-memory address.
    pub base: u64,
    /// Number of cells.
    pub cells: u64,
    /// Stored width of every cell.
    pub cell_bits: u8,
}

/// One all-fixed-state property retained independently of source syntax.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectProperty {
    /// Source declaration name.
    pub name: String,
    /// Temporal cell address compared by the property.
    pub address: u64,
    /// Unsigned word comparison.
    pub comparison: PropertyComparison,
    /// Right-hand literal word.
    pub value: u64,
    /// Extended bounded Boolean predicate; `None` is the legacy atom above.
    pub predicate: Option<PropertyPredicate>,
    /// Sorted exact address slice mentioned by the predicate.
    pub touched_addresses: Vec<u64>,
}

/// One retained named temporal-cell schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectNamedTemporal {
    pub name: String,
    pub address: u64,
    /// Declaration metadata only; present still starts at zero.
    pub default: u64,
}

/// One source file referenced by bytecode source-map entries.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ObjectSourceFile {
    /// Module-local source identity. Entries are canonical and contiguous.
    pub id: u32,
    /// Diagnostic display name, not a filesystem authority.
    pub name: String,
    /// Exact UTF-8 byte length at compile time.
    pub byte_len: u64,
    /// Stable FNV-1a digest of the exact UTF-8 source bytes.
    pub content_digest: u64,
}

/// Kind of optional verification payload carried by one object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VerificationArtifactKind {
    /// Deterministic bytecode-verifier report or proof transcript.
    BytecodeReport,
    /// Solver certificate or counterexample artifact.
    SolverCertificate,
}

/// Opaque, bounded, explicitly versioned verification payload.
///
/// The linker deliberately invalidates this section after rewriting code;
/// consumers must interpret it according to `kind` and `format_version`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ObjectVerificationArtifact {
    /// Closed payload category.
    pub kind: VerificationArtifactKind,
    /// Category-specific nonzero encoding version.
    pub format_version: u16,
    /// Bounded exact payload bytes.
    pub payload: Vec<u8>,
}

/// Stable compiler metadata retained next to relocatable code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectMetadata {
    /// Exact executable target.
    pub target: ObjectTarget,
    /// Required portable runtime ABI.
    pub runtime_abi: u16,
    /// Sorted unique literal-word pool, exactly matching `PushWord` operands.
    pub constants: Vec<u64>,
    /// Exact `TemporalEnter` schemas in instruction order.
    pub temporal_regions: Vec<ObjectTemporalRegion>,
    /// Named temporal-cell schemas exported by this source object.
    pub named_temporals: Vec<ObjectNamedTemporal>,
    /// Source properties in deterministic declaration order.
    pub properties: Vec<ObjectProperty>,
    /// Whole-module effects.
    pub effects: ObjectEffectSummary,
    /// Canonical local source manifest. Instruction spans remain in bytecode.
    pub source_files: Vec<ObjectSourceFile>,
    /// Optional proof material for this exact pre-link object.
    pub verification: Option<ObjectVerificationArtifact>,
}

/// A linked executable plus metadata preserved or rebuilt across relocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkedProgram {
    /// Validated linked executable bytecode.
    pub code: BytecodeProgram,
    /// Metadata whose instruction/source identities match `code`.
    pub metadata: ObjectMetadata,
}

/// One relocatable object. `code` may be invalid only because operands at
/// declared relocation sites are unresolved; all other structure is checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectModule {
    /// Stable module name used for deterministic link ordering.
    pub name: String,
    /// Local bytecode and tables.
    pub code: BytecodeProgram,
    /// Definitions made visible to other objects.
    pub exports: Vec<ObjectExport>,
    /// Required external definitions.
    pub imports: Vec<ObjectImport>,
    /// Symbolic operand replacements.
    pub relocations: Vec<ObjectRelocation>,
    /// Stable compiled-module metadata.
    pub metadata: ObjectMetadata,
}

const OBJECT_MAGIC: &[u8; 8] = b"OUROOBJ\0";
const OBJECT_VERSION: u16 = 4;
const OBJECT_FLAGS: u16 = 0;

/// Maximum encoded size of one relocatable object.
pub const MAX_OBJECT_BYTES: usize = 128 * 1024 * 1024;
/// Maximum number of object modules accepted by one public link operation.
pub const MAX_LINK_OBJECTS: usize = 4_096;
/// Maximum UTF-8 byte length of a module or link-visible symbol name.
pub const MAX_OBJECT_NAME_BYTES: usize = 4 * 1024;
/// Maximum number of exports in one object.
pub const MAX_OBJECT_EXPORTS: usize = 100_000;
/// Maximum number of imports in one object.
pub const MAX_OBJECT_IMPORTS: usize = 100_000;
/// Maximum number of relocations in one object.
pub const MAX_OBJECT_RELOCATIONS: usize = MAX_INSTRUCTIONS;
/// Maximum number of literal words in a deterministic object constant pool.
pub const MAX_OBJECT_CONSTANTS: usize = MAX_INSTRUCTIONS;
/// Maximum number of finite temporal-region schemas.
pub const MAX_OBJECT_TEMPORAL_REGIONS: usize = MAX_INSTRUCTIONS;
/// Maximum number of retained named temporal schemas.
pub const MAX_OBJECT_NAMED_TEMPORALS: usize = 100_000;
/// Maximum number of retained property declarations.
pub const MAX_OBJECT_PROPERTIES: usize = 100_000;
/// Maximum number of retained source-file manifest entries.
pub const MAX_OBJECT_SOURCE_FILES: usize = 100_000;
/// Maximum UTF-8 byte length of a source display name.
pub const MAX_OBJECT_SOURCE_NAME_BYTES: usize = 16 * 1024;
/// Maximum optional proof/certificate payload size.
pub const MAX_OBJECT_VERIFICATION_BYTES: usize = 1024 * 1024;

/// Relocatable object encoding, decoding, or validation failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObjectError {
    /// A bounded byte or table count exceeded its hard limit.
    LimitExceeded {
        /// Limited entity.
        what: &'static str,
        /// Observed count.
        count: usize,
        /// Maximum accepted count.
        limit: usize,
    },
    /// Object magic was absent or incorrect.
    BadMagic,
    /// Object version is not supported.
    UnsupportedVersion(u16),
    /// Reserved format flags were nonzero.
    UnsupportedFlags(u16),
    /// Input ended before a complete field could be decoded.
    Truncated {
        /// Byte offset at which the field began.
        offset: usize,
        /// Bytes required by the field.
        needed: usize,
    },
    /// Bytes remained after the declared object ended.
    TrailingBytes(usize),
    /// A module or symbol name was not valid UTF-8.
    InvalidUtf8 {
        /// Name field being decoded.
        field: &'static str,
    },
    /// A serialized symbol-kind tag is unknown.
    UnknownSymbolKind(u8),
    /// A serialized symbol-target tag is unknown.
    UnknownSymbolTarget(u8),
    /// A serialized relocation-kind tag is unknown.
    UnknownRelocationKind(u8),
    /// A serialized target tag is unknown.
    UnknownTarget(u16),
    /// A serialized property-comparison tag is unknown.
    UnknownPropertyComparison(u8),
    /// A serialized verification-artifact tag is unknown.
    UnknownVerificationKind(u8),
    /// Stable effect bits included an undefined category.
    UnknownEffectFlags(u64),
    /// Compiler metadata disagreed with its retained code/HIR/source inputs.
    InvalidMetadata(String),
    /// A string or table count could not fit the object format.
    IndexOverflow(&'static str),
    /// Embedded local bytecode was malformed.
    Bytecode(BytecodeError),
    /// Object-aware structural validation failed.
    InvalidObject(LinkError),
}

impl fmt::Display for ObjectError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LimitExceeded { what, count, limit } => {
                write!(
                    formatter,
                    "object {what} count {count} exceeds limit {limit}"
                )
            }
            Self::BadMagic => write!(formatter, "invalid object magic"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported object version {version}")
            }
            Self::UnsupportedFlags(flags) => {
                write!(formatter, "unsupported object flags 0x{flags:04x}")
            }
            Self::Truncated { offset, needed } => write!(
                formatter,
                "truncated object at offset {offset}: need {needed} bytes"
            ),
            Self::TrailingBytes(count) => write!(formatter, "{count} trailing object bytes"),
            Self::InvalidUtf8 { field } => write!(formatter, "object {field} is not valid UTF-8"),
            Self::UnknownSymbolKind(tag) => write!(formatter, "unknown symbol-kind tag {tag}"),
            Self::UnknownSymbolTarget(tag) => {
                write!(formatter, "unknown symbol-target tag {tag}")
            }
            Self::UnknownRelocationKind(tag) => {
                write!(formatter, "unknown relocation-kind tag {tag}")
            }
            Self::UnknownTarget(tag) => write!(formatter, "unknown object target tag {tag}"),
            Self::UnknownPropertyComparison(tag) => {
                write!(formatter, "unknown property-comparison tag {tag}")
            }
            Self::UnknownVerificationKind(tag) => {
                write!(formatter, "unknown verification-artifact tag {tag}")
            }
            Self::UnknownEffectFlags(flags) => {
                write!(formatter, "unknown object effect flags 0x{flags:016x}")
            }
            Self::InvalidMetadata(message) => {
                write!(formatter, "invalid object metadata: {message}")
            }
            Self::IndexOverflow(what) => write!(formatter, "object {what} index overflow"),
            Self::Bytecode(error) => write!(formatter, "embedded bytecode: {error}"),
            Self::InvalidObject(error) => write!(formatter, "{error}"),
        }
    }
}

impl Error for ObjectError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Bytecode(error) => Some(error),
            Self::InvalidObject(error) => Some(error),
            _ => None,
        }
    }
}

impl From<BytecodeError> for ObjectError {
    fn from(value: BytecodeError) -> Self {
        Self::Bytecode(value)
    }
}

impl ObjectMetadata {
    /// Rebuild all metadata that can be proven from bytecode alone.
    pub fn for_bytecode(code: &BytecodeProgram) -> Self {
        Self {
            target: ObjectTarget::PortableVm64Le,
            runtime_abi: CURRENT_RUNTIME_ABI,
            constants: bytecode_constants(code),
            temporal_regions: bytecode_temporal_regions(code),
            named_temporals: Vec::new(),
            properties: Vec::new(),
            effects: ObjectEffectSummary {
                declared: ObjectEffects::default(),
                observed: observed_effects(code),
            },
            source_files: synthetic_source_manifest(code),
            verification: None,
        }
    }

    /// Deterministic named-cell relevance slice for one retained property.
    /// This reports metadata only; it does not claim incremental solver reuse.
    pub fn property_named_slice(&self, property_name: &str) -> Option<Vec<ObjectNamedTemporal>> {
        let property = self
            .properties
            .iter()
            .find(|property| property.name.eq_ignore_ascii_case(property_name))?;
        let addresses = &property.touched_addresses;
        let mut temporals = self
            .named_temporals
            .iter()
            .filter(|temporal| addresses.binary_search(&temporal.address).is_ok())
            .cloned()
            .collect::<Vec<_>>();
        temporals.sort_by(|left, right| {
            left.address
                .cmp(&right.address)
                .then_with(|| left.name.cmp(&right.name))
        });
        Some(temporals)
    }
}

fn bytecode_constants(code: &BytecodeProgram) -> Vec<u64> {
    let mut constants = code
        .instructions
        .iter()
        .filter_map(|instruction| match instruction {
            Instruction::PushWord(value) => Some(*value),
            _ => None,
        })
        .collect::<Vec<_>>();
    constants.sort_unstable();
    constants.dedup();
    constants
}

fn bytecode_temporal_regions(code: &BytecodeProgram) -> Vec<ObjectTemporalRegion> {
    code.instructions
        .iter()
        .enumerate()
        .filter_map(|(instruction, opcode)| match opcode {
            Instruction::TemporalEnter {
                base,
                size,
                cell_bits,
                ..
            } => Some(ObjectTemporalRegion {
                instruction: instruction as u32,
                base: *base,
                cells: *size,
                cell_bits: *cell_bits,
            }),
            _ => None,
        })
        .collect()
}

fn observed_effects(code: &BytecodeProgram) -> ObjectEffects {
    let mut effects = ObjectEffects::default();
    for instruction in &code.instructions {
        match instruction {
            Instruction::Primitive(opcode) => {
                let broad = match opcode.effect_class() {
                    EffectClass::Pure => ObjectEffects::default(),
                    EffectClass::External => ObjectEffects(ObjectEffects::EXTERNAL_IO),
                    EffectClass::NonDeterministic => ObjectEffects(ObjectEffects::NONDETERMINISTIC),
                };
                effects = effects.union(broad).union(opcode_effects(*opcode));
            }
            Instruction::TemporalEnter { .. } | Instruction::TemporalExit { .. } => {
                effects = effects.union(ObjectEffects(ObjectEffects::TEMPORAL));
            }
            Instruction::CallForeign(id) => {
                effects = effects.union(ObjectEffects(ObjectEffects::FOREIGN));
                if let Some(foreign) = code.foreigns.get(id.index()) {
                    effects = effects.union(foreign_effects(foreign));
                }
            }
            _ => {}
        }
    }
    effects
}

fn opcode_effects(opcode: OpCode) -> ObjectEffects {
    use OpCode::*;
    let bits = match opcode {
        Oracle | Prophecy | PresentRead | Paradox => ObjectEffects::TEMPORAL,
        Output | Emit => ObjectEffects::BUFFERED_OUTPUT,
        Input => ObjectEffects::INPUT,
        VecNew | VecPush | VecPop | VecGet | VecSet | VecLen | HashNew | HashPut | HashGet
        | HashDel | HashHas | HashLen | SetNew | SetAdd | SetHas | SetDel | SetLen | BufferNew
        | BufferFromStack | BufferToStack | BufferLen | BufferReadByte | BufferWriteByte
        | BufferFree => ObjectEffects::ALLOCATION,
        FFICall | FFICallNamed => ObjectEffects::FOREIGN | ObjectEffects::EXTERNAL_IO,
        _ => 0,
    };
    ObjectEffects(bits)
}

fn foreign_effects(foreign: &ForeignEntry) -> ObjectEffects {
    use crate::bytecode::ForeignEffects;
    let raw = foreign.effects.bits();
    let mut bits = 0;
    if raw & ForeignEffects::IO != 0
        || raw & ForeignEffects::READS != 0
        || raw & ForeignEffects::WRITES != 0
    {
        bits |= ObjectEffects::EXTERNAL_IO;
    }
    if raw & ForeignEffects::TEMPORAL != 0 {
        bits |= ObjectEffects::TEMPORAL;
    }
    if raw & ForeignEffects::ALLOC != 0 {
        bits |= ObjectEffects::ALLOCATION;
    }
    ObjectEffects(bits)
}

fn declared_effects(hir: &HirProgram) -> ObjectEffects {
    let mut bits = 0;
    for effect in hir
        .procedures
        .iter()
        .flat_map(|procedure| procedure.effects.iter())
    {
        bits |= match effect {
            Effect::Pure => 0,
            Effect::Reads(_) | Effect::Writes(_) | Effect::Temporal => ObjectEffects::TEMPORAL,
            Effect::IO => ObjectEffects::INPUT | ObjectEffects::BUFFERED_OUTPUT,
            Effect::Alloc => ObjectEffects::ALLOCATION,
            Effect::FFI => ObjectEffects::FOREIGN | ObjectEffects::EXTERNAL_IO,
            Effect::FileIO | Effect::Network | Effect::System => ObjectEffects::EXTERNAL_IO,
        };
    }
    for foreign in &hir.foreigns {
        use crate::runtime::ffi::FFIEffect;
        for effect in &foreign.declaration.signature.effects {
            bits |= match effect {
                FFIEffect::Pure => 0,
                FFIEffect::IO | FFIEffect::Reads | FFIEffect::Writes => ObjectEffects::EXTERNAL_IO,
                FFIEffect::Temporal => ObjectEffects::TEMPORAL,
                FFIEffect::Alloc => ObjectEffects::ALLOCATION,
            };
        }
        // Every declared foreign is link-visible even when declared PURE.
        bits |= ObjectEffects::FOREIGN;
    }
    ObjectEffects(bits)
}

fn synthetic_source_manifest(code: &BytecodeProgram) -> Vec<ObjectSourceFile> {
    let Some(max_source) = code
        .source_map
        .iter()
        .map(|entry| entry.span.source.index())
        .max()
    else {
        return Vec::new();
    };
    // Do not let a sparse, attacker-constructed SourceId turn a compatibility
    // constructor into a huge dense allocation. The resulting empty manifest
    // is rejected by `validate_metadata` with a normal structural error.
    if max_source >= MAX_OBJECT_SOURCE_FILES {
        return Vec::new();
    }
    let mut byte_lengths = vec![0u64; max_source.saturating_add(1)];
    for entry in &code.source_map {
        let end = u64::try_from(entry.span.range.end).unwrap_or(u64::MAX);
        byte_lengths[entry.span.source.index()] = byte_lengths[entry.span.source.index()].max(end);
    }
    byte_lengths
        .into_iter()
        .enumerate()
        .map(|(id, byte_len)| ObjectSourceFile {
            id: id as u32,
            name: format!("<source:{id}>"),
            byte_len,
            content_digest: 0,
        })
        .collect()
}

fn source_manifest(sources: &SourceManager) -> Result<Vec<ObjectSourceFile>, ObjectError> {
    object_limit("source file", sources.len(), MAX_OBJECT_SOURCE_FILES)?;
    (0..sources.len())
        .map(|index| {
            let id = SourceId::new(index);
            let source = sources
                .get(id)
                .map_err(|error| ObjectError::InvalidMetadata(error.to_string()))?;
            object_limit(
                "source name",
                source.display_name().len(),
                MAX_OBJECT_SOURCE_NAME_BYTES,
            )?;
            Ok(ObjectSourceFile {
                id: u32::try_from(index).map_err(|_| ObjectError::IndexOverflow("source file"))?,
                name: source.display_name().to_string(),
                byte_len: u64::try_from(source.text().len())
                    .map_err(|_| ObjectError::IndexOverflow("source byte length"))?,
                content_digest: fnv1a64(source.text().as_bytes()),
            })
        })
        .collect()
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

impl ObjectModule {
    /// Construct a self-contained local object from bytecode alone.
    ///
    /// This compatibility constructor rebuilds all code-derived metadata. If
    /// source maps exist it creates explicit synthetic manifest entries using
    /// the largest referenced byte boundary; callers with a [`SourceManager`]
    /// should use [`Self::from_compiled`] to retain real names and digests.
    pub fn new(name: impl Into<String>, code: BytecodeProgram) -> Self {
        let metadata = ObjectMetadata::for_bytecode(&code);
        Self {
            name: name.into(),
            code,
            exports: Vec::new(),
            imports: Vec::new(),
            relocations: Vec::new(),
            metadata,
        }
    }

    /// Construct a self-contained compiled module from its typed HIR,
    /// bytecode, and exact source manager.
    ///
    /// This compatibility constructor expects one self-contained HIR/bytecode
    /// unit and records every contributing source. Canonical source graphs use
    /// [`crate::object_compiler::compile_objects`] and
    /// [`Self::from_relocatable_compiled`] instead.
    pub fn from_compiled(
        name: impl Into<String>,
        hir: &HirProgram,
        code: BytecodeProgram,
        sources: &SourceManager,
    ) -> Result<Self, ObjectError> {
        Self::from_relocatable_compiled(
            name,
            hir,
            code,
            sources,
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
    }

    /// Construct a compiled module with typed exports, imports, and symbolic
    /// operand relocations.
    ///
    /// Unlike [`Self::from_compiled`], validation is delayed until the symbol
    /// tables are installed, so imported procedure/quotation/foreign operands
    /// may remain outside local tables only at declared relocation sites.
    pub fn from_relocatable_compiled(
        name: impl Into<String>,
        hir: &HirProgram,
        code: BytecodeProgram,
        sources: &SourceManager,
        exports: Vec<ObjectExport>,
        imports: Vec<ObjectImport>,
        relocations: Vec<ObjectRelocation>,
    ) -> Result<Self, ObjectError> {
        if hir.procedures.len() != code.procedures.len()
            || hir.quotes.len() != code.quotations.len()
            || hir.foreigns.len() != code.foreigns.len()
        {
            return Err(ObjectError::InvalidMetadata(
                "HIR and bytecode typed table lengths differ".into(),
            ));
        }

        let mut metadata = ObjectMetadata::for_bytecode(&code);
        metadata.named_temporals = hir
            .temporals
            .iter()
            .map(|temporal| ObjectNamedTemporal {
                name: temporal.name.clone(),
                address: temporal.address,
                default: temporal.default,
            })
            .collect();
        metadata.properties = hir
            .properties
            .iter()
            .map(|property| ObjectProperty {
                name: property.declaration.name.clone(),
                address: property.declaration.address,
                comparison: property.declaration.comparison,
                value: property.declaration.value,
                predicate: property.declaration.predicate.clone(),
                touched_addresses: property.declaration.touched_addresses(),
            })
            .collect();
        metadata.effects.declared = declared_effects(hir);
        metadata.source_files = source_manifest(sources)?;

        let object = Self {
            name: name.into(),
            code,
            exports,
            imports,
            relocations,
            metadata,
        };
        object.validate().map_err(ObjectError::InvalidObject)?;
        Ok(object)
    }

    /// Validate local bytecode, typed symbol tables, and relocation coverage.
    ///
    /// Every non-relocated identity operand must resolve in a local typed
    /// table. Only a correctly typed instruction named by exactly one
    /// relocation may carry an unresolved identity.
    pub fn validate(&self) -> Result<(), LinkError> {
        validate_object(self)
    }

    /// Encode this validated object in deterministic versioned little-endian
    /// form. The embedded bytecode retains unresolved operands because the
    /// surrounding typed relocation table is their authority.
    pub fn to_bytes(&self) -> Result<Vec<u8>, ObjectError> {
        self.validate().map_err(ObjectError::InvalidObject)?;
        let code = self.code.to_relocatable_bytes()?;
        object_limit("embedded bytecode byte", code.len(), MAX_ARTIFACT_BYTES)?;

        let mut bytes = Vec::new();
        bytes.extend_from_slice(OBJECT_MAGIC);
        object_put_u16(&mut bytes, OBJECT_VERSION);
        object_put_u16(&mut bytes, OBJECT_FLAGS);
        object_put_u32(&mut bytes, object_count(self.name.len(), "module name")?);
        object_put_u32(&mut bytes, object_count(code.len(), "embedded bytecode")?);
        object_put_u32(&mut bytes, object_count(self.exports.len(), "export")?);
        object_put_u32(&mut bytes, object_count(self.imports.len(), "import")?);
        object_put_u32(
            &mut bytes,
            object_count(self.relocations.len(), "relocation")?,
        );
        object_put_u16(&mut bytes, encode_object_target(self.metadata.target));
        object_put_u16(&mut bytes, self.metadata.runtime_abi);
        object_put_u32(
            &mut bytes,
            object_count(self.metadata.constants.len(), "constant")?,
        );
        object_put_u32(
            &mut bytes,
            object_count(self.metadata.temporal_regions.len(), "temporal region")?,
        );
        object_put_u32(
            &mut bytes,
            object_count(self.metadata.properties.len(), "property")?,
        );
        object_put_u32(
            &mut bytes,
            object_count(self.metadata.source_files.len(), "source file")?,
        );
        object_put_u64(&mut bytes, self.metadata.effects.declared.bits());
        object_put_u64(&mut bytes, self.metadata.effects.observed.bits());
        let (verification_kind, verification_version, verification_len) = self
            .metadata
            .verification
            .as_ref()
            .map(|artifact| {
                (
                    encode_verification_kind(artifact.kind),
                    artifact.format_version,
                    artifact.payload.len(),
                )
            })
            .unwrap_or((0, 0, 0));
        bytes.push(verification_kind);
        object_put_u16(&mut bytes, verification_version);
        object_put_u32(
            &mut bytes,
            object_count(verification_len, "verification artifact byte")?,
        );
        bytes.extend_from_slice(self.name.as_bytes());
        bytes.extend_from_slice(&code);

        for export in &self.exports {
            encode_object_string(&mut bytes, &export.name, "export name")?;
            encode_symbol_target(&mut bytes, export.target);
            encode_signature(&mut bytes, export.signature);
        }
        for import in &self.imports {
            encode_object_string(&mut bytes, &import.name, "import name")?;
            bytes.push(encode_symbol_kind(import.kind));
            encode_signature(&mut bytes, import.signature);
        }
        for relocation in &self.relocations {
            object_put_u32(&mut bytes, relocation.instruction);
            object_put_u32(&mut bytes, relocation.import);
            bytes.push(encode_relocation_kind(relocation.kind));
        }
        for constant in &self.metadata.constants {
            object_put_u64(&mut bytes, *constant);
        }
        for region in &self.metadata.temporal_regions {
            object_put_u32(&mut bytes, region.instruction);
            object_put_u64(&mut bytes, region.base);
            object_put_u64(&mut bytes, region.cells);
            bytes.push(region.cell_bits);
        }
        for property in &self.metadata.properties {
            encode_object_string(&mut bytes, &property.name, "property name")?;
            object_put_u64(&mut bytes, property.address);
            bytes.push(encode_property_comparison(property.comparison));
            object_put_u64(&mut bytes, property.value);
            let predicate = encode_property_predicate(property.predicate.as_ref())?;
            object_put_u32(
                &mut bytes,
                object_count(predicate.len(), "property predicate byte")?,
            );
            bytes.extend_from_slice(&predicate);
            object_put_u32(
                &mut bytes,
                object_count(property.touched_addresses.len(), "property touched address")?,
            );
            for address in &property.touched_addresses {
                object_put_u64(&mut bytes, *address);
            }
        }
        for source in &self.metadata.source_files {
            object_put_u32(&mut bytes, source.id);
            encode_bounded_string(
                &mut bytes,
                &source.name,
                "source name",
                MAX_OBJECT_SOURCE_NAME_BYTES,
            )?;
            object_put_u64(&mut bytes, source.byte_len);
            object_put_u64(&mut bytes, source.content_digest);
        }
        object_put_u32(
            &mut bytes,
            object_count(self.metadata.named_temporals.len(), "named temporal")?,
        );
        for temporal in &self.metadata.named_temporals {
            encode_object_string(&mut bytes, &temporal.name, "named temporal name")?;
            object_put_u64(&mut bytes, temporal.address);
            object_put_u64(&mut bytes, temporal.default);
        }
        if let Some(verification) = &self.metadata.verification {
            bytes.extend_from_slice(&verification.payload);
        }

        object_limit("artifact byte", bytes.len(), MAX_OBJECT_BYTES)?;
        Ok(bytes)
    }

    /// Decode a bounded object, reject unknown tags and trailing bytes, then
    /// perform complete relocation-aware structural validation.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ObjectError> {
        object_limit("artifact byte", bytes.len(), MAX_OBJECT_BYTES)?;
        let mut reader = ObjectReader::new(bytes);
        if reader.take(OBJECT_MAGIC.len())? != OBJECT_MAGIC {
            return Err(ObjectError::BadMagic);
        }
        let version = reader.u16()?;
        if version != OBJECT_VERSION {
            return Err(ObjectError::UnsupportedVersion(version));
        }
        let flags = reader.u16()?;
        if flags != OBJECT_FLAGS {
            return Err(ObjectError::UnsupportedFlags(flags));
        }

        let module_name_len = reader.bounded_count("module name", MAX_OBJECT_NAME_BYTES)?;
        let code_len = reader.bounded_count("embedded bytecode byte", MAX_ARTIFACT_BYTES)?;
        let export_count = reader.bounded_count("export", MAX_OBJECT_EXPORTS)?;
        let import_count = reader.bounded_count("import", MAX_OBJECT_IMPORTS)?;
        let relocation_count = reader.bounded_count("relocation", MAX_OBJECT_RELOCATIONS)?;
        let target = decode_object_target(reader.u16()?)?;
        let runtime_abi = reader.u16()?;
        let constant_count = reader.bounded_count("constant", MAX_OBJECT_CONSTANTS)?;
        let temporal_region_count =
            reader.bounded_count("temporal region", MAX_OBJECT_TEMPORAL_REGIONS)?;
        let property_count = reader.bounded_count("property", MAX_OBJECT_PROPERTIES)?;
        let source_count = reader.bounded_count("source file", MAX_OBJECT_SOURCE_FILES)?;
        let declared_effects = decode_object_effects(reader.u64()?)?;
        let observed_effects = decode_object_effects(reader.u64()?)?;
        let verification_kind = reader.u8()?;
        let verification_version = reader.u16()?;
        let verification_len =
            reader.bounded_count("verification artifact byte", MAX_OBJECT_VERIFICATION_BYTES)?;

        let name = reader.string_with_len(module_name_len, "module name")?;
        let code = BytecodeProgram::from_relocatable_bytes(reader.take(code_len)?)?;

        let mut exports = Vec::with_capacity(export_count);
        for _ in 0..export_count {
            exports.push(ObjectExport {
                name: reader.string("export name")?,
                target: decode_symbol_target(&mut reader)?,
                signature: decode_signature(&mut reader)?,
            });
        }

        let mut imports = Vec::with_capacity(import_count);
        for _ in 0..import_count {
            imports.push(ObjectImport {
                name: reader.string("import name")?,
                kind: decode_symbol_kind(reader.u8()?)?,
                signature: decode_signature(&mut reader)?,
            });
        }

        let mut relocations = Vec::with_capacity(relocation_count);
        for _ in 0..relocation_count {
            relocations.push(ObjectRelocation {
                instruction: reader.u32()?,
                import: reader.u32()?,
                kind: decode_relocation_kind(reader.u8()?)?,
            });
        }

        let mut constants = Vec::with_capacity(constant_count);
        for _ in 0..constant_count {
            constants.push(reader.u64()?);
        }
        let mut temporal_regions = Vec::with_capacity(temporal_region_count);
        for _ in 0..temporal_region_count {
            temporal_regions.push(ObjectTemporalRegion {
                instruction: reader.u32()?,
                base: reader.u64()?,
                cells: reader.u64()?,
                cell_bits: reader.u8()?,
            });
        }
        let mut properties = Vec::with_capacity(property_count);
        for _ in 0..property_count {
            let name = reader.string("property name")?;
            let address = reader.u64()?;
            let comparison = decode_property_comparison(reader.u8()?)?;
            let value = reader.u64()?;
            let predicate_len =
                reader.bounded_count("property predicate byte", MAX_OBJECT_VERIFICATION_BYTES)?;
            let predicate = decode_property_predicate(reader.take(predicate_len)?)?;
            let touched_count =
                reader.bounded_count("property touched address", MAX_OBJECT_NAMED_TEMPORALS)?;
            let mut touched_addresses = Vec::with_capacity(touched_count);
            for _ in 0..touched_count {
                touched_addresses.push(reader.u64()?);
            }
            properties.push(ObjectProperty {
                name,
                address,
                comparison,
                value,
                predicate,
                touched_addresses,
            });
        }
        let mut source_files = Vec::with_capacity(source_count);
        for _ in 0..source_count {
            source_files.push(ObjectSourceFile {
                id: reader.u32()?,
                name: reader.bounded_string("source name", MAX_OBJECT_SOURCE_NAME_BYTES)?,
                byte_len: reader.u64()?,
                content_digest: reader.u64()?,
            });
        }
        let named_temporal_count =
            reader.bounded_count("named temporal", MAX_OBJECT_NAMED_TEMPORALS)?;
        let mut named_temporals = Vec::with_capacity(named_temporal_count);
        for _ in 0..named_temporal_count {
            named_temporals.push(ObjectNamedTemporal {
                name: reader.string("named temporal name")?,
                address: reader.u64()?,
                default: reader.u64()?,
            });
        }
        let verification = decode_verification_artifact(
            verification_kind,
            verification_version,
            reader.take(verification_len)?.to_vec(),
        )?;

        if reader.remaining() != 0 {
            return Err(ObjectError::TrailingBytes(reader.remaining()));
        }

        let object = Self {
            name,
            code,
            exports,
            imports,
            relocations,
            metadata: ObjectMetadata {
                target,
                runtime_abi,
                constants,
                temporal_regions,
                named_temporals,
                properties,
                effects: ObjectEffectSummary {
                    declared: declared_effects,
                    observed: observed_effects,
                },
                source_files,
                verification,
            },
        };
        object.validate().map_err(ObjectError::InvalidObject)?;
        Ok(object)
    }
}

/// Deterministic link failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkError {
    /// Module names form object identity and must be unique and nonempty.
    InvalidModuleName(String),
    /// Two objects used the same module name.
    DuplicateModule(String),
    /// An object requires a runtime ABI this linker cannot produce.
    UnsupportedRuntimeAbi {
        /// Owning module.
        module: String,
        /// Required ABI.
        required: u16,
        /// ABI implemented by the linked runtime.
        supported: u16,
    },
    /// Input objects describe incompatible executable targets.
    IncompatibleTarget {
        /// Owning module.
        module: String,
        /// Target established by the first object.
        expected: ObjectTarget,
        /// Conflicting target.
        actual: ObjectTarget,
    },
    /// An object exceeded a bytecode table limit.
    LimitExceeded {
        /// Limited entity.
        what: &'static str,
        /// Observed count.
        count: usize,
        /// Maximum allowed count.
        limit: usize,
    },
    /// An object table/range/relocation is malformed.
    MalformedObject {
        /// Owning module.
        module: String,
        /// Exact finding.
        message: String,
    },
    /// Two exports, including cross-kind exports, used the same name.
    DuplicateSymbol(String),
    /// No export provides this import.
    UnresolvedSymbol {
        /// Importing module.
        module: String,
        /// Required symbol.
        symbol: String,
    },
    /// Import and export categories differ.
    SymbolKindMismatch {
        /// Symbol name.
        symbol: String,
        /// Required category.
        expected: SymbolKind,
        /// Exported category.
        actual: SymbolKind,
    },
    /// Import and export signatures differ.
    SignatureMismatch {
        /// Symbol name.
        symbol: String,
        /// Required signature.
        expected: SymbolSignature,
        /// Exported signature.
        actual: SymbolSignature,
    },
    /// A linked identity or target could not fit the bytecode format.
    IndexOverflow(&'static str),
    /// Final executable structural validation failed.
    InvalidLinkedBytecode(BytecodeError),
}

impl fmt::Display for LinkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidModuleName(name) => write!(formatter, "invalid module name {name:?}"),
            Self::DuplicateModule(name) => write!(formatter, "duplicate module {name:?}"),
            Self::UnsupportedRuntimeAbi {
                module,
                required,
                supported,
            } => write!(
                formatter,
                "module {module:?} requires runtime ABI {required}, but this linker supports {supported}"
            ),
            Self::IncompatibleTarget {
                module,
                expected,
                actual,
            } => write!(
                formatter,
                "module {module:?} targets {actual:?}, incompatible with {expected:?}"
            ),
            Self::LimitExceeded { what, count, limit } => {
                write!(
                    formatter,
                    "linked {what} count {count} exceeds limit {limit}"
                )
            }
            Self::MalformedObject { module, message } => {
                write!(formatter, "malformed object {module:?}: {message}")
            }
            Self::DuplicateSymbol(symbol) => write!(formatter, "duplicate symbol {symbol:?}"),
            Self::UnresolvedSymbol { module, symbol } => {
                write!(
                    formatter,
                    "module {module:?} has unresolved symbol {symbol:?}"
                )
            }
            Self::SymbolKindMismatch {
                symbol,
                expected,
                actual,
            } => write!(
                formatter,
                "symbol {symbol:?} has kind {actual:?}, expected {expected:?}"
            ),
            Self::SignatureMismatch {
                symbol,
                expected,
                actual,
            } => write!(
                formatter,
                "symbol {symbol:?} has signature {actual:?}, expected {expected:?}"
            ),
            Self::IndexOverflow(what) => write!(formatter, "linked {what} index overflow"),
            Self::InvalidLinkedBytecode(error) => write!(formatter, "linked bytecode: {error}"),
        }
    }
}

impl Error for LinkError {}

impl From<BytecodeError> for LinkError {
    fn from(value: BytecodeError) -> Self {
        Self::InvalidLinkedBytecode(value)
    }
}

#[derive(Debug, Clone)]
struct ModuleMaps {
    procedures: Vec<ProcedureId>,
    quotes: Vec<QuoteId>,
    foreigns: Vec<ForeignId>,
    sources_base: usize,
}

#[derive(Debug, Clone, Copy)]
struct ResolvedExport {
    target: SymbolTarget,
    signature: SymbolSignature,
}

/// Link relocatable objects into one structurally validated executable.
/// Input order is irrelevant; stable module-name order determines layout.
pub fn link(objects: &[ObjectModule]) -> Result<BytecodeProgram, LinkError> {
    Ok(link_with_metadata(objects)?.code)
}

/// Link relocatable objects while retaining metadata whose identities are
/// rebuilt to match the final executable.
///
/// Verification artifacts are intentionally cleared because relocation and
/// main-body concatenation change the code they certified.
pub fn link_with_metadata(objects: &[ObjectModule]) -> Result<LinkedProgram, LinkError> {
    check_limit("link object", objects.len(), MAX_LINK_OBJECTS)?;
    let mut order: Vec<usize> = (0..objects.len()).collect();
    order.sort_by(|left, right| objects[*left].name.cmp(&objects[*right].name));

    let mut previous_name: Option<&str> = None;
    let expected_target = order
        .first()
        .map(|index| objects[*index].metadata.target)
        .unwrap_or(ObjectTarget::PortableVm64Le);
    for &object_index in &order {
        let object = &objects[object_index];
        if object.name.trim().is_empty() {
            return Err(LinkError::InvalidModuleName(object.name.clone()));
        }
        if previous_name == Some(object.name.as_str()) {
            return Err(LinkError::DuplicateModule(object.name.clone()));
        }
        previous_name = Some(&object.name);
        validate_object(object)?;
        if object.metadata.target != expected_target {
            return Err(LinkError::IncompatibleTarget {
                module: object.name.clone(),
                expected: expected_target,
                actual: object.metadata.target,
            });
        }
        if object.metadata.runtime_abi != CURRENT_RUNTIME_ABI {
            return Err(LinkError::UnsupportedRuntimeAbi {
                module: object.name.clone(),
                required: object.metadata.runtime_abi,
                supported: CURRENT_RUNTIME_ABI,
            });
        }
    }

    let (instruction_capacity, source_map_capacity) = preflight_link_sizes(objects, &order)?;

    let maps = allocate_maps(objects, &order)?;
    let exports = collect_exports(objects, &order, &maps)?;
    let resolutions = resolve_imports(objects, &order, &exports)?;

    let mut builder = LinkBuilder::new(
        objects,
        &maps,
        &resolutions,
        instruction_capacity,
        source_map_capacity,
    );
    let main_start = 0;
    for &object_index in &order {
        builder.append_range(object_index, objects[object_index].code.main, true)?;
    }
    builder.instructions.push(Instruction::Return);
    let main = CodeRange {
        start: main_start,
        end: u32_len(builder.instructions.len(), "main")?,
    };

    let mut quotations = Vec::new();
    for &object_index in &order {
        for entry in &objects[object_index].code.quotations {
            let range = builder.append_range(object_index, entry.range, false)?;
            quotations.push(QuoteEntry {
                id: maps[object_index].quotes[entry.id.as_u64() as usize],
                range,
            });
        }
    }

    let mut procedures = Vec::new();
    for &object_index in &order {
        for entry in &objects[object_index].code.procedures {
            let range = builder.append_range(object_index, entry.range, false)?;
            procedures.push(ProcedureEntry {
                id: maps[object_index].procedures[entry.id.index()],
                range,
            });
        }
    }

    check_limit("instruction", builder.instructions.len(), MAX_INSTRUCTIONS)?;
    check_limit(
        "source map",
        builder.source_map.len(),
        MAX_SOURCE_MAP_ENTRIES,
    )?;
    let foreigns = linked_foreigns(objects, &order, &maps)?;
    let program = BytecodeProgram {
        instructions: builder.instructions,
        main,
        quotations,
        procedures,
        foreigns,
        source_map: builder.source_map,
    };
    program.validate()?;
    let metadata = linked_metadata(objects, &order, &maps, &program)?;
    Ok(LinkedProgram {
        code: program,
        metadata,
    })
}

fn preflight_link_sizes(
    objects: &[ObjectModule],
    order: &[usize],
) -> Result<(usize, usize), LinkError> {
    let mut instructions = 1usize; // One linked-main Return.
    let mut source_maps = 0usize;
    let mut exports = 0usize;
    let mut imports = 0usize;
    let mut named_temporals = 0usize;
    let mut properties = 0usize;
    let mut source_files = 0usize;
    for &index in order {
        let object = &objects[index];
        let main_len = usize::try_from(object.code.main.end - object.code.main.start)
            .map_err(|_| LinkError::IndexOverflow("main instruction count"))?;
        instructions = instructions
            .checked_add(main_len.saturating_sub(1))
            .ok_or(LinkError::IndexOverflow("aggregate instruction count"))?;
        for entry in &object.code.quotations {
            instructions = instructions
                .checked_add((entry.range.end - entry.range.start) as usize)
                .ok_or(LinkError::IndexOverflow("aggregate instruction count"))?;
        }
        for entry in &object.code.procedures {
            instructions = instructions
                .checked_add((entry.range.end - entry.range.start) as usize)
                .ok_or(LinkError::IndexOverflow("aggregate instruction count"))?;
        }
        source_maps = source_maps
            .checked_add(object.code.source_map.len())
            .ok_or(LinkError::IndexOverflow("aggregate source map count"))?;
        exports = exports
            .checked_add(object.exports.len())
            .ok_or(LinkError::IndexOverflow("aggregate export count"))?;
        imports = imports
            .checked_add(object.imports.len())
            .ok_or(LinkError::IndexOverflow("aggregate import count"))?;
        named_temporals = named_temporals
            .checked_add(object.metadata.named_temporals.len())
            .ok_or(LinkError::IndexOverflow("aggregate named temporal count"))?;
        properties = properties
            .checked_add(object.metadata.properties.len())
            .ok_or(LinkError::IndexOverflow("aggregate property count"))?;
        source_files = source_files
            .checked_add(object.metadata.source_files.len())
            .ok_or(LinkError::IndexOverflow("aggregate source file count"))?;
    }
    check_limit("instruction", instructions, MAX_INSTRUCTIONS)?;
    check_limit("source map", source_maps, MAX_SOURCE_MAP_ENTRIES)?;
    check_limit("export", exports, MAX_OBJECT_EXPORTS)?;
    check_limit("import", imports, MAX_OBJECT_IMPORTS)?;
    check_limit(
        "named temporal",
        named_temporals,
        MAX_OBJECT_NAMED_TEMPORALS,
    )?;
    check_limit("property", properties, MAX_OBJECT_PROPERTIES)?;
    check_limit("source file", source_files, MAX_OBJECT_SOURCE_FILES)?;
    Ok((instructions, source_maps))
}

fn linked_metadata(
    objects: &[ObjectModule],
    order: &[usize],
    maps: &[ModuleMaps],
    code: &BytecodeProgram,
) -> Result<ObjectMetadata, LinkError> {
    let mut metadata = ObjectMetadata::for_bytecode(code);
    metadata.target = order
        .first()
        .map(|index| objects[*index].metadata.target)
        .unwrap_or(ObjectTarget::PortableVm64Le);
    metadata.runtime_abi = CURRENT_RUNTIME_ABI;
    metadata.effects.declared = order
        .iter()
        .fold(ObjectEffects::default(), |effects, index| {
            effects.union(objects[*index].metadata.effects.declared)
        });
    let named_temporal_count = order
        .iter()
        .map(|index| objects[*index].metadata.named_temporals.len())
        .try_fold(0usize, |total, count| total.checked_add(count))
        .ok_or(LinkError::IndexOverflow("aggregate named temporal count"))?;
    check_limit(
        "named temporal",
        named_temporal_count,
        MAX_OBJECT_NAMED_TEMPORALS,
    )?;
    metadata.named_temporals = Vec::with_capacity(named_temporal_count);
    for &index in order {
        metadata
            .named_temporals
            .extend(objects[index].metadata.named_temporals.iter().cloned());
    }
    let mut named_temporal_names = HashSet::new();
    let mut named_temporal_addresses = HashSet::new();
    for temporal in &metadata.named_temporals {
        if !named_temporal_names.insert(temporal.name.to_lowercase()) {
            return Err(LinkError::DuplicateSymbol(temporal.name.clone()));
        }
        if !named_temporal_addresses.insert(temporal.address) {
            return Err(LinkError::MalformedObject {
                module: "<linked-metadata>".into(),
                message: format!("duplicate named temporal address {}", temporal.address),
            });
        }
    }
    let property_count = order
        .iter()
        .map(|index| objects[*index].metadata.properties.len())
        .try_fold(0usize, |total, count| total.checked_add(count))
        .ok_or(LinkError::IndexOverflow("aggregate property count"))?;
    check_limit("property", property_count, MAX_OBJECT_PROPERTIES)?;
    metadata.properties = Vec::with_capacity(property_count);
    for &index in order {
        metadata
            .properties
            .extend(objects[index].metadata.properties.iter().cloned());
    }

    metadata.source_files.clear();
    for &object_index in order {
        for source in &objects[object_index].metadata.source_files {
            let linked_id = maps[object_index]
                .sources_base
                .checked_add(source.id as usize)
                .ok_or(LinkError::IndexOverflow("source identity"))?;
            metadata.source_files.push(ObjectSourceFile {
                id: u32::try_from(linked_id)
                    .map_err(|_| LinkError::IndexOverflow("source identity"))?,
                name: source.name.clone(),
                byte_len: source.byte_len,
                content_digest: source.content_digest,
            });
        }
    }
    check_limit(
        "source file",
        metadata.source_files.len(),
        MAX_OBJECT_SOURCE_FILES,
    )?;
    metadata.verification = None;
    Ok(metadata)
}

fn allocate_maps(objects: &[ObjectModule], order: &[usize]) -> Result<Vec<ModuleMaps>, LinkError> {
    let procedure_count = order
        .iter()
        .map(|index| objects[*index].code.procedures.len())
        .sum::<usize>();
    let quote_count = order
        .iter()
        .map(|index| objects[*index].code.quotations.len())
        .sum::<usize>();
    let foreign_count = order
        .iter()
        .map(|index| objects[*index].code.foreigns.len())
        .sum::<usize>();
    check_limit("procedure", procedure_count, MAX_PROCEDURES)?;
    check_limit("quotation", quote_count, MAX_QUOTES)?;
    check_limit("foreign", foreign_count, MAX_FOREIGNS)?;

    let mut maps = vec![
        ModuleMaps {
            procedures: Vec::new(),
            quotes: Vec::new(),
            foreigns: Vec::new(),
            sources_base: 0,
        };
        objects.len()
    ];
    let mut next_procedure = 0usize;
    let mut next_quote = 0usize;
    let mut next_foreign = 0usize;
    let mut next_source = 0usize;
    for &object_index in order {
        let object = &objects[object_index];
        let map = &mut maps[object_index];
        map.sources_base = next_source;
        map.procedures = (0..object.code.procedures.len())
            .map(|_| {
                let id = ProcedureId::try_from_index(next_procedure)
                    .ok_or(LinkError::IndexOverflow("procedure"))?;
                next_procedure += 1;
                Ok(id)
            })
            .collect::<Result<_, LinkError>>()?;
        map.quotes = (0..object.code.quotations.len())
            .map(|_| {
                let raw =
                    u64::try_from(next_quote).map_err(|_| LinkError::IndexOverflow("quotation"))?;
                next_quote += 1;
                Ok(QuoteId::new(raw))
            })
            .collect::<Result<_, LinkError>>()?;
        map.foreigns = (0..object.code.foreigns.len())
            .map(|_| {
                let id = ForeignId::try_from_index(next_foreign)
                    .ok_or(LinkError::IndexOverflow("foreign"))?;
                next_foreign += 1;
                Ok(id)
            })
            .collect::<Result<_, LinkError>>()?;

        next_source = next_source
            .checked_add(object.metadata.source_files.len())
            .ok_or(LinkError::IndexOverflow("source identity"))?;
    }
    Ok(maps)
}

fn linked_foreigns(
    objects: &[ObjectModule],
    order: &[usize],
    maps: &[ModuleMaps],
) -> Result<Vec<ForeignEntry>, LinkError> {
    let count = order
        .iter()
        .map(|index| objects[*index].code.foreigns.len())
        .sum();
    let mut foreigns = Vec::with_capacity(count);
    for &object_index in order {
        for foreign in &objects[object_index].code.foreigns {
            let mut linked = foreign.clone();
            linked.id = maps[object_index]
                .foreigns
                .get(foreign.id.index())
                .copied()
                .ok_or_else(|| {
                    malformed(&objects[object_index], "foreign entry is out of range")
                })?;
            foreigns.push(linked);
        }
    }
    Ok(foreigns)
}

fn collect_exports(
    objects: &[ObjectModule],
    order: &[usize],
    maps: &[ModuleMaps],
) -> Result<BTreeMap<String, ResolvedExport>, LinkError> {
    let mut exports = BTreeMap::new();
    for &object_index in order {
        for export in &objects[object_index].exports {
            let target =
                relocate_symbol(export.target, &maps[object_index], &objects[object_index])?;
            if exports
                .insert(
                    export.name.clone(),
                    ResolvedExport {
                        target,
                        signature: export.signature,
                    },
                )
                .is_some()
            {
                return Err(LinkError::DuplicateSymbol(export.name.clone()));
            }
        }
    }
    Ok(exports)
}

fn resolve_imports(
    objects: &[ObjectModule],
    order: &[usize],
    exports: &BTreeMap<String, ResolvedExport>,
) -> Result<Vec<Vec<SymbolTarget>>, LinkError> {
    let mut resolutions = vec![Vec::new(); objects.len()];
    for &object_index in order {
        for import in &objects[object_index].imports {
            let Some(export) = exports.get(&import.name) else {
                return Err(LinkError::UnresolvedSymbol {
                    module: objects[object_index].name.clone(),
                    symbol: import.name.clone(),
                });
            };
            if export.target.kind() != import.kind {
                return Err(LinkError::SymbolKindMismatch {
                    symbol: import.name.clone(),
                    expected: import.kind,
                    actual: export.target.kind(),
                });
            }
            if export.signature != import.signature {
                return Err(LinkError::SignatureMismatch {
                    symbol: import.name.clone(),
                    expected: import.signature,
                    actual: export.signature,
                });
            }
            resolutions[object_index].push(export.target);
        }
    }
    Ok(resolutions)
}

fn relocate_symbol(
    target: SymbolTarget,
    map: &ModuleMaps,
    object: &ObjectModule,
) -> Result<SymbolTarget, LinkError> {
    match target {
        SymbolTarget::Constant(value) => Some(SymbolTarget::Constant(value)),
        SymbolTarget::Procedure(id) => map
            .procedures
            .get(id.index())
            .copied()
            .map(SymbolTarget::Procedure),
        SymbolTarget::Quote(id) => usize::try_from(id.as_u64())
            .ok()
            .and_then(|index| map.quotes.get(index).copied())
            .map(SymbolTarget::Quote),
        SymbolTarget::Foreign(id) => map
            .foreigns
            .get(id.index())
            .copied()
            .map(SymbolTarget::Foreign),
    }
    .ok_or_else(|| LinkError::MalformedObject {
        module: object.name.clone(),
        message: format!("export target {target:?} is outside its local table"),
    })
}

struct LinkBuilder<'a> {
    objects: &'a [ObjectModule],
    maps: &'a [ModuleMaps],
    resolutions: &'a [Vec<SymbolTarget>],
    relocations: Vec<HashMap<u32, ObjectRelocation>>,
    sources: Vec<HashMap<u32, SourceSpan>>,
    instructions: Vec<Instruction>,
    source_map: Vec<SourceMapEntry>,
}

impl<'a> LinkBuilder<'a> {
    fn new(
        objects: &'a [ObjectModule],
        maps: &'a [ModuleMaps],
        resolutions: &'a [Vec<SymbolTarget>],
        instruction_capacity: usize,
        source_map_capacity: usize,
    ) -> Self {
        let relocations = objects
            .iter()
            .map(|object| {
                object
                    .relocations
                    .iter()
                    .map(|relocation| (relocation.instruction, *relocation))
                    .collect()
            })
            .collect();
        let sources = objects
            .iter()
            .map(|object| {
                object
                    .code
                    .source_map
                    .iter()
                    .map(|entry| (entry.instruction, entry.span))
                    .collect()
            })
            .collect();
        Self {
            objects,
            maps,
            resolutions,
            relocations,
            sources,
            instructions: Vec::with_capacity(instruction_capacity),
            source_map: Vec::with_capacity(source_map_capacity),
        }
    }

    fn append_range(
        &mut self,
        object_index: usize,
        range: CodeRange,
        omit_return: bool,
    ) -> Result<CodeRange, LinkError> {
        let new_start = u32_len(self.instructions.len(), "instruction")?;
        let old_end = if omit_return {
            range.end - 1
        } else {
            range.end
        };
        for old_pc in range.start..old_end {
            if self.instructions.len() >= MAX_INSTRUCTIONS {
                return Err(LinkError::LimitExceeded {
                    what: "instruction",
                    count: self.instructions.len().saturating_add(1),
                    limit: MAX_INSTRUCTIONS,
                });
            }
            let instruction = self.relocate_instruction(object_index, range, new_start, old_pc)?;
            let new_pc = u32_len(self.instructions.len(), "instruction")?;
            self.instructions.push(instruction);
            if let Some(span) = self.sources[object_index].get(&old_pc).copied() {
                if self.source_map.len() >= MAX_SOURCE_MAP_ENTRIES {
                    return Err(LinkError::LimitExceeded {
                        what: "source map",
                        count: self.source_map.len().saturating_add(1),
                        limit: MAX_SOURCE_MAP_ENTRIES,
                    });
                }
                let source_index = self.maps[object_index]
                    .sources_base
                    .checked_add(span.source.index())
                    .ok_or(LinkError::IndexOverflow("source identity"))?;
                self.source_map.push(SourceMapEntry {
                    instruction: new_pc,
                    span: SourceSpan::new(SourceId::new(source_index), span.range),
                });
            }
        }
        Ok(CodeRange {
            start: new_start,
            end: u32_len(self.instructions.len(), "instruction")?,
        })
    }

    fn relocate_instruction(
        &self,
        object_index: usize,
        range: CodeRange,
        new_start: u32,
        old_pc: u32,
    ) -> Result<Instruction, LinkError> {
        let object = &self.objects[object_index];
        let instruction = object.code.instructions[old_pc as usize];
        if let Some(relocation) = self.relocations[object_index].get(&old_pc) {
            let target = self.resolutions[object_index]
                .get(relocation.import as usize)
                .copied()
                .ok_or_else(|| malformed(object, "relocation import is out of range"))?;
            return match (relocation.kind, target) {
                (RelocationKind::ConstantLiteral, SymbolTarget::Constant(value)) => {
                    Ok(Instruction::PushWord(value))
                }
                (RelocationKind::ProcedureCall, SymbolTarget::Procedure(id)) => {
                    Ok(Instruction::CallProcedure(id))
                }
                (RelocationKind::QuoteLiteral, SymbolTarget::Quote(id)) => {
                    Ok(Instruction::PushQuote(id))
                }
                (RelocationKind::ForeignCall, SymbolTarget::Foreign(id)) => {
                    Ok(Instruction::CallForeign(id))
                }
                _ => Err(malformed(
                    object,
                    "relocation resolved to the wrong symbol kind",
                )),
            };
        }

        let map = &self.maps[object_index];
        match instruction {
            Instruction::Primitive(_) | Instruction::PushWord(_) | Instruction::Return => {
                Ok(instruction)
            }
            Instruction::PushQuote(id) => map
                .quotes
                .get(usize::try_from(id.as_u64()).unwrap_or(usize::MAX))
                .copied()
                .map(Instruction::PushQuote)
                .ok_or_else(|| malformed(object, "quotation operand is out of range")),
            Instruction::CallProcedure(id) => map
                .procedures
                .get(id.index())
                .copied()
                .map(Instruction::CallProcedure)
                .ok_or_else(|| malformed(object, "procedure operand is out of range")),
            Instruction::CallForeign(id) => map
                .foreigns
                .get(id.index())
                .copied()
                .map(Instruction::CallForeign)
                .ok_or_else(|| malformed(object, "foreign operand is out of range")),
            Instruction::IfFalse {
                else_target,
                end_target,
                has_else,
            } => Ok(Instruction::IfFalse {
                else_target: translate_target(object, range, new_start, else_target)?,
                end_target: translate_target(object, range, new_start, end_target)?,
                has_else,
            }),
            Instruction::Jump { target } => Ok(Instruction::Jump {
                target: translate_target(object, range, new_start, target)?,
            }),
            Instruction::WhileFalse {
                loop_start,
                end_target,
            } => Ok(Instruction::WhileFalse {
                loop_start: translate_target(object, range, new_start, loop_start)?,
                end_target: translate_target(object, range, new_start, end_target)?,
            }),
            Instruction::LoopBack { target } => Ok(Instruction::LoopBack {
                target: translate_target(object, range, new_start, target)?,
            }),
            Instruction::TemporalEnter {
                base,
                size,
                cell_bits,
                exit_target,
            } => Ok(Instruction::TemporalEnter {
                base,
                size,
                cell_bits,
                exit_target: translate_target(object, range, new_start, exit_target)?,
            }),
            Instruction::TemporalExit { enter_target } => Ok(Instruction::TemporalExit {
                enter_target: translate_target(object, range, new_start, enter_target)?,
            }),
        }
    }
}

fn translate_target(
    object: &ObjectModule,
    range: CodeRange,
    new_start: u32,
    target: u32,
) -> Result<u32, LinkError> {
    if target < range.start || target >= range.end {
        return Err(malformed(
            object,
            "control target leaves its local code range",
        ));
    }
    new_start
        .checked_add(target - range.start)
        .ok_or(LinkError::IndexOverflow("control target"))
}

fn validate_object(object: &ObjectModule) -> Result<(), LinkError> {
    if object.name.trim().is_empty()
        || object.name.contains('\0')
        || object.name.len() > MAX_OBJECT_NAME_BYTES
    {
        return Err(LinkError::InvalidModuleName(object.name.clone()));
    }
    check_limit(
        "instruction",
        object.code.instructions.len(),
        MAX_INSTRUCTIONS,
    )?;
    check_limit("procedure", object.code.procedures.len(), MAX_PROCEDURES)?;
    check_limit("quotation", object.code.quotations.len(), MAX_QUOTES)?;
    check_limit("foreign", object.code.foreigns.len(), MAX_FOREIGNS)?;
    check_limit(
        "source map",
        object.code.source_map.len(),
        MAX_SOURCE_MAP_ENTRIES,
    )?;
    check_limit("export", object.exports.len(), MAX_OBJECT_EXPORTS)?;
    check_limit("import", object.imports.len(), MAX_OBJECT_IMPORTS)?;
    check_limit(
        "relocation",
        object.relocations.len(),
        MAX_OBJECT_RELOCATIONS,
    )?;
    validate_metadata(object)?;

    let instruction_end = u32_len(object.code.instructions.len(), "instruction")?;

    for export in &object.exports {
        validate_symbol_name(object, &export.name, "export")?;
        let local = match export.target {
            SymbolTarget::Constant(_) => true,
            SymbolTarget::Procedure(id) => id.index() < object.code.procedures.len(),
            SymbolTarget::Quote(id) => id.as_u64() < object.code.quotations.len() as u64,
            SymbolTarget::Foreign(id) => id.index() < object.code.foreigns.len(),
        };
        if !local {
            return Err(malformed(
                object,
                format!(
                    "export target {:?} is outside its local table",
                    export.target
                ),
            ));
        }
        if let SymbolTarget::Foreign(id) = export.target {
            let foreign = &object.code.foreigns[id.index()];
            let retained = SymbolSignature::foreign(foreign);
            if export.signature != retained {
                return Err(malformed(
                    object,
                    format!(
                        "foreign export {:?} signature {:?} differs from retained ABI {:?}",
                        export.name, export.signature, retained
                    ),
                ));
            }
        }
    }
    for import in &object.imports {
        validate_symbol_name(object, &import.name, "import")?;
    }

    let mut unresolved = HashSet::with_capacity(object.relocations.len());
    for relocation in &object.relocations {
        if relocation.instruction >= instruction_end {
            return Err(malformed(object, "relocation instruction is out of range"));
        }
        if relocation.import as usize >= object.imports.len() {
            return Err(malformed(object, "relocation import is out of range"));
        }
        if object.imports[relocation.import as usize].kind != relocation.kind.symbol_kind() {
            return Err(malformed(object, "relocation and import kinds differ"));
        }
        let instruction = object.code.instructions[relocation.instruction as usize];
        let shape_matches = matches!(
            (relocation.kind, instruction),
            (RelocationKind::ConstantLiteral, Instruction::PushWord(_))
                | (RelocationKind::ProcedureCall, Instruction::CallProcedure(_))
                | (RelocationKind::QuoteLiteral, Instruction::PushQuote(_))
                | (RelocationKind::ForeignCall, Instruction::CallForeign(_))
        );
        if !shape_matches {
            return Err(malformed(
                object,
                "relocation instruction has the wrong operand form",
            ));
        }
        if !unresolved.insert(relocation.instruction) {
            return Err(malformed(object, "instruction has multiple relocations"));
        }
    }

    object
        .code
        .validate_relocatable(&unresolved)
        .map(|_| ())
        .map_err(|error| malformed(object, format!("local bytecode: {error}")))
}

fn validate_metadata(object: &ObjectModule) -> Result<(), LinkError> {
    let metadata = &object.metadata;
    if metadata.runtime_abi == 0 {
        return Err(malformed(object, "runtime ABI must be nonzero"));
    }
    check_limit("constant", metadata.constants.len(), MAX_OBJECT_CONSTANTS)?;
    if metadata.constants.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(malformed(
            object,
            "constant pool is not strictly increasing",
        ));
    }
    if metadata.constants != bytecode_constants(&object.code) {
        return Err(malformed(
            object,
            "constant pool does not exactly match bytecode literals",
        ));
    }

    check_limit(
        "temporal region",
        metadata.temporal_regions.len(),
        MAX_OBJECT_TEMPORAL_REGIONS,
    )?;
    if metadata.temporal_regions != bytecode_temporal_regions(&object.code) {
        return Err(malformed(
            object,
            "temporal-region schema does not exactly match bytecode",
        ));
    }

    check_limit(
        "named temporal",
        metadata.named_temporals.len(),
        MAX_OBJECT_NAMED_TEMPORALS,
    )?;
    let mut temporal_names = HashSet::with_capacity(metadata.named_temporals.len());
    let mut temporal_addresses = HashSet::with_capacity(metadata.named_temporals.len());
    for temporal in &metadata.named_temporals {
        validate_symbol_name(object, &temporal.name, "named temporal")?;
        if !temporal_names.insert(temporal.name.to_lowercase()) {
            return Err(malformed(object, "duplicate named temporal name"));
        }
        if !temporal_addresses.insert(temporal.address) {
            return Err(malformed(object, "duplicate named temporal address"));
        }
    }

    check_limit("property", metadata.properties.len(), MAX_OBJECT_PROPERTIES)?;
    let mut property_names = HashSet::with_capacity(metadata.properties.len());
    for property in &metadata.properties {
        validate_symbol_name(object, &property.name, "property")?;
        if !property_names.insert(property.name.as_str()) {
            return Err(malformed(
                object,
                format!("duplicate property name {:?}", property.name),
            ));
        }
        encode_property_predicate(property.predicate.as_ref())
            .map_err(|error| malformed(object, error.to_string()))?;
        let expected = property
            .predicate
            .as_ref()
            .map(PropertyPredicate::touched_addresses)
            .unwrap_or_else(|| vec![property.address]);
        if property.touched_addresses != expected {
            return Err(malformed(
                object,
                format!(
                    "property {:?} touched-address slice is not canonical",
                    property.name
                ),
            ));
        }
    }

    if ObjectEffects::from_bits(metadata.effects.declared.bits()).is_none()
        || ObjectEffects::from_bits(metadata.effects.observed.bits()).is_none()
    {
        return Err(malformed(object, "effect summary contains unknown flags"));
    }
    let observed = observed_effects(&object.code);
    if metadata.effects.observed != observed {
        return Err(malformed(
            object,
            "observed effect summary does not exactly match bytecode",
        ));
    }

    check_limit(
        "source file",
        metadata.source_files.len(),
        MAX_OBJECT_SOURCE_FILES,
    )?;
    for (index, source) in metadata.source_files.iter().enumerate() {
        if source.id as usize != index {
            return Err(malformed(
                object,
                "source manifest identities are not canonical and contiguous",
            ));
        }
        if source.name.trim().is_empty()
            || source.name.contains('\0')
            || source.name.len() > MAX_OBJECT_SOURCE_NAME_BYTES
        {
            return Err(malformed(
                object,
                format!("source display name {:?} is invalid", source.name),
            ));
        }
    }
    for entry in &object.code.source_map {
        let Some(source) = metadata.source_files.get(entry.span.source.index()) else {
            return Err(malformed(
                object,
                "source-map entry references an absent source manifest identity",
            ));
        };
        let start = u64::try_from(entry.span.range.start)
            .map_err(|_| malformed(object, "source-map start does not fit the object format"))?;
        let end = u64::try_from(entry.span.range.end)
            .map_err(|_| malformed(object, "source-map end does not fit the object format"))?;
        if start > end || end > source.byte_len {
            return Err(malformed(
                object,
                "source-map range exceeds its source manifest byte length",
            ));
        }
    }

    if let Some(verification) = &metadata.verification {
        if verification.format_version == 0 {
            return Err(malformed(
                object,
                "verification artifact version must be nonzero",
            ));
        }
        check_limit(
            "verification artifact byte",
            verification.payload.len(),
            MAX_OBJECT_VERIFICATION_BYTES,
        )?;
    }
    Ok(())
}

fn validate_symbol_name(object: &ObjectModule, name: &str, table: &str) -> Result<(), LinkError> {
    if name.trim().is_empty() || name.contains('\0') || name.len() > MAX_OBJECT_NAME_BYTES {
        Err(malformed(
            object,
            format!("{table} symbol name {name:?} is invalid"),
        ))
    } else {
        Ok(())
    }
}

fn malformed(object: &ObjectModule, message: impl Into<String>) -> LinkError {
    LinkError::MalformedObject {
        module: object.name.clone(),
        message: message.into(),
    }
}

fn check_limit(what: &'static str, count: usize, limit: usize) -> Result<(), LinkError> {
    if count > limit {
        Err(LinkError::LimitExceeded { what, count, limit })
    } else {
        Ok(())
    }
}

fn u32_len(value: usize, what: &'static str) -> Result<u32, LinkError> {
    u32::try_from(value).map_err(|_| LinkError::IndexOverflow(what))
}

const SYMBOL_PROCEDURE: u8 = 0;
const SYMBOL_QUOTE: u8 = 1;
const SYMBOL_FOREIGN: u8 = 2;
const SYMBOL_CONSTANT: u8 = 3;
const RELOCATION_PROCEDURE_CALL: u8 = 0;
const RELOCATION_QUOTE_LITERAL: u8 = 1;
const RELOCATION_FOREIGN_CALL: u8 = 2;
const RELOCATION_CONSTANT_LITERAL: u8 = 3;
const TARGET_PORTABLE_VM_64_LE: u16 = 1;
const PROPERTY_EQ: u8 = 0;
const PROPERTY_NE: u8 = 1;
const PROPERTY_ULT: u8 = 2;
const PROPERTY_ULE: u8 = 3;
const PROPERTY_UGT: u8 = 4;
const PROPERTY_UGE: u8 = 5;
const VERIFICATION_NONE: u8 = 0;
const VERIFICATION_BYTECODE_REPORT: u8 = 1;
const VERIFICATION_SOLVER_CERTIFICATE: u8 = 2;

fn object_limit(what: &'static str, count: usize, limit: usize) -> Result<(), ObjectError> {
    if count > limit {
        Err(ObjectError::LimitExceeded { what, count, limit })
    } else {
        Ok(())
    }
}

fn object_count(count: usize, what: &'static str) -> Result<u32, ObjectError> {
    u32::try_from(count).map_err(|_| ObjectError::IndexOverflow(what))
}

fn encode_object_string(
    bytes: &mut Vec<u8>,
    value: &str,
    what: &'static str,
) -> Result<(), ObjectError> {
    object_limit(what, value.len(), MAX_OBJECT_NAME_BYTES)?;
    object_put_u32(bytes, object_count(value.len(), what)?);
    bytes.extend_from_slice(value.as_bytes());
    Ok(())
}

fn encode_bounded_string(
    bytes: &mut Vec<u8>,
    value: &str,
    what: &'static str,
    limit: usize,
) -> Result<(), ObjectError> {
    object_limit(what, value.len(), limit)?;
    object_put_u32(bytes, object_count(value.len(), what)?);
    bytes.extend_from_slice(value.as_bytes());
    Ok(())
}

fn encode_object_target(target: ObjectTarget) -> u16 {
    match target {
        ObjectTarget::PortableVm64Le => TARGET_PORTABLE_VM_64_LE,
    }
}

fn decode_object_target(tag: u16) -> Result<ObjectTarget, ObjectError> {
    match tag {
        TARGET_PORTABLE_VM_64_LE => Ok(ObjectTarget::PortableVm64Le),
        _ => Err(ObjectError::UnknownTarget(tag)),
    }
}

fn decode_object_effects(bits: u64) -> Result<ObjectEffects, ObjectError> {
    ObjectEffects::from_bits(bits).ok_or(ObjectError::UnknownEffectFlags(bits))
}

fn encode_property_comparison(comparison: PropertyComparison) -> u8 {
    match comparison {
        PropertyComparison::Eq => PROPERTY_EQ,
        PropertyComparison::Ne => PROPERTY_NE,
        PropertyComparison::Ult => PROPERTY_ULT,
        PropertyComparison::Ule => PROPERTY_ULE,
        PropertyComparison::Ugt => PROPERTY_UGT,
        PropertyComparison::Uge => PROPERTY_UGE,
    }
}

fn decode_property_comparison(tag: u8) -> Result<PropertyComparison, ObjectError> {
    match tag {
        PROPERTY_EQ => Ok(PropertyComparison::Eq),
        PROPERTY_NE => Ok(PropertyComparison::Ne),
        PROPERTY_ULT => Ok(PropertyComparison::Ult),
        PROPERTY_ULE => Ok(PropertyComparison::Ule),
        PROPERTY_UGT => Ok(PropertyComparison::Ugt),
        PROPERTY_UGE => Ok(PropertyComparison::Uge),
        _ => Err(ObjectError::UnknownPropertyComparison(tag)),
    }
}

const PROPERTY_PREDICATE_COMPARE: u8 = 0;
const PROPERTY_PREDICATE_NOT: u8 = 1;
const PROPERTY_PREDICATE_AND: u8 = 2;
const PROPERTY_PREDICATE_OR: u8 = 3;
const MAX_PROPERTY_PREDICATE_DEPTH: usize = 32;
const MAX_PROPERTY_PREDICATE_NODES: usize = 256;

fn encode_property_predicate(
    predicate: Option<&PropertyPredicate>,
) -> Result<Vec<u8>, ObjectError> {
    let Some(predicate) = predicate else {
        return Ok(Vec::new());
    };
    let mut bytes = Vec::new();
    let mut nodes = 0usize;
    encode_property_predicate_node(predicate, 0, &mut nodes, &mut bytes)?;
    Ok(bytes)
}

fn encode_property_predicate_node(
    predicate: &PropertyPredicate,
    depth: usize,
    nodes: &mut usize,
    bytes: &mut Vec<u8>,
) -> Result<(), ObjectError> {
    if depth >= MAX_PROPERTY_PREDICATE_DEPTH {
        return Err(ObjectError::InvalidMetadata(
            "property predicate nesting exceeds 32".into(),
        ));
    }
    *nodes += 1;
    if *nodes > MAX_PROPERTY_PREDICATE_NODES {
        return Err(ObjectError::InvalidMetadata(
            "property predicate exceeds 256 nodes".into(),
        ));
    }
    match predicate {
        PropertyPredicate::Compare {
            cell,
            comparison,
            value,
        } => {
            bytes.push(PROPERTY_PREDICATE_COMPARE);
            object_put_u64(bytes, cell.address);
            if let Some(name) = &cell.name {
                encode_object_string(bytes, name, "property cell name")?;
            } else {
                object_put_u32(bytes, 0);
            }
            bytes.push(encode_property_comparison(*comparison));
            object_put_u64(bytes, *value);
        }
        PropertyPredicate::Not(inner) => {
            bytes.push(PROPERTY_PREDICATE_NOT);
            encode_property_predicate_node(inner, depth + 1, nodes, bytes)?;
        }
        PropertyPredicate::And(left, right) | PropertyPredicate::Or(left, right) => {
            bytes.push(if matches!(predicate, PropertyPredicate::And(_, _)) {
                PROPERTY_PREDICATE_AND
            } else {
                PROPERTY_PREDICATE_OR
            });
            encode_property_predicate_node(left, depth + 1, nodes, bytes)?;
            encode_property_predicate_node(right, depth + 1, nodes, bytes)?;
        }
    }
    Ok(())
}

fn decode_property_predicate(bytes: &[u8]) -> Result<Option<PropertyPredicate>, ObjectError> {
    if bytes.is_empty() {
        return Ok(None);
    }
    let mut reader = ObjectReader::new(bytes);
    let mut nodes = 0usize;
    let predicate = decode_property_predicate_node(&mut reader, 0, &mut nodes)?;
    if reader.remaining() != 0 {
        return Err(ObjectError::InvalidMetadata(
            "property predicate has trailing bytes".into(),
        ));
    }
    Ok(Some(predicate))
}

fn decode_property_predicate_node(
    reader: &mut ObjectReader<'_>,
    depth: usize,
    nodes: &mut usize,
) -> Result<PropertyPredicate, ObjectError> {
    if depth >= MAX_PROPERTY_PREDICATE_DEPTH {
        return Err(ObjectError::InvalidMetadata(
            "property predicate nesting exceeds 32".into(),
        ));
    }
    *nodes += 1;
    if *nodes > MAX_PROPERTY_PREDICATE_NODES {
        return Err(ObjectError::InvalidMetadata(
            "property predicate exceeds 256 nodes".into(),
        ));
    }
    match reader.u8()? {
        PROPERTY_PREDICATE_COMPARE => {
            let address = reader.u64()?;
            let name_len = reader.bounded_count("property cell name", MAX_OBJECT_NAME_BYTES)?;
            let name = if name_len == 0 {
                None
            } else {
                Some(reader.string_with_len(name_len, "property cell name")?)
            };
            let comparison = decode_property_comparison(reader.u8()?)?;
            let value = reader.u64()?;
            Ok(PropertyPredicate::Compare {
                cell: PropertyCell { address, name },
                comparison,
                value,
            })
        }
        PROPERTY_PREDICATE_NOT => Ok(PropertyPredicate::Not(Box::new(
            decode_property_predicate_node(reader, depth + 1, nodes)?,
        ))),
        PROPERTY_PREDICATE_AND => Ok(PropertyPredicate::And(
            Box::new(decode_property_predicate_node(reader, depth + 1, nodes)?),
            Box::new(decode_property_predicate_node(reader, depth + 1, nodes)?),
        )),
        PROPERTY_PREDICATE_OR => Ok(PropertyPredicate::Or(
            Box::new(decode_property_predicate_node(reader, depth + 1, nodes)?),
            Box::new(decode_property_predicate_node(reader, depth + 1, nodes)?),
        )),
        tag => Err(ObjectError::InvalidMetadata(format!(
            "unknown property predicate tag {tag}"
        ))),
    }
}

fn encode_verification_kind(kind: VerificationArtifactKind) -> u8 {
    match kind {
        VerificationArtifactKind::BytecodeReport => VERIFICATION_BYTECODE_REPORT,
        VerificationArtifactKind::SolverCertificate => VERIFICATION_SOLVER_CERTIFICATE,
    }
}

fn decode_verification_artifact(
    tag: u8,
    format_version: u16,
    payload: Vec<u8>,
) -> Result<Option<ObjectVerificationArtifact>, ObjectError> {
    let kind = match tag {
        VERIFICATION_NONE if format_version == 0 && payload.is_empty() => return Ok(None),
        VERIFICATION_NONE => {
            return Err(ObjectError::InvalidMetadata(
                "absent verification artifact has nonzero version or payload".into(),
            ))
        }
        VERIFICATION_BYTECODE_REPORT => VerificationArtifactKind::BytecodeReport,
        VERIFICATION_SOLVER_CERTIFICATE => VerificationArtifactKind::SolverCertificate,
        _ => return Err(ObjectError::UnknownVerificationKind(tag)),
    };
    if format_version == 0 {
        return Err(ObjectError::InvalidMetadata(
            "verification artifact version must be nonzero".into(),
        ));
    }
    Ok(Some(ObjectVerificationArtifact {
        kind,
        format_version,
        payload,
    }))
}

fn encode_symbol_kind(kind: SymbolKind) -> u8 {
    match kind {
        SymbolKind::Constant => SYMBOL_CONSTANT,
        SymbolKind::Procedure => SYMBOL_PROCEDURE,
        SymbolKind::Quote => SYMBOL_QUOTE,
        SymbolKind::Foreign => SYMBOL_FOREIGN,
    }
}

fn decode_symbol_kind(tag: u8) -> Result<SymbolKind, ObjectError> {
    match tag {
        SYMBOL_CONSTANT => Ok(SymbolKind::Constant),
        SYMBOL_PROCEDURE => Ok(SymbolKind::Procedure),
        SYMBOL_QUOTE => Ok(SymbolKind::Quote),
        SYMBOL_FOREIGN => Ok(SymbolKind::Foreign),
        _ => Err(ObjectError::UnknownSymbolKind(tag)),
    }
}

fn encode_symbol_target(bytes: &mut Vec<u8>, target: SymbolTarget) {
    match target {
        SymbolTarget::Constant(value) => {
            bytes.push(SYMBOL_CONSTANT);
            object_put_u64(bytes, value);
        }
        SymbolTarget::Procedure(id) => {
            bytes.push(SYMBOL_PROCEDURE);
            object_put_u32(bytes, id.as_u32());
        }
        SymbolTarget::Quote(id) => {
            bytes.push(SYMBOL_QUOTE);
            object_put_u64(bytes, id.as_u64());
        }
        SymbolTarget::Foreign(id) => {
            bytes.push(SYMBOL_FOREIGN);
            object_put_u32(bytes, id.as_u32());
        }
    }
}

fn decode_symbol_target(reader: &mut ObjectReader<'_>) -> Result<SymbolTarget, ObjectError> {
    match reader.u8()? {
        SYMBOL_CONSTANT => Ok(SymbolTarget::Constant(reader.u64()?)),
        SYMBOL_PROCEDURE => {
            let raw = reader.u32()?;
            let id = ProcedureId::try_from_index(raw as usize)
                .ok_or(ObjectError::IndexOverflow("procedure target"))?;
            Ok(SymbolTarget::Procedure(id))
        }
        SYMBOL_QUOTE => Ok(SymbolTarget::Quote(QuoteId::new(reader.u64()?))),
        SYMBOL_FOREIGN => {
            let raw = reader.u32()?;
            let id = ForeignId::try_from_index(raw as usize)
                .ok_or(ObjectError::IndexOverflow("foreign target"))?;
            Ok(SymbolTarget::Foreign(id))
        }
        tag => Err(ObjectError::UnknownSymbolTarget(tag)),
    }
}

fn encode_signature(bytes: &mut Vec<u8>, signature: SymbolSignature) {
    object_put_u32(bytes, signature.inputs);
    object_put_u32(bytes, signature.outputs);
    object_put_u64(bytes, signature.effects);
    object_put_u64(bytes, signature.abi);
}

fn decode_signature(reader: &mut ObjectReader<'_>) -> Result<SymbolSignature, ObjectError> {
    Ok(SymbolSignature {
        inputs: reader.u32()?,
        outputs: reader.u32()?,
        effects: reader.u64()?,
        abi: reader.u64()?,
    })
}

fn encode_relocation_kind(kind: RelocationKind) -> u8 {
    match kind {
        RelocationKind::ConstantLiteral => RELOCATION_CONSTANT_LITERAL,
        RelocationKind::ProcedureCall => RELOCATION_PROCEDURE_CALL,
        RelocationKind::QuoteLiteral => RELOCATION_QUOTE_LITERAL,
        RelocationKind::ForeignCall => RELOCATION_FOREIGN_CALL,
    }
}

fn decode_relocation_kind(tag: u8) -> Result<RelocationKind, ObjectError> {
    match tag {
        RELOCATION_CONSTANT_LITERAL => Ok(RelocationKind::ConstantLiteral),
        RELOCATION_PROCEDURE_CALL => Ok(RelocationKind::ProcedureCall),
        RELOCATION_QUOTE_LITERAL => Ok(RelocationKind::QuoteLiteral),
        RELOCATION_FOREIGN_CALL => Ok(RelocationKind::ForeignCall),
        _ => Err(ObjectError::UnknownRelocationKind(tag)),
    }
}

fn object_put_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn object_put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn object_put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

struct ObjectReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ObjectReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], ObjectError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or(ObjectError::Truncated {
                offset: self.offset,
                needed: count,
            })?;
        let result = self
            .bytes
            .get(self.offset..end)
            .ok_or(ObjectError::Truncated {
                offset: self.offset,
                needed: count,
            })?;
        self.offset = end;
        Ok(result)
    }

    fn u8(&mut self) -> Result<u8, ObjectError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, ObjectError> {
        let bytes: [u8; 2] = self.take(2)?.try_into().expect("exact object field");
        Ok(u16::from_le_bytes(bytes))
    }

    fn u32(&mut self) -> Result<u32, ObjectError> {
        let bytes: [u8; 4] = self.take(4)?.try_into().expect("exact object field");
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, ObjectError> {
        let bytes: [u8; 8] = self.take(8)?.try_into().expect("exact object field");
        Ok(u64::from_le_bytes(bytes))
    }

    fn bounded_count(&mut self, what: &'static str, limit: usize) -> Result<usize, ObjectError> {
        let count = self.u32()? as usize;
        object_limit(what, count, limit)?;
        Ok(count)
    }

    fn string(&mut self, field: &'static str) -> Result<String, ObjectError> {
        let length = self.bounded_count(field, MAX_OBJECT_NAME_BYTES)?;
        self.string_with_len(length, field)
    }

    fn bounded_string(&mut self, field: &'static str, limit: usize) -> Result<String, ObjectError> {
        let length = self.bounded_count(field, limit)?;
        self.string_with_len(length, field)
    }

    fn string_with_len(
        &mut self,
        length: usize,
        field: &'static str,
    ) -> Result<String, ObjectError> {
        std::str::from_utf8(self.take(length)?)
            .map(str::to_owned)
            .map_err(|_| ObjectError::InvalidUtf8 { field })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{OpCode, Procedure, Program, Stmt};
    use crate::core::Value;
    use crate::hir::HirProgram;

    fn procedure(name: &str, body: Vec<Stmt>) -> Procedure {
        Procedure {
            name: name.to_string(),
            params: Vec::new(),
            returns: 1,
            effects: Vec::new(),
            body,
        }
    }

    fn object(name: &str, program: Program) -> ObjectModule {
        let hir = HirProgram::resolve(&program).expect("resolve object fixture");
        ObjectModule::new(
            name,
            BytecodeProgram::compile(&hir).expect("compile object fixture"),
        )
    }

    fn relocatable_fixture() -> ObjectModule {
        let mut program = Program::new();
        program.quotes.push(vec![Stmt::Op(OpCode::Nop)]);
        program
            .procedures
            .push(procedure("local_proc", vec![Stmt::Push(Value::new(5))]));
        let mut object = object("fixture.α", program);
        object.code.foreigns.push(ForeignEntry {
            id: ForeignId::try_from_index(0).unwrap(),
            library: "host".into(),
            symbol: "foreign".into(),
            parameters: vec![crate::bytecode::ForeignScalarType::U64],
            result: Some(crate::bytecode::ForeignScalarType::U64),
            effects: crate::bytecode::ForeignEffects::from_bits(
                crate::bytecode::ForeignEffects::PURE,
            )
            .unwrap(),
        });

        object.code.instructions.splice(
            0..0,
            [
                Instruction::CallProcedure(ProcedureId::try_from_index(99).unwrap()),
                Instruction::PushQuote(QuoteId::new(98)),
                Instruction::CallForeign(ForeignId::try_from_index(97).unwrap()),
            ],
        );
        object.code.main.end += 3;
        for entry in &mut object.code.quotations {
            entry.range.start += 3;
            entry.range.end += 3;
        }
        for entry in &mut object.code.procedures {
            entry.range.start += 3;
            entry.range.end += 3;
        }

        let signature = SymbolSignature {
            inputs: 2,
            outputs: 1,
            effects: 0x0102_0304_0506_0708,
            abi: 0,
        };
        let foreign_signature = SymbolSignature::foreign(&object.code.foreigns[0]);
        object.exports = vec![
            ObjectExport {
                name: "local.proc".into(),
                target: SymbolTarget::Procedure(ProcedureId::try_from_index(0).unwrap()),
                signature,
            },
            ObjectExport {
                name: "local.quote".into(),
                target: SymbolTarget::Quote(QuoteId::new(0)),
                signature: SymbolSignature::pure(0, 0),
            },
            ObjectExport {
                name: "local.foreign".into(),
                target: SymbolTarget::Foreign(ForeignId::try_from_index(0).unwrap()),
                signature: foreign_signature,
            },
        ];
        object.imports = vec![
            ObjectImport {
                name: "external.proc".into(),
                kind: SymbolKind::Procedure,
                signature,
            },
            ObjectImport {
                name: "external.quote".into(),
                kind: SymbolKind::Quote,
                signature: SymbolSignature::pure(0, 0),
            },
            ObjectImport {
                name: "external.foreign".into(),
                kind: SymbolKind::Foreign,
                signature: foreign_signature,
            },
        ];
        object.relocations = vec![
            ObjectRelocation {
                instruction: 0,
                import: 0,
                kind: RelocationKind::ProcedureCall,
            },
            ObjectRelocation {
                instruction: 1,
                import: 1,
                kind: RelocationKind::QuoteLiteral,
            },
            ObjectRelocation {
                instruction: 2,
                import: 2,
                kind: RelocationKind::ForeignCall,
            },
        ];
        object.metadata = ObjectMetadata::for_bytecode(&object.code);
        object
    }

    #[test]
    fn object_round_trip_is_deterministic_and_byte_stable() {
        let object = relocatable_fixture();
        object.validate().expect("relocatable fixture validates");
        assert!(
            object.code.validate().is_err(),
            "raw object is not executable"
        );

        let first = object.to_bytes().expect("encode object");
        let second = object.to_bytes().expect("repeat deterministic encoding");
        assert_eq!(first, second);
        let decoded = ObjectModule::from_bytes(&first).expect("decode object");
        assert_eq!(decoded, object);
        assert_eq!(decoded.to_bytes().expect("re-encode object"), first);
    }

    #[test]
    fn only_declared_typed_relocation_sites_may_be_unresolved() {
        let valid = relocatable_fixture();
        valid.validate().expect("all unresolved sites declared");

        let mut missing = valid.clone();
        missing.relocations.remove(1);
        assert!(matches!(
            missing.validate(),
            Err(LinkError::MalformedObject { .. })
        ));

        let mut wrong_shape = valid.clone();
        wrong_shape.relocations[0].kind = RelocationKind::QuoteLiteral;
        assert!(matches!(
            wrong_shape.validate(),
            Err(LinkError::MalformedObject { .. })
        ));

        let mut duplicate = valid;
        duplicate.relocations.push(duplicate.relocations[0]);
        assert!(matches!(
            duplicate.validate(),
            Err(LinkError::MalformedObject { .. })
        ));
    }

    #[test]
    fn malformed_object_byte_corpus_is_rejected() {
        let fixture = relocatable_fixture();
        let good = fixture.to_bytes().unwrap();

        for length in 0..good.len() {
            assert!(
                ObjectModule::from_bytes(&good[..length]).is_err(),
                "accepted object truncation at {length}"
            );
        }

        let mut trailing = good.clone();
        trailing.push(0);
        assert!(matches!(
            ObjectModule::from_bytes(&trailing),
            Err(ObjectError::TrailingBytes(1))
        ));

        let mut bad_magic = good.clone();
        bad_magic[0] ^= 0xff;
        assert_eq!(
            ObjectModule::from_bytes(&bad_magic).unwrap_err(),
            ObjectError::BadMagic
        );

        let mut bad_version = good.clone();
        let unsupported_version = OBJECT_VERSION + 1;
        bad_version[8..10].copy_from_slice(&unsupported_version.to_le_bytes());
        assert_eq!(
            ObjectModule::from_bytes(&bad_version).unwrap_err(),
            ObjectError::UnsupportedVersion(unsupported_version)
        );

        let mut bad_flags = good.clone();
        bad_flags[10..12].copy_from_slice(&1u16.to_le_bytes());
        assert_eq!(
            ObjectModule::from_bytes(&bad_flags).unwrap_err(),
            ObjectError::UnsupportedFlags(1)
        );

        let mut excessive_imports = good.clone();
        excessive_imports[24..28].copy_from_slice(&(MAX_OBJECT_IMPORTS as u32 + 1).to_le_bytes());
        assert!(matches!(
            ObjectModule::from_bytes(&excessive_imports),
            Err(ObjectError::LimitExceeded { what: "import", .. })
        ));

        let mut unknown_object_target = good.clone();
        unknown_object_target[32..34].copy_from_slice(&0xffffu16.to_le_bytes());
        assert_eq!(
            ObjectModule::from_bytes(&unknown_object_target).unwrap_err(),
            ObjectError::UnknownTarget(0xffff)
        );

        let mut excessive_constants = good.clone();
        excessive_constants[36..40]
            .copy_from_slice(&(MAX_OBJECT_CONSTANTS as u32 + 1).to_le_bytes());
        assert!(matches!(
            ObjectModule::from_bytes(&excessive_constants),
            Err(ObjectError::LimitExceeded {
                what: "constant",
                ..
            })
        ));

        let mut unknown_effects = good.clone();
        unknown_effects[52..60].copy_from_slice(&(1u64 << 63).to_le_bytes());
        assert_eq!(
            ObjectModule::from_bytes(&unknown_effects).unwrap_err(),
            ObjectError::UnknownEffectFlags(1u64 << 63)
        );

        let mut unknown_verification = good.clone();
        unknown_verification[68] = 0xff;
        assert_eq!(
            ObjectModule::from_bytes(&unknown_verification).unwrap_err(),
            ObjectError::UnknownVerificationKind(0xff)
        );

        let mut excessive_verification = good.clone();
        excessive_verification[71..75]
            .copy_from_slice(&(MAX_OBJECT_VERIFICATION_BYTES as u32 + 1).to_le_bytes());
        assert!(matches!(
            ObjectModule::from_bytes(&excessive_verification),
            Err(ObjectError::LimitExceeded {
                what: "verification artifact byte",
                ..
            })
        ));

        let name_len = u32::from_le_bytes(good[12..16].try_into().unwrap()) as usize;
        let code_len = u32::from_le_bytes(good[16..20].try_into().unwrap()) as usize;
        let code_start = 75 + name_len;
        let mut bad_embedded_magic = good.clone();
        bad_embedded_magic[code_start] ^= 0xff;
        assert!(matches!(
            ObjectModule::from_bytes(&bad_embedded_magic),
            Err(ObjectError::Bytecode(BytecodeError::BadMagic))
        ));

        let first_export = code_start + code_len;
        let export_name_len =
            u32::from_le_bytes(good[first_export..first_export + 4].try_into().unwrap()) as usize;
        let target_tag = first_export + 4 + export_name_len;
        let mut unknown_target = good.clone();
        unknown_target[target_tag] = 0xff;
        assert_eq!(
            ObjectModule::from_bytes(&unknown_target).unwrap_err(),
            ObjectError::UnknownSymbolTarget(0xff)
        );

        let mut first_import = first_export;
        for target_width in [5usize, 9, 5] {
            let name_len =
                u32::from_le_bytes(good[first_import..first_import + 4].try_into().unwrap())
                    as usize;
            first_import += 4 + name_len + target_width + 24;
        }
        let import_name_len =
            u32::from_le_bytes(good[first_import..first_import + 4].try_into().unwrap()) as usize;
        let import_kind = first_import + 4 + import_name_len;
        let mut unknown_kind = good.clone();
        unknown_kind[import_kind] = 0xff;
        assert_eq!(
            ObjectModule::from_bytes(&unknown_kind).unwrap_err(),
            ObjectError::UnknownSymbolKind(0xff)
        );

        let mut first_relocation = first_import;
        for _ in 0..3 {
            let name_len = u32::from_le_bytes(
                good[first_relocation..first_relocation + 4]
                    .try_into()
                    .unwrap(),
            ) as usize;
            first_relocation += 4 + name_len + 1 + 24;
        }
        let mut unknown_relocation = good;
        unknown_relocation[first_relocation + 8] = 0xff;
        assert_eq!(
            ObjectModule::from_bytes(&unknown_relocation).unwrap_err(),
            ObjectError::UnknownRelocationKind(0xff)
        );

        let mut property_object = object("property-tags", Program::new());
        property_object.metadata.properties.push(ObjectProperty {
            name: "p".into(),
            address: 0,
            comparison: PropertyComparison::Eq,
            value: 0,
            predicate: None,
            touched_addresses: vec![0],
        });
        let mut unknown_comparison = property_object.to_bytes().unwrap();
        let property_name_start = 75
            + property_object.name.len()
            + u32::from_le_bytes(unknown_comparison[16..20].try_into().unwrap()) as usize;
        let comparison = property_name_start + 4 + 1 + 8;
        unknown_comparison[comparison] = 0xff;
        assert_eq!(
            ObjectModule::from_bytes(&unknown_comparison).unwrap_err(),
            ObjectError::UnknownPropertyComparison(0xff)
        );
    }

    #[test]
    fn object_decoder_rejects_invalid_utf8_names() {
        let object = object("m", Program::new());
        let mut bytes = object.to_bytes().unwrap();
        // Fixed v3 header is 75 bytes and this fixture has a one-byte module name.
        bytes[75] = 0xff;
        assert_eq!(
            ObjectModule::from_bytes(&bytes).unwrap_err(),
            ObjectError::InvalidUtf8 {
                field: "module name"
            }
        );

        let encoded = relocatable_fixture().to_bytes().unwrap();
        let module_name_len = u32::from_le_bytes(encoded[12..16].try_into().unwrap()) as usize;
        let code_len = u32::from_le_bytes(encoded[16..20].try_into().unwrap()) as usize;
        let first_export = 75 + module_name_len + code_len;
        let mut bad_export = encoded;
        bad_export[first_export + 4] = 0xff;
        assert_eq!(
            ObjectModule::from_bytes(&bad_export).unwrap_err(),
            ObjectError::InvalidUtf8 {
                field: "export name"
            }
        );
    }

    #[test]
    fn link_order_is_module_name_deterministic() {
        let mut alpha_program = Program::new();
        alpha_program.body.push(Stmt::Push(Value::new(1)));
        let mut beta_program = Program::new();
        beta_program.body.push(Stmt::Push(Value::new(2)));
        let alpha = object("alpha", alpha_program);
        let beta = object("beta", beta_program);

        let forward = link(&[alpha.clone(), beta.clone()]).unwrap();
        let reverse = link(&[beta, alpha]).unwrap();
        assert_eq!(forward, reverse);
        assert_eq!(
            forward.instructions,
            vec![
                Instruction::PushWord(1),
                Instruction::PushWord(2),
                Instruction::Return
            ]
        );
        assert_eq!(forward.to_bytes().unwrap(), reverse.to_bytes().unwrap());
    }

    #[test]
    fn link_rejects_object_count_before_building_aggregate_tables() {
        let template = object("template", Program::new());
        let objects = (0..=MAX_LINK_OBJECTS)
            .map(|index| {
                let mut object = template.clone();
                object.name = format!("module-{index:04}");
                object
            })
            .collect::<Vec<_>>();
        assert!(matches!(
            link(&objects),
            Err(LinkError::LimitExceeded {
                what: "link object",
                count,
                limit: MAX_LINK_OBJECTS,
            }) if count == MAX_LINK_OBJECTS + 1
        ));
    }

    #[test]
    fn compiled_metadata_round_trips_every_stable_section() {
        let mut program = Program::new();
        program.body.push(Stmt::Push(Value::new(7)));
        program.body.push(Stmt::TemporalScope {
            base: 3,
            size: 2,
            cell_bits: 8,
            body: vec![Stmt::Op(OpCode::Nop)],
        });
        program.procedures.push(Procedure {
            name: "effectful".into(),
            params: Vec::new(),
            returns: 0,
            effects: vec![Effect::IO, Effect::Alloc],
            body: Vec::new(),
        });
        program
            .temporal_properties
            .push(crate::ast::TemporalPropertyDeclaration {
                name: "cell_is_seven".into(),
                address: 3,
                comparison: PropertyComparison::Eq,
                value: 7,
                predicate: None,
            });
        let hir = HirProgram::resolve(&program).unwrap();
        let mut code = BytecodeProgram::compile(&hir).unwrap();
        code.source_map.push(SourceMapEntry {
            instruction: 0,
            span: SourceSpan::new(SourceId::new(0), crate::source::TextRange::new(0, 1)),
        });
        code.validate().unwrap();
        let mut sources = SourceManager::new();
        sources.add_virtual("fixture.ouro", "7\nTEMPORAL 3 2 BITS 8 { NOP }\n");

        let mut object = ObjectModule::from_compiled("rich", &hir, code, &sources).unwrap();
        object.metadata.verification = Some(ObjectVerificationArtifact {
            kind: VerificationArtifactKind::BytecodeReport,
            format_version: 1,
            payload: b"verified:structural".to_vec(),
        });
        object.validate().unwrap();

        assert_eq!(object.metadata.constants, vec![7]);
        assert_eq!(object.metadata.temporal_regions.len(), 1);
        assert_eq!(object.metadata.properties[0].name, "cell_is_seven");
        assert_ne!(object.metadata.effects.declared.bits(), 0);
        assert_ne!(object.metadata.effects.observed.bits(), 0);
        assert_eq!(object.metadata.source_files[0].name, "fixture.ouro");
        assert_ne!(object.metadata.source_files[0].content_digest, 0);

        let bytes = object.to_bytes().unwrap();
        let decoded = ObjectModule::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, object);
        assert_eq!(decoded.to_bytes().unwrap(), bytes);
    }

    #[test]
    fn metadata_validation_rejects_code_drift_and_unbounded_proofs() {
        let object = object(
            "metadata",
            Program {
                body: vec![Stmt::Push(Value::new(9))],
                ..Program::new()
            },
        );

        let mut constants = object.clone();
        constants.metadata.constants.push(9);
        assert!(matches!(
            constants.validate(),
            Err(LinkError::MalformedObject { .. })
        ));

        let mut effects = object.clone();
        effects.metadata.effects.observed = ObjectEffects(ObjectEffects::TEMPORAL);
        assert!(matches!(
            effects.validate(),
            Err(LinkError::MalformedObject { .. })
        ));

        let mut sparse_code = object.code.clone();
        sparse_code.source_map.push(SourceMapEntry {
            instruction: 0,
            span: SourceSpan::new(
                SourceId::new(MAX_OBJECT_SOURCE_FILES + 1),
                crate::source::TextRange::new(0, 1),
            ),
        });
        let sparse_source = ObjectModule::new("sparse-source", sparse_code);
        assert!(matches!(
            sparse_source.validate(),
            Err(LinkError::MalformedObject { .. })
        ));

        let mut proof = object;
        proof.metadata.verification = Some(ObjectVerificationArtifact {
            kind: VerificationArtifactKind::SolverCertificate,
            format_version: 1,
            payload: vec![0; MAX_OBJECT_VERIFICATION_BYTES + 1],
        });
        assert!(matches!(
            proof.validate(),
            Err(LinkError::LimitExceeded {
                what: "verification artifact byte",
                ..
            })
        ));
    }

    #[test]
    fn metadata_linking_is_deterministic_preserves_manifests_and_checks_abi() {
        let mut alpha_program = Program::new();
        alpha_program.body.push(Stmt::Push(Value::new(11)));
        let mut alpha = object("alpha", alpha_program);
        alpha.code.source_map.push(SourceMapEntry {
            instruction: 0,
            span: SourceSpan::new(SourceId::new(0), crate::source::TextRange::new(0, 1)),
        });
        alpha.metadata = ObjectMetadata::for_bytecode(&alpha.code);
        alpha.metadata.properties.push(ObjectProperty {
            name: "alpha_property".into(),
            address: 0,
            comparison: PropertyComparison::Eq,
            value: 11,
            predicate: None,
            touched_addresses: vec![0],
        });
        alpha.metadata.verification = Some(ObjectVerificationArtifact {
            kind: VerificationArtifactKind::BytecodeReport,
            format_version: 1,
            payload: vec![1, 2, 3],
        });

        let mut beta_program = Program::new();
        beta_program.body.push(Stmt::Push(Value::new(22)));
        let mut beta = object("beta", beta_program);
        beta.code.source_map.push(SourceMapEntry {
            instruction: 0,
            span: SourceSpan::new(SourceId::new(0), crate::source::TextRange::new(0, 1)),
        });
        beta.metadata = ObjectMetadata::for_bytecode(&beta.code);
        beta.metadata.properties.push(ObjectProperty {
            name: "beta_property".into(),
            address: 1,
            comparison: PropertyComparison::Eq,
            value: 22,
            predicate: None,
            touched_addresses: vec![1],
        });

        let forward = link_with_metadata(&[alpha.clone(), beta.clone()]).unwrap();
        let reverse = link_with_metadata(&[beta.clone(), alpha.clone()]).unwrap();
        assert_eq!(forward, reverse);
        assert_eq!(forward.metadata.constants, vec![11, 22]);
        assert_eq!(
            forward
                .metadata
                .properties
                .iter()
                .map(|property| property.name.as_str())
                .collect::<Vec<_>>(),
            vec!["alpha_property", "beta_property"]
        );
        assert_eq!(forward.metadata.source_files.len(), 2);
        assert_eq!(
            forward
                .code
                .source_map
                .iter()
                .map(|entry| entry.span.source.index())
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(forward.metadata.verification.is_none());

        beta.metadata.runtime_abi = CURRENT_RUNTIME_ABI + 1;
        assert!(matches!(
            link(&[alpha, beta]),
            Err(LinkError::UnsupportedRuntimeAbi { .. })
        ));
    }

    #[test]
    fn relocates_external_procedure_calls_to_typed_ids() {
        let mut provider_program = Program::new();
        provider_program
            .procedures
            .push(procedure("worker", vec![Stmt::Push(Value::new(7))]));
        let mut provider = object("provider", provider_program);
        provider.exports.push(ObjectExport {
            name: "worker".into(),
            target: SymbolTarget::Procedure(ProcedureId::try_from_index(0).unwrap()),
            signature: SymbolSignature::pure(0, 1),
        });

        let mut consumer = object("consumer", Program::new());
        consumer.code.instructions.insert(
            0,
            Instruction::CallProcedure(ProcedureId::try_from_index(0).unwrap()),
        );
        consumer.code.main.end += 1;
        consumer.imports.push(ObjectImport {
            name: "worker".into(),
            kind: SymbolKind::Procedure,
            signature: SymbolSignature::pure(0, 1),
        });
        consumer.relocations.push(ObjectRelocation {
            instruction: 0,
            import: 0,
            kind: RelocationKind::ProcedureCall,
        });

        let linked = link(&[provider, consumer]).unwrap();
        assert!(matches!(
            linked.instructions[0],
            Instruction::CallProcedure(id) if id.index() == 0
        ));
        linked.validate().unwrap();
    }

    #[test]
    fn relocates_local_quotation_ids_across_objects() {
        let mut alpha_program = Program::new();
        alpha_program.quotes.push(vec![Stmt::Op(OpCode::Nop)]);
        alpha_program.body.push(Stmt::PushQuote(QuoteId::new(0)));
        let mut beta_program = Program::new();
        beta_program.quotes.push(vec![Stmt::Op(OpCode::Dup)]);
        beta_program.body.push(Stmt::PushQuote(QuoteId::new(0)));

        let linked = link(&[object("beta", beta_program), object("alpha", alpha_program)]).unwrap();
        assert!(matches!(linked.instructions[0], Instruction::PushQuote(id) if id.as_u64() == 0));
        assert!(matches!(linked.instructions[1], Instruction::PushQuote(id) if id.as_u64() == 1));
        assert_eq!(linked.quotations[0].id.as_u64(), 0);
        assert_eq!(linked.quotations[1].id.as_u64(), 1);
    }

    #[test]
    fn rejects_unresolved_kind_signature_and_duplicate_collisions() {
        let signature = SymbolSignature::pure(0, 1);
        let mut consumer = object("consumer", Program::new());
        consumer.imports.push(ObjectImport {
            name: "missing".into(),
            kind: SymbolKind::Procedure,
            signature,
        });
        assert!(matches!(
            link(&[consumer]),
            Err(LinkError::UnresolvedSymbol { .. })
        ));

        let mut provider_program = Program::new();
        provider_program.procedures.push(procedure("p", Vec::new()));
        let mut provider = object("provider", provider_program);
        provider.exports.push(ObjectExport {
            name: "symbol".into(),
            target: SymbolTarget::Procedure(ProcedureId::try_from_index(0).unwrap()),
            signature,
        });

        let mut wrong_kind = object("wrong-kind", Program::new());
        wrong_kind.imports.push(ObjectImport {
            name: "symbol".into(),
            kind: SymbolKind::Quote,
            signature,
        });
        assert!(matches!(
            link(&[provider.clone(), wrong_kind]),
            Err(LinkError::SymbolKindMismatch { .. })
        ));

        let mut wrong_signature = object("wrong-signature", Program::new());
        wrong_signature.imports.push(ObjectImport {
            name: "symbol".into(),
            kind: SymbolKind::Procedure,
            signature: SymbolSignature::pure(1, 1),
        });
        assert!(matches!(
            link(&[provider.clone(), wrong_signature]),
            Err(LinkError::SignatureMismatch { .. })
        ));

        let mut duplicate = provider.clone();
        duplicate.name = "provider-2".into();
        assert_eq!(
            link(&[provider, duplicate]).unwrap_err(),
            LinkError::DuplicateSymbol("symbol".into())
        );
    }

    #[test]
    fn rejects_malformed_relocations_and_final_control_flow() {
        let mut malformed = object("malformed", Program::new());
        malformed.relocations.push(ObjectRelocation {
            instruction: 0,
            import: 0,
            kind: RelocationKind::ProcedureCall,
        });
        assert!(matches!(
            link(&[malformed]),
            Err(LinkError::MalformedObject { .. })
        ));

        let mut control_program = Program::new();
        control_program.body.push(Stmt::If {
            then_branch: Vec::new(),
            else_branch: None,
        });
        let mut bad_control = object("bad-control", control_program);
        let Instruction::IfFalse { end_target, .. } = &mut bad_control.code.instructions[0] else {
            panic!("expected IF");
        };
        *end_target = u32::MAX;
        assert!(matches!(
            link(&[bad_control]),
            Err(LinkError::MalformedObject { .. })
        ));
    }

    #[test]
    fn foreign_imports_match_every_scalar_type_not_only_stack_width() {
        let target = ForeignId::try_from_index(0).unwrap();
        let descriptor = ForeignEntry {
            id: target,
            library: "process".into(),
            symbol: "typed".into(),
            parameters: vec![crate::bytecode::ForeignScalarType::U64],
            result: Some(crate::bytecode::ForeignScalarType::I64),
            effects: crate::bytecode::ForeignEffects::from_bits(
                crate::bytecode::ForeignEffects::IO,
            )
            .unwrap(),
        };
        let exact = SymbolSignature::foreign(&descriptor);
        let mut provider = object("provider", Program::new());
        provider.code.foreigns.push(descriptor);
        provider.exports.push(ObjectExport {
            name: "typed".into(),
            target: SymbolTarget::Foreign(target),
            signature: exact,
        });

        let mut consumer = object("consumer", Program::new());
        consumer
            .code
            .instructions
            .insert(0, Instruction::CallForeign(target));
        consumer.code.main.end += 1;
        let mut wrong = exact;
        wrong.abi ^= 1 << 2; // Require i64 rather than the exported u64 input.
        consumer.imports.push(ObjectImport {
            name: "typed".into(),
            kind: SymbolKind::Foreign,
            signature: wrong,
        });
        consumer.relocations.push(ObjectRelocation {
            instruction: 0,
            import: 0,
            kind: RelocationKind::ForeignCall,
        });
        consumer.metadata = ObjectMetadata::for_bytecode(&consumer.code);

        assert!(matches!(
            link(&[provider, consumer]),
            Err(LinkError::SignatureMismatch { .. })
        ));
    }

    #[test]
    fn foreign_relocation_preserves_the_complete_scalar_abi_descriptor() {
        let target = ForeignId::try_from_index(0).unwrap();
        let descriptor = ForeignEntry {
            id: target,
            library: "safe-host".into(),
            symbol: "signed_round_trip".into(),
            parameters: vec![
                crate::bytecode::ForeignScalarType::I64,
                crate::bytecode::ForeignScalarType::U64,
            ],
            result: Some(crate::bytecode::ForeignScalarType::I64),
            effects: crate::bytecode::ForeignEffects::from_bits(
                crate::bytecode::ForeignEffects::READS,
            )
            .unwrap(),
        };
        let signature = SymbolSignature::foreign(&descriptor);

        let mut provider = object("provider", Program::new());
        provider.code.foreigns.push(descriptor.clone());
        provider.exports.push(ObjectExport {
            name: "typed.foreign".into(),
            target: SymbolTarget::Foreign(target),
            signature,
        });

        let mut consumer = object("consumer", Program::new());
        consumer.code.instructions.insert(
            0,
            Instruction::CallForeign(ForeignId::try_from_index(99).unwrap()),
        );
        consumer.code.main.end += 1;
        consumer.imports.push(ObjectImport {
            name: "typed.foreign".into(),
            kind: SymbolKind::Foreign,
            signature,
        });
        consumer.relocations.push(ObjectRelocation {
            instruction: 0,
            import: 0,
            kind: RelocationKind::ForeignCall,
        });
        consumer.metadata = ObjectMetadata::for_bytecode(&consumer.code);

        let linked = link(&[provider, consumer]).unwrap();
        assert_eq!(linked.foreigns, vec![descriptor]);
        assert!(matches!(
            linked.instructions[0],
            Instruction::CallForeign(id) if id == target
        ));
    }
}
