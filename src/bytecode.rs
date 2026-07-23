//! Deterministic, structurally validated stack bytecode.
//!
//! This is the first flat executable representation of [`HirProgram`].  It
//! deliberately keeps compiler identities typed: a word literal, quotation,
//! procedure, and foreign call can never be confused merely because their
//! serialized integers happen to have the same value.
//!
//! Bytecode validation is structural.  In particular it does **not** prove
//! stack heights, stack element types, effect contracts, or temporal
//! linearity.  A successful mandatory HIR semantic/type report is a preceding
//! compiler phase; [`BytecodeValidationReport::stack_semantics`] makes that
//! boundary explicit instead of treating artifact validation as a type proof.

use crate::ast::{OpCode, QuoteId};
use crate::hir::{
    CallTarget, ConstantId, ForeignId, HirProgram, HirStmt, HirStmtKind, ProcedureId,
};
use crate::source::{SourceId, SourceSpan, TextRange};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

const MAGIC: &[u8; 8] = b"OUROBC\0\0";
const FORMAT_VERSION: u16 = 2;
const FORMAT_FLAGS: u16 = 0;

/// Hard artifact limits, checked before allocation during decoding.
pub const MAX_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;
/// Maximum number of flat bytecode instructions in one artifact.
pub const MAX_INSTRUCTIONS: usize = 1_000_000;
/// Maximum number of procedures in one artifact.
pub const MAX_PROCEDURES: usize = 100_000;
/// Maximum number of quotations in one artifact.
pub const MAX_QUOTES: usize = 100_000;
/// Maximum number of foreign declarations referenced by one artifact.
pub const MAX_FOREIGNS: usize = 100_000;
/// Maximum UTF-8 byte length of a foreign library or symbol name.
pub const MAX_FOREIGN_NAME_BYTES: usize = 4 * 1024;
/// Maximum number of scalar parameters accepted by the linked host ABI.
pub const MAX_FOREIGN_PARAMETERS: usize = 16;
/// Maximum number of source-map records in one artifact.
pub const MAX_SOURCE_MAP_ENTRIES: usize = 1_000_000;

/// A half-open range of instructions belonging to one callable unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CodeRange {
    /// First instruction in the unit.
    pub start: u32,
    /// One past the final instruction in the unit.
    pub end: u32,
}

impl CodeRange {
    fn contains(self, pc: u32) -> bool {
        self.start <= pc && pc < self.end
    }
}

/// Procedure entry point retaining its typed compiler identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProcedureEntry {
    /// Procedure selected by this entry.
    pub id: ProcedureId,
    /// Complete procedure code, including its final [`Instruction::Return`].
    pub range: CodeRange,
}

/// Quotation entry point retaining its typed compiler identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuoteEntry {
    /// Quotation selected by this entry.
    pub id: QuoteId,
    /// Complete quotation code, including its final [`Instruction::Return`].
    pub range: CodeRange,
}

/// One source location associated with one flat instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceMapEntry {
    /// Instruction index.  Source map entries are strictly increasing.
    pub instruction: u32,
    /// Exact retained source location.
    pub span: SourceSpan,
}

/// One symbolic MANIFEST use emitted as a `PushWord`. Object compilation uses
/// this side result to create a typed constant-literal relocation; executable
/// `OUROBC` remains free of unresolved symbol records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstantReference {
    pub instruction: u32,
    pub constant: ConstantId,
}

/// One exactly marshalable word type in the linked foreign ABI.
///
/// Both variants occupy one Ourochronos word. Signed values retain their
/// two's-complement bit pattern at the boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ForeignScalarType {
    /// Unsigned 64-bit scalar.
    U64,
    /// Signed 64-bit scalar transported as its exact 64-bit representation.
    I64,
}

/// Effect flags retained for a linked foreign declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ForeignEffects(u8);

impl ForeignEffects {
    /// Referentially transparent declaration.
    pub const PURE: u8 = 1 << 0;
    /// General I/O.
    pub const IO: u8 = 1 << 1;
    /// External read.
    pub const READS: u8 = 1 << 2;
    /// External write.
    pub const WRITES: u8 = 1 << 3;
    /// Explicit temporal interaction.
    pub const TEMPORAL: u8 = 1 << 4;
    /// External allocation.
    pub const ALLOC: u8 = 1 << 5;
    const ALL: u8 =
        Self::PURE | Self::IO | Self::READS | Self::WRITES | Self::TEMPORAL | Self::ALLOC;

    /// Construct a checked bit set.
    pub const fn from_bits(bits: u8) -> Option<Self> {
        if bits != 0 && bits & !Self::ALL == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }

    /// Raw stable artifact bits.
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Whether every declared effect is pure.
    pub const fn is_pure(self) -> bool {
        self.0 == Self::PURE
    }
}

/// A linked foreign target with the complete narrow scalar ABI contract.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ForeignEntry {
    /// Canonical table identity.
    pub id: ForeignId,
    /// Host table namespace (the source `FOREIGN` library string).
    pub library: String,
    /// Exact external symbol, including a source `AS` alias when present.
    pub symbol: String,
    /// Scalar arguments in source/stack order.
    pub parameters: Vec<ForeignScalarType>,
    /// Optional single scalar result.
    pub result: Option<ForeignScalarType>,
    /// Declared external effects.
    pub effects: ForeignEffects,
}

/// Flat executable instruction.
///
/// Control instructions contain instruction indices, never byte offsets.
/// Their redundant pairing metadata lets the validator reject malformed or
/// crossing structured control flow before a runtime sees the program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Instruction {
    /// Execute a source primitive.
    Primitive(OpCode),
    /// Push an ordinary 64-bit machine word.
    PushWord(u64),
    /// Push a checked quotation identity.
    PushQuote(QuoteId),
    /// Call an Ourochronos procedure.
    CallProcedure(ProcedureId),
    /// Call a declared foreign function.
    CallForeign(ForeignId),
    /// Pop a condition and enter the false branch when it is zero.
    IfFalse {
        /// First instruction of the else branch, or `end_target` without one.
        else_target: u32,
        /// First instruction following the entire conditional.
        end_target: u32,
        /// Distinguishes no else branch from a present but empty else branch.
        has_else: bool,
    },
    /// Canonical forward jump separating an IF's then and else branches.
    Jump {
        /// First instruction after the IF.
        target: u32,
    },
    /// Pop a loop condition and leave the loop when it is zero.
    WhileFalse {
        /// First instruction of the condition block.
        loop_start: u32,
        /// First instruction after the loop.
        end_target: u32,
    },
    /// Canonical back edge at the end of a WHILE body.
    LoopBack {
        /// First instruction of the condition block.
        target: u32,
    },
    /// Enter an isolated temporal-memory scope.
    TemporalEnter {
        /// Base temporal cell address.
        base: u64,
        /// Number of cells in the region.
        size: u64,
        /// Stored width of every cell.
        cell_bits: u8,
        /// Index of the paired [`Instruction::TemporalExit`].
        exit_target: u32,
    },
    /// Exit a temporal-memory scope.
    TemporalExit {
        /// Index of the paired [`Instruction::TemporalEnter`].
        enter_target: u32,
    },
    /// Return from the main body, procedure, or quotation.
    Return,
}

/// Whether structural bytecode validation includes stack semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StackSemanticStatus {
    /// Stack semantics are intentionally a mandatory preceding HIR report.
    NotChecked,
}

/// Successful bytecode validation and its deliberately limited proof scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BytecodeValidationReport {
    /// Number of structurally validated instructions.
    pub instructions: usize,
    /// Stack/type checking is not duplicated by artifact validation.
    pub stack_semantics: StackSemanticStatus,
}

/// A deterministic bytecode artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BytecodeProgram {
    /// Flat instruction vector.
    pub instructions: Vec<Instruction>,
    /// Top-level program entry range.
    pub main: CodeRange,
    /// Quotation entries in typed identity order.
    pub quotations: Vec<QuoteEntry>,
    /// Procedure entries in typed identity order.
    pub procedures: Vec<ProcedureEntry>,
    /// Linked foreign targets in typed identity order.
    pub foreigns: Vec<ForeignEntry>,
    /// Optional parallel instruction-to-source map.
    pub source_map: Vec<SourceMapEntry>,
}

/// Bytecode compilation, decoding, or structural validation failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BytecodeError {
    /// A bounded artifact table exceeded its declared hard limit.
    LimitExceeded {
        /// Table or byte sequence being limited.
        what: &'static str,
        /// Observed count.
        count: usize,
        /// Maximum accepted count.
        limit: usize,
    },
    /// A HIR table did not contain its required source-order typed identity.
    InvalidHirIdentity {
        /// HIR table name.
        table: &'static str,
        /// Source-order table index.
        index: usize,
        /// Encoded identity found at the index.
        actual: u64,
    },
    /// HIR attempted to treat a runtime/provenance-bearing value as a literal.
    NonLiteralValue,
    /// A source foreign declaration is outside the deliberately narrow host
    /// ABI and therefore cannot enter an executable artifact.
    UnsupportedForeignSignature {
        /// Source-level foreign function name.
        foreign: String,
        /// Exact rejected shape or declaration defect.
        reason: String,
    },
    /// An index could not be represented by the artifact format.
    IndexOverflow {
        /// Kind of index.
        what: &'static str,
        /// Unrepresentable value.
        value: usize,
    },
    /// Artifact magic was absent or incorrect.
    BadMagic,
    /// Artifact version is not supported by this implementation.
    UnsupportedVersion(u16),
    /// Reserved format flags were nonzero.
    UnsupportedFlags(u16),
    /// Input ended before a complete field could be decoded.
    Truncated {
        /// Byte offset at which the field began.
        offset: usize,
        /// Bytes required for the field.
        needed: usize,
    },
    /// Bytes remained after the declared artifact ended.
    TrailingBytes(usize),
    /// An instruction tag is not defined by this format version.
    UnknownInstructionTag(u8),
    /// A primitive index is outside [`OpCode::ALL`].
    UnknownPrimitive(u16),
    /// An opcode was missing from [`OpCode::ALL`] while encoding.
    UnregisteredPrimitive(&'static str),
    /// A boolean field used a byte other than zero or one.
    InvalidBoolean(u8),
    /// A decoded source identity or offset cannot fit this host's `usize`.
    HostIndexOverflow {
        /// Field name.
        what: &'static str,
        /// Serialized value.
        value: u64,
    },
    /// A structural invariant was violated.
    InvalidStructure(String),
}

impl fmt::Display for BytecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LimitExceeded { what, count, limit } => {
                write!(formatter, "{what} count {count} exceeds limit {limit}")
            }
            Self::InvalidHirIdentity {
                table,
                index,
                actual,
            } => write!(
                formatter,
                "{table} entry {index} carries noncanonical identity {actual}"
            ),
            Self::NonLiteralValue => write!(
                formatter,
                "HIR Push contains runtime provenance and cannot become a literal"
            ),
            Self::UnsupportedForeignSignature { foreign, reason } => {
                write!(formatter, "foreign {foreign:?} is not linkable: {reason}")
            }
            Self::IndexOverflow { what, value } => {
                write!(
                    formatter,
                    "{what} index {value} does not fit the bytecode format"
                )
            }
            Self::BadMagic => write!(formatter, "invalid bytecode magic"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported bytecode version {version}")
            }
            Self::UnsupportedFlags(flags) => {
                write!(formatter, "unsupported bytecode flags 0x{flags:04x}")
            }
            Self::Truncated { offset, needed } => write!(
                formatter,
                "truncated bytecode at offset {offset}: need {needed} bytes"
            ),
            Self::TrailingBytes(count) => write!(formatter, "{count} trailing artifact bytes"),
            Self::UnknownInstructionTag(tag) => {
                write!(formatter, "unknown bytecode instruction tag {tag}")
            }
            Self::UnknownPrimitive(opcode) => {
                write!(formatter, "unknown primitive opcode index {opcode}")
            }
            Self::UnregisteredPrimitive(name) => {
                write!(formatter, "primitive {name} is absent from OpCode::ALL")
            }
            Self::InvalidBoolean(value) => {
                write!(formatter, "invalid serialized boolean byte {value}")
            }
            Self::HostIndexOverflow { what, value } => {
                write!(
                    formatter,
                    "serialized {what} value {value} exceeds host usize"
                )
            }
            Self::InvalidStructure(message) => write!(formatter, "invalid bytecode: {message}"),
        }
    }
}

impl Error for BytecodeError {}

fn lower_foreign_entry(
    id: ForeignId,
    declaration: &crate::parser::FFIDeclaration,
) -> Result<ForeignEntry, BytecodeError> {
    use crate::runtime::ffi::{FFIEffect, FFIType};

    let signature = &declaration.signature;
    let reject = |reason: String| BytecodeError::UnsupportedForeignSignature {
        foreign: signature.name.clone(),
        reason,
    };
    let scalar = |typ: FFIType| match typ {
        FFIType::U64 => Ok(ForeignScalarType::U64),
        FFIType::I64 => Ok(ForeignScalarType::I64),
        other => Err(reject(format!(
            "type {other:?} is outside the scalar u64/i64 ABI"
        ))),
    };

    if signature.library.trim().is_empty() || signature.library.contains('\0') {
        return Err(reject("library name is empty or contains NUL".into()));
    }
    if signature.library.len() > MAX_FOREIGN_NAME_BYTES {
        return Err(reject(format!(
            "library name exceeds {MAX_FOREIGN_NAME_BYTES} UTF-8 bytes"
        )));
    }
    let symbol = declaration
        .symbol_name
        .clone()
        .unwrap_or_else(|| signature.name.clone());
    if symbol.trim().is_empty() || symbol.contains('\0') {
        return Err(reject("symbol name is empty or contains NUL".into()));
    }
    if symbol.len() > MAX_FOREIGN_NAME_BYTES {
        return Err(reject(format!(
            "symbol name exceeds {MAX_FOREIGN_NAME_BYTES} UTF-8 bytes"
        )));
    }
    if signature.params.len() > MAX_FOREIGN_PARAMETERS {
        return Err(reject(format!(
            "{} parameters exceed the ABI limit {MAX_FOREIGN_PARAMETERS}",
            signature.params.len()
        )));
    }

    let parameters = signature
        .params
        .iter()
        .map(|parameter| scalar(parameter.typ))
        .collect::<Result<Vec<_>, _>>()?;
    let result = match signature.returns.as_slice() {
        [] | [FFIType::Void] => None,
        [typ] => Some(scalar(*typ)?),
        returns => {
            return Err(reject(format!(
                "{} return values exceed the ABI limit 1",
                returns.len()
            )))
        }
    };
    let mut effect_bits = 0u8;
    for effect in &signature.effects {
        effect_bits |= match effect {
            FFIEffect::Pure => ForeignEffects::PURE,
            FFIEffect::IO => ForeignEffects::IO,
            FFIEffect::Reads => ForeignEffects::READS,
            FFIEffect::Writes => ForeignEffects::WRITES,
            FFIEffect::Temporal => ForeignEffects::TEMPORAL,
            FFIEffect::Alloc => ForeignEffects::ALLOC,
        };
    }
    let effects = ForeignEffects::from_bits(effect_bits)
        .ok_or_else(|| reject("effect set is empty or invalid".into()))?;
    if effects.bits() & ForeignEffects::PURE != 0 && !effects.is_pure() {
        return Err(reject(
            "PURE cannot be combined with external effect annotations".into(),
        ));
    }

    Ok(ForeignEntry {
        id,
        library: signature.library.clone(),
        symbol,
        parameters,
        result,
        effects,
    })
}

fn validate_foreign_entry(foreign: &ForeignEntry) -> Result<(), BytecodeError> {
    let invalid_name = |value: &str| {
        value.trim().is_empty() || value.contains('\0') || value.len() > MAX_FOREIGN_NAME_BYTES
    };
    if invalid_name(&foreign.library) {
        return invalid(format!(
            "foreign {} has an invalid library name",
            foreign.id.as_u32()
        ));
    }
    if invalid_name(&foreign.symbol) {
        return invalid(format!(
            "foreign {} has an invalid symbol name",
            foreign.id.as_u32()
        ));
    }
    check_limit(
        "foreign parameter",
        foreign.parameters.len(),
        MAX_FOREIGN_PARAMETERS,
    )?;
    if ForeignEffects::from_bits(foreign.effects.bits()) != Some(foreign.effects)
        || (foreign.effects.bits() & ForeignEffects::PURE != 0 && !foreign.effects.is_pure())
    {
        return invalid(format!(
            "foreign {} has an invalid effect set",
            foreign.id.as_u32()
        ));
    }
    Ok(())
}

impl BytecodeProgram {
    /// Lower typed HIR into deterministic flat bytecode.
    ///
    /// This method performs identity and structural checks, but it is not a
    /// substitute for the mandatory preceding HIR stack/type report.  Runtime
    /// provenance on a purported literal is rejected rather than silently
    /// discarded.
    pub fn compile(program: &HirProgram) -> Result<Self, BytecodeError> {
        Self::compile_with_symbols(program).map(|(code, _)| code)
    }

    /// Lower HIR and retain exact instruction sites originating from typed
    /// MANIFEST uses for relocation-aware object construction.
    pub fn compile_with_symbols(
        program: &HirProgram,
    ) -> Result<(Self, Vec<ConstantReference>), BytecodeError> {
        check_limit("constant", program.constants.len(), MAX_INSTRUCTIONS)?;
        check_limit("procedure", program.procedures.len(), MAX_PROCEDURES)?;
        check_limit("quotation", program.quotes.len(), MAX_QUOTES)?;
        check_limit("foreign", program.foreigns.len(), MAX_FOREIGNS)?;

        for (index, constant) in program.constants.iter().enumerate() {
            let expected =
                ConstantId::try_from_index(index).ok_or(BytecodeError::IndexOverflow {
                    what: "constant",
                    value: index,
                })?;
            if constant.id != expected {
                return Err(BytecodeError::InvalidHirIdentity {
                    table: "constant",
                    index,
                    actual: u64::from(constant.id.as_u32()),
                });
            }
        }
        for (index, procedure) in program.procedures.iter().enumerate() {
            let expected =
                ProcedureId::try_from_index(index).ok_or(BytecodeError::IndexOverflow {
                    what: "procedure",
                    value: index,
                })?;
            if procedure.id != expected {
                return Err(BytecodeError::InvalidHirIdentity {
                    table: "procedure",
                    index,
                    actual: u64::from(procedure.id.as_u32()),
                });
            }
        }
        for (index, quote) in program.quotes.iter().enumerate() {
            let expected = u64::try_from(index).map_err(|_| BytecodeError::IndexOverflow {
                what: "quotation",
                value: index,
            })?;
            if quote.id != QuoteId::new(expected) {
                return Err(BytecodeError::InvalidHirIdentity {
                    table: "quotation",
                    index,
                    actual: quote.id.as_u64(),
                });
            }
        }
        for (index, foreign) in program.foreigns.iter().enumerate() {
            let expected =
                ForeignId::try_from_index(index).ok_or(BytecodeError::IndexOverflow {
                    what: "foreign",
                    value: index,
                })?;
            if foreign.id != expected {
                return Err(BytecodeError::InvalidHirIdentity {
                    table: "foreign",
                    index,
                    actual: u64::from(foreign.id.as_u32()),
                });
            }
        }

        let foreigns = program
            .foreigns
            .iter()
            .map(|foreign| lower_foreign_entry(foreign.id, &foreign.declaration))
            .collect::<Result<Vec<_>, _>>()?;
        let mut compiler = Compiler::default();
        let main = compiler.compile_unit(&program.body)?;

        let mut quotations = Vec::with_capacity(program.quotes.len());
        for quote in &program.quotes {
            quotations.push(QuoteEntry {
                id: quote.id,
                range: compiler.compile_unit(&quote.body)?,
            });
        }

        let mut procedures = Vec::with_capacity(program.procedures.len());
        for procedure in &program.procedures {
            procedures.push(ProcedureEntry {
                id: procedure.id,
                range: compiler.compile_unit(&procedure.body)?,
            });
        }

        let Compiler {
            instructions,
            source_map,
            constant_references,
        } = compiler;
        let bytecode = Self {
            instructions,
            main,
            quotations,
            procedures,
            foreigns,
            source_map,
        };
        bytecode.validate()?;
        Ok((bytecode, constant_references))
    }

    /// Validate artifact structure without claiming to check stack semantics.
    pub fn validate(&self) -> Result<BytecodeValidationReport, BytecodeError> {
        self.validate_with_unresolved(&HashSet::new())
    }

    /// Validate relocatable bytecode while permitting typed identity operands
    /// to remain unresolved at exactly the declared instruction sites.
    pub(crate) fn validate_relocatable(
        &self,
        unresolved: &HashSet<u32>,
    ) -> Result<BytecodeValidationReport, BytecodeError> {
        self.validate_with_unresolved(unresolved)
    }

    fn validate_with_unresolved(
        &self,
        unresolved: &HashSet<u32>,
    ) -> Result<BytecodeValidationReport, BytecodeError> {
        check_limit("instruction", self.instructions.len(), MAX_INSTRUCTIONS)?;
        check_limit("procedure", self.procedures.len(), MAX_PROCEDURES)?;
        check_limit("quotation", self.quotations.len(), MAX_QUOTES)?;
        check_limit("foreign", self.foreigns.len(), MAX_FOREIGNS)?;
        check_limit("source map", self.source_map.len(), MAX_SOURCE_MAP_ENTRIES)?;

        for (index, foreign) in self.foreigns.iter().enumerate() {
            let expected =
                ForeignId::try_from_index(index).ok_or(BytecodeError::IndexOverflow {
                    what: "foreign",
                    value: index,
                })?;
            if foreign.id != expected {
                return invalid(format!(
                    "foreign entry {index} has identity {}",
                    foreign.id.as_u32()
                ));
            }
            validate_foreign_entry(foreign)?;
        }

        self.validate_layout()?;
        self.validate_source_map()?;

        for &pc in unresolved {
            let instruction = self.instruction(pc)?;
            if !matches!(
                instruction,
                Instruction::PushWord(_)
                    | Instruction::PushQuote(_)
                    | Instruction::CallProcedure(_)
                    | Instruction::CallForeign(_)
            ) {
                return invalid(format!(
                    "declared unresolved site {pc} is not a relocatable operand"
                ));
            }
        }

        let mut expected_jumps = HashMap::<u32, u32>::new();
        let mut expected_loop_backs = HashMap::<u32, u32>::new();
        let mut intervals = Vec::<ControlInterval>::new();

        for range in self.all_ranges() {
            self.validate_unit(
                range,
                &mut expected_jumps,
                &mut expected_loop_backs,
                &mut intervals,
                unresolved,
            )?;
        }

        for (index, instruction) in self.instructions.iter().enumerate() {
            let pc = u32_index(index, "instruction")?;
            match instruction {
                Instruction::Jump { target } => match expected_jumps.get(&pc) {
                    Some(expected) if expected == target => {}
                    _ => {
                        return invalid(format!(
                            "instruction {pc} is an unpaired or mistargeted IF jump"
                        ));
                    }
                },
                Instruction::LoopBack { target } => match expected_loop_backs.get(&pc) {
                    Some(expected) if expected == target => {}
                    _ => {
                        return invalid(format!(
                            "instruction {pc} is an unpaired or mistargeted loop back edge"
                        ));
                    }
                },
                _ => {}
            }
        }

        validate_laminar_intervals(&mut intervals)?;

        Ok(BytecodeValidationReport {
            instructions: self.instructions.len(),
            stack_semantics: StackSemanticStatus::NotChecked,
        })
    }

    /// Encode a validated artifact in the versioned little-endian format.
    pub fn to_bytes(&self) -> Result<Vec<u8>, BytecodeError> {
        self.validate()?;

        self.encode_bytes()
    }

    /// Encode bytecode whose unresolved identity operands have been validated
    /// by the object layer.
    pub(crate) fn to_relocatable_bytes(&self) -> Result<Vec<u8>, BytecodeError> {
        self.encode_bytes()
    }

    fn encode_bytes(&self) -> Result<Vec<u8>, BytecodeError> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        put_u16(&mut bytes, FORMAT_VERSION);
        put_u16(&mut bytes, FORMAT_FLAGS);
        put_u32(
            &mut bytes,
            count_u32(self.instructions.len(), "instruction")?,
        );
        put_u32(&mut bytes, count_u32(self.procedures.len(), "procedure")?);
        put_u32(&mut bytes, count_u32(self.quotations.len(), "quotation")?);
        put_u32(&mut bytes, count_u32(self.foreigns.len(), "foreign")?);
        put_u32(&mut bytes, count_u32(self.source_map.len(), "source map")?);
        put_range(&mut bytes, self.main);

        for instruction in &self.instructions {
            encode_instruction(&mut bytes, *instruction)?;
            if bytes.len() > MAX_ARTIFACT_BYTES {
                return Err(BytecodeError::LimitExceeded {
                    what: "artifact byte",
                    count: bytes.len(),
                    limit: MAX_ARTIFACT_BYTES,
                });
            }
        }
        for entry in &self.procedures {
            put_u32(&mut bytes, entry.id.as_u32());
            put_range(&mut bytes, entry.range);
        }
        for entry in &self.quotations {
            put_u64(&mut bytes, entry.id.as_u64());
            put_range(&mut bytes, entry.range);
        }
        for entry in &self.foreigns {
            encode_foreign_entry(&mut bytes, entry)?;
        }
        for entry in &self.source_map {
            put_u32(&mut bytes, entry.instruction);
            put_u64(
                &mut bytes,
                u64::try_from(entry.span.source.index()).map_err(|_| {
                    BytecodeError::IndexOverflow {
                        what: "source identity",
                        value: entry.span.source.index(),
                    }
                })?,
            );
            put_u64(
                &mut bytes,
                u64::try_from(entry.span.range.start).map_err(|_| {
                    BytecodeError::IndexOverflow {
                        what: "source range start",
                        value: entry.span.range.start,
                    }
                })?,
            );
            put_u64(
                &mut bytes,
                u64::try_from(entry.span.range.end).map_err(|_| BytecodeError::IndexOverflow {
                    what: "source range end",
                    value: entry.span.range.end,
                })?,
            );
        }

        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(BytecodeError::LimitExceeded {
                what: "artifact byte",
                count: bytes.len(),
                limit: MAX_ARTIFACT_BYTES,
            });
        }
        Ok(bytes)
    }

    /// Decode, bound-check, and structurally validate an artifact.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BytecodeError> {
        let program = Self::decode_bytes(bytes)?;
        program.validate()?;
        Ok(program)
    }

    /// Decode bounded bytecode for subsequent object-aware relocation
    /// validation.
    pub(crate) fn from_relocatable_bytes(bytes: &[u8]) -> Result<Self, BytecodeError> {
        Self::decode_bytes(bytes)
    }

    fn decode_bytes(bytes: &[u8]) -> Result<Self, BytecodeError> {
        check_limit("artifact byte", bytes.len(), MAX_ARTIFACT_BYTES)?;
        let mut reader = Reader::new(bytes);
        if reader.take(MAGIC.len())? != MAGIC {
            return Err(BytecodeError::BadMagic);
        }
        let version = reader.u16()?;
        if version != FORMAT_VERSION {
            return Err(BytecodeError::UnsupportedVersion(version));
        }
        let flags = reader.u16()?;
        if flags != FORMAT_FLAGS {
            return Err(BytecodeError::UnsupportedFlags(flags));
        }

        let instruction_count = reader.bounded_count("instruction", MAX_INSTRUCTIONS)?;
        let procedure_count = reader.bounded_count("procedure", MAX_PROCEDURES)?;
        let quotation_count = reader.bounded_count("quotation", MAX_QUOTES)?;
        let foreign_count = reader.bounded_count("foreign", MAX_FOREIGNS)?;
        let source_map_count = reader.bounded_count("source map", MAX_SOURCE_MAP_ENTRIES)?;
        let main = reader.range()?;

        let mut instructions = Vec::with_capacity(instruction_count);
        for _ in 0..instruction_count {
            instructions.push(decode_instruction(&mut reader)?);
        }

        let mut procedures = Vec::with_capacity(procedure_count);
        for _ in 0..procedure_count {
            let raw = reader.u32()?;
            let id = ProcedureId::try_from_index(raw as usize).ok_or(
                BytecodeError::HostIndexOverflow {
                    what: "procedure identity",
                    value: u64::from(raw),
                },
            )?;
            procedures.push(ProcedureEntry {
                id,
                range: reader.range()?,
            });
        }

        let mut quotations = Vec::with_capacity(quotation_count);
        for _ in 0..quotation_count {
            quotations.push(QuoteEntry {
                id: QuoteId::new(reader.u64()?),
                range: reader.range()?,
            });
        }

        let mut foreigns = Vec::with_capacity(foreign_count);
        for index in 0..foreign_count {
            foreigns.push(decode_foreign_entry(&mut reader, index)?);
        }

        let mut source_map = Vec::with_capacity(source_map_count);
        for _ in 0..source_map_count {
            let instruction = reader.u32()?;
            let source_raw = reader.u64()?;
            let start_raw = reader.u64()?;
            let end_raw = reader.u64()?;
            let source = usize_from_u64("source identity", source_raw)?;
            let start = usize_from_u64("source range start", start_raw)?;
            let end = usize_from_u64("source range end", end_raw)?;
            source_map.push(SourceMapEntry {
                instruction,
                span: SourceSpan::new(SourceId::new(source), TextRange::new(start, end)),
            });
        }

        if reader.remaining() != 0 {
            return Err(BytecodeError::TrailingBytes(reader.remaining()));
        }

        let program = Self {
            instructions,
            main,
            quotations,
            procedures,
            foreigns,
            source_map,
        };
        Ok(program)
    }

    fn all_ranges(&self) -> Vec<CodeRange> {
        let mut ranges = Vec::with_capacity(1 + self.quotations.len() + self.procedures.len());
        ranges.push(self.main);
        ranges.extend(self.quotations.iter().map(|entry| entry.range));
        ranges.extend(self.procedures.iter().map(|entry| entry.range));
        ranges
    }

    fn validate_layout(&self) -> Result<(), BytecodeError> {
        for (index, entry) in self.quotations.iter().enumerate() {
            let expected = u64::try_from(index).map_err(|_| BytecodeError::IndexOverflow {
                what: "quotation",
                value: index,
            })?;
            if entry.id != QuoteId::new(expected) {
                return invalid(format!(
                    "quotation entry {index} has identity {}",
                    entry.id.as_u64()
                ));
            }
        }
        for (index, entry) in self.procedures.iter().enumerate() {
            let expected =
                ProcedureId::try_from_index(index).ok_or(BytecodeError::IndexOverflow {
                    what: "procedure",
                    value: index,
                })?;
            if entry.id != expected {
                return invalid(format!(
                    "procedure entry {index} has identity {}",
                    entry.id.as_u32()
                ));
            }
        }

        let ranges = self.all_ranges();
        let mut expected_start = 0u32;
        let instruction_end = u32_index(self.instructions.len(), "instruction count")?;
        for (index, range) in ranges.iter().copied().enumerate() {
            if range.start != expected_start {
                return invalid(format!(
                    "code range {index} starts at {}, expected {expected_start}",
                    range.start
                ));
            }
            if range.end <= range.start || range.end > instruction_end {
                return invalid(format!(
                    "code range {index} is invalid: {}..{} for {instruction_end} instructions",
                    range.start, range.end
                ));
            }
            expected_start = range.end;
        }
        if expected_start != instruction_end {
            return invalid(format!(
                "code ranges end at {expected_start}, but instruction vector ends at {instruction_end}"
            ));
        }
        Ok(())
    }

    fn validate_source_map(&self) -> Result<(), BytecodeError> {
        let instruction_end = u32_index(self.instructions.len(), "instruction count")?;
        let mut previous = None;
        for entry in &self.source_map {
            if entry.instruction >= instruction_end {
                return invalid(format!(
                    "source map instruction {} is out of range",
                    entry.instruction
                ));
            }
            if previous.is_some_and(|pc| entry.instruction <= pc) {
                return invalid("source map entries are not strictly increasing".to_string());
            }
            if entry.span.range.start > entry.span.range.end {
                return invalid(format!(
                    "source map instruction {} has reversed range {}..{}",
                    entry.instruction, entry.span.range.start, entry.span.range.end
                ));
            }
            previous = Some(entry.instruction);
        }
        Ok(())
    }

    fn validate_unit(
        &self,
        range: CodeRange,
        expected_jumps: &mut HashMap<u32, u32>,
        expected_loop_backs: &mut HashMap<u32, u32>,
        intervals: &mut Vec<ControlInterval>,
        unresolved: &HashSet<u32>,
    ) -> Result<(), BytecodeError> {
        let return_pc = range.end - 1;
        if self.instruction(return_pc)? != &Instruction::Return {
            return invalid(format!(
                "unit {}..{} does not terminate with RETURN",
                range.start, range.end
            ));
        }
        for pc in range.start..return_pc {
            if self.instruction(pc)? == &Instruction::Return {
                return invalid(format!("unit contains early RETURN at instruction {pc}"));
            }
        }

        let depths = self.temporal_depths(range, intervals)?;
        for pc in range.start..range.end {
            let instruction = *self.instruction(pc)?;
            match instruction {
                Instruction::Primitive(_) | Instruction::PushWord(_) | Instruction::Return => {}
                Instruction::PushQuote(id) => {
                    if !unresolved.contains(&pc) && id.as_u64() >= self.quotations.len() as u64 {
                        return invalid(format!(
                            "instruction {pc} references missing quotation {}",
                            id.as_u64()
                        ));
                    }
                }
                Instruction::CallProcedure(id) => {
                    if !unresolved.contains(&pc) && id.index() >= self.procedures.len() {
                        return invalid(format!(
                            "instruction {pc} references missing procedure {}",
                            id.as_u32()
                        ));
                    }
                }
                Instruction::CallForeign(id) => {
                    if !unresolved.contains(&pc) && id.index() >= self.foreigns.len() {
                        return invalid(format!(
                            "instruction {pc} references missing foreign {}",
                            id.as_u32()
                        ));
                    }
                }
                Instruction::IfFalse {
                    else_target,
                    end_target,
                    has_else,
                } => {
                    require_target(range, else_target, "IF else", pc)?;
                    require_target(range, end_target, "IF end", pc)?;
                    if else_target <= pc || end_target < else_target {
                        return invalid(format!(
                            "instruction {pc} has malformed IF targets {else_target} and {end_target}"
                        ));
                    }
                    require_same_depth(&depths, range, pc, else_target, "IF else")?;
                    require_same_depth(&depths, range, pc, end_target, "IF end")?;
                    if has_else {
                        let jump_pc = else_target.checked_sub(1).ok_or_else(|| {
                            BytecodeError::InvalidStructure(format!(
                                "instruction {pc} has impossible else target"
                            ))
                        })?;
                        if jump_pc <= pc {
                            return invalid(format!(
                                "instruction {pc} has no then branch boundary jump"
                            ));
                        }
                        insert_unique(expected_jumps, jump_pc, end_target, "IF jump")?;
                    } else if else_target != end_target {
                        return invalid(format!(
                            "instruction {pc} has distinct else target without an else branch"
                        ));
                    }
                    intervals.push(ControlInterval::new(pc, end_target, "IF", pc)?);
                }
                Instruction::Jump { target } => {
                    require_target(range, target, "jump", pc)?;
                    require_same_depth(&depths, range, pc, target, "jump")?;
                }
                Instruction::WhileFalse {
                    loop_start,
                    end_target,
                } => {
                    require_target(range, loop_start, "WHILE start", pc)?;
                    require_target(range, end_target, "WHILE end", pc)?;
                    if loop_start > pc || end_target <= pc + 1 {
                        return invalid(format!(
                            "instruction {pc} has malformed WHILE targets {loop_start} and {end_target}"
                        ));
                    }
                    let back_pc = end_target - 1;
                    require_same_depth(&depths, range, pc, loop_start, "WHILE start")?;
                    require_same_depth(&depths, range, pc, end_target, "WHILE end")?;
                    insert_unique(expected_loop_backs, back_pc, loop_start, "WHILE back edge")?;
                    intervals.push(ControlInterval::new(loop_start, end_target, "WHILE", pc)?);
                }
                Instruction::LoopBack { target } => {
                    require_target(range, target, "loop back", pc)?;
                    require_same_depth(&depths, range, pc, target, "loop back")?;
                }
                Instruction::TemporalEnter {
                    base,
                    size,
                    cell_bits,
                    exit_target,
                } => {
                    if size == 0
                        || !(1..=64).contains(&cell_bits)
                        || base.checked_add(size).is_none()
                    {
                        return invalid(format!(
                            "instruction {pc} has invalid temporal region base={base}, size={size}, bits={cell_bits}"
                        ));
                    }
                    require_target(range, exit_target, "temporal exit", pc)?;
                    if exit_target <= pc {
                        return invalid(format!(
                            "instruction {pc} has non-forward temporal exit {exit_target}"
                        ));
                    }
                    match self.instruction(exit_target)? {
                        Instruction::TemporalExit { enter_target } if *enter_target == pc => {}
                        _ => {
                            return invalid(format!(
                                "instruction {pc} does not pair with temporal exit {exit_target}"
                            ));
                        }
                    }
                }
                Instruction::TemporalExit { enter_target } => {
                    require_target(range, enter_target, "temporal enter", pc)?;
                    match self.instruction(enter_target)? {
                        Instruction::TemporalEnter { exit_target, .. } if *exit_target == pc => {}
                        _ => {
                            return invalid(format!(
                                "instruction {pc} does not pair with temporal enter {enter_target}"
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn temporal_depths(
        &self,
        range: CodeRange,
        intervals: &mut Vec<ControlInterval>,
    ) -> Result<Vec<usize>, BytecodeError> {
        let mut depths = vec![0usize; (range.end - range.start) as usize];
        let mut stack = Vec::<(u32, u32)>::new();
        for pc in range.start..range.end {
            depths[(pc - range.start) as usize] = stack.len();
            match *self.instruction(pc)? {
                Instruction::TemporalEnter { exit_target, .. } => {
                    stack.push((pc, exit_target));
                }
                Instruction::TemporalExit { enter_target } => {
                    let Some((open, expected_exit)) = stack.pop() else {
                        return invalid(format!(
                            "instruction {pc} exits a temporal scope that is not open"
                        ));
                    };
                    if open != enter_target || expected_exit != pc {
                        return invalid(format!(
                            "instruction {pc} closes temporal scope {enter_target}, expected {open}"
                        ));
                    }
                    intervals.push(ControlInterval::new(open, pc + 1, "TEMPORAL", pc)?);
                }
                _ => {}
            }
        }
        if let Some((open, _)) = stack.last() {
            return invalid(format!("temporal scope at instruction {open} is unclosed"));
        }
        Ok(depths)
    }

    fn instruction(&self, pc: u32) -> Result<&Instruction, BytecodeError> {
        self.instructions.get(pc as usize).ok_or_else(|| {
            BytecodeError::InvalidStructure(format!("instruction index {pc} is out of range"))
        })
    }
}

#[derive(Default)]
struct Compiler {
    instructions: Vec<Instruction>,
    source_map: Vec<SourceMapEntry>,
    constant_references: Vec<ConstantReference>,
}

enum Work<'a> {
    Block(&'a [HirStmt]),
    Statement(&'a HirStmt),
    FinishIfThen {
        marker: u32,
        has_else: bool,
        location: Option<SourceSpan>,
    },
    FinishIf {
        marker: u32,
    },
    FinishWhileCondition {
        loop_start: u32,
        body: &'a [HirStmt],
        location: Option<SourceSpan>,
    },
    FinishWhile {
        marker: u32,
        loop_start: u32,
        location: Option<SourceSpan>,
    },
    FinishTemporal {
        enter: u32,
        location: Option<SourceSpan>,
    },
}

impl Compiler {
    fn compile_unit(&mut self, block: &[HirStmt]) -> Result<CodeRange, BytecodeError> {
        let start = self.pc()?;
        let mut work = vec![Work::Block(block)];

        while let Some(item) = work.pop() {
            match item {
                Work::Block(statements) => {
                    work.extend(statements.iter().rev().map(Work::Statement));
                }
                Work::Statement(statement) => match &statement.kind {
                    HirStmtKind::Op(opcode) => {
                        self.emit(Instruction::Primitive(*opcode), statement.location)?;
                    }
                    HirStmtKind::Push(value) => {
                        if !value.prov.is_pure() {
                            return Err(BytecodeError::NonLiteralValue);
                        }
                        self.emit(Instruction::PushWord(value.val), statement.location)?;
                    }
                    HirStmtKind::PushConstant { id, value } => {
                        let instruction =
                            self.emit(Instruction::PushWord(*value), statement.location)?;
                        self.constant_references.push(ConstantReference {
                            instruction,
                            constant: *id,
                        });
                    }
                    HirStmtKind::ReadTemporal { address, .. } => {
                        self.emit(Instruction::PushWord(*address), statement.location)?;
                        self.emit(Instruction::Primitive(OpCode::Oracle), statement.location)?;
                    }
                    HirStmtKind::PushQuote(id) => {
                        self.emit(Instruction::PushQuote(*id), statement.location)?;
                    }
                    HirStmtKind::Call { target } => {
                        let instruction = match target {
                            CallTarget::Procedure(id) => Instruction::CallProcedure(*id),
                            CallTarget::Foreign(id) => Instruction::CallForeign(*id),
                        };
                        self.emit(instruction, statement.location)?;
                    }
                    HirStmtKind::Block(body) => work.push(Work::Block(body)),
                    HirStmtKind::If {
                        then_branch,
                        else_branch,
                    } => {
                        let marker = self.emit(
                            Instruction::IfFalse {
                                else_target: 0,
                                end_target: 0,
                                has_else: else_branch.is_some(),
                            },
                            statement.location,
                        )?;
                        work.push(Work::FinishIf { marker });
                        if let Some(else_branch) = else_branch {
                            work.push(Work::Block(else_branch));
                        }
                        work.push(Work::FinishIfThen {
                            marker,
                            has_else: else_branch.is_some(),
                            location: statement.location,
                        });
                        work.push(Work::Block(then_branch));
                    }
                    HirStmtKind::While { cond, body } => {
                        let loop_start = self.pc()?;
                        work.push(Work::FinishWhileCondition {
                            loop_start,
                            body,
                            location: statement.location,
                        });
                        work.push(Work::Block(cond));
                    }
                    HirStmtKind::TemporalScope {
                        base,
                        size,
                        cell_bits,
                        body,
                    } => {
                        let enter = self.emit(
                            Instruction::TemporalEnter {
                                base: *base,
                                size: *size,
                                cell_bits: *cell_bits,
                                exit_target: 0,
                            },
                            statement.location,
                        )?;
                        work.push(Work::FinishTemporal {
                            enter,
                            location: statement.location,
                        });
                        work.push(Work::Block(body));
                    }
                },
                Work::FinishIfThen {
                    marker,
                    has_else,
                    location,
                } => {
                    let else_target = if has_else {
                        self.emit(Instruction::Jump { target: 0 }, location)?;
                        self.pc()?
                    } else {
                        self.pc()?
                    };
                    let Instruction::IfFalse {
                        else_target: target,
                        ..
                    } = &mut self.instructions[marker as usize]
                    else {
                        return invalid("IF patch marker changed instruction kind".to_string());
                    };
                    *target = else_target;
                }
                Work::FinishIf { marker } => {
                    let end_target = self.pc()?;
                    let Instruction::IfFalse {
                        else_target,
                        end_target: end,
                        has_else,
                    } = &mut self.instructions[marker as usize]
                    else {
                        return invalid(
                            "IF completion marker changed instruction kind".to_string(),
                        );
                    };
                    *end = end_target;
                    if *has_else {
                        let jump_pc = else_target.checked_sub(1).ok_or_else(|| {
                            BytecodeError::InvalidStructure(
                                "compiled IF has impossible else target".to_string(),
                            )
                        })?;
                        let Instruction::Jump { target } = &mut self.instructions[jump_pc as usize]
                        else {
                            return invalid("compiled IF lost its boundary jump".to_string());
                        };
                        *target = end_target;
                    }
                }
                Work::FinishWhileCondition {
                    loop_start,
                    body,
                    location,
                } => {
                    let marker = self.emit(
                        Instruction::WhileFalse {
                            loop_start,
                            end_target: 0,
                        },
                        location,
                    )?;
                    work.push(Work::FinishWhile {
                        marker,
                        loop_start,
                        location,
                    });
                    work.push(Work::Block(body));
                }
                Work::FinishWhile {
                    marker,
                    loop_start,
                    location,
                } => {
                    self.emit(Instruction::LoopBack { target: loop_start }, location)?;
                    let end_target = self.pc()?;
                    let Instruction::WhileFalse {
                        end_target: end, ..
                    } = &mut self.instructions[marker as usize]
                    else {
                        return invalid("WHILE patch marker changed instruction kind".to_string());
                    };
                    *end = end_target;
                }
                Work::FinishTemporal { enter, location } => {
                    let exit = self.emit(
                        Instruction::TemporalExit {
                            enter_target: enter,
                        },
                        location,
                    )?;
                    let Instruction::TemporalEnter { exit_target, .. } =
                        &mut self.instructions[enter as usize]
                    else {
                        return invalid(
                            "temporal patch marker changed instruction kind".to_string(),
                        );
                    };
                    *exit_target = exit;
                }
            }
        }

        self.emit(Instruction::Return, None)?;
        Ok(CodeRange {
            start,
            end: self.pc()?,
        })
    }

    fn pc(&self) -> Result<u32, BytecodeError> {
        check_limit("instruction", self.instructions.len(), MAX_INSTRUCTIONS)?;
        u32_index(self.instructions.len(), "instruction")
    }

    fn emit(
        &mut self,
        instruction: Instruction,
        location: Option<SourceSpan>,
    ) -> Result<u32, BytecodeError> {
        if self.instructions.len() >= MAX_INSTRUCTIONS {
            return Err(BytecodeError::LimitExceeded {
                what: "instruction",
                count: self.instructions.len() + 1,
                limit: MAX_INSTRUCTIONS,
            });
        }
        let pc = self.pc()?;
        self.instructions.push(instruction);
        if let Some(span) = location {
            if self.source_map.len() >= MAX_SOURCE_MAP_ENTRIES {
                return Err(BytecodeError::LimitExceeded {
                    what: "source map",
                    count: self.source_map.len() + 1,
                    limit: MAX_SOURCE_MAP_ENTRIES,
                });
            }
            self.source_map.push(SourceMapEntry {
                instruction: pc,
                span,
            });
        }
        Ok(pc)
    }
}

#[derive(Debug, Clone, Copy)]
struct ControlInterval {
    start: u32,
    end: u32,
    kind: &'static str,
    marker: u32,
}

impl ControlInterval {
    fn new(start: u32, end: u32, kind: &'static str, marker: u32) -> Result<Self, BytecodeError> {
        if start >= end {
            return invalid(format!(
                "{kind} at instruction {marker} has empty/reversed interval {start}..{end}"
            ));
        }
        Ok(Self {
            start,
            end,
            kind,
            marker,
        })
    }
}

fn validate_laminar_intervals(intervals: &mut [ControlInterval]) -> Result<(), BytecodeError> {
    intervals.sort_by_key(|interval| (interval.start, std::cmp::Reverse(interval.end)));
    let mut stack = Vec::<ControlInterval>::new();
    for interval in intervals.iter().copied() {
        while stack
            .last()
            .is_some_and(|outer| interval.start >= outer.end)
        {
            stack.pop();
        }
        if let Some(outer) = stack.last() {
            if interval.start == outer.start && interval.end == outer.end {
                return invalid(format!(
                    "{} interval at {} duplicates {} interval at {}",
                    interval.kind, interval.marker, outer.kind, outer.marker
                ));
            }
            if interval.end > outer.end {
                return invalid(format!(
                    "{} interval {}..{} at {} crosses {} interval {}..{} at {}",
                    interval.kind,
                    interval.start,
                    interval.end,
                    interval.marker,
                    outer.kind,
                    outer.start,
                    outer.end,
                    outer.marker
                ));
            }
        }
        stack.push(interval);
    }
    Ok(())
}

fn require_target(
    range: CodeRange,
    target: u32,
    kind: &str,
    source: u32,
) -> Result<(), BytecodeError> {
    if !range.contains(target) {
        return invalid(format!(
            "instruction {source} has {kind} target {target} outside unit {}..{}",
            range.start, range.end
        ));
    }
    Ok(())
}

fn require_same_depth(
    depths: &[usize],
    range: CodeRange,
    source: u32,
    target: u32,
    kind: &str,
) -> Result<(), BytecodeError> {
    let source_depth = depths[(source - range.start) as usize];
    let target_depth = depths[(target - range.start) as usize];
    if source_depth != target_depth {
        return invalid(format!(
            "instruction {source} has {kind} target {target} across temporal scope depth {source_depth}->{target_depth}"
        ));
    }
    Ok(())
}

fn insert_unique(
    map: &mut HashMap<u32, u32>,
    instruction: u32,
    target: u32,
    kind: &str,
) -> Result<(), BytecodeError> {
    if map.insert(instruction, target).is_some() {
        return invalid(format!(
            "instruction {instruction} is claimed by multiple {kind} records"
        ));
    }
    Ok(())
}

fn check_limit(what: &'static str, count: usize, limit: usize) -> Result<(), BytecodeError> {
    if count > limit {
        Err(BytecodeError::LimitExceeded { what, count, limit })
    } else {
        Ok(())
    }
}

fn count_u32(count: usize, what: &'static str) -> Result<u32, BytecodeError> {
    u32::try_from(count).map_err(|_| BytecodeError::IndexOverflow { what, value: count })
}

fn u32_index(index: usize, what: &'static str) -> Result<u32, BytecodeError> {
    u32::try_from(index).map_err(|_| BytecodeError::IndexOverflow { what, value: index })
}

fn usize_from_u64(what: &'static str, value: u64) -> Result<usize, BytecodeError> {
    usize::try_from(value).map_err(|_| BytecodeError::HostIndexOverflow { what, value })
}

fn invalid<T>(message: String) -> Result<T, BytecodeError> {
    Err(BytecodeError::InvalidStructure(message))
}

const TAG_PRIMITIVE: u8 = 0;
const TAG_PUSH_WORD: u8 = 1;
const TAG_PUSH_QUOTE: u8 = 2;
const TAG_CALL_PROCEDURE: u8 = 3;
const TAG_CALL_FOREIGN: u8 = 4;
const TAG_IF_FALSE: u8 = 5;
const TAG_JUMP: u8 = 6;
const TAG_WHILE_FALSE: u8 = 7;
const TAG_LOOP_BACK: u8 = 8;
const TAG_TEMPORAL_ENTER: u8 = 9;
const TAG_TEMPORAL_EXIT: u8 = 10;
const TAG_RETURN: u8 = 11;

fn encode_instruction(bytes: &mut Vec<u8>, instruction: Instruction) -> Result<(), BytecodeError> {
    match instruction {
        Instruction::Primitive(opcode) => {
            bytes.push(TAG_PRIMITIVE);
            let index = OpCode::ALL
                .iter()
                .position(|candidate| *candidate == opcode)
                .ok_or(BytecodeError::UnregisteredPrimitive(opcode.name()))?;
            put_u16(
                bytes,
                u16::try_from(index).map_err(|_| BytecodeError::IndexOverflow {
                    what: "primitive opcode",
                    value: index,
                })?,
            );
        }
        Instruction::PushWord(value) => {
            bytes.push(TAG_PUSH_WORD);
            put_u64(bytes, value);
        }
        Instruction::PushQuote(id) => {
            bytes.push(TAG_PUSH_QUOTE);
            put_u64(bytes, id.as_u64());
        }
        Instruction::CallProcedure(id) => {
            bytes.push(TAG_CALL_PROCEDURE);
            put_u32(bytes, id.as_u32());
        }
        Instruction::CallForeign(id) => {
            bytes.push(TAG_CALL_FOREIGN);
            put_u32(bytes, id.as_u32());
        }
        Instruction::IfFalse {
            else_target,
            end_target,
            has_else,
        } => {
            bytes.push(TAG_IF_FALSE);
            put_u32(bytes, else_target);
            put_u32(bytes, end_target);
            bytes.push(u8::from(has_else));
        }
        Instruction::Jump { target } => {
            bytes.push(TAG_JUMP);
            put_u32(bytes, target);
        }
        Instruction::WhileFalse {
            loop_start,
            end_target,
        } => {
            bytes.push(TAG_WHILE_FALSE);
            put_u32(bytes, loop_start);
            put_u32(bytes, end_target);
        }
        Instruction::LoopBack { target } => {
            bytes.push(TAG_LOOP_BACK);
            put_u32(bytes, target);
        }
        Instruction::TemporalEnter {
            base,
            size,
            cell_bits,
            exit_target,
        } => {
            bytes.push(TAG_TEMPORAL_ENTER);
            put_u64(bytes, base);
            put_u64(bytes, size);
            bytes.push(cell_bits);
            put_u32(bytes, exit_target);
        }
        Instruction::TemporalExit { enter_target } => {
            bytes.push(TAG_TEMPORAL_EXIT);
            put_u32(bytes, enter_target);
        }
        Instruction::Return => bytes.push(TAG_RETURN),
    }
    Ok(())
}

fn decode_instruction(reader: &mut Reader<'_>) -> Result<Instruction, BytecodeError> {
    match reader.u8()? {
        TAG_PRIMITIVE => {
            let raw = reader.u16()?;
            let opcode = OpCode::ALL
                .get(raw as usize)
                .copied()
                .ok_or(BytecodeError::UnknownPrimitive(raw))?;
            Ok(Instruction::Primitive(opcode))
        }
        TAG_PUSH_WORD => Ok(Instruction::PushWord(reader.u64()?)),
        TAG_PUSH_QUOTE => Ok(Instruction::PushQuote(QuoteId::new(reader.u64()?))),
        TAG_CALL_PROCEDURE => {
            let raw = reader.u32()?;
            let id = ProcedureId::try_from_index(raw as usize).ok_or(
                BytecodeError::HostIndexOverflow {
                    what: "procedure call identity",
                    value: u64::from(raw),
                },
            )?;
            Ok(Instruction::CallProcedure(id))
        }
        TAG_CALL_FOREIGN => {
            let raw = reader.u32()?;
            let id = ForeignId::try_from_index(raw as usize).ok_or(
                BytecodeError::HostIndexOverflow {
                    what: "foreign call identity",
                    value: u64::from(raw),
                },
            )?;
            Ok(Instruction::CallForeign(id))
        }
        TAG_IF_FALSE => {
            let else_target = reader.u32()?;
            let end_target = reader.u32()?;
            let has_else = match reader.u8()? {
                0 => false,
                1 => true,
                value => return Err(BytecodeError::InvalidBoolean(value)),
            };
            Ok(Instruction::IfFalse {
                else_target,
                end_target,
                has_else,
            })
        }
        TAG_JUMP => Ok(Instruction::Jump {
            target: reader.u32()?,
        }),
        TAG_WHILE_FALSE => Ok(Instruction::WhileFalse {
            loop_start: reader.u32()?,
            end_target: reader.u32()?,
        }),
        TAG_LOOP_BACK => Ok(Instruction::LoopBack {
            target: reader.u32()?,
        }),
        TAG_TEMPORAL_ENTER => Ok(Instruction::TemporalEnter {
            base: reader.u64()?,
            size: reader.u64()?,
            cell_bits: reader.u8()?,
            exit_target: reader.u32()?,
        }),
        TAG_TEMPORAL_EXIT => Ok(Instruction::TemporalExit {
            enter_target: reader.u32()?,
        }),
        TAG_RETURN => Ok(Instruction::Return),
        tag => Err(BytecodeError::UnknownInstructionTag(tag)),
    }
}

fn encode_foreign_entry(bytes: &mut Vec<u8>, entry: &ForeignEntry) -> Result<(), BytecodeError> {
    put_u32(bytes, entry.id.as_u32());
    put_string(bytes, &entry.library, "foreign library")?;
    put_string(bytes, &entry.symbol, "foreign symbol")?;
    put_u32(
        bytes,
        count_u32(entry.parameters.len(), "foreign parameter")?,
    );
    for parameter in &entry.parameters {
        bytes.push(encode_foreign_scalar(*parameter));
    }
    match entry.result {
        None => bytes.push(0),
        Some(result) => {
            bytes.push(1);
            bytes.push(encode_foreign_scalar(result));
        }
    }
    bytes.push(entry.effects.bits());
    Ok(())
}

fn decode_foreign_entry(
    reader: &mut Reader<'_>,
    index: usize,
) -> Result<ForeignEntry, BytecodeError> {
    let raw_id = reader.u32()?;
    let id =
        ForeignId::try_from_index(raw_id as usize).ok_or(BytecodeError::HostIndexOverflow {
            what: "foreign identity",
            value: u64::from(raw_id),
        })?;
    let library = reader.string("foreign library", MAX_FOREIGN_NAME_BYTES)?;
    let symbol = reader.string("foreign symbol", MAX_FOREIGN_NAME_BYTES)?;
    let parameter_count = reader.bounded_count("foreign parameter", MAX_FOREIGN_PARAMETERS)?;
    let mut parameters = Vec::with_capacity(parameter_count);
    for _ in 0..parameter_count {
        parameters.push(decode_foreign_scalar(reader.u8()?)?);
    }
    let result = match reader.u8()? {
        0 => None,
        1 => Some(decode_foreign_scalar(reader.u8()?)?),
        value => return Err(BytecodeError::InvalidBoolean(value)),
    };
    let effect_bits = reader.u8()?;
    let effects = ForeignEffects::from_bits(effect_bits).ok_or_else(|| {
        BytecodeError::InvalidStructure(format!(
            "foreign entry {index} has invalid effect bits 0x{effect_bits:02x}"
        ))
    })?;
    let entry = ForeignEntry {
        id,
        library,
        symbol,
        parameters,
        result,
        effects,
    };
    validate_foreign_entry(&entry)?;
    Ok(entry)
}

const fn encode_foreign_scalar(typ: ForeignScalarType) -> u8 {
    match typ {
        ForeignScalarType::U64 => 0,
        ForeignScalarType::I64 => 1,
    }
}

fn decode_foreign_scalar(tag: u8) -> Result<ForeignScalarType, BytecodeError> {
    match tag {
        0 => Ok(ForeignScalarType::U64),
        1 => Ok(ForeignScalarType::I64),
        value => Err(BytecodeError::InvalidStructure(format!(
            "unknown foreign scalar type tag {value}"
        ))),
    }
}

fn put_string(bytes: &mut Vec<u8>, value: &str, what: &'static str) -> Result<(), BytecodeError> {
    check_limit(what, value.len(), MAX_FOREIGN_NAME_BYTES)?;
    put_u32(bytes, count_u32(value.len(), what)?);
    bytes.extend_from_slice(value.as_bytes());
    Ok(())
}

fn put_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_range(bytes: &mut Vec<u8>, range: CodeRange) {
    put_u32(bytes, range.start);
    put_u32(bytes, range.end);
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], BytecodeError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or(BytecodeError::Truncated {
                offset: self.offset,
                needed: count,
            })?;
        let result = self
            .bytes
            .get(self.offset..end)
            .ok_or(BytecodeError::Truncated {
                offset: self.offset,
                needed: count,
            })?;
        self.offset = end;
        Ok(result)
    }

    fn u8(&mut self) -> Result<u8, BytecodeError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, BytecodeError> {
        let bytes: [u8; 2] = self.take(2)?.try_into().expect("exact reader width");
        Ok(u16::from_le_bytes(bytes))
    }

    fn u32(&mut self) -> Result<u32, BytecodeError> {
        let bytes: [u8; 4] = self.take(4)?.try_into().expect("exact reader width");
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, BytecodeError> {
        let bytes: [u8; 8] = self.take(8)?.try_into().expect("exact reader width");
        Ok(u64::from_le_bytes(bytes))
    }

    fn bounded_count(&mut self, what: &'static str, limit: usize) -> Result<usize, BytecodeError> {
        let count = self.u32()? as usize;
        check_limit(what, count, limit)?;
        Ok(count)
    }

    fn range(&mut self) -> Result<CodeRange, BytecodeError> {
        Ok(CodeRange {
            start: self.u32()?,
            end: self.u32()?,
        })
    }

    fn string(&mut self, what: &'static str, limit: usize) -> Result<String, BytecodeError> {
        let count = self.bounded_count(what, limit)?;
        let raw = self.take(count)?;
        std::str::from_utf8(raw)
            .map(str::to_owned)
            .map_err(|_| BytecodeError::InvalidStructure(format!("{what} is not valid UTF-8")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Procedure, Program, Stmt};
    use crate::core::{Provenance, Value};
    use crate::hir::{HirForeign, HirProcedure, HirQuote};
    use crate::parser::FFIDeclaration;
    use crate::runtime::ffi::{FFIEffect, FFISignature, FFIType};

    fn procedure(name: &str, body: Vec<Stmt>) -> Procedure {
        Procedure {
            name: name.to_string(),
            params: Vec::new(),
            returns: 0,
            effects: Vec::new(),
            body,
        }
    }

    fn foreign(name: &str) -> FFIDeclaration {
        FFIDeclaration {
            signature: FFISignature::new(name, "test").effects(vec![FFIEffect::Pure]),
            symbol_name: Some(format!("native_{name}")),
        }
    }

    fn resolved(program: &Program) -> HirProgram {
        HirProgram::resolve(program).expect("test program resolves")
    }

    #[test]
    fn round_trip_is_byte_stable_and_deterministic() {
        let mut program = Program::new();
        program.quotes.push(vec![Stmt::Op(OpCode::Dup)]);
        program.procedures.push(procedure(
            "worker",
            vec![Stmt::PushQuote(QuoteId::new(0)), Stmt::Op(OpCode::Exec)],
        ));
        program.ffi_declarations.push(foreign("native"));
        program.body = vec![
            Stmt::Push(Value::new(7)),
            Stmt::If {
                then_branch: vec![Stmt::Call {
                    name: "worker".to_string(),
                }],
                else_branch: Some(vec![Stmt::Call {
                    name: "native".to_string(),
                }]),
            },
        ];
        let hir = resolved(&program);

        let first = BytecodeProgram::compile(&hir).expect("compile");
        let second = BytecodeProgram::compile(&hir).expect("compile again");
        assert_eq!(first, second);
        let bytes = first.to_bytes().expect("encode");
        let decoded = BytecodeProgram::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded, first);
        assert_eq!(decoded.to_bytes().expect("re-encode"), bytes);
        assert_eq!(
            decoded.validate().expect("validate").stack_semantics,
            StackSemanticStatus::NotChecked
        );
        assert_eq!(decoded.foreigns[0].library, "test");
        assert_eq!(decoded.foreigns[0].symbol, "native_native");
        assert!(decoded.foreigns[0].effects.is_pure());
    }

    #[test]
    fn accepts_only_the_bounded_u64_i64_scalar_foreign_abi() {
        let mut accepted = Program::new();
        accepted.ffi_declarations.push(FFIDeclaration {
            signature: FFISignature::new("mix", "process")
                .param("unsigned", FFIType::U64)
                .param("signed", FFIType::I64)
                .returns_type(FFIType::I64)
                .effects(vec![FFIEffect::IO]),
            symbol_name: Some("host_mix".into()),
        });
        let artifact = BytecodeProgram::compile(&resolved(&accepted)).unwrap();
        assert_eq!(
            artifact.foreigns[0].parameters,
            [ForeignScalarType::U64, ForeignScalarType::I64]
        );
        assert_eq!(artifact.foreigns[0].result, Some(ForeignScalarType::I64));

        for unsupported in [FFIType::U32, FFIType::Bool, FFIType::Ptr, FFIType::Str] {
            let mut rejected = Program::new();
            rejected.ffi_declarations.push(FFIDeclaration {
                signature: FFISignature::new("bad", "process")
                    .param("value", unsupported)
                    .returns_type(FFIType::U64)
                    .pure(),
                symbol_name: None,
            });
            assert!(matches!(
                BytecodeProgram::compile(&resolved(&rejected)),
                Err(BytecodeError::UnsupportedForeignSignature { .. })
            ));
        }

        let mut too_many_results = Program::new();
        too_many_results.ffi_declarations.push(FFIDeclaration {
            signature: FFISignature::new("many", "process")
                .returns_type(FFIType::U64)
                .returns_type(FFIType::I64)
                .pure(),
            symbol_name: None,
        });
        assert!(matches!(
            BytecodeProgram::compile(&resolved(&too_many_results)),
            Err(BytecodeError::UnsupportedForeignSignature { .. })
        ));
    }

    #[test]
    fn lowers_structured_control_flow_to_checked_targets() {
        let mut program = Program::new();
        program.body = vec![Stmt::While {
            cond: vec![Stmt::Push(Value::ONE)],
            body: vec![Stmt::If {
                then_branch: vec![Stmt::Op(OpCode::Nop)],
                else_branch: Some(vec![Stmt::Op(OpCode::Pop)]),
            }],
        }];
        let bytecode = BytecodeProgram::compile(&resolved(&program)).expect("compile");
        bytecode.validate().expect("valid control flow");

        assert!(bytecode
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::WhileFalse { .. })));
        assert!(bytecode
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::IfFalse { .. })));

        let mut corrupt = bytecode.clone();
        let Instruction::WhileFalse { loop_start, .. } = &mut corrupt.instructions[1] else {
            panic!("expected WHILE marker");
        };
        *loop_start = corrupt.main.end;
        assert!(corrupt.validate().is_err());
    }

    #[test]
    fn keeps_word_quote_procedure_and_foreign_operands_typed() {
        let mut program = Program::new();
        program.quotes.push(Vec::new());
        program.procedures.push(procedure("local", Vec::new()));
        program.ffi_declarations.push(foreign("native"));
        program.body = vec![
            Stmt::Push(Value::new(0)),
            Stmt::PushQuote(QuoteId::new(0)),
            Stmt::Call {
                name: "local".to_string(),
            },
            Stmt::Call {
                name: "native".to_string(),
            },
        ];
        let bytecode = BytecodeProgram::compile(&resolved(&program)).expect("compile");
        assert!(matches!(bytecode.instructions[0], Instruction::PushWord(0)));
        assert!(matches!(
            bytecode.instructions[1],
            Instruction::PushQuote(id) if id == QuoteId::new(0)
        ));
        assert!(matches!(
            bytecode.instructions[2],
            Instruction::CallProcedure(id) if id.index() == 0
        ));
        assert!(matches!(
            bytecode.instructions[3],
            Instruction::CallForeign(id) if id.index() == 0
        ));
        assert_eq!(bytecode.quotations[0].id, QuoteId::new(0));
        assert_eq!(bytecode.procedures[0].id.index(), 0);
    }

    #[test]
    fn emits_and_validates_explicit_temporal_scope_pairs() {
        let mut program = Program::new();
        program.body.push(Stmt::TemporalScope {
            base: 4,
            size: 3,
            cell_bits: 12,
            body: vec![Stmt::Op(OpCode::Oracle)],
        });
        let bytecode = BytecodeProgram::compile(&resolved(&program)).expect("compile");
        assert!(matches!(
            bytecode.instructions[0],
            Instruction::TemporalEnter {
                base: 4,
                size: 3,
                cell_bits: 12,
                exit_target: 2
            }
        ));
        assert_eq!(
            bytecode.instructions[2],
            Instruction::TemporalExit { enter_target: 0 }
        );

        let mut corrupt = bytecode;
        corrupt.instructions[2] = Instruction::TemporalExit { enter_target: 1 };
        assert!(corrupt.validate().is_err());
    }

    #[test]
    fn retains_source_locations_in_a_parallel_map() {
        let span = SourceSpan::new(SourceId::new(3), TextRange::new(5, 11));
        let hir = HirProgram {
            constants: Vec::new(),
            temporals: Vec::new(),
            procedures: Vec::new(),
            quotes: Vec::new(),
            body: vec![HirStmt {
                kind: HirStmtKind::Push(Value::new(1)),
                location: Some(span),
            }],
            foreigns: Vec::new(),
            family_declaration: None,
            markov_declaration: None,
            quantum_declaration: None,
            properties: Vec::new(),
        };
        let bytecode = BytecodeProgram::compile(&hir).expect("compile");
        assert_eq!(
            bytecode.source_map,
            vec![SourceMapEntry {
                instruction: 0,
                span
            }]
        );
        let decoded = BytecodeProgram::from_bytes(&bytecode.to_bytes().unwrap()).unwrap();
        assert_eq!(decoded.source_map, bytecode.source_map);
    }

    #[test]
    fn rejects_runtime_provenance_in_a_literal() {
        let hir = HirProgram {
            constants: Vec::new(),
            temporals: Vec::new(),
            procedures: Vec::<HirProcedure>::new(),
            quotes: Vec::<HirQuote>::new(),
            body: vec![HirStmt {
                kind: HirStmtKind::Push(Value::with_provenance(1, Provenance::single(0))),
                location: None,
            }],
            foreigns: Vec::<HirForeign>::new(),
            family_declaration: None,
            markov_declaration: None,
            quantum_declaration: None,
            properties: Vec::new(),
        };
        assert_eq!(
            BytecodeProgram::compile(&hir).unwrap_err(),
            BytecodeError::NonLiteralValue
        );
    }

    #[test]
    fn malformed_artifact_corpus_is_rejected() {
        let bytecode = BytecodeProgram::compile(&resolved(&Program::new())).expect("compile");
        let good = bytecode.to_bytes().expect("encode");

        for length in 0..good.len() {
            assert!(
                BytecodeProgram::from_bytes(&good[..length]).is_err(),
                "accepted truncation at {length}"
            );
        }

        let mut trailing = good.clone();
        trailing.push(0);
        assert!(matches!(
            BytecodeProgram::from_bytes(&trailing),
            Err(BytecodeError::TrailingBytes(1))
        ));

        let mut bad_magic = good.clone();
        bad_magic[0] ^= 0xff;
        assert_eq!(
            BytecodeProgram::from_bytes(&bad_magic).unwrap_err(),
            BytecodeError::BadMagic
        );

        let mut bad_version = good.clone();
        let unsupported_version = FORMAT_VERSION + 1;
        bad_version[8..10].copy_from_slice(&unsupported_version.to_le_bytes());
        assert_eq!(
            BytecodeProgram::from_bytes(&bad_version).unwrap_err(),
            BytecodeError::UnsupportedVersion(unsupported_version)
        );

        let mut bad_flags = good.clone();
        bad_flags[10..12].copy_from_slice(&1u16.to_le_bytes());
        assert_eq!(
            BytecodeProgram::from_bytes(&bad_flags).unwrap_err(),
            BytecodeError::UnsupportedFlags(1)
        );

        let mut excessive_count = good.clone();
        excessive_count[12..16].copy_from_slice(&(MAX_INSTRUCTIONS as u32 + 1).to_le_bytes());
        assert!(matches!(
            BytecodeProgram::from_bytes(&excessive_count),
            Err(BytecodeError::LimitExceeded {
                what: "instruction",
                ..
            })
        ));

        // Header is 40 bytes: the empty main unit's only instruction follows.
        let mut unknown_tag = good.clone();
        unknown_tag[40] = 0xff;
        assert_eq!(
            BytecodeProgram::from_bytes(&unknown_tag).unwrap_err(),
            BytecodeError::UnknownInstructionTag(0xff)
        );

        let primitive = BytecodeProgram {
            instructions: vec![Instruction::Primitive(OpCode::Nop), Instruction::Return],
            main: CodeRange { start: 0, end: 2 },
            quotations: Vec::new(),
            procedures: Vec::new(),
            foreigns: Vec::new(),
            source_map: Vec::new(),
        }
        .to_bytes()
        .unwrap();
        let mut unknown_primitive = primitive;
        unknown_primitive[41..43].copy_from_slice(&u16::MAX.to_le_bytes());
        assert_eq!(
            BytecodeProgram::from_bytes(&unknown_primitive).unwrap_err(),
            BytecodeError::UnknownPrimitive(u16::MAX)
        );

        let mut if_program = Program::new();
        if_program.body.push(Stmt::If {
            then_branch: Vec::new(),
            else_branch: None,
        });
        let mut bad_boolean = BytecodeProgram::compile(&resolved(&if_program))
            .unwrap()
            .to_bytes()
            .unwrap();
        // IF starts immediately after the 40-byte header; its bool is byte 49.
        bad_boolean[49] = 2;
        assert_eq!(
            BytecodeProgram::from_bytes(&bad_boolean).unwrap_err(),
            BytecodeError::InvalidBoolean(2)
        );
    }

    #[test]
    fn validator_rejects_bad_ranges_calls_and_termination() {
        let valid = BytecodeProgram::compile(&resolved(&Program::new())).expect("compile");

        let mut bad_range = valid.clone();
        bad_range.main.start = 1;
        assert!(bad_range.validate().is_err());

        let mut bad_return = valid.clone();
        bad_return.instructions[0] = Instruction::Primitive(OpCode::Nop);
        assert!(bad_return.validate().is_err());

        let mut bad_call = valid;
        bad_call.instructions.insert(
            0,
            Instruction::CallProcedure(ProcedureId::try_from_index(0).unwrap()),
        );
        bad_call.main.end += 1;
        assert!(bad_call.validate().is_err());
    }

    #[test]
    fn opcode_all_has_unique_deterministic_encoding() {
        let unique: std::collections::HashSet<_> = OpCode::ALL.iter().copied().collect();
        assert_eq!(unique.len(), OpCode::ALL.len());
        for (index, opcode) in OpCode::ALL.iter().copied().enumerate() {
            let mut bytes = Vec::new();
            encode_instruction(&mut bytes, Instruction::Primitive(opcode)).unwrap();
            assert_eq!(bytes[0], TAG_PRIMITIVE);
            assert_eq!(u16::from_le_bytes([bytes[1], bytes[2]]) as usize, index);
            let mut reader = Reader::new(&bytes);
            assert_eq!(
                decode_instruction(&mut reader).unwrap(),
                Instruction::Primitive(opcode)
            );
        }
    }
}
