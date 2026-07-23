//! Deterministic, resolved high-level intermediate representation.
//!
//! The parser's [`Program`] still names callees with strings.  This module is
//! the first representation that is safe for later compiler stages: callable
//! references are typed table indices, quotation references have been checked,
//! and every declaration retains a stable source-order identity.

use crate::ast::{
    Effect, MarkovDeclaration, OpCode, Program, ProgramLocations, PropertyPredicate,
    PspaceFamilyDeclaration, QuantumChannelDeclaration, QuoteId, Stmt, StmtLocations,
    TemporalPropertyDeclaration,
};
use crate::core::Value;
use crate::parser::FFIDeclaration;
use crate::source::SourceSpan;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

macro_rules! define_u32_id {
    ($name:ident, $description:literal) => {
        #[doc = $description]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(u32);

        impl $name {
            /// Constructs an ID from a table index, rejecting indices that do
            /// not fit in the representation used by compiler artifacts.
            pub fn try_from_index(index: usize) -> Option<Self> {
                u32::try_from(index).ok().map(Self)
            }

            /// Returns the zero-based table index.
            pub const fn index(self) -> usize {
                self.0 as usize
            }

            /// Returns the serialized integer representation.
            pub const fn as_u32(self) -> u32 {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "{}", self.0)
            }
        }
    };
}

define_u32_id!(
    ProcedureId,
    "Stable index of a procedure in [`HirProgram::procedures`]."
);
define_u32_id!(
    ForeignId,
    "Stable index of a foreign declaration in [`HirProgram::foreigns`]."
);
define_u32_id!(
    PropertyId,
    "Stable index of a property in [`HirProgram::properties`]."
);
define_u32_id!(
    ConstantId,
    "Stable index of a MANIFEST word in [`HirProgram::constants`]."
);
define_u32_id!(
    TemporalId,
    "Stable index of a named temporal cell in [`HirProgram::temporals`]."
);

/// A callable selected by name resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallTarget {
    /// An Ourochronos procedure.
    Procedure(ProcedureId),
    /// A declared foreign function.
    Foreign(ForeignId),
}

impl CallTarget {
    fn kind(self) -> CallableKind {
        match self {
            Self::Procedure(_) => CallableKind::Procedure,
            Self::Foreign(_) => CallableKind::Foreign,
        }
    }
}

/// Kind of a declaration occupying the callable namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallableKind {
    /// A language procedure.
    Procedure,
    /// A foreign function declaration.
    Foreign,
}

/// Compiler table whose index could not be represented by its typed ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexKind {
    /// MANIFEST table.
    Constant,
    /// Named temporal-cell table.
    Temporal,
    /// Procedure table.
    Procedure,
    /// Foreign declaration table.
    Foreign,
    /// Property table.
    Property,
    /// Quotation table.
    Quotation,
}

/// Statement-list owner used to retain diagnostic context before source spans
/// are attached to individual AST statements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HirOwner {
    /// Top-level program body.
    ProgramBody,
    /// A procedure body.
    Procedure(ProcedureId),
    /// An anonymous quotation body.
    Quote(QuoteId),
}

/// Specific failure reported while resolving an AST program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HirErrorKind {
    /// Two declarations occupy the same case-insensitive callable name.
    DuplicateCallable {
        /// Normalized name that collided.
        name: String,
        /// Kind of the declaration encountered first.
        first: CallableKind,
        /// Kind of the later declaration.
        second: CallableKind,
    },
    /// Two properties have the same case-insensitive name.
    DuplicateProperty {
        /// Normalized property name.
        name: String,
    },
    /// Two retained MANIFEST declarations use the same name.
    DuplicateConstant { name: String },
    /// Two retained temporal declarations use the same name.
    DuplicateTemporal { name: String },
    /// A temporal cell and MANIFEST occupy the same value-symbol spelling.
    TemporalConstantCollision { name: String },
    /// Two retained temporal declarations use the same absolute address.
    DuplicateTemporalAddress {
        address: u64,
        first: String,
        second: String,
    },
    /// A named temporal-cell reference has no retained declaration.
    UnknownTemporal { name: String },
    /// Parser-retained reference and declaration addresses disagree.
    TemporalAddressMismatch {
        name: String,
        declared: u64,
        used: u64,
    },
    /// A symbolic constant push has no retained declaration.
    UnknownConstant { name: String },
    /// Parser-retained use and declaration values disagree.
    ConstantValueMismatch {
        name: String,
        declared: u64,
        used: u64,
    },
    /// A call has no procedure or foreign declaration.
    UnknownCall {
        /// Source spelling of the unresolved name.
        name: String,
    },
    /// A pushed quotation identity falls outside the quotation table.
    InvalidQuote {
        /// Invalid quotation identity.
        quote: QuoteId,
        /// Number of quotations available in the program.
        quote_count: usize,
    },
    /// A declaration table is too large for its artifact ID representation.
    IndexOverflow {
        /// Table whose index overflowed.
        kind: IndexKind,
        /// First unrepresentable index.
        index: usize,
    },
    /// An internal lowering invariant was violated after successful validation.
    InternalInvariant {
        /// Description suitable for a compiler-bug diagnostic.
        message: String,
    },
}

/// Resolution error with extensible source and ownership context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HirError {
    /// Specific error category and data.
    pub kind: HirErrorKind,
    /// Statement-list owner, when the error arose in executable code.
    pub owner: Option<HirOwner>,
    /// Exact source span once located AST statements are wired into resolution.
    pub location: Option<SourceSpan>,
}

impl HirError {
    fn declaration(kind: HirErrorKind) -> Self {
        Self {
            kind,
            owner: None,
            location: None,
        }
    }

    fn declaration_at(kind: HirErrorKind, location: Option<SourceSpan>) -> Self {
        Self {
            kind,
            owner: None,
            location,
        }
    }

    fn statement_at(kind: HirErrorKind, owner: HirOwner, location: Option<SourceSpan>) -> Self {
        Self {
            kind,
            owner: Some(owner),
            location,
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self::declaration(HirErrorKind::InternalInvariant {
            message: message.into(),
        })
    }
}

impl fmt::Display for HirError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            HirErrorKind::DuplicateCallable {
                name,
                first,
                second,
            } => write!(
                formatter,
                "duplicate callable {name:?}: {second:?} conflicts with earlier {first:?}"
            ),
            HirErrorKind::DuplicateProperty { name } => {
                write!(formatter, "duplicate property {name:?}")
            }
            HirErrorKind::DuplicateConstant { name } => {
                write!(formatter, "duplicate constant {name:?}")
            }
            HirErrorKind::DuplicateTemporal { name } => {
                write!(formatter, "duplicate temporal cell {name:?}")
            }
            HirErrorKind::TemporalConstantCollision { name } => write!(
                formatter,
                "temporal cell {name:?} conflicts with a retained MANIFEST"
            ),
            HirErrorKind::DuplicateTemporalAddress {
                address,
                first,
                second,
            } => write!(
                formatter,
                "temporal cell {second:?} reuses address {address} declared by {first:?}"
            ),
            HirErrorKind::UnknownTemporal { name } => {
                write!(formatter, "unknown temporal cell {name:?}")
            }
            HirErrorKind::TemporalAddressMismatch {
                name,
                declared,
                used,
            } => write!(
                formatter,
                "temporal cell {name:?} retains address {used}, but its declaration is {declared}"
            ),
            HirErrorKind::UnknownConstant { name } => {
                write!(formatter, "unknown constant {name:?}")
            }
            HirErrorKind::ConstantValueMismatch {
                name,
                declared,
                used,
            } => write!(
                formatter,
                "constant {name:?} retains value {used}, but its declaration is {declared}"
            ),
            HirErrorKind::UnknownCall { name } => {
                write!(formatter, "unknown callable {name:?}")
            }
            HirErrorKind::InvalidQuote { quote, quote_count } => write!(
                formatter,
                "quotation {} is outside the table of {quote_count} quotations",
                quote.as_u64()
            ),
            HirErrorKind::IndexOverflow { kind, index } => {
                write!(
                    formatter,
                    "{kind:?} index {index} exceeds the u32 artifact limit"
                )
            }
            HirErrorKind::InternalInvariant { message } => {
                write!(formatter, "HIR lowering invariant failed: {message}")
            }
        }?;
        if let Some(location) = self.location {
            write!(
                formatter,
                " at source {} bytes {}..{}",
                location.source.index(),
                location.range.start,
                location.range.end
            )?;
        }
        Ok(())
    }
}

impl Error for HirError {}

/// A resolved statement and its future source location.
#[derive(Debug, Clone, PartialEq)]
pub struct HirStmt {
    /// Statement semantics.
    pub kind: HirStmtKind,
    /// Exact source location once the located AST migration reaches HIR.
    pub location: Option<SourceSpan>,
}

impl HirStmt {
    fn unlocated(kind: HirStmtKind) -> Self {
        Self {
            kind,
            location: None,
        }
    }
}

/// Resolved equivalent of every [`Stmt`] variant.
#[derive(Debug, Clone, PartialEq)]
pub enum HirStmtKind {
    /// Execute a primitive opcode.
    Op(OpCode),
    /// Push an ordinary machine word.
    Push(Value),
    /// Push a resolved MANIFEST word while retaining link identity.
    PushConstant { id: ConstantId, value: u64 },
    /// Read a resolved named temporal cell.
    ReadTemporal { id: TemporalId, address: u64 },
    /// Push a validated quotation identity.
    PushQuote(QuoteId),
    /// Conditional structured control flow.
    If {
        /// Statements selected by a true condition.
        then_branch: Vec<HirStmt>,
        /// Optional false branch.
        else_branch: Option<Vec<HirStmt>>,
    },
    /// Structured loop.
    While {
        /// Condition-producing statements.
        cond: Vec<HirStmt>,
        /// Loop body.
        body: Vec<HirStmt>,
    },
    /// Scoped statement grouping.
    Block(Vec<HirStmt>),
    /// Call a resolved language or foreign declaration.
    Call {
        /// Typed declaration reference.
        target: CallTarget,
    },
    /// Isolated temporal memory scope.
    TemporalScope {
        /// Base address of the temporal region.
        base: u64,
        /// Number of cells in the temporal region.
        size: u64,
        /// Stored word width.
        cell_bits: u8,
        /// Statements executed within the region.
        body: Vec<HirStmt>,
    },
}

/// Procedure declaration with a stable typed identity.
#[derive(Debug, Clone, PartialEq)]
pub struct HirProcedure {
    /// Source-order identity.
    pub id: ProcedureId,
    /// Source spelling of the declaration name.
    pub name: String,
    /// Stack parameter documentation.
    pub params: Vec<String>,
    /// Declared number of results.
    pub returns: usize,
    /// Declared effects.
    pub effects: Vec<Effect>,
    /// Resolved procedure body.
    pub body: Vec<HirStmt>,
}

/// Quotation declaration with its checked quotation-table identity.
#[derive(Debug, Clone, PartialEq)]
pub struct HirQuote {
    /// Source-order quotation identity.
    pub id: QuoteId,
    /// Resolved quotation body.
    pub body: Vec<HirStmt>,
}

/// Foreign declaration with a stable typed identity.
#[derive(Debug, Clone)]
pub struct HirForeign {
    /// Source-order identity.
    pub id: ForeignId,
    /// Complete parsed foreign declaration, including signature and symbol.
    pub declaration: FFIDeclaration,
}

/// Temporal property declaration with a stable typed identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HirProperty {
    /// Source-order identity.
    pub id: PropertyId,
    /// Complete source declaration.
    pub declaration: TemporalPropertyDeclaration,
    /// Typed named-cell occurrences in predicate traversal order.
    pub cells: Vec<HirPropertyCell>,
}

/// Retained MANIFEST declaration with stable typed identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HirConstant {
    pub id: ConstantId,
    pub name: String,
    pub value: u64,
}

/// Retained named temporal declaration with stable typed identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HirTemporal {
    pub id: TemporalId,
    pub name: String,
    pub address: u64,
    pub default: u64,
    /// Exact declaration location for file-backed source.
    pub location: Option<SourceSpan>,
}

/// A typed predicate cell. Numeric `CELL` operands have no declaration ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HirPropertyCell {
    pub temporal: Option<TemporalId>,
    pub address: u64,
}

/// Fully resolved high-level program consumed by later checking and lowering.
#[derive(Debug, Clone)]
pub struct HirProgram {
    /// MANIFEST declarations in deterministic source/link order.
    pub constants: Vec<HirConstant>,
    /// Named temporal cells in deterministic source/link order.
    pub temporals: Vec<HirTemporal>,
    /// Procedures in deterministic source/link order.
    pub procedures: Vec<HirProcedure>,
    /// Anonymous quotations in deterministic table order.
    pub quotes: Vec<HirQuote>,
    /// Resolved top-level body.
    pub body: Vec<HirStmt>,
    /// Foreign declarations in deterministic source/link order.
    pub foreigns: Vec<HirForeign>,
    /// Optional PSPACE family contract.
    pub family_declaration: Option<PspaceFamilyDeclaration>,
    /// Optional finite stochastic CTC declaration.
    pub markov_declaration: Option<MarkovDeclaration>,
    /// Optional source-level quantum channel.
    pub quantum_declaration: Option<QuantumChannelDeclaration>,
    /// All-fixed-state properties in deterministic source/link order.
    pub properties: Vec<HirProperty>,
}

#[derive(Debug, Clone, Copy)]
struct CallableSymbol {
    target: CallTarget,
}

struct ResolutionSymbols<'a> {
    callables: &'a HashMap<String, CallableSymbol>,
    constants: &'a HashMap<String, (ConstantId, u64)>,
    temporals: &'a HashMap<String, (TemporalId, u64)>,
}

impl HirProgram {
    /// Resolves a linked AST program into typed HIR.
    ///
    /// IDs are assigned strictly by declaration-vector order.  Name lookup is
    /// case-insensitive, while declaration spellings are retained.  Statement
    /// validation and tree reconstruction use explicit work stacks so deeply
    /// nested source cannot consume the Rust call stack.
    pub fn resolve(program: &Program) -> Result<Self, Vec<HirError>> {
        Self::resolve_internal(program, None)
    }

    /// Resolve a legacy AST together with the exact sidecar produced by
    /// file-backed parsing.
    pub fn resolve_located(
        program: &Program,
        locations: &ProgramLocations,
    ) -> Result<Self, Vec<HirError>> {
        if !locations.matches(program) {
            return Err(vec![HirError::internal(
                "source-location sidecar does not match the AST",
            )]);
        }
        Self::resolve_internal(program, Some(locations))
    }

    fn resolve_internal(
        program: &Program,
        locations: Option<&ProgramLocations>,
    ) -> Result<Self, Vec<HirError>> {
        let mut errors = Vec::new();
        let mut index_overflowed = false;
        let mut callables = HashMap::<String, CallableSymbol>::new();
        let mut constants = HashMap::<String, (ConstantId, u64)>::new();
        let mut temporals = HashMap::<String, (TemporalId, u64)>::new();

        let mut constant_ids = Vec::with_capacity(program.manifests.len());
        for (index, declaration) in program.manifests.iter().enumerate() {
            let Some(id) = ConstantId::try_from_index(index) else {
                errors.push(HirError::declaration(HirErrorKind::IndexOverflow {
                    kind: IndexKind::Constant,
                    index,
                }));
                index_overflowed = true;
                continue;
            };
            constant_ids.push(id);
            let name = normalize_name(&declaration.name);
            if constants
                .insert(name.clone(), (id, declaration.value))
                .is_some()
            {
                errors.push(HirError::declaration_at(
                    HirErrorKind::DuplicateConstant { name },
                    locations.and_then(|locations| locations.manifests[index]),
                ));
            }
        }

        let mut temporal_ids = Vec::with_capacity(program.temporal_declarations.len());
        let mut temporal_addresses = HashMap::<u64, String>::new();
        for (index, declaration) in program.temporal_declarations.iter().enumerate() {
            let Some(id) = TemporalId::try_from_index(index) else {
                errors.push(HirError::declaration(HirErrorKind::IndexOverflow {
                    kind: IndexKind::Temporal,
                    index,
                }));
                index_overflowed = true;
                continue;
            };
            temporal_ids.push(id);
            let name = normalize_name(&declaration.name);
            if constants.contains_key(&name) {
                errors.push(HirError::declaration_at(
                    HirErrorKind::TemporalConstantCollision { name: name.clone() },
                    locations.and_then(|locations| locations.temporals[index]),
                ));
            }
            if temporals
                .insert(name.clone(), (id, declaration.address))
                .is_some()
            {
                errors.push(HirError::declaration_at(
                    HirErrorKind::DuplicateTemporal { name: name.clone() },
                    locations.and_then(|locations| locations.temporals[index]),
                ));
            }
            if let Some(first) = temporal_addresses.insert(declaration.address, name.clone()) {
                errors.push(HirError::declaration_at(
                    HirErrorKind::DuplicateTemporalAddress {
                        address: declaration.address,
                        first,
                        second: name,
                    },
                    locations.and_then(|locations| locations.temporals[index]),
                ));
            }
        }

        let mut procedure_ids = Vec::with_capacity(program.procedures.len());
        for (index, procedure) in program.procedures.iter().enumerate() {
            let Some(id) = ProcedureId::try_from_index(index) else {
                errors.push(HirError::declaration(HirErrorKind::IndexOverflow {
                    kind: IndexKind::Procedure,
                    index,
                }));
                index_overflowed = true;
                continue;
            };
            procedure_ids.push(id);
            register_callable(
                &mut callables,
                &procedure.name,
                CallTarget::Procedure(id),
                locations
                    .and_then(|locations| locations.procedures[index].as_ref())
                    .map(|locations| locations.declaration),
                &mut errors,
            );
        }

        let mut foreign_ids = Vec::with_capacity(program.ffi_declarations.len());
        for (index, foreign) in program.ffi_declarations.iter().enumerate() {
            let Some(id) = ForeignId::try_from_index(index) else {
                errors.push(HirError::declaration(HirErrorKind::IndexOverflow {
                    kind: IndexKind::Foreign,
                    index,
                }));
                index_overflowed = true;
                continue;
            };
            foreign_ids.push(id);
            register_callable(
                &mut callables,
                &foreign.signature.name,
                CallTarget::Foreign(id),
                locations.and_then(|locations| locations.foreigns[index]),
                &mut errors,
            );
        }

        let mut property_ids = Vec::with_capacity(program.temporal_properties.len());
        let mut property_names = HashMap::<String, PropertyId>::new();
        for (index, property) in program.temporal_properties.iter().enumerate() {
            let Some(id) = PropertyId::try_from_index(index) else {
                errors.push(HirError::declaration(HirErrorKind::IndexOverflow {
                    kind: IndexKind::Property,
                    index,
                }));
                index_overflowed = true;
                continue;
            };
            property_ids.push(id);
            let name = normalize_name(&property.name);
            if property_names.insert(name.clone(), id).is_some() {
                errors.push(HirError::declaration_at(
                    HirErrorKind::DuplicateProperty { name },
                    locations.and_then(|locations| locations.properties[index]),
                ));
            }
            validate_property_cells(
                property,
                &temporals,
                locations.and_then(|locations| locations.properties[index]),
                &mut errors,
            );
        }

        let mut quote_ids = Vec::with_capacity(program.quotes.len());
        for index in 0..program.quotes.len() {
            let Ok(raw_id) = u64::try_from(index) else {
                errors.push(HirError::declaration(HirErrorKind::IndexOverflow {
                    kind: IndexKind::Quotation,
                    index,
                }));
                index_overflowed = true;
                continue;
            };
            quote_ids.push(QuoteId::new(raw_id));
        }

        // Overflow makes owner and output vectors incomplete.  It must be
        // handled before code validation rather than manufacturing IDs.
        if index_overflowed {
            return Err(errors);
        }

        let resolution_symbols = ResolutionSymbols {
            callables: &callables,
            constants: &constants,
            temporals: &temporals,
        };

        validate_block(
            &program.body,
            locations.map(|locations| locations.body.as_slice()),
            HirOwner::ProgramBody,
            program.quotes.len(),
            &resolution_symbols,
            &mut errors,
        );
        for (index, (id, quote)) in quote_ids.iter().copied().zip(&program.quotes).enumerate() {
            validate_block(
                quote,
                locations.map(|locations| locations.quotes[index].as_slice()),
                HirOwner::Quote(id),
                program.quotes.len(),
                &resolution_symbols,
                &mut errors,
            );
        }
        for (index, (id, procedure)) in procedure_ids
            .iter()
            .copied()
            .zip(&program.procedures)
            .enumerate()
        {
            validate_block(
                &procedure.body,
                locations
                    .and_then(|locations| locations.procedures[index].as_ref())
                    .map(|locations| locations.body.as_slice()),
                HirOwner::Procedure(id),
                program.quotes.len(),
                &resolution_symbols,
                &mut errors,
            );
        }

        if !errors.is_empty() {
            return Err(errors);
        }

        let mut body = lower_block(&program.body, &callables, &constants, &temporals)
            .map_err(|error| vec![error])?;
        if let Some(locations) = locations {
            attach_locations(&mut body, &locations.body);
        }

        let mut quotes = Vec::with_capacity(program.quotes.len());
        for (index, (id, quote)) in quote_ids.into_iter().zip(&program.quotes).enumerate() {
            let mut body = lower_block(quote, &callables, &constants, &temporals)
                .map_err(|error| vec![error])?;
            if let Some(locations) = locations {
                attach_locations(&mut body, &locations.quotes[index]);
            }
            quotes.push(HirQuote { id, body });
        }

        let mut procedures = Vec::with_capacity(program.procedures.len());
        for (index, (id, procedure)) in procedure_ids
            .into_iter()
            .zip(&program.procedures)
            .enumerate()
        {
            let mut body = lower_block(&procedure.body, &callables, &constants, &temporals)
                .map_err(|error| vec![error])?;
            if let Some(body_locations) = locations
                .and_then(|locations| locations.procedures[index].as_ref())
                .map(|locations| locations.body.as_slice())
            {
                attach_locations(&mut body, body_locations);
            }
            procedures.push(HirProcedure {
                id,
                name: procedure.name.clone(),
                params: procedure.params.clone(),
                returns: procedure.returns,
                effects: procedure.effects.clone(),
                body,
            });
        }

        let foreigns = foreign_ids
            .into_iter()
            .zip(&program.ffi_declarations)
            .map(|(id, declaration)| HirForeign {
                id,
                declaration: declaration.clone(),
            })
            .collect();
        let properties = property_ids
            .into_iter()
            .zip(&program.temporal_properties)
            .map(|(id, declaration)| HirProperty {
                id,
                declaration: declaration.clone(),
                cells: resolve_property_cells(declaration, &temporals),
            })
            .collect();

        Ok(Self {
            constants: constant_ids
                .into_iter()
                .zip(&program.manifests)
                .map(|(id, declaration)| HirConstant {
                    id,
                    name: declaration.name.clone(),
                    value: declaration.value,
                })
                .collect(),
            temporals: temporal_ids
                .into_iter()
                .zip(&program.temporal_declarations)
                .map(|(id, declaration)| HirTemporal {
                    id,
                    name: declaration.name.clone(),
                    address: declaration.address,
                    default: declaration.default,
                    location: locations.and_then(|locations| locations.temporals[id.index()]),
                })
                .collect(),
            procedures,
            quotes,
            body,
            foreigns,
            family_declaration: program.family_declaration.clone(),
            markov_declaration: program.markov_declaration.clone(),
            quantum_declaration: program.quantum_declaration.clone(),
            properties,
        })
    }
}

fn normalize_name(name: &str) -> String {
    name.to_lowercase()
}

fn register_callable(
    callables: &mut HashMap<String, CallableSymbol>,
    spelling: &str,
    target: CallTarget,
    location: Option<SourceSpan>,
    errors: &mut Vec<HirError>,
) {
    let name = normalize_name(spelling);
    if let Some(first) = callables.get(&name) {
        errors.push(HirError::declaration_at(
            HirErrorKind::DuplicateCallable {
                name,
                first: first.target.kind(),
                second: target.kind(),
            },
            location,
        ));
    } else {
        callables.insert(name, CallableSymbol { target });
    }
}

fn property_named_cells(predicate: &PropertyPredicate) -> Vec<(&str, u64)> {
    let mut cells = Vec::new();
    let mut work = vec![predicate];
    while let Some(predicate) = work.pop() {
        match predicate {
            PropertyPredicate::Compare { cell, .. } => {
                if let Some(name) = &cell.name {
                    cells.push((name.as_str(), cell.address));
                }
            }
            PropertyPredicate::Not(inner) => work.push(inner),
            PropertyPredicate::And(left, right) | PropertyPredicate::Or(left, right) => {
                work.push(right);
                work.push(left);
            }
        }
    }
    cells
}

fn validate_property_cells(
    property: &TemporalPropertyDeclaration,
    temporals: &HashMap<String, (TemporalId, u64)>,
    location: Option<SourceSpan>,
    errors: &mut Vec<HirError>,
) {
    let Some(predicate) = &property.predicate else {
        return;
    };
    for (name, used) in property_named_cells(predicate) {
        match temporals.get(&normalize_name(name)) {
            None => errors.push(HirError::declaration_at(
                HirErrorKind::UnknownTemporal {
                    name: name.to_string(),
                },
                location,
            )),
            Some((_, declared)) if *declared != used => errors.push(HirError::declaration_at(
                HirErrorKind::TemporalAddressMismatch {
                    name: name.to_string(),
                    declared: *declared,
                    used,
                },
                location,
            )),
            Some(_) => {}
        }
    }
}

fn resolve_property_cells(
    property: &TemporalPropertyDeclaration,
    temporals: &HashMap<String, (TemporalId, u64)>,
) -> Vec<HirPropertyCell> {
    let Some(predicate) = &property.predicate else {
        return vec![HirPropertyCell {
            temporal: None,
            address: property.address,
        }];
    };
    let mut cells = Vec::new();
    let mut work = vec![predicate];
    while let Some(predicate) = work.pop() {
        match predicate {
            PropertyPredicate::Compare { cell, .. } => cells.push(HirPropertyCell {
                temporal: cell
                    .name
                    .as_ref()
                    .and_then(|name| temporals.get(&normalize_name(name)))
                    .map(|(id, _)| *id),
                address: cell.address,
            }),
            PropertyPredicate::Not(inner) => work.push(inner),
            PropertyPredicate::And(left, right) | PropertyPredicate::Or(left, right) => {
                work.push(right);
                work.push(left);
            }
        }
    }
    cells
}

fn validate_block(
    block: &[Stmt],
    locations: Option<&[StmtLocations]>,
    owner: HirOwner,
    quote_count: usize,
    symbols: &ResolutionSymbols<'_>,
    errors: &mut Vec<HirError>,
) {
    let mut work = Vec::new();
    push_validation_block(&mut work, block, locations);

    while let Some((statement, location)) = work.pop() {
        let source_span = location.map(|location| location.span);
        match statement {
            Stmt::PushConstant { name, value } => {
                match symbols.constants.get(&normalize_name(name)) {
                    None => errors.push(HirError::statement_at(
                        HirErrorKind::UnknownConstant { name: name.clone() },
                        owner,
                        source_span,
                    )),
                    Some((_, declared)) if declared != value => {
                        errors.push(HirError::statement_at(
                            HirErrorKind::ConstantValueMismatch {
                                name: name.clone(),
                                declared: *declared,
                                used: *value,
                            },
                            owner,
                            source_span,
                        ))
                    }
                    Some(_) => {}
                }
            }
            Stmt::Call { name } => {
                if !symbols.callables.contains_key(&normalize_name(name)) {
                    errors.push(HirError::statement_at(
                        HirErrorKind::UnknownCall { name: name.clone() },
                        owner,
                        source_span,
                    ));
                }
            }
            Stmt::ReadTemporal { name, address } => {
                match symbols.temporals.get(&normalize_name(name)) {
                    None => errors.push(HirError::statement_at(
                        HirErrorKind::UnknownTemporal { name: name.clone() },
                        owner,
                        source_span,
                    )),
                    Some((_, declared)) if declared != address => {
                        errors.push(HirError::statement_at(
                            HirErrorKind::TemporalAddressMismatch {
                                name: name.clone(),
                                declared: *declared,
                                used: *address,
                            },
                            owner,
                            source_span,
                        ));
                    }
                    Some(_) => {}
                }
            }
            Stmt::PushQuote(quote) if quote.as_u64() >= quote_count as u64 => {
                errors.push(HirError::statement_at(
                    HirErrorKind::InvalidQuote {
                        quote: *quote,
                        quote_count,
                    },
                    owner,
                    source_span,
                ));
            }
            Stmt::Op(_) | Stmt::Push(_) | Stmt::PushQuote(_) => {}
            Stmt::If {
                then_branch,
                else_branch,
            } => {
                if let Some(else_branch) = else_branch {
                    push_validation_block(
                        &mut work,
                        else_branch,
                        location
                            .and_then(|location| location.child_blocks.get(1))
                            .map(Vec::as_slice),
                    );
                }
                push_validation_block(
                    &mut work,
                    then_branch,
                    location
                        .and_then(|location| location.child_blocks.first())
                        .map(Vec::as_slice),
                );
            }
            Stmt::While { cond, body } => {
                push_validation_block(
                    &mut work,
                    body,
                    location
                        .and_then(|location| location.child_blocks.get(1))
                        .map(Vec::as_slice),
                );
                push_validation_block(
                    &mut work,
                    cond,
                    location
                        .and_then(|location| location.child_blocks.first())
                        .map(Vec::as_slice),
                );
            }
            Stmt::Block(body) | Stmt::TemporalScope { body, .. } => {
                push_validation_block(
                    &mut work,
                    body,
                    location
                        .and_then(|location| location.child_blocks.first())
                        .map(Vec::as_slice),
                );
            }
        }
    }
}

fn push_validation_block<'a>(
    work: &mut Vec<(&'a Stmt, Option<&'a StmtLocations>)>,
    block: &'a [Stmt],
    locations: Option<&'a [StmtLocations]>,
) {
    for index in (0..block.len()).rev() {
        work.push((
            &block[index],
            locations.and_then(|locations| locations.get(index)),
        ));
    }
}

fn attach_locations(block: &mut [HirStmt], locations: &[StmtLocations]) {
    let mut work = vec![(block, locations)];
    while let Some((statements, locations)) = work.pop() {
        debug_assert_eq!(statements.len(), locations.len());
        for (statement, location) in statements.iter_mut().zip(locations) {
            statement.location = Some(location.span);
            match (&mut statement.kind, location.child_blocks.as_slice()) {
                (
                    HirStmtKind::If {
                        then_branch,
                        else_branch,
                    },
                    child_blocks,
                ) => {
                    if let (Some(else_branch), Some(else_locations)) =
                        (else_branch.as_mut(), child_blocks.get(1))
                    {
                        work.push((else_branch.as_mut_slice(), else_locations.as_slice()));
                    }
                    if let Some(then_locations) = child_blocks.first() {
                        work.push((then_branch.as_mut_slice(), then_locations.as_slice()));
                    }
                }
                (HirStmtKind::While { cond, body }, child_blocks) => {
                    if let Some(body_locations) = child_blocks.get(1) {
                        work.push((body.as_mut_slice(), body_locations.as_slice()));
                    }
                    if let Some(cond_locations) = child_blocks.first() {
                        work.push((cond.as_mut_slice(), cond_locations.as_slice()));
                    }
                }
                (HirStmtKind::Block(body), child_blocks)
                | (HirStmtKind::TemporalScope { body, .. }, child_blocks) => {
                    if let Some(body_locations) = child_blocks.first() {
                        work.push((body.as_mut_slice(), body_locations.as_slice()));
                    }
                }
                (
                    HirStmtKind::Op(_)
                    | HirStmtKind::Push(_)
                    | HirStmtKind::PushConstant { .. }
                    | HirStmtKind::ReadTemporal { .. }
                    | HirStmtKind::PushQuote(_)
                    | HirStmtKind::Call { .. },
                    _,
                ) => {}
            }
        }
    }
}

enum LowerWork<'a> {
    Block(&'a [Stmt]),
    Statement(&'a Stmt),
    FinishList(usize),
    FinishIf { has_else: bool },
    FinishWhile,
    FinishBlock,
    FinishTemporal { base: u64, size: u64, cell_bits: u8 },
}

enum LowerPiece {
    Statement(HirStmt),
    List(Vec<HirStmt>),
}

fn lower_block(
    block: &[Stmt],
    callables: &HashMap<String, CallableSymbol>,
    constants: &HashMap<String, (ConstantId, u64)>,
    temporals: &HashMap<String, (TemporalId, u64)>,
) -> Result<Vec<HirStmt>, HirError> {
    let mut work = vec![LowerWork::Block(block)];
    let mut pieces = Vec::new();

    while let Some(item) = work.pop() {
        match item {
            LowerWork::Block(statements) => {
                work.push(LowerWork::FinishList(statements.len()));
                work.extend(statements.iter().rev().map(LowerWork::Statement));
            }
            LowerWork::Statement(statement) => match statement {
                Stmt::Op(op) => pieces.push(LowerPiece::Statement(HirStmt::unlocated(
                    HirStmtKind::Op(*op),
                ))),
                Stmt::Push(value) => pieces.push(LowerPiece::Statement(HirStmt::unlocated(
                    HirStmtKind::Push(value.clone()),
                ))),
                Stmt::PushConstant { name, value } => {
                    let (id, declared) = constants.get(&normalize_name(name)).ok_or_else(|| {
                        HirError::internal(format!(
                            "validated constant {name:?} disappeared from the symbol table"
                        ))
                    })?;
                    debug_assert_eq!(declared, value);
                    pieces.push(LowerPiece::Statement(HirStmt::unlocated(
                        HirStmtKind::PushConstant {
                            id: *id,
                            value: *value,
                        },
                    )));
                }
                Stmt::ReadTemporal { name, address } => {
                    let (id, declared) = temporals.get(&normalize_name(name)).ok_or_else(|| {
                        HirError::internal(format!(
                            "validated temporal cell {name:?} disappeared from the symbol table"
                        ))
                    })?;
                    debug_assert_eq!(declared, address);
                    pieces.push(LowerPiece::Statement(HirStmt::unlocated(
                        HirStmtKind::ReadTemporal {
                            id: *id,
                            address: *address,
                        },
                    )));
                }
                Stmt::PushQuote(quote) => pieces.push(LowerPiece::Statement(HirStmt::unlocated(
                    HirStmtKind::PushQuote(*quote),
                ))),
                Stmt::Call { name } => {
                    let target = callables
                        .get(&normalize_name(name))
                        .ok_or_else(|| {
                            HirError::internal(format!(
                                "validated call {name:?} disappeared from the symbol table"
                            ))
                        })?
                        .target;
                    pieces.push(LowerPiece::Statement(HirStmt::unlocated(
                        HirStmtKind::Call { target },
                    )));
                }
                Stmt::If {
                    then_branch,
                    else_branch,
                } => {
                    work.push(LowerWork::FinishIf {
                        has_else: else_branch.is_some(),
                    });
                    if let Some(else_branch) = else_branch {
                        work.push(LowerWork::Block(else_branch));
                    }
                    work.push(LowerWork::Block(then_branch));
                }
                Stmt::While { cond, body } => {
                    work.push(LowerWork::FinishWhile);
                    work.push(LowerWork::Block(body));
                    work.push(LowerWork::Block(cond));
                }
                Stmt::Block(body) => {
                    work.push(LowerWork::FinishBlock);
                    work.push(LowerWork::Block(body));
                }
                Stmt::TemporalScope {
                    base,
                    size,
                    cell_bits,
                    body,
                } => {
                    work.push(LowerWork::FinishTemporal {
                        base: *base,
                        size: *size,
                        cell_bits: *cell_bits,
                    });
                    work.push(LowerWork::Block(body));
                }
            },
            LowerWork::FinishList(length) => {
                if pieces.len() < length {
                    return Err(HirError::internal("statement-list result underflow"));
                }
                let start = pieces.len() - length;
                let tail = pieces.split_off(start);
                let mut statements = Vec::with_capacity(length);
                for piece in tail {
                    let LowerPiece::Statement(statement) = piece else {
                        return Err(HirError::internal(
                            "statement list contained a nested list result",
                        ));
                    };
                    statements.push(statement);
                }
                pieces.push(LowerPiece::List(statements));
            }
            LowerWork::FinishIf { has_else } => {
                let else_branch = if has_else {
                    Some(pop_list(&mut pieces, "IF else branch")?)
                } else {
                    None
                };
                let then_branch = pop_list(&mut pieces, "IF then branch")?;
                pieces.push(LowerPiece::Statement(HirStmt::unlocated(HirStmtKind::If {
                    then_branch,
                    else_branch,
                })));
            }
            LowerWork::FinishWhile => {
                let body = pop_list(&mut pieces, "WHILE body")?;
                let cond = pop_list(&mut pieces, "WHILE condition")?;
                pieces.push(LowerPiece::Statement(HirStmt::unlocated(
                    HirStmtKind::While { cond, body },
                )));
            }
            LowerWork::FinishBlock => {
                let body = pop_list(&mut pieces, "BLOCK body")?;
                pieces.push(LowerPiece::Statement(HirStmt::unlocated(
                    HirStmtKind::Block(body),
                )));
            }
            LowerWork::FinishTemporal {
                base,
                size,
                cell_bits,
            } => {
                let body = pop_list(&mut pieces, "TEMPORAL body")?;
                pieces.push(LowerPiece::Statement(HirStmt::unlocated(
                    HirStmtKind::TemporalScope {
                        base,
                        size,
                        cell_bits,
                        body,
                    },
                )));
            }
        }
    }

    if pieces.len() != 1 {
        return Err(HirError::internal(format!(
            "root lowering produced {} results instead of one",
            pieces.len()
        )));
    }
    pop_list(&mut pieces, "root statement list")
}

fn pop_list(pieces: &mut Vec<LowerPiece>, context: &str) -> Result<Vec<HirStmt>, HirError> {
    match pieces.pop() {
        Some(LowerPiece::List(statements)) => Ok(statements),
        Some(LowerPiece::Statement(_)) => Err(HirError::internal(format!(
            "{context} produced a statement instead of a list"
        ))),
        None => Err(HirError::internal(format!(
            "{context} produced no lowering result"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Procedure, PropertyComparison, RationalLiteral};
    use crate::runtime::ffi::{FFIEffect, FFISignature};

    fn procedure(name: &str, body: Vec<Stmt>) -> crate::ast::Procedure {
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
            symbol_name: Some(format!("external_{name}")),
        }
    }

    #[test]
    fn resolves_procedure_and_foreign_calls_case_insensitively() {
        let mut program = Program::new();
        program.procedures.push(procedure("Local", Vec::new()));
        program.ffi_declarations.push(foreign("Native"));
        program.body = vec![
            Stmt::Call {
                name: "LOCAL".to_string(),
            },
            Stmt::Call {
                name: "native".to_string(),
            },
        ];

        let hir = HirProgram::resolve(&program).expect("program resolves");
        assert!(matches!(
            hir.body[0].kind,
            HirStmtKind::Call {
                target: CallTarget::Procedure(id)
            } if id == ProcedureId::try_from_index(0).unwrap()
        ));
        assert!(matches!(
            hir.body[1].kind,
            HirStmtKind::Call {
                target: CallTarget::Foreign(id)
            } if id == ForeignId::try_from_index(0).unwrap()
        ));
        assert_eq!(
            hir.foreigns[0].declaration.symbol_name.as_deref(),
            Some("external_Native")
        );
    }

    #[test]
    fn rejects_unknown_duplicate_and_cross_kind_callable_names() {
        let mut unknown = Program::new();
        unknown.body.push(Stmt::Call {
            name: "missing".to_string(),
        });
        let errors = HirProgram::resolve(&unknown).unwrap_err();
        assert!(matches!(
            errors[0].kind,
            HirErrorKind::UnknownCall { ref name } if name == "missing"
        ));

        let mut duplicate = Program::new();
        duplicate.procedures.push(procedure("Same", Vec::new()));
        duplicate.procedures.push(procedure("sAME", Vec::new()));
        duplicate.ffi_declarations.push(foreign("SAME"));
        let errors = HirProgram::resolve(&duplicate).unwrap_err();
        assert_eq!(errors.len(), 2);
        assert!(matches!(
            errors[0].kind,
            HirErrorKind::DuplicateCallable {
                first: CallableKind::Procedure,
                second: CallableKind::Procedure,
                ..
            }
        ));
        assert!(matches!(
            errors[1].kind,
            HirErrorKind::DuplicateCallable {
                first: CallableKind::Procedure,
                second: CallableKind::Foreign,
                ..
            }
        ));
    }

    #[test]
    fn validates_quote_references_in_every_nested_owner() {
        let invalid = Stmt::PushQuote(QuoteId::new(7));
        let mut program = Program::new();
        program.body = vec![Stmt::If {
            then_branch: vec![Stmt::Block(vec![invalid.clone()])],
            else_branch: None,
        }];
        program.quotes.push(vec![Stmt::While {
            cond: vec![Stmt::Push(Value::ONE)],
            body: vec![Stmt::TemporalScope {
                base: 0,
                size: 1,
                cell_bits: 64,
                body: vec![invalid.clone()],
            }],
        }]);
        program.procedures.push(procedure("nested", vec![invalid]));

        let errors = HirProgram::resolve(&program).unwrap_err();
        assert_eq!(errors.len(), 3);
        assert_eq!(errors[0].owner, Some(HirOwner::ProgramBody));
        assert_eq!(errors[1].owner, Some(HirOwner::Quote(QuoteId::new(0))));
        assert_eq!(
            errors[2].owner,
            Some(HirOwner::Procedure(ProcedureId::try_from_index(0).unwrap()))
        );
        assert!(errors.iter().all(|error| matches!(
            error.kind,
            HirErrorKind::InvalidQuote {
                quote,
                quote_count: 1
            } if quote == QuoteId::new(7)
        )));
    }

    #[test]
    fn assigns_deterministic_source_order_ids_and_retains_declarations() {
        let mut program = Program::new();
        program.procedures.push(procedure("zeta", Vec::new()));
        program.procedures.push(procedure("alpha", Vec::new()));
        program.ffi_declarations.push(foreign("second_table"));
        program.quotes.push(Vec::new());
        program
            .temporal_properties
            .push(TemporalPropertyDeclaration {
                name: "safety".to_string(),
                address: 4,
                comparison: PropertyComparison::Eq,
                value: 9,
                predicate: None,
            });
        program.markov_declaration = Some(MarkovDeclaration {
            name: "m".to_string(),
            states: 1,
            transition: vec![vec![RationalLiteral {
                numerator: 1,
                denominator: 1,
            }]],
            accepting_states: vec![0],
            accept_at_least: RationalLiteral {
                numerator: 1,
                denominator: 1,
            },
            reject_at_most: RationalLiteral {
                numerator: 0,
                denominator: 1,
            },
        });

        let first = HirProgram::resolve(&program).expect("first resolution");
        let second = HirProgram::resolve(&program).expect("second resolution");
        let first_ids: Vec<_> = first.procedures.iter().map(|proc| proc.id).collect();
        let second_ids: Vec<_> = second.procedures.iter().map(|proc| proc.id).collect();
        assert_eq!(first_ids, second_ids);
        assert_eq!(first_ids[0].index(), 0);
        assert_eq!(first_ids[1].index(), 1);
        assert_eq!(first.foreigns[0].id.index(), 0);
        assert_eq!(first.quotes[0].id, QuoteId::new(0));
        assert_eq!(first.properties[0].id.index(), 0);
        assert_eq!(first.markov_declaration, program.markov_declaration);
    }

    #[test]
    fn preserves_every_statement_shape_during_iterative_lowering() {
        let mut program = Program::new();
        program.procedures.push(procedure("callee", Vec::new()));
        program.quotes.push(vec![Stmt::Op(OpCode::Nop)]);
        program.body = vec![
            Stmt::Op(OpCode::Add),
            Stmt::Push(Value::new(12)),
            Stmt::PushQuote(QuoteId::new(0)),
            Stmt::If {
                then_branch: vec![Stmt::Op(OpCode::Dup)],
                else_branch: Some(vec![Stmt::Op(OpCode::Pop)]),
            },
            Stmt::While {
                cond: vec![Stmt::Push(Value::ONE)],
                body: vec![Stmt::Op(OpCode::Nop)],
            },
            Stmt::Block(vec![Stmt::Op(OpCode::Swap)]),
            Stmt::Call {
                name: "callee".to_string(),
            },
            Stmt::TemporalScope {
                base: 8,
                size: 3,
                cell_bits: 16,
                body: vec![Stmt::Op(OpCode::Oracle)],
            },
        ];

        let hir = HirProgram::resolve(&program).expect("program resolves");
        assert!(matches!(hir.body[0].kind, HirStmtKind::Op(OpCode::Add)));
        assert!(matches!(hir.body[1].kind, HirStmtKind::Push(ref value) if value.val == 12));
        assert!(matches!(
            hir.body[2].kind,
            HirStmtKind::PushQuote(id) if id == QuoteId::new(0)
        ));
        assert!(matches!(
            hir.body[3].kind,
            HirStmtKind::If {
                ref then_branch,
                else_branch: Some(ref else_branch)
            } if matches!(then_branch[0].kind, HirStmtKind::Op(OpCode::Dup))
                && matches!(else_branch[0].kind, HirStmtKind::Op(OpCode::Pop))
        ));
        assert!(matches!(
            hir.body[4].kind,
            HirStmtKind::While { ref cond, ref body }
                if matches!(cond[0].kind, HirStmtKind::Push(_))
                    && matches!(body[0].kind, HirStmtKind::Op(OpCode::Nop))
        ));
        assert!(matches!(
            hir.body[5].kind,
            HirStmtKind::Block(ref body)
                if matches!(body[0].kind, HirStmtKind::Op(OpCode::Swap))
        ));
        assert!(matches!(
            hir.body[6].kind,
            HirStmtKind::Call {
                target: CallTarget::Procedure(_)
            }
        ));
        assert!(matches!(
            hir.body[7].kind,
            HirStmtKind::TemporalScope {
                base: 8,
                size: 3,
                cell_bits: 16,
                ref body
            } if matches!(body[0].kind, HirStmtKind::Op(OpCode::Oracle))
        ));
    }

    #[test]
    fn u32_ids_check_the_artifact_boundary() {
        assert_eq!(ProcedureId::try_from_index(0).unwrap().as_u32(), 0);
        assert_eq!(
            ProcedureId::try_from_index(u32::MAX as usize)
                .unwrap()
                .as_u32(),
            u32::MAX
        );
        if usize::BITS > u32::BITS {
            assert!(ProcedureId::try_from_index(u32::MAX as usize + 1).is_none());
            assert!(ForeignId::try_from_index(u32::MAX as usize + 1).is_none());
            assert!(PropertyId::try_from_index(u32::MAX as usize + 1).is_none());
        }
    }

    fn located_source(source: &str) -> crate::ast::LocatedProgram {
        let source_id = crate::source::SourceId::new(23);
        let located_tokens = crate::lexer::lex(source_id, source).unwrap();
        let tokens = located_tokens
            .iter()
            .map(|token| token.token.clone())
            .collect::<Vec<_>>();
        let spans = located_tokens
            .iter()
            .map(|token| token.span)
            .collect::<Vec<_>>();
        let mut parser = crate::parser::Parser::new_with_source_spans(&tokens, &spans).unwrap();
        parser.parse_located_program().unwrap()
    }

    #[test]
    fn located_resolution_reports_the_exact_call_site() {
        let source = "PROCEDURE known { 1 }\nknown\n";
        let mut located = located_source(source);
        let Stmt::Call { name } = &mut located.program.body[0] else {
            panic!("expected parsed call");
        };
        *name = "missing".to_string();

        let errors = HirProgram::resolve_located(&located.program, &located.locations).unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            errors[0].kind,
            HirErrorKind::UnknownCall { ref name } if name == "missing"
        ));
        let span = errors[0].location.unwrap();
        assert_eq!(&source[span.range.start..span.range.end], "known");
    }

    #[test]
    fn located_resolution_reports_the_later_duplicate_declaration() {
        let source = "PROCEDURE first { 1 }\nPROCEDURE second { 2 }\n0\n";
        let mut located = located_source(source);
        located.program.procedures[1].name = "first".to_string();

        let errors = HirProgram::resolve_located(&located.program, &located.locations).unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            errors[0].kind,
            HirErrorKind::DuplicateCallable { ref name, .. } if name == "first"
        ));
        let span = errors[0].location.unwrap();
        assert_eq!(
            &source[span.range.start..span.range.end],
            "PROCEDURE second { 2 }"
        );
    }
}
