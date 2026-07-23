//! Compile-time type system for OUROCHRONOS.
//!
//! This module provides static analysis to track **temporal tainting** - which values
//! depend on oracle reads (temporal) vs. which are pure constants (ground truth).
//!
//! # Type Lattice
//!
//! ```text
//!        Temporal (depends on oracle)
//!           ↑
//!        Unknown (not yet determined)
//!           ↑
//!         Pure (constant, no oracle dependency)
//! ```
//!
//! # Effect System
//!
//! The effect system tracks side effects:
//! - `Pure`: No effects (referentially transparent)
//! - `Reads`: Reads from memory
//! - `Writes`: Writes to memory
//! - `Temporal`: Uses ORACLE/PROPHECY
//! - `IO`: Performs input/output
//! - `Alloc`: Allocates or mutates runtime data structures
//! - `FFI`: Calls foreign code
//! - `FileIO`: Accesses files
//! - `Network`: Accesses sockets
//! - `System`: Accesses process, clock, sleep, or randomness services
//!
//! Procedure contracts are sets of those capabilities. `Pure` covers only an
//! empty computed set; it is not an alias for an unchecked procedure.
//!
//! # Type Rules
//!
//! | Operation | Input Types | Output Type | Effect |
//! |-----------|-------------|-------------|--------|
//! | `ORACLE` | `Pure` (address) | `Temporal` | Temporal |
//! | `PROPHECY` | `Any`, `Pure` (address) | - | Temporal |
//! | `FETCH` | `Pure` (address) | `Unknown` | Reads |
//! | `STORE` | `Any`, `Pure` (address) | - | Writes |
//! | Arithmetic | `Temporal × Any` | `Temporal` | Pure |
//! | `OUTPUT` | `Any` | - | IO |
//!
//! # Safety Properties
//!
//! The type checker enforces:
//! 1. **Causal Integrity**: Temporal values must flow through verification
//! 2. **Stability Guarantee**: Pure values cannot spontaneously become temporal
//! 3. **Effect Soundness**: Declared effects must cover actual effects
//! 4. **Taint Tracking**: The programmer knows which computations depend on the future

use crate::ast::{Effect as AstEffect, OpCode, Procedure, Program, Stmt};
use crate::core::Address;
use std::collections::{HashMap, HashSet};

/// Temporal type for compile-time causal tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TemporalType {
    /// Value has no temporal dependency (constant, ground truth).
    Pure,
    /// Value depends on at least one oracle read.
    Temporal,
    /// Type is not yet determined (for inference).
    #[default]
    Unknown,
}

impl TemporalType {
    /// Join two types in the lattice.
    /// Temporal ⊔ anything = Temporal
    /// Pure ⊔ Pure = Pure
    /// Unknown ⊔ x = x
    pub fn join(self, other: Self) -> Self {
        match (self, other) {
            (TemporalType::Temporal, _) | (_, TemporalType::Temporal) => TemporalType::Temporal,
            (TemporalType::Pure, TemporalType::Pure) => TemporalType::Pure,
            (TemporalType::Unknown, x) | (x, TemporalType::Unknown) => x,
        }
    }

    /// Check if this type is more specific than another.
    pub fn is_subtype_of(self, other: Self) -> bool {
        match (self, other) {
            (_, TemporalType::Unknown) => true,
            (TemporalType::Pure, TemporalType::Pure) => true,
            (TemporalType::Temporal, TemporalType::Temporal) => true,
            (TemporalType::Pure, TemporalType::Temporal) => true, // Pure can be used where Temporal expected
            _ => false,
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            TemporalType::Pure => "Pure",
            TemporalType::Temporal => "Temporal",
            TemporalType::Unknown => "Unknown",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Linear Types
// ═══════════════════════════════════════════════════════════════════════════

/// Linearity qualifier for values.
///
/// Linear types enforce that certain values (particularly those from ORACLE)
/// can only be used exactly once. This prevents free duplication of temporal
/// information, which is essential for CTC semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Linearity {
    /// Value can be freely copied and discarded.
    #[default]
    Unrestricted,
    /// Value must be consumed exactly once.
    /// Cannot be duplicated via DUP, OVER, PICK, etc.
    Linear,
}

impl Linearity {
    /// Check if this value can be duplicated.
    pub fn can_duplicate(&self) -> bool {
        matches!(self, Linearity::Unrestricted)
    }

    /// Check if this is a linear value.
    pub fn is_linear(&self) -> bool {
        matches!(self, Linearity::Linear)
    }

    /// Join two linearity qualifiers.
    /// Linear values remain linear when combined with anything.
    pub fn join(self, other: Self) -> Self {
        match (self, other) {
            (Linearity::Linear, _) | (_, Linearity::Linear) => Linearity::Linear,
            (Linearity::Unrestricted, Linearity::Unrestricted) => Linearity::Unrestricted,
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Linearity::Unrestricted => "Unrestricted",
            Linearity::Linear => "Linear",
        }
    }
}

/// A stack type combining temporal type and linearity.
///
/// This is the type assigned to values on the abstract stack during
/// type checking. It tracks both temporal tainting and linearity constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct StackType {
    /// The temporal type (Pure, Temporal, Unknown).
    pub temporal: TemporalType,
    /// The linearity qualifier.
    pub linearity: Linearity,
}

impl StackType {
    /// Create a pure, unrestricted type.
    pub const PURE: StackType = StackType {
        temporal: TemporalType::Pure,
        linearity: Linearity::Unrestricted,
    };

    /// Create an unknown, unrestricted type.
    pub const UNKNOWN: StackType = StackType {
        temporal: TemporalType::Unknown,
        linearity: Linearity::Unrestricted,
    };

    /// Create a temporal, linear type (for ORACLE results).
    pub const TEMPORAL_LINEAR: StackType = StackType {
        temporal: TemporalType::Temporal,
        linearity: Linearity::Linear,
    };

    /// Create a temporal, unrestricted type (after consuming linearity).
    pub const TEMPORAL: StackType = StackType {
        temporal: TemporalType::Temporal,
        linearity: Linearity::Unrestricted,
    };

    /// Create a new stack type.
    pub fn new(temporal: TemporalType, linearity: Linearity) -> Self {
        Self {
            temporal,
            linearity,
        }
    }

    /// Create from a temporal type with unrestricted linearity.
    pub fn from_temporal(temporal: TemporalType) -> Self {
        Self {
            temporal,
            linearity: Linearity::Unrestricted,
        }
    }

    /// Check if this value can be duplicated.
    pub fn can_duplicate(&self) -> bool {
        self.linearity.can_duplicate()
    }

    /// Check if this is a linear value.
    pub fn is_linear(&self) -> bool {
        self.linearity.is_linear()
    }

    /// Join two stack types.
    pub fn join(self, other: Self) -> Self {
        Self {
            temporal: self.temporal.join(other.temporal),
            linearity: self.linearity.join(other.linearity),
        }
    }

    /// Consume linearity, returning an unrestricted version.
    /// Used when a linear value is properly consumed (e.g., by arithmetic).
    pub fn consume(self) -> Self {
        Self {
            temporal: self.temporal,
            linearity: Linearity::Unrestricted,
        }
    }

    /// Description for error messages.
    pub fn describe(&self) -> String {
        format!("{}+{}", self.temporal.name(), self.linearity.name())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Effect System
// ═══════════════════════════════════════════════════════════════════════════

/// Computational effects for effect tracking and enforcement.
///
/// Effects form a lattice where composition is join (least upper bound).
/// This enables modular reasoning about side effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ComputedEffect {
    /// No effects: pure, referentially transparent computation.
    #[default]
    Pure,
    /// Reads from memory (INDEX, FETCH-like operations).
    Reads,
    /// Writes to memory (STORE-like operations).
    Writes,
    /// Temporal effects: ORACLE and/or PROPHECY operations.
    Temporal,
    /// I/O effects: INPUT, OUTPUT, EMIT.
    IO,
}

impl ComputedEffect {
    /// Join two effects in the effect lattice.
    ///
    /// The result is the least effect that subsumes both.
    pub fn join(self, other: Self) -> Self {
        use ComputedEffect::*;
        match (self, other) {
            (Pure, x) | (x, Pure) => x,
            (IO, _) | (_, IO) => IO,
            (Temporal, _) | (_, Temporal) => Temporal,
            (Reads, Writes) | (Writes, Reads) => IO, // Both reads and writes = IO
            (Reads, Reads) => Reads,
            (Writes, Writes) => Writes,
        }
    }

    /// Check if this effect is subsumed by another.
    ///
    /// Returns true if `self` can be used where `other` is expected.
    pub fn is_subeffect_of(self, other: Self) -> bool {
        use ComputedEffect::*;
        match (self, other) {
            (Pure, _) => true, // Pure can go anywhere
            (_, IO) => true,   // IO subsumes all
            (Reads, Reads) => true,
            (Writes, Writes) => true,
            (Temporal, Temporal) => true,
            _ => false,
        }
    }

    /// Get human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            ComputedEffect::Pure => "pure",
            ComputedEffect::Reads => "reads",
            ComputedEffect::Writes => "writes",
            ComputedEffect::Temporal => "temporal",
            ComputedEffect::IO => "io",
        }
    }
}

/// Set of effects for comprehensive effect tracking.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EffectSet {
    /// Whether this computation reads from memory.
    pub reads: bool,
    /// Whether this computation writes to memory.
    pub writes: bool,
    /// Whether this computation uses ORACLE.
    pub oracle: bool,
    /// Whether this computation uses PROPHECY.
    pub prophecy: bool,
    /// Whether this computation performs I/O.
    pub io: bool,
    /// Whether this computation allocates or manipulates data structures.
    pub alloc: bool,
    /// Whether this computation invokes foreign code.
    pub ffi: bool,
    /// Whether this computation performs file I/O.
    pub file_io: bool,
    /// Whether this computation performs network I/O.
    pub network: bool,
    /// Whether this computation invokes process/system services.
    pub system: bool,
    /// Specific ORACLE addresses read (if known statically).
    pub read_addresses: HashSet<Address>,
    /// Specific PROPHECY addresses written (if known statically).
    pub write_addresses: HashSet<Address>,
    /// At least one ORACLE address was runtime-computed.
    pub unknown_oracle_address: bool,
    /// At least one PROPHECY address was runtime-computed.
    pub unknown_prophecy_address: bool,
}

impl EffectSet {
    /// Create an empty (pure) effect set.
    pub fn pure() -> Self {
        Self::default()
    }

    /// Create an effect set for ORACLE operation.
    pub fn oracle(addr: Option<Address>) -> Self {
        let mut set = Self {
            oracle: true,
            ..Default::default()
        };
        if let Some(a) = addr {
            set.read_addresses.insert(a);
        } else {
            set.unknown_oracle_address = true;
        }
        set
    }

    /// Create an effect set for PROPHECY operation.
    pub fn prophecy(addr: Option<Address>) -> Self {
        let mut set = Self {
            prophecy: true,
            ..Default::default()
        };
        if let Some(a) = addr {
            set.write_addresses.insert(a);
        } else {
            set.unknown_prophecy_address = true;
        }
        set
    }

    /// Create an effect set for memory read.
    pub fn reads() -> Self {
        Self {
            reads: true,
            ..Default::default()
        }
    }

    /// Create an effect set for memory write.
    pub fn writes() -> Self {
        Self {
            writes: true,
            ..Default::default()
        }
    }

    /// Create an effect set for I/O.
    pub fn io() -> Self {
        Self {
            io: true,
            ..Default::default()
        }
    }

    /// Join two effect sets.
    pub fn join(&self, other: &Self) -> Self {
        EffectSet {
            reads: self.reads || other.reads,
            writes: self.writes || other.writes,
            oracle: self.oracle || other.oracle,
            prophecy: self.prophecy || other.prophecy,
            io: self.io || other.io,
            alloc: self.alloc || other.alloc,
            ffi: self.ffi || other.ffi,
            file_io: self.file_io || other.file_io,
            network: self.network || other.network,
            system: self.system || other.system,
            read_addresses: self
                .read_addresses
                .union(&other.read_addresses)
                .cloned()
                .collect(),
            write_addresses: self
                .write_addresses
                .union(&other.write_addresses)
                .cloned()
                .collect(),
            unknown_oracle_address: self.unknown_oracle_address || other.unknown_oracle_address,
            unknown_prophecy_address: self.unknown_prophecy_address
                || other.unknown_prophecy_address,
        }
    }

    /// Check if this effect set is pure (no effects).
    pub fn is_pure(&self) -> bool {
        !self.reads
            && !self.writes
            && !self.oracle
            && !self.prophecy
            && !self.io
            && !self.alloc
            && !self.ffi
            && !self.file_io
            && !self.network
            && !self.system
    }

    /// Check if this effect set has temporal effects.
    pub fn is_temporal(&self) -> bool {
        self.oracle || self.prophecy
    }

    /// Check if declared AST effects cover this computed effect set.
    pub fn is_covered_by(&self, declared: &[AstEffect]) -> bool {
        // PURE is deliberately exclusive in meaning: even `PURE IO` cannot
        // launder an effectful body through the contract checker.
        if declared.iter().any(|e| matches!(e, AstEffect::Pure)) {
            return self.is_pure();
        }

        let temporal = declared
            .iter()
            .any(|effect| matches!(effect, AstEffect::Temporal));
        let declared_reads: HashSet<Address> = declared
            .iter()
            .filter_map(|effect| match effect {
                AstEffect::Reads(address) => Some(*address),
                _ => None,
            })
            .collect();
        let declared_writes: HashSet<Address> = declared
            .iter()
            .filter_map(|effect| match effect {
                AstEffect::Writes(address) => Some(*address),
                _ => None,
            })
            .collect();

        // A general TEMPORAL capability covers both temporal directions. A
        // list of address capabilities covers them only when every address is
        // statically known and explicitly named.
        let oracle_covered = !self.oracle
            || temporal
            || (!self.unknown_oracle_address
                && !self.read_addresses.is_empty()
                && self.read_addresses.is_subset(&declared_reads));
        let prophecy_covered = !self.prophecy
            || temporal
            || (!self.unknown_prophecy_address
                && !self.write_addresses.is_empty()
                && self.write_addresses.is_subset(&declared_writes));

        // PRESENT/array operations do not yet retain enough address identity
        // to validate the numeric argument; require the correct capability
        // category and keep that limitation explicit until located HIR does.
        let reads_covered = !self.reads || !declared_reads.is_empty();
        let writes_covered = !self.writes || !declared_writes.is_empty();

        oracle_covered
            && prophecy_covered
            && reads_covered
            && writes_covered
            && (!self.io
                || declared
                    .iter()
                    .any(|effect| matches!(effect, AstEffect::IO)))
            && (!self.alloc
                || declared
                    .iter()
                    .any(|effect| matches!(effect, AstEffect::Alloc)))
            && (!self.ffi
                || declared
                    .iter()
                    .any(|effect| matches!(effect, AstEffect::FFI)))
            && (!self.file_io
                || declared
                    .iter()
                    .any(|effect| matches!(effect, AstEffect::FileIO)))
            && (!self.network
                || declared
                    .iter()
                    .any(|effect| matches!(effect, AstEffect::Network)))
            && (!self.system
                || declared
                    .iter()
                    .any(|effect| matches!(effect, AstEffect::System)))
    }

    /// Convert to a simple computed effect for compatibility.
    pub fn to_computed_effect(&self) -> ComputedEffect {
        if self.io || self.alloc || self.ffi || self.file_io || self.network || self.system {
            ComputedEffect::IO
        } else if self.oracle || self.prophecy {
            ComputedEffect::Temporal
        } else if self.writes {
            ComputedEffect::Writes
        } else if self.reads {
            ComputedEffect::Reads
        } else {
            ComputedEffect::Pure
        }
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        if self.is_pure() {
            return "pure".to_string();
        }

        let mut parts = Vec::new();
        if self.oracle {
            parts.push("oracle");
        }
        if self.prophecy {
            parts.push("prophecy");
        }
        if self.reads {
            parts.push("reads");
        }
        if self.writes {
            parts.push("writes");
        }
        if self.io {
            parts.push("io");
        }
        if self.alloc {
            parts.push("alloc");
        }
        if self.ffi {
            parts.push("ffi");
        }
        if self.file_io {
            parts.push("file-io");
        }
        if self.network {
            parts.push("network");
        }
        if self.system {
            parts.push("system");
        }
        parts.join(", ")
    }
}

/// Kind of type error for categorization and handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeErrorKind {
    /// Stack underflow during operation.
    StackUnderflow,
    /// Type mismatch in operation.
    TypeMismatch,
    /// Linear value illegally duplicated (DUP, OVER, PICK on ORACLE result).
    LinearityViolation,
    /// Effect declaration doesn't match actual effects.
    EffectViolation,
    /// Temporal type error (e.g., temporal address in ORACLE).
    TemporalError,
    /// Unknown or undefined identifier.
    UndefinedIdentifier,
    /// Other/generic error.
    Other,
}

impl TypeErrorKind {
    /// Get a human-readable name for this error kind.
    pub fn name(&self) -> &'static str {
        match self {
            TypeErrorKind::StackUnderflow => "stack underflow",
            TypeErrorKind::TypeMismatch => "type mismatch",
            TypeErrorKind::LinearityViolation => "linearity violation",
            TypeErrorKind::EffectViolation => "effect violation",
            TypeErrorKind::TemporalError => "temporal error",
            TypeErrorKind::UndefinedIdentifier => "undefined identifier",
            TypeErrorKind::Other => "error",
        }
    }

    /// Check if this error is a linearity violation.
    pub fn is_linearity(&self) -> bool {
        matches!(self, TypeErrorKind::LinearityViolation)
    }
}

/// Type error during checking with full source location support.
#[derive(Debug, Clone)]
pub struct TypeError {
    /// Description of the error.
    pub message: String,
    /// Kind of error for categorization.
    pub kind: TypeErrorKind,
    /// Location hint (statement index).
    pub location: Option<usize>,
    /// Source span (line, column) if available.
    pub line: Option<usize>,
    pub column: Option<usize>,
    /// Optional help text.
    pub help: Option<String>,
    /// Optional note.
    pub note: Option<String>,
}

impl TypeError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: TypeErrorKind::Other,
            location: None,
            line: None,
            column: None,
            help: None,
            note: None,
        }
    }

    /// Create a linearity violation error.
    pub fn linearity(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: TypeErrorKind::LinearityViolation,
            location: None,
            line: None,
            column: None,
            help: Some("Linear values from ORACLE can only be used once. Use SWAP/ROT to move them without duplication.".to_string()),
            note: None,
        }
    }

    /// Create a stack underflow error.
    pub fn stack_underflow(operation: &str, required: usize, available: usize) -> Self {
        Self {
            message: format!(
                "{} requires {} element(s) but stack has {}",
                operation, required, available
            ),
            kind: TypeErrorKind::StackUnderflow,
            location: None,
            line: None,
            column: None,
            help: None,
            note: None,
        }
    }

    pub fn with_kind(mut self, kind: TypeErrorKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn with_location(mut self, loc: usize) -> Self {
        self.location = Some(loc);
        self
    }

    pub fn with_line_col(mut self, line: usize, column: usize) -> Self {
        self.line = Some(line);
        self.column = Some(column);
        self
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Format: "error[E0001]: message at line:col"
        write!(f, "error[{}]", self.kind.name())?;

        if let (Some(line), Some(col)) = (self.line, self.column) {
            write!(f, " at {}:{}", line, col)?;
        } else if let Some(loc) = self.location {
            write!(f, " at stmt {}", loc)?;
        }

        write!(f, ": {}", self.message)?;

        if let Some(help) = &self.help {
            write!(f, "\n  = help: {}", help)?;
        }
        if let Some(note) = &self.note {
            write!(f, "\n  = note: {}", note)?;
        }
        Ok(())
    }
}

impl std::error::Error for TypeError {}

/// Result of type checking.
#[derive(Debug, Clone)]
pub struct TypeCheckResult {
    /// Whether the program is well-typed.
    pub is_valid: bool,
    /// Any type errors found.
    pub errors: Vec<TypeError>,
    /// Warnings (non-fatal issues).
    pub warnings: Vec<String>,
    /// Inferred types for memory cells (only for statically known addresses).
    pub cell_types: HashMap<Address, TemporalType>,
    /// PROPHECY writes whose address is computed at runtime and therefore
    /// not represented in cell_types.
    pub unknown_prophecy_writes: usize,
    /// Stack type at end of program.
    pub final_stack_types: Vec<StackType>,
    /// Inferred effects for the program.
    pub effects: EffectSet,
    /// Effect violations found (if any).
    pub effect_violations: Vec<EffectViolation>,
    /// Linearity violations found (if any).
    pub linear_violations: Vec<LinearityViolation>,
}

/// An effect violation: declared effect doesn't match actual effect.
#[derive(Debug, Clone)]
pub struct EffectViolation {
    /// Name of the procedure with the violation.
    pub procedure_name: String,
    /// The declared effects.
    pub declared: Vec<AstEffect>,
    /// The computed actual effects.
    pub actual: EffectSet,
    /// Location hint.
    pub location: Option<usize>,
}

impl TypeCheckResult {
    pub fn success() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            cell_types: HashMap::new(),
            unknown_prophecy_writes: 0,
            final_stack_types: Vec::new(),
            effects: EffectSet::pure(),
            effect_violations: Vec::new(),
            linear_violations: Vec::new(),
        }
    }

    pub fn with_error(error: TypeError) -> Self {
        Self {
            is_valid: false,
            errors: vec![error],
            warnings: Vec::new(),
            cell_types: HashMap::new(),
            unknown_prophecy_writes: 0,
            final_stack_types: Vec::new(),
            effects: EffectSet::pure(),
            effect_violations: Vec::new(),
            linear_violations: Vec::new(),
        }
    }

    /// Check if there are effect violations.
    pub fn has_effect_violations(&self) -> bool {
        !self.effect_violations.is_empty()
    }

    /// Check if there are linearity violations.
    pub fn has_linear_violations(&self) -> bool {
        !self.linear_violations.is_empty()
    }
}

/// Type checker for OUROCHRONOS programs.
pub struct TypeChecker {
    /// Abstract stack of types (with linearity tracking).
    stack: Vec<StackType>,
    /// Types of memory cells (from PROPHECY writes).
    cell_types: HashMap<Address, TemporalType>,
    /// Collected errors.
    errors: Vec<TypeError>,
    /// Collected warnings.
    warnings: Vec<String>,
    /// Current statement index.
    stmt_index: usize,
    /// Whether we're in a temporal-controlled branch.
    in_temporal_branch: bool,
    /// Literal pushed by the immediately preceding statement, if any. This
    /// is the peephole that recovers statically known addresses for the
    /// dominant `addr ORACLE` / `value addr PROPHECY` idiom.
    last_pushed_literal: Option<u64>,
    /// PROPHECY writes whose address could not be determined statically.
    unknown_prophecy_writes: usize,
    /// Accumulated effects for the current scope.
    current_effects: EffectSet,
    /// Effect violations found.
    effect_violations: Vec<EffectViolation>,
    /// Linear type violations found.
    linear_violations: Vec<LinearityViolation>,
    /// Whether to enforce strict linearity (if false, only warn).
    strict_linearity: bool,
}

/// A linearity violation detected during type checking.
///
/// Linear types (from ORACLE) can only be used once. Attempting to duplicate
/// them via DUP, OVER, or PICK is a compile-time error in strict mode.
#[derive(Debug, Clone)]
pub struct LinearityViolation {
    /// The operation that caused the violation.
    pub operation: String,
    /// Description of what went wrong.
    pub message: String,
    /// Statement index where violation occurred.
    pub stmt_index: usize,
    /// The type that was illegally duplicated.
    pub violating_type: StackType,
    /// Stack position of the linear value (for OVER/PICK).
    pub stack_position: Option<usize>,
}

impl LinearityViolation {
    /// Convert this violation to a TypeError for unified error handling.
    pub fn to_type_error(&self) -> TypeError {
        TypeError::linearity(&self.message)
            .with_location(self.stmt_index)
            .with_note(format!(
                "The {} operation attempted to duplicate a linear value (type: {})",
                self.operation,
                self.violating_type.describe()
            ))
    }
}

impl std::fmt::Display for LinearityViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "linearity violation at stmt {}: {} - {}",
            self.stmt_index, self.operation, self.message
        )
    }
}

impl TypeChecker {
    /// Create a new type checker.
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            cell_types: HashMap::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
            stmt_index: 0,
            in_temporal_branch: false,
            last_pushed_literal: None,
            unknown_prophecy_writes: 0,
            current_effects: EffectSet::pure(),
            effect_violations: Vec::new(),
            linear_violations: Vec::new(),
            strict_linearity: true,
        }
    }

    /// Type-check a program.
    pub fn check(&mut self, program: &Program) -> TypeCheckResult {
        // First, check all procedures for effect compliance
        for proc in &program.procedures {
            self.check_procedure(proc);
        }

        // Then check the main body
        self.check_statements(&program.body);

        TypeCheckResult {
            is_valid: self.errors.is_empty()
                && self.effect_violations.is_empty()
                && (self.linear_violations.is_empty() || !self.strict_linearity),
            errors: self.errors.clone(),
            warnings: self.warnings.clone(),
            cell_types: self.cell_types.clone(),
            unknown_prophecy_writes: self.unknown_prophecy_writes,
            final_stack_types: self.stack.clone(),
            effects: self.current_effects.clone(),
            effect_violations: self.effect_violations.clone(),
            linear_violations: self.linear_violations.clone(),
        }
    }

    /// Enable or disable strict linearity checking.
    pub fn set_strict_linearity(&mut self, strict: bool) {
        self.strict_linearity = strict;
    }

    /// Check a procedure's direct body for effect compliance.
    ///
    /// Calls are intentionally opaque in this migration stage: this does not
    /// yet compute transitive callee effects or validate stack signatures.
    /// Resolved HIR will make both analyses mandatory without name guessing.
    fn check_procedure(&mut self, proc: &Procedure) {
        // Save current state
        let saved_effects = std::mem::take(&mut self.current_effects);
        let saved_stack = std::mem::take(&mut self.stack);

        // Check procedure body
        self.check_statements(&proc.body);

        // Verify declared effects cover actual effects
        if !proc.effects.is_empty() && !self.current_effects.is_covered_by(&proc.effects) {
            self.effect_violations.push(EffectViolation {
                procedure_name: proc.name.clone(),
                declared: proc.effects.clone(),
                actual: self.current_effects.clone(),
                location: Some(self.stmt_index),
            });
        }

        // Restore state
        self.current_effects = saved_effects.join(&self.current_effects);
        self.stack = saved_stack;
    }

    /// Check a block of statements.
    fn check_statements(&mut self, stmts: &[Stmt]) {
        for stmt in stmts {
            self.check_stmt(stmt);
            self.stmt_index += 1;
        }
    }

    /// Check a single statement.
    fn check_stmt(&mut self, stmt: &Stmt) {
        // Sound only for one statement of lookback: any statement other than
        // Push leaves the top of stack unpredictable, so the literal is
        // cleared before dispatch and re-established only by Push.
        let addr_literal = self.last_pushed_literal.take();
        match stmt {
            Stmt::Push(value) => {
                // Constants are always pure and unrestricted
                let ty = if value.prov.is_pure() {
                    StackType::PURE
                } else {
                    StackType::TEMPORAL
                };
                self.stack.push(ty);
                self.last_pushed_literal = Some(value.val);
            }

            Stmt::PushConstant { value, .. } => {
                self.stack.push(StackType::PURE);
                self.last_pushed_literal = Some(*value);
            }

            Stmt::ReadTemporal { .. } => {
                self.stack.push(StackType::TEMPORAL);
                self.current_effects.oracle = true;
            }

            Stmt::PushQuote(_) => {
                // Quotation references are pure, unrestricted stack values at
                // runtime, but are not word literals and therefore must not be
                // reused by the literal-address peephole.
                self.stack.push(StackType::PURE);
            }

            Stmt::Op(op) => {
                self.check_op(*op, addr_literal);
            }

            Stmt::If {
                then_branch,
                else_branch,
            } => {
                // Pop condition
                let cond_type = self.pop_type();

                // If condition is temporal, both branches are tainted
                let was_temporal = self.in_temporal_branch;
                if cond_type.temporal == TemporalType::Temporal {
                    self.in_temporal_branch = true;
                }

                // Check branches
                let stack_before = self.stack.clone();
                self.check_statements(then_branch);
                let then_stack = self.stack.clone();

                self.stack = stack_before;
                if let Some(else_stmts) = else_branch {
                    self.check_statements(else_stmts);
                }
                let else_stack = self.stack.clone();

                // Merge branch stacks (join types)
                self.stack = self.merge_stacks(&then_stack, &else_stack);

                self.in_temporal_branch = was_temporal;
            }

            Stmt::While { cond, body } => {
                // Check condition
                let stack_before = self.stack.clone();
                self.check_statements(cond);
                let cond_type = self.pop_type();

                // Loops with temporal conditions are especially tricky
                if cond_type.temporal == TemporalType::Temporal {
                    self.warnings.push(format!(
                        "Loop condition is Temporal at stmt {}: loop behavior may depend on future",
                        self.stmt_index
                    ));
                    self.in_temporal_branch = true;
                }

                // Check body
                self.check_statements(body);

                // Restore stack (approximate: assume loop terminates)
                self.stack = stack_before;
            }

            Stmt::Block(stmts) => {
                self.check_statements(stmts);
            }

            Stmt::Call { name: _ } => {
                // Direct-body checking deliberately does not infer transitive
                // callee effects yet; see check_procedure's contract caveat.
                self.stack.push(StackType::UNKNOWN);
            }

            Stmt::TemporalScope { body, .. } => {
                // Temporal scope: check the body
                // Memory isolation doesn't affect type checking at this level
                self.check_statements(body);
            }
        }
    }

    /// Check an opcode.
    fn check_op(&mut self, op: OpCode, addr_literal: Option<u64>) {
        match op {
            // ===== Temporal Operations =====
            OpCode::Oracle => {
                let addr_type = self.pop_type();
                if addr_type.temporal == TemporalType::Temporal {
                    self.warnings.push(format!(
                        "ORACLE address is Temporal at stmt {}: address depends on future",
                        self.stmt_index
                    ));
                }
                // Temporal information is duplicable. This matches both
                // runtimes, the verifier, and the language's flagship
                // examples; temporal taint is not an affine resource.
                self.stack.push(StackType::TEMPORAL);

                // Track effect: ORACLE is a temporal read
                self.current_effects.oracle = true;
                if let Some(address) = addr_literal {
                    self.current_effects
                        .read_addresses
                        .insert(address as Address);
                } else {
                    self.current_effects.unknown_oracle_address = true;
                }
            }

            OpCode::Prophecy => {
                let addr_type = self.pop_type();
                let value_type = self.pop_type();

                if addr_type.temporal == TemporalType::Temporal {
                    self.warnings.push(format!(
                        "PROPHECY address is Temporal at stmt {}: writing to future-dependent address",
                        self.stmt_index
                    ));
                }

                // Record the cell type only when the address is statically
                // known; fabricating an entry at address 0 would make every
                // multi-address programme's cell map wrong.
                match addr_literal {
                    Some(addr) => {
                        self.cell_types.insert(addr as Address, value_type.temporal);
                    }
                    _ => {
                        self.unknown_prophecy_writes += 1;
                    }
                }

                // Track effect: PROPHECY is a temporal write
                self.current_effects.prophecy = true;
                if let Some(address) = addr_literal {
                    self.current_effects
                        .write_addresses
                        .insert(address as Address);
                } else {
                    self.current_effects.unknown_prophecy_address = true;
                }
            }

            OpCode::PresentRead => {
                let addr_type = self.pop_type();
                // Reading from present can be temporal if written with temporal value
                let result = if self.in_temporal_branch {
                    StackType::TEMPORAL
                } else {
                    StackType::from_temporal(addr_type.temporal) // Approximate: same as address type
                };
                self.stack.push(result);

                // Track effect: PRESENT is a memory read
                self.current_effects.reads = true;
            }

            // ===== Stack Manipulation (with linearity checks) =====
            OpCode::Pop => {
                self.pop_type();
            }

            OpCode::Dup => {
                // Check linearity before duplicating
                self.check_can_duplicate("DUP");
                let ty = self.peek_type();
                // If duplicating, result is unrestricted (consumed linearity)
                self.stack.push(ty.consume());
            }

            OpCode::Swap => {
                // Swap doesn't duplicate, so no linearity check needed
                if self.stack.len() >= 2 {
                    let len = self.stack.len();
                    self.stack.swap(len - 1, len - 2);
                }
            }

            OpCode::Over => {
                // OVER duplicates the second element
                self.check_can_duplicate_at(1, "OVER");
                if self.stack.len() >= 2 {
                    let ty = self.stack[self.stack.len() - 2];
                    // If duplicating, result is unrestricted
                    self.stack.push(ty.consume());
                }
            }

            OpCode::Rot => {
                // ROT doesn't duplicate, just rotates
                if self.stack.len() >= 3 {
                    let len = self.stack.len();
                    let a = self.stack[len - 3];
                    self.stack[len - 3] = self.stack[len - 2];
                    self.stack[len - 2] = self.stack[len - 1];
                    self.stack[len - 1] = a;
                }
            }

            OpCode::Depth => {
                // Depth is always pure (it's a constant at that point)
                self.stack.push(StackType::PURE);
            }

            OpCode::Pick => {
                let _ = self.pop_type(); // index
                                         // PICK duplicates a value at a computed index.
                                         // Since we can't statically determine which element will be picked,
                                         // we must conservatively reject if ANY stack element is linear.
                                         // This is strict but sound - it prevents potential linear duplication.
                self.check_stack_has_no_linear("PICK");
                // Result type is unknown but unrestricted (we've validated no linear)
                self.stack.push(StackType::UNKNOWN);
            }

            OpCode::Nop => {
                // No effect
            }

            OpCode::Neg | OpCode::Abs | OpCode::Sign => {
                // Unary operations: pop, transform, push
                // These consume the linear value (if any)
                let a = self.pop_type();
                self.stack.push(a.consume()); // Consuming makes it unrestricted
            }

            // ===== Arithmetic (joins types, consumes linearity) =====
            OpCode::Add
            | OpCode::Sub
            | OpCode::Mul
            | OpCode::Div
            | OpCode::Mod
            | OpCode::Min
            | OpCode::Max => {
                let b = self.pop_type();
                let a = self.pop_type();
                // Arithmetic consumes linear values, result is unrestricted
                self.stack.push(a.join(b).consume());
            }

            // ===== Bitwise (joins types, consumes linearity) =====
            OpCode::And | OpCode::Or | OpCode::Xor | OpCode::Shl | OpCode::Shr => {
                let b = self.pop_type();
                let a = self.pop_type();
                self.stack.push(a.join(b).consume());
            }

            OpCode::Not => {
                // Unary: consumes and produces unrestricted
                let a = self.pop_type();
                self.stack.push(a.consume());
            }

            // ===== Comparison (joins types, consumes linearity) =====
            OpCode::Eq
            | OpCode::Neq
            | OpCode::Lt
            | OpCode::Gt
            | OpCode::Lte
            | OpCode::Gte
            | OpCode::Slt
            | OpCode::Sgt
            | OpCode::Slte
            | OpCode::Sgte => {
                let b = self.pop_type();
                let a = self.pop_type();
                self.stack.push(a.join(b).consume());
            }

            // ===== I/O =====
            OpCode::Input => {
                // Input is pure (external, not from oracle)
                self.stack.push(StackType::PURE);

                // Track effect: INPUT is an I/O effect
                self.current_effects.io = true;
            }

            OpCode::Output | OpCode::Emit => {
                let ty = self.pop_type();
                if ty.temporal == TemporalType::Temporal && !self.in_temporal_branch {
                    // Outputting temporal value: this is significant
                    // The output depends on the fixed point
                }

                // Track effect: OUTPUT/EMIT are I/O effects
                self.current_effects.io = true;
            }

            // ===== Control =====
            OpCode::Halt | OpCode::Paradox => {
                // No stack effect
            }

            // ===== Arrays =====
            OpCode::Pack => {
                // Consumes n+2 values, produces none
                self.pop_type(); // n
                self.pop_type(); // base
                                 // Additional values popped dynamically

                // Track effect: PACK writes to memory
                self.current_effects.writes = true;
            }
            OpCode::Unpack => {
                // Consumes 2, produces n values - type is unknown
                self.pop_type(); // n
                self.pop_type(); // base
                self.stack.push(StackType::UNKNOWN);

                // Track effect: UNPACK reads from memory
                self.current_effects.reads = true;
            }
            OpCode::Index => {
                // Read from memory - result may be temporal
                let _index_type = self.pop_type();
                let _base_type = self.pop_type();
                self.stack.push(StackType::TEMPORAL);

                // Track effect: INDEX reads from memory
                self.current_effects.reads = true;
            }
            OpCode::Store => {
                // Write to memory
                self.pop_type(); // index
                self.pop_type(); // base
                self.pop_type(); // value

                // Track effect: STORE writes to memory
                self.current_effects.writes = true;
            }

            // ===== Vector Operations =====
            OpCode::VecNew => {
                // Creates a new vector handle
                self.stack.push(StackType::PURE);
                self.current_effects.alloc = true;
            }
            OpCode::VecPush => {
                self.pop_type(); // value
                let handle = self.pop_type(); // handle
                self.stack.push(handle);
                self.current_effects.alloc = true;
            }
            OpCode::VecPop => {
                let handle = self.pop_type();
                self.stack.push(handle);
                self.stack.push(StackType::UNKNOWN); // Value could be temporal if stored
                self.current_effects.alloc = true;
            }
            OpCode::VecGet => {
                self.pop_type(); // index
                let handle = self.pop_type();
                self.stack.push(handle);
                self.stack.push(StackType::UNKNOWN); // Value type unknown
                self.current_effects.alloc = true;
            }
            OpCode::VecSet => {
                self.pop_type(); // index
                self.pop_type(); // value
                let handle = self.pop_type();
                self.stack.push(handle);
                self.current_effects.alloc = true;
            }
            OpCode::VecLen => {
                let handle = self.pop_type();
                self.stack.push(handle);
                self.stack.push(StackType::PURE);
                self.current_effects.alloc = true;
            }

            // ===== Hash Table Operations =====
            OpCode::HashNew => {
                self.stack.push(StackType::PURE);
                self.current_effects.alloc = true;
            }
            OpCode::HashPut => {
                self.pop_type(); // value
                self.pop_type(); // key
                let handle = self.pop_type();
                self.stack.push(handle);
                self.current_effects.alloc = true;
            }
            OpCode::HashGet => {
                self.pop_type(); // key
                let handle = self.pop_type();
                self.stack.push(handle);
                self.stack.push(StackType::UNKNOWN); // value
                self.stack.push(StackType::PURE); // found flag
                self.current_effects.alloc = true;
            }
            OpCode::HashDel => {
                self.pop_type(); // key
                let handle = self.pop_type();
                self.stack.push(handle);
                self.current_effects.alloc = true;
            }
            OpCode::HashHas => {
                self.pop_type(); // key
                let handle = self.pop_type();
                self.stack.push(handle);
                self.stack.push(StackType::PURE); // found flag
                self.current_effects.alloc = true;
            }
            OpCode::HashLen => {
                let handle = self.pop_type();
                self.stack.push(handle);
                self.stack.push(StackType::PURE);
                self.current_effects.alloc = true;
            }

            // ===== Set Operations =====
            OpCode::SetNew => {
                self.stack.push(StackType::PURE);
                self.current_effects.alloc = true;
            }
            OpCode::SetAdd => {
                self.pop_type(); // value
                let handle = self.pop_type();
                self.stack.push(handle);
                self.current_effects.alloc = true;
            }
            OpCode::SetHas => {
                self.pop_type(); // value
                let handle = self.pop_type();
                self.stack.push(handle);
                self.stack.push(StackType::PURE); // found flag
                self.current_effects.alloc = true;
            }
            OpCode::SetDel => {
                self.pop_type(); // value
                let handle = self.pop_type();
                self.stack.push(handle);
                self.current_effects.alloc = true;
            }
            OpCode::SetLen => {
                let handle = self.pop_type();
                self.stack.push(handle);
                self.stack.push(StackType::PURE);
                self.current_effects.alloc = true;
            }

            // Dynamic stack structure changes cannot be statically typed easily
            OpCode::Roll
            | OpCode::Reverse
            | OpCode::StrRev
            | OpCode::StrCat
            | OpCode::StrSplit
            | OpCode::Assert
            | OpCode::Exec
            | OpCode::Dip
            | OpCode::Keep
            | OpCode::Bi
            | OpCode::Rec => {
                // Conservative approach: do nothing or invalidate stack?
                // For this prototype, we ignore their effect on type stack structure
            }

            // ===== FFI Operations =====
            OpCode::FFICall | OpCode::FFICallNamed => {
                // FFI calls have an opaque foreign effect and may modify stack.
                // We can't statically determine the stack effect
                self.pop_type(); // FFI ID or name handle
                self.current_effects.ffi = true;
            }

            // ===== File I/O Operations =====
            OpCode::FileOpen => {
                self.pop_type(); // mode
                self.pop_type(); // path_handle
                self.stack.push(StackType::PURE); // file_handle
                self.current_effects.file_io = true;
            }
            OpCode::FileRead => {
                self.pop_type(); // max_bytes
                let handle = self.pop_type();
                self.stack.push(handle); // file_handle
                self.stack.push(StackType::PURE); // buffer_handle
                self.stack.push(StackType::PURE); // bytes_read
                self.current_effects.file_io = true;
            }
            OpCode::FileWrite => {
                self.pop_type(); // buffer_handle
                let handle = self.pop_type();
                self.stack.push(handle); // file_handle
                self.stack.push(StackType::PURE); // bytes_written
                self.current_effects.file_io = true;
            }
            OpCode::FileSeek => {
                self.pop_type(); // origin
                self.pop_type(); // offset
                let handle = self.pop_type();
                self.stack.push(handle); // file_handle
                self.stack.push(StackType::PURE); // new_position
                self.current_effects.file_io = true;
            }
            OpCode::FileFlush => {
                let handle = self.pop_type();
                self.stack.push(handle);
                self.current_effects.file_io = true;
            }
            OpCode::FileClose => {
                self.pop_type(); // file_handle
                self.current_effects.file_io = true;
            }
            OpCode::FileExists => {
                self.pop_type(); // path_handle
                self.stack.push(StackType::PURE); // exists flag
                self.current_effects.file_io = true;
            }
            OpCode::FileSize => {
                self.pop_type(); // path_handle
                self.stack.push(StackType::PURE); // size
                self.current_effects.file_io = true;
            }

            // ===== Buffer Operations =====
            OpCode::BufferNew => {
                self.pop_type(); // capacity
                self.stack.push(StackType::PURE); // buffer_handle
                self.current_effects.alloc = true;
            }
            OpCode::BufferFromStack => {
                // Variable number of pops, just pop the count
                self.pop_type(); // n
                                 // Would need to pop n more values, but we can't know n statically
                self.stack.push(StackType::PURE); // buffer_handle
                self.current_effects.alloc = true;
            }
            OpCode::BufferToStack => {
                self.pop_type(); // buffer_handle
                                 // Would push variable number of values
                self.stack.push(StackType::PURE); // n
                self.current_effects.alloc = true;
            }
            OpCode::BufferLen => {
                let handle = self.pop_type();
                self.stack.push(handle); // buffer_handle
                self.stack.push(StackType::PURE); // length
                self.current_effects.alloc = true;
            }
            OpCode::BufferReadByte => {
                let handle = self.pop_type();
                self.stack.push(handle); // buffer_handle
                self.stack.push(StackType::PURE); // byte
                self.current_effects.alloc = true;
            }
            OpCode::BufferWriteByte => {
                self.pop_type(); // byte
                let handle = self.pop_type();
                self.stack.push(handle); // buffer_handle
                self.current_effects.alloc = true;
            }
            OpCode::BufferFree => {
                self.pop_type(); // buffer_handle
                self.current_effects.alloc = true;
            }

            // ===== Network Operations =====
            OpCode::TcpConnect => {
                self.pop_type(); // port
                self.pop_type(); // host_handle
                self.stack.push(StackType::PURE); // socket_handle
                self.current_effects.network = true;
            }
            OpCode::SocketSend => {
                self.pop_type(); // buffer_handle
                let handle = self.pop_type();
                self.stack.push(handle); // socket_handle
                self.stack.push(StackType::PURE); // bytes_sent
                self.current_effects.network = true;
            }
            OpCode::SocketRecv => {
                self.pop_type(); // max_bytes
                let handle = self.pop_type();
                self.stack.push(handle); // socket_handle
                self.stack.push(StackType::PURE); // buffer_handle
                self.stack.push(StackType::PURE); // bytes_recv
                self.current_effects.network = true;
            }
            OpCode::SocketClose => {
                self.pop_type(); // socket_handle
                self.current_effects.network = true;
            }

            // ===== Process Operations =====
            OpCode::ProcExec => {
                self.pop_type(); // command_handle
                self.stack.push(StackType::PURE); // output_handle
                self.stack.push(StackType::PURE); // exit_code
                self.current_effects.system = true;
            }

            // ===== System Operations =====
            OpCode::Clock => {
                self.stack.push(StackType::PURE); // timestamp
                self.current_effects.system = true;
            }
            OpCode::Sleep => {
                self.pop_type(); // milliseconds
                self.current_effects.system = true;
            }
            OpCode::Random => {
                self.stack.push(StackType::PURE); // random_value
                self.current_effects.system = true;
            }
        }
    }

    /// Pop a type from the abstract stack (consuming it).
    fn pop_type(&mut self) -> StackType {
        self.stack.pop().unwrap_or(StackType::UNKNOWN)
    }

    /// Peek at the top type.
    fn peek_type(&self) -> StackType {
        self.stack.last().copied().unwrap_or(StackType::UNKNOWN)
    }

    /// Check if the top of stack can be duplicated (not linear).
    ///
    /// In strict mode (default), duplicating a linear value is a compile error.
    /// Linear values come from ORACLE and can only be consumed once.
    fn check_can_duplicate(&mut self, operation: &str) -> bool {
        let top = self.peek_type();
        if top.is_linear() {
            let violation = LinearityViolation {
                operation: operation.to_string(),
                message: format!(
                    "Cannot {} linear value (type: {}). Linear values from ORACLE can only be used once.",
                    operation.to_lowercase(),
                    top.describe()
                ),
                stmt_index: self.stmt_index,
                violating_type: top,
                stack_position: Some(0),
            };
            if self.strict_linearity {
                self.linear_violations.push(violation);
            } else {
                self.warnings.push(format!(
                    "Warning at stmt {}: {} of linear value",
                    self.stmt_index, operation
                ));
            }
            false
        } else {
            true
        }
    }

    /// Check if a specific stack position can be duplicated.
    ///
    /// Used for OVER (index 1) and PICK (dynamic index).
    fn check_can_duplicate_at(&mut self, index: usize, operation: &str) -> bool {
        let stack_idx = self.stack.len().saturating_sub(1 + index);
        if let Some(st) = self.stack.get(stack_idx).copied() {
            if st.is_linear() {
                let violation = LinearityViolation {
                    operation: operation.to_string(),
                    message: format!(
                        "Cannot {} linear value at index {} (type: {}). Linear values can only be used once.",
                        operation.to_lowercase(),
                        index,
                        st.describe()
                    ),
                    stmt_index: self.stmt_index,
                    violating_type: st,
                    stack_position: Some(index),
                };
                if self.strict_linearity {
                    self.linear_violations.push(violation);
                } else {
                    self.warnings.push(format!(
                        "Warning at stmt {}: {} of linear value at index {}",
                        self.stmt_index, operation, index
                    ));
                }
                return false;
            }
        }
        true
    }

    /// Check if any element on the stack is linear (for PICK with dynamic index).
    ///
    /// Returns true if all elements are safe to duplicate, false if any are linear.
    fn check_stack_has_no_linear(&mut self, operation: &str) -> bool {
        for (i, st) in self.stack.iter().enumerate().rev() {
            if st.is_linear() {
                let violation = LinearityViolation {
                    operation: operation.to_string(),
                    message: format!(
                        "{} may duplicate linear value at position {} (type: {}). \
                         Cannot statically verify safety with dynamic index.",
                        operation,
                        self.stack.len() - 1 - i,
                        st.describe()
                    ),
                    stmt_index: self.stmt_index,
                    violating_type: *st,
                    stack_position: Some(self.stack.len() - 1 - i),
                };
                if self.strict_linearity {
                    self.linear_violations.push(violation);
                } else {
                    self.warnings.push(format!(
                        "Warning at stmt {}: {} may duplicate linear value at position {}",
                        self.stmt_index,
                        operation,
                        self.stack.len() - 1 - i
                    ));
                }
                return false;
            }
        }
        true
    }

    /// Merge two stacks from branches (join corresponding types).
    fn merge_stacks(&self, a: &[StackType], b: &[StackType]) -> Vec<StackType> {
        let max_len = a.len().max(b.len());
        let mut result = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let ta = a.get(i).copied().unwrap_or(StackType::UNKNOWN);
            let tb = b.get(i).copied().unwrap_or(StackType::UNKNOWN);
            result.push(ta.join(tb));
        }

        result
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Type-check a program and return the result.
pub fn type_check(program: &Program) -> TypeCheckResult {
    let mut checker = TypeChecker::new();
    checker.check(program)
}

/// Display type information for a program.
pub fn display_types(result: &TypeCheckResult) -> String {
    let mut output = String::new();

    output.push_str("=== Type Check Result ===\n");
    output.push_str(&format!("Valid: {}\n", result.is_valid));

    if !result.errors.is_empty() {
        output.push_str("\nErrors:\n");
        for err in &result.errors {
            output.push_str(&format!("  - {}\n", err));
        }
    }

    if !result.warnings.is_empty() {
        output.push_str("\nWarnings:\n");
        for warn in &result.warnings {
            output.push_str(&format!("  - {}\n", warn));
        }
    }

    if !result.cell_types.is_empty() {
        output.push_str("\nCell Types:\n");
        for (addr, ty) in &result.cell_types {
            output.push_str(&format!("  [{}]: {}\n", addr, ty.name()));
        }
    }

    if result.unknown_prophecy_writes > 0 {
        output.push_str(&format!(
            "\n{} PROPHECY write(s) to runtime-computed addresses (not typed)\n",
            result.unknown_prophecy_writes
        ));
    }

    if !result.final_stack_types.is_empty() {
        output.push_str("\nFinal Stack Types:\n");
        for (i, ty) in result.final_stack_types.iter().enumerate() {
            output.push_str(&format!("  {}: {}\n", i, ty.describe()));
        }
    }

    if !result.linear_violations.is_empty() {
        output.push_str("\nLinearity Violations:\n");
        for violation in &result.linear_violations {
            output.push_str(&format!(
                "  - [stmt {}] {}: {}\n",
                violation.stmt_index, violation.operation, violation.message
            ));
        }
    }

    if !result.effect_violations.is_empty() {
        output.push_str("\nEffect Violations:\n");
        for violation in &result.effect_violations {
            output.push_str(&format!(
                "  - procedure '{}': declared {:?}, inferred {}\n",
                violation.procedure_name,
                violation.declared,
                violation.actual.summary()
            ));
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn check(source: &str) -> TypeCheckResult {
        let program = parse(source).expect("Parse failed");
        type_check(&program)
    }

    #[test]
    fn prophecy_cell_type_uses_the_literal_address() {
        // `value addr PROPHECY` with a literal address types that cell, not
        // the formerly hardcoded address 0.
        let result = check("42 5 PROPHECY");
        assert!(result.cell_types.contains_key(&5));
        assert!(!result.cell_types.contains_key(&0));
        assert_eq!(result.unknown_prophecy_writes, 0);
    }

    #[test]
    fn prophecy_cell_types_support_configurable_wide_addresses() {
        let program = crate::parser::parse("1 65536 PROPHECY").unwrap();
        let result = type_check(&program);
        assert_eq!(result.cell_types.get(&65_536), Some(&TemporalType::Pure));
        assert_eq!(result.unknown_prophecy_writes, 0);
    }

    #[test]
    fn prophecy_with_computed_address_is_recorded_as_unknown() {
        // The address 3+4 is computed at runtime; no cell entry may be
        // fabricated for it.
        let result = check("42 3 4 ADD PROPHECY");
        assert!(result.cell_types.is_empty());
        assert_eq!(result.unknown_prophecy_writes, 1);
    }

    #[test]
    fn test_pure_program() {
        let result = check("10 20 ADD OUTPUT");
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
        // Final stack should be empty (OUTPUT consumes)
    }

    #[test]
    fn test_temporal_from_oracle() {
        let result = check("0 ORACLE 0 PROPHECY");
        assert!(result.is_valid);
        // Should have no errors but might have info about temporal tainting
    }

    #[test]
    fn test_temporal_arithmetic() {
        let result = check("0 ORACLE 1 ADD 0 PROPHECY");
        assert!(result.is_valid);
        // The ADD of temporal + pure should be temporal
    }

    #[test]
    fn test_temporal_condition() {
        let result = check("0 ORACLE 0 EQ IF { 1 } ELSE { 2 } OUTPUT");
        assert!(result.is_valid);
        // Condition is temporal, so branches are tainted
    }

    #[test]
    fn test_type_lattice_join() {
        assert_eq!(
            TemporalType::Pure.join(TemporalType::Pure),
            TemporalType::Pure
        );
        assert_eq!(
            TemporalType::Pure.join(TemporalType::Temporal),
            TemporalType::Temporal
        );
        assert_eq!(
            TemporalType::Temporal.join(TemporalType::Pure),
            TemporalType::Temporal
        );
        assert_eq!(
            TemporalType::Unknown.join(TemporalType::Pure),
            TemporalType::Pure
        );
    }

    #[test]
    fn test_subtype_relation() {
        assert!(TemporalType::Pure.is_subtype_of(TemporalType::Pure));
        assert!(TemporalType::Pure.is_subtype_of(TemporalType::Temporal));
        assert!(TemporalType::Temporal.is_subtype_of(TemporalType::Temporal));
        assert!(!TemporalType::Temporal.is_subtype_of(TemporalType::Pure));
    }

    // ═══════════════════════════════════════════════════════════════════
    // Effect System Tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_effect_set_pure() {
        let effects = EffectSet::pure();
        assert!(effects.is_pure());
        assert!(!effects.is_temporal());
        assert_eq!(effects.summary(), "pure");
    }

    #[test]
    fn test_effect_set_oracle() {
        let effects = EffectSet::oracle(Some(0));
        assert!(!effects.is_pure());
        assert!(effects.is_temporal());
        assert!(effects.oracle);
    }

    #[test]
    fn test_effect_set_prophecy() {
        let effects = EffectSet::prophecy(Some(0));
        assert!(!effects.is_pure());
        assert!(effects.is_temporal());
        assert!(effects.prophecy);
    }

    #[test]
    fn test_effect_set_join() {
        let oracle = EffectSet::oracle(None);
        let prophecy = EffectSet::prophecy(None);
        let joined = oracle.join(&prophecy);

        assert!(joined.oracle);
        assert!(joined.prophecy);
        assert!(joined.is_temporal());
    }

    #[test]
    fn test_computed_effect_join() {
        use super::ComputedEffect;

        assert_eq!(
            ComputedEffect::Pure.join(ComputedEffect::Pure),
            ComputedEffect::Pure
        );
        assert_eq!(
            ComputedEffect::Pure.join(ComputedEffect::Reads),
            ComputedEffect::Reads
        );
        assert_eq!(
            ComputedEffect::Reads.join(ComputedEffect::Writes),
            ComputedEffect::IO
        );
        assert_eq!(
            ComputedEffect::Temporal.join(ComputedEffect::Pure),
            ComputedEffect::Temporal
        );
    }

    #[test]
    fn test_computed_effect_subeffect() {
        use super::ComputedEffect;

        assert!(ComputedEffect::Pure.is_subeffect_of(ComputedEffect::IO));
        assert!(ComputedEffect::Pure.is_subeffect_of(ComputedEffect::Reads));
        assert!(ComputedEffect::Reads.is_subeffect_of(ComputedEffect::Reads));
        assert!(!ComputedEffect::Reads.is_subeffect_of(ComputedEffect::Writes));
    }

    #[test]
    fn test_effect_tracking_oracle() {
        let result = check("0 ORACLE 0 PROPHECY");
        assert!(result.effects.oracle);
        assert!(result.effects.prophecy);
        assert!(result.effects.is_temporal());
    }

    #[test]
    fn test_effect_tracking_io() {
        let result = check("10 OUTPUT");
        assert!(result.effects.io);
        assert!(!result.effects.is_temporal());
    }

    #[test]
    fn test_effect_tracking_pure() {
        let result = check("10 20 ADD");
        assert!(result.effects.is_pure());
    }

    #[test]
    fn test_effect_tracking_memory() {
        // Use PRESENT which is a memory read operation
        let result = check("100 PRESENT");
        assert!(result.effects.reads);
        assert!(!result.effects.is_pure());
    }

    #[test]
    fn matching_address_contracts_cover_temporal_accesses() {
        let result = check("PROCEDURE ok READS(0) WRITES(1) { 0 ORACLE 9 1 PROPHECY } 0");
        assert!(result.effect_violations.is_empty());
        assert!(result.is_valid);
    }

    #[test]
    fn temporal_address_contracts_reject_wrong_or_runtime_addresses() {
        let wrong = check("PROCEDURE bad READS(1) { 0 ORACLE } 0");
        assert_eq!(wrong.effect_violations.len(), 1);

        let runtime = check("PROCEDURE bad READS(0) { 1 2 ADD ORACLE } 0");
        assert_eq!(runtime.effect_violations.len(), 1);

        let general = check("PROCEDURE ok TEMPORAL { 1 2 ADD ORACLE } 0");
        assert!(general.effect_violations.is_empty());
    }

    #[test]
    fn pure_and_wrong_category_contracts_do_not_cover_effects() {
        for source in [
            "PROCEDURE bad PURE { 1 OUTPUT } 0",
            "PROCEDURE bad PURE IO { 1 OUTPUT } 0",
            "PROCEDURE bad IO { 0 ORACLE } 0",
            "PROCEDURE bad FILE_IO { CLOCK } 0",
        ] {
            let result = check(source);
            assert_eq!(result.effect_violations.len(), 1, "{source}");
            assert!(!result.is_valid, "{source}");
        }
    }

    #[test]
    fn each_declared_effect_category_covers_its_direct_operations() {
        for (annotation, body) in [
            ("IO", "1 OUTPUT"),
            ("ALLOC", "VEC_NEW"),
            ("FFI", "0 FFI_CALL"),
            ("FILE_IO", "0 0 FILE_OPEN"),
            ("NETWORK", "0 0 TCP_CONNECT"),
            ("SYSTEM", "CLOCK"),
        ] {
            let source = format!("PROCEDURE ok {annotation} {{ {body} }} 0");
            let result = check(&source);
            assert!(
                result.effect_violations.is_empty(),
                "{source}: {:?}",
                result.effect_violations
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Linear Type Tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_linearity_enum() {
        // Test basic linearity properties
        assert!(Linearity::Unrestricted.can_duplicate());
        assert!(!Linearity::Linear.can_duplicate());

        assert!(!Linearity::Unrestricted.is_linear());
        assert!(Linearity::Linear.is_linear());

        // Test join: linear values stay linear
        assert_eq!(
            Linearity::Linear.join(Linearity::Unrestricted),
            Linearity::Linear
        );
        assert_eq!(
            Linearity::Unrestricted.join(Linearity::Linear),
            Linearity::Linear
        );
        assert_eq!(
            Linearity::Unrestricted.join(Linearity::Unrestricted),
            Linearity::Unrestricted
        );
    }

    #[test]
    fn test_stack_type_creation() {
        // Test StackType constants
        assert!(StackType::PURE.can_duplicate());
        assert!(!StackType::PURE.is_linear());
        assert_eq!(StackType::PURE.temporal, TemporalType::Pure);

        assert!(StackType::TEMPORAL_LINEAR.is_linear());
        assert!(!StackType::TEMPORAL_LINEAR.can_duplicate());
        assert_eq!(StackType::TEMPORAL_LINEAR.temporal, TemporalType::Temporal);

        // Test consume: makes linear -> unrestricted
        let linear = StackType::TEMPORAL_LINEAR;
        let consumed = linear.consume();
        assert!(!consumed.is_linear());
        assert_eq!(consumed.temporal, TemporalType::Temporal);
    }

    #[test]
    fn test_stack_type_join() {
        // Joining linear with unrestricted produces linear
        let result = StackType::PURE.join(StackType::TEMPORAL_LINEAR);
        assert!(result.is_linear());
        assert_eq!(result.temporal, TemporalType::Temporal);
    }

    #[test]
    fn test_oracle_duplication_is_allowed() {
        let result = check("0 ORACLE DUP");
        assert!(!result.has_linear_violations());
        assert!(result.is_valid);
        assert_eq!(result.final_stack_types.len(), 2);
        assert!(result
            .final_stack_types
            .iter()
            .all(|ty| ty.temporal == TemporalType::Temporal && !ty.is_linear()));
    }

    #[test]
    fn test_oracle_over_duplication_is_allowed() {
        let result = check("0 ORACLE 1 OVER");
        assert!(!result.has_linear_violations());
        assert!(result.is_valid);
        assert_eq!(result.final_stack_types.last(), Some(&StackType::TEMPORAL));
    }

    #[test]
    fn test_linear_consumed_by_arithmetic() {
        // Arithmetic consumes linear values - no violation should occur
        let result = check("0 ORACLE 1 ADD 0 PROPHECY");

        // No linear violations - the value was properly consumed
        assert!(!result.has_linear_violations());
        assert!(result.is_valid);
    }

    #[test]
    fn test_linear_consumed_by_prophecy() {
        // PROPHECY consumes the linear value - no violation
        let result = check("0 ORACLE 0 PROPHECY");

        assert!(!result.has_linear_violations());
        assert!(result.is_valid);
    }

    #[test]
    fn test_linear_consumed_by_comparison() {
        // Comparison operations consume linear values
        let result = check("0 ORACLE 42 EQ");

        // No linear violations - comparison consumed the value
        assert!(!result.has_linear_violations());
    }

    #[test]
    fn test_no_linear_violation_on_pure_values() {
        // DUP of pure values should not produce violations
        let result = check("42 DUP ADD");

        assert!(!result.has_linear_violations());
        assert!(result.is_valid);
    }

    #[test]
    fn test_pick_can_duplicate_oracle_values() {
        let result = check("0 ORACLE 1 2 3 1 PICK");
        assert!(!result.has_linear_violations());
        assert!(result.is_valid);
    }

    #[test]
    fn test_pick_without_linear_is_ok() {
        // PICK without linear values on stack should be fine
        let result = check("1 2 3 4 1 PICK");

        // No linear violations since no ORACLE values
        assert!(!result.has_linear_violations());
        assert!(result.is_valid);
    }

    #[test]
    fn test_strict_linearity_mode_does_not_make_oracle_affine() {
        let program = parse("0 ORACLE DUP").expect("Parse failed");
        let mut checker = TypeChecker::new();
        checker.set_strict_linearity(true);
        let result = checker.check(&program);

        assert!(!result.has_linear_violations());
        assert!(result.is_valid);
    }

    #[test]
    fn test_permissive_linearity_mode() {
        // Test that permissive mode allows linear violations (with warnings)
        let program = parse("0 ORACLE DUP").expect("Parse failed");
        let mut checker = TypeChecker::new();
        checker.set_strict_linearity(false);
        let result = checker.check(&program);

        assert!(!result.has_linear_violations());
        assert!(result.is_valid);
    }

    #[test]
    fn flagship_examples_do_not_violate_linearity() {
        for (name, source) in [
            (
                "mutual exclusion",
                include_str!("../../examples/case_studies/mutual_exclusion.ouro"),
            ),
            (
                "circular dataflow",
                include_str!("../../examples/case_studies/circular_dataflow.ouro"),
            ),
            (
                "retrocausal game",
                include_str!("../../examples/case_studies/retrocausal_game.ouro"),
            ),
        ] {
            let program = parse(source).unwrap_or_else(|error| panic!("{}: {}", name, error));
            let result = type_check(&program);
            assert!(
                !result.has_linear_violations(),
                "{}: {:?}",
                name,
                result.linear_violations
            );
        }
    }

    #[test]
    fn test_oracle_returns_temporal_unrestricted() {
        let result = check("0 ORACLE");

        assert_eq!(result.final_stack_types.len(), 1);
        let oracle_type = result.final_stack_types[0];
        assert!(!oracle_type.is_linear());
        assert!(oracle_type.can_duplicate());
        assert_eq!(oracle_type.temporal, TemporalType::Temporal);
    }

    #[test]
    fn test_multiple_oracle_linear_tracking() {
        // Multiple ORACLE calls, each is independently linear
        let result = check("0 ORACLE 1 ORACLE ADD 0 PROPHECY");

        // Both linear values consumed by ADD - no violations
        assert!(!result.has_linear_violations());
        assert!(result.is_valid);
    }

    #[test]
    fn test_swap_preserves_linearity() {
        // SWAP doesn't duplicate, should not violate linearity
        let result = check("0 ORACLE 42 SWAP");

        // No violation - swap just moves values around
        assert!(!result.has_linear_violations());

        assert!(result.final_stack_types.iter().all(|ty| !ty.is_linear()));
    }

    #[test]
    fn test_rot_preserves_linearity() {
        // ROT doesn't duplicate, should not violate linearity
        let result = check("0 ORACLE 1 2 ROT");

        // No violation - rot just rotates values
        assert!(!result.has_linear_violations());
    }
}
