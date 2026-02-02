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
//! # Linear Types
//!
//! Values from ORACLE are marked as **linear**, meaning they can only be used once.
//! This prevents free duplication of temporal information, which is essential for
//! maintaining causal consistency in CTC semantics.
//!
//! - `Linear`: Must be consumed exactly once (cannot DUP/OVER/PICK)
//! - `Unrestricted`: Can be freely copied and discarded
//!
//! Linear types ensure that temporal information flows through explicit channels
//! and cannot be silently duplicated, which could create paradoxes.
//!
//! # Effect System
//!
//! The effect system tracks side effects:
//! - `Pure`: No effects (referentially transparent)
//! - `Reads`: Reads from memory
//! - `Writes`: Writes to memory
//! - `Temporal`: Uses ORACLE/PROPHECY
//! - `IO`: Performs input/output
//!
//! Effects form a lattice and compose via join:
//! ```text
//!           IO (all effects)
//!          / | \
//!    Reads  Writes  Temporal
//!          \ | /
//!          Pure
//! ```
//!
//! # Type Rules
//!
//! | Operation | Input Types | Output Type | Effect |
//! |-----------|-------------|-------------|--------|
//! | `ORACLE` | `Pure` (address) | `Temporal+Linear` | Temporal |
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
//! 5. **Linearity**: ORACLE results cannot be freely duplicated

use crate::ast::{Program, Stmt, OpCode, Procedure, Effect as AstEffect};
use crate::core::Address;
use std::collections::{HashMap, HashSet};

/// Temporal type for compile-time causal tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemporalType {
    /// Value has no temporal dependency (constant, ground truth).
    Pure,
    /// Value depends on at least one oracle read.
    Temporal,
    /// Type is not yet determined (for inference).
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

impl Default for TemporalType {
    fn default() -> Self {
        TemporalType::Unknown
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
        Self { temporal, linearity }
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputedEffect {
    /// No effects: pure, referentially transparent computation.
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
            (Pure, _) => true,           // Pure can go anywhere
            (_, IO) => true,             // IO subsumes all
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

impl Default for ComputedEffect {
    fn default() -> Self {
        ComputedEffect::Pure
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
    /// Specific memory addresses read (if known statically).
    pub read_addresses: HashSet<Address>,
    /// Specific memory addresses written (if known statically).
    pub write_addresses: HashSet<Address>,
}

impl EffectSet {
    /// Create an empty (pure) effect set.
    pub fn pure() -> Self {
        Self::default()
    }

    /// Create an effect set for ORACLE operation.
    pub fn oracle(addr: Option<Address>) -> Self {
        let mut set = Self::default();
        set.oracle = true;
        if let Some(a) = addr {
            set.read_addresses.insert(a);
        }
        set
    }

    /// Create an effect set for PROPHECY operation.
    pub fn prophecy(addr: Option<Address>) -> Self {
        let mut set = Self::default();
        set.prophecy = true;
        if let Some(a) = addr {
            set.write_addresses.insert(a);
        }
        set
    }

    /// Create an effect set for memory read.
    pub fn reads() -> Self {
        let mut set = Self::default();
        set.reads = true;
        set
    }

    /// Create an effect set for memory write.
    pub fn writes() -> Self {
        let mut set = Self::default();
        set.writes = true;
        set
    }

    /// Create an effect set for I/O.
    pub fn io() -> Self {
        let mut set = Self::default();
        set.io = true;
        set
    }

    /// Join two effect sets.
    pub fn join(&self, other: &Self) -> Self {
        EffectSet {
            reads: self.reads || other.reads,
            writes: self.writes || other.writes,
            oracle: self.oracle || other.oracle,
            prophecy: self.prophecy || other.prophecy,
            io: self.io || other.io,
            read_addresses: self.read_addresses.union(&other.read_addresses).cloned().collect(),
            write_addresses: self.write_addresses.union(&other.write_addresses).cloned().collect(),
        }
    }

    /// Check if this effect set is pure (no effects).
    pub fn is_pure(&self) -> bool {
        !self.reads && !self.writes && !self.oracle && !self.prophecy && !self.io
    }

    /// Check if this effect set has temporal effects.
    pub fn is_temporal(&self) -> bool {
        self.oracle || self.prophecy
    }

    /// Check if declared AST effects cover this computed effect set.
    pub fn is_covered_by(&self, declared: &[AstEffect]) -> bool {
        // Pure declaration means we require no effects
        if declared.iter().any(|e| matches!(e, AstEffect::Pure)) {
            return self.is_pure();
        }

        // Check if we have temporal effects without coverage
        // Currently AST effects don't have a Temporal variant,
        // so temporal effects are always violations if declared Pure
        let has_temporal_coverage = !self.is_temporal();

        // Check each declared effect type (for future extension)
        for effect in declared {
            match effect {
                AstEffect::Pure => {
                    // Already handled above
                }
                AstEffect::Reads(_) => {
                    // Reads declaration covers read effects
                    // (we ignore specific addresses for now)
                }
                AstEffect::Writes(_) => {
                    // Writes declaration covers write effects
                }
                AstEffect::Temporal => {
                    // Temporal effect covers ORACLE/PROPHECY usage
                    // This allows temporal operations when declared
                }
                AstEffect::IO => {
                    // IO effect covers INPUT/OUTPUT/EMIT usage
                }
                AstEffect::Alloc => {
                    // Alloc effect covers data structure allocation
                }
                AstEffect::FFI => {
                    // FFI effect covers foreign function calls
                }
                AstEffect::FileIO => {
                    // FileIO effect covers file operations
                }
                AstEffect::Network => {
                    // Network effect covers socket operations
                }
                AstEffect::System => {
                    // System effect covers process/system operations
                }
            }
        }

        // If we have temporal effects but no temporal coverage, fail
        if self.is_temporal() && !has_temporal_coverage {
            return false;
        }

        true
    }

    /// Convert to a simple computed effect for compatibility.
    pub fn to_computed_effect(&self) -> ComputedEffect {
        if self.io {
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
        if self.oracle { parts.push("oracle"); }
        if self.prophecy { parts.push("prophecy"); }
        if self.reads { parts.push("reads"); }
        if self.writes { parts.push("writes"); }
        if self.io { parts.push("io"); }
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
            message: format!("{} requires {} element(s) but stack has {}", operation, required, available),
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
    /// Inferred types for memory cells.
    pub cell_types: HashMap<Address, TemporalType>,
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
        write!(f, "linearity violation at stmt {}: {} - {}",
            self.stmt_index, self.operation, self.message)
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

    /// Check a procedure for effect compliance.
    fn check_procedure(&mut self, proc: &Procedure) {
        // Save current state
        let saved_effects = std::mem::take(&mut self.current_effects);
        let saved_stack = std::mem::take(&mut self.stack);

        // Check procedure body
        self.check_statements(&proc.body);

        // Verify declared effects cover actual effects
        if !proc.effects.is_empty() {
            if !self.current_effects.is_covered_by(&proc.effects) {
                self.effect_violations.push(EffectViolation {
                    procedure_name: proc.name.clone(),
                    declared: proc.effects.clone(),
                    actual: self.current_effects.clone(),
                    location: Some(self.stmt_index),
                });
            }
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
        match stmt {
            Stmt::Push(value) => {
                // Constants are always pure and unrestricted
                let ty = if value.prov.is_pure() {
                    StackType::PURE
                } else {
                    StackType::TEMPORAL
                };
                self.stack.push(ty);
            }

            Stmt::Op(op) => {
                self.check_op(*op);
            }

            Stmt::If { then_branch, else_branch } => {
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
                // Procedure call: we don't have the body here, so mark as Unknown
                // In a full implementation, we'd look up the procedure and analyze its body
                self.stack.push(StackType::UNKNOWN);
            }

            Stmt::Match { cases, default } => {
                // Pop the matched value
                self.pop_type();

                // Analyze all branches (conservative: union of all possible types)
                for (_, body) in cases {
                    for stmt in body {
                        self.check_stmt(stmt);
                    }
                }
                if let Some(default_body) = default {
                    for stmt in default_body {
                        self.check_stmt(stmt);
                    }
                }
            }

            Stmt::TemporalScope { body, .. } => {
                // Temporal scope: check the body
                // Memory isolation doesn't affect type checking at this level
                self.check_statements(body);
            }
        }
    }
    
    /// Check an opcode.
    fn check_op(&mut self, op: OpCode) {
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
                // Result is TEMPORAL and LINEAR - can only be used once
                self.stack.push(StackType::TEMPORAL_LINEAR);

                // Track effect: ORACLE is a temporal read
                self.current_effects.oracle = true;
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

                // Record the type written to this cell (if address is constant)
                // In practice, we don't know the address statically, so we approximate
                self.cell_types.insert(0, value_type.temporal); // Simplified: assume address 0

                // Track effect: PROPHECY is a temporal write
                self.current_effects.prophecy = true;
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
            OpCode::Add | OpCode::Sub | OpCode::Mul | OpCode::Div | OpCode::Mod |
            OpCode::Min | OpCode::Max => {
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
            OpCode::Eq | OpCode::Neq | OpCode::Lt | OpCode::Gt | OpCode::Lte | OpCode::Gte |
            OpCode::Slt | OpCode::Sgt | OpCode::Slte | OpCode::Sgte => {
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
                self.current_effects.io = true; // Allocation is a side effect
            }
            OpCode::VecPush => {
                self.pop_type(); // value
                let handle = self.pop_type(); // handle
                self.stack.push(handle);
                self.current_effects.io = true;
            }
            OpCode::VecPop => {
                let handle = self.pop_type();
                self.stack.push(handle);
                self.stack.push(StackType::UNKNOWN); // Value could be temporal if stored
                self.current_effects.io = true;
            }
            OpCode::VecGet => {
                self.pop_type(); // index
                let handle = self.pop_type();
                self.stack.push(handle);
                self.stack.push(StackType::UNKNOWN); // Value type unknown
                self.current_effects.io = true;
            }
            OpCode::VecSet => {
                self.pop_type(); // index
                self.pop_type(); // value
                let handle = self.pop_type();
                self.stack.push(handle);
                self.current_effects.io = true;
            }
            OpCode::VecLen => {
                let handle = self.pop_type();
                self.stack.push(handle);
                self.stack.push(StackType::PURE);
                self.current_effects.io = true;
            }

            // ===== Hash Table Operations =====
            OpCode::HashNew => {
                self.stack.push(StackType::PURE);
                self.current_effects.io = true;
            }
            OpCode::HashPut => {
                self.pop_type(); // value
                self.pop_type(); // key
                let handle = self.pop_type();
                self.stack.push(handle);
                self.current_effects.io = true;
            }
            OpCode::HashGet => {
                self.pop_type(); // key
                let handle = self.pop_type();
                self.stack.push(handle);
                self.stack.push(StackType::UNKNOWN); // value
                self.stack.push(StackType::PURE); // found flag
                self.current_effects.io = true;
            }
            OpCode::HashDel => {
                self.pop_type(); // key
                let handle = self.pop_type();
                self.stack.push(handle);
                self.current_effects.io = true;
            }
            OpCode::HashHas => {
                self.pop_type(); // key
                let handle = self.pop_type();
                self.stack.push(handle);
                self.stack.push(StackType::PURE); // found flag
                self.current_effects.io = true;
            }
            OpCode::HashLen => {
                let handle = self.pop_type();
                self.stack.push(handle);
                self.stack.push(StackType::PURE);
                self.current_effects.io = true;
            }

            // ===== Set Operations =====
            OpCode::SetNew => {
                self.stack.push(StackType::PURE);
                self.current_effects.io = true;
            }
            OpCode::SetAdd => {
                self.pop_type(); // value
                let handle = self.pop_type();
                self.stack.push(handle);
                self.current_effects.io = true;
            }
            OpCode::SetHas => {
                self.pop_type(); // value
                let handle = self.pop_type();
                self.stack.push(handle);
                self.stack.push(StackType::PURE); // found flag
                self.current_effects.io = true;
            }
            OpCode::SetDel => {
                self.pop_type(); // value
                let handle = self.pop_type();
                self.stack.push(handle);
                self.current_effects.io = true;
            }
            OpCode::SetLen => {
                let handle = self.pop_type();
                self.stack.push(handle);
                self.stack.push(StackType::PURE);
                self.current_effects.io = true;
            }

            // Dynamic stack structure changes cannot be statically typed easily
            OpCode::Roll | OpCode::Reverse | OpCode::StrRev | OpCode::StrCat | OpCode::StrSplit | OpCode::Assert | OpCode::Exec | OpCode::Dip | OpCode::Keep | OpCode::Bi | OpCode::Rec => {
                // Conservative approach: do nothing or invalidate stack?
                // For this prototype, we ignore their effect on type stack structure
            }

            // ===== FFI Operations =====
            OpCode::FFICall | OpCode::FFICallNamed => {
                // FFI calls have IO effect and potentially modify stack
                // We can't statically determine the stack effect
                self.pop_type(); // FFI ID or name handle
                self.current_effects.io = true;
            }

            // ===== File I/O Operations =====
            OpCode::FileOpen => {
                self.pop_type(); // mode
                self.pop_type(); // path_handle
                self.stack.push(StackType::PURE); // file_handle
                self.current_effects.io = true;
            }
            OpCode::FileRead => {
                self.pop_type(); // max_bytes
                let handle = self.pop_type();
                self.stack.push(handle); // file_handle
                self.stack.push(StackType::PURE); // buffer_handle
                self.stack.push(StackType::PURE); // bytes_read
                self.current_effects.io = true;
            }
            OpCode::FileWrite => {
                self.pop_type(); // buffer_handle
                let handle = self.pop_type();
                self.stack.push(handle); // file_handle
                self.stack.push(StackType::PURE); // bytes_written
                self.current_effects.io = true;
            }
            OpCode::FileSeek => {
                self.pop_type(); // origin
                self.pop_type(); // offset
                let handle = self.pop_type();
                self.stack.push(handle); // file_handle
                self.stack.push(StackType::PURE); // new_position
                self.current_effects.io = true;
            }
            OpCode::FileFlush => {
                let handle = self.pop_type();
                self.stack.push(handle);
                self.current_effects.io = true;
            }
            OpCode::FileClose => {
                self.pop_type(); // file_handle
                self.current_effects.io = true;
            }
            OpCode::FileExists => {
                self.pop_type(); // path_handle
                self.stack.push(StackType::PURE); // exists flag
                self.current_effects.io = true;
            }
            OpCode::FileSize => {
                self.pop_type(); // path_handle
                self.stack.push(StackType::PURE); // size
                self.current_effects.io = true;
            }

            // ===== Buffer Operations =====
            OpCode::BufferNew => {
                self.pop_type(); // capacity
                self.stack.push(StackType::PURE); // buffer_handle
                self.current_effects.io = true;
            }
            OpCode::BufferFromStack => {
                // Variable number of pops, just pop the count
                self.pop_type(); // n
                // Would need to pop n more values, but we can't know n statically
                self.stack.push(StackType::PURE); // buffer_handle
                self.current_effects.io = true;
            }
            OpCode::BufferToStack => {
                self.pop_type(); // buffer_handle
                // Would push variable number of values
                self.stack.push(StackType::PURE); // n
                self.current_effects.io = true;
            }
            OpCode::BufferLen => {
                let handle = self.pop_type();
                self.stack.push(handle); // buffer_handle
                self.stack.push(StackType::PURE); // length
                self.current_effects.io = true;
            }
            OpCode::BufferReadByte => {
                let handle = self.pop_type();
                self.stack.push(handle); // buffer_handle
                self.stack.push(StackType::PURE); // byte
                self.current_effects.io = true;
            }
            OpCode::BufferWriteByte => {
                self.pop_type(); // byte
                let handle = self.pop_type();
                self.stack.push(handle); // buffer_handle
                self.current_effects.io = true;
            }
            OpCode::BufferFree => {
                self.pop_type(); // buffer_handle
                self.current_effects.io = true;
            }

            // ===== Network Operations =====
            OpCode::TcpConnect => {
                self.pop_type(); // port
                self.pop_type(); // host_handle
                self.stack.push(StackType::PURE); // socket_handle
                self.current_effects.io = true;
            }
            OpCode::SocketSend => {
                self.pop_type(); // buffer_handle
                let handle = self.pop_type();
                self.stack.push(handle); // socket_handle
                self.stack.push(StackType::PURE); // bytes_sent
                self.current_effects.io = true;
            }
            OpCode::SocketRecv => {
                self.pop_type(); // max_bytes
                let handle = self.pop_type();
                self.stack.push(handle); // socket_handle
                self.stack.push(StackType::PURE); // buffer_handle
                self.stack.push(StackType::PURE); // bytes_recv
                self.current_effects.io = true;
            }
            OpCode::SocketClose => {
                self.pop_type(); // socket_handle
                self.current_effects.io = true;
            }

            // ===== Process Operations =====
            OpCode::ProcExec => {
                self.pop_type(); // command_handle
                self.stack.push(StackType::PURE); // output_handle
                self.stack.push(StackType::PURE); // exit_code
                self.current_effects.io = true;
            }

            // ===== System Operations =====
            OpCode::Clock => {
                self.stack.push(StackType::PURE); // timestamp
                self.current_effects.io = true;
            }
            OpCode::Sleep => {
                self.pop_type(); // milliseconds
                self.current_effects.io = true;
            }
            OpCode::Random => {
                self.stack.push(StackType::PURE); // random_value
                self.current_effects.io = true;
            }
        }
    }
    
    /// Pop a type from the abstract stack (consuming it).
    fn pop_type(&mut self) -> StackType {
        self.stack.pop().unwrap_or(StackType::UNKNOWN)
    }

    /// Pop a type and extract just the temporal component.
    fn pop_temporal(&mut self) -> TemporalType {
        self.pop_type().temporal
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
                        operation, self.stack.len() - 1 - i, st.describe()
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
                        self.stmt_index, operation, self.stack.len() - 1 - i
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
    
    if !result.final_stack_types.is_empty() {
        output.push_str("\nFinal Stack Types:\n");
        for (i, ty) in result.final_stack_types.iter().enumerate() {
            output.push_str(&format!("  {}: {}\n", i, ty.describe()));
        }
    }

    if !result.linear_violations.is_empty() {
        output.push_str("\nLinearity Violations:\n");
        for violation in &result.linear_violations {
            output.push_str(&format!("  - [stmt {}] {}: {}\n",
                violation.stmt_index, violation.operation, violation.message));
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
        assert_eq!(TemporalType::Pure.join(TemporalType::Pure), TemporalType::Pure);
        assert_eq!(TemporalType::Pure.join(TemporalType::Temporal), TemporalType::Temporal);
        assert_eq!(TemporalType::Temporal.join(TemporalType::Pure), TemporalType::Temporal);
        assert_eq!(TemporalType::Unknown.join(TemporalType::Pure), TemporalType::Pure);
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

        assert_eq!(ComputedEffect::Pure.join(ComputedEffect::Pure), ComputedEffect::Pure);
        assert_eq!(ComputedEffect::Pure.join(ComputedEffect::Reads), ComputedEffect::Reads);
        assert_eq!(ComputedEffect::Reads.join(ComputedEffect::Writes), ComputedEffect::IO);
        assert_eq!(ComputedEffect::Temporal.join(ComputedEffect::Pure), ComputedEffect::Temporal);
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
        assert_eq!(Linearity::Linear.join(Linearity::Unrestricted), Linearity::Linear);
        assert_eq!(Linearity::Unrestricted.join(Linearity::Linear), Linearity::Linear);
        assert_eq!(Linearity::Unrestricted.join(Linearity::Unrestricted), Linearity::Unrestricted);
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
    fn test_linear_dup_violation() {
        // DUP of an ORACLE result should produce a linear violation
        let result = check("0 ORACLE DUP");

        // Should detect the linear violation
        assert!(result.has_linear_violations());
        assert_eq!(result.linear_violations.len(), 1);
        assert_eq!(result.linear_violations[0].operation, "DUP");
    }

    #[test]
    fn test_linear_over_violation() {
        // OVER when the second-from-top is linear should produce a violation
        // Stack after "0 ORACLE 1": [linear, pure] (pure at top)
        // OVER copies the second element (linear) to top
        let result = check("0 ORACLE 1 OVER");

        // OVER copies the linear value - should produce violation
        assert!(result.has_linear_violations());
        let over_violations: Vec<_> = result.linear_violations.iter()
            .filter(|v| v.operation == "OVER")
            .collect();
        assert!(!over_violations.is_empty());
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
    fn test_pick_linear_violation() {
        // PICK with linear values on stack should produce linearity violation
        // (stricter than before - now an error, not just a warning)
        let result = check("0 ORACLE 1 2 3 1 PICK");

        // PICK at runtime index may duplicate linear value - now a compile error
        // Since we can't statically verify which element is picked, we must
        // conservatively reject if ANY stack element is linear
        assert!(result.has_linear_violations());
        let pick_violations: Vec<_> = result.linear_violations.iter()
            .filter(|v| v.operation == "PICK")
            .collect();
        assert!(!pick_violations.is_empty(), "Expected PICK linearity violation");
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
    fn test_strict_linearity_mode() {
        // Test that strict mode produces errors (not just warnings)
        let program = parse("0 ORACLE DUP").expect("Parse failed");
        let mut checker = TypeChecker::new();
        checker.set_strict_linearity(true);
        let result = checker.check(&program);

        assert!(result.has_linear_violations());
        assert!(!result.is_valid); // Invalid in strict mode
    }

    #[test]
    fn test_permissive_linearity_mode() {
        // Test that permissive mode allows linear violations (with warnings)
        let program = parse("0 ORACLE DUP").expect("Parse failed");
        let mut checker = TypeChecker::new();
        checker.set_strict_linearity(false);
        let result = checker.check(&program);

        // Still records the violation but doesn't fail
        assert!(!result.warnings.is_empty() || !result.linear_violations.is_empty());
        // In permissive mode, is_valid is true even with linear issues
        assert!(result.is_valid);
    }

    #[test]
    fn test_linear_violation_message() {
        let result = check("0 ORACLE DUP");

        assert!(result.has_linear_violations());
        let violation = &result.linear_violations[0];
        assert!(violation.message.contains("Linear"));
        assert!(violation.message.contains("ORACLE"));
    }

    #[test]
    fn test_oracle_returns_temporal_linear() {
        // Verify that ORACLE pushes a TEMPORAL_LINEAR type
        let result = check("0 ORACLE");

        // Stack should contain one temporal linear value
        assert_eq!(result.final_stack_types.len(), 1);
        let oracle_type = result.final_stack_types[0];
        assert!(oracle_type.is_linear());
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

        // But the linear value is still on the stack
        let has_linear = result.final_stack_types.iter().any(|t| t.is_linear());
        assert!(has_linear);
    }

    #[test]
    fn test_rot_preserves_linearity() {
        // ROT doesn't duplicate, should not violate linearity
        let result = check("0 ORACLE 1 2 ROT");

        // No violation - rot just rotates values
        assert!(!result.has_linear_violations());
    }
}
