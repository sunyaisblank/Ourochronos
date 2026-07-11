// ═══════════════════════════════════════════════════════════════════════════
// Layer 0: Core (No internal dependencies)
// ═══════════════════════════════════════════════════════════════════════════
pub mod core;

// ═══════════════════════════════════════════════════════════════════════════
// Layer 1: Compiler (depends on core)
// ═══════════════════════════════════════════════════════════════════════════
pub mod ast;
pub mod parser;
pub mod compiler;

// ═══════════════════════════════════════════════════════════════════════════
// Layer 2: Temporal & Types (depends on core, compiler)
// ═══════════════════════════════════════════════════════════════════════════
pub mod temporal;
pub mod types;

// ═══════════════════════════════════════════════════════════════════════════
// Layer 3: VM (depends on core, compiler, temporal)
// ═══════════════════════════════════════════════════════════════════════════
pub mod vm;

// ═══════════════════════════════════════════════════════════════════════════
// Layer 4: Runtime (depends on core, vm)
// ═══════════════════════════════════════════════════════════════════════════
pub mod runtime;

// ═══════════════════════════════════════════════════════════════════════════
// Layer 5: Tooling (depends on all)
// ═══════════════════════════════════════════════════════════════════════════
pub mod tooling;

// ═══════════════════════════════════════════════════════════════════════════
// Standard Library & Cross-cutting
// ═══════════════════════════════════════════════════════════════════════════
pub mod stdlib;
pub mod audit;

// ═══════════════════════════════════════════════════════════════════════════
// Public API Re-exports
// ═══════════════════════════════════════════════════════════════════════════

// Core types
pub use core::error::{OuroError, OuroResult, ErrorConfig, ErrorCategory, SourceLocation,
                 DivisionByZeroPolicy, BoundsPolicy, StackPolicy, MemoryOperation, ErrorCollector};
pub use core::{Value, Address, Memory, OutputItem, Stack, MEMORY_SIZE};
pub use core::Handle;
pub use core::provenance::{Provenance, DEFAULT_PROVENANCE_SATURATION_LIMIT};

// AST and parser
pub use ast::{OpCode, Stmt, Program};
pub use parser::{tokenize, Parser};

// Temporal
pub use temporal::timeloop::{TimeLoop, ConvergenceStatus, TimeLoopConfig as Config, ExecutionMode};
pub use temporal::smt_encoder::SmtEncoder;
pub use temporal::action::{ActionPrinciple, ActionConfig};

// Type system
pub use types::{TemporalType, TypeChecker, type_check};

// VM
pub use vm::{Executor, EpochStatus, FastExecutor};

// Tooling
pub use tooling::repl::{Repl, ReplConfig};
pub use tooling::debugger::Debugger;

// Standard library
pub use stdlib::StdLib;

pub mod analysis;

mod tests;
mod determinism_tests;
mod property_tests;
