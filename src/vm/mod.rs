//! Virtual Machine for OUROCHRONOS epoch execution.
//!
//! This module provides compatibility facades over one executable authority:
//!
//! - **Executor**: source-level epoch API that resolves, semantically checks,
//!   independently verifies, and executes bytecode
//! - **FastExecutor**: historical pure-program API backed by the same bytecode VM
//!
//! # Architecture
//!
//! Neither facade interprets source AST in production. The bytecode VM executes
//! a single epoch: given an anamnesis (read-only "future" memory), it produces a
//! present (read-write "current" memory) and buffered output.
//!
//! The fixed-point search (in timeloop) repeatedly runs epochs until
//! Present = Anamnesis (temporal consistency achieved).

pub mod executor;
pub mod fast_vm;

// Re-export from executor
pub use executor::{
    EffectsPolicy, EpochContext, EpochResult, EpochStatus, Executor, ExecutorConfig, VmState,
};

// Re-export from fast_vm
pub use fast_vm::{execute_with_fast_path, is_program_pure, FastExecutor, FastStack};
