//! Virtual Machine for OUROCHRONOS epoch execution.
//!
//! This module provides two VM implementations:
//!
//! - **Executor**: The full-featured VM with provenance tracking for temporal operations
//! - **FastExecutor**: An optimised VM for pure programmes without temporal operations
//!
//! # Architecture
//!
//! The VM executes a single epoch: given an anamnesis (read-only "future" memory),
//! it produces a present (read-write "current" memory) and output.
//!
//! The fixed-point search (in timeloop) repeatedly runs epochs until
//! Present = Anamnesis (temporal consistency achieved).
//!
//! # VM Traits
//!
//! The `traits` module defines common operation interfaces that can be specialized:
//! - `ArithmeticOps`: Add, sub, mul, div, etc.
//! - `BitwiseOps`: And, or, xor, shifts
//! - `ComparisonOps`: Eq, lt, gt, etc.
//!
//! Two implementations are provided:
//! - `ProvenanceOps`: For values with causal dependency tracking
//! - `PureOps`: For raw u64 values without provenance

pub mod executor;
pub mod fast_vm;
pub mod traits;

// Re-export from executor
pub use executor::{Executor, EpochStatus, VmState, EpochResult, ExecutorConfig};

// Re-export from fast_vm
pub use fast_vm::{FastExecutor, FastStack, is_program_pure, execute_with_fast_path};

// Re-export traits
pub use traits::{ArithmeticOps, BitwiseOps, ComparisonOps, ProvenanceOps, PureOps};
