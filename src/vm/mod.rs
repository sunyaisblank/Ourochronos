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

pub mod executor;
pub mod fast_vm;

// Re-export from executor
pub use executor::{Executor, EpochStatus, VmState, EpochResult, ExecutorConfig};

// Re-export from fast_vm
pub use fast_vm::{FastExecutor, FastStack, is_program_pure, execute_with_fast_path};
