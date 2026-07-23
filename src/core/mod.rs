//! Core types for the OUROCHRONOS virtual machine.
//!
//! This module defines the fundamental data types that form the foundation
//! of the OUROCHRONOS language runtime:
//!
//! - **Address**: 64-bit indices into a run-configured finite memory
//! - **Value**: 64-bit unsigned integers with provenance tracking
//! - **Memory**: The memory state (65,536 cells by default; configurable)
//! - **Stack**: Bounds-checked stack operations
//! - **Provenance**: Temporal dependency tracking
//! - **Error**: Comprehensive error hierarchy
//!
//! # Layer 0 - No Internal Dependencies
//!
//! This module has no dependencies on other OUROCHRONOS modules,
//! allowing it to be imported by all other layers.

pub mod address;
pub mod data_structures;
pub mod error;
pub mod memory;
pub mod paged_memory;
pub mod provenance;
pub mod stack;
pub mod value;

// Re-export primary types at module level
pub use address::{Address, Handle, MEMORY_SIZE};
pub use data_structures::{DataStructures, HashStore, SetStore, VecStore};
pub use error::{
    BoundsPolicy, DivisionByZeroPolicy, ErrorCategory, ErrorCollector, ErrorConfig,
    MemoryOperation, OuroError, OuroResult, SourceLocation, StackPolicy,
};
pub use memory::{ExactMemoryState, Memory, MAX_DENSE_MEMORY_CELLS};
pub use paged_memory::{PagedMemory, PagedMemoryError, PagedMemoryLimits, PAGE_CELLS};
pub use provenance::{AlgebraicOp, Provenance};
pub use stack::Stack;
pub use value::{OutputItem, Value};
