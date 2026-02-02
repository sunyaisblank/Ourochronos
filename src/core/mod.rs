//! Core types for the OUROCHRONOS virtual machine.
//!
//! This module defines the fundamental data types that form the foundation
//! of the OUROCHRONOS language runtime:
//!
//! - **Address**: 16-bit memory indices
//! - **Value**: 64-bit unsigned integers with provenance tracking
//! - **Memory**: The memory state (65536 cells)
//! - **Stack**: Bounds-checked stack operations
//! - **Provenance**: Temporal dependency tracking
//! - **Error**: Comprehensive error hierarchy
//!
//! # Layer 0 - No Internal Dependencies
//!
//! This module has no dependencies on other OUROCHRONOS modules,
//! allowing it to be imported by all other layers.

pub mod address;
pub mod provenance;
pub mod error;
pub mod value;
pub mod memory;
pub mod stack;
pub mod data_structures;

// Re-export primary types at module level
pub use address::{Address, MEMORY_SIZE, Handle};
pub use provenance::{Provenance, AlgebraicOp};
pub use error::{
    OuroError, OuroResult, ErrorConfig, ErrorCategory, SourceLocation,
    DivisionByZeroPolicy, BoundsPolicy, StackPolicy, MemoryOperation, ErrorCollector
};
pub use value::{Value, OutputItem};
pub use memory::Memory;
pub use stack::Stack;
pub use data_structures::{VecStore, HashStore, SetStore, DataStructures};
