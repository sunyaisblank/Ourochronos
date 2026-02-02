//! Memory address types and constants for OUROCHRONOS.
//!
//! This module defines the fundamental address type and memory size constant.
//! These are separated out to break circular dependencies between core modules.

/// Memory address (16-bit index).
pub type Address = u16;

/// The size of the memory space (2^16 = 65536 cells).
pub const MEMORY_SIZE: usize = 65536;

/// Handle type for referencing data structures.
///
/// Data structures are stored externally and referenced by handles.
/// This keeps the core Value type as a simple u64 while allowing
/// complex data structures to be manipulated.
pub type Handle = u64;
