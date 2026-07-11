//! Runtime layer for OUROCHRONOS.
//!
//! This module provides runtime infrastructure including:
//!
//! - **FFI**: Foreign function interface for calling external libraries
//! - **IO**: File, network, and process I/O operations

pub mod ffi;
pub mod io;

// Re-export from ffi
pub use ffi::{
    FFIRegistry, FFIContext, FFISignature, FFIFunction, FFIEffect, FFIType, FFICaller,
    FFIParam, FFIError, FFIFn, DynamicLibraryManager, ExtendedFFIContext,
};

// Re-export from io
pub use io::{IOContext, FileMode, SeekOrigin, Buffer, IOStats, FileHandle, SocketHandle, IOLogEntry};

