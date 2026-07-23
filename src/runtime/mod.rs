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
    DynamicLibraryManager, ExtendedFFIContext, FFICaller, FFIContext, FFIEffect, FFIError, FFIFn,
    FFIFunction, FFIParam, FFIRegistry, FFISignature, FFIType,
};

// Re-export from io
pub use io::{
    Buffer, FileHandle, FileMode, IOContext, IOLogEntry, IOStats, SeekOrigin, SocketHandle,
};
