//! Runtime layer for OUROCHRONOS.
//!
//! This module provides runtime infrastructure including:
//!
//! - **FFI**: Foreign function interface for calling external libraries
//! - **IO**: File, network, and process I/O operations
//! - **Module**: Module system for organizing code
//! - **Package**: Package management and dependencies

pub mod ffi;
pub mod io;
pub mod module;
pub mod package;

// Re-export from ffi
pub use ffi::{
    FFIRegistry, FFIContext, FFISignature, FFIFunction, FFIEffect, FFIType, FFICaller,
    FFIParam, FFIError, FFIFn, DynamicLibraryManager, ExtendedFFIContext,
};

// Re-export from io
pub use io::{IOContext, FileMode, SeekOrigin, Buffer, IOStats, FileHandle, SocketHandle, IOLogEntry};

// Re-export from module
pub use module::{Module, ModuleRegistry, parse_module_declaration, parse_exports};

// Re-export from package
pub use package::{PackageManager, PackageManifest, Package, Dependency, PackageSource};
