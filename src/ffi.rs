//! Foreign Function Interface (FFI) for OUROCHRONOS.
//!
//! This module provides a safe mechanism for calling external functions from
//! Ourochronos programs while maintaining temporal semantics and type safety.
//!
//! # Design Principles
//!
//! 1. **Type Safety at Boundaries**: All FFI functions have explicit type signatures
//! 2. **Effect Tracking**: FFI functions declare their effects (Pure, IO, Temporal)
//! 3. **Memory Safety**: Handles are used for external resources
//! 4. **Temporal Awareness**: FFI calls cannot bypass temporal consistency checks
//!
//! # Usage
//!
//! ```text
//! FOREIGN "libc" {
//!     PROC strlen ( ptr: u64 -- len: u64 ) PURE;
//!     PROC puts ( ptr: u64 -- result: i32 ) IO;
//! }
//! ```

use crate::core_types::{Value, Handle};
use crate::vm::VmState;
use crate::error::{OuroError, OuroResult, SourceLocation};
use std::collections::HashMap;
use std::sync::Arc;
use std::path::PathBuf;

#[cfg(feature = "dynamic-ffi")]
use libloading::{Library, Symbol};

// ═══════════════════════════════════════════════════════════════════════════════
// FFI Effect Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Effect annotations for FFI functions.
///
/// These effects integrate with the type system to ensure temporal consistency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FFIEffect {
    /// Function has no side effects (referentially transparent).
    Pure,
    /// Function performs I/O operations.
    IO,
    /// Function reads from external state.
    Reads,
    /// Function writes to external state.
    Writes,
    /// Function may affect temporal state (use with caution).
    Temporal,
    /// Function allocates resources.
    Alloc,
}

impl FFIEffect {
    /// Check if this effect is compatible with a pure context.
    pub fn is_pure(&self) -> bool {
        matches!(self, FFIEffect::Pure)
    }

    /// Check if this effect involves I/O.
    pub fn involves_io(&self) -> bool {
        matches!(self, FFIEffect::IO | FFIEffect::Reads | FFIEffect::Writes)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FFI Type System
// ═══════════════════════════════════════════════════════════════════════════════

/// Types available at FFI boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FFIType {
    /// 64-bit unsigned integer (default Ourochronos value).
    U64,
    /// 64-bit signed integer.
    I64,
    /// 32-bit unsigned integer.
    U32,
    /// 32-bit signed integer.
    I32,
    /// Boolean (1 or 0).
    Bool,
    /// Pointer/handle to external data.
    Ptr,
    /// String handle (pointer + length).
    Str,
    /// Void (for functions returning nothing).
    Void,
    /// File handle.
    FileHandle,
    /// Socket handle.
    SocketHandle,
    /// Buffer handle.
    BufferHandle,
}

impl FFIType {
    /// Parse a type name from a string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "U64" | "UINT64" | "ULONG" => Some(FFIType::U64),
            "I64" | "INT64" | "LONG" => Some(FFIType::I64),
            "U32" | "UINT32" | "UINT" => Some(FFIType::U32),
            "I32" | "INT32" | "INT" => Some(FFIType::I32),
            "BOOL" | "BOOLEAN" => Some(FFIType::Bool),
            "PTR" | "POINTER" | "ADDR" => Some(FFIType::Ptr),
            "STR" | "STRING" => Some(FFIType::Str),
            "VOID" | "()" => Some(FFIType::Void),
            "FILE" | "FILEHANDLE" => Some(FFIType::FileHandle),
            "SOCKET" | "SOCKETHANDLE" => Some(FFIType::SocketHandle),
            "BUFFER" | "BUFFERHANDLE" => Some(FFIType::BufferHandle),
            _ => None,
        }
    }

    /// Get the stack representation size (number of values).
    pub fn stack_size(&self) -> usize {
        match self {
            FFIType::Void => 0,
            FFIType::Str => 2, // pointer + length
            _ => 1,
        }
    }

    /// Convert a value to this FFI type.
    pub fn from_value(&self, val: &Value) -> u64 {
        match self {
            FFIType::Bool => if val.val != 0 { 1 } else { 0 },
            FFIType::U32 => val.val as u32 as u64,
            FFIType::I32 => val.val as i32 as i64 as u64,
            FFIType::I64 => val.val,
            _ => val.val,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FFI Function Signature
// ═══════════════════════════════════════════════════════════════════════════════

/// A parameter in an FFI function signature.
#[derive(Debug, Clone)]
pub struct FFIParam {
    /// Parameter name (for documentation).
    pub name: String,
    /// Parameter type.
    pub typ: FFIType,
}

/// Signature of an FFI function.
#[derive(Debug, Clone)]
pub struct FFISignature {
    /// Function name.
    pub name: String,
    /// Library the function belongs to.
    pub library: String,
    /// Input parameters (popped from stack).
    pub params: Vec<FFIParam>,
    /// Return types (pushed to stack).
    pub returns: Vec<FFIType>,
    /// Effect annotations.
    pub effects: Vec<FFIEffect>,
}

impl FFISignature {
    /// Create a new FFI signature.
    pub fn new(name: impl Into<String>, library: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            library: library.into(),
            params: Vec::new(),
            returns: Vec::new(),
            effects: vec![FFIEffect::IO], // Default to IO for safety
        }
    }

    /// Add an input parameter.
    pub fn param(mut self, name: impl Into<String>, typ: FFIType) -> Self {
        self.params.push(FFIParam { name: name.into(), typ });
        self
    }

    /// Add a return type.
    pub fn returns_type(mut self, typ: FFIType) -> Self {
        self.returns.push(typ);
        self
    }

    /// Set effects.
    pub fn effects(mut self, effects: Vec<FFIEffect>) -> Self {
        self.effects = effects;
        self
    }

    /// Mark as pure.
    pub fn pure(mut self) -> Self {
        self.effects = vec![FFIEffect::Pure];
        self
    }

    /// Calculate total input stack consumption.
    pub fn input_stack_size(&self) -> usize {
        self.params.iter().map(|p| p.typ.stack_size()).sum()
    }

    /// Calculate total output stack production.
    pub fn output_stack_size(&self) -> usize {
        self.returns.iter().map(|t| t.stack_size()).sum()
    }

    /// Check if this function is pure.
    pub fn is_pure(&self) -> bool {
        self.effects.iter().all(|e| e.is_pure())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FFI Function Implementation
// ═══════════════════════════════════════════════════════════════════════════════

/// Type alias for FFI function implementation.
pub type FFIFn = Arc<dyn Fn(&mut VmState, &[Value]) -> OuroResult<Vec<Value>> + Send + Sync>;

/// A registered FFI function with its implementation.
#[derive(Clone)]
pub struct FFIFunction {
    /// Function signature.
    pub signature: FFISignature,
    /// Function ID in the registry.
    pub id: u32,
    /// Implementation.
    implementation: FFIFn,
}

impl FFIFunction {
    /// Create a new FFI function.
    pub fn new<F>(signature: FFISignature, id: u32, implementation: F) -> Self
    where
        F: Fn(&mut VmState, &[Value]) -> OuroResult<Vec<Value>> + Send + Sync + 'static,
    {
        Self {
            signature,
            id,
            implementation: Arc::new(implementation),
        }
    }

    /// Call the FFI function.
    pub fn call(&self, state: &mut VmState, args: &[Value]) -> OuroResult<Vec<Value>> {
        (self.implementation)(state, args)
    }
}

impl std::fmt::Debug for FFIFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FFIFunction")
            .field("signature", &self.signature)
            .field("id", &self.id)
            .finish()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FFI Registry
// ═══════════════════════════════════════════════════════════════════════════════

/// Registry for FFI functions.
///
/// The registry maintains all registered FFI functions and their mappings.
#[derive(Default)]
pub struct FFIRegistry {
    /// Functions indexed by ID.
    functions: Vec<FFIFunction>,
    /// Function name -> ID mapping.
    name_map: HashMap<String, u32>,
    /// Library name -> function IDs mapping.
    library_map: HashMap<String, Vec<u32>>,
}

impl FFIRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
            name_map: HashMap::new(),
            library_map: HashMap::new(),
        }
    }

    /// Register a new FFI function.
    pub fn register<F>(&mut self, signature: FFISignature, implementation: F) -> u32
    where
        F: Fn(&mut VmState, &[Value]) -> OuroResult<Vec<Value>> + Send + Sync + 'static,
    {
        let id = self.functions.len() as u32;
        let name = signature.name.clone();
        let library = signature.library.clone();

        let function = FFIFunction::new(signature, id, implementation);
        self.functions.push(function);

        // Create qualified name: library::function
        let qualified_name = format!("{}::{}", library, name);
        self.name_map.insert(qualified_name.to_lowercase(), id);
        self.name_map.insert(name.to_lowercase(), id);

        // Add to library map
        self.library_map
            .entry(library.to_lowercase())
            .or_default()
            .push(id);

        id
    }

    /// Get a function by ID.
    pub fn get(&self, id: u32) -> Option<&FFIFunction> {
        self.functions.get(id as usize)
    }

    /// Get a function by name.
    pub fn get_by_name(&self, name: &str) -> Option<&FFIFunction> {
        let id = self.name_map.get(&name.to_lowercase())?;
        self.get(*id)
    }

    /// Get all functions in a library.
    pub fn get_library_functions(&self, library: &str) -> Vec<&FFIFunction> {
        self.library_map
            .get(&library.to_lowercase())
            .map(|ids| ids.iter().filter_map(|id| self.get(*id)).collect())
            .unwrap_or_default()
    }

    /// Get the effect set for a function.
    pub fn get_effects(&self, id: u32) -> Vec<FFIEffect> {
        self.get(id)
            .map(|f| f.signature.effects.clone())
            .unwrap_or_default()
    }

    /// Check if a function exists.
    pub fn contains(&self, name: &str) -> bool {
        self.name_map.contains_key(&name.to_lowercase())
    }

    /// Get the number of registered functions.
    pub fn len(&self) -> usize {
        self.functions.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.functions.is_empty()
    }

    /// List all registered function names.
    pub fn list_functions(&self) -> Vec<&str> {
        self.functions.iter().map(|f| f.signature.name.as_str()).collect()
    }
}

impl std::fmt::Debug for FFIRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FFIRegistry")
            .field("function_count", &self.functions.len())
            .field("libraries", &self.library_map.keys().collect::<Vec<_>>())
            .finish()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FFI Context
// ═══════════════════════════════════════════════════════════════════════════════

/// Context for FFI operations during VM execution.
///
/// This holds all state needed for FFI calls, including the registry
/// and any external resources that need to persist across calls.
#[derive(Default)]
pub struct FFIContext {
    /// Registry of available FFI functions.
    pub registry: FFIRegistry,
    /// Buffer storage for FFI operations.
    buffers: Vec<Vec<u8>>,
    /// External data storage (opaque handles).
    external_data: HashMap<Handle, Box<dyn std::any::Any + Send + Sync>>,
    /// Next handle ID for external data.
    next_handle: Handle,
}

impl FFIContext {
    /// Create a new FFI context.
    pub fn new() -> Self {
        Self {
            registry: FFIRegistry::new(),
            buffers: Vec::new(),
            external_data: HashMap::new(),
            next_handle: 1, // 0 is reserved for null
        }
    }

    /// Create a context with the standard FFI functions pre-registered.
    pub fn with_stdlib() -> Self {
        let mut ctx = Self::new();
        ctx.register_stdlib();
        ctx
    }

    /// Allocate a new buffer and return its handle.
    pub fn alloc_buffer(&mut self, size: usize) -> Handle {
        let handle = self.buffers.len() as Handle;
        self.buffers.push(vec![0u8; size]);
        handle
    }

    /// Get a buffer by handle.
    pub fn get_buffer(&self, handle: Handle) -> Option<&[u8]> {
        self.buffers.get(handle as usize).map(|v| v.as_slice())
    }

    /// Get a mutable buffer by handle.
    pub fn get_buffer_mut(&mut self, handle: Handle) -> Option<&mut Vec<u8>> {
        self.buffers.get_mut(handle as usize)
    }

    /// Store external data and return a handle.
    pub fn store_external<T: Send + Sync + 'static>(&mut self, data: T) -> Handle {
        let handle = self.next_handle;
        self.next_handle += 1;
        self.external_data.insert(handle, Box::new(data));
        handle
    }

    /// Retrieve external data by handle.
    pub fn get_external<T: 'static>(&self, handle: Handle) -> Option<&T> {
        self.external_data.get(&handle)?.downcast_ref()
    }

    /// Retrieve mutable external data by handle.
    pub fn get_external_mut<T: 'static>(&mut self, handle: Handle) -> Option<&mut T> {
        self.external_data.get_mut(&handle)?.downcast_mut()
    }

    /// Remove external data by handle.
    pub fn remove_external(&mut self, handle: Handle) -> bool {
        self.external_data.remove(&handle).is_some()
    }

    /// Register standard library FFI functions.
    fn register_stdlib(&mut self) {
        // Buffer operations
        self.registry.register(
            FFISignature::new("buffer_alloc", "core")
                .param("size", FFIType::U64)
                .returns_type(FFIType::BufferHandle)
                .effects(vec![FFIEffect::Alloc]),
            |_state, args| {
                let size = args.get(0).map(|v| v.val as usize).unwrap_or(0);
                // Note: Actual allocation happens via FFIContext, not here
                // This is a placeholder - actual implementation in VM
                Ok(vec![Value::new(size as u64)])
            },
        );

        // String length (pure)
        self.registry.register(
            FFISignature::new("str_len", "core")
                .param("handle", FFIType::BufferHandle)
                .returns_type(FFIType::U64)
                .pure(),
            |_state, args| {
                let _handle = args.get(0).map(|v| v.val).unwrap_or(0);
                // Placeholder - actual implementation accesses buffer
                Ok(vec![Value::new(0)])
            },
        );

        // Clock/time (IO effect)
        self.registry.register(
            FFISignature::new("clock", "core")
                .returns_type(FFIType::U64)
                .effects(vec![FFIEffect::IO]),
            |_state, _args| {
                use std::time::{SystemTime, UNIX_EPOCH};
                let duration = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default();
                Ok(vec![Value::new(duration.as_millis() as u64)])
            },
        );

        // Random number (IO effect - non-deterministic)
        self.registry.register(
            FFISignature::new("random", "core")
                .returns_type(FFIType::U64)
                .effects(vec![FFIEffect::IO]),
            |_state, _args| {
                // Simple pseudo-random using time
                use std::time::{SystemTime, UNIX_EPOCH};
                let nanos = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .subsec_nanos();
                Ok(vec![Value::new(nanos as u64)])
            },
        );

        // Sleep (IO effect)
        self.registry.register(
            FFISignature::new("sleep", "core")
                .param("milliseconds", FFIType::U64)
                .effects(vec![FFIEffect::IO]),
            |_state, args| {
                let ms = args.get(0).map(|v| v.val).unwrap_or(0);
                std::thread::sleep(std::time::Duration::from_millis(ms));
                Ok(vec![])
            },
        );

        // Environment variable (IO effect)
        self.registry.register(
            FFISignature::new("getenv", "core")
                .param("name_ptr", FFIType::Ptr)
                .param("name_len", FFIType::U64)
                .returns_type(FFIType::U64) // Returns 0 if not found, or value handle
                .effects(vec![FFIEffect::IO, FFIEffect::Reads]),
            |_state, _args| {
                // Placeholder - needs string handling
                Ok(vec![Value::new(0)])
            },
        );
    }
}

impl std::fmt::Debug for FFIContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FFIContext")
            .field("registry", &self.registry)
            .field("buffer_count", &self.buffers.len())
            .field("external_data_count", &self.external_data.len())
            .finish()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FFI Errors
// ═══════════════════════════════════════════════════════════════════════════════

/// Errors specific to FFI operations.
#[derive(Debug, Clone)]
pub enum FFIError {
    /// Function not found in registry.
    FunctionNotFound { name: String },
    /// Wrong number of arguments.
    ArgumentMismatch { expected: usize, got: usize },
    /// Type conversion failed.
    TypeConversion { expected: FFIType, got: String },
    /// Library not loaded.
    LibraryNotLoaded { name: String },
    /// Effect violation (e.g., calling IO function from pure context).
    EffectViolation { function: String, effect: FFIEffect },
    /// Invalid handle.
    InvalidHandle { handle: Handle },
    /// Buffer overflow.
    BufferOverflow { handle: Handle, requested: usize, available: usize },
    /// External call failed.
    ExternalCallFailed { function: String, message: String },
}

impl std::fmt::Display for FFIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FFIError::FunctionNotFound { name } => {
                write!(f, "FFI function not found: '{}'", name)
            }
            FFIError::ArgumentMismatch { expected, got } => {
                write!(f, "FFI argument mismatch: expected {} arguments, got {}", expected, got)
            }
            FFIError::TypeConversion { expected, got } => {
                write!(f, "FFI type conversion failed: expected {:?}, got {}", expected, got)
            }
            FFIError::LibraryNotLoaded { name } => {
                write!(f, "FFI library not loaded: '{}'", name)
            }
            FFIError::EffectViolation { function, effect } => {
                write!(f, "FFI effect violation: function '{}' has {:?} effect", function, effect)
            }
            FFIError::InvalidHandle { handle } => {
                write!(f, "FFI invalid handle: {}", handle)
            }
            FFIError::BufferOverflow { handle, requested, available } => {
                write!(f, "FFI buffer overflow: handle {}, requested {} bytes, available {}",
                       handle, requested, available)
            }
            FFIError::ExternalCallFailed { function, message } => {
                write!(f, "FFI external call failed: '{}': {}", function, message)
            }
        }
    }
}

impl std::error::Error for FFIError {}

impl From<FFIError> for OuroError {
    fn from(err: FFIError) -> Self {
        OuroError::FFI {
            message: err.to_string(),
            location: SourceLocation::default(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FFI Caller
// ═══════════════════════════════════════════════════════════════════════════════

/// Helper for calling FFI functions from the VM.
pub struct FFICaller;

impl FFICaller {
    /// Call an FFI function by ID.
    pub fn call_by_id(
        ctx: &mut FFIContext,
        state: &mut VmState,
        id: u32,
        location: SourceLocation,
    ) -> OuroResult<()> {
        let function = ctx.registry.get(id).ok_or_else(|| OuroError::FFI {
            message: format!("FFI function ID {} not found", id),
            location: location.clone(),
        })?;

        let signature = function.signature.clone();
        let input_size = signature.input_stack_size();

        // Pop arguments from stack
        if state.stack.len() < input_size {
            return Err(OuroError::StackUnderflow {
                operation: format!("FFI call '{}'", signature.name),
                required: input_size,
                available: state.stack.len(),
                location,
            });
        }

        let mut args = Vec::with_capacity(input_size);
        for _ in 0..input_size {
            args.push(state.stack.pop().unwrap());
        }
        args.reverse(); // Stack pops in reverse order

        // Clone function for call (to avoid borrow issues)
        let func = ctx.registry.get(id).unwrap().clone();

        // Call the function
        let results = func.call(state, &args)?;

        // Push results to stack
        for result in results {
            state.stack.push(result);
        }

        Ok(())
    }

    /// Call an FFI function by name.
    pub fn call_by_name(
        ctx: &mut FFIContext,
        state: &mut VmState,
        name: &str,
        location: SourceLocation,
    ) -> OuroResult<()> {
        let id = ctx.registry.name_map.get(&name.to_lowercase())
            .copied()
            .ok_or_else(|| OuroError::FFI {
                message: format!("FFI function '{}' not found", name),
                location: location.clone(),
            })?;

        Self::call_by_id(ctx, state, id, location)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dynamic Library Loading
// ═══════════════════════════════════════════════════════════════════════════════

/// Manager for dynamically loaded external libraries.
///
/// This provides a safe way to load and call functions from shared libraries
/// (.so on Linux, .dll on Windows, .dylib on macOS).
#[derive(Default)]
pub struct DynamicLibraryManager {
    /// Loaded libraries by name.
    #[cfg(feature = "dynamic-ffi")]
    libraries: HashMap<String, Arc<Library>>,
    #[cfg(not(feature = "dynamic-ffi"))]
    libraries: HashMap<String, ()>,
    /// Library search paths.
    search_paths: Vec<PathBuf>,
}

impl DynamicLibraryManager {
    /// Create a new library manager.
    pub fn new() -> Self {
        let mut manager = Self {
            libraries: HashMap::new(),
            search_paths: Vec::new(),
        };

        // Add default search paths
        #[cfg(target_os = "linux")]
        {
            manager.search_paths.push(PathBuf::from("/usr/lib"));
            manager.search_paths.push(PathBuf::from("/usr/local/lib"));
            manager.search_paths.push(PathBuf::from("/lib"));
        }

        #[cfg(target_os = "macos")]
        {
            manager.search_paths.push(PathBuf::from("/usr/lib"));
            manager.search_paths.push(PathBuf::from("/usr/local/lib"));
        }

        #[cfg(target_os = "windows")]
        {
            if let Ok(sys_root) = std::env::var("SYSTEMROOT") {
                manager.search_paths.push(PathBuf::from(format!("{}\\System32", sys_root)));
            }
        }

        // Add current directory
        if let Ok(cwd) = std::env::current_dir() {
            manager.search_paths.push(cwd);
        }

        manager
    }

    /// Add a search path for libraries.
    pub fn add_search_path(&mut self, path: impl Into<PathBuf>) {
        self.search_paths.push(path.into());
    }

    /// Find a library file by name.
    fn find_library(&self, name: &str) -> Option<PathBuf> {
        // Try different file extensions based on platform
        let extensions = if cfg!(target_os = "windows") {
            vec![".dll", ""]
        } else if cfg!(target_os = "macos") {
            vec![".dylib", ".so", ""]
        } else {
            vec![".so", ""]
        };

        // Try with "lib" prefix on Unix
        let prefixes = if cfg!(target_os = "windows") {
            vec!["", "lib"]
        } else {
            vec!["lib", ""]
        };

        for path in &self.search_paths {
            for prefix in &prefixes {
                for ext in &extensions {
                    let lib_name = format!("{}{}{}", prefix, name, ext);
                    let full_path = path.join(&lib_name);
                    if full_path.exists() {
                        return Some(full_path);
                    }
                }
            }
        }

        // Try as-is (for full paths)
        let direct_path = PathBuf::from(name);
        if direct_path.exists() {
            return Some(direct_path);
        }

        None
    }

    /// Load a library by name.
    #[cfg(feature = "dynamic-ffi")]
    pub fn load(&mut self, name: &str) -> OuroResult<()> {
        if self.libraries.contains_key(name) {
            return Ok(()); // Already loaded
        }

        let path = self.find_library(name).ok_or_else(|| OuroError::FFI {
            message: format!("Library not found: {}", name),
            location: SourceLocation::default(),
        })?;

        unsafe {
            let library = Library::new(&path).map_err(|e| OuroError::FFI {
                message: format!("Failed to load library '{}': {}", path.display(), e),
                location: SourceLocation::default(),
            })?;

            self.libraries.insert(name.to_string(), Arc::new(library));
        }

        Ok(())
    }

    /// Load a library (stub for when dynamic-ffi feature is disabled).
    #[cfg(not(feature = "dynamic-ffi"))]
    pub fn load(&mut self, name: &str) -> OuroResult<()> {
        Err(OuroError::FFI {
            message: format!(
                "Dynamic FFI is disabled. Cannot load library '{}'. \
                 Recompile with --features dynamic-ffi to enable.",
                name
            ),
            location: SourceLocation::default(),
        })
    }

    /// Unload a library.
    pub fn unload(&mut self, name: &str) -> bool {
        self.libraries.remove(name).is_some()
    }

    /// Check if a library is loaded.
    pub fn is_loaded(&self, name: &str) -> bool {
        self.libraries.contains_key(name)
    }

    /// Get list of loaded libraries.
    pub fn loaded_libraries(&self) -> Vec<&str> {
        self.libraries.keys().map(|s| s.as_str()).collect()
    }

    /// Call a function from a loaded library.
    ///
    /// # Safety
    ///
    /// This is inherently unsafe as it calls external code. The caller must ensure:
    /// - The function exists and has the expected signature
    /// - Arguments are correctly formatted
    /// - The function is safe to call
    #[cfg(feature = "dynamic-ffi")]
    pub unsafe fn call_extern<T, F>(&self, library: &str, symbol: &str) -> OuroResult<Symbol<F>>
    where
        F: Copy,
    {
        let lib = self.libraries.get(library).ok_or_else(|| OuroError::FFI {
            message: format!("Library not loaded: {}", library),
            location: SourceLocation::default(),
        })?;

        let sym: Symbol<F> = lib.get(symbol.as_bytes()).map_err(|e| OuroError::FFI {
            message: format!("Symbol '{}' not found in '{}': {}", symbol, library, e),
            location: SourceLocation::default(),
        })?;

        Ok(sym)
    }
}

impl std::fmt::Debug for DynamicLibraryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicLibraryManager")
            .field("loaded_libraries", &self.libraries.keys().collect::<Vec<_>>())
            .field("search_paths", &self.search_paths)
            .finish()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FFI Integration with Dynamic Loading
// ═══════════════════════════════════════════════════════════════════════════════

/// Extended FFI context with dynamic library support.
pub struct ExtendedFFIContext {
    /// Base FFI context.
    pub base: FFIContext,
    /// Dynamic library manager.
    pub library_manager: DynamicLibraryManager,
}

impl ExtendedFFIContext {
    /// Create a new extended FFI context.
    pub fn new() -> Self {
        Self {
            base: FFIContext::with_stdlib(),
            library_manager: DynamicLibraryManager::new(),
        }
    }

    /// Load a library and register its functions.
    pub fn load_library(&mut self, name: &str) -> OuroResult<()> {
        self.library_manager.load(name)
    }

    /// Register an FFI function from a declaration.
    ///
    /// This creates a wrapper that will call the external function when invoked.
    pub fn register_from_declaration(
        &mut self,
        decl: &crate::parser::FFIDeclaration,
    ) -> OuroResult<u32> {
        let sig = decl.signature.clone();
        let _symbol = decl.symbol_name.clone().unwrap_or_else(|| sig.name.clone());
        let library = sig.library.clone();

        // Ensure library is loaded
        if !self.library_manager.is_loaded(&library) {
            self.library_manager.load(&library)?;
        }

        // Create a stub implementation that returns placeholder values
        // Real FFI calls require unsafe and platform-specific code
        let id = self.base.registry.register(sig, move |_state, _args| {
            // This is a stub - actual implementation would use libffi or similar
            // to call the external function with proper argument marshalling
            Err(OuroError::FFI {
                message: format!(
                    "Dynamic FFI call not implemented. Use built-in functions or \
                     implement native bridge for library '{}'",
                    library
                ),
                location: SourceLocation::default(),
            })
        });

        Ok(id)
    }
}

impl Default for ExtendedFFIContext {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_types::Memory;

    #[test]
    fn test_ffi_registry_register() {
        let mut registry = FFIRegistry::new();

        let sig = FFISignature::new("test_func", "test_lib")
            .param("a", FFIType::U64)
            .param("b", FFIType::U64)
            .returns_type(FFIType::U64)
            .pure();

        let id = registry.register(sig, |_state, args| {
            let sum = args.iter().map(|v| v.val).sum::<u64>();
            Ok(vec![Value::new(sum)])
        });

        assert_eq!(id, 0);
        assert!(registry.contains("test_func"));
        assert!(registry.contains("test_lib::test_func"));
    }

    #[test]
    fn test_ffi_signature_stack_size() {
        let sig = FFISignature::new("func", "lib")
            .param("a", FFIType::U64)
            .param("b", FFIType::Str) // Str is 2 stack slots
            .returns_type(FFIType::U64);

        assert_eq!(sig.input_stack_size(), 3);
        assert_eq!(sig.output_stack_size(), 1);
    }

    #[test]
    fn test_ffi_context_stdlib() {
        let ctx = FFIContext::with_stdlib();

        assert!(ctx.registry.contains("clock"));
        assert!(ctx.registry.contains("random"));
        assert!(ctx.registry.contains("sleep"));
    }

    #[test]
    fn test_ffi_context_buffer() {
        let mut ctx = FFIContext::new();

        let handle = ctx.alloc_buffer(1024);
        assert_eq!(handle, 0);

        let buffer = ctx.get_buffer(handle);
        assert!(buffer.is_some());
        assert_eq!(buffer.unwrap().len(), 1024);
    }

    #[test]
    fn test_ffi_context_external_data() {
        let mut ctx = FFIContext::new();

        let handle = ctx.store_external::<String>("hello".to_string());

        let data: Option<&String> = ctx.get_external(handle);
        assert!(data.is_some());
        assert_eq!(data.unwrap(), "hello");

        assert!(ctx.remove_external(handle));
        assert!(ctx.get_external::<String>(handle).is_none());
    }

    #[test]
    fn test_ffi_type_parse() {
        assert_eq!(FFIType::from_str("u64"), Some(FFIType::U64));
        assert_eq!(FFIType::from_str("i32"), Some(FFIType::I32));
        assert_eq!(FFIType::from_str("bool"), Some(FFIType::Bool));
        assert_eq!(FFIType::from_str("ptr"), Some(FFIType::Ptr));
        assert_eq!(FFIType::from_str("VOID"), Some(FFIType::Void));
        assert_eq!(FFIType::from_str("unknown"), None);
    }

    #[test]
    fn test_ffi_effect_is_pure() {
        assert!(FFIEffect::Pure.is_pure());
        assert!(!FFIEffect::IO.is_pure());
        assert!(!FFIEffect::Reads.is_pure());
    }

    #[test]
    fn test_ffi_call() {
        let mut ctx = FFIContext::new();

        // Register a simple add function
        ctx.registry.register(
            FFISignature::new("add", "math")
                .param("a", FFIType::U64)
                .param("b", FFIType::U64)
                .returns_type(FFIType::U64)
                .pure(),
            |_state, args| {
                let a = args.get(0).map(|v| v.val).unwrap_or(0);
                let b = args.get(1).map(|v| v.val).unwrap_or(0);
                Ok(vec![Value::new(a + b)])
            },
        );

        let mut state = VmState::new(Memory::new());
        state.stack.push(Value::new(10));
        state.stack.push(Value::new(20));

        FFICaller::call_by_name(&mut ctx, &mut state, "add", SourceLocation::default()).unwrap();

        assert_eq!(state.stack.len(), 1);
        assert_eq!(state.stack[0].val, 30);
    }
}
