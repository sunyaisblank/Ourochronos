//! Comprehensive error types for OUROCHRONOS.
//!
//! This module provides a unified error hierarchy for the language,
//! enabling precise error handling, recovery, and debugging.
//!
//! # Error Categories
//!
//! - **Runtime Errors**: Occur during VM execution (stack, memory, arithmetic)
//! - **Temporal Errors**: Related to oracle/prophecy and fixed-point computation
//! - **Parse Errors**: Lexing and parsing failures
//! - **Type Errors**: Static type checking failures
//!
//! # Design Principles
//!
//! 1. **First-Principles**: Errors derive from primitive invariants
//! 2. **Recoverability**: Each error specifies if recovery is possible
//! 3. **Context-Rich**: Errors carry location and diagnostic information
//! 4. **Lightweight**: Minimal overhead when not triggered

use std::fmt;
use super::address::Address;

/// Configuration for error handling behavior.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorConfig {
    /// How to handle division by zero.
    pub division_by_zero: DivisionByZeroPolicy,
    /// How to handle memory bounds violations.
    pub memory_bounds: BoundsPolicy,
    /// How to handle stack underflow.
    pub stack_underflow: StackPolicy,
    /// Maximum stack depth (0 = unlimited).
    pub max_stack_depth: usize,
    /// Whether to track error provenance.
    pub track_provenance: bool,
}

impl Default for ErrorConfig {
    fn default() -> Self {
        Self {
            division_by_zero: DivisionByZeroPolicy::ReturnZero,
            memory_bounds: BoundsPolicy::Wrap,
            stack_underflow: StackPolicy::Error,
            max_stack_depth: 0,
            track_provenance: true,
        }
    }
}

impl ErrorConfig {
    /// Strict mode: all violations produce errors.
    pub fn strict() -> Self {
        Self {
            division_by_zero: DivisionByZeroPolicy::Error,
            memory_bounds: BoundsPolicy::Error,
            stack_underflow: StackPolicy::Error,
            max_stack_depth: 10_000,
            track_provenance: true,
        }
    }

    /// Permissive mode: maximum compatibility, silent handling.
    pub fn permissive() -> Self {
        Self {
            division_by_zero: DivisionByZeroPolicy::ReturnZero,
            memory_bounds: BoundsPolicy::Wrap,
            stack_underflow: StackPolicy::ReturnZero,
            max_stack_depth: 0,
            track_provenance: false,
        }
    }
}

/// Policy for handling division by zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivisionByZeroPolicy {
    /// Return zero (current behavior, maintains backward compatibility).
    ReturnZero,
    /// Produce a runtime error.
    Error,
    /// Return a configurable sentinel value.
    Sentinel(u64),
}

/// Policy for handling memory bounds violations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundsPolicy {
    /// Wrap addresses modulo memory size (permissive).
    Wrap,
    /// Produce a runtime error (strict).
    Error,
    /// Clamp to valid range.
    Clamp,
}

/// Policy for handling stack underflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackPolicy {
    /// Produce a runtime error.
    Error,
    /// Return zero value (permissive).
    ReturnZero,
}

/// Source location for error reporting.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SourceLocation {
    /// Line number (1-indexed).
    pub line: usize,
    /// Column number (1-indexed).
    pub column: usize,
    /// Statement index within the program.
    pub stmt_index: usize,
    /// Optional source file name.
    pub file: Option<String>,
}

impl SourceLocation {
    pub fn new(line: usize, column: usize) -> Self {
        Self {
            line,
            column,
            stmt_index: 0,
            file: None,
        }
    }

    pub fn at_stmt(stmt_index: usize) -> Self {
        Self {
            line: 0,
            column: 0,
            stmt_index,
            file: None,
        }
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref file) = self.file {
            write!(f, "{}:", file)?;
        }
        if self.line > 0 {
            write!(f, "{}:{}", self.line, self.column)
        } else {
            write!(f, "stmt {}", self.stmt_index)
        }
    }
}

/// Comprehensive error type for OUROCHRONOS.
#[derive(Debug, Clone)]
pub enum OuroError {
    // ═══════════════════════════════════════════════════════════════════
    // Runtime Errors
    // ═══════════════════════════════════════════════════════════════════

    /// Stack underflow: attempted to pop from empty stack.
    StackUnderflow {
        operation: String,
        required: usize,
        available: usize,
        location: SourceLocation,
    },

    /// Stack overflow: exceeded maximum stack depth.
    StackOverflow {
        max_depth: usize,
        location: SourceLocation,
    },

    /// Division by zero.
    DivisionByZero {
        dividend: u64,
        location: SourceLocation,
    },

    /// Modulo by zero.
    ModuloByZero {
        dividend: u64,
        location: SourceLocation,
    },

    /// Memory address out of bounds.
    MemoryBoundsViolation {
        address: u64,
        max_address: Address,
        operation: MemoryOperation,
        location: SourceLocation,
    },

    /// Instruction limit exceeded (infinite loop protection).
    InstructionLimitExceeded {
        limit: u64,
        location: SourceLocation,
    },

    /// Invalid quote/quotation ID.
    InvalidQuoteId {
        id: usize,
        max_id: usize,
        location: SourceLocation,
    },

    /// Assertion failure.
    AssertionFailed {
        message: String,
        location: SourceLocation,
    },

    // ═══════════════════════════════════════════════════════════════════
    // Temporal Errors
    // ═══════════════════════════════════════════════════════════════════

    /// Explicit paradox triggered by PARADOX instruction.
    ExplicitParadox {
        location: SourceLocation,
    },

    /// Temporal consistency violation (fixed-point not achieved).
    TemporalInconsistency {
        epoch: usize,
        differing_cells: Vec<Address>,
        message: String,
    },

    /// Oscillation detected (state cycles without convergence).
    Oscillation {
        period: usize,
        oscillating_cells: Vec<Address>,
        epoch: usize,
    },

    /// Divergence detected (state grows unboundedly).
    Divergence {
        epoch: usize,
        diverging_cells: Vec<Address>,
    },

    /// Maximum epochs exceeded.
    MaxEpochsExceeded {
        max_epochs: usize,
    },

    /// Negative causal loop detected (grandfather paradox).
    NegativeCausalLoop {
        involved_addresses: Vec<Address>,
        epoch: usize,
    },

    // ═══════════════════════════════════════════════════════════════════
    // Parse Errors
    // ═══════════════════════════════════════════════════════════════════

    /// Unexpected token during parsing.
    UnexpectedToken {
        expected: String,
        found: String,
        location: SourceLocation,
    },

    /// Unexpected end of input.
    UnexpectedEof {
        context: String,
        location: SourceLocation,
    },

    /// Unknown opcode or keyword.
    UnknownKeyword {
        keyword: String,
        suggestions: Vec<String>,
        location: SourceLocation,
    },

    /// Undefined procedure call.
    UndefinedProcedure {
        name: String,
        location: SourceLocation,
    },

    /// Invalid numeric literal.
    InvalidNumber {
        text: String,
        location: SourceLocation,
    },

    /// Unclosed block or string.
    Unclosed {
        construct: String,
        opened_at: SourceLocation,
    },

    // ═══════════════════════════════════════════════════════════════════
    // Type Errors
    // ═══════════════════════════════════════════════════════════════════

    /// Effect violation: procedure declared PURE but has side effects.
    EffectViolation {
        procedure: String,
        declared_effect: String,
        actual_effect: String,
        location: SourceLocation,
    },

    /// Temporal taint violation.
    TemporalTaintViolation {
        message: String,
        location: SourceLocation,
    },

    /// Stack effect mismatch.
    StackEffectMismatch {
        expected_inputs: usize,
        expected_outputs: usize,
        actual_inputs: usize,
        actual_outputs: usize,
        location: SourceLocation,
    },

    /// Linear type violation: value was duplicated or dropped improperly.
    LinearViolation {
        message: String,
        value_description: String,
        location: SourceLocation,
    },

    // ═══════════════════════════════════════════════════════════════════
    // Data Structure Errors
    // ═══════════════════════════════════════════════════════════════════

    /// Invalid data structure handle.
    InvalidHandle {
        handle_type: String,
        handle: u64,
        max_handle: u64,
        location: SourceLocation,
    },

    /// Data structure is empty when operation requires elements.
    EmptyStructure {
        structure_type: String,
        operation: String,
        location: SourceLocation,
    },

    /// Index out of bounds for data structure.
    IndexOutOfBounds {
        structure_type: String,
        index: u64,
        length: u64,
        location: SourceLocation,
    },

    /// Key not found in hash table.
    KeyNotFound {
        key: u64,
        location: SourceLocation,
    },

    // ═══════════════════════════════════════════════════════════════════
    // FFI Errors
    // ═══════════════════════════════════════════════════════════════════

    /// FFI function not found.
    FFIFunctionNotFound {
        name: String,
        location: SourceLocation,
    },

    /// FFI argument mismatch.
    FFIArgumentMismatch {
        function: String,
        expected: usize,
        got: usize,
        location: SourceLocation,
    },

    /// FFI type conversion error.
    FFITypeConversion {
        expected: String,
        got: String,
        location: SourceLocation,
    },

    /// FFI library not loaded.
    FFILibraryNotLoaded {
        library: String,
        location: SourceLocation,
    },

    /// FFI call failed.
    FFICallFailed {
        function: String,
        message: String,
        location: SourceLocation,
    },

    /// Generic FFI error.
    FFI {
        message: String,
        location: SourceLocation,
    },

    // ═══════════════════════════════════════════════════════════════════
    // I/O Errors
    // ═══════════════════════════════════════════════════════════════════

    /// File/network I/O error.
    IO {
        operation: String,
        path: Option<String>,
        message: String,
        location: SourceLocation,
    },

    /// Invalid file handle.
    InvalidFileHandle {
        handle: u64,
        location: SourceLocation,
    },

    /// Invalid socket handle.
    InvalidSocketHandle {
        handle: u64,
        location: SourceLocation,
    },

    /// Invalid buffer handle.
    InvalidBufferHandle {
        handle: u64,
        location: SourceLocation,
    },

    /// Network connection error.
    ConnectionFailed {
        host: String,
        port: u16,
        message: String,
        location: SourceLocation,
    },

    /// Network timeout.
    Timeout {
        operation: String,
        timeout_ms: u64,
        location: SourceLocation,
    },

    // ═══════════════════════════════════════════════════════════════════
    // Internal Errors
    // ═══════════════════════════════════════════════════════════════════

    /// Internal error (should not occur in normal operation).
    Internal {
        message: String,
        location: Option<SourceLocation>,
    },
}

/// Memory operation type for error context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOperation {
    Read,
    Write,
    Oracle,
    Prophecy,
    Index,
    Store,
    Pack,
    Unpack,
}

impl fmt::Display for MemoryOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryOperation::Read => write!(f, "read"),
            MemoryOperation::Write => write!(f, "write"),
            MemoryOperation::Oracle => write!(f, "ORACLE"),
            MemoryOperation::Prophecy => write!(f, "PROPHECY"),
            MemoryOperation::Index => write!(f, "INDEX"),
            MemoryOperation::Store => write!(f, "STORE"),
            MemoryOperation::Pack => write!(f, "PACK"),
            MemoryOperation::Unpack => write!(f, "UNPACK"),
        }
    }
}

impl fmt::Display for OuroError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // Runtime Errors
            OuroError::StackUnderflow { operation, required, available, location } => {
                write!(f, "[{}] Stack underflow in {}: requires {} values, but only {} available",
                       location, operation, required, available)
            }
            OuroError::StackOverflow { max_depth, location } => {
                write!(f, "[{}] Stack overflow: exceeded maximum depth of {}", location, max_depth)
            }
            OuroError::DivisionByZero { dividend, location } => {
                write!(f, "[{}] Division by zero: {} / 0", location, dividend)
            }
            OuroError::ModuloByZero { dividend, location } => {
                write!(f, "[{}] Modulo by zero: {} % 0", location, dividend)
            }
            OuroError::MemoryBoundsViolation { address, max_address, operation, location } => {
                write!(f, "[{}] Memory bounds violation: {} at address {} (max: {})",
                       location, operation, address, max_address)
            }
            OuroError::InstructionLimitExceeded { limit, location } => {
                write!(f, "[{}] Instruction limit exceeded: {} instructions", location, limit)
            }
            OuroError::InvalidQuoteId { id, max_id, location } => {
                write!(f, "[{}] Invalid quote ID: {} (valid range: 0..{})", location, id, max_id)
            }
            OuroError::AssertionFailed { message, location } => {
                write!(f, "[{}] Assertion failed: {}", location, message)
            }

            // Temporal Errors
            OuroError::ExplicitParadox { location } => {
                write!(f, "[{}] Explicit paradox triggered", location)
            }
            OuroError::TemporalInconsistency { epoch, differing_cells, message } => {
                write!(f, "[epoch {}] Temporal inconsistency: {} (cells: {:?})",
                       epoch, message, differing_cells)
            }
            OuroError::Oscillation { period, oscillating_cells, epoch } => {
                write!(f, "[epoch {}] Oscillation detected with period {} at cells {:?}",
                       epoch, period, oscillating_cells)
            }
            OuroError::Divergence { epoch, diverging_cells } => {
                write!(f, "[epoch {}] Divergence detected at cells {:?}", epoch, diverging_cells)
            }
            OuroError::MaxEpochsExceeded { max_epochs } => {
                write!(f, "Maximum epochs exceeded: {}", max_epochs)
            }
            OuroError::NegativeCausalLoop { involved_addresses, epoch } => {
                write!(f, "[epoch {}] Negative causal loop (grandfather paradox) at addresses {:?}",
                       epoch, involved_addresses)
            }

            // Parse Errors
            OuroError::UnexpectedToken { expected, found, location } => {
                write!(f, "[{}] Unexpected token: expected {}, found '{}'", location, expected, found)
            }
            OuroError::UnexpectedEof { context, location } => {
                write!(f, "[{}] Unexpected end of input while parsing {}", location, context)
            }
            OuroError::UnknownKeyword { keyword, suggestions, location } => {
                let mut msg = format!("[{}] Unknown keyword: '{}'", location, keyword);
                if !suggestions.is_empty() {
                    msg.push_str(&format!(". Did you mean: {}?", suggestions.join(", ")));
                }
                write!(f, "{}", msg)
            }
            OuroError::UndefinedProcedure { name, location } => {
                write!(f, "[{}] Undefined procedure: '{}'", location, name)
            }
            OuroError::InvalidNumber { text, location } => {
                write!(f, "[{}] Invalid numeric literal: '{}'", location, text)
            }
            OuroError::Unclosed { construct, opened_at } => {
                write!(f, "[{}] Unclosed {}", opened_at, construct)
            }

            // Type Errors
            OuroError::EffectViolation { procedure, declared_effect, actual_effect, location } => {
                write!(f, "[{}] Effect violation in '{}': declared {} but performs {}",
                       location, procedure, declared_effect, actual_effect)
            }
            OuroError::TemporalTaintViolation { message, location } => {
                write!(f, "[{}] Temporal taint violation: {}", location, message)
            }
            OuroError::StackEffectMismatch { expected_inputs, expected_outputs,
                                              actual_inputs, actual_outputs, location } => {
                write!(f, "[{}] Stack effect mismatch: expected ({} -- {}) but got ({} -- {})",
                       location, expected_inputs, expected_outputs, actual_inputs, actual_outputs)
            }
            OuroError::LinearViolation { message, value_description, location } => {
                write!(f, "[{}] Linear type violation: {} (value: {})",
                       location, message, value_description)
            }

            // Data Structure Errors
            OuroError::InvalidHandle { handle_type, handle, max_handle, location } => {
                write!(f, "[{}] Invalid {} handle: {} (valid range: 0..{})",
                       location, handle_type, handle, max_handle)
            }
            OuroError::EmptyStructure { structure_type, operation, location } => {
                write!(f, "[{}] Cannot {} empty {}", location, operation, structure_type)
            }
            OuroError::IndexOutOfBounds { structure_type, index, length, location } => {
                write!(f, "[{}] {} index out of bounds: {} (length: {})",
                       location, structure_type, index, length)
            }
            OuroError::KeyNotFound { key, location } => {
                write!(f, "[{}] Key not found: {}", location, key)
            }

            // FFI Errors
            OuroError::FFIFunctionNotFound { name, location } => {
                write!(f, "[{}] FFI function not found: '{}'", location, name)
            }
            OuroError::FFIArgumentMismatch { function, expected, got, location } => {
                write!(f, "[{}] FFI argument mismatch in '{}': expected {} arguments, got {}",
                       location, function, expected, got)
            }
            OuroError::FFITypeConversion { expected, got, location } => {
                write!(f, "[{}] FFI type conversion failed: expected {}, got {}",
                       location, expected, got)
            }
            OuroError::FFILibraryNotLoaded { library, location } => {
                write!(f, "[{}] FFI library not loaded: '{}'", location, library)
            }
            OuroError::FFICallFailed { function, message, location } => {
                write!(f, "[{}] FFI call to '{}' failed: {}", location, function, message)
            }
            OuroError::FFI { message, location } => {
                write!(f, "[{}] FFI error: {}", location, message)
            }

            // I/O Errors
            OuroError::IO { operation, path, message, location } => {
                if let Some(p) = path {
                    write!(f, "[{}] I/O error during {} on '{}': {}", location, operation, p, message)
                } else {
                    write!(f, "[{}] I/O error during {}: {}", location, operation, message)
                }
            }
            OuroError::InvalidFileHandle { handle, location } => {
                write!(f, "[{}] Invalid file handle: {}", location, handle)
            }
            OuroError::InvalidSocketHandle { handle, location } => {
                write!(f, "[{}] Invalid socket handle: {}", location, handle)
            }
            OuroError::InvalidBufferHandle { handle, location } => {
                write!(f, "[{}] Invalid buffer handle: {}", location, handle)
            }
            OuroError::ConnectionFailed { host, port, message, location } => {
                write!(f, "[{}] Connection to {}:{} failed: {}", location, host, port, message)
            }
            OuroError::Timeout { operation, timeout_ms, location } => {
                write!(f, "[{}] Operation '{}' timed out after {}ms", location, operation, timeout_ms)
            }

            // Internal Errors
            OuroError::Internal { message, location } => {
                if let Some(loc) = location {
                    write!(f, "[{}] Internal error: {}", loc, message)
                } else {
                    write!(f, "Internal error: {}", message)
                }
            }
        }
    }
}

impl std::error::Error for OuroError {}

impl OuroError {
    /// Check if this error is recoverable.
    pub fn is_recoverable(&self) -> bool {
        matches!(self,
            OuroError::DivisionByZero { .. } |
            OuroError::ModuloByZero { .. } |
            OuroError::StackUnderflow { .. } |
            OuroError::MemoryBoundsViolation { .. }
        )
    }

    /// Get the error category.
    pub fn category(&self) -> ErrorCategory {
        match self {
            OuroError::StackUnderflow { .. } |
            OuroError::StackOverflow { .. } |
            OuroError::DivisionByZero { .. } |
            OuroError::ModuloByZero { .. } |
            OuroError::MemoryBoundsViolation { .. } |
            OuroError::InstructionLimitExceeded { .. } |
            OuroError::InvalidQuoteId { .. } |
            OuroError::AssertionFailed { .. } => ErrorCategory::Runtime,

            OuroError::ExplicitParadox { .. } |
            OuroError::TemporalInconsistency { .. } |
            OuroError::Oscillation { .. } |
            OuroError::Divergence { .. } |
            OuroError::MaxEpochsExceeded { .. } |
            OuroError::NegativeCausalLoop { .. } => ErrorCategory::Temporal,

            OuroError::UnexpectedToken { .. } |
            OuroError::UnexpectedEof { .. } |
            OuroError::UnknownKeyword { .. } |
            OuroError::UndefinedProcedure { .. } |
            OuroError::InvalidNumber { .. } |
            OuroError::Unclosed { .. } => ErrorCategory::Parse,

            OuroError::EffectViolation { .. } |
            OuroError::TemporalTaintViolation { .. } |
            OuroError::StackEffectMismatch { .. } |
            OuroError::LinearViolation { .. } => ErrorCategory::Type,

            OuroError::InvalidHandle { .. } |
            OuroError::EmptyStructure { .. } |
            OuroError::IndexOutOfBounds { .. } |
            OuroError::KeyNotFound { .. } => ErrorCategory::Runtime,

            OuroError::FFIFunctionNotFound { .. } |
            OuroError::FFIArgumentMismatch { .. } |
            OuroError::FFITypeConversion { .. } |
            OuroError::FFILibraryNotLoaded { .. } |
            OuroError::FFICallFailed { .. } |
            OuroError::FFI { .. } => ErrorCategory::FFI,

            OuroError::IO { .. } |
            OuroError::InvalidFileHandle { .. } |
            OuroError::InvalidSocketHandle { .. } |
            OuroError::InvalidBufferHandle { .. } |
            OuroError::ConnectionFailed { .. } |
            OuroError::Timeout { .. } => ErrorCategory::IO,

            OuroError::Internal { .. } => ErrorCategory::Internal,
        }
    }

    /// Get the error code for programmatic handling.
    pub fn code(&self) -> u32 {
        match self {
            // Runtime: 1000-1999
            OuroError::StackUnderflow { .. } => 1001,
            OuroError::StackOverflow { .. } => 1002,
            OuroError::DivisionByZero { .. } => 1003,
            OuroError::ModuloByZero { .. } => 1004,
            OuroError::MemoryBoundsViolation { .. } => 1005,
            OuroError::InstructionLimitExceeded { .. } => 1006,
            OuroError::InvalidQuoteId { .. } => 1007,
            OuroError::AssertionFailed { .. } => 1008,

            // Temporal: 2000-2999
            OuroError::ExplicitParadox { .. } => 2001,
            OuroError::TemporalInconsistency { .. } => 2002,
            OuroError::Oscillation { .. } => 2003,
            OuroError::Divergence { .. } => 2004,
            OuroError::MaxEpochsExceeded { .. } => 2005,
            OuroError::NegativeCausalLoop { .. } => 2006,

            // Parse: 3000-3999
            OuroError::UnexpectedToken { .. } => 3001,
            OuroError::UnexpectedEof { .. } => 3002,
            OuroError::UnknownKeyword { .. } => 3003,
            OuroError::UndefinedProcedure { .. } => 3004,
            OuroError::InvalidNumber { .. } => 3005,
            OuroError::Unclosed { .. } => 3006,

            // Type: 4000-4999
            OuroError::EffectViolation { .. } => 4001,
            OuroError::TemporalTaintViolation { .. } => 4002,
            OuroError::StackEffectMismatch { .. } => 4003,
            OuroError::LinearViolation { .. } => 4004,

            // Data Structure: 5000-5999
            OuroError::InvalidHandle { .. } => 5001,
            OuroError::EmptyStructure { .. } => 5002,
            OuroError::IndexOutOfBounds { .. } => 5003,
            OuroError::KeyNotFound { .. } => 5004,

            // FFI: 6000-6999
            OuroError::FFIFunctionNotFound { .. } => 6001,
            OuroError::FFIArgumentMismatch { .. } => 6002,
            OuroError::FFITypeConversion { .. } => 6003,
            OuroError::FFILibraryNotLoaded { .. } => 6004,
            OuroError::FFICallFailed { .. } => 6005,
            OuroError::FFI { .. } => 6099,

            // I/O: 7000-7999
            OuroError::IO { .. } => 7001,
            OuroError::InvalidFileHandle { .. } => 7002,
            OuroError::InvalidSocketHandle { .. } => 7003,
            OuroError::InvalidBufferHandle { .. } => 7004,
            OuroError::ConnectionFailed { .. } => 7005,
            OuroError::Timeout { .. } => 7006,

            // Internal: 9000-9999
            OuroError::Internal { .. } => 9001,
        }
    }

    /// Get the source location if available.
    pub fn location(&self) -> Option<&SourceLocation> {
        match self {
            OuroError::StackUnderflow { location, .. } |
            OuroError::StackOverflow { location, .. } |
            OuroError::DivisionByZero { location, .. } |
            OuroError::ModuloByZero { location, .. } |
            OuroError::MemoryBoundsViolation { location, .. } |
            OuroError::InstructionLimitExceeded { location, .. } |
            OuroError::InvalidQuoteId { location, .. } |
            OuroError::AssertionFailed { location, .. } |
            OuroError::ExplicitParadox { location, .. } |
            OuroError::UnexpectedToken { location, .. } |
            OuroError::UnexpectedEof { location, .. } |
            OuroError::UnknownKeyword { location, .. } |
            OuroError::UndefinedProcedure { location, .. } |
            OuroError::InvalidNumber { location, .. } |
            OuroError::Unclosed { opened_at: location, .. } |
            OuroError::EffectViolation { location, .. } |
            OuroError::TemporalTaintViolation { location, .. } |
            OuroError::StackEffectMismatch { location, .. } |
            OuroError::LinearViolation { location, .. } |
            OuroError::InvalidHandle { location, .. } |
            OuroError::EmptyStructure { location, .. } |
            OuroError::IndexOutOfBounds { location, .. } |
            OuroError::KeyNotFound { location, .. } |
            OuroError::FFIFunctionNotFound { location, .. } |
            OuroError::FFIArgumentMismatch { location, .. } |
            OuroError::FFITypeConversion { location, .. } |
            OuroError::FFILibraryNotLoaded { location, .. } |
            OuroError::FFICallFailed { location, .. } |
            OuroError::FFI { location, .. } |
            OuroError::IO { location, .. } |
            OuroError::InvalidFileHandle { location, .. } |
            OuroError::InvalidSocketHandle { location, .. } |
            OuroError::InvalidBufferHandle { location, .. } |
            OuroError::ConnectionFailed { location, .. } |
            OuroError::Timeout { location, .. } => Some(location),

            OuroError::Internal { location, .. } => location.as_ref(),

            _ => None,
        }
    }
}

/// Error category for filtering and routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    Runtime,
    Temporal,
    Parse,
    Type,
    FFI,
    IO,
    Internal,
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCategory::Runtime => write!(f, "runtime"),
            ErrorCategory::Temporal => write!(f, "temporal"),
            ErrorCategory::Parse => write!(f, "parse"),
            ErrorCategory::Type => write!(f, "type"),
            ErrorCategory::FFI => write!(f, "ffi"),
            ErrorCategory::IO => write!(f, "io"),
            ErrorCategory::Internal => write!(f, "internal"),
        }
    }
}

/// Result type alias for OUROCHRONOS operations.
pub type OuroResult<T> = Result<T, OuroError>;

/// Collect multiple errors.
#[derive(Debug, Clone, Default)]
pub struct ErrorCollector {
    errors: Vec<OuroError>,
    warnings: Vec<String>,
}

impl ErrorCollector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_error(&mut self, error: OuroError) {
        self.errors.push(error);
    }

    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn errors(&self) -> &[OuroError] {
        &self.errors
    }

    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    pub fn into_result<T>(self, value: T) -> Result<T, Vec<OuroError>> {
        if self.errors.is_empty() {
            Ok(value)
        } else {
            Err(self.errors)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = OuroError::StackUnderflow {
            operation: "ADD".to_string(),
            required: 2,
            available: 1,
            location: SourceLocation::new(10, 5),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Stack underflow"));
        assert!(msg.contains("ADD"));
        assert!(msg.contains("10:5"));
    }

    #[test]
    fn test_error_category() {
        let runtime_err = OuroError::DivisionByZero {
            dividend: 42,
            location: SourceLocation::default(),
        };
        assert_eq!(runtime_err.category(), ErrorCategory::Runtime);

        let temporal_err = OuroError::Oscillation {
            period: 2,
            oscillating_cells: vec![0, 1],
            epoch: 5,
        };
        assert_eq!(temporal_err.category(), ErrorCategory::Temporal);
    }

    #[test]
    fn test_error_recoverability() {
        let recoverable = OuroError::DivisionByZero {
            dividend: 10,
            location: SourceLocation::default(),
        };
        assert!(recoverable.is_recoverable());

        let not_recoverable = OuroError::ExplicitParadox {
            location: SourceLocation::default(),
        };
        assert!(!not_recoverable.is_recoverable());
    }

    #[test]
    fn test_error_config() {
        let strict = ErrorConfig::strict();
        assert_eq!(strict.division_by_zero, DivisionByZeroPolicy::Error);

        let permissive = ErrorConfig::permissive();
        assert_eq!(permissive.division_by_zero, DivisionByZeroPolicy::ReturnZero);
    }

    #[test]
    fn test_error_collector() {
        let mut collector = ErrorCollector::new();
        assert!(!collector.has_errors());

        collector.add_error(OuroError::StackUnderflow {
            operation: "POP".to_string(),
            required: 1,
            available: 0,
            location: SourceLocation::default(),
        });
        assert!(collector.has_errors());
        assert_eq!(collector.errors().len(), 1);
    }
}
