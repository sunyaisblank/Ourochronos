//! Tooling layer for OUROCHRONOS.
//!
//! This module provides developer tools including:
//!
//! - **LSP**: Language Server Protocol implementation
//! - **Debugger**: Interactive debugging with breakpoints and watchpoints
//! - **Profiler**: Performance profiling and optimization hints
//! - **REPL**: Interactive Read-Eval-Print Loop

pub mod lsp;
pub mod debugger;
pub mod profiler;
pub mod repl;

// Re-export from lsp
pub use lsp::{
    LanguageAnalyzer, Diagnostic, Severity, CompletionItem, CompletionKind,
    HoverInfo, Document, Location, DocumentSymbol, SymbolKind,
    SemanticToken, SemanticTokenType, DiagnosticRelated,
};

// Re-export from debugger
pub use debugger::{
    Debugger, DebugEvent, EpochSnapshot, Breakpoint, Watchpoint,
    WatchType, BreakCondition,
};

// Re-export from profiler
pub use profiler::{
    Profiler, ProfilerConfig, EpochProfile, InstructionStats,
    MemoryProfile, TemporalStats, ProfileSummary, OptimizationHints,
};

// Re-export from repl
pub use repl::{Repl, ReplConfig};
