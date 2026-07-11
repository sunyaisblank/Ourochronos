//! Tooling layer for OUROCHRONOS.
//!
//! This module provides developer tools including:
//!
//! - **LSP**: Language Server Protocol implementation
//! - **Debugger**: Interactive debugging with breakpoints and watchpoints
//! - **REPL**: Interactive Read-Eval-Print Loop

pub mod lsp;
pub mod debugger;
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

// Re-export from repl
pub use repl::{Repl, ReplConfig};
