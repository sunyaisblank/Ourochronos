//! Tooling layer for OUROCHRONOS.
//!
//! This module provides developer tools including:
//!
//! - **LSP**: Language Server Protocol implementation
//! - **REPL**: Interactive Read-Eval-Print Loop

pub mod lsp;
pub mod repl;

// Re-export from lsp
pub use lsp::{
    LanguageAnalyzer, Diagnostic, Severity, CompletionItem, CompletionKind,
    HoverInfo, Document, Location, DocumentSymbol, SymbolKind,
    SemanticToken, SemanticTokenType, DiagnosticRelated,
};

// Re-export from repl
pub use repl::{Repl, ReplConfig};
