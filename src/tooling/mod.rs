//! Tooling layer for OUROCHRONOS.
//!
//! This module provides developer tools including:
//!
//! - **LSP**: Language Server Protocol implementation
//! - **REPL**: Interactive Read-Eval-Print Loop

mod frontend;
pub mod lsp;
pub mod repl;

// Re-export from lsp
pub use lsp::{
    CompletionItem, CompletionKind, Diagnostic, DiagnosticRelated, Document, DocumentSymbol,
    HoverInfo, LanguageAnalyzer, Location, SemanticToken, SemanticTokenType, Severity, SymbolKind,
    MAX_LSP_ANALYSIS_UNITS, MAX_LSP_DOCUMENTS, MAX_LSP_SOURCE_BYTES,
};

// Re-export from repl
pub use repl::{Repl, ReplConfig};
