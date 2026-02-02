//! Compiler infrastructure for OUROCHRONOS.
//!
//! This module provides a unified interface to the compiler pipeline:
//!
//! - **Lexer**: Tokenization with span tracking (`lexer/`)
//! - **Parser**: Domain-specific parsing (`parser/`)
//! - **AST**: Abstract syntax tree types (`ast/`)
//! - **Analysis**: Type checking, provenance, temporal graph (`analysis/`)
//!
//! # Compiler Pipeline
//!
//! ```text
//! Source → Tokenize → Parse → Analyze → Program
//!            ↓          ↓        ↓
//!         Spans     AST      Types/Effects
//! ```
//!
//! # Semantics Preservation
//!
//! The compiler phases form a semantics-preserving pipeline:
//! ```text
//! Semantics(Source) = Semantics(Parse(Lex(Source))) = Semantics(Analyze(Parse(Lex(Source))))
//! ```

pub mod lexer;
pub mod ast;
pub mod analysis;

// Re-export parser from top-level (already well-organized)
pub use crate::parser as parser;

// Re-export key types for convenience
pub use lexer::{Span, Token, SpannedToken, tokenize, tokenize_with_spans};
pub use crate::parser::{Parser, ParseError, ParseContext, DomainParser, DomainRegistry};
pub use ast::{OpCode, Stmt, Program, Procedure, Effect};
pub use analysis::{CompilerAnalysis, AnalysisResult};

use crate::types::TypeChecker;

/// Compilation error with source location.
#[derive(Debug, Clone)]
pub struct CompileError {
    /// Error category.
    pub kind: ErrorKind,
    /// Error message.
    pub message: String,
    /// Source location.
    pub span: Option<Span>,
    /// Help text.
    pub help: Option<String>,
    /// Related errors.
    pub related: Vec<CompileError>,
}

/// Categories of compilation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// Lexical error (invalid token).
    Lexer,
    /// Syntax error (parse failure).
    Parser,
    /// Type error (type mismatch, linearity violation).
    Type,
    /// Temporal error (causality violation).
    Temporal,
    /// Effect error (unauthorized side effect).
    Effect,
}

impl CompileError {
    /// Create a lexer error.
    pub fn lexer(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Lexer,
            message: message.into(),
            span: None,
            help: None,
            related: Vec::new(),
        }
    }

    /// Create a parser error.
    pub fn parser(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Parser,
            message: message.into(),
            span: None,
            help: None,
            related: Vec::new(),
        }
    }

    /// Create a type error.
    pub fn type_error(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Type,
            message: message.into(),
            span: None,
            help: None,
            related: Vec::new(),
        }
    }

    /// Create a temporal error.
    pub fn temporal(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Temporal,
            message: message.into(),
            span: None,
            help: None,
            related: Vec::new(),
        }
    }

    /// Add source span.
    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    /// Add help text.
    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(span) = &self.span {
            write!(f, "error[{:?}] at {}: {}", self.kind, span, self.message)
        } else {
            write!(f, "error[{:?}]: {}", self.kind, self.message)
        }
    }
}

impl std::error::Error for CompileError {}

/// Compilation result.
pub type CompileResult<T> = Result<T, CompileError>;

/// Compiler pipeline configuration.
#[derive(Debug, Clone)]
pub struct CompilerConfig {
    /// Enable type checking.
    pub type_check: bool,
    /// Enable temporal analysis.
    pub temporal_analysis: bool,
    /// Strict linearity enforcement (errors instead of warnings).
    pub strict_linearity: bool,
    /// Enable effect analysis.
    pub effect_analysis: bool,
    /// Maximum parse depth (prevent stack overflow).
    pub max_depth: usize,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            type_check: true,
            temporal_analysis: true,
            strict_linearity: true,
            effect_analysis: true,
            max_depth: 256,
        }
    }
}

/// Unified compiler pipeline.
///
/// Orchestrates lexing, parsing, and analysis phases.
pub struct Compiler {
    /// Configuration.
    pub config: CompilerConfig,
}

impl Compiler {
    /// Create a new compiler with default configuration.
    pub fn new() -> Self {
        Self {
            config: CompilerConfig::default(),
        }
    }

    /// Create a compiler with custom configuration.
    pub fn with_config(config: CompilerConfig) -> Self {
        Self { config }
    }

    /// Compile source code to a program.
    ///
    /// Runs the full pipeline: lex → parse → analyze.
    pub fn compile(&self, source: &str) -> CompileResult<CompiledProgram> {
        // Phase 1: Tokenize with spans
        let tokens = tokenize_with_spans(source);

        // Phase 2: Parse to AST
        let program = self.parse(&tokens)?;

        // Phase 3: Analyze
        let analysis = if self.config.type_check || self.config.temporal_analysis {
            Some(self.analyze(&program)?)
        } else {
            None
        };

        Ok(CompiledProgram {
            program,
            analysis,
            source: source.to_string(),
        })
    }

    /// Parse tokens to AST.
    fn parse(&self, tokens: &[SpannedToken]) -> CompileResult<Program> {
        // Convert spanned tokens to plain tokens for existing parser
        let plain_tokens: Vec<Token> = tokens.iter().map(|st| st.token.clone()).collect();

        let mut parser = Parser::new(&plain_tokens);
        parser.parse_program().map_err(|e| {
            CompileError::parser(e)
        })
    }

    /// Run analysis passes on the program.
    fn analyze(&self, program: &Program) -> CompileResult<AnalysisResult> {
        let mut result = AnalysisResult::default();

        // Type checking
        if self.config.type_check {
            let mut checker = TypeChecker::new();
            let type_result = checker.check(program);

            // Check for linearity violations if strict mode
            if self.config.strict_linearity {
                for warning in &type_result.warnings {
                    if warning.contains("linear") || warning.contains("ORACLE") {
                        return Err(CompileError::type_error(warning.clone())
                            .with_help("Linear values from ORACLE can only be used once"));
                    }
                }
            }

            result.type_result = Some(type_result);
        }

        // Temporal analysis
        if self.config.temporal_analysis {
            let tdg = crate::analysis::TemporalDependencyGraph::build(program);
            result.temporal_graph = Some(tdg);
        }

        Ok(result)
    }
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of compilation.
#[derive(Debug, Clone)]
pub struct CompiledProgram {
    /// The parsed program.
    pub program: Program,
    /// Analysis results (if enabled).
    pub analysis: Option<AnalysisResult>,
    /// Original source code.
    pub source: String,
}

impl CompiledProgram {
    /// Check if the program is pure (no temporal operations).
    pub fn is_pure(&self) -> bool {
        if let Some(ref analysis) = self.analysis {
            if let Some(ref type_result) = analysis.type_result {
                return type_result.effects.is_pure();
            }
        }
        // Conservative: assume not pure if we don't know
        false
    }

    /// Get the program.
    pub fn into_program(self) -> Program {
        self.program
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiler_basic() {
        let compiler = Compiler::new();
        let result = compiler.compile("1 2 ADD");
        assert!(result.is_ok());

        let compiled = result.unwrap();
        assert!(compiled.is_pure());
    }

    #[test]
    fn test_compiler_temporal() {
        let compiler = Compiler::new();
        let result = compiler.compile("0 ORACLE 1 ADD 0 PROPHECY");
        assert!(result.is_ok());

        let compiled = result.unwrap();
        assert!(!compiled.is_pure());
    }

    #[test]
    fn test_compile_error_display() {
        let err = CompileError::parser("unexpected token")
            .with_span(Span::new(5, 12, 45, 3))
            .with_help("expected a number");

        let display = format!("{}", err);
        assert!(display.contains("5:12"));
        assert!(display.contains("unexpected token"));
    }

    #[test]
    fn test_compiler_config() {
        let config = CompilerConfig {
            type_check: false,
            temporal_analysis: false,
            strict_linearity: false,
            effect_analysis: false,
            max_depth: 128,
        };

        let compiler = Compiler::with_config(config);
        let result = compiler.compile("1 2 ADD");
        assert!(result.is_ok());

        // Analysis should be None when disabled
        let compiled = result.unwrap();
        assert!(compiled.analysis.is_none());
    }
}
