//! Lexer and Parser for OUROCHRONOS.
//!
//! Syntax:
//! - Numbers: 42, 0xFF (hex), 0b1010 (binary)
//! - Opcodes: ADD, SUB, ORACLE, PROPHECY, etc.
//! - Control flow: IF { } ELSE { }, WHILE { cond } { body }
//! - Comments: # line comment
//!
//! # Architecture
//!
//! The parser is organized into domain-specific modules for maintainability:
//! - `domain`: Domain parser trait for modular keyword handling
//! - `stack_ops`: Stack manipulation (DUP, SWAP, ROT, etc.)
//! - `arithmetic`: Arithmetic and comparison operations
//! - `temporal`: ORACLE, PROPHECY, PRESENT, PARADOX
//! - `data_structures`: VEC, HASH, SET operations
//! - `io_ops`: File, buffer, network, system operations
//! - `string_ops`: String manipulation
//! - `memory_ops`: Computed memory access (INDEX, STORE, PACK, UNPACK)
//! - `keyword_map`: Fast keyword-to-domain lookup

// Domain parser modules
pub mod arithmetic;
pub mod data_structures;
pub mod domain;
pub mod io_ops;
pub mod keyword_map;
pub mod memory_ops;
pub mod stack_ops;
pub mod string_ops;
pub mod temporal;

use crate::ast::{
    Effect, LocatedProgram, OpCode, Procedure, ProcedureLocations, Program, ProgramLocations,
    PropertyCell, PropertyComparison, PropertyPredicate, QuoteId, Stmt, StmtLocations,
    TemporalDeclaration,
};
use crate::core::Value;
use crate::runtime::ffi::{FFIEffect, FFISignature, FFIType};
use crate::source::{SourceSpan, TextRange};
use std::collections::HashMap;
use std::iter::Peekable;
use std::slice::Iter;
use std::str::FromStr;

// Re-export domain parser types for external use
pub use crate::lexer::Token;
pub use domain::{DomainParser, ParseContext};
pub use keyword_map::{Domain, DomainRegistry};

/// Maximum recursive nesting accepted for blocks, quotations, and ordinary
/// expressions. PROPERTY predicates have their own stricter structural bound.
pub const MAX_PARSER_NESTING: usize = 64;

/// Maximum number of statement nodes that one parser may materialize.
///
/// This matches the bytecode instruction limit so source sugar cannot build an
/// AST that the canonical compiler is guaranteed to reject later. The count is
/// shared by the main body, procedure bodies, nested blocks, and quotations.
pub const MAX_EXPANDED_STATEMENTS: usize = crate::bytecode::MAX_INSTRUCTIONS;

/// Maximum dimension accepted for a dense MARKOV declaration.
///
/// The stationary-analysis pipeline constructs dense matrices from this
/// scalar, so it must be bounded before allocating state-indexed vectors.
pub const MAX_MARKOV_STATES: usize = 256;

/// Source location span for error reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    /// Line number (1-indexed).
    pub line: usize,
    /// Column number (1-indexed).
    pub column: usize,
    /// Byte offset in source.
    pub offset: usize,
    /// Length in bytes.
    pub len: usize,
}

impl Span {
    /// Create a new span.
    pub fn new(line: usize, column: usize, offset: usize, len: usize) -> Self {
        Self {
            line,
            column,
            offset,
            len,
        }
    }
}

impl std::fmt::Display for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

/// A token with its source span.
#[derive(Debug, Clone, PartialEq)]
pub struct SpannedToken {
    /// The token.
    pub token: Token,
    /// Source location.
    pub span: Span,
}

impl SpannedToken {
    /// Create a new spanned token.
    pub fn new(token: Token, span: Span) -> Self {
        Self { token, span }
    }
}

/// Parse error with location and helpful message.
#[derive(Debug, Clone)]
pub struct ParseError {
    /// Error message.
    pub message: String,
    /// Source location where error occurred.
    pub span: Option<Span>,
    /// Optional help text.
    pub help: Option<String>,
    /// Optional note.
    pub note: Option<String>,
}

impl ParseError {
    /// Create a new parse error.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            span: None,
            help: None,
            note: None,
        }
    }

    /// Add span to error.
    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    /// Add help text.
    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    /// Add note.
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(span) = &self.span {
            write!(f, "error at {}: {}", span, self.message)?;
        } else {
            write!(f, "error: {}", self.message)?;
        }
        if let Some(help) = &self.help {
            write!(f, "\n  = help: {}", help)?;
        }
        if let Some(note) = &self.note {
            write!(f, "\n  = note: {}", note)?;
        }
        Ok(())
    }
}

impl std::error::Error for ParseError {}

impl From<ParseError> for String {
    fn from(e: ParseError) -> String {
        e.to_string()
    }
}

impl Span {
    /// Combine two spans into a span covering both.
    pub fn merge(&self, other: &Self) -> Self {
        if self.offset <= other.offset {
            Self {
                line: self.line,
                column: self.column,
                offset: self.offset,
                len: (other.offset + other.len).saturating_sub(self.offset),
            }
        } else {
            other.merge(self)
        }
    }

    /// Extract the text this span covers from source.
    pub fn extract<'a>(&self, source: &'a str) -> &'a str {
        let end = (self.offset + self.len).min(source.len());
        &source[self.offset..end]
    }
}

/// Tokenize source code into a sequence of tokens (without spans).
/// For backward compatibility. Use [`crate::lexer::lex`] for fallible parsing.
/// This infallible legacy adapter panics rather than returning a truncated token
/// stream when the source-byte or token allocation ceiling is exceeded.
pub fn tokenize(input: &str) -> Vec<Token> {
    tokenize_with_spans(input)
        .into_iter()
        .map(|st| st.token)
        .collect()
}

/// Tokenize source code with full span tracking.
///
/// This is the preferred tokenization function as it provides precise
/// source location for error reporting.
///
/// # Precision Requirements
/// - Byte offsets are exact for UTF-8
/// - Column tracking uses character count (not bytes)
/// - Line numbers are 1-indexed
pub fn tokenize_with_spans(input: &str) -> Vec<SpannedToken> {
    tokenize_with_spans_limit(input, crate::lexer::MAX_SOURCE_TOKENS)
}

fn tokenize_with_spans_limit(input: &str, token_limit: usize) -> Vec<SpannedToken> {
    assert!(
        input.len() <= crate::source::MAX_SOURCE_FILE_BYTES,
        "legacy tokenizer source exceeds the {}-byte limit; use lexer::lex for a fallible diagnostic",
        crate::source::MAX_SOURCE_FILE_BYTES
    );
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    // Position tracking for spans
    let mut line = 1usize;
    let mut column = 1usize;

    // Helper to compute byte offset for char position
    let byte_offsets: Vec<usize> = {
        let mut offsets = Vec::with_capacity(chars.len() + 1);
        let mut off = 0;
        for c in &chars {
            offsets.push(off);
            off += c.len_utf8();
        }
        offsets.push(off); // End offset
        offsets
    };

    while i < chars.len() {
        let start_line = line;
        let start_column = column;
        let start_offset = byte_offsets[i];

        match chars[i] {
            // Whitespace - track position but don't emit token
            ' ' | '\t' => {
                column += 1;
                i += 1;
            }
            '\n' => {
                line += 1;
                column = 1;
                i += 1;
            }
            '\r' => {
                // Handle \r\n as single newline
                if i + 1 < chars.len() && chars[i + 1] == '\n' {
                    i += 1;
                }
                line += 1;
                column = 1;
                i += 1;
            }

            // Block delimiters
            '{' => {
                let span = Span::new(start_line, start_column, start_offset, 1);
                push_legacy_token(&mut tokens, Token::LBrace, span, token_limit);
                column += 1;
                i += 1;
            }
            '}' => {
                let span = Span::new(start_line, start_column, start_offset, 1);
                push_legacy_token(&mut tokens, Token::RBrace, span, token_limit);
                column += 1;
                i += 1;
            }
            '(' => {
                let span = Span::new(start_line, start_column, start_offset, 1);
                push_legacy_token(&mut tokens, Token::LParen, span, token_limit);
                column += 1;
                i += 1;
            }
            ')' => {
                let span = Span::new(start_line, start_column, start_offset, 1);
                push_legacy_token(&mut tokens, Token::RParen, span, token_limit);
                column += 1;
                i += 1;
            }
            '[' => {
                let span = Span::new(start_line, start_column, start_offset, 1);
                push_legacy_token(&mut tokens, Token::LBracket, span, token_limit);
                column += 1;
                i += 1;
            }
            ']' => {
                let span = Span::new(start_line, start_column, start_offset, 1);
                push_legacy_token(&mut tokens, Token::RBracket, span, token_limit);
                column += 1;
                i += 1;
            }
            ',' => {
                let span = Span::new(start_line, start_column, start_offset, 1);
                push_legacy_token(&mut tokens, Token::Comma, span, token_limit);
                column += 1;
                i += 1;
            }

            // Line comment
            '#' => {
                while i < chars.len() && chars[i] != '\n' {
                    column += 1;
                    i += 1;
                }
            }

            // String literal "..."
            '"' => {
                i += 1; // Skip opening quote
                column += 1;
                let mut string_val = String::new();
                while i < chars.len() && chars[i] != '"' {
                    if chars[i] == '\n' {
                        line += 1;
                        column = 1;
                    } else {
                        column += 1;
                    }
                    if chars[i] == '\\' && i + 1 < chars.len() {
                        // Escape sequence
                        i += 1;
                        column += 1;
                        match chars[i] {
                            'n' => string_val.push('\n'),
                            't' => string_val.push('\t'),
                            'r' => string_val.push('\r'),
                            '"' => string_val.push('"'),
                            '\\' => string_val.push('\\'),
                            c => string_val.push(c),
                        }
                    } else {
                        string_val.push(chars[i]);
                    }
                    i += 1;
                }
                if i < chars.len() {
                    i += 1; // Skip closing quote
                    column += 1;
                }
                let end_offset = byte_offsets[i];
                let span = Span::new(
                    start_line,
                    start_column,
                    start_offset,
                    end_offset - start_offset,
                );
                push_legacy_token(&mut tokens, Token::StringLit(string_val), span, token_limit);
            }

            // Character literal '...'
            '\'' => {
                i += 1; // Skip opening quote
                column += 1;
                if i < chars.len() {
                    let ch = if chars[i] == '\\' && i + 1 < chars.len() {
                        i += 1;
                        column += 1;
                        match chars[i] {
                            'n' => '\n',
                            't' => '\t',
                            'r' => '\r',
                            '\'' => '\'',
                            '\\' => '\\',
                            c => c,
                        }
                    } else {
                        chars[i]
                    };
                    i += 1;
                    column += 1;
                    if i < chars.len() && chars[i] == '\'' {
                        i += 1; // Skip closing quote
                        column += 1;
                    }
                    let end_offset = byte_offsets[i];
                    let span = Span::new(
                        start_line,
                        start_column,
                        start_offset,
                        end_offset - start_offset,
                    );
                    push_legacy_token(&mut tokens, Token::CharLit(ch), span, token_limit);
                }
            }

            // Semicolon comment (alternative)
            ';' if i + 1 < chars.len() && chars[i + 1] == ';' => {
                while i < chars.len() && chars[i] != '\n' {
                    column += 1;
                    i += 1;
                }
            }

            // Numeric literal
            c if c.is_ascii_digit() => {
                let start_i = i;

                // Check for hex (0x) or binary (0b)
                if c == '0' && i + 1 < chars.len() {
                    match chars[i + 1] {
                        'x' | 'X' => {
                            i += 2; // Skip 0x
                            column += 2;
                            while i < chars.len() && chars[i].is_ascii_hexdigit() {
                                i += 1;
                                column += 1;
                            }
                            let hex_str: String = chars[start_i + 2..i].iter().collect();
                            if let Ok(n) = u64::from_str_radix(&hex_str, 16) {
                                let end_offset = byte_offsets[i];
                                let span = Span::new(
                                    start_line,
                                    start_column,
                                    start_offset,
                                    end_offset - start_offset,
                                );
                                push_legacy_token(&mut tokens, Token::Number(n), span, token_limit);
                            }
                            continue;
                        }
                        'b' | 'B' => {
                            i += 2; // Skip 0b
                            column += 2;
                            while i < chars.len() && (chars[i] == '0' || chars[i] == '1') {
                                i += 1;
                                column += 1;
                            }
                            let bin_str: String = chars[start_i + 2..i].iter().collect();
                            if let Ok(n) = u64::from_str_radix(&bin_str, 2) {
                                let end_offset = byte_offsets[i];
                                let span = Span::new(
                                    start_line,
                                    start_column,
                                    start_offset,
                                    end_offset - start_offset,
                                );
                                push_legacy_token(&mut tokens, Token::Number(n), span, token_limit);
                            }
                            continue;
                        }
                        _ => {}
                    }
                }

                // Decimal number
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                    column += 1;
                }
                let num_str: String = chars[start_i..i].iter().collect();
                if let Ok(n) = num_str.parse::<u64>() {
                    let end_offset = byte_offsets[i];
                    let span = Span::new(
                        start_line,
                        start_column,
                        start_offset,
                        end_offset - start_offset,
                    );
                    push_legacy_token(&mut tokens, Token::Number(n), span, token_limit);
                }
            }

            // Word (identifier or keyword)
            c if c.is_alphabetic() || c == '_' => {
                let start_i = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                    column += 1;
                }
                let word: String = chars[start_i..i].iter().collect();
                let end_offset = byte_offsets[i];
                let span = Span::new(
                    start_line,
                    start_column,
                    start_offset,
                    end_offset - start_offset,
                );
                push_legacy_token(&mut tokens, Token::Word(word), span, token_limit);
            }

            // Equals sign
            '=' if i + 1 < chars.len() && chars[i + 1] == '=' => {
                // == is a comparison operator
                i += 2;
                column += 2;
                let end_offset = byte_offsets[i];
                let span = Span::new(
                    start_line,
                    start_column,
                    start_offset,
                    end_offset - start_offset,
                );
                push_legacy_token(
                    &mut tokens,
                    Token::Word("==".to_string()),
                    span,
                    token_limit,
                );
            }
            '=' => {
                let span = Span::new(start_line, start_column, start_offset, 1);
                push_legacy_token(&mut tokens, Token::Equals, span, token_limit);
                column += 1;
                i += 1;
            }

            // Semicolon (statement terminator)
            ';' => {
                // Check if it's a comment (;;)
                if i + 1 < chars.len() && chars[i + 1] == ';' {
                    while i < chars.len() && chars[i] != '\n' {
                        column += 1;
                        i += 1;
                    }
                } else {
                    let span = Span::new(start_line, start_column, start_offset, 1);
                    push_legacy_token(&mut tokens, Token::Semicolon, span, token_limit);
                    column += 1;
                    i += 1;
                }
            }

            '/' => {
                if i + 1 < chars.len() && chars[i + 1] == '/' {
                    // Line comment
                    while i < chars.len() && chars[i] != '\n' {
                        column += 1;
                        i += 1;
                    }
                } else {
                    let span = Span::new(start_line, start_column, start_offset, 1);
                    push_legacy_token(&mut tokens, Token::Word("/".to_string()), span, token_limit);
                    column += 1;
                    i += 1;
                }
            }

            // Symbolic operators (single character words)
            '+' | '-' | '*' | '%' | '&' | '|' | '^' | '~' | '<' | '>' | '!' | '@' | ':' => {
                // Check for two-character operators
                let mut op = String::new();
                op.push(chars[i]);
                i += 1;
                column += 1;

                if i < chars.len() {
                    match (op.chars().next().unwrap(), chars[i]) {
                        ('<', '=')
                        | ('>', '=')
                        | ('!', '=')
                        | ('<', '<')
                        | ('>', '>')
                        | ('&', '&')
                        | ('|', '|') => {
                            op.push(chars[i]);
                            i += 1;
                            column += 1;
                        }
                        _ => {}
                    }
                }

                let end_offset = byte_offsets[i];
                let span = Span::new(
                    start_line,
                    start_column,
                    start_offset,
                    end_offset - start_offset,
                );
                push_legacy_token(&mut tokens, Token::Word(op), span, token_limit);
            }

            // Skip unknown characters
            _ => {
                column += 1;
                i += 1;
            }
        }
    }

    tokens
}

fn push_legacy_token(tokens: &mut Vec<SpannedToken>, token: Token, span: Span, token_limit: usize) {
    assert!(
        tokens.len() < token_limit,
        "legacy tokenizer source exceeds the {token_limit}-token limit; use lexer::lex for a fallible diagnostic"
    );
    tokens.push(SpannedToken::new(token, span));
}

/// Variable binding with its stack position.
#[derive(Debug, Clone)]
struct VariableBinding {
    /// Stack depth when this variable was created.
    stack_depth: usize,
}

/// Temporal variable binding backed by oracle address.
#[derive(Debug, Clone)]
struct TemporalBinding {
    /// Memory address this variable reads from/writes to.
    address: u64,
}

const MAX_PROPERTY_NODES: usize = 256;

fn property_node(nodes: &mut usize) -> Result<(), String> {
    *nodes = nodes.saturating_add(1);
    if *nodes > MAX_PROPERTY_NODES {
        Err(format!(
            "PROPERTY predicate exceeds {MAX_PROPERTY_NODES} Boolean nodes"
        ))
    } else {
        Ok(())
    }
}

fn parse_property_comparison(token: Option<&Token>) -> Result<PropertyComparison, String> {
    match token {
        Some(Token::Word(word)) => match word.to_ascii_uppercase().as_str() {
            "EQ" => Ok(PropertyComparison::Eq),
            "NE" | "NEQ" => Ok(PropertyComparison::Ne),
            "LT" => Ok(PropertyComparison::Ult),
            "LTE" => Ok(PropertyComparison::Ule),
            "GT" => Ok(PropertyComparison::Ugt),
            "GTE" => Ok(PropertyComparison::Uge),
            other => Err(format!(
                "Unknown PROPERTY comparison '{}'; expected EQ, NEQ, LT, LTE, GT, or GTE",
                other
            )),
        },
        _ => Err("Expected comparison operator in PROPERTY".into()),
    }
}

fn first_property_atom(predicate: &PropertyPredicate) -> Option<(u64, PropertyComparison, u64)> {
    match predicate {
        PropertyPredicate::Compare {
            cell,
            comparison,
            value,
        } => Some((cell.address, *comparison, *value)),
        PropertyPredicate::Not(inner) => first_property_atom(inner),
        PropertyPredicate::And(left, right) | PropertyPredicate::Or(left, right) => {
            first_property_atom(left).or_else(|| first_property_atom(right))
        }
    }
}

/// Procedure binding with its definition.
#[derive(Debug, Clone)]
struct ProcedureBinding {
    /// Number of parameters the procedure takes.
    param_count: usize,
    /// Number of return values.
    return_count: usize,
}

struct ParsedBlock {
    statements: Vec<Stmt>,
    locations: Vec<StmtLocations>,
}

/// Discover the callable shapes of source-local procedures before parsing
/// their bodies. Name resolution proper remains a HIR responsibility, but the
/// source parser needs these shapes so forward and recursive calls are not
/// mistaken for unknown variables while constructing the legacy AST adapter.
fn discover_local_procedure_interfaces(tokens: &[Token]) -> HashMap<String, ProcedureBinding> {
    let mut procedures = HashMap::new();
    let mut index = 0;
    while index < tokens.len() {
        let is_declaration = matches!(
            tokens.get(index),
            Some(Token::Word(keyword))
                if keyword.eq_ignore_ascii_case("PROCEDURE")
                    || keyword.eq_ignore_ascii_case("PROC")
        );
        if !is_declaration {
            index += 1;
            continue;
        }

        let Some(Token::Word(name)) = tokens.get(index + 1) else {
            index += 1;
            continue;
        };
        let mut param_count = 0;
        let mut cursor = index + 2;
        if matches!(tokens.get(cursor), Some(Token::LParen)) {
            cursor += 1;
            while let Some(token) = tokens.get(cursor) {
                match token {
                    Token::RParen => break,
                    Token::Word(_) => param_count += 1,
                    _ => {}
                }
                cursor += 1;
            }
        }
        procedures.insert(
            name.to_lowercase(),
            ProcedureBinding {
                param_count,
                return_count: 1,
            },
        );
        index += 2;
    }
    procedures
}

/// FFI function declaration parsed from FOREIGN block.
#[derive(Debug, Clone)]
pub struct FFIDeclaration {
    /// Function signature.
    pub signature: FFISignature,
    /// External symbol name (may differ from Ouro name).
    pub symbol_name: Option<String>,
}

/// Parser for OUROCHRONOS programs.
pub struct Parser<'a> {
    tokens: Peekable<Iter<'a, Token>>,
    token_count: usize,
    source_spans: Option<&'a [SourceSpan]>,
    constants: HashMap<String, u64>,
    manifest_declarations: Vec<crate::ast::ManifestDeclaration>,
    /// Variable bindings (name -> binding info).
    variables: HashMap<String, VariableBinding>,
    /// Temporal variable bindings (name -> temporal binding).
    temporal_vars: HashMap<String, TemporalBinding>,
    temporal_declarations: Vec<TemporalDeclaration>,
    /// Procedure bindings (name -> binding info).
    procedures: HashMap<String, ProcedureBinding>,
    /// Parsed procedure definitions.
    procedure_defs: Vec<Procedure>,
    /// Parsed quotations (anonymous blocks).
    quotes: Vec<Vec<Stmt>>,
    /// Current stack depth (for variable resolution).
    stack_depth: usize,
    /// Token position for error reporting.
    token_pos: usize,
    /// FFI declarations from FOREIGN blocks.
    ffi_declarations: Vec<FFIDeclaration>,
    /// Domain parser registry for modular keyword handling.
    domain_registry: DomainRegistry,
    family_declaration: Option<crate::ast::PspaceFamilyDeclaration>,
    markov_declaration: Option<crate::ast::MarkovDeclaration>,
    quantum_declaration: Option<crate::ast::QuantumChannelDeclaration>,
    temporal_properties: Vec<crate::ast::TemporalPropertyDeclaration>,
    /// When true, IMPORT directives are recorded for an external canonical
    /// module loader instead of being read relative to the process CWD.
    defer_imports: bool,
    deferred_imports: Vec<String>,
    procedure_locations: Vec<Option<ProcedureLocations>>,
    quote_locations: Vec<Vec<StmtLocations>>,
    foreign_locations: Vec<SourceSpan>,
    property_locations: Vec<SourceSpan>,
    manifest_locations: Vec<SourceSpan>,
    temporal_locations: Vec<SourceSpan>,
    pending_stmt_children: Option<Vec<Vec<StmtLocations>>>,
    last_error_source_span: Option<SourceSpan>,
    nesting_depth: usize,
    expanded_statements: usize,
    expanded_statement_limit: usize,
}

impl<'a> Parser<'a> {
    /// Create a new parser from a token slice.
    pub fn new(tokens: &'a [Token]) -> Self {
        Self {
            tokens: tokens.iter().peekable(),
            token_count: tokens.len(),
            source_spans: None,
            constants: HashMap::new(),
            manifest_declarations: Vec::new(),
            variables: HashMap::new(),
            temporal_vars: HashMap::new(),
            temporal_declarations: Vec::new(),
            procedures: discover_local_procedure_interfaces(tokens),
            procedure_defs: Vec::new(),
            quotes: Vec::new(),
            stack_depth: 0,
            token_pos: 0,
            ffi_declarations: Vec::new(),
            domain_registry: DomainRegistry::new(),
            family_declaration: None,
            markov_declaration: None,
            quantum_declaration: None,
            temporal_properties: Vec::new(),
            defer_imports: false,
            deferred_imports: Vec::new(),
            procedure_locations: Vec::new(),
            quote_locations: Vec::new(),
            foreign_locations: Vec::new(),
            property_locations: Vec::new(),
            manifest_locations: Vec::new(),
            temporal_locations: Vec::new(),
            pending_stmt_children: None,
            last_error_source_span: None,
            nesting_depth: 0,
            expanded_statements: 0,
            expanded_statement_limit: MAX_EXPANDED_STATEMENTS,
        }
    }

    /// Create a parser retaining one exact source span per token.
    ///
    /// This leaves [`Parser::new`] and the legacy AST API unchanged. The
    /// located constructor rejects misaligned sidecars rather than compiling
    /// with silently shifted diagnostics.
    pub fn new_with_source_spans(
        tokens: &'a [Token],
        spans: &'a [SourceSpan],
    ) -> Result<Self, String> {
        if tokens.len() != spans.len() {
            return Err(format!(
                "token/span count mismatch: {} tokens and {} spans",
                tokens.len(),
                spans.len()
            ));
        }
        let mut parser = Self::new(tokens);
        parser.source_spans = Some(spans);
        Ok(parser)
    }

    /// Create a parser that records IMPORT directives without loading them.
    /// Canonical module-graph construction uses this mode after resolving and
    /// parsing dependencies relative to their importing source.
    pub fn new_with_deferred_imports(tokens: &'a [Token]) -> Self {
        let mut parser = Self::new(tokens);
        parser.defer_imports = true;
        parser
    }

    /// Located counterpart of [`Parser::new_with_deferred_imports`].
    pub fn new_with_deferred_imports_and_spans(
        tokens: &'a [Token],
        spans: &'a [SourceSpan],
    ) -> Result<Self, String> {
        Self::new_with_deferred_imports_spans_and_statement_limit(
            tokens,
            spans,
            MAX_EXPANDED_STATEMENTS,
        )
    }

    /// Located deferred-import parser with a caller-supplied remaining graph
    /// statement budget. The ceiling may only tighten the global parser limit.
    pub fn new_with_deferred_imports_spans_and_statement_limit(
        tokens: &'a [Token],
        spans: &'a [SourceSpan],
        statement_limit: usize,
    ) -> Result<Self, String> {
        let mut parser = Self::new_with_source_spans(tokens, spans)?;
        parser.defer_imports = true;
        parser.expanded_statement_limit = statement_limit.min(MAX_EXPANDED_STATEMENTS);
        Ok(parser)
    }

    /// Number of statement-budget units reserved by the current parse.
    pub const fn expanded_statement_count(&self) -> usize {
        self.expanded_statements
    }

    fn position(&self) -> usize {
        self.token_count - self.tokens.len()
    }

    fn consumed_span(&self, start: usize) -> Option<SourceSpan> {
        let spans = self.source_spans?;
        let end = self.position();
        if start >= end || end > spans.len() {
            return spans.get(start.min(spans.len().saturating_sub(1))).copied();
        }
        let first = spans[start];
        let last = spans[end - 1];
        debug_assert_eq!(first.source, last.source);
        Some(SourceSpan::new(
            first.source,
            TextRange::new(first.range.start, last.range.end),
        ))
    }

    /// Best exact token location for a parse failure at the current cursor.
    pub fn error_source_span(&self) -> Option<SourceSpan> {
        if let Some(span) = self.last_error_source_span {
            return Some(span);
        }
        let spans = self.source_spans?;
        spans
            .get(self.position())
            .or_else(|| {
                self.position()
                    .checked_sub(1)
                    .and_then(|index| spans.get(index))
            })
            .copied()
    }

    /// Take IMPORT path strings encountered by a deferred-import parser.
    pub fn take_deferred_imports(&mut self) -> Vec<String> {
        std::mem::take(&mut self.deferred_imports)
    }

    /// Register the callable interface of an already parsed dependency
    /// without copying its definitions into this parser's output program.
    pub fn register_imported_interface(&mut self, program: &Program) {
        for declaration in &program.manifests {
            self.constants
                .insert(declaration.name.to_uppercase(), declaration.value);
        }
        for declaration in &program.temporal_declarations {
            self.temporal_vars.insert(
                declaration.name.to_lowercase(),
                TemporalBinding {
                    address: declaration.address,
                },
            );
        }
        for procedure in &program.procedures {
            self.register_procedure_interface(procedure);
        }
        for declaration in &program.ffi_declarations {
            self.procedures.insert(
                declaration.signature.name.to_lowercase(),
                ProcedureBinding {
                    param_count: declaration.signature.input_stack_size(),
                    return_count: declaration.signature.output_stack_size(),
                },
            );
        }
    }

    /// Get parsed FFI declarations (to be registered with FFIContext).
    pub fn get_ffi_declarations(&self) -> &[FFIDeclaration] {
        &self.ffi_declarations
    }

    /// Register external procedures (e.g. standard library).
    pub fn register_procedures(&mut self, procs: Vec<Procedure>) {
        for mut proc in procs {
            proc.name = proc.name.to_lowercase();
            self.register_procedure_interface(&proc);
            self.procedure_defs.push(proc);
            self.procedure_locations.push(None);
        }
    }

    /// Register callable procedure signatures without retaining definitions in
    /// the output program. Module parsing uses this for preludes and imports;
    /// the linker owns the single deterministic copy of each definition.
    pub fn register_procedure_interfaces(&mut self, procedures: &[Procedure]) {
        for procedure in procedures {
            self.register_procedure_interface(procedure);
        }
    }

    fn register_procedure_interface(&mut self, procedure: &Procedure) {
        self.procedures.insert(
            procedure.name.to_lowercase(),
            ProcedureBinding {
                param_count: procedure.params.len(),
                return_count: procedure.returns,
            },
        );
    }

    /// Create an error at the current position.
    fn error(&self, message: impl Into<String>) -> ParseError {
        ParseError::new(message).with_span(Span::new(1, self.token_pos + 1, 0, 1))
    }

    /// Create an error with help text.
    fn error_with_help(&self, message: impl Into<String>, help: impl Into<String>) -> ParseError {
        self.error(message).with_help(help)
    }

    fn with_nesting<T>(
        &mut self,
        context: &'static str,
        parse: impl FnOnce(&mut Self) -> Result<T, String>,
    ) -> Result<T, String> {
        if self.nesting_depth >= MAX_PARSER_NESTING {
            return Err(format!(
                "parser nesting exceeds {MAX_PARSER_NESTING} while parsing {context}"
            ));
        }
        self.nesting_depth += 1;
        let result = parse(self);
        self.nesting_depth -= 1;
        result
    }

    /// Reserve parser-wide statement budget before materializing AST nodes.
    fn reserve_expanded_statements(
        &mut self,
        additional: usize,
        context: &'static str,
    ) -> Result<(), String> {
        let requested = self
            .expanded_statements
            .checked_add(additional)
            .ok_or_else(|| {
                format!("expanded statement count overflows usize while parsing {context}")
            })?;
        if requested > self.expanded_statement_limit {
            return Err(format!(
                "expanded statement budget exceeds {} while parsing {context}",
                self.expanded_statement_limit
            ));
        }
        self.expanded_statements = requested;
        Ok(())
    }

    #[cfg(test)]
    fn new_with_expanded_statement_limit(tokens: &'a [Token], limit: usize) -> Self {
        let mut parser = Self::new(tokens);
        parser.expanded_statement_limit = limit;
        parser
    }

    /// Consume next token and track position.
    fn _next_token(&mut self) -> Option<&'a Token> {
        self.token_pos += 1;
        self.tokens.next()
    }

    /// Emit an opcode and track its stack effect.
    fn emit_op(&mut self, op: OpCode) -> Result<Stmt, String> {
        let (inputs, outputs) = op.stack_effect();
        // Saturating sub to avoid negative depth (parser doesn't validate thoroughly)
        self.stack_depth = self.stack_depth.saturating_sub(inputs);
        self.stack_depth += outputs;
        Ok(Stmt::Op(op))
    }

    /// Parse a complete program.
    pub fn parse_program(&mut self) -> Result<Program, String> {
        self.parse_program_parts().map(|located| located.program)
    }

    /// Parse a complete program and retain its exact source-location sidecar.
    pub fn parse_located_program(&mut self) -> Result<LocatedProgram, String> {
        if self.source_spans.is_none() {
            return Err("located parsing requires token source spans".to_string());
        }
        let located = self.parse_program_parts()?;
        if !located.locations.matches(&located.program) {
            return Err("internal parser location tree does not match the AST".to_string());
        }
        Ok(located)
    }

    fn parse_program_parts(&mut self) -> Result<LocatedProgram, String> {
        // Parse declarations (MANIFEST), FOREIGN blocks, and procedures
        loop {
            if self.peek_word_eq("MANIFEST") {
                let start = self.position();
                self.tokens.next(); // Consume MANIFEST
                self.parse_declaration(start)?;
            } else if self.peek_temporal_declaration() {
                let start = self.position();
                self.tokens.next();
                self.parse_temporal_declaration(start)?;
            } else if self.peek_word_eq("FAMILY") {
                self.tokens.next();
                self.parse_family_declaration()?;
            } else if self.peek_word_eq("MARKOV") {
                self.tokens.next();
                self.parse_markov_declaration()?;
            } else if self.peek_word_eq("QCHANNEL") {
                self.tokens.next();
                self.parse_quantum_declaration()?;
            } else if self.peek_word_eq("PROPERTY") {
                self.tokens.next();
                self.parse_temporal_property()?;
            } else if self.peek_word_eq("IMPORT") {
                // parse_import consumes IMPORT itself
                self.parse_import()?;
            } else if self.peek_word_eq("FOREIGN") {
                self.tokens.next(); // Consume FOREIGN
                self.parse_foreign_block()?;
            } else if self.peek_word_eq("PROCEDURE") || self.peek_word_eq("PROC") {
                self.tokens.next(); // Consume PROCEDURE/PROC
                self.parse_procedure_def()?;
            } else {
                break;
            }
        }

        let mut stmts = Vec::new();
        let mut body_locations = Vec::new();
        while self.tokens.peek().is_some() {
            let (statement, location) = self.parse_stmt()?;
            stmts.push(statement);
            if let Some(location) = location {
                body_locations.push(location);
            }
        }
        let program = Program {
            manifests: std::mem::take(&mut self.manifest_declarations),
            temporal_declarations: std::mem::take(&mut self.temporal_declarations),
            procedures: std::mem::take(&mut self.procedure_defs),
            quotes: self.quotes.clone(),
            body: stmts,
            ffi_declarations: std::mem::take(&mut self.ffi_declarations),
            family_declaration: self.family_declaration.take(),
            markov_declaration: self.markov_declaration.take(),
            quantum_declaration: self.quantum_declaration.take(),
            temporal_properties: std::mem::take(&mut self.temporal_properties),
        };
        Ok(LocatedProgram {
            locations: ProgramLocations {
                manifests: std::mem::take(&mut self.manifest_locations)
                    .into_iter()
                    .map(Some)
                    .collect(),
                temporals: std::mem::take(&mut self.temporal_locations)
                    .into_iter()
                    .map(Some)
                    .collect(),
                procedures: std::mem::take(&mut self.procedure_locations),
                quotes: std::mem::take(&mut self.quote_locations),
                body: body_locations,
                foreigns: std::mem::take(&mut self.foreign_locations)
                    .into_iter()
                    .map(Some)
                    .collect(),
                properties: std::mem::take(&mut self.property_locations)
                    .into_iter()
                    .map(Some)
                    .collect(),
            },
            program,
        })
    }

    fn peek_temporal_declaration(&self) -> bool {
        let mut tokens = self.tokens.clone();
        matches!(tokens.next(), Some(Token::Word(word)) if word.eq_ignore_ascii_case("TEMPORAL"))
            && matches!(tokens.next(), Some(Token::Word(_)))
    }

    /// Parse a bounded `ALL_FIXED` Boolean predicate.
    fn parse_temporal_property(&mut self) -> Result<(), String> {
        let declaration_start = self.position().saturating_sub(1);
        let name = match self.tokens.next() {
            Some(Token::Word(name)) => name.clone(),
            _ => return Err("Expected property name after PROPERTY".into()),
        };
        if self
            .temporal_properties
            .iter()
            .any(|property| property.name.eq_ignore_ascii_case(&name))
        {
            return Err(format!("Duplicate PROPERTY name '{}'", name));
        }
        match self.tokens.next() {
            Some(Token::LBrace) => {}
            _ => return Err("Expected '{' after PROPERTY name".into()),
        }
        match self.tokens.next() {
            Some(Token::Word(word)) if word.eq_ignore_ascii_case("ALL_FIXED") => {}
            _ => return Err("PROPERTY requires ALL_FIXED quantification".into()),
        }
        let mut nodes = 0usize;
        let predicate = self.parse_property_or(0, &mut nodes)?;
        let (address, comparison, value) = first_property_atom(&predicate)
            .ok_or_else(|| "PROPERTY predicate contains no cell comparison".to_string())?;
        self.expect_semicolon("PROPERTY predicate")?;
        match self.tokens.next() {
            Some(Token::RBrace) => {}
            _ => return Err("Expected '}' after PROPERTY predicate".into()),
        }
        self.temporal_properties
            .push(crate::ast::TemporalPropertyDeclaration {
                name,
                address,
                comparison,
                value,
                predicate: Some(predicate),
            });
        if let Some(span) = self.consumed_span(declaration_start) {
            self.property_locations.push(span);
        }
        Ok(())
    }

    fn parse_property_or(
        &mut self,
        depth: usize,
        nodes: &mut usize,
    ) -> Result<PropertyPredicate, String> {
        let mut expression = self.parse_property_and(depth, nodes)?;
        while self.peek_word_eq("OR") {
            self.tokens.next();
            property_node(nodes)?;
            expression = PropertyPredicate::Or(
                Box::new(expression),
                Box::new(self.parse_property_and(depth, nodes)?),
            );
        }
        Ok(expression)
    }

    fn parse_property_and(
        &mut self,
        depth: usize,
        nodes: &mut usize,
    ) -> Result<PropertyPredicate, String> {
        let mut expression = self.parse_property_unary(depth, nodes)?;
        while self.peek_word_eq("AND") {
            self.tokens.next();
            property_node(nodes)?;
            expression = PropertyPredicate::And(
                Box::new(expression),
                Box::new(self.parse_property_unary(depth, nodes)?),
            );
        }
        Ok(expression)
    }

    fn parse_property_unary(
        &mut self,
        depth: usize,
        nodes: &mut usize,
    ) -> Result<PropertyPredicate, String> {
        const MAX_PROPERTY_DEPTH: usize = 32;
        if depth >= MAX_PROPERTY_DEPTH {
            return Err(format!("PROPERTY nesting exceeds {MAX_PROPERTY_DEPTH}"));
        }
        if self.peek_word_eq("NOT") {
            self.tokens.next();
            property_node(nodes)?;
            return Ok(PropertyPredicate::Not(Box::new(
                self.parse_property_unary(depth + 1, nodes)?,
            )));
        }
        if matches!(self.tokens.peek(), Some(Token::LParen)) {
            self.tokens.next();
            let expression = self.parse_property_or(depth + 1, nodes)?;
            match self.tokens.next() {
                Some(Token::RParen) => return Ok(expression),
                _ => return Err("Expected ')' in PROPERTY predicate".into()),
            }
        }
        self.parse_property_comparison(nodes)
    }

    fn parse_property_comparison(
        &mut self,
        nodes: &mut usize,
    ) -> Result<PropertyPredicate, String> {
        property_node(nodes)?;
        if self.peek_word_eq("CELL") {
            self.tokens.next();
        }
        let cell = match self.tokens.next() {
            Some(Token::Number(address)) => PropertyCell {
                address: *address,
                name: self
                    .temporal_vars
                    .iter()
                    .filter(|(_, binding)| binding.address == *address)
                    .map(|(name, _)| name.clone())
                    .min(),
            },
            Some(Token::Word(name)) => {
                let binding = self
                    .temporal_vars
                    .get(&name.to_lowercase())
                    .ok_or_else(|| format!("Unknown temporal cell '{}' in PROPERTY", name))?;
                PropertyCell {
                    address: binding.address,
                    name: Some(name.clone()),
                }
            }
            _ => return Err("Expected cell address or temporal name in PROPERTY".into()),
        };
        let comparison = parse_property_comparison(self.tokens.next())?;
        let value = match self.tokens.next() {
            Some(Token::Number(value)) => *value,
            _ => return Err("Expected comparison value in PROPERTY".into()),
        };
        Ok(PropertyPredicate::Compare {
            cell,
            comparison,
            value,
        })
    }

    /// Parse a finite exact stochastic CTC model.
    fn parse_markov_declaration(&mut self) -> Result<(), String> {
        if self.markov_declaration.is_some() {
            return Err("Only one MARKOV declaration is permitted".into());
        }
        let name = match self.tokens.next() {
            Some(Token::Word(name)) => name.clone(),
            _ => return Err("Expected model name after MARKOV".into()),
        };
        match self.tokens.next() {
            Some(Token::LBrace) => {}
            _ => return Err("Expected '{' after MARKOV name".into()),
        }

        let mut states = None;
        let mut rows: Vec<(usize, Vec<crate::ast::RationalLiteral>)> = Vec::new();
        let mut accepting_states = None;
        let mut accept_at_least = crate::ast::RationalLiteral {
            numerator: 2,
            denominator: 3,
        };
        let mut reject_at_most = crate::ast::RationalLiteral {
            numerator: 1,
            denominator: 3,
        };

        loop {
            let field = match self.tokens.next() {
                Some(Token::RBrace) => break,
                Some(Token::Word(field)) => field.to_uppercase(),
                _ => return Err("Expected MARKOV field or '}'".into()),
            };
            match field.as_str() {
                "STATES" => {
                    if states.is_some() {
                        return Err("Duplicate STATES field in MARKOV declaration".into());
                    }
                    let count = match self.tokens.next() {
                        Some(Token::Number(count)) => usize::try_from(*count)
                            .map_err(|_| "MARKOV state count exceeds usize")?,
                        _ => return Err("Expected state count after STATES".into()),
                    };
                    if count == 0 {
                        return Err("MARKOV must contain at least one state".into());
                    }
                    if count > MAX_MARKOV_STATES {
                        return Err(format!(
                            "MARKOV state count {count} exceeds maximum {MAX_MARKOV_STATES}"
                        ));
                    }
                    states = Some(count);
                    self.expect_semicolon("STATES")?;
                }
                "ROW" => {
                    let index = match self.tokens.next() {
                        Some(Token::Number(index)) => {
                            usize::try_from(*index).map_err(|_| "MARKOV row index exceeds usize")?
                        }
                        _ => return Err("Expected row index after ROW".into()),
                    };
                    match self.tokens.next() {
                        Some(Token::LBrace) => {}
                        _ => return Err("Expected '{' after MARKOV row index".into()),
                    }
                    let mut probabilities = Vec::new();
                    loop {
                        match self.tokens.peek() {
                            Some(Token::RBrace) => {
                                self.tokens.next();
                                break;
                            }
                            Some(_) => probabilities.push(self.parse_rational_literal("ROW")?),
                            None => return Err("Unterminated MARKOV ROW".into()),
                        }
                    }
                    self.expect_semicolon("ROW")?;
                    rows.push((index, probabilities));
                }
                "ACCEPTING" => {
                    match self.tokens.next() {
                        Some(Token::LBrace) => {}
                        _ => return Err("Expected '{' after ACCEPTING".into()),
                    }
                    let mut values = Vec::new();
                    loop {
                        match self.tokens.next() {
                            Some(Token::RBrace) => break,
                            Some(Token::Number(state)) => values.push(
                                usize::try_from(*state)
                                    .map_err(|_| "accepting-state index exceeds usize")?,
                            ),
                            Some(Token::Comma) => {}
                            _ => {
                                return Err(
                                    "Expected state index, comma, or '}' in ACCEPTING".into()
                                )
                            }
                        }
                    }
                    self.expect_semicolon("ACCEPTING")?;
                    accepting_states = Some(values);
                }
                "ACCEPT_AT_LEAST" => {
                    accept_at_least = self.parse_rational_literal("ACCEPT_AT_LEAST")?;
                    self.expect_semicolon("ACCEPT_AT_LEAST")?;
                }
                "REJECT_AT_MOST" => {
                    reject_at_most = self.parse_rational_literal("REJECT_AT_MOST")?;
                    self.expect_semicolon("REJECT_AT_MOST")?;
                }
                other => return Err(format!("Unknown MARKOV field '{}'", other)),
            }
        }

        let states = states.ok_or("MARKOV requires STATES")?;
        let mut transition: Vec<Option<Vec<crate::ast::RationalLiteral>>> = vec![None; states];
        for (index, row) in rows {
            if index >= states {
                return Err(format!("MARKOV row {} is outside 0..{}", index, states));
            }
            if transition[index].is_some() {
                return Err(format!("Duplicate MARKOV row {}", index));
            }
            if row.len() != states {
                return Err(format!(
                    "MARKOV row {} has {} probabilities, expected {}",
                    index,
                    row.len(),
                    states
                ));
            }
            transition[index] = Some(row);
        }
        let transition = transition
            .into_iter()
            .enumerate()
            .map(|(index, row)| row.ok_or_else(|| format!("MARKOV is missing row {}", index)))
            .collect::<Result<Vec<_>, _>>()?;
        let mut accepting_states = accepting_states.ok_or("MARKOV requires ACCEPTING")?;
        accepting_states.sort_unstable();
        accepting_states.dedup();
        if let Some(state) = accepting_states.iter().find(|state| **state >= states) {
            return Err(format!(
                "accepting state {} is outside 0..{}",
                state, states
            ));
        }
        let reject_scaled = reject_at_most.numerator as u128 * accept_at_least.denominator as u128;
        let accept_scaled = accept_at_least.numerator as u128 * reject_at_most.denominator as u128;
        if reject_scaled > accept_scaled {
            return Err("REJECT_AT_MOST must not exceed ACCEPT_AT_LEAST".into());
        }

        self.markov_declaration = Some(crate::ast::MarkovDeclaration {
            name,
            states,
            transition,
            accepting_states,
            accept_at_least,
            reject_at_most,
        });
        Ok(())
    }

    /// Parse a source-level qubit CPTP channel in Kraus form. Each `C` entry
    /// contains real and imaginary signed rationals.
    fn parse_quantum_declaration(&mut self) -> Result<(), String> {
        if self.quantum_declaration.is_some() {
            return Err("Only one QCHANNEL declaration is permitted".into());
        }
        let name = match self.tokens.next() {
            Some(Token::Word(name)) => name.clone(),
            _ => return Err("Expected channel name after QCHANNEL".into()),
        };
        match self.tokens.next() {
            Some(Token::LBrace) => {}
            _ => return Err("Expected '{' after QCHANNEL name".into()),
        }

        let mut qubit = false;
        let mut kraus = Vec::new();
        let mut accepting_basis = None;
        let mut accept_at_least = crate::ast::RationalLiteral {
            numerator: 2,
            denominator: 3,
        };
        let mut reject_at_most = crate::ast::RationalLiteral {
            numerator: 1,
            denominator: 3,
        };
        let mut validation_tolerance = crate::ast::RationalLiteral {
            numerator: 1,
            denominator: 1_000_000_000,
        };
        let mut analysis_tolerance = crate::ast::RationalLiteral {
            numerator: 1,
            denominator: 1_000_000_000,
        };

        loop {
            let field = match self.tokens.next() {
                Some(Token::RBrace) => break,
                Some(Token::Word(field)) => field.to_uppercase(),
                _ => return Err("Expected QCHANNEL field or '}'".into()),
            };
            match field.as_str() {
                "QUBIT" => {
                    if qubit {
                        return Err("Duplicate QUBIT field".into());
                    }
                    qubit = true;
                    self.expect_semicolon("QUBIT")?;
                }
                "KRAUS" => {
                    match self.tokens.next() {
                        Some(Token::LBrace) => {}
                        _ => return Err("Expected '{' after KRAUS".into()),
                    }
                    let mut entries = Vec::new();
                    loop {
                        match self.tokens.peek() {
                            Some(Token::RBrace) => {
                                self.tokens.next();
                                break;
                            }
                            Some(Token::Word(word)) if word.eq_ignore_ascii_case("C") => {
                                self.tokens.next();
                                let real =
                                    self.parse_signed_rational_literal("KRAUS real component")?;
                                let imaginary = self
                                    .parse_signed_rational_literal("KRAUS imaginary component")?;
                                entries
                                    .push(crate::ast::ComplexRationalLiteral { real, imaginary });
                            }
                            _ => return Err("Expected C or '}' inside KRAUS".into()),
                        }
                    }
                    self.expect_semicolon("KRAUS")?;
                    if entries.len() != 4 {
                        return Err(format!(
                            "Qubit KRAUS operator has {} entries, expected 4",
                            entries.len()
                        ));
                    }
                    kraus.push(entries);
                }
                "ACCEPT_BASIS" => {
                    let basis = match self.tokens.next() {
                        Some(Token::Number(value @ 0..=1)) => *value as usize,
                        _ => return Err("ACCEPT_BASIS must be 0 or 1".into()),
                    };
                    accepting_basis = Some(basis);
                    self.expect_semicolon("ACCEPT_BASIS")?;
                }
                "ACCEPT_AT_LEAST" => {
                    accept_at_least = self.parse_rational_literal("ACCEPT_AT_LEAST")?;
                    self.expect_semicolon("ACCEPT_AT_LEAST")?;
                }
                "REJECT_AT_MOST" => {
                    reject_at_most = self.parse_rational_literal("REJECT_AT_MOST")?;
                    self.expect_semicolon("REJECT_AT_MOST")?;
                }
                "VALIDATION_TOLERANCE" => {
                    validation_tolerance = self.parse_rational_literal("VALIDATION_TOLERANCE")?;
                    self.expect_semicolon("VALIDATION_TOLERANCE")?;
                }
                "ANALYSIS_TOLERANCE" => {
                    analysis_tolerance = self.parse_rational_literal("ANALYSIS_TOLERANCE")?;
                    self.expect_semicolon("ANALYSIS_TOLERANCE")?;
                }
                other => return Err(format!("Unknown QCHANNEL field '{}'", other)),
            }
        }
        if !qubit {
            return Err("QCHANNEL currently requires QUBIT".into());
        }
        if kraus.is_empty() {
            return Err("QCHANNEL requires at least one KRAUS operator".into());
        }
        let accepting_basis = accepting_basis.ok_or("QCHANNEL requires ACCEPT_BASIS")?;
        let reject_scaled = reject_at_most.numerator as u128 * accept_at_least.denominator as u128;
        let accept_scaled = accept_at_least.numerator as u128 * reject_at_most.denominator as u128;
        if reject_scaled > accept_scaled {
            return Err("REJECT_AT_MOST must not exceed ACCEPT_AT_LEAST".into());
        }
        if validation_tolerance.numerator == 0 || analysis_tolerance.numerator == 0 {
            return Err("QCHANNEL tolerances must be positive".into());
        }
        self.quantum_declaration = Some(crate::ast::QuantumChannelDeclaration {
            name,
            kraus,
            accepting_basis,
            accept_at_least,
            reject_at_most,
            validation_tolerance,
            analysis_tolerance,
        });
        Ok(())
    }

    fn parse_signed_rational_literal(
        &mut self,
        context: &str,
    ) -> Result<crate::ast::SignedRationalLiteral, String> {
        let negative = match self.tokens.peek() {
            Some(Token::Word(sign)) if sign == "-" => {
                self.tokens.next();
                true
            }
            Some(Token::Word(sign)) if sign == "+" => {
                self.tokens.next();
                false
            }
            _ => false,
        };
        let magnitude = match self.tokens.next() {
            Some(Token::Number(value)) => i64::try_from(*value)
                .map_err(|_| format!("Signed numerator in {} exceeds i64", context))?,
            _ => return Err(format!("Expected signed rational numerator in {}", context)),
        };
        match self.tokens.next() {
            Some(Token::Word(slash)) if slash == "/" => {}
            _ => return Err(format!("Expected '/' in {} rational", context)),
        }
        let denominator = match self.tokens.next() {
            Some(Token::Number(value)) if *value > 0 => *value,
            _ => return Err(format!("Expected positive denominator in {}", context)),
        };
        Ok(crate::ast::SignedRationalLiteral {
            numerator: if negative { -magnitude } else { magnitude },
            denominator,
        })
    }

    fn parse_rational_literal(
        &mut self,
        context: &str,
    ) -> Result<crate::ast::RationalLiteral, String> {
        let numerator = match self.tokens.next() {
            Some(Token::Number(value)) => *value,
            _ => return Err(format!("Expected rational numerator in {}", context)),
        };
        match self.tokens.next() {
            Some(Token::Word(slash)) if slash == "/" => {}
            _ => return Err(format!("Expected '/' in {} rational", context)),
        }
        let denominator = match self.tokens.next() {
            Some(Token::Number(value)) if *value > 0 => *value,
            Some(Token::Number(_)) => {
                return Err(format!(
                    "Rational denominator in {} must be positive",
                    context
                ))
            }
            _ => return Err(format!("Expected rational denominator in {}", context)),
        };
        Ok(crate::ast::RationalLiteral {
            numerator,
            denominator,
        })
    }

    /// Parse a source-level PSPACE family contract.
    ///
    /// ```text
    /// FAMILY name {
    ///   CTC_CELLS POLY coefficient degree additive;
    ///   CHRONOLOGY_BITS POLY coefficient degree additive;
    ///   TRANSITION_STEPS POLY coefficient degree additive;
    ///   UNIFORM; TOTAL; READOUT_INVARIANT; IDEAL_DEUTSCH; EFFECTS_FROZEN;
    /// }
    /// ```
    fn parse_family_declaration(&mut self) -> Result<(), String> {
        if self.family_declaration.is_some() {
            return Err("Only one FAMILY declaration is permitted".into());
        }
        let name = match self.tokens.next() {
            Some(Token::Word(name)) => name.clone(),
            _ => return Err("Expected family name after FAMILY".into()),
        };
        match self.tokens.next() {
            Some(Token::LBrace) => {}
            _ => return Err("Expected '{' after FAMILY name".into()),
        }

        let mut ctc_cells = None;
        let mut chronology_bits = None;
        let mut transition_steps = None;
        let mut uniform = false;
        let mut total = false;
        let mut readout = false;
        let mut deutsch = false;
        let mut effects = false;

        loop {
            let field = match self.tokens.next() {
                Some(Token::RBrace) => break,
                Some(Token::Word(field)) => field.to_uppercase(),
                _ => return Err("Expected FAMILY field or '}'".into()),
            };
            match field.as_str() {
                "CTC_CELLS" => ctc_cells = Some(self.parse_polynomial_declaration("CTC_CELLS")?),
                "CHRONOLOGY_BITS" => {
                    chronology_bits = Some(self.parse_polynomial_declaration("CHRONOLOGY_BITS")?)
                }
                "TRANSITION_STEPS" => {
                    transition_steps = Some(self.parse_polynomial_declaration("TRANSITION_STEPS")?)
                }
                "UNIFORM" => {
                    uniform = true;
                    self.expect_semicolon("UNIFORM")?;
                }
                "TOTAL" => {
                    total = true;
                    self.expect_semicolon("TOTAL")?;
                }
                "READOUT_INVARIANT" => {
                    readout = true;
                    self.expect_semicolon("READOUT_INVARIANT")?;
                }
                "IDEAL_DEUTSCH" => {
                    deutsch = true;
                    self.expect_semicolon("IDEAL_DEUTSCH")?;
                }
                "EFFECTS_FROZEN" => {
                    effects = true;
                    self.expect_semicolon("EFFECTS_FROZEN")?;
                }
                other => return Err(format!("Unknown FAMILY field '{}'", other)),
            }
        }

        self.family_declaration = Some(crate::ast::PspaceFamilyDeclaration {
            name,
            ctc_cells: ctc_cells.ok_or("FAMILY requires CTC_CELLS")?,
            chronology_respecting_bits: chronology_bits.ok_or("FAMILY requires CHRONOLOGY_BITS")?,
            transition_steps: transition_steps.ok_or("FAMILY requires TRANSITION_STEPS")?,
            polynomial_time_uniform: uniform,
            total_transition: total,
            all_fixed_points_agree: readout,
            ideal_deutsch_selector: deutsch,
            effects_frozen_or_modeled: effects,
        });
        Ok(())
    }

    fn parse_polynomial_declaration(
        &mut self,
        field: &str,
    ) -> Result<crate::ast::PolynomialDeclaration, String> {
        match self.tokens.next() {
            Some(Token::Word(word)) if word.eq_ignore_ascii_case("POLY") => {}
            _ => return Err(format!("Expected POLY after {}", field)),
        }
        let coefficient = match self.tokens.next() {
            Some(Token::Number(value)) => *value,
            _ => return Err(format!("Expected coefficient in {} POLY", field)),
        };
        let degree_u64 = match self.tokens.next() {
            Some(Token::Number(value)) => *value,
            _ => return Err(format!("Expected degree in {} POLY", field)),
        };
        let degree = u32::try_from(degree_u64)
            .map_err(|_| format!("Polynomial degree in {} exceeds u32", field))?;
        let additive = match self.tokens.next() {
            Some(Token::Number(value)) => *value,
            _ => return Err(format!("Expected additive constant in {} POLY", field)),
        };
        self.expect_semicolon(field)?;
        Ok(crate::ast::PolynomialDeclaration {
            coefficient,
            degree,
            additive,
        })
    }

    fn expect_semicolon(&mut self, context: &str) -> Result<(), String> {
        match self.tokens.next() {
            Some(Token::Semicolon) => Ok(()),
            Some(Token::Word(word)) if word == ";" => Ok(()),
            _ => Err(format!("Expected ';' after {}", context)),
        }
    }

    /// Parse a declaration: MANIFEST name = value;
    /// Parse a declaration: MANIFEST name = value;
    fn parse_declaration(&mut self, declaration_start: usize) -> Result<(), String> {
        // MANIFEST token already consumed by parse_program

        let name = match self.tokens.next() {
            Some(Token::Word(w)) => w.to_uppercase(),
            _ => return Err("Expected identifier after MANIFEST".to_string()),
        };

        // Expect '='
        match self.tokens.peek() {
            Some(Token::Equals) => {
                self.tokens.next();
            }
            Some(Token::Word(w)) if w == "=" => {
                self.tokens.next();
            }
            _ => return Err("Expected '=' in declaration".to_string()),
        }

        // Expect value (simple integer for now)
        let value = match self.tokens.next() {
            Some(Token::Number(n)) => *n,
            _ => return Err("Expected integer value in declaration".to_string()),
        };

        // Expect ';'
        match self.tokens.peek() {
            Some(Token::Semicolon) => {
                self.tokens.next();
            }
            Some(Token::Word(w)) if w == ";" => {
                self.tokens.next();
            }
            _ => return Err("Expected ';' after declaration".to_string()),
        }

        if self.constants.contains_key(&name) {
            return Err(format!("Duplicate MANIFEST name '{}'", name));
        }
        self.constants.insert(name.clone(), value);
        self.manifest_declarations
            .push(crate::ast::ManifestDeclaration { name, value });
        if let Some(span) = self.consumed_span(declaration_start) {
            self.manifest_locations.push(span);
        }
        Ok(())
    }

    /// Parse a single statement.
    fn parse_stmt(&mut self) -> Result<(Stmt, Option<StmtLocations>), String> {
        debug_assert!(self.pending_stmt_children.is_none());
        let start = self.position();
        // Every textual statement contributes at least its root AST node. Any
        // source sugar below reserves its generated children separately.
        self.reserve_expanded_statements(1, "statement")?;
        let parsed = (|| {
            match self.tokens.next() {
                Some(Token::Number(n)) => {
                    self.stack_depth += 1;
                    Ok(Stmt::Push(Value::new(*n)))
                }

                Some(Token::Word(w)) => self.parse_word(w, start),

                Some(Token::LBrace) => {
                    let content = self.parse_block_content()?;
                    self.retain_stmt_children(vec![content.locations]);
                    Ok(Stmt::Block(content.statements))
                }

                Some(Token::StringLit(s)) => {
                    // The ubiquitous source form `"text" OUTPUT` means textual
                    // output, not "push a length-suffixed string and print only
                    // its length". Compile it to balanced character emissions so
                    // no hidden character words leak onto the operand stack.
                    if matches!(self.tokens.peek(), Some(Token::Word(word)) if word.eq_ignore_ascii_case("OUTPUT"))
                    {
                        self.tokens.next();
                        let character_count = s.chars().count();
                        let generated = character_count
                            .checked_mul(2)
                            .ok_or_else(|| "string OUTPUT expansion overflows usize".to_string())?;
                        self.reserve_expanded_statements(generated, "string literal")?;
                        let mut stmts = Vec::with_capacity(generated);
                        for character in s.chars() {
                            stmts.push(Stmt::Push(Value::new(character as u64)));
                            stmts.push(Stmt::Op(OpCode::Emit));
                        }
                        Ok(Stmt::Block(stmts))
                    } else {
                        // Push each character as a value onto the stack
                        // Then push the length at the end
                        let character_count = s.chars().count();
                        let generated = character_count.checked_add(1).ok_or_else(|| {
                            "string literal expansion overflows usize".to_string()
                        })?;
                        self.reserve_expanded_statements(generated, "string literal")?;
                        let mut stmts = Vec::with_capacity(generated);
                        for character in s.chars() {
                            self.stack_depth += 1;
                            stmts.push(Stmt::Push(Value::new(character as u64)));
                        }
                        // Push length
                        self.stack_depth += 1;
                        stmts.push(Stmt::Push(Value::new(character_count as u64)));
                        Ok(Stmt::Block(stmts))
                    }
                }

                Some(Token::CharLit(c)) => {
                    self.stack_depth += 1;
                    Ok(Stmt::Push(Value::new(*c as u64)))
                }

                Some(Token::LBracket) => self.parse_quote(),

                Some(Token::LParen) => {
                    // Parenthesized expression - parse and return as block
                    let stmts = self.parse_expr_or()?;
                    match self.tokens.next() {
                        Some(Token::RParen) => {}
                        _ => return Err("Expected ')' to close expression".to_string()),
                    }
                    if stmts.len() == 1 {
                        Ok(stmts.into_iter().next().unwrap())
                    } else {
                        Ok(Stmt::Block(stmts))
                    }
                }

                Some(Token::RBrace) => Err("Unexpected '}'".to_string()),
                Some(Token::RBracket) => Err("Unexpected ']'".to_string()),
                Some(Token::RParen) => Err("Unexpected ')'".to_string()),
                Some(Token::Comma) => Err("Unexpected ','".to_string()),
                Some(Token::Equals) => Err("Unexpected '='".to_string()),
                Some(Token::Semicolon) => Err("Unexpected ';'".to_string()),

                None => Err("Unexpected end of input".to_string()),
            }
        })();
        let statement = match parsed {
            Ok(statement) => statement,
            Err(error) => {
                self.pending_stmt_children = None;
                self.last_error_source_span = self.consumed_span(start);
                return Err(error);
            }
        };

        let Some(span) = self.consumed_span(start) else {
            self.pending_stmt_children = None;
            return Ok((statement, None));
        };
        let location = if let Some(child_blocks) = self.pending_stmt_children.take() {
            StmtLocations { span, child_blocks }
        } else {
            StmtLocations::uniform(&statement, span)
        };
        if !location.matches(&statement) {
            return Err("internal parser statement location shape mismatch".to_string());
        }
        Ok((statement, Some(location)))
    }

    fn retain_stmt_children(&mut self, child_blocks: Vec<Vec<StmtLocations>>) {
        if self.source_spans.is_some() {
            debug_assert!(self.pending_stmt_children.is_none());
            self.pending_stmt_children = Some(child_blocks);
        }
    }

    /// Parse a word (opcode or keyword).
    ///
    /// This method uses the domain parser registry for simple opcodes,
    /// and handles special cases (control flow, declarations) directly.
    fn parse_word(&mut self, word: &str, statement_start: usize) -> Result<Stmt, String> {
        let upper = word.to_uppercase();

        // Special cases that need access to parser state (control flow, declarations)
        // These must be checked before delegating to domain parsers
        match upper.as_str() {
            // OUTPUT and EMIT have special expression syntax: OUTPUT(expr)
            "OUTPUT" => {
                if matches!(self.tokens.peek(), Some(Token::LParen)) {
                    let expr_stmts = self.parse_expression()?;
                    let mut stmts = expr_stmts;
                    self.reserve_expanded_statements(1, "OUTPUT expression")?;
                    stmts.push(Stmt::Op(OpCode::Output));
                    Ok(Stmt::Block(stmts))
                } else {
                    self.emit_op(OpCode::Output)
                }
            }

            "EMIT" => {
                if matches!(self.tokens.peek(), Some(Token::LParen)) {
                    let expr_stmts = self.parse_expression()?;
                    let mut stmts = expr_stmts;
                    self.reserve_expanded_statements(1, "EMIT expression")?;
                    stmts.push(Stmt::Op(OpCode::Emit));
                    Ok(Stmt::Block(stmts))
                } else {
                    self.emit_op(OpCode::Emit)
                }
            }

            // Control flow keywords - require parser state for blocks
            "ASSERT" => {
                // Optional condition expression
                let mut stmts = if self.is_expression_start() {
                    self.parse_expr_or()?
                } else {
                    Vec::new()
                };

                // Optional TEMPORAL keyword
                if self.peek_word_eq("TEMPORAL") {
                    self.tokens.next();
                }

                // Expect String literal
                let msg = match self.tokens.next() {
                    Some(Token::StringLit(s)) => s,
                    _ => {
                        return Err(
                            "Expected string literal message after ASSERT condition".to_string()
                        )
                    }
                };

                // Push message string (chars + len)
                let character_count = msg.chars().count();
                let generated = character_count
                    .checked_add(2)
                    .ok_or_else(|| "ASSERT message expansion overflows usize".to_string())?;
                self.reserve_expanded_statements(generated, "ASSERT message")?;
                for character in msg.chars() {
                    self.stack_depth += 1;
                    stmts.push(Stmt::Push(Value::new(character as u64)));
                }
                self.stack_depth += 1;
                stmts.push(Stmt::Push(Value::new(character_count as u64)));

                // Emit ASSERT op
                // OpCode::Assert consumes (cond, chars, len)
                // Stack effect (2,0) handles cond and len.
                // We must manually subtract chars.
                self.stack_depth = self.stack_depth.saturating_sub(character_count);
                self.emit_op(OpCode::Assert)?;
                stmts.push(Stmt::Op(OpCode::Assert));

                // Optional semicolon
                if matches!(self.tokens.peek(), Some(Token::Semicolon)) {
                    self.tokens.next();
                }

                Ok(Stmt::Block(stmts))
            }

            "IF" => {
                // Check for new expression syntax: IF (expr) { }
                if matches!(self.tokens.peek(), Some(Token::LParen)) {
                    let expr_stmts = self.parse_expression()?;
                    let then_block = self.parse_block()?;
                    let else_block = if self.peek_word_eq("ELSE") {
                        self.tokens.next(); // consume ELSE
                        Some(self.parse_block()?)
                    } else {
                        None
                    };
                    // Wrap: evaluate condition, then IF
                    let mut all_stmts = expr_stmts;
                    let then_locations = then_block.locations;
                    let else_locations = else_block.as_ref().map(|block| block.locations.clone());
                    self.reserve_expanded_statements(1, "IF expression")?;
                    all_stmts.push(Stmt::If {
                        then_branch: then_block.statements,
                        else_branch: else_block.map(|block| block.statements),
                    });
                    if let Some(span) = self.consumed_span(statement_start) {
                        let mut locations = all_stmts[..all_stmts.len() - 1]
                            .iter()
                            .map(|statement| StmtLocations::uniform(statement, span))
                            .collect::<Vec<_>>();
                        let mut if_children = vec![then_locations];
                        if let Some(else_locations) = else_locations {
                            if_children.push(else_locations);
                        }
                        locations.push(StmtLocations {
                            span,
                            child_blocks: if_children,
                        });
                        self.retain_stmt_children(vec![locations]);
                    }
                    Ok(Stmt::Block(all_stmts))
                } else {
                    // Old syntax: condition already on stack
                    self.stack_depth = self.stack_depth.saturating_sub(1);
                    let then_block = self.parse_block()?;
                    let else_block = if self.peek_word_eq("ELSE") {
                        self.tokens.next(); // consume ELSE
                        Some(self.parse_block()?)
                    } else {
                        None
                    };
                    let mut child_locations = vec![then_block.locations];
                    if let Some(block) = &else_block {
                        child_locations.push(block.locations.clone());
                    }
                    self.retain_stmt_children(child_locations);
                    Ok(Stmt::If {
                        then_branch: then_block.statements,
                        else_branch: else_block.map(|block| block.statements),
                    })
                }
            }

            "WHILE" => {
                let cond_block = self.parse_block()?;
                let body_block = self.parse_block()?;
                self.retain_stmt_children(vec![cond_block.locations, body_block.locations]);
                Ok(Stmt::While {
                    cond: cond_block.statements,
                    body: body_block.statements,
                })
            }

            "ELSE" => Err("Unexpected ELSE without IF".to_string()),

            // Procedure definition
            "PROCEDURE" => {
                self.parse_procedure_def()?;
                Ok(Stmt::Block(vec![]))
            }

            // TEMPORAL: Either scope or variable binding
            // Scope: TEMPORAL <base> <size> [BITS <cell_bits>] { body }
            // Variable: TEMPORAL <name> @ <addr> DEFAULT <val>;
            "TEMPORAL" => {
                match self.tokens.peek() {
                    // If next is a number, it's the scope syntax
                    Some(Token::Number(_)) => {
                        // Parse base address
                        let base = match self.tokens.next() {
                            Some(Token::Number(n)) => *n,
                            _ => return Err("Expected base address after TEMPORAL".to_string()),
                        };

                        // Parse size
                        let size = match self.tokens.next() {
                            Some(Token::Number(n)) => *n,
                            _ => return Err("Expected size after TEMPORAL base".to_string()),
                        };
                        if size == 0 {
                            return Err("TEMPORAL scope size must be greater than zero".to_string());
                        }

                        let cell_bits = match self.tokens.peek() {
                            Some(Token::Word(word)) if word.eq_ignore_ascii_case("BITS") => {
                                self.tokens.next();
                                match self.tokens.next() {
                                    Some(Token::Number(bits @ 1..=64)) => *bits as u8,
                                    Some(Token::Number(bits)) => {
                                        return Err(format!(
                                            "TEMPORAL BITS must be in 1..=64, got {}",
                                            bits
                                        ));
                                    }
                                    _ => {
                                        return Err("Expected bit width after TEMPORAL ... BITS"
                                            .to_string());
                                    }
                                }
                            }
                            _ => 64,
                        };

                        // Parse body block
                        let body = self.parse_block()?;
                        self.retain_stmt_children(vec![body.locations]);

                        Ok(Stmt::TemporalScope {
                            base,
                            size,
                            cell_bits,
                            body: body.statements,
                        })
                    }
                    // If next is a word (identifier), it's variable binding syntax
                    Some(Token::Word(_)) => {
                        let start = self.position().saturating_sub(1);
                        self.parse_temporal_declaration(start)?;
                        Ok(Stmt::Block(vec![]))
                    }
                    _ => Err(self
                        .error("Expected address or variable name after TEMPORAL")
                        .into()),
                }
            }

            // LET bindings: LET name = expr;
            "LET" => self.parse_let(),

            // Handle domain parser keywords, constants, variables, or unknown words
            other => {
                // First, try the domain parser registry for simple opcodes
                // This delegates stack ops, arithmetic, temporal, data structures, I/O, strings
                // to their respective domain parsers
                {
                    let mut ctx = ParseContext::new(&mut self.stack_depth, domain::emit_op_helper);
                    if let Some(result) = self.domain_registry.parse(other, &mut ctx) {
                        return result;
                    }
                }

                // Check if it is a defined constant
                if let Some(&val) = self.constants.get(other) {
                    self.stack_depth += 1;
                    return Ok(Stmt::PushConstant {
                        name: other.to_uppercase(),
                        value: val,
                    });
                }

                // Check if it is a defined variable (case-insensitive)
                let lower = word.to_lowercase();
                if let Some(binding) = self.variables.get(&lower).cloned() {
                    // Calculate PICK index: distance from current stack top to variable position
                    let pick_index = self.stack_depth - binding.stack_depth - 1;
                    self.stack_depth += 1; // PICK adds to stack
                                           // Generate: <pick_index> PICK
                    self.reserve_expanded_statements(2, "variable reference")?;
                    return Ok(Stmt::Block(vec![
                        Stmt::Push(Value::new(pick_index as u64)),
                        Stmt::Op(OpCode::Pick),
                    ]));
                }

                // Check if it is a temporal variable (oracle-backed)
                if let Some(binding) = self.temporal_vars.get(&lower).cloned() {
                    self.stack_depth += 1; // ORACLE adds to stack
                    return Ok(Stmt::ReadTemporal {
                        name: lower,
                        address: binding.address,
                    });
                }

                // Check if it is a defined procedure
                if let Some(binding) = self.procedures.get(&lower).cloned() {
                    // Procedure call: consumes params, produces returns
                    self.stack_depth = self.stack_depth.saturating_sub(binding.param_count);
                    self.stack_depth += binding.return_count;
                    return Ok(Stmt::Call { name: lower });
                }

                // If it is strictly uppercase, it might be a misspelled opcode
                if other.chars().all(|c| c.is_uppercase() || c == '_') {
                    // Suggest similar opcodes
                    let suggestion = self.suggest_opcode(other);
                    let mut err = self.error(format!("Unknown opcode: '{}'", other));
                    if let Some(suggested) = suggestion {
                        err = err.with_help(format!("Did you mean '{}'?", suggested));
                    }
                    err = err.with_note("Available opcodes: ADD, SUB, MUL, DIV, DUP, SWAP, ORACLE, PROPHECY, IF, WHILE, LET");
                    return Err(err.into());
                }
                // Otherwise, error with variable context
                let mut err = self.error(format!("Unknown variable or procedure: '{}'", other));
                if !self.variables.is_empty() {
                    let vars: Vec<_> = self.variables.keys().take(3).cloned().collect();
                    err = err.with_note(format!("Defined variables: {}", vars.join(", ")));
                }
                if !self.procedures.is_empty() {
                    let procs: Vec<_> = self.procedures.keys().take(3).cloned().collect();
                    err = err.with_help(format!("Available procedures: {}", procs.join(", ")));
                }
                Err(err.into())
            }
        }
    }

    /// Suggest a similar opcode for typo detection.
    fn suggest_opcode(&self, unknown: &str) -> Option<&'static str> {
        const OPCODES: &[&str] = &[
            "ADD",
            "SUB",
            "MUL",
            "DIV",
            "MOD",
            "NEG",
            "DUP",
            "DROP",
            "SWAP",
            "OVER",
            "ROT",
            "PICK",
            "DEPTH",
            "AND",
            "OR",
            "XOR",
            "NOT",
            "SHL",
            "SHR",
            "EQ",
            "NEQ",
            "LT",
            "GT",
            "LTE",
            "GTE",
            "ORACLE",
            "PROPHECY",
            "PRESENT",
            "PARADOX",
            "INPUT",
            "OUTPUT",
            "HALT",
            "NOP",
            "ROLL",
            "REVERSE",
            "STR_REV",
            "STR_CAT",
            "STR_SPLIT",
            // FFI and I/O
            "FFI_CALL",
            "FFI_CALL_NAMED",
            "FILE_OPEN",
            "FILE_READ",
            "FILE_WRITE",
            "FILE_SEEK",
            "FILE_FLUSH",
            "FILE_CLOSE",
            "FILE_EXISTS",
            "FILE_SIZE",
            "BUFFER_NEW",
            "BUFFER_FROM_STACK",
            "BUFFER_TO_STACK",
            "BUFFER_LEN",
            "BUFFER_READ_BYTE",
            "BUFFER_WRITE_BYTE",
            "BUFFER_FREE",
            "TCP_CONNECT",
            "SOCKET_SEND",
            "SOCKET_RECV",
            "SOCKET_CLOSE",
            "PROC_EXEC",
            "CLOCK",
            "SLEEP",
            "RANDOM",
        ];

        // Simple prefix matching
        for &op in OPCODES {
            if op.starts_with(unknown) || unknown.starts_with(op) {
                return Some(op);
            }
        }

        // Levenshtein distance 1 check (simple character swap/add/remove)
        for &op in OPCODES {
            if (unknown.len() as i32 - op.len() as i32).abs() <= 1 {
                let mut diff = 0;
                for (a, b) in unknown.chars().zip(op.chars()) {
                    if a != b {
                        diff += 1;
                    }
                }
                if diff <= 1 {
                    return Some(op);
                }
            }
        }

        None
    }

    /// Parse a LET binding: LET name = expr;
    fn parse_let(&mut self) -> Result<Stmt, String> {
        // Get variable name
        let name = match self.tokens.next() {
            Some(Token::Word(w)) => w.to_lowercase(),
            _ => return Err("Expected variable name after LET".to_string()),
        };

        // Expect '='
        match self.tokens.next() {
            Some(Token::Equals) => {}
            Some(Token::Word(w)) if w == "=" => {}
            _ => return Err("Expected '=' after variable name in LET".to_string()),
        }

        // Parse the expression (collect statements until semicolon)
        let mut expr_stmts = Vec::new();
        loop {
            match self.tokens.peek() {
                Some(Token::Semicolon) => {
                    self.tokens.next(); // consume ;
                    break;
                }
                Some(Token::RBrace) | None => {
                    // End of block or input - allow missing semicolon
                    break;
                }
                _ => {
                    let (statement, _) = self.parse_stmt()?;
                    expr_stmts.push(statement);
                }
            }
        }

        // Register the variable at current stack depth
        // (the expression left its result on the stack)
        self.variables.insert(
            name,
            VariableBinding {
                stack_depth: self.stack_depth - 1,
            },
        );

        // Return the expression statements as a block
        if expr_stmts.len() == 1 {
            Ok(expr_stmts.remove(0))
        } else {
            Ok(Stmt::Block(expr_stmts))
        }
    }

    /// Parse and retain `TEMPORAL <name> @ <addr> DEFAULT <val>;`.
    fn parse_temporal_declaration(&mut self, declaration_start: usize) -> Result<(), String> {
        // Get variable name
        let name = match self.tokens.next() {
            Some(Token::Word(w)) => w.to_lowercase(),
            _ => return Err(self.error("Expected variable name after TEMPORAL").into()),
        };

        // Expect '@' symbol
        match self.tokens.next() {
            Some(Token::Word(w)) if w == "@" => {}
            _ => {
                return Err(self
                    .error_with_help(
                        "Expected '@' after TEMPORAL variable name",
                        "Syntax: TEMPORAL name @ address DEFAULT value;",
                    )
                    .into())
            }
        }

        // Get address
        let address = match self.tokens.next() {
            Some(Token::Number(n)) => *n,
            _ => return Err(self.error("Expected address after '@'").into()),
        };

        // Expect 'DEFAULT' keyword
        match self.tokens.next() {
            Some(Token::Word(w)) if w.to_uppercase() == "DEFAULT" => {}
            _ => {
                return Err(self
                    .error_with_help(
                        "Expected 'DEFAULT' after address",
                        "Syntax: TEMPORAL name @ address DEFAULT value;",
                    )
                    .into())
            }
        }

        let default = match self.tokens.next() {
            Some(Token::Number(value)) => *value,
            _ => return Err(self.error("Expected default value after DEFAULT").into()),
        };

        self.expect_semicolon("TEMPORAL declaration")?;

        if self.temporal_vars.contains_key(&name) {
            self.last_error_source_span = self.consumed_span(declaration_start);
            return Err(format!("Duplicate TEMPORAL name '{}'", name));
        }
        if let Some(first_name) = self
            .temporal_declarations
            .iter()
            .find(|declaration| declaration.address == address)
            .map(|declaration| declaration.name.clone())
        {
            self.last_error_source_span = self.consumed_span(declaration_start);
            return Err(format!(
                "TEMPORAL '{}' reuses address {} declared by '{}'",
                name, address, first_name
            ));
        }
        self.temporal_vars
            .insert(name.clone(), TemporalBinding { address });
        self.temporal_declarations.push(TemporalDeclaration {
            name,
            address,
            default,
        });
        if let Some(span) = self.consumed_span(declaration_start) {
            self.temporal_locations.push(span);
        }
        Ok(())
    }

    /// Parse a procedure definition: PROCEDURE name(params) [EFFECTS] { body }
    fn parse_procedure_def(&mut self) -> Result<(), String> {
        // Procedure token already consumed by parse_statement dispatch
        let declaration_start = self.position().saturating_sub(1);

        // Get procedure name
        let name = match self.tokens.next() {
            Some(Token::Word(w)) => w.to_lowercase(),
            _ => return Err("Expected procedure name after PROCEDURE".to_string()),
        };

        // Parse optional parameter list
        let mut params = Vec::new();
        // Check for LParen (preferred) or Word("(") (legacy)
        let has_paren = matches!(self.tokens.peek(), Some(Token::LParen)) || self.peek_word_eq("(");

        if has_paren {
            self.tokens.next(); // consume (
            loop {
                match self.tokens.peek() {
                    Some(Token::RParen) => {
                        self.tokens.next();
                        break;
                    }
                    Some(Token::Word(w)) if w == ")" => {
                        self.tokens.next();
                        break;
                    }
                    Some(Token::Comma) => {
                        self.tokens.next();
                    }
                    Some(Token::Word(w)) if w == "," => {
                        self.tokens.next();
                    }
                    Some(Token::Word(w)) => {
                        params.push(w.to_lowercase());
                        self.tokens.next();
                    }
                    _ => break,
                }
            }
        }

        // Parse effect annotations. Keep this exhaustive over the source-level
        // Effect vocabulary so every contract accepted by the AST can also be
        // written on an ordinary procedure.
        let mut effects = Vec::new();
        loop {
            // Stop if we see '{' which starts the body
            if matches!(self.tokens.peek(), Some(Token::LBrace)) {
                break;
            }

            match self.tokens.peek() {
                Some(Token::Word(w)) => {
                    let w_upper = w.to_uppercase();
                    let effect = match w_upper.as_str() {
                        "PURE" => {
                            self.tokens.next();
                            Effect::Pure
                        }
                        "READS" => {
                            self.tokens.next();
                            Effect::Reads(self.parse_effect_arg()?)
                        }
                        "WRITES" => {
                            self.tokens.next();
                            Effect::Writes(self.parse_effect_arg()?)
                        }
                        "TEMPORAL" => {
                            self.tokens.next();
                            Effect::Temporal
                        }
                        "IO" => {
                            self.tokens.next();
                            Effect::IO
                        }
                        "ALLOC" => {
                            self.tokens.next();
                            Effect::Alloc
                        }
                        "FFI" => {
                            self.tokens.next();
                            Effect::FFI
                        }
                        "FILE_IO" => {
                            self.tokens.next();
                            Effect::FileIO
                        }
                        "NETWORK" => {
                            self.tokens.next();
                            Effect::Network
                        }
                        "SYSTEM" => {
                            self.tokens.next();
                            Effect::System
                        }
                        // Not an effect keyword: let parse_block produce the
                        // useful "expected '{'" diagnostic for stray words.
                        _ => break,
                    };
                    effects.push(effect);
                }
                _ => break,
            }
        }

        // Parse body block
        let body = self.parse_block()?;

        // Register procedure
        let param_count = params.len();
        self.procedures.insert(
            name.clone(),
            ProcedureBinding {
                param_count,
                return_count: 1, // Default: procedures return 1 value
            },
        );

        self.procedure_defs.push(Procedure {
            name,
            params,
            returns: 1,
            effects,
            body: body.statements,
        });
        if let Some(declaration) = self.consumed_span(declaration_start) {
            self.procedure_locations.push(Some(ProcedureLocations {
                declaration,
                body: body.locations,
            }));
        }

        Ok(())
    }

    /// Parse a FOREIGN block: FOREIGN "library" { declarations... }
    ///
    /// Syntax:
    /// ```text
    /// FOREIGN "libc" {
    ///     PROC strlen ( ptr: u64 -- len: u64 ) PURE;
    ///     PROC puts ( ptr: u64 -- result: i32 ) IO;
    /// }
    /// ```
    fn parse_foreign_block(&mut self) -> Result<(), String> {
        // FOREIGN token already consumed

        // Get library name (string literal)
        let library_name = match self.tokens.next() {
            Some(Token::StringLit(s)) => s.clone(),
            _ => return Err("Expected library name string after FOREIGN".to_string()),
        };

        // Expect '{'
        match self.tokens.next() {
            Some(Token::LBrace) => {}
            _ => return Err("Expected '{' after FOREIGN library name".to_string()),
        }

        // Parse function declarations until '}'
        loop {
            match self.tokens.peek() {
                Some(Token::RBrace) => {
                    self.tokens.next(); // consume }
                    break;
                }
                Some(Token::Word(w)) if w.to_uppercase() == "PROC" || w.to_uppercase() == "FN" => {
                    self.tokens.next(); // consume PROC/FN
                    self.parse_foreign_function(&library_name)?;
                }
                None => return Err("Unclosed FOREIGN block, expected '}'".to_string()),
                _ => {
                    // Skip unknown tokens or consume semicolons
                    if let Some(Token::Semicolon) = self.tokens.peek() {
                        self.tokens.next();
                    } else {
                        return Err("Expected PROC or FN declaration in FOREIGN block".to_string());
                    }
                }
            }
        }

        Ok(())
    }

    /// Parse a single FFI function declaration.
    ///
    /// Syntax: PROC name ( param: type, ... -- return_type, ... ) EFFECT;
    fn parse_foreign_function(&mut self, library: &str) -> Result<(), String> {
        let declaration_start = self.position().saturating_sub(1);
        // Get function name
        let name = match self.tokens.next() {
            Some(Token::Word(w)) => w.clone(),
            _ => return Err("Expected function name after PROC in FOREIGN block".to_string()),
        };

        // Optional external symbol name: PROC ouro_name AS "c_name"
        let symbol_name = if self.peek_word_eq("AS") {
            self.tokens.next(); // consume AS
            match self.tokens.next() {
                Some(Token::StringLit(s)) => Some(s.clone()),
                _ => return Err("Expected string literal after AS".to_string()),
            }
        } else {
            None
        };

        // Parse parameter list: ( param: type, ... -- return_type, ... )
        let mut params = Vec::new();
        let mut returns = Vec::new();
        let mut is_return_section = false;

        match self.tokens.next() {
            Some(Token::LParen) => {}
            _ => return Err("Expected '(' after function name in FOREIGN".to_string()),
        }

        loop {
            match self.tokens.peek() {
                Some(Token::RParen) => {
                    self.tokens.next(); // consume )
                    break;
                }
                Some(Token::Word(w)) if w == "--" => {
                    self.tokens.next(); // consume --
                    is_return_section = true;
                }
                Some(Token::Word(w)) if w == "-" => {
                    // Check for -- as two separate - tokens
                    self.tokens.next(); // consume first -
                    if let Some(Token::Word(next)) = self.tokens.peek() {
                        if next == "-" {
                            self.tokens.next(); // consume second -
                            is_return_section = true;
                        }
                    }
                }
                Some(Token::Comma) => {
                    self.tokens.next(); // consume comma
                }
                Some(Token::Word(w)) if w != ":" && !w.chars().all(|c| c == '-') => {
                    let param_or_type_name = w.clone();
                    self.tokens.next();

                    // Check for `:` (param: type) or just type
                    match self.tokens.peek() {
                        Some(Token::Word(colon)) if colon == ":" => {
                            self.tokens.next(); // consume :
                                                // Get type name
                            let type_name = match self.tokens.next() {
                                Some(Token::Word(t)) => t.clone(),
                                _ => return Err("Expected type name after ':'".to_string()),
                            };
                            let ffi_type = FFIType::from_str(&type_name)?;

                            if is_return_section {
                                returns.push(ffi_type);
                            } else {
                                params.push((param_or_type_name, ffi_type));
                            }
                        }
                        _ => {
                            // Just a type name (no parameter name)
                            let ffi_type = FFIType::from_str(&param_or_type_name)?;

                            if is_return_section {
                                returns.push(ffi_type);
                            } else {
                                params.push((format!("arg{}", params.len()), ffi_type));
                            }
                        }
                    }
                }
                None => return Err("Unclosed parameter list in FOREIGN function".to_string()),
                _ => {
                    self.tokens.next(); // skip unknown (including standalone :)
                }
            }
        }

        // Parse effect annotations (PURE, IO, READS, WRITES, TEMPORAL, ALLOC)
        let mut effects = Vec::new();
        loop {
            match self.tokens.peek() {
                Some(Token::Word(w)) => {
                    let upper = w.to_uppercase();
                    match upper.as_str() {
                        "PURE" => {
                            self.tokens.next();
                            effects.push(FFIEffect::Pure);
                        }
                        "IO" => {
                            self.tokens.next();
                            effects.push(FFIEffect::IO);
                        }
                        "READS" => {
                            self.tokens.next();
                            effects.push(FFIEffect::Reads);
                        }
                        "WRITES" => {
                            self.tokens.next();
                            effects.push(FFIEffect::Writes);
                        }
                        "TEMPORAL" => {
                            self.tokens.next();
                            effects.push(FFIEffect::Temporal);
                        }
                        "ALLOC" => {
                            self.tokens.next();
                            effects.push(FFIEffect::Alloc);
                        }
                        _ => break,
                    }
                }
                Some(Token::Semicolon) => {
                    self.tokens.next(); // consume ;
                    break;
                }
                _ => break,
            }
        }

        // Default to IO if no effects specified
        if effects.is_empty() {
            effects.push(FFIEffect::IO);
        }

        // Build FFI signature
        let mut signature = FFISignature::new(&name, library);
        for (param_name, param_type) in params {
            signature = signature.param(param_name, param_type);
        }
        for ret_type in returns {
            signature = signature.returns_type(ret_type);
        }
        signature = signature.effects(effects);

        // Save sizes before moving signature
        let param_count = signature.input_stack_size();
        let return_count = signature.output_stack_size();

        // Register declaration
        self.ffi_declarations.push(FFIDeclaration {
            signature,
            symbol_name,
        });
        if let Some(span) = self.consumed_span(declaration_start) {
            self.foreign_locations.push(span);
        }

        // Also register as a procedure binding for type checking
        self.procedures.insert(
            name.to_lowercase(),
            ProcedureBinding {
                param_count,
                return_count,
            },
        );

        Ok(())
    }

    /// Parse argument for effect: (addr)
    fn parse_effect_arg(&mut self) -> Result<u64, String> {
        // Expect '('
        match self.tokens.next() {
            Some(Token::LParen) => {}
            Some(Token::Word(w)) if w == "(" => {}
            _ => return Err("Expected '(' after effect keyword".to_string()),
        }

        // Expect number (for now only constant addresses supported)
        let addr = match self.tokens.next() {
            Some(Token::Number(n)) => *n,
            _ => return Err("Expected address number in effect annotation".to_string()),
        };

        // Expect ')'
        match self.tokens.next() {
            Some(Token::RParen) => {}
            Some(Token::Word(w)) if w == ")" => {}
            _ => return Err("Expected ')' after effect address".to_string()),
        }

        Ok(addr)
    }

    /// Parse a block enclosed in braces.
    fn parse_block(&mut self) -> Result<ParsedBlock, String> {
        match self.tokens.next() {
            Some(Token::LBrace) => self.parse_block_content(),
            _ => Err("Expected '{'".to_string()),
        }
    }

    /// Parse the content of a block (after '{').
    fn parse_block_content(&mut self) -> Result<ParsedBlock, String> {
        self.with_nesting("block", Self::parse_block_content_inner)
    }

    fn parse_block_content_inner(&mut self) -> Result<ParsedBlock, String> {
        let mut stmts = Vec::new();
        let mut locations = Vec::new();

        loop {
            match self.tokens.peek() {
                Some(Token::RBrace) => {
                    self.tokens.next(); // consume '}'
                    return Ok(ParsedBlock {
                        statements: stmts,
                        locations,
                    });
                }
                None => return Err("Unclosed block, expected '}'".to_string()),
                _ => {
                    let (statement, location) = self.parse_stmt()?;
                    stmts.push(statement);
                    if let Some(location) = location {
                        locations.push(location);
                    }
                }
            }
        }
    }

    /// Check if the next token is a word equal to the given string.
    fn peek_word_eq(&mut self, expected: &str) -> bool {
        match self.tokens.peek() {
            Some(Token::Word(w)) => w.to_uppercase() == expected,
            _ => false,
        }
    }

    // ========================================================================
    // Expression Parser (Operator Precedence)
    // ========================================================================

    /// Check if next token starts an expression (for detecting expression context)
    fn is_expression_start(&mut self) -> bool {
        match self.tokens.peek() {
            Some(Token::LParen) | Some(Token::Number(_)) => true,
            Some(Token::Word(w)) => !self.is_keyword_str(w),
            _ => false,
        }
    }

    /// Check if a string is a control-flow keyword
    fn is_keyword_str(&self, word: &str) -> bool {
        matches!(
            word.to_uppercase().as_str(),
            "IF" | "ELSE"
                | "WHILE"
                | "LET"
                | "MANIFEST"
                | "FAMILY"
                | "MARKOV"
                | "QCHANNEL"
                | "PROCEDURE"
                | "PROC"
                | "TEMPORAL"
                | "ASSERT"
        )
    }

    /// Parse a quotation: [ stmts ]
    /// Stores the block in self.quotes and returns Stmt::PushQuote(quote_id).
    fn parse_quote(&mut self) -> Result<Stmt, String> {
        self.with_nesting("quotation", Self::parse_quote_inner)
    }

    fn parse_quote_inner(&mut self) -> Result<Stmt, String> {
        // LBracket already consumed by parse_stmt

        let mut stmts = Vec::new();
        let mut locations = Vec::new();
        // Quote acts as a nested stack context?
        // Actually, internal statements don't affect current stack depth until executed.
        // But for parsing verification (stack effect), we assume it's standalone?
        // Or do we defer check? For now, we just parse stmts.

        // Save current depth to restore it? No, parser tracks linear stack depth.
        // But code inside [ ] doesn't run "now".
        // So effectively [ ] pushes 1 item (the quote ID).
        // The effects of *running* the quote are unknown to simple parser.
        // We act like [ ] consumes nothing (except internal) and pushes 1.

        let depth_before = self.stack_depth;

        loop {
            match self.tokens.peek() {
                Some(Token::RBracket) => {
                    self.tokens.next(); // consume ]
                    break;
                }
                None => return Err("Unclosed quotation, expected ']'".to_string()),
                _ => {
                    let (statement, location) = self.parse_stmt()?;
                    stmts.push(statement);
                    if let Some(location) = location {
                        locations.push(location);
                    }
                }
            }
        }

        // Restore stack depth because the stmts inside didn't actually happen to *us*
        self.stack_depth = depth_before;
        // But the quote itself is a value pushed to stack
        self.stack_depth += 1;

        let id = self.quotes.len() as u64;
        self.quotes.push(stmts);
        if self.source_spans.is_some() {
            self.quote_locations.push(locations);
        }

        Ok(Stmt::PushQuote(QuoteId::new(id)))
    }

    /// Parse a parenthesized expression: (expr)
    /// Returns statements that evaluate the expression and leave result on stack
    fn parse_expression(&mut self) -> Result<Vec<Stmt>, String> {
        // Must start with (
        match self.tokens.next() {
            Some(Token::LParen) => {}
            _ => return Err("Expected '(' to start expression".to_string()),
        }

        let stmts = self.parse_expr_or()?;

        // Must end with )
        match self.tokens.next() {
            Some(Token::RParen) => {}
            _ => return Err("Expected ')' to close expression".to_string()),
        }

        Ok(stmts)
    }

    /// Parse OR expression: expr && expr && ...
    fn parse_expr_or(&mut self) -> Result<Vec<Stmt>, String> {
        let mut stmts = self.parse_expr_and()?;

        while self.peek_operator("||") || self.peek_word_eq("OR") {
            self.tokens.next(); // consume ||
            let right = self.parse_expr_and()?;
            stmts.extend(right);
            self.reserve_expanded_statements(1, "OR expression")?;
            stmts.push(Stmt::Op(OpCode::Or));
            self.stack_depth = self.stack_depth.saturating_sub(1);
        }

        Ok(stmts)
    }

    /// Parse AND expression: expr || expr || ...
    fn parse_expr_and(&mut self) -> Result<Vec<Stmt>, String> {
        let mut stmts = self.parse_expr_comparison()?;

        while self.peek_operator("&&") || self.peek_word_eq("AND") {
            self.tokens.next(); // consume &&
            let right = self.parse_expr_comparison()?;
            stmts.extend(right);
            self.reserve_expanded_statements(1, "AND expression")?;
            stmts.push(Stmt::Op(OpCode::And));
            self.stack_depth = self.stack_depth.saturating_sub(1);
        }

        Ok(stmts)
    }

    /// Parse comparison: expr (< | > | <= | >= | == | !=) expr
    fn parse_expr_comparison(&mut self) -> Result<Vec<Stmt>, String> {
        let mut stmts = self.parse_expr_additive()?;

        loop {
            let op = if self.peek_operator("==") || self.peek_word_eq("EQ") {
                self.tokens.next();
                Some(OpCode::Eq)
            } else if self.peek_operator("!=") || self.peek_word_eq("NEQ") {
                self.tokens.next();
                Some(OpCode::Neq)
            } else if self.peek_operator("<=") || self.peek_word_eq("LTE") {
                self.tokens.next();
                Some(OpCode::Lte)
            } else if self.peek_operator(">=") || self.peek_word_eq("GTE") {
                self.tokens.next();
                Some(OpCode::Gte)
            } else if self.peek_operator("<") || self.peek_word_eq("LT") {
                self.tokens.next();
                Some(OpCode::Lt)
            } else if self.peek_operator(">") || self.peek_word_eq("GT") {
                self.tokens.next();
                Some(OpCode::Gt)
            } else {
                None
            };

            if let Some(opcode) = op {
                let right = self.parse_expr_additive()?;
                stmts.extend(right);
                self.reserve_expanded_statements(1, "comparison expression")?;
                stmts.push(Stmt::Op(opcode));
                self.stack_depth = self.stack_depth.saturating_sub(1);
            } else {
                break;
            }
        }

        Ok(stmts)
    }

    /// Parse additive: expr (+ | -) expr
    fn parse_expr_additive(&mut self) -> Result<Vec<Stmt>, String> {
        let mut stmts = self.parse_expr_multiplicative()?;

        loop {
            let op = if self.peek_operator("+") {
                self.tokens.next();
                Some(OpCode::Add)
            } else if self.peek_operator("-") {
                self.tokens.next();
                Some(OpCode::Sub)
            } else {
                None
            };

            if let Some(opcode) = op {
                let right = self.parse_expr_multiplicative()?;
                stmts.extend(right);
                self.reserve_expanded_statements(1, "additive expression")?;
                stmts.push(Stmt::Op(opcode));
                self.stack_depth = self.stack_depth.saturating_sub(1);
            } else {
                break;
            }
        }

        Ok(stmts)
    }

    /// Parse multiplicative: expr (* | / | %) expr
    fn parse_expr_multiplicative(&mut self) -> Result<Vec<Stmt>, String> {
        let mut stmts = self.parse_expr_unary()?;

        loop {
            let op = if self.peek_operator("*") {
                self.tokens.next();
                Some(OpCode::Mul)
            } else if self.peek_operator("/") {
                self.tokens.next();
                Some(OpCode::Div)
            } else if self.peek_operator("%") {
                self.tokens.next();
                Some(OpCode::Mod)
            } else {
                None
            };

            if let Some(opcode) = op {
                let right = self.parse_expr_unary()?;
                stmts.extend(right);
                self.reserve_expanded_statements(1, "multiplicative expression")?;
                stmts.push(Stmt::Op(opcode));
                self.stack_depth = self.stack_depth.saturating_sub(1);
            } else {
                break;
            }
        }

        Ok(stmts)
    }

    /// Parse unary: !expr, ~expr, -expr, or primary
    fn parse_expr_unary(&mut self) -> Result<Vec<Stmt>, String> {
        self.with_nesting("expression", Self::parse_expr_unary_inner)
    }

    fn parse_expr_unary_inner(&mut self) -> Result<Vec<Stmt>, String> {
        if self.peek_operator("!") || self.peek_operator("~") {
            self.tokens.next();
            let mut stmts = self.parse_expr_unary()?;
            self.reserve_expanded_statements(1, "unary expression")?;
            stmts.push(Stmt::Op(OpCode::Not));
            Ok(stmts)
        } else {
            self.parse_expr_primary()
        }
    }

    /// Parse primary expression: number, variable, function call, or (expr)
    fn parse_expr_primary(&mut self) -> Result<Vec<Stmt>, String> {
        // Peek and clone necessary data before consuming
        let peeked = self.tokens.peek().cloned();
        match peeked {
            Some(Token::Number(n)) => {
                self.tokens.next();
                self.stack_depth += 1;
                self.reserve_expanded_statements(1, "numeric expression")?;
                Ok(vec![Stmt::Push(Value::new(*n))])
            }

            Some(Token::LParen) => self.parse_expression(),

            Some(Token::Word(w)) => {
                let word = w;
                let upper = word.to_uppercase();

                // Check for ORACLE(addr) syntax
                if upper == "ORACLE" {
                    self.tokens.next();
                    // Check for function-call syntax
                    if matches!(self.tokens.peek(), Some(Token::LParen)) {
                        self.tokens.next(); // consume (
                        let addr_stmts = self.parse_expr_or()?; // parse address expression
                        match self.tokens.next() {
                            Some(Token::RParen) => {}
                            _ => return Err("Expected ')' after ORACLE address".to_string()),
                        }
                        let mut stmts = addr_stmts;
                        self.reserve_expanded_statements(1, "ORACLE expression")?;
                        stmts.push(Stmt::Op(OpCode::Oracle));
                        // Oracle pops 1 (address), pushes 1 (value) - net: 0, but we added 1 for address
                        return Ok(stmts);
                    } else {
                        // Plain ORACLE token - let parse_word handle it
                        self.stack_depth += 1;
                        self.reserve_expanded_statements(1, "ORACLE expression")?;
                        return Ok(vec![Stmt::Op(OpCode::Oracle)]);
                    }
                }

                // Check for PROPHECY(addr, value) syntax
                if upper == "PROPHECY" {
                    self.tokens.next();
                    if matches!(self.tokens.peek(), Some(Token::LParen)) {
                        self.tokens.next(); // consume (
                        let value_stmts = self.parse_expr_or()?;
                        match self.tokens.next() {
                            Some(Token::Comma) => {}
                            _ => return Err("Expected ',' in PROPHECY(value, addr)".to_string()),
                        }
                        let addr_stmts = self.parse_expr_or()?;
                        match self.tokens.next() {
                            Some(Token::RParen) => {}
                            _ => return Err("Expected ')' after PROPHECY".to_string()),
                        }
                        let mut stmts = value_stmts;
                        stmts.extend(addr_stmts);
                        self.reserve_expanded_statements(1, "PROPHECY expression")?;
                        stmts.push(Stmt::Op(OpCode::Prophecy));
                        self.stack_depth = self.stack_depth.saturating_sub(2);
                        return Ok(stmts);
                    } else {
                        self.reserve_expanded_statements(1, "PROPHECY expression")?;
                        return Ok(vec![Stmt::Op(OpCode::Prophecy)]);
                    }
                }

                self.tokens.next();

                // Check if it's a defined constant
                if let Some(&val) = self.constants.get(&upper) {
                    self.stack_depth += 1;
                    self.reserve_expanded_statements(1, "constant expression")?;
                    return Ok(vec![Stmt::PushConstant {
                        name: upper,
                        value: val,
                    }]);
                }

                // Check if it's a defined variable
                let lower = word.to_lowercase();
                if let Some(binding) = self.variables.get(&lower).cloned() {
                    let pick_index = self.stack_depth - binding.stack_depth - 1;
                    self.stack_depth += 1;
                    self.reserve_expanded_statements(2, "variable expression")?;
                    return Ok(vec![
                        Stmt::Push(Value::new(pick_index as u64)),
                        Stmt::Op(OpCode::Pick),
                    ]);
                }

                // Check if it's a temporal variable. Expands exactly like the
                // statement path: <address> ORACLE. The DEFAULT clause is not
                // an ORACLE operand; a second push here would be popped as
                // the address (reading anamnesis[default]) and would leave
                // the real address behind as stack litter.
                if let Some(binding) = self.temporal_vars.get(&lower).cloned() {
                    self.stack_depth += 1;
                    self.reserve_expanded_statements(1, "temporal expression")?;
                    return Ok(vec![Stmt::ReadTemporal {
                        name: lower,
                        address: binding.address,
                    }]);
                }

                // Procedure call?
                if let Some(proc) = self.procedures.get(&lower) {
                    // Check argument count? Parser doesn't strictly enforce stack depth yet...
                    self.stack_depth = self.stack_depth.saturating_sub(proc.param_count);
                    self.stack_depth += proc.return_count;
                    self.reserve_expanded_statements(1, "procedure-call expression")?;
                    return Ok(vec![Stmt::Call { name: lower }]);
                }

                Err(format!("Unknown identifier: {}", word))
            }

            _ => Err("Expected expression".to_string()),
        }
    }

    /// Parse IMPORT statement: IMPORT "filename"
    fn parse_import(&mut self) -> Result<(), String> {
        self.tokens.next(); // Consume IMPORT

        let filename = match self.tokens.next() {
            Some(Token::StringLit(s)) => s.clone(),
            _ => return Err("Expected filename string after IMPORT".to_string()),
        };

        if self.defer_imports {
            self.deferred_imports.push(filename);
            return Ok(());
        }
        Err(format!(
            "IMPORT requires a file-backed ModuleGraph (found: {filename})"
        ))
    }

    /// Check if next token is a specific operator
    fn peek_operator(&mut self, op: &str) -> bool {
        match self.tokens.peek() {
            Some(Token::Word(w)) => w == op,
            _ => false,
        }
    }
}

/// Parse source code directly into a program.
pub fn parse(source: &str) -> Result<Program, String> {
    let tokens: Vec<Token> = crate::lexer::lex(crate::source::SourceId::new(0), source)
        .map_err(|errors| {
            errors
                .into_iter()
                .map(|error| error.to_string())
                .collect::<Vec<_>>()
                .join("\n")
        })?
        .into_iter()
        .map(|located| located.token)
        .collect();
    let mut parser = Parser::new_with_deferred_imports(&tokens);
    let program = parser.parse_program()?;
    let imports = parser.take_deferred_imports();
    if imports.is_empty() {
        Ok(program)
    } else {
        Err(format!(
            "IMPORT requires a file-backed ModuleGraph (found: {})",
            imports.join(", ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_numbers() {
        let tokens = tokenize("42 0xFF 0b1010");
        assert_eq!(
            tokens,
            vec![Token::Number(42), Token::Number(255), Token::Number(10),]
        );
    }

    #[test]
    fn test_tokenize_words() {
        let tokens = tokenize("ADD SUB ORACLE");
        assert_eq!(
            tokens,
            vec![
                Token::Word("ADD".to_string()),
                Token::Word("SUB".to_string()),
                Token::Word("ORACLE".to_string()),
            ]
        );
    }

    #[test]
    fn test_tokenize_comments() {
        let tokens = tokenize("ADD # this is a comment\nSUB");
        assert_eq!(
            tokens,
            vec![
                Token::Word("ADD".to_string()),
                Token::Word("SUB".to_string()),
            ]
        );
    }

    #[test]
    fn test_parse_simple() {
        let program = parse("10 20 ADD OUTPUT").unwrap();
        assert_eq!(program.body.len(), 4);
    }

    #[test]
    fn markov_state_dimension_is_bounded_before_dense_allocation() {
        for count in [MAX_MARKOV_STATES as u64 + 1, usize::MAX as u64] {
            let error = parse(&format!("MARKOV huge {{ STATES {count}; }}")).unwrap_err();
            assert_eq!(
                error,
                format!("MARKOV state count {count} exceeds maximum {MAX_MARKOV_STATES}")
            );
        }
    }

    #[test]
    fn compact_string_expansion_reserves_budget_before_allocation() {
        let tokens = tokenize("\"12345678\" OUTPUT");
        let mut parser = Parser::new_with_expanded_statement_limit(&tokens, 16);
        assert_eq!(
            parser.parse_program().unwrap_err(),
            "expanded statement budget exceeds 16 while parsing string literal"
        );
    }

    #[test]
    fn expansion_budget_is_cumulative_across_all_statement_owners() {
        let tokens = tokenize("PROCEDURE p { \"a\" OUTPUT } [ { \"b\" OUTPUT } ] \"c\" OUTPUT");
        let mut parser = Parser::new_with_expanded_statement_limit(&tokens, 10);
        assert_eq!(
            parser.parse_program().unwrap_err(),
            "expanded statement budget exceeds 10 while parsing string literal"
        );
    }

    #[test]
    fn infallible_legacy_tokenizer_fails_closed_at_its_token_limit() {
        let panic = std::panic::catch_unwind(|| tokenize_with_spans_limit("1 2 3", 2));
        assert!(panic.is_err());
    }

    #[test]
    fn ordinary_procedures_parse_the_complete_effect_vocabulary() {
        let program = parse(
            "PROCEDURE contracted PURE READS(1) WRITES(2) TEMPORAL IO \
             ALLOC FFI FILE_IO NETWORK SYSTEM { 0 }",
        )
        .unwrap();

        assert_eq!(
            program.procedures[0].effects,
            vec![
                Effect::Pure,
                Effect::Reads(1),
                Effect::Writes(2),
                Effect::Temporal,
                Effect::IO,
                Effect::Alloc,
                Effect::FFI,
                Effect::FileIO,
                Effect::Network,
                Effect::System,
            ]
        );
    }

    #[test]
    fn direct_parse_rejects_unknown_characters() {
        let error = parse("1 $ OUTPUT").unwrap_err();
        assert!(error.contains("unknown character '$'"), "{error}");
    }

    #[test]
    fn string_followed_by_output_lowers_to_balanced_character_emission() {
        let program = parse("\"Hi\" OUTPUT").unwrap();
        let Stmt::Block(body) = &program.body[0] else {
            panic!("string output should lower to one block");
        };
        assert_eq!(body.len(), 4);
        assert!(matches!(&body[0], Stmt::Push(value) if value.val == 'H' as u64));
        assert!(matches!(&body[1], Stmt::Op(OpCode::Emit)));
        assert!(matches!(&body[2], Stmt::Push(value) if value.val == 'i' as u64));
        assert!(matches!(&body[3], Stmt::Op(OpCode::Emit)));
    }

    #[test]
    fn quotation_syntax_emits_a_typed_reference_not_a_word_literal() {
        let program = parse("7 [ 8 ]").unwrap();
        assert!(matches!(&program.body[0], Stmt::Push(value) if value.val == 7));
        assert!(matches!(
            &program.body[1],
            Stmt::PushQuote(id) if id.as_u64() == 0
        ));
        assert!(matches!(&program.quotes[0][0], Stmt::Push(value) if value.val == 8));
    }

    #[test]
    fn deferred_imports_record_paths_and_use_registered_interfaces() {
        let dependency = parse("PROCEDURE answer { 42 }").unwrap();
        let tokens = tokenize("IMPORT \"dep/answer.ouro\" answer");
        let mut parser = Parser::new_with_deferred_imports(&tokens);
        parser.register_imported_interface(&dependency);
        let program = parser.parse_program().unwrap();

        assert_eq!(parser.take_deferred_imports(), vec!["dep/answer.ouro"]);
        assert!(matches!(
            &program.body[0],
            Stmt::Call { name } if name == "answer"
        ));
        assert!(program.procedures.is_empty());
    }

    #[test]
    fn ordinary_parser_rejects_imports_without_filesystem_io() {
        let tokens = tokenize("IMPORT \"present-but-untrusted.ouro\"");
        let mut parser = Parser::new(&tokens);

        assert_eq!(
            parser.parse_program().unwrap_err(),
            "IMPORT requires a file-backed ModuleGraph (found: present-but-untrusted.ouro)"
        );
    }

    #[test]
    fn recursive_language_constructs_fail_at_the_parser_nesting_limit() {
        let cases = [
            (
                format!(
                    "{}0{}",
                    "{".repeat(MAX_PARSER_NESTING + 1),
                    "}".repeat(MAX_PARSER_NESTING + 1)
                ),
                "block",
            ),
            (
                format!(
                    "{}0{}",
                    "[".repeat(MAX_PARSER_NESTING + 1),
                    "]".repeat(MAX_PARSER_NESTING + 1)
                ),
                "quotation",
            ),
            (
                format!(
                    "{}0{}",
                    "(".repeat(MAX_PARSER_NESTING + 1),
                    ")".repeat(MAX_PARSER_NESTING + 1)
                ),
                "expression",
            ),
            (
                format!("({}0)", "!".repeat(MAX_PARSER_NESTING + 1)),
                "expression",
            ),
        ];

        for (source, context) in cases {
            assert!(source.len() < 1_024);
            let error = parse(&source).unwrap_err();
            assert_eq!(
                error,
                format!("parser nesting exceeds {MAX_PARSER_NESTING} while parsing {context}")
            );
        }
    }

    #[test]
    fn imported_nested_quotations_are_import_order_independent() {
        use crate::core::{Memory, OutputItem};
        use crate::vm::{EpochStatus, Executor};

        let unique = format!(
            "ouro-quote-import-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let directory = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&directory).unwrap();
        let alpha = directory.join("alpha.ouro");
        let beta = directory.join("beta.ouro");

        // Each procedure references an outer quote, and that quote references
        // an inner quote. The trailing integer zero must never be relocated.
        std::fs::write(
            &alpha,
            "PROCEDURE alpha { [ [ 11 OUTPUT ] EXEC ] EXEC 0 OUTPUT }",
        )
        .unwrap();
        std::fs::write(
            &beta,
            "PROCEDURE beta { [ [ 22 OUTPUT ] EXEC ] EXEC 0 OUTPUT }",
        )
        .unwrap();

        let forward_root = directory.join("forward.ouro");
        let reverse_root = directory.join("reverse.ouro");
        std::fs::write(
            &forward_root,
            "IMPORT \"alpha.ouro\" IMPORT \"beta.ouro\" alpha beta",
        )
        .unwrap();
        std::fs::write(
            &reverse_root,
            "IMPORT \"beta.ouro\" IMPORT \"alpha.ouro\" alpha beta",
        )
        .unwrap();
        let alpha_then_beta = crate::module_graph::ModuleGraph::load(forward_root, Vec::new())
            .unwrap()
            .into_program();
        let beta_then_alpha = crate::module_graph::ModuleGraph::load(reverse_root, Vec::new())
            .unwrap()
            .into_program();

        let execute = |program: Program| {
            let program = program.inline_procedures();
            let mut executor = Executor::new();
            executor.config.immediate_output = false;
            let result = executor.run_epoch(&program, &Memory::new());
            assert_eq!(result.status, EpochStatus::Finished);
            result
                .output
                .into_iter()
                .map(|item| match item {
                    OutputItem::Val(value) => value.val,
                    OutputItem::Char(value) => value as u64,
                })
                .collect::<Vec<_>>()
        };

        let forward = execute(alpha_then_beta);
        let reverse = execute(beta_then_alpha);
        assert_eq!(forward, vec![11, 0, 22, 0]);
        assert_eq!(reverse, forward);

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn test_parse_if() {
        let program = parse("1 IF { 42 OUTPUT } ELSE { 0 OUTPUT }").unwrap();
        match &program.body[1] {
            Stmt::If {
                then_branch,
                else_branch,
            } => {
                assert_eq!(then_branch.len(), 2);
                assert!(else_branch.is_some());
            }
            _ => panic!("Expected If statement"),
        }
    }

    #[test]
    fn test_trivially_consistent() {
        let program = parse("10 20 ADD OUTPUT").unwrap();
        assert!(program.is_trivially_consistent());

        let temporal = parse("0 ORACLE 0 PROPHECY").unwrap();
        assert!(!temporal.is_trivially_consistent());
    }

    #[test]
    fn test_tokenize_let() {
        let tokens = tokenize("LET x = 42;");
        assert_eq!(
            tokens,
            vec![
                Token::Word("LET".to_string()),
                Token::Word("x".to_string()),
                Token::Equals,
                Token::Number(42),
                Token::Semicolon,
            ]
        );
    }

    #[test]
    fn test_parse_let_simple() {
        // LET x = 42; should parse to Push(42)
        let program = parse("LET x = 42;").unwrap();
        assert_eq!(program.body.len(), 1);
        match &program.body[0] {
            Stmt::Push(v) => assert_eq!(v.val, 42),
            other => panic!("Expected Push, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_let_with_reference() {
        // LET x = 10; x should generate: Push(10), then a PICK to get x
        let program = parse("LET x = 10; x OUTPUT").unwrap();
        // Should have: Push(10), Block([Push(0), Pick]), Output
        assert_eq!(program.body.len(), 3);

        // First should be Push(10)
        match &program.body[0] {
            Stmt::Push(v) => assert_eq!(v.val, 10),
            _ => panic!("Expected Push for LET"),
        }

        // Second should be Block with PICK
        match &program.body[1] {
            Stmt::Block(stmts) => {
                assert_eq!(stmts.len(), 2);
                match &stmts[1] {
                    Stmt::Op(OpCode::Pick) => {}
                    _ => panic!("Expected PICK in variable reference"),
                }
            }
            _ => panic!("Expected Block for variable reference"),
        }
    }

    #[test]
    fn test_parse_let_expression() {
        // LET x = 10 20 ADD;
        let program = parse("LET x = 10 20 ADD;").unwrap();
        // Should have one Block containing Push(10), Push(20), Add
        match &program.body[0] {
            Stmt::Block(stmts) => {
                assert_eq!(stmts.len(), 3);
            }
            _ => panic!("Expected Block for multi-statement LET expression"),
        }
    }

    // ========================================================================
    // Expression Syntax Tests
    // ========================================================================

    #[test]
    fn test_expression_arithmetic() {
        // (10 + 20) should compile to Push(10), Push(20), Add
        let program = parse("(10 + 20) OUTPUT").unwrap();
        assert!(!program.body.is_empty());
    }

    #[test]
    fn test_expression_comparison() {
        // (x > 1) style comparison
        let program = parse("LET x = 5; (x > 1) OUTPUT").unwrap();
        assert!(program.body.len() >= 2);
    }

    #[test]
    fn test_expression_if_syntax() {
        // IF (1 > 0) { ... } new expression syntax
        let program = parse("IF (1 > 0) { 42 OUTPUT }").unwrap();
        assert!(!program.body.is_empty());
        // Should wrap expression eval and IF in a block
        match &program.body[0] {
            Stmt::Block(stmts) => {
                // Last statement should be IF
                assert!(matches!(stmts.last(), Some(Stmt::If { .. })));
            }
            _ => panic!("Expected Block wrapping expression and IF"),
        }
    }

    #[test]
    fn test_expression_complex() {
        // Complex expression: (x > 1 && x < 15)
        let program = parse("LET x = 5; (x > 1 && x < 15) OUTPUT").unwrap();
        assert!(program.body.len() >= 2);
    }

    #[test]
    fn test_expression_oracle_function() {
        // ORACLE(0) function syntax
        let program = parse("(ORACLE(0) > 1) OUTPUT").unwrap();
        assert!(!program.body.is_empty());
    }

    #[test]
    fn test_tokenize_parentheses() {
        let tokens = tokenize("(10 + 20)");
        assert!(matches!(tokens[0], Token::LParen));
        assert!(matches!(tokens[4], Token::RParen));
    }

    // ========================================================================
    // FOREIGN Declaration Tests
    // ========================================================================

    #[test]
    fn test_foreign_single_function() {
        let source = r#"
            FOREIGN "math" {
                PROC add ( a: u64, b: u64 -- result: u64 ) PURE;
            }
            42 OUTPUT
        "#;

        let program = parse(source).unwrap();
        assert_eq!(program.ffi_declarations.len(), 1);

        let decl = &program.ffi_declarations[0];
        assert_eq!(decl.signature.name, "add");
        assert_eq!(decl.signature.library, "math");
        assert_eq!(decl.signature.params.len(), 2);
        assert_eq!(decl.signature.returns.len(), 1);
    }

    #[test]
    fn test_foreign_multiple_functions() {
        let source = r#"
            FOREIGN "libc" {
                PROC strlen ( ptr: u64 -- len: u64 ) PURE;
                PROC puts ( ptr: u64 -- result: i32 ) IO;
            }
            42 OUTPUT
        "#;

        let program = parse(source).unwrap();
        assert_eq!(program.ffi_declarations.len(), 2);

        assert_eq!(program.ffi_declarations[0].signature.name, "strlen");
        assert_eq!(program.ffi_declarations[1].signature.name, "puts");
    }

    #[test]
    fn test_foreign_with_symbol_alias() {
        let source = r#"
            FOREIGN "c" {
                PROC ouro_sleep AS "sleep" ( seconds: u32 -- ) IO;
            }
            42 OUTPUT
        "#;

        let program = parse(source).unwrap();
        assert_eq!(program.ffi_declarations.len(), 1);

        let decl = &program.ffi_declarations[0];
        assert_eq!(decl.signature.name, "ouro_sleep");
        assert_eq!(decl.symbol_name, Some("sleep".to_string()));
    }

    #[test]
    fn test_foreign_effect_annotations() {
        let source = r#"
            FOREIGN "io" {
                PROC alloc ( size: u64 -- ptr: u64 ) ALLOC;
                PROC read ( fd: u64 -- data: u64 ) IO READS;
                PROC write ( fd: u64, data: u64 -- ) IO WRITES;
            }
            42 OUTPUT
        "#;

        let program = parse(source).unwrap();
        assert_eq!(program.ffi_declarations.len(), 3);

        // Check that alloc has ALLOC effect
        assert!(program.ffi_declarations[0]
            .signature
            .effects
            .iter()
            .any(|e| matches!(e, FFIEffect::Alloc)));

        // Check that read has IO and READS effects
        assert!(program.ffi_declarations[1]
            .signature
            .effects
            .iter()
            .any(|e| matches!(e, FFIEffect::IO)));
        assert!(program.ffi_declarations[1]
            .signature
            .effects
            .iter()
            .any(|e| matches!(e, FFIEffect::Reads)));
    }

    // ========================================================================
    // Span Tracking Tests (Phase 2)
    // ========================================================================

    #[test]
    fn test_tokenize_with_spans_basic() {
        let tokens = tokenize_with_spans("42 ADD");
        assert_eq!(tokens.len(), 2);

        // First token: 42 at line 1, column 1
        assert_eq!(tokens[0].token, Token::Number(42));
        assert_eq!(tokens[0].span.line, 1);
        assert_eq!(tokens[0].span.column, 1);
        assert_eq!(tokens[0].span.offset, 0);
        assert_eq!(tokens[0].span.len, 2);

        // Second token: ADD at line 1, column 4
        assert_eq!(tokens[1].token, Token::Word("ADD".to_string()));
        assert_eq!(tokens[1].span.line, 1);
        assert_eq!(tokens[1].span.column, 4);
        assert_eq!(tokens[1].span.offset, 3);
        assert_eq!(tokens[1].span.len, 3);
    }

    #[test]
    fn test_tokenize_with_spans_multiline() {
        let tokens = tokenize_with_spans("10\n20\n30");
        assert_eq!(tokens.len(), 3);

        assert_eq!(tokens[0].span.line, 1);
        assert_eq!(tokens[0].span.column, 1);

        assert_eq!(tokens[1].span.line, 2);
        assert_eq!(tokens[1].span.column, 1);

        assert_eq!(tokens[2].span.line, 3);
        assert_eq!(tokens[2].span.column, 1);
    }

    #[test]
    fn test_tokenize_with_spans_hex_binary() {
        let tokens = tokenize_with_spans("0xFF 0b1010");
        assert_eq!(tokens.len(), 2);

        // 0xFF = 4 bytes
        assert_eq!(tokens[0].token, Token::Number(255));
        assert_eq!(tokens[0].span.len, 4);

        // 0b1010 = 6 bytes
        assert_eq!(tokens[1].token, Token::Number(10));
        assert_eq!(tokens[1].span.len, 6);
    }

    #[test]
    fn test_tokenize_with_spans_string() {
        let tokens = tokenize_with_spans("\"hello\" ADD");
        assert_eq!(tokens.len(), 2);

        // "hello" = 7 bytes (including quotes)
        assert_eq!(tokens[0].token, Token::StringLit("hello".to_string()));
        assert_eq!(tokens[0].span.len, 7);
    }

    #[test]
    fn test_tokenize_with_spans_char() {
        let tokens = tokenize_with_spans("'a' 'b'");
        assert_eq!(tokens.len(), 2);

        // 'a' = 3 bytes
        assert_eq!(tokens[0].token, Token::CharLit('a'));
        assert_eq!(tokens[0].span.len, 3);
        assert_eq!(tokens[0].span.column, 1);

        // 'b' at column 5
        assert_eq!(tokens[1].token, Token::CharLit('b'));
        assert_eq!(tokens[1].span.column, 5);
    }

    #[test]
    fn test_tokenize_with_spans_braces() {
        let tokens = tokenize_with_spans("IF { 1 }");
        assert_eq!(tokens.len(), 4);

        assert_eq!(tokens[1].token, Token::LBrace);
        assert_eq!(tokens[1].span.column, 4);
        assert_eq!(tokens[1].span.len, 1);

        assert_eq!(tokens[3].token, Token::RBrace);
        assert_eq!(tokens[3].span.column, 8);
    }

    #[test]
    fn test_tokenize_with_spans_operators() {
        let tokens = tokenize_with_spans("<= >= != <<");
        assert_eq!(tokens.len(), 4);

        // Two-character operators should have len 2
        assert_eq!(tokens[0].span.len, 2);
        assert_eq!(tokens[1].span.len, 2);
        assert_eq!(tokens[2].span.len, 2);
        assert_eq!(tokens[3].span.len, 2);
    }

    #[test]
    fn test_tokenize_with_spans_comment_skipping() {
        let tokens = tokenize_with_spans("ADD # comment\nSUB");
        assert_eq!(tokens.len(), 2);

        assert_eq!(tokens[0].span.line, 1);
        assert_eq!(tokens[1].span.line, 2);
    }

    #[test]
    fn test_span_extract() {
        let source = "42 ADD 100";
        let tokens = tokenize_with_spans(source);

        // Verify that span.extract() returns the correct text
        assert_eq!(tokens[0].span.extract(source), "42");
        assert_eq!(tokens[1].span.extract(source), "ADD");
        assert_eq!(tokens[2].span.extract(source), "100");
    }

    #[test]
    fn test_span_merge() {
        let span1 = Span::new(1, 1, 0, 2); // "42"
        let span2 = Span::new(1, 4, 3, 3); // "ADD"

        let merged = span1.merge(&span2);
        assert_eq!(merged.line, 1);
        assert_eq!(merged.column, 1);
        assert_eq!(merged.offset, 0);
        assert_eq!(merged.len, 6); // "42 ADD"
    }

    #[test]
    fn test_tokenize_with_spans_utf8() {
        // Test that byte offsets are correct for UTF-8
        let tokens = tokenize_with_spans("x = 42");
        assert_eq!(tokens.len(), 3);

        assert_eq!(tokens[0].span.offset, 0); // x
        assert_eq!(tokens[1].span.offset, 2); // =
        assert_eq!(tokens[2].span.offset, 4); // 42
    }

    #[test]
    fn test_tokenize_backward_compatible() {
        // Ensure tokenize() still works as before
        let tokens_old = tokenize("10 20 ADD");
        let tokens_new: Vec<Token> = tokenize_with_spans("10 20 ADD")
            .into_iter()
            .map(|st| st.token)
            .collect();

        assert_eq!(tokens_old, tokens_new);
    }

    #[test]
    fn local_procedure_interfaces_allow_forward_and_recursive_calls() {
        let tokens = tokenize("PROCEDURE first(x) { second } PROCEDURE second { first } first");
        let mut parser = Parser::new(&tokens);
        let program = parser.parse_program().expect("program parses");

        assert_eq!(program.procedures.len(), 2);
        assert!(matches!(
            program.procedures[0].body.as_slice(),
            [Stmt::Call { name }] if name == "second"
        ));
        assert!(matches!(
            program.procedures[1].body.as_slice(),
            [Stmt::Call { name }] if name == "first"
        ));
        assert!(matches!(
            program.body.as_slice(),
            [Stmt::Call { name }] if name == "first"
        ));
    }

    #[test]
    fn located_parser_retains_utf8_and_nested_statement_ranges() {
        let source = "PROCEDURE alpha { 1 IF { [ 2 ] EXEC } ELSE { WHILE { 0 } { 3 } } }\n\"λ\" OUTPUT\nalpha\n";
        let source_id = crate::source::SourceId::new(7);
        let located_tokens = crate::lexer::lex(source_id, source).unwrap();
        let tokens = located_tokens
            .iter()
            .map(|token| token.token.clone())
            .collect::<Vec<_>>();
        let spans = located_tokens
            .iter()
            .map(|token| token.span)
            .collect::<Vec<_>>();
        let mut parser = Parser::new_with_source_spans(&tokens, &spans).unwrap();
        let located = parser.parse_located_program().unwrap();

        assert!(located.locations.matches(&located.program));
        let procedure = located.locations.procedures[0].as_ref().unwrap();
        assert_eq!(
            &source[procedure.declaration.range.start..procedure.declaration.range.end],
            "PROCEDURE alpha { 1 IF { [ 2 ] EXEC } ELSE { WHILE { 0 } { 3 } } }"
        );
        assert_eq!(
            &source[procedure.body[0].span.range.start..procedure.body[0].span.range.end],
            "1"
        );
        let conditional = &procedure.body[1];
        assert!(source[conditional.span.range.start..conditional.span.range.end].starts_with("IF"));
        assert_eq!(conditional.child_blocks.len(), 2);
        let nested_loop = &conditional.child_blocks[1][0];
        assert!(
            source[nested_loop.span.range.start..nested_loop.span.range.end].starts_with("WHILE")
        );
        assert_eq!(nested_loop.child_blocks.len(), 2);

        let unicode_output = &located.locations.body[0];
        assert_eq!(
            &source[unicode_output.span.range.start..unicode_output.span.range.end],
            "\"λ\" OUTPUT"
        );
        assert_eq!(unicode_output.span.range.len(), "\"λ\" OUTPUT".len());
        assert_eq!(
            &source[located.locations.body[1].span.range.start
                ..located.locations.body[1].span.range.end],
            "alpha"
        );
    }

    #[test]
    fn located_expression_if_retains_nested_branch_locations() {
        let source = "IF (1 > 0) { 2 } ELSE { 3 }";
        let source_id = crate::source::SourceId::new(8);
        let located_tokens = crate::lexer::lex(source_id, source).unwrap();
        let tokens = located_tokens
            .iter()
            .map(|token| token.token.clone())
            .collect::<Vec<_>>();
        let spans = located_tokens
            .iter()
            .map(|token| token.span)
            .collect::<Vec<_>>();
        let mut parser = Parser::new_with_source_spans(&tokens, &spans).unwrap();
        let located = parser.parse_located_program().unwrap();

        assert!(located.locations.matches(&located.program));
        let wrapper = &located.locations.body[0];
        let conditional = wrapper.child_blocks[0].last().unwrap();
        assert_eq!(conditional.child_blocks.len(), 2);
        assert_eq!(
            &source[conditional.child_blocks[0][0].span.range.start
                ..conditional.child_blocks[0][0].span.range.end],
            "2"
        );
        assert_eq!(
            &source[conditional.child_blocks[1][0].span.range.start
                ..conditional.child_blocks[1][0].span.range.end],
            "3"
        );
    }

    #[test]
    fn named_temporal_and_boolean_property_are_retained_and_located() {
        let source = "TEMPORAL alpha @ 2 DEFAULT 17;\nTEMPORAL beta @ 5 DEFAULT 23;\nPROPERTY safe { ALL_FIXED NOT (CELL alpha NE 0 AND CELL beta GT 9) OR CELL 8 EQ 1; }\nalpha POP\n";
        let source_id = crate::source::SourceId::new(9);
        let located_tokens = crate::lexer::lex(source_id, source).unwrap();
        let tokens = located_tokens
            .iter()
            .map(|token| token.token.clone())
            .collect::<Vec<_>>();
        let spans = located_tokens
            .iter()
            .map(|token| token.span)
            .collect::<Vec<_>>();
        let mut parser = Parser::new_with_source_spans(&tokens, &spans).unwrap();
        let located = parser.parse_located_program().unwrap();

        assert_eq!(located.program.temporal_declarations[0].default, 17);
        assert_eq!(located.program.temporal_declarations[1].default, 23);
        assert_eq!(
            located.program.temporal_properties[0].touched_addresses(),
            vec![2, 5, 8]
        );
        assert_eq!(located.locations.temporals.len(), 2);
        assert!(located.locations.temporals.iter().all(Option::is_some));
        assert!(matches!(
            located.program.body[0],
            Stmt::ReadTemporal { address: 2, .. }
        ));
    }

    #[test]
    fn temporal_declarations_reject_duplicate_names_addresses_and_missing_terminators() {
        assert!(parse("TEMPORAL a @ 1 DEFAULT 0; TEMPORAL A @ 2 DEFAULT 0;")
            .unwrap_err()
            .contains("Duplicate TEMPORAL name"));
        assert!(parse("TEMPORAL a @ 1 DEFAULT 0; TEMPORAL b @ 1 DEFAULT 0;")
            .unwrap_err()
            .contains("reuses address 1"));
        assert!(parse("TEMPORAL a @ 1 DEFAULT 0")
            .unwrap_err()
            .contains("TEMPORAL declaration"));
    }

    #[test]
    fn property_predicate_nesting_is_bounded() {
        let source = format!(
            "PROPERTY too_deep {{ ALL_FIXED {}CELL 0 EQ 0; }}",
            "NOT ".repeat(33)
        );
        assert!(parse(&source).unwrap_err().contains("nesting exceeds 32"));
    }
}
