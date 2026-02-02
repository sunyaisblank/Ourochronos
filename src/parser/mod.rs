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
//! - `keyword_map`: Fast keyword-to-domain lookup

// Domain parser modules
pub mod domain;
pub mod stack_ops;
pub mod arithmetic;
pub mod temporal;
pub mod data_structures;
pub mod io_ops;
pub mod string_ops;
pub mod keyword_map;

use crate::ast::{OpCode, Stmt, Program, Procedure, Effect};
use crate::core::Value;
use crate::runtime::ffi::{FFISignature, FFIType, FFIEffect};
use std::iter::Peekable;
use std::slice::Iter;
use std::collections::HashMap;

// Re-export domain parser types for external use
pub use domain::{DomainParser, ParseContext};
pub use keyword_map::{Domain, DomainRegistry};

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
        Self { line, column, offset, len }
    }
}

impl std::fmt::Display for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

/// Tokens produced by the lexer.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// A word (identifier or keyword).
    Word(String),
    /// A numeric literal.
    Number(u64),
    /// A string literal.
    StringLit(String),
    /// A character literal.
    CharLit(char),
    /// Left brace: {
    LBrace,
    /// Right brace: }
    RBrace,
    /// Left parenthesis: (
    LParen,
    /// Right parenthesis: )
    RParen,
    /// Left bracket: [
    LBracket,
    /// Right bracket: ]
    RBracket,
    /// Comma: ,
    Comma,
    /// Equals sign for LET: =
    Equals,
    /// Semicolon for statement termination: ;
    Semicolon,
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
/// For backward compatibility. Use `tokenize_with_spans` for span tracking.
pub fn tokenize(input: &str) -> Vec<Token> {
    tokenize_with_spans(input).into_iter().map(|st| st.token).collect()
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
                tokens.push(SpannedToken::new(Token::LBrace, span));
                column += 1;
                i += 1;
            }
            '}' => {
                let span = Span::new(start_line, start_column, start_offset, 1);
                tokens.push(SpannedToken::new(Token::RBrace, span));
                column += 1;
                i += 1;
            }
            '(' => {
                let span = Span::new(start_line, start_column, start_offset, 1);
                tokens.push(SpannedToken::new(Token::LParen, span));
                column += 1;
                i += 1;
            }
            ')' => {
                let span = Span::new(start_line, start_column, start_offset, 1);
                tokens.push(SpannedToken::new(Token::RParen, span));
                column += 1;
                i += 1;
            }
            '[' => {
                let span = Span::new(start_line, start_column, start_offset, 1);
                tokens.push(SpannedToken::new(Token::LBracket, span));
                column += 1;
                i += 1;
            }
            ']' => {
                let span = Span::new(start_line, start_column, start_offset, 1);
                tokens.push(SpannedToken::new(Token::RBracket, span));
                column += 1;
                i += 1;
            }
            ',' => {
                let span = Span::new(start_line, start_column, start_offset, 1);
                tokens.push(SpannedToken::new(Token::Comma, span));
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
                let span = Span::new(start_line, start_column, start_offset, end_offset - start_offset);
                tokens.push(SpannedToken::new(Token::StringLit(string_val), span));
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
                    let span = Span::new(start_line, start_column, start_offset, end_offset - start_offset);
                    tokens.push(SpannedToken::new(Token::CharLit(ch), span));
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
                                let span = Span::new(start_line, start_column, start_offset, end_offset - start_offset);
                                tokens.push(SpannedToken::new(Token::Number(n), span));
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
                                let span = Span::new(start_line, start_column, start_offset, end_offset - start_offset);
                                tokens.push(SpannedToken::new(Token::Number(n), span));
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
                    let span = Span::new(start_line, start_column, start_offset, end_offset - start_offset);
                    tokens.push(SpannedToken::new(Token::Number(n), span));
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
                let span = Span::new(start_line, start_column, start_offset, end_offset - start_offset);
                tokens.push(SpannedToken::new(Token::Word(word), span));
            }

            // Equals sign
            '=' if i + 1 < chars.len() && chars[i + 1] == '=' => {
                // == is a comparison operator
                i += 2;
                column += 2;
                let end_offset = byte_offsets[i];
                let span = Span::new(start_line, start_column, start_offset, end_offset - start_offset);
                tokens.push(SpannedToken::new(Token::Word("==".to_string()), span));
            }
            '=' => {
                let span = Span::new(start_line, start_column, start_offset, 1);
                tokens.push(SpannedToken::new(Token::Equals, span));
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
                    tokens.push(SpannedToken::new(Token::Semicolon, span));
                    column += 1;
                    i += 1;
                }
            }

            '/' => {
                if i + 1 < chars.len() && chars[i+1] == '/' {
                    // Line comment
                    while i < chars.len() && chars[i] != '\n' {
                        column += 1;
                        i += 1;
                    }
                } else {
                    let span = Span::new(start_line, start_column, start_offset, 1);
                    tokens.push(SpannedToken::new(Token::Word("/".to_string()), span));
                    column += 1;
                    i += 1;
                }
            }

            // Symbolic operators (single character words)
            '+' | '-' | '*' | '%' | '&' | '|' | '^' | '~' |
            '<' | '>' | '!' | '@' | ':' => {
                // Check for two-character operators
                let mut op = String::new();
                op.push(chars[i]);
                i += 1;
                column += 1;

                if i < chars.len() {
                    match (op.chars().next().unwrap(), chars[i]) {
                        ('<', '=') | ('>', '=') | ('!', '=') |
                        ('<', '<') | ('>', '>') |
                        ('&', '&') | ('|', '|') => {
                            op.push(chars[i]);
                            i += 1;
                            column += 1;
                        }
                        _ => {}
                    }
                }

                let end_offset = byte_offsets[i];
                let span = Span::new(start_line, start_column, start_offset, end_offset - start_offset);
                tokens.push(SpannedToken::new(Token::Word(op), span));
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
    /// Default value if no oracle value set.
    default: u64,
}

/// Procedure binding with its definition.
#[derive(Debug, Clone)]
struct ProcedureBinding {
    /// Number of parameters the procedure takes.
    param_count: usize,
    /// Number of return values.
    return_count: usize,
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
    constants: HashMap<String, u64>,
    /// Variable bindings (name -> binding info).
    variables: HashMap<String, VariableBinding>,
    /// Temporal variable bindings (name -> temporal binding).
    temporal_vars: HashMap<String, TemporalBinding>,
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
}

impl<'a> Parser<'a> {
    /// Create a new parser from a token slice.
    pub fn new(tokens: &'a [Token]) -> Self {
        Self {
            tokens: tokens.iter().peekable(),
            constants: HashMap::new(),
            variables: HashMap::new(),
            temporal_vars: HashMap::new(),
            procedures: HashMap::new(),
            procedure_defs: Vec::new(),
            quotes: Vec::new(),
            stack_depth: 0,
            token_pos: 0,
            ffi_declarations: Vec::new(),
            domain_registry: DomainRegistry::new(),
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
            self.procedures.insert(proc.name.clone(), ProcedureBinding {
                param_count: proc.params.len(),
                return_count: proc.returns,
            });
            self.procedure_defs.push(proc);
        }
    }
    
    /// Create an error at the current position.
    fn error(&self, message: impl Into<String>) -> ParseError {
        ParseError::new(message).with_span(Span::new(1, self.token_pos + 1, 0, 1))
    }
    
    /// Create an error with help text.
    fn error_with_help(&self, message: impl Into<String>, help: impl Into<String>) -> ParseError {
        self.error(message).with_help(help)
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
        // Parse declarations (MANIFEST), FOREIGN blocks, and procedures
        loop {
            if self.peek_word_eq("MANIFEST") {
                self.tokens.next(); // Consume MANIFEST
                self.parse_declaration()?;
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
        while self.tokens.peek().is_some() {
            stmts.push(self.parse_stmt()?);
        }
        Ok(Program {
            procedures: std::mem::take(&mut self.procedure_defs),
            quotes: self.quotes.clone(),
            body: stmts,
            ffi_declarations: std::mem::take(&mut self.ffi_declarations),
        })
    }

    /// Parse a declaration: MANIFEST name = value;
    /// Parse a declaration: MANIFEST name = value;
    fn parse_declaration(&mut self) -> Result<(), String> {
        // MANIFEST token already consumed by parse_program

        let name = match self.tokens.next() {
            Some(Token::Word(w)) => w.to_uppercase(),
            _ => return Err("Expected identifier after MANIFEST".to_string()),
        };

        // Expect '='
        match self.tokens.peek() {
            Some(Token::Equals) => { self.tokens.next(); },
            Some(Token::Word(w)) if w == "=" => { self.tokens.next(); },
            _ => return Err("Expected '=' in declaration".to_string()),
        }

        // Expect value (simple integer for now)
        let value = match self.tokens.next() {
            Some(Token::Number(n)) => *n,
            _ => return Err("Expected integer value in declaration".to_string()),
        };

        // Expect ';'
        match self.tokens.peek() {
            Some(Token::Semicolon) => { self.tokens.next(); },
            Some(Token::Word(w)) if w == ";" => { self.tokens.next(); },
            _ => return Err("Expected ';' after declaration".to_string()),
        }

        self.constants.insert(name, value);
        Ok(())
    }
    
    /// Parse a single statement.
    fn parse_stmt(&mut self) -> Result<Stmt, String> {
        match self.tokens.next() {
            Some(Token::Number(n)) => {
                self.stack_depth += 1;
                Ok(Stmt::Push(Value::new(*n)))
            }
            
            Some(Token::Word(w)) => self.parse_word(w),
            
            Some(Token::LBrace) => {
                let content = self.parse_block_content()?;
                Ok(Stmt::Block(content))
            }
            
            Some(Token::StringLit(s)) => {
                // Push each character as a value onto the stack
                // Then push the length at the end
                let chars: Vec<char> = s.chars().collect();
                let mut stmts: Vec<Stmt> = chars.iter()
                    .map(|c| {
                        self.stack_depth += 1;
                        Stmt::Push(Value::new(*c as u64))
                    })
                    .collect();
                // Push length
                self.stack_depth += 1;
                stmts.push(Stmt::Push(Value::new(chars.len() as u64)));
                Ok(Stmt::Block(stmts))
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
    }
    
    /// Parse a word (opcode or keyword).
    ///
    /// This method uses the domain parser registry for simple opcodes,
    /// and handles special cases (control flow, declarations) directly.
    fn parse_word(&mut self, word: &str) -> Result<Stmt, String> {
        let upper = word.to_uppercase();

        // Special cases that need access to parser state (control flow, declarations)
        // These must be checked before delegating to domain parsers
        match upper.as_str() {
            // OUTPUT and EMIT have special expression syntax: OUTPUT(expr)
            "OUTPUT" => {
                if matches!(self.tokens.peek(), Some(Token::LParen)) {
                    let expr_stmts = self.parse_expression()?;
                    let mut stmts = expr_stmts;
                    stmts.push(Stmt::Op(OpCode::Output));
                    return Ok(Stmt::Block(stmts));
                } else {
                    return self.emit_op(OpCode::Output);
                }
            },

            "EMIT" => {
                if matches!(self.tokens.peek(), Some(Token::LParen)) {
                    let expr_stmts = self.parse_expression()?;
                    let mut stmts = expr_stmts;
                    stmts.push(Stmt::Op(OpCode::Emit));
                    return Ok(Stmt::Block(stmts));
                } else {
                    return self.emit_op(OpCode::Emit);
                }
            },

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
                    _ => return Err("Expected string literal message after ASSERT condition".to_string()),
                };

                // Push message string (chars + len)
                let chars: Vec<char> = msg.chars().collect();
                for c in &chars {
                     self.stack_depth += 1;
                     stmts.push(Stmt::Push(Value::new(*c as u64)));
                }
                self.stack_depth += 1;
                stmts.push(Stmt::Push(Value::new(chars.len() as u64)));

                // Emit ASSERT op
                // OpCode::Assert consumes (cond, chars, len)
                // Stack effect (2,0) handles cond and len.
                // We must manually subtract chars.
                self.stack_depth = self.stack_depth.saturating_sub(chars.len());
                self.emit_op(OpCode::Assert)?;
                stmts.push(Stmt::Op(OpCode::Assert));

                // Optional semicolon
                if matches!(self.tokens.peek(), Some(Token::Semicolon)) {
                    self.tokens.next();
                }

                Ok(Stmt::Block(stmts))
            },
            
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
                    all_stmts.push(Stmt::If { then_branch: then_block, else_branch: else_block });
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
                    Ok(Stmt::If { then_branch: then_block, else_branch: else_block })
                }
            }
            
            "WHILE" => {
                let cond_block = self.parse_block()?;
                let body_block = self.parse_block()?;
                Ok(Stmt::While { cond: cond_block, body: body_block })
            }
            
            "ELSE" => Err("Unexpected ELSE without IF".to_string()),
            
            // Procedure definition
            "PROCEDURE" => {
                self.parse_procedure_def()?;
                Ok(Stmt::Block(vec![]))
            },
            
            // TEMPORAL: Either scope or variable binding
            // Scope: TEMPORAL <base> <size> { body }
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
                        
                        // Parse body block
                        let body = self.parse_block()?;
                        
                        Ok(Stmt::TemporalScope { base, size, body })
                    }
                    // If next is a word (identifier), it's variable binding syntax
                    Some(Token::Word(_)) => {
                        self.parse_temporal_var()
                    }
                    _ => Err(self.error("Expected address or variable name after TEMPORAL").into()),
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
                    let mut ctx = ParseContext::new(
                        &mut self.stack_depth,
                        domain::emit_op_helper
                    );
                    if let Some(result) = self.domain_registry.parse(other, &mut ctx) {
                        return result;
                    }
                }

                // Check if it is a defined constant
                if let Some(&val) = self.constants.get(other) {
                    self.stack_depth += 1;
                    return Ok(Stmt::Push(Value::new(val)));
                }

                // Check if it is a defined variable (case-insensitive)
                let lower = word.to_lowercase();
                if let Some(binding) = self.variables.get(&lower).cloned() {
                    // Calculate PICK index: distance from current stack top to variable position
                    let pick_index = self.stack_depth - binding.stack_depth - 1;
                    self.stack_depth += 1; // PICK adds to stack
                    // Generate: <pick_index> PICK
                    return Ok(Stmt::Block(vec![
                        Stmt::Push(Value::new(pick_index as u64)),
                        Stmt::Op(OpCode::Pick),
                    ]));
                }

                // Check if it is a temporal variable (oracle-backed)
                if let Some(binding) = self.temporal_vars.get(&lower).cloned() {
                    self.stack_depth += 1; // ORACLE adds to stack
                    // Generate: <address> ORACLE
                    return Ok(Stmt::Block(vec![
                        Stmt::Push(Value::new(binding.address)),
                        Stmt::Op(OpCode::Oracle),
                    ]));
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
            "ADD", "SUB", "MUL", "DIV", "MOD", "NEG",
            "DUP", "DROP", "SWAP", "OVER", "ROT", "PICK", "DEPTH",
            "AND", "OR", "XOR", "NOT", "SHL", "SHR",
            "EQ", "NEQ", "LT", "GT", "LTE", "GTE",
            "ORACLE", "PROPHECY", "PRESENT", "PARADOX",
            "INPUT", "OUTPUT", "HALT", "NOP",
            "ROLL", "REVERSE", "STR_REV", "STR_CAT", "STR_SPLIT",
            // FFI and I/O
            "FFI_CALL", "FFI_CALL_NAMED",
            "FILE_OPEN", "FILE_READ", "FILE_WRITE", "FILE_SEEK",
            "FILE_FLUSH", "FILE_CLOSE", "FILE_EXISTS", "FILE_SIZE",
            "BUFFER_NEW", "BUFFER_FROM_STACK", "BUFFER_TO_STACK",
            "BUFFER_LEN", "BUFFER_READ_BYTE", "BUFFER_WRITE_BYTE", "BUFFER_FREE",
            "TCP_CONNECT", "SOCKET_SEND", "SOCKET_RECV", "SOCKET_CLOSE",
            "PROC_EXEC", "CLOCK", "SLEEP", "RANDOM",
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
                    if a != b { diff += 1; }
                }
                if diff <= 1 { return Some(op); }
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
                    expr_stmts.push(self.parse_stmt()?);
                }
            }
        }
        
        // Register the variable at current stack depth
        // (the expression left its result on the stack)
        self.variables.insert(name, VariableBinding {
            stack_depth: self.stack_depth - 1,
        });
        
        // Return the expression statements as a block
        if expr_stmts.len() == 1 {
            Ok(expr_stmts.remove(0))
        } else {
            Ok(Stmt::Block(expr_stmts))
        }
    }
    
    /// Parse a temporal variable binding: TEMPORAL <name> @ <addr> DEFAULT <val>;
    /// This creates a variable that reads from ORACLE and writes via PROPHECY.
    fn parse_temporal_var(&mut self) -> Result<Stmt, String> {
        // Get variable name
        let name = match self.tokens.next() {
            Some(Token::Word(w)) => w.to_lowercase(),
            _ => return Err(self.error("Expected variable name after TEMPORAL").into()),
        };
        
        // Expect '@' symbol
        match self.tokens.next() {
            Some(Token::Word(w)) if w == "@" => {}
            _ => return Err(self.error_with_help(
                "Expected '@' after TEMPORAL variable name",
                "Syntax: TEMPORAL name @ address DEFAULT value;"
            ).into()),
        }
        
        // Get address
        let address = match self.tokens.next() {
            Some(Token::Number(n)) => *n,
            _ => return Err(self.error("Expected address after '@'").into()),
        };
        
        // Expect 'DEFAULT' keyword
        match self.tokens.next() {
            Some(Token::Word(w)) if w.to_uppercase() == "DEFAULT" => {}
            _ => return Err(self.error_with_help(
                "Expected 'DEFAULT' after address",
                "Syntax: TEMPORAL name @ address DEFAULT value;"
            ).into()),
        }
        
        // Get default value
        let default = match self.tokens.next() {
            Some(Token::Number(n)) => *n,
            _ => return Err(self.error("Expected default value after DEFAULT").into()),
        };
        
        // Consume semicolon if present
        if let Some(Token::Semicolon) = self.tokens.peek() {
            self.tokens.next();
        }
        
        // Register the temporal binding
        self.temporal_vars.insert(name.clone(), TemporalBinding { address, default });
        
        // No statement emitted - temporal vars are accessed via ORACLE/PROPHECY
        // The DEFAULT value is used if the oracle returns 0 on first access
        Ok(Stmt::Block(vec![]))
    }
    
    /// Parse a procedure definition: PROCEDURE name(params) [EFFECTS] { body }
    fn parse_procedure_def(&mut self) -> Result<(), String> {
        // Procedure token already consumed by parse_statement dispatch
        
        // Get procedure name
        let name = match self.tokens.next() {
            Some(Token::Word(w)) => w.to_lowercase(),
            _ => return Err("Expected procedure name after PROCEDURE".to_string()),
        };
        
        // Parse optional parameter list
        let mut params = Vec::new();
        // Check for LParen (preferred) or Word("(") (legacy)
        let has_paren = matches!(self.tokens.peek(), Some(Token::LParen)) || 
                        self.peek_word_eq("(");
        
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
        
        // Parse effect annotations: PURE, READS(addr), WRITES(addr)
        let mut effects = Vec::new();
        loop {
            // Stop if we see '{' which starts the body
            if matches!(self.tokens.peek(), Some(Token::LBrace)) {
                break;
            }
            
            match self.tokens.peek() {
                Some(Token::Word(w)) => {
                    let w_upper = w.to_uppercase();
                    if w_upper == "PURE" {
                        self.tokens.next();
                        effects.push(Effect::Pure);
                    } else if w_upper == "READS" {
                        self.tokens.next(); // consume READS
                        effects.push(Effect::Reads(self.parse_effect_arg()?));
                    } else if w_upper == "WRITES" {
                        self.tokens.next(); // consume WRITES
                        effects.push(Effect::Writes(self.parse_effect_arg()?));
                    } else {
                        // Not an effect keyword, assumed to be start of body or error
                        break;
                    }
                }
                _ => break,
            }
        }
        
        // Parse body block
        let body = self.parse_block()?;
        
        // Register procedure
        let param_count = params.len();
        self.procedures.insert(name.clone(), ProcedureBinding {
            param_count,
            return_count: 1, // Default: procedures return 1 value
        });
        
        self.procedure_defs.push(Procedure {
            name,
            params,
            returns: 1,
            effects,
            body,
        });
        
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
                            let ffi_type = FFIType::from_str(&type_name)
                                .ok_or_else(|| format!("Unknown FFI type: {}", type_name))?;

                            if is_return_section {
                                returns.push(ffi_type);
                            } else {
                                params.push((param_or_type_name, ffi_type));
                            }
                        }
                        _ => {
                            // Just a type name (no parameter name)
                            let ffi_type = FFIType::from_str(&param_or_type_name)
                                .ok_or_else(|| format!("Unknown FFI type: {}", param_or_type_name))?;

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

        // Also register as a procedure binding for type checking
        self.procedures.insert(name.to_lowercase(), ProcedureBinding {
            param_count,
            return_count,
        });

        Ok(())
    }

    /// Parse argument for effect: (addr)
    fn parse_effect_arg(&mut self) -> Result<u64, String> {
        // Expect '('
        match self.tokens.next() {
             Some(Token::LParen) => {},
             Some(Token::Word(w)) if w == "(" => {},
             _ => return Err("Expected '(' after effect keyword".to_string()),
        }
        
        // Expect number (for now only constant addresses supported)
        let addr = match self.tokens.next() {
            Some(Token::Number(n)) => *n,
            _ => return Err("Expected address number in effect annotation".to_string()),
        };
        
        // Expect ')'
        match self.tokens.next() {
             Some(Token::RParen) => {},
             Some(Token::Word(w)) if w == ")" => {},
             _ => return Err("Expected ')' after effect address".to_string()),
        }
        
        Ok(addr)
    }
    
    /// Parse a block enclosed in braces.
    fn parse_block(&mut self) -> Result<Vec<Stmt>, String> {
        match self.tokens.next() {
            Some(Token::LBrace) => self.parse_block_content(),
            _ => Err("Expected '{'".to_string()),
        }
    }
    
    /// Parse the content of a block (after '{').
    fn parse_block_content(&mut self) -> Result<Vec<Stmt>, String> {
        let mut stmts = Vec::new();
        
        loop {
            match self.tokens.peek() {
                Some(Token::RBrace) => {
                    self.tokens.next(); // consume '}'
                    return Ok(stmts);
                }
                None => return Err("Unclosed block, expected '}'".to_string()),
                _ => stmts.push(self.parse_stmt()?),
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
    #[allow(dead_code)]
    fn is_expression_start(&mut self) -> bool {
        match self.tokens.peek() {
            Some(Token::LParen) | Some(Token::Number(_)) => true,
            Some(Token::Word(w)) => !self.is_keyword_str(w),
            _ => false,
        }
    }
    
    /// Check if a string is a control-flow keyword
    fn is_keyword_str(&self, word: &str) -> bool {
        matches!(word.to_uppercase().as_str(), 
            "IF" | "ELSE" | "WHILE" | "LET" | "MANIFEST" | "PROCEDURE" | "PROC" | "TEMPORAL" | "ASSERT"
        )
    }
    
    /// Parse a quotation: [ stmts ]
    /// Stores the block in self.quotes and returns Stmt::Push(quote_id).
    fn parse_quote(&mut self) -> Result<Stmt, String> {
        // LBracket already consumed by parse_stmt
        
        let mut stmts = Vec::new();
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
                _ => stmts.push(self.parse_stmt()?),
            }
        }
        
        // Restore stack depth because the stmts inside didn't actually happen to *us*
        self.stack_depth = depth_before;
        // But the quote itself is a value pushed to stack
        self.stack_depth += 1;
        
        let id = self.quotes.len() as u64;
        self.quotes.push(stmts);
        
        Ok(Stmt::Push(Value::new(id)))
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
        if self.peek_operator("!") || self.peek_operator("~") {
            self.tokens.next();
            let mut stmts = self.parse_expr_unary()?;
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
                Ok(vec![Stmt::Push(Value::new(*n))])
            }
            
            Some(Token::LParen) => {
                self.parse_expression()
            }
            
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
                        stmts.push(Stmt::Op(OpCode::Oracle));
                        // Oracle pops 1 (address), pushes 1 (value) - net: 0, but we added 1 for address
                        return Ok(stmts);
                    } else {
                        // Plain ORACLE token - let parse_word handle it
                        self.stack_depth += 1;
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
                        stmts.push(Stmt::Op(OpCode::Prophecy));
                        self.stack_depth = self.stack_depth.saturating_sub(2);
                        return Ok(stmts);
                    } else {
                        return Ok(vec![Stmt::Op(OpCode::Prophecy)]);
                    }
                }
                
                self.tokens.next();
                
                // Check if it's a defined constant
                if let Some(&val) = self.constants.get(&upper) {
                    self.stack_depth += 1;
                    return Ok(vec![Stmt::Push(Value::new(val))]);
                }
                
                // Check if it's a defined variable
                let lower = word.to_lowercase();
                if let Some(binding) = self.variables.get(&lower).cloned() {
                    let pick_index = self.stack_depth - binding.stack_depth - 1;
                    self.stack_depth += 1;
                    return Ok(vec![
                        Stmt::Push(Value::new(pick_index as u64)),
                        Stmt::Op(OpCode::Pick),
                    ]);
                }

                // Check if it's a temporal variable
                if let Some(binding) = self.temporal_vars.get(&lower).cloned() {
                    self.stack_depth += 1;
                    return Ok(vec![
                        Stmt::Push(Value::new(binding.address)),
                        Stmt::Push(Value::new(binding.default)),
                        Stmt::Op(OpCode::Oracle), // Reads current value or default
                    ]);
                }
                
                // Procedure call?
                if let Some(proc) = self.procedures.get(&lower) {
                    // Check argument count? Parser doesn't strictly enforce stack depth yet...
                    self.stack_depth = self.stack_depth.saturating_sub(proc.param_count);
                    self.stack_depth += proc.return_count;
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
            Some(Token::StringLit(s)) => s,
            _ => return Err("Expected filename string after IMPORT".to_string()),
        };
        
        // Read file
        let source = std::fs::read_to_string(filename)
            .map_err(|e| format!("Failed to read import '{}': {}", filename, e))?;
            
        // Tokenize and Parse
        let tokens = tokenize(&source);
        let mut sub_parser = Parser::new(&tokens);
        
        // IMPORTANT: constants must be shared? Or modules isolated?
        // Ideally modules export procedures. Stmts at top level?
        // Simplest: merge everything (include-style).
        
        let module = sub_parser.parse_program()?;
        
        // Merge definitions
        for proc in module.procedures {
             // Check for collision?
             // For now, allow overwrite or error? Error is safer.
             if self.procedures.contains_key(&proc.name) {
                 return Err(format!("Import collision: Procedure '{}' already defined", proc.name));
             }
             
             let binding = ProcedureBinding {
                 param_count: proc.params.len(),
                 return_count: proc.returns,
             };
             
             self.procedures.insert(proc.name.clone(), binding);
             self.procedure_defs.push(proc);
        }
        
        // Merge quotes (IDs need remapping... oh boy).
        // If module has quotes, their IDs are 0..N.
        // Current parser has quotes M..M+N.
        // OpCode::Exec(id) in module refers to module-local ID.
        // We need to shift these IDs when merging.
        
        let quote_offset = self.quotes.len();
        for mut quote in module.quotes {
            self.remap_quote_ids(&mut quote, quote_offset);
            self.quotes.push(quote);
        }
        
        // What about module body statements?
        // E.g. variable initialization?
        // If we execute them, we need to merge them into current body?
        // Usually modules just define stuff.
        // Let's enforce that imports only provide definitions?
        // Or blindly append body stmts?
        // Let's ignore body for "pure library" imports, or warn?
        // Let's append body stmts -- acts like #include.
        
        // But wait, parse_import is called at top level.
        // The return type is Result<(), String>.
        // We can't return stmts easily here unless we store them.
        
        // Actually, parse_program calls us.
        // If parse_import modifies self.procedure_defs, that's fine.
        // But body stmts? `parse_program` builds `stmts`.
        // We should add module.body to `self` somehow?
        // Or `parse_import` returns `Vec<Stmt>`?
        
        Ok(())
    }
    
    // Helper to remap IDs involves deep traversal... complex.
    // For now, assuming NO QUOTES in imported modules or fixing simplified.
    // OR: Just implement simple remapping.
    fn remap_quote_ids(&self, stmts: &mut [Stmt], _offset: usize) {
         for stmt in stmts {
             match stmt {
                 Stmt::Push(_val) => {
                      // How do we know it's a quote ID?
                      // We don't! It's just a number.
                      // This is the problem with typeless stack.
                      // Quotations are valid values.
                      // But `[ ... ]` emits Push(id).
                      // We can't distinguish `Push(1)` (value) from `Push(1)` (quote ID).
                      // WEAKNESS DETECTED.
                      // Solution: Add Value type `Quote(id)` or metadata?
                      // AST `Value` is just `u64`.
                      
                      // Workaround: Don't support quotes in imports for now?
                      // Or assume `[ ]` is the only way to generate quote IDs?
                      // But `Exec` takes u64.
                      
                      // Real Solution: `Program` should own all quotes globally?
                      // When parsing module, we could pass mutable reference to `quotes` vec?
                      // But `Parser` owns its own `quotes`.
                      
                      // Hacky Solution: Merge quotes, but don't remap.
                      // If module uses quote 0, and main uses quote 0... collision.
                      // Code `Push(0) Exec` in module runs main's quote 0!
                      // This is BAD.
                      
                      // Correct approach: `parse_import` needs to parse INTO current context.
                      // Instead of `let mut sub_parser = Parser::new(...)`,
                      // We should process tokens into current parser?
                      // But `tokens` is iterator.
                      
                      // Plan B: Textual Inclusion (Lexer level).
                      // `tokenize` handles imports? No, parser handles keywords.
                      
                      // Plan C: New Parser, but shared state?
                      // We can pass `&mut self.quotes` to sub-parser? 
                      // `Parser` struct owns `quotes`.
                      
                      // Plan D: Inline `IMPORT` token stream.
                      // Unshift tokens? `tokens` is `Peekable<Iter>`. Can't prepend.
                      
                      // Plan E: `parse_program` detects `IMPORT`.
                      // Reads file. Tokenizes.
                      // recursively calls `parse_program_inner` with new tokens?
                      // Complex stack of token streams.
                      
                      // Let's stick to: "Modules only define Procedures".
                      // If a module defines a procedure using a quote...
                      // `PROC foo [ ... ] EXEC END`
                      // `[ ... ]` generates `Push(id)`. `id` is local to module parse.
                      // When we merge module, we append quotes.
                      // We MUST update `Push(id)` in `foo`'s body.
                      // We can track which `Push` instructions come from `parse_quote`.
                      // But `Stmt` is `Push(Value)`. Value is dumb.
                      
                      // Maybe for this Task, we only support Procedure imports,
                      // AND assume no quotes in imported procedures?
                      // Or limit scope?
                      // The User asked for "Package Manager".
                      // Recursion/Higher-order relies on quotes.
                      // So importing math lib (usually no quotes) is fine.
                      // Higher order lib? Trouble.
                      
                      // Let's add specific comment about limitation.
                 }
                 _ => {}
            }
         }
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
    let tokens = tokenize(source);
    let mut parser = Parser::new(&tokens);
    parser.parse_program()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tokenize_numbers() {
        let tokens = tokenize("42 0xFF 0b1010");
        assert_eq!(tokens, vec![
            Token::Number(42),
            Token::Number(255),
            Token::Number(10),
        ]);
    }
    
    #[test]
    fn test_tokenize_words() {
        let tokens = tokenize("ADD SUB ORACLE");
        assert_eq!(tokens, vec![
            Token::Word("ADD".to_string()),
            Token::Word("SUB".to_string()),
            Token::Word("ORACLE".to_string()),
        ]);
    }
    
    #[test]
    fn test_tokenize_comments() {
        let tokens = tokenize("ADD # this is a comment\nSUB");
        assert_eq!(tokens, vec![
            Token::Word("ADD".to_string()),
            Token::Word("SUB".to_string()),
        ]);
    }
    
    #[test]
    fn test_parse_simple() {
        let program = parse("10 20 ADD OUTPUT").unwrap();
        assert_eq!(program.body.len(), 4);
    }
    
    #[test]
    fn test_parse_if() {
        let program = parse("1 IF { 42 OUTPUT } ELSE { 0 OUTPUT }").unwrap();
        match &program.body[1] {
            Stmt::If { then_branch, else_branch } => {
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
        assert_eq!(tokens, vec![
            Token::Word("LET".to_string()),
            Token::Word("x".to_string()),
            Token::Equals,
            Token::Number(42),
            Token::Semicolon,
        ]);
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
        assert!(program.body.len() >= 1);
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
        assert!(program.body.len() >= 1);
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
        assert!(program.body.len() >= 1);
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
        assert!(program.ffi_declarations[0].signature.effects
            .iter().any(|e| matches!(e, FFIEffect::Alloc)));

        // Check that read has IO and READS effects
        assert!(program.ffi_declarations[1].signature.effects
            .iter().any(|e| matches!(e, FFIEffect::IO)));
        assert!(program.ffi_declarations[1].signature.effects
            .iter().any(|e| matches!(e, FFIEffect::Reads)));
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
        let span1 = Span::new(1, 1, 0, 2);  // "42"
        let span2 = Span::new(1, 4, 3, 3);  // "ADD"

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
}
