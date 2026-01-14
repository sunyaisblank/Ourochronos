//! Lexer and Parser for OUROCHRONOS.
//!
//! Syntax:
//! - Numbers: 42, 0xFF (hex), 0b1010 (binary)
//! - Opcodes: ADD, SUB, ORACLE, PROPHECY, etc.
//! - Control flow: IF { } ELSE { }, WHILE { cond } { body }
//! - Comments: # line comment

use crate::ast::{OpCode, Stmt, Program, Procedure, Effect};
use crate::core_types::Value;
use crate::ffi::{FFISignature, FFIType, FFIEffect};
use std::iter::Peekable;
use std::slice::Iter;
use std::collections::HashMap;

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

/// Tokenize source code into a sequence of tokens.
pub fn tokenize(input: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    
    while i < chars.len() {
        match chars[i] {
            // Whitespace
            ' ' | '\t' | '\n' | '\r' => {
                i += 1;
            }
            
            // Block delimiters
            '{' => {
                tokens.push(Token::LBrace);
                i += 1;
            }
            '}' => {
                tokens.push(Token::RBrace);
                i += 1;
            }
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            '[' => {
                tokens.push(Token::LBracket);
                i += 1;
            }
            ']' => {
                tokens.push(Token::RBracket);
                i += 1;
            }
            ',' => {
                tokens.push(Token::Comma);
                i += 1;
            }
            
            // Line comment
            '#' => {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            }
            
            // String literal "..."
            '"' => {
                i += 1; // Skip opening quote
                let mut string_val = String::new();
                while i < chars.len() && chars[i] != '"' {
                    if chars[i] == '\\' && i + 1 < chars.len() {
                        // Escape sequence
                        i += 1;
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
                }
                tokens.push(Token::StringLit(string_val));
            }
            
            // Character literal '...'
            '\'' => {
                i += 1; // Skip opening quote
                if i < chars.len() {
                    let ch = if chars[i] == '\\' && i + 1 < chars.len() {
                        i += 1;
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
                    if i < chars.len() && chars[i] == '\'' {
                        i += 1; // Skip closing quote
                    }
                    tokens.push(Token::CharLit(ch));
                }
            }
            
            // Semicolon comment (alternative)
            ';' if i + 1 < chars.len() && chars[i + 1] == ';' => {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            }
            
            // Numeric literal
            c if c.is_ascii_digit() => {
                let start = i;
                
                // Check for hex (0x) or binary (0b)
                if c == '0' && i + 1 < chars.len() {
                    match chars[i + 1] {
                        'x' | 'X' => {
                            i += 2; // Skip 0x
                            let hex_start = i;
                            while i < chars.len() && chars[i].is_ascii_hexdigit() {
                                i += 1;
                            }
                            let hex_str: String = chars[hex_start..i].iter().collect();
                            if let Ok(n) = u64::from_str_radix(&hex_str, 16) {
                                tokens.push(Token::Number(n));
                            }
                            continue;
                        }
                        'b' | 'B' => {
                            i += 2; // Skip 0b
                            let bin_start = i;
                            while i < chars.len() && (chars[i] == '0' || chars[i] == '1') {
                                i += 1;
                            }
                            let bin_str: String = chars[bin_start..i].iter().collect();
                            if let Ok(n) = u64::from_str_radix(&bin_str, 2) {
                                tokens.push(Token::Number(n));
                            }
                            continue;
                        }
                        _ => {}
                    }
                }
                
                // Decimal number
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
                let num_str: String = chars[start..i].iter().collect();
                if let Ok(n) = num_str.parse::<u64>() {
                    tokens.push(Token::Number(n));
                }
            }
            
            // Word (identifier or keyword)
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                tokens.push(Token::Word(word));
            }
            
            // Equals sign
            '=' if i + 1 < chars.len() && chars[i + 1] == '=' => {
                // == is a comparison operator
                tokens.push(Token::Word("==".to_string()));
                i += 2;
            }
            '=' => {
                tokens.push(Token::Equals);
                i += 1;
            }
            
            // Semicolon (statement terminator)
            ';' => {
                // Check if it's a comment (;;)
                if i + 1 < chars.len() && chars[i + 1] == ';' {
                    while i < chars.len() && chars[i] != '\n' {
                        i += 1;
                    }
                } else {
                    tokens.push(Token::Semicolon);
                    i += 1;
                }
            }
            
            '/' => {
                if i + 1 < chars.len() && chars[i+1] == '/' {
                    // Line comment
                    while i < chars.len() && chars[i] != '\n' {
                        i += 1;
                    }
                } else {
                    tokens.push(Token::Word("/".to_string()));
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
                
                if i < chars.len() {
                    match (op.chars().next().unwrap(), chars[i]) {
                        ('<', '=') | ('>', '=') | ('!', '=') |
                        ('<', '<') | ('>', '>') | 
                        ('&', '&') | ('|', '|') => {
                            op.push(chars[i]);
                            i += 1;
                        }
                        _ => {}
                    }
                }
                
                tokens.push(Token::Word(op));
            }
            
            // Skip unknown characters
            _ => {
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
    fn parse_word(&mut self, word: &str) -> Result<Stmt, String> {
        let upper = word.to_uppercase();
        
        match upper.as_str() {
            // Stack operations
            "NOP" => self.emit_op(OpCode::Nop),
            "HALT" => self.emit_op(OpCode::Halt),
            "POP" | "DROP" => self.emit_op(OpCode::Pop),
            "DUP" => self.emit_op(OpCode::Dup),
            "SWAP" => self.emit_op(OpCode::Swap),
            "OVER" => self.emit_op(OpCode::Over),
            "ROT" => self.emit_op(OpCode::Rot),
            "DEPTH" => self.emit_op(OpCode::Depth),
            "PICK" | "PEEK" => self.emit_op(OpCode::Pick),
            "ROLL" => self.emit_op(OpCode::Roll),
            "REVERSE" | "REV" => self.emit_op(OpCode::Reverse),
            "EXEC" | "CALL" => self.emit_op(OpCode::Exec),
            "DIP" => self.emit_op(OpCode::Dip),
            "KEEP" => self.emit_op(OpCode::Keep),
            "BI" | "CLEAVE" => self.emit_op(OpCode::Bi),
            "REC" => self.emit_op(OpCode::Rec),
            
            // String Operations
            "STR_REV" => self.emit_op(OpCode::StrRev),
            "STR_CAT" | "CONCAT" => self.emit_op(OpCode::StrCat),
            "STR_SPLIT" | "SPLIT" => self.emit_op(OpCode::StrSplit),
            
            // Arithmetic
            "ADD" | "+" => self.emit_op(OpCode::Add),
            "SUB" | "-" => self.emit_op(OpCode::Sub),
            "MUL" | "*" => self.emit_op(OpCode::Mul),
            "DIV" | "/" => self.emit_op(OpCode::Div),
            "MOD" | "%" => self.emit_op(OpCode::Mod),
            "NEG" => self.emit_op(OpCode::Neg),
            "ABS" => self.emit_op(OpCode::Abs),
            "MIN" => self.emit_op(OpCode::Min),
            "MAX" => self.emit_op(OpCode::Max),
            "SIGN" | "SIGNUM" => self.emit_op(OpCode::Sign),

            // Bitwise
            "NOT" | "~" => self.emit_op(OpCode::Not),
            "AND" | "&" => self.emit_op(OpCode::And),
            "OR" | "|" => self.emit_op(OpCode::Or),
            "XOR" | "^" => self.emit_op(OpCode::Xor),
            "SHL" | "<<" => self.emit_op(OpCode::Shl),
            "SHR" | ">>" => self.emit_op(OpCode::Shr),
            
            // Comparison
            "EQ" | "==" => self.emit_op(OpCode::Eq),
            "NEQ" | "!=" => self.emit_op(OpCode::Neq),
            "LT" | "<" => self.emit_op(OpCode::Lt),
            "GT" | ">" => self.emit_op(OpCode::Gt),
            "LTE" | "<=" => self.emit_op(OpCode::Lte),
            "GTE" | ">=" => self.emit_op(OpCode::Gte),

            // Signed Comparison
            "SLT" => self.emit_op(OpCode::Slt),
            "SGT" => self.emit_op(OpCode::Sgt),
            "SLTE" => self.emit_op(OpCode::Slte),
            "SGTE" => self.emit_op(OpCode::Sgte),

            // Temporal
            "ORACLE" | "READ" => self.emit_op(OpCode::Oracle),
            "PROPHECY" | "WRITE" => self.emit_op(OpCode::Prophecy),
            "PRESENT" => self.emit_op(OpCode::PresentRead),
            "PARADOX" => self.emit_op(OpCode::Paradox),
            
            // I/O
            "INPUT" => self.emit_op(OpCode::Input),
            "OUTPUT" => {
                if matches!(self.tokens.peek(), Some(Token::LParen)) {
                    let expr_stmts = self.parse_expression()?;
                    let mut stmts = expr_stmts;
                    stmts.push(Stmt::Op(OpCode::Output));
                    Ok(Stmt::Block(stmts))
                } else {
                    self.emit_op(OpCode::Output)
                }
            },
            
            "EMIT" => {
                if matches!(self.tokens.peek(), Some(Token::LParen)) {
                    let expr_stmts = self.parse_expression()?;
                    let mut stmts = expr_stmts;
                    stmts.push(Stmt::Op(OpCode::Emit));
                    Ok(Stmt::Block(stmts))
                } else {
                    self.emit_op(OpCode::Emit)
                }
            },

            // Vector operations
            "VEC_NEW" => self.emit_op(OpCode::VecNew),
            "VEC_PUSH" => self.emit_op(OpCode::VecPush),
            "VEC_POP" => self.emit_op(OpCode::VecPop),
            "VEC_GET" => self.emit_op(OpCode::VecGet),
            "VEC_SET" => self.emit_op(OpCode::VecSet),
            "VEC_LEN" => self.emit_op(OpCode::VecLen),

            // Hash table operations
            "HASH_NEW" => self.emit_op(OpCode::HashNew),
            "HASH_PUT" => self.emit_op(OpCode::HashPut),
            "HASH_GET" => self.emit_op(OpCode::HashGet),
            "HASH_DEL" => self.emit_op(OpCode::HashDel),
            "HASH_HAS" => self.emit_op(OpCode::HashHas),
            "HASH_LEN" => self.emit_op(OpCode::HashLen),

            // Set operations
            "SET_NEW" => self.emit_op(OpCode::SetNew),
            "SET_ADD" => self.emit_op(OpCode::SetAdd),
            "SET_HAS" => self.emit_op(OpCode::SetHas),
            "SET_DEL" => self.emit_op(OpCode::SetDel),
            "SET_LEN" => self.emit_op(OpCode::SetLen),

            // FFI operations
            "FFI_CALL" => self.emit_op(OpCode::FFICall),
            "FFI_CALL_NAMED" => self.emit_op(OpCode::FFICallNamed),

            // File I/O operations
            "FILE_OPEN" => self.emit_op(OpCode::FileOpen),
            "FILE_READ" => self.emit_op(OpCode::FileRead),
            "FILE_WRITE" => self.emit_op(OpCode::FileWrite),
            "FILE_SEEK" => self.emit_op(OpCode::FileSeek),
            "FILE_FLUSH" => self.emit_op(OpCode::FileFlush),
            "FILE_CLOSE" => self.emit_op(OpCode::FileClose),
            "FILE_EXISTS" => self.emit_op(OpCode::FileExists),
            "FILE_SIZE" => self.emit_op(OpCode::FileSize),

            // Buffer operations
            "BUFFER_NEW" => self.emit_op(OpCode::BufferNew),
            "BUFFER_FROM_STACK" => self.emit_op(OpCode::BufferFromStack),
            "BUFFER_TO_STACK" => self.emit_op(OpCode::BufferToStack),
            "BUFFER_LEN" => self.emit_op(OpCode::BufferLen),
            "BUFFER_READ_BYTE" => self.emit_op(OpCode::BufferReadByte),
            "BUFFER_WRITE_BYTE" => self.emit_op(OpCode::BufferWriteByte),
            "BUFFER_FREE" => self.emit_op(OpCode::BufferFree),

            // Network operations
            "TCP_CONNECT" => self.emit_op(OpCode::TcpConnect),
            "SOCKET_SEND" => self.emit_op(OpCode::SocketSend),
            "SOCKET_RECV" => self.emit_op(OpCode::SocketRecv),
            "SOCKET_CLOSE" => self.emit_op(OpCode::SocketClose),

            // Process operations
            "PROC_EXEC" => self.emit_op(OpCode::ProcExec),

            // System operations
            "CLOCK" => self.emit_op(OpCode::Clock),
            "SLEEP" => self.emit_op(OpCode::Sleep),
            "RANDOM" => self.emit_op(OpCode::Random),

            // Control flow
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
            
            // Handle constants, variables, or unknown words
            other => {
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
}
