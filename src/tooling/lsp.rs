//! Language Server Protocol support for OUROCHRONOS.
//!
//! Provides IDE integration with:
//! - Diagnostics (parse errors, type errors, temporal safety warnings)
//! - Code completion (opcodes, keywords, procedures, variables)
//! - Hover documentation
//! - Go to definition
//! - Document symbols
//! - Semantic tokens (syntax highlighting)
//!
//! Requires the `lsp` feature: `cargo build --features lsp`

use std::collections::HashMap;

use crate::ast::{OpCode, Program, Stmt, Effect as AstEffect};
use crate::parser::{Parser, tokenize, Token};
use crate::types::{type_check, TypeCheckResult};

// ═══════════════════════════════════════════════════════════════════════════════
// Core Types
// ═══════════════════════════════════════════════════════════════════════════════

/// LSP diagnostic severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Info,
    Hint,
}

/// A diagnostic message with source location.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    /// Start line (0-indexed).
    pub line: usize,
    /// Start column (0-indexed).
    pub column: usize,
    /// End line.
    pub end_line: usize,
    /// End column.
    pub end_column: usize,
    /// Diagnostic message.
    pub message: String,
    /// Severity level.
    pub severity: Severity,
    /// Optional diagnostic code.
    pub code: Option<String>,
    /// Related information (for multi-location diagnostics).
    pub related: Vec<DiagnosticRelated>,
}

/// Related diagnostic information.
#[derive(Debug, Clone)]
pub struct DiagnosticRelated {
    pub uri: String,
    pub line: usize,
    pub column: usize,
    pub message: String,
}

/// Completion item kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionKind {
    Keyword,
    Opcode,
    Procedure,
    Variable,
    Snippet,
    Constant,
}

/// A completion suggestion.
#[derive(Debug, Clone)]
pub struct CompletionItem {
    /// Label shown in completion list.
    pub label: String,
    /// Kind of completion.
    pub kind: CompletionKind,
    /// Detail text (type info, signature).
    pub detail: Option<String>,
    /// Documentation markdown.
    pub documentation: Option<String>,
    /// Text to insert (if different from label).
    pub insert_text: Option<String>,
    /// Sort order (lower = higher priority).
    pub sort_text: Option<String>,
    /// Filter text for fuzzy matching.
    pub filter_text: Option<String>,
}

/// Hover information.
#[derive(Debug, Clone)]
pub struct HoverInfo {
    /// Content in markdown format.
    pub contents: String,
    /// Optional range that the hover applies to.
    pub range: Option<(usize, usize, usize, usize)>, // start_line, start_col, end_line, end_col
}

/// A symbol in the document.
#[derive(Debug, Clone)]
pub struct DocumentSymbol {
    /// Symbol name.
    pub name: String,
    /// Symbol kind.
    pub kind: SymbolKind,
    /// Symbol detail (e.g., signature).
    pub detail: Option<String>,
    /// Range in document.
    pub range: (usize, usize, usize, usize),
    /// Selection range (name location).
    pub selection_range: (usize, usize, usize, usize),
    /// Child symbols.
    pub children: Vec<DocumentSymbol>,
}

/// Symbol kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    Function,
    Variable,
    Constant,
    Module,
    Operator,
}

/// Location for go-to-definition.
#[derive(Debug, Clone)]
pub struct Location {
    pub uri: String,
    pub line: usize,
    pub column: usize,
    pub end_line: usize,
    pub end_column: usize,
}

/// Semantic token types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticTokenType {
    Keyword,
    Operator,
    Number,
    String,
    Comment,
    Function,
    Variable,
    Parameter,
    Type,
    Macro,
}

/// A semantic token.
#[derive(Debug, Clone)]
pub struct SemanticToken {
    pub line: usize,
    pub start_char: usize,
    pub length: usize,
    pub token_type: SemanticTokenType,
    pub modifiers: u32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Document Management
// ═══════════════════════════════════════════════════════════════════════════════

/// A tracked document with parsed state.
#[derive(Debug, Clone)]
pub struct Document {
    /// Document URI.
    pub uri: String,
    /// Current document text.
    pub text: String,
    /// Version number for sync.
    pub version: i32,
    /// Parsed program (if successful).
    pub program: Option<Program>,
    /// Type check result (if parsed successfully).
    pub type_result: Option<TypeCheckResult>,
    /// Cached diagnostics.
    pub diagnostics: Vec<Diagnostic>,
    /// Tokens from last parse.
    pub tokens: Vec<Token>,
    /// Line offsets for position calculation.
    pub line_offsets: Vec<usize>,
}

impl Document {
    /// Create a new document.
    pub fn new(uri: String, text: String, version: i32) -> Self {
        let line_offsets = Self::compute_line_offsets(&text);
        Self {
            uri,
            text,
            version,
            program: None,
            type_result: None,
            diagnostics: Vec::new(),
            tokens: Vec::new(),
            line_offsets,
        }
    }

    /// Compute line offsets for position calculation.
    fn compute_line_offsets(text: &str) -> Vec<usize> {
        let mut offsets = vec![0];
        for (i, c) in text.char_indices() {
            if c == '\n' {
                offsets.push(i + 1);
            }
        }
        offsets
    }

    /// Convert byte offset to line/column.
    pub fn offset_to_position(&self, offset: usize) -> (usize, usize) {
        let line = self.line_offsets.partition_point(|&o| o <= offset).saturating_sub(1);
        let line_start = self.line_offsets.get(line).copied().unwrap_or(0);
        let column = offset.saturating_sub(line_start);
        (line, column)
    }

    /// Convert line/column to byte offset.
    pub fn position_to_offset(&self, line: usize, column: usize) -> usize {
        let line_start = self.line_offsets.get(line).copied().unwrap_or(0);
        line_start + column
    }

    /// Get word at position (for completions/hover).
    /// Returns the word under or immediately before the cursor.
    pub fn word_at_position(&self, line: usize, column: usize) -> Option<&str> {
        let offset = self.position_to_offset(line, column);
        let bytes = self.text.as_bytes();

        if bytes.is_empty() {
            return None;
        }

        // Determine start position for word search
        let start_offset = if offset >= bytes.len() {
            // Past end of text - check if last char is a word char
            if is_word_char(bytes[bytes.len() - 1]) {
                bytes.len() - 1
            } else {
                return None; // After whitespace/punctuation at end
            }
        } else if offset > 0 && !is_word_char(bytes[offset]) && is_word_char(bytes[offset - 1]) {
            // Cursor right after a word (e.g., "ADD|" where | is cursor)
            offset - 1
        } else if is_word_char(bytes[offset]) {
            // Cursor on a word char
            offset
        } else {
            // Cursor on whitespace/punctuation with no word char before
            return None;
        };

        // Find word boundaries
        let mut start = start_offset;
        while start > 0 && is_word_char(bytes[start - 1]) {
            start -= 1;
        }

        let mut end = start_offset;
        while end < bytes.len() && is_word_char(bytes[end]) {
            end += 1;
        }

        if start <= end {
            Some(&self.text[start..end])
        } else {
            None
        }
    }

    /// Update document with new text.
    pub fn update(&mut self, text: String, version: i32) {
        self.text = text;
        self.version = version;
        self.line_offsets = Self::compute_line_offsets(&self.text);
        self.program = None;
        self.type_result = None;
        self.diagnostics.clear();
        self.tokens.clear();
    }
}

fn is_word_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

// ═══════════════════════════════════════════════════════════════════════════════
// Language Analyzer
// ═══════════════════════════════════════════════════════════════════════════════

/// OUROCHRONOS language analyzer for LSP.
#[derive(Debug, Default)]
pub struct LanguageAnalyzer {
    /// Open documents by URI.
    documents: HashMap<String, Document>,
    /// Known procedure definitions across all documents.
    procedure_locations: HashMap<String, Location>,
}

impl LanguageAnalyzer {
    /// Create a new analyzer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Open a document.
    pub fn open_document(&mut self, uri: &str, text: &str, version: i32) -> Vec<Diagnostic> {
        let mut doc = Document::new(uri.to_string(), text.to_string(), version);
        self.analyze_document(&mut doc);
        let diagnostics = doc.diagnostics.clone();
        self.documents.insert(uri.to_string(), doc);
        diagnostics
    }

    /// Update a document.
    pub fn update_document(&mut self, uri: &str, text: &str, version: i32) -> Vec<Diagnostic> {
        // Remove document, update it, analyze, and re-insert
        if let Some(mut doc) = self.documents.remove(uri) {
            doc.update(text.to_string(), version);
            self.analyze_document(&mut doc);
            let diagnostics = doc.diagnostics.clone();
            self.documents.insert(uri.to_string(), doc);
            diagnostics
        } else {
            self.open_document(uri, text, version)
        }
    }

    /// Close a document.
    pub fn close_document(&mut self, uri: &str) {
        self.documents.remove(uri);
    }

    /// Get document.
    pub fn get_document(&self, uri: &str) -> Option<&Document> {
        self.documents.get(uri)
    }

    /// Analyze a document and populate diagnostics.
    fn analyze_document(&mut self, doc: &mut Document) {
        doc.diagnostics.clear();

        // Tokenize
        doc.tokens = tokenize(&doc.text);

        // Parse
        let mut parser = Parser::new(&doc.tokens);
        parser.register_procedures(crate::StdLib::procedures());

        match parser.parse_program() {
            Ok(program) => {
                // Check temporal safety
                doc.diagnostics.extend(self.check_temporal_safety(&program, doc));

                // Type check
                let type_result = type_check(&program);

                // Add type errors as diagnostics
                for error in &type_result.errors {
                    let line = error.location.unwrap_or(0);
                    doc.diagnostics.push(Diagnostic {
                        line,
                        column: 0,
                        end_line: line,
                        end_column: 100,
                        message: error.message.clone(),
                        severity: Severity::Error,
                        code: Some("T001".to_string()),
                        related: Vec::new(),
                    });
                }

                // Add type warnings (filter out stdlib warnings by checking statement number)
                let line_count = doc.text.lines().count();
                for warning in &type_result.warnings {
                    // Extract statement number from warning like "at stmt 66"
                    let is_from_stdlib = if let Some(idx) = warning.find("at stmt ") {
                        let rest = &warning[idx + 8..];
                        if let Some(end) = rest.find(|c: char| !c.is_ascii_digit()) {
                            rest[..end].parse::<usize>().map(|n| n > line_count * 3).unwrap_or(false)
                        } else {
                            rest.parse::<usize>().map(|n| n > line_count * 3).unwrap_or(false)
                        }
                    } else {
                        false
                    };

                    // Only add warnings from user code
                    if !is_from_stdlib {
                        doc.diagnostics.push(Diagnostic {
                            line: 0,
                            column: 0,
                            end_line: 0,
                            end_column: 100,
                            message: warning.clone(),
                            severity: Severity::Warning,
                            code: Some("T002".to_string()),
                            related: Vec::new(),
                        });
                    }
                }

                // Register procedures
                for (i, proc) in program.procedures.iter().enumerate() {
                    self.procedure_locations.insert(
                        proc.name.clone(),
                        Location {
                            uri: doc.uri.clone(),
                            line: self.find_procedure_line(doc, &proc.name).unwrap_or(i),
                            column: 0,
                            end_line: self.find_procedure_line(doc, &proc.name).unwrap_or(i),
                            end_column: proc.name.len(),
                        },
                    );
                }

                doc.type_result = Some(type_result);
                doc.program = Some(program);
            }
            Err(e) => {
                // Parse error - try to extract location
                let (line, col) = self.parse_error_location(&e);
                doc.diagnostics.push(Diagnostic {
                    line,
                    column: col,
                    end_line: line,
                    end_column: col + 10,
                    message: format!("Parse error: {}", e),
                    severity: Severity::Error,
                    code: Some("P001".to_string()),
                    related: Vec::new(),
                });
            }
        }
    }

    /// Find the line where a procedure is defined.
    fn find_procedure_line(&self, doc: &Document, name: &str) -> Option<usize> {
        let pattern = format!("PROCEDURE {}", name);
        for (i, line) in doc.text.lines().enumerate() {
            if line.contains(&pattern) {
                return Some(i);
            }
        }
        None
    }

    /// Try to extract location from parse error message.
    fn parse_error_location(&self, error: &str) -> (usize, usize) {
        // Try to parse "at line X" or "line X, column Y" patterns
        if let Some(idx) = error.find("line ") {
            let rest = &error[idx + 5..];
            if let Some(end) = rest.find(|c: char| !c.is_ascii_digit()) {
                if let Ok(line) = rest[..end].parse::<usize>() {
                    return (line.saturating_sub(1), 0);
                }
            }
        }
        (0, 0)
    }

    /// Check for temporal safety issues.
    fn check_temporal_safety(&self, program: &Program, doc: &Document) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        let mut context = CheckContext::new(doc);

        // Collect procedure names from the program itself
        let program_procs: std::collections::HashSet<_> = program.procedures.iter()
            .map(|p| p.name.clone())
            .collect();

        // Check main body
        self.check_stmts_temporal(&program.body, &mut context, &mut diagnostics, &program_procs);

        // Check procedures
        for proc in &program.procedures {
            // Check for effect violations
            let body_effects = self.compute_effects(&proc.body);
            if proc.effects.iter().any(|e| matches!(e, AstEffect::Pure)) && !body_effects.is_pure() {
                if let Some(line) = self.find_procedure_line(doc, &proc.name) {
                    diagnostics.push(Diagnostic {
                        line,
                        column: 0,
                        end_line: line,
                        end_column: proc.name.len() + 10,
                        message: format!(
                            "Procedure '{}' is declared PURE but has effects: {}",
                            proc.name,
                            body_effects.describe()
                        ),
                        severity: Severity::Error,
                        code: Some("E001".to_string()),
                        related: Vec::new(),
                    });
                }
            }

            self.check_stmts_temporal(&proc.body, &mut context, &mut diagnostics, &program_procs);
        }

        diagnostics
    }

    fn check_stmts_temporal(
        &self,
        stmts: &[Stmt],
        ctx: &mut CheckContext,
        diagnostics: &mut Vec<Diagnostic>,
        program_procs: &std::collections::HashSet<String>,
    ) {
        for stmt in stmts {
            self.check_stmt_temporal(stmt, ctx, diagnostics, program_procs);
            ctx.advance_line();
        }
    }

    fn check_stmt_temporal(
        &self,
        stmt: &Stmt,
        ctx: &mut CheckContext,
        diagnostics: &mut Vec<Diagnostic>,
        program_procs: &std::collections::HashSet<String>,
    ) {
        match stmt {
            Stmt::Op(OpCode::Paradox) => {
                diagnostics.push(Diagnostic {
                    line: ctx.line,
                    column: 0,
                    end_line: ctx.line,
                    end_column: 7,
                    message: "PARADOX will abort execution - is this intentional?".to_string(),
                    severity: Severity::Warning,
                    code: Some("W001".to_string()),
                    related: Vec::new(),
                });
            }
            Stmt::Op(OpCode::Oracle) => {
                ctx.oracle_count += 1;
                if ctx.in_loop {
                    diagnostics.push(Diagnostic {
                        line: ctx.line,
                        column: 0,
                        end_line: ctx.line,
                        end_column: 6,
                        message: "ORACLE inside loop may cause convergence issues".to_string(),
                        severity: Severity::Info,
                        code: Some("I001".to_string()),
                        related: Vec::new(),
                    });
                }
            }
            Stmt::Op(OpCode::Prophecy) => {
                ctx.prophecy_count += 1;
            }
            Stmt::Block(stmts) => {
                self.check_stmts_temporal(stmts, ctx, diagnostics, program_procs);
            }
            Stmt::If { then_branch, else_branch } => {
                self.check_stmts_temporal(then_branch, ctx, diagnostics, program_procs);
                if let Some(eb) = else_branch {
                    self.check_stmts_temporal(eb, ctx, diagnostics, program_procs);
                }
            }
            Stmt::While { cond, body } => {
                let was_in_loop = ctx.in_loop;
                ctx.in_loop = true;
                self.check_stmts_temporal(cond, ctx, diagnostics, program_procs);
                self.check_stmts_temporal(body, ctx, diagnostics, program_procs);
                ctx.in_loop = was_in_loop;
            }
            Stmt::TemporalScope { body, .. } => {
                let prev_oracle = ctx.oracle_count;
                let prev_prophecy = ctx.prophecy_count;
                self.check_stmts_temporal(body, ctx, diagnostics, program_procs);

                // Check for unbalanced temporal operations in scope
                if ctx.oracle_count > prev_oracle && ctx.prophecy_count == prev_prophecy {
                    diagnostics.push(Diagnostic {
                        line: ctx.line,
                        column: 0,
                        end_line: ctx.line,
                        end_column: 15,
                        message: "TEMPORAL scope reads from oracle but never writes - possible causality issue".to_string(),
                        severity: Severity::Hint,
                        code: Some("H001".to_string()),
                        related: Vec::new(),
                    });
                }
            }
            Stmt::Match { cases, default } => {
                for (_, body) in cases {
                    self.check_stmts_temporal(body, ctx, diagnostics, program_procs);
                }
                if let Some(def) = default {
                    self.check_stmts_temporal(def, ctx, diagnostics, program_procs);
                }
            }
            Stmt::Call { name } => {
                // Check if calling undefined procedure
                if !program_procs.contains(name)
                    && !self.procedure_locations.contains_key(name)
                    && !crate::StdLib::procedures().iter().any(|p| p.name == *name)
                {
                    diagnostics.push(Diagnostic {
                        line: ctx.line,
                        column: 0,
                        end_line: ctx.line,
                        end_column: name.len(),
                        message: format!("Unknown procedure: '{}'", name),
                        severity: Severity::Error,
                        code: Some("E002".to_string()),
                        related: Vec::new(),
                    });
                }
            }
            _ => {}
        }
    }

    /// Compute effects of a statement list.
    fn compute_effects(&self, stmts: &[Stmt]) -> EffectInfo {
        let mut info = EffectInfo::default();
        for stmt in stmts {
            info = info.join(&self.compute_stmt_effects(stmt));
        }
        info
    }

    fn compute_stmt_effects(&self, stmt: &Stmt) -> EffectInfo {
        match stmt {
            Stmt::Op(OpCode::Oracle) => EffectInfo { oracle: true, ..Default::default() },
            Stmt::Op(OpCode::Prophecy) => EffectInfo { prophecy: true, ..Default::default() },
            Stmt::Op(OpCode::PresentRead) => EffectInfo { reads: true, ..Default::default() },
            Stmt::Op(OpCode::Store) => EffectInfo { writes: true, ..Default::default() },
            Stmt::Op(OpCode::Input) | Stmt::Op(OpCode::Output) | Stmt::Op(OpCode::Emit) => {
                EffectInfo { io: true, ..Default::default() }
            }
            Stmt::Block(stmts) => self.compute_effects(stmts),
            Stmt::If { then_branch, else_branch } => {
                let mut info = self.compute_effects(then_branch);
                if let Some(eb) = else_branch {
                    info = info.join(&self.compute_effects(eb));
                }
                info
            }
            Stmt::While { cond, body } => {
                self.compute_effects(cond).join(&self.compute_effects(body))
            }
            Stmt::TemporalScope { body, .. } => self.compute_effects(body),
            Stmt::Match { cases, default } => {
                let mut info = EffectInfo::default();
                for (_, body) in cases {
                    info = info.join(&self.compute_effects(body));
                }
                if let Some(def) = default {
                    info = info.join(&self.compute_effects(def));
                }
                info
            }
            _ => EffectInfo::default(),
        }
    }

    /// Get completions at a position.
    pub fn get_completions(&self, uri: &str, line: usize, column: usize) -> Vec<CompletionItem> {
        let mut items = Vec::new();
        let doc = self.documents.get(uri);

        // Context-aware completion prefix
        let prefix = doc.and_then(|d| d.word_at_position(line, column)).unwrap_or("");
        let prefix_upper = prefix.to_uppercase();

        // Keywords with snippets
        let keywords = [
            ("IF", "IF { $1 }", "Conditional branch"),
            ("IF...ELSE", "IF { $1 } ELSE { $2 }", "Conditional with else branch"),
            ("WHILE", "WHILE { $1 } { $2 }", "Loop while condition is true"),
            ("PROCEDURE", "PROCEDURE $1 {\n    $2\n}", "Define a procedure"),
            ("PROCEDURE PURE", "PROCEDURE $1 PURE {\n    $2\n}", "Define a pure procedure"),
            ("LET", "LET $1 = $2;", "Bind a value to a name"),
            ("MODULE", "MODULE $1 {\n    $2\n}", "Define a module"),
            ("IMPORT", "IMPORT $1;", "Import a module"),
            ("EXPORT", "EXPORT $1;", "Export a symbol"),
            ("TEMPORAL", "TEMPORAL $1 $2 {\n    $3\n}", "Temporal isolation scope"),
            ("MATCH", "MATCH {\n    $1 => { $2 }\n    _ => { $3 }\n}", "Pattern matching"),
        ];

        for (kw, snippet, doc) in keywords {
            if prefix.is_empty() || kw.starts_with(&prefix_upper) {
                items.push(CompletionItem {
                    label: kw.to_string(),
                    kind: CompletionKind::Keyword,
                    detail: Some("keyword".to_string()),
                    documentation: Some(doc.to_string()),
                    insert_text: Some(snippet.to_string()),
                    sort_text: Some(format!("0{}", kw)),
                    filter_text: Some(kw.to_string()),
                });
            }
        }

        // All opcodes with documentation
        for (op, stack, desc, effect) in OPCODE_DOCS {
            if prefix.is_empty() || op.starts_with(&prefix_upper) {
                items.push(CompletionItem {
                    label: op.to_string(),
                    kind: CompletionKind::Opcode,
                    detail: Some(format!("Stack: {}", stack)),
                    documentation: Some(format!("{}\n\nEffect: {}", desc, effect)),
                    insert_text: None,
                    sort_text: Some(format!("1{}", op)),
                    filter_text: Some(op.to_string()),
                });
            }
        }

        // Known procedures from all open documents
        for (name, loc) in &self.procedure_locations {
            if prefix.is_empty() || name.to_uppercase().starts_with(&prefix_upper) {
                items.push(CompletionItem {
                    label: name.clone(),
                    kind: CompletionKind::Procedure,
                    detail: Some("procedure".to_string()),
                    documentation: Some(format!("Defined in {}", loc.uri)),
                    insert_text: Some(name.clone()),
                    sort_text: Some(format!("2{}", name)),
                    filter_text: Some(name.clone()),
                });
            }
        }

        // Standard library procedures
        for (name, params, returns, effects, doc) in STDLIB_DOCS {
            if prefix.is_empty() || name.to_uppercase().starts_with(&prefix_upper) {
                items.push(CompletionItem {
                    label: name.to_string(),
                    kind: CompletionKind::Procedure,
                    detail: Some(format!("({}) -> {} [{}]", params, returns, effects)),
                    documentation: Some(doc.to_string()),
                    insert_text: Some(name.to_string()),
                    sort_text: Some(format!("2{}", name)),
                    filter_text: Some(name.to_string()),
                });
            }
        }

        // Common number literals
        if prefix.is_empty() || prefix.chars().all(|c| c.is_ascii_digit()) {
            for n in [0, 1, 2, 10, 100] {
                items.push(CompletionItem {
                    label: n.to_string(),
                    kind: CompletionKind::Constant,
                    detail: Some("literal".to_string()),
                    documentation: None,
                    insert_text: None,
                    sort_text: Some(format!("9{:05}", n)),
                    filter_text: None,
                });
            }
        }

        items
    }

    /// Get hover information at a position.
    pub fn get_hover(&self, uri: &str, line: usize, column: usize) -> Option<HoverInfo> {
        let doc = self.documents.get(uri)?;
        let word = doc.word_at_position(line, column)?;
        let word_upper = word.to_uppercase();

        // Check opcodes
        for (op, stack, desc, effect) in OPCODE_DOCS {
            if *op == word_upper {
                return Some(HoverInfo {
                    contents: format!(
                        "**{}**\n\nStack: `{}`\n\n{}\n\n*Effect*: {}",
                        op, stack, desc, effect
                    ),
                    range: Some((line, column, line, column + word.len())),
                });
            }
        }

        // Check keywords
        let keyword_docs = [
            ("IF", "**IF** `{ then }` [**ELSE** `{ else }`]\n\nConditional branching. Pops condition from stack; executes then-branch if non-zero."),
            ("ELSE", "**ELSE** `{ else }`\n\nOptional else branch for IF statement."),
            ("WHILE", "**WHILE** `{ cond }` `{ body }`\n\nLoop while condition block produces non-zero value."),
            ("PROCEDURE", "**PROCEDURE** `name` [effects] `{ body }`\n\nDefines a reusable procedure. Call by using `name` as a statement.\n\nEffects: PURE, TEMPORAL, IO, ALLOC"),
            ("LET", "**LET** `name = expr;`\n\nBinds the result of expression to a name for later use."),
            ("MODULE", "**MODULE** `name` `{ ... }`\n\nDefines a module namespace."),
            ("IMPORT", "**IMPORT** `module;`\n\nImports symbols from a module."),
            ("EXPORT", "**EXPORT** `symbol;`\n\nExports a symbol from the current module."),
            ("TEMPORAL", "**TEMPORAL** `base size` `{ body }`\n\nCreates an isolated temporal memory region.\n\nOps within are relative to base address. Paradoxes discard changes."),
            ("MATCH", "**MATCH** `{ cases }`\n\nPattern match on top-of-stack value."),
            ("PURE", "**PURE** effect annotation\n\nDeclares that a procedure has no side effects."),
        ];

        for (kw, doc) in keyword_docs {
            if kw == word_upper {
                return Some(HoverInfo {
                    contents: doc.to_string(),
                    range: Some((line, column, line, column + word.len())),
                });
            }
        }

        // Check stdlib
        for (name, params, returns, effects, doc) in STDLIB_DOCS {
            if *name == word_upper || name.to_lowercase() == word.to_lowercase() {
                return Some(HoverInfo {
                    contents: format!(
                        "**{}**\n\nSignature: `({}) -> {}`\n\nEffects: {}\n\n{}",
                        name, params, returns, effects, doc
                    ),
                    range: Some((line, column, line, column + word.len())),
                });
            }
        }

        // Check user-defined procedures
        if let Some(loc) = self.procedure_locations.get(word) {
            return Some(HoverInfo {
                contents: format!(
                    "**{}** (user procedure)\n\nDefined at {}:{}",
                    word, loc.uri, loc.line + 1
                ),
                range: Some((line, column, line, column + word.len())),
            });
        }

        None
    }

    /// Go to definition.
    pub fn get_definition(&self, uri: &str, line: usize, column: usize) -> Option<Location> {
        let doc = self.documents.get(uri)?;
        let word = doc.word_at_position(line, column)?;

        // Check user procedures
        if let Some(loc) = self.procedure_locations.get(word) {
            return Some(loc.clone());
        }

        // Check in same document
        if let Some(program) = &doc.program {
            for proc in &program.procedures {
                if proc.name == word {
                    if let Some(proc_line) = self.find_procedure_line(doc, &proc.name) {
                        return Some(Location {
                            uri: uri.to_string(),
                            line: proc_line,
                            column: 0,
                            end_line: proc_line,
                            end_column: proc.name.len(),
                        });
                    }
                }
            }
        }

        None
    }

    /// Get document symbols.
    pub fn get_document_symbols(&self, uri: &str) -> Vec<DocumentSymbol> {
        let mut symbols = Vec::new();

        let doc = match self.documents.get(uri) {
            Some(d) => d,
            None => return symbols,
        };

        if let Some(program) = &doc.program {
            // Add procedures
            for proc in &program.procedures {
                let line = self.find_procedure_line(doc, &proc.name).unwrap_or(0);
                let end_line = line + self.count_procedure_lines(&proc.body);

                symbols.push(DocumentSymbol {
                    name: proc.name.clone(),
                    kind: SymbolKind::Function,
                    detail: Some(format!(
                        "({}) -> {} [{}]",
                        proc.params.join(", "),
                        proc.returns,
                        proc.effects.iter()
                            .map(|e| format!("{:?}", e))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )),
                    range: (line, 0, end_line, 1),
                    selection_range: (line, 10, line, 10 + proc.name.len()),
                    children: Vec::new(),
                });
            }
        }

        symbols
    }

    fn count_procedure_lines(&self, stmts: &[Stmt]) -> usize {
        // Rough estimate
        stmts.len().max(3)
    }

    /// Get semantic tokens for syntax highlighting.
    pub fn get_semantic_tokens(&self, uri: &str) -> Vec<SemanticToken> {
        let mut tokens_out = Vec::new();

        let doc = match self.documents.get(uri) {
            Some(d) => d,
            None => return tokens_out,
        };

        for (i, line) in doc.text.lines().enumerate() {
            let mut col = 0;
            for word in line.split_whitespace() {
                let word_start = line[col..].find(word).map(|p| col + p).unwrap_or(col);
                let word_upper = word.to_uppercase();

                let token_type = if KEYWORDS.contains(&word_upper.as_str()) {
                    Some(SemanticTokenType::Keyword)
                } else if OPCODE_DOCS.iter().any(|(op, _, _, _)| *op == word_upper) {
                    Some(SemanticTokenType::Operator)
                } else if word.parse::<u64>().is_ok() || word.starts_with("0x") {
                    Some(SemanticTokenType::Number)
                } else if word.starts_with('"') || word.starts_with('\'') {
                    Some(SemanticTokenType::String)
                } else if word.starts_with("//") || word.starts_with('#') {
                    Some(SemanticTokenType::Comment)
                } else if self.procedure_locations.contains_key(word) || self.procedure_locations.contains_key(&word.to_lowercase()) {
                    Some(SemanticTokenType::Function)
                } else {
                    None
                };

                if let Some(tt) = token_type {
                    tokens_out.push(SemanticToken {
                        line: i,
                        start_char: word_start,
                        length: word.len(),
                        token_type: tt,
                        modifiers: 0,
                    });
                }

                col = word_start + word.len();
            }
        }

        tokens_out
    }

    /// Analyze source code directly (convenience method).
    pub fn analyze(&mut self, uri: &str, source: &str) -> Vec<Diagnostic> {
        self.open_document(uri, source, 0)
    }
}

/// Context for temporal checking.
struct CheckContext<'a> {
    #[allow(dead_code)]
    doc: &'a Document,
    line: usize,
    in_loop: bool,
    oracle_count: usize,
    prophecy_count: usize,
}

impl<'a> CheckContext<'a> {
    fn new(doc: &'a Document) -> Self {
        Self {
            doc,
            line: 0,
            in_loop: false,
            oracle_count: 0,
            prophecy_count: 0,
        }
    }

    fn advance_line(&mut self) {
        self.line += 1;
    }
}

/// Effect information for analysis.
#[derive(Debug, Default, Clone)]
struct EffectInfo {
    oracle: bool,
    prophecy: bool,
    reads: bool,
    writes: bool,
    io: bool,
}

impl EffectInfo {
    fn is_pure(&self) -> bool {
        !self.oracle && !self.prophecy && !self.reads && !self.writes && !self.io
    }

    fn join(&self, other: &Self) -> Self {
        Self {
            oracle: self.oracle || other.oracle,
            prophecy: self.prophecy || other.prophecy,
            reads: self.reads || other.reads,
            writes: self.writes || other.writes,
            io: self.io || other.io,
        }
    }

    fn describe(&self) -> String {
        let mut parts = Vec::new();
        if self.oracle { parts.push("oracle"); }
        if self.prophecy { parts.push("prophecy"); }
        if self.reads { parts.push("reads"); }
        if self.writes { parts.push("writes"); }
        if self.io { parts.push("io"); }
        if parts.is_empty() {
            "pure".to_string()
        } else {
            parts.join(", ")
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Documentation Data
// ═══════════════════════════════════════════════════════════════════════════════

const KEYWORDS: &[&str] = &[
    "IF", "ELSE", "WHILE", "PROCEDURE", "LET", "MODULE",
    "IMPORT", "EXPORT", "TEMPORAL", "MATCH", "PURE", "ALLOC", "IO"
];

/// Opcode documentation: (name, stack_effect, description, effect)
const OPCODE_DOCS: &[(&str, &str, &str, &str)] = &[
    // Stack manipulation
    ("NOP", "( -- )", "No operation", "pure"),
    ("HALT", "( -- )", "Halt execution of the current epoch", "pure"),
    ("POP", "( a -- )", "Pop and discard top of stack", "pure"),
    ("DUP", "( a -- a a )", "Duplicate top of stack", "pure"),
    ("SWAP", "( a b -- b a )", "Swap top two elements", "pure"),
    ("OVER", "( a b -- a b a )", "Copy second element to top", "pure"),
    ("ROT", "( a b c -- b c a )", "Rotate top three elements", "pure"),
    ("DEPTH", "( -- n )", "Push current stack depth", "pure"),
    ("PICK", "( n -- v )", "Copy nth element to top (0=DUP)", "pure"),
    ("ROLL", "( n -- )", "Roll nth element to top", "pure"),
    ("REVERSE", "( n -- )", "Reverse top n elements", "pure"),
    ("EXEC", "( quote -- ... )", "Execute a quotation", "varies"),
    ("DIP", "( x quote -- x )", "Execute quote with x hidden", "varies"),
    ("KEEP", "( x quote -- ... x )", "Keep x, execute quote, restore x", "varies"),
    ("BI", "( x p q -- ... )", "Apply p then q to x", "varies"),
    ("REC", "( quote -- ... )", "Recursive combinator", "varies"),

    // Arithmetic
    ("ADD", "( a b -- a+b )", "Add two values", "pure"),
    ("SUB", "( a b -- a-b )", "Subtract b from a", "pure"),
    ("MUL", "( a b -- a*b )", "Multiply two values", "pure"),
    ("DIV", "( a b -- a/b )", "Integer division (0 if b=0)", "pure"),
    ("MOD", "( a b -- a%b )", "Modulo (0 if b=0)", "pure"),
    ("NEG", "( a -- -a )", "Two's complement negation", "pure"),
    ("ABS", "( a -- |a| )", "Absolute value (signed)", "pure"),
    ("MIN", "( a b -- min )", "Minimum of two values", "pure"),
    ("MAX", "( a b -- max )", "Maximum of two values", "pure"),
    ("SIGN", "( a -- sign )", "Sign: -1, 0, or 1", "pure"),
    ("ASSERT", "( cond chars len -- )", "Assert condition, panic with message if false", "pure"),

    // Bitwise
    ("NOT", "( a -- ~a )", "Bitwise NOT", "pure"),
    ("AND", "( a b -- a&b )", "Bitwise AND", "pure"),
    ("OR", "( a b -- a|b )", "Bitwise OR", "pure"),
    ("XOR", "( a b -- a^b )", "Bitwise XOR", "pure"),
    ("SHL", "( a n -- a<<n )", "Left shift", "pure"),
    ("SHR", "( a n -- a>>n )", "Right shift (logical)", "pure"),

    // Comparison
    ("EQ", "( a b -- a==b )", "Equal (1 if true, 0 if false)", "pure"),
    ("NEQ", "( a b -- a!=b )", "Not equal", "pure"),
    ("LT", "( a b -- a<b )", "Less than (unsigned)", "pure"),
    ("GT", "( a b -- a>b )", "Greater than (unsigned)", "pure"),
    ("LTE", "( a b -- a<=b )", "Less than or equal (unsigned)", "pure"),
    ("GTE", "( a b -- a>=b )", "Greater than or equal (unsigned)", "pure"),
    ("SLT", "( a b -- a<b )", "Less than (signed)", "pure"),
    ("SGT", "( a b -- a>b )", "Greater than (signed)", "pure"),
    ("SLTE", "( a b -- a<=b )", "Less than or equal (signed)", "pure"),
    ("SGTE", "( a b -- a>=b )", "Greater than or equal (signed)", "pure"),

    // Temporal operations
    ("ORACLE", "( addr -- value )", "Read from anamnesis (the future). Returns value written by PROPHECY in a consistent timeline.", "temporal"),
    ("PROPHECY", "( value addr -- )", "Write to present memory (fulfilling the future). Value becomes readable by ORACLE in past epochs.", "temporal"),
    ("PRESENT", "( addr -- value )", "Read from current present memory", "reads"),
    ("PARADOX", "( -- )", "Signal explicit temporal paradox. Aborts current epoch.", "temporal"),

    // Memory/Array
    ("PACK", "( v1..vn base n -- )", "Pack n values into memory at base", "writes"),
    ("UNPACK", "( base n -- v1..vn )", "Unpack n values from memory at base", "reads"),
    ("INDEX", "( base index -- value )", "Read from base+index", "reads"),
    ("STORE", "( value base index -- )", "Write value to base+index", "writes"),

    // I/O
    ("INPUT", "( -- value )", "Read a value from input", "io"),
    ("OUTPUT", "( value -- )", "Output a value", "io"),
    ("EMIT", "( char -- )", "Output a character", "io"),

    // Strings
    ("STR_REV", "( chars len -- reversed len )", "Reverse a string", "pure"),
    ("STR_CAT", "( c1 len1 c2 len2 -- c1c2 len )", "Concatenate strings", "pure"),
    ("STR_SPLIT", "( chars len delim -- parts count )", "Split string by delimiter", "pure"),

    // Vectors
    ("VEC_NEW", "( -- vec )", "Create empty vector", "alloc"),
    ("VEC_PUSH", "( vec value -- vec )", "Push to vector end", "alloc"),
    ("VEC_POP", "( vec -- vec value )", "Pop from vector end", "alloc"),
    ("VEC_GET", "( vec index -- vec value )", "Get value at index", "pure"),
    ("VEC_SET", "( vec value index -- vec )", "Set value at index", "alloc"),
    ("VEC_LEN", "( vec -- vec len )", "Get vector length", "pure"),

    // Hash tables
    ("HASH_NEW", "( -- hash )", "Create empty hash table", "alloc"),
    ("HASH_PUT", "( hash key value -- hash )", "Insert/update key-value", "alloc"),
    ("HASH_GET", "( hash key -- hash value found )", "Get value by key", "pure"),
    ("HASH_DEL", "( hash key -- hash )", "Delete key", "alloc"),
    ("HASH_HAS", "( hash key -- hash found )", "Check if key exists", "pure"),
    ("HASH_LEN", "( hash -- hash count )", "Get entry count", "pure"),

    // Sets
    ("SET_NEW", "( -- set )", "Create empty set", "alloc"),
    ("SET_ADD", "( set value -- set )", "Add value to set", "alloc"),
    ("SET_HAS", "( set value -- set found )", "Check if value in set", "pure"),
    ("SET_DEL", "( set value -- set )", "Remove value from set", "alloc"),
    ("SET_LEN", "( set -- set count )", "Get set size", "pure"),
];

/// Standard library documentation: (name, params, returns, effects, description)
const STDLIB_DOCS: &[(&str, &str, &str, &str, &str)] = &[
    ("double", "x", "1", "pure", "Double a value: x * 2"),
    ("square", "x", "1", "pure", "Square a value: x * x"),
    ("abs", "x", "1", "pure", "Absolute value"),
    ("max", "a, b", "1", "pure", "Maximum of two values"),
    ("min", "a, b", "1", "pure", "Minimum of two values"),
    ("factorial", "n", "1", "pure", "Factorial: n!"),
    ("fib", "n", "1", "pure", "Fibonacci number"),
    ("gcd", "a, b", "1", "pure", "Greatest common divisor"),
    ("is_prime", "n", "1", "pure", "Primality test (1 if prime)"),
    ("sum_to", "n", "1", "pure", "Sum 1 to n"),
    ("pow", "base, exp", "1", "pure", "Integer exponentiation"),
    ("print_num", "n", "0", "io", "Print a number"),
    ("print_char", "c", "0", "io", "Print a character"),
    ("newline", "", "0", "io", "Print newline"),
    ("read_num", "", "1", "io", "Read a number from input"),
    ("inc", "x", "1", "pure", "Increment: x + 1"),
    ("dec", "x", "1", "pure", "Decrement: x - 1"),
    ("negate", "x", "1", "pure", "Negate: -x"),
    ("is_zero", "x", "1", "pure", "Check if zero (1 if true)"),
    ("is_positive", "x", "1", "pure", "Check if positive (signed)"),
    ("is_negative", "x", "1", "pure", "Check if negative (signed)"),
];

// ═══════════════════════════════════════════════════════════════════════════════
// LSP Server Implementation
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "lsp")]
pub mod server {
    //! LSP wire protocol implementation using lsp-server crate.

    use lsp_server::{Connection, Message, Notification, Request, Response};
    use lsp_types::{
        self, CompletionOptions, CompletionParams, CompletionItemKind, DidChangeTextDocumentParams,
        DidCloseTextDocumentParams, DidOpenTextDocumentParams, DidSaveTextDocumentParams,
        DocumentSymbolParams, DocumentSymbolResponse, GotoDefinitionParams, GotoDefinitionResponse,
        Hover, HoverContents, HoverParams, HoverProviderCapability, InitializeResult,
        InsertTextFormat, MarkupContent, MarkupKind, NumberOrString, OneOf, Position,
        PublishDiagnosticsParams, Range, SaveOptions, SemanticTokens, SemanticTokensFullOptions,
        SemanticTokensLegend, SemanticTokensOptions, SemanticTokensParams, SemanticTokensResult,
        SemanticTokensServerCapabilities, SemanticTokenType as LspSemanticTokenType, ServerCapabilities,
        ServerInfo, TextDocumentSyncCapability, TextDocumentSyncKind, TextDocumentSyncOptions,
        TextDocumentSyncSaveOptions, Url, DiagnosticSeverity, DiagnosticRelatedInformation,
        Documentation,
    };
    use serde::de::DeserializeOwned;
    use serde_json::Value;
    use super::{
        LanguageAnalyzer, Diagnostic, Severity, CompletionKind,
        DocumentSymbol, SymbolKind, SemanticTokenType,
    };

    /// Run the LSP server (main entry point).
    pub fn run_server() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        eprintln!("OUROCHRONOS LSP Server starting...");

        let (connection, io_threads) = Connection::stdio();

        // Server capabilities
        let capabilities = ServerCapabilities {
            text_document_sync: Some(TextDocumentSyncCapability::Options(
                TextDocumentSyncOptions {
                    open_close: Some(true),
                    change: Some(TextDocumentSyncKind::FULL),
                    will_save: Some(false),
                    will_save_wait_until: Some(false),
                    save: Some(TextDocumentSyncSaveOptions::SaveOptions(SaveOptions {
                        include_text: Some(true),
                    })),
                }
            )),
            completion_provider: Some(CompletionOptions {
                resolve_provider: Some(false),
                trigger_characters: Some(vec![".".to_string(), ":".to_string()]),
                all_commit_characters: None,
                work_done_progress_options: Default::default(),
                completion_item: None,
            }),
            hover_provider: Some(HoverProviderCapability::Simple(true)),
            definition_provider: Some(OneOf::Left(true)),
            document_symbol_provider: Some(OneOf::Left(true)),
            semantic_tokens_provider: Some(
                SemanticTokensServerCapabilities::SemanticTokensOptions(
                    SemanticTokensOptions {
                        work_done_progress_options: Default::default(),
                        legend: SemanticTokensLegend {
                            token_types: vec![
                                LspSemanticTokenType::KEYWORD,
                                LspSemanticTokenType::OPERATOR,
                                LspSemanticTokenType::NUMBER,
                                LspSemanticTokenType::STRING,
                                LspSemanticTokenType::COMMENT,
                                LspSemanticTokenType::FUNCTION,
                                LspSemanticTokenType::VARIABLE,
                                LspSemanticTokenType::PARAMETER,
                                LspSemanticTokenType::TYPE,
                                LspSemanticTokenType::MACRO,
                            ],
                            token_modifiers: vec![],
                        },
                        range: Some(false),
                        full: Some(SemanticTokensFullOptions::Bool(true)),
                    }
                )
            ),
            ..Default::default()
        };

        let init_result = InitializeResult {
            capabilities,
            server_info: Some(ServerInfo {
                name: "ourochronos-lsp".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
        };

        let init_result_json = serde_json::to_value(init_result)?;
        connection.initialize(init_result_json)?;

        eprintln!("OUROCHRONOS LSP Server initialized");

        // Main message loop
        let mut analyzer = LanguageAnalyzer::new();

        for msg in &connection.receiver {
            match msg {
                Message::Request(req) => {
                    if connection.handle_shutdown(&req)? {
                        eprintln!("Shutdown requested");
                        break;
                    }

                    handle_request(&mut analyzer, &connection, req)?;
                }
                Message::Notification(notif) => {
                    handle_notification(&mut analyzer, &connection, notif)?;
                }
                Message::Response(_) => {
                    // We don't send requests, so we don't expect responses
                }
            }
        }

        io_threads.join()?;
        eprintln!("OUROCHRONOS LSP Server stopped");
        Ok(())
    }

    /// Handle an LSP request.
    fn handle_request(
        analyzer: &mut LanguageAnalyzer,
        conn: &Connection,
        req: Request,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let id = req.id.clone();
        let method = req.method.as_str();

        let result: Result<Value, String> = match method {
            "textDocument/completion" => {
                handle_completion(analyzer, &req)
            }
            "textDocument/hover" => {
                handle_hover(analyzer, &req)
            }
            "textDocument/definition" => {
                handle_definition(analyzer, &req)
            }
            "textDocument/documentSymbol" => {
                handle_document_symbol(analyzer, &req)
            }
            "textDocument/semanticTokens/full" => {
                handle_semantic_tokens(analyzer, &req)
            }
            _ => {
                eprintln!("Unhandled request: {}", method);
                Err(format!("Method not supported: {}", method))
            }
        };

        let response = match result {
            Ok(value) => Response::new_ok(id, value),
            Err(msg) => Response::new_err(
                id,
                lsp_server::ErrorCode::MethodNotFound as i32,
                msg,
            ),
        };

        conn.sender.send(Message::Response(response))?;
        Ok(())
    }

    /// Handle an LSP notification.
    fn handle_notification(
        analyzer: &mut LanguageAnalyzer,
        conn: &Connection,
        notif: Notification,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match notif.method.as_str() {
            "textDocument/didOpen" => {
                let params: DidOpenTextDocumentParams = extract_params(notif.params)?;
                let uri = params.text_document.uri.to_string();
                let text = params.text_document.text;
                let version = params.text_document.version;

                let diagnostics = analyzer.open_document(&uri, &text, version);
                publish_diagnostics(conn, &uri, diagnostics, Some(version))?;
            }
            "textDocument/didChange" => {
                let params: DidChangeTextDocumentParams = extract_params(notif.params)?;
                let uri = params.text_document.uri.to_string();
                let version = params.text_document.version;

                // We use full sync, so there's exactly one change with the full text
                if let Some(change) = params.content_changes.into_iter().next() {
                    let diagnostics = analyzer.update_document(&uri, &change.text, version);
                    publish_diagnostics(conn, &uri, diagnostics, Some(version))?;
                }
            }
            "textDocument/didSave" => {
                let params: DidSaveTextDocumentParams = extract_params(notif.params)?;
                let uri = params.text_document.uri.to_string();

                // Re-analyze on save if text is provided
                if let Some(text) = params.text {
                    let diagnostics = analyzer.update_document(&uri, &text, 0);
                    publish_diagnostics(conn, &uri, diagnostics, None)?;
                }
            }
            "textDocument/didClose" => {
                let params: DidCloseTextDocumentParams = extract_params(notif.params)?;
                let uri = params.text_document.uri.to_string();
                analyzer.close_document(&uri);

                // Clear diagnostics
                publish_diagnostics(conn, &uri, vec![], None)?;
            }
            "initialized" => {
                eprintln!("Client initialized");
            }
            _ => {
                eprintln!("Unhandled notification: {}", notif.method);
            }
        }

        Ok(())
    }

    /// Extract typed parameters from JSON.
    fn extract_params<T: DeserializeOwned>(params: Value) -> Result<T, Box<dyn std::error::Error + Send + Sync>> {
        Ok(serde_json::from_value(params)?)
    }

    /// Publish diagnostics to the client.
    fn publish_diagnostics(
        conn: &Connection,
        uri: &str,
        diagnostics: Vec<Diagnostic>,
        version: Option<i32>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let lsp_diagnostics: Vec<lsp_types::Diagnostic> = diagnostics
            .into_iter()
            .map(|d| lsp_types::Diagnostic {
                range: Range {
                    start: Position {
                        line: d.line as u32,
                        character: d.column as u32,
                    },
                    end: Position {
                        line: d.end_line as u32,
                        character: d.end_column as u32,
                    },
                },
                severity: Some(match d.severity {
                    Severity::Error => DiagnosticSeverity::ERROR,
                    Severity::Warning => DiagnosticSeverity::WARNING,
                    Severity::Info => DiagnosticSeverity::INFORMATION,
                    Severity::Hint => DiagnosticSeverity::HINT,
                }),
                code: d.code.map(NumberOrString::String),
                code_description: None,
                source: Some("ourochronos".to_string()),
                message: d.message,
                related_information: if d.related.is_empty() {
                    None
                } else {
                    Some(d.related.into_iter().map(|r| DiagnosticRelatedInformation {
                        location: lsp_types::Location {
                            uri: r.uri.parse().unwrap_or_else(|_| Url::parse("file:///").unwrap()),
                            range: Range {
                                start: Position { line: r.line as u32, character: r.column as u32 },
                                end: Position { line: r.line as u32, character: r.column as u32 + 10 },
                            },
                        },
                        message: r.message,
                    }).collect())
                },
                tags: None,
                data: None,
            })
            .collect();

        let params = PublishDiagnosticsParams {
            uri: uri.parse()?,
            diagnostics: lsp_diagnostics,
            version,
        };

        let notif = Notification::new(
            "textDocument/publishDiagnostics".to_string(),
            serde_json::to_value(params)?,
        );

        conn.sender.send(Message::Notification(notif))?;
        Ok(())
    }

    /// Handle textDocument/completion request.
    fn handle_completion(
        analyzer: &LanguageAnalyzer,
        req: &Request,
    ) -> Result<Value, String> {
        let params: CompletionParams = serde_json::from_value(req.params.clone())
            .map_err(|e| e.to_string())?;

        let uri = params.text_document_position.text_document.uri.to_string();
        let line = params.text_document_position.position.line as usize;
        let column = params.text_document_position.position.character as usize;

        let items = analyzer.get_completions(&uri, line, column);

        let lsp_items: Vec<lsp_types::CompletionItem> = items
            .into_iter()
            .map(|item| lsp_types::CompletionItem {
                label: item.label,
                kind: Some(match item.kind {
                    CompletionKind::Keyword => CompletionItemKind::KEYWORD,
                    CompletionKind::Opcode => CompletionItemKind::OPERATOR,
                    CompletionKind::Procedure => CompletionItemKind::FUNCTION,
                    CompletionKind::Variable => CompletionItemKind::VARIABLE,
                    CompletionKind::Snippet => CompletionItemKind::SNIPPET,
                    CompletionKind::Constant => CompletionItemKind::CONSTANT,
                }),
                detail: item.detail,
                documentation: item.documentation.map(|d| {
                    Documentation::MarkupContent(MarkupContent {
                        kind: MarkupKind::Markdown,
                        value: d,
                    })
                }),
                insert_text: item.insert_text,
                insert_text_format: Some(InsertTextFormat::SNIPPET),
                sort_text: item.sort_text,
                filter_text: item.filter_text,
                ..Default::default()
            })
            .collect();

        serde_json::to_value(lsp_items).map_err(|e| e.to_string())
    }

    /// Handle textDocument/hover request.
    fn handle_hover(
        analyzer: &LanguageAnalyzer,
        req: &Request,
    ) -> Result<Value, String> {
        let params: HoverParams = serde_json::from_value(req.params.clone())
            .map_err(|e| e.to_string())?;

        let uri = params.text_document_position_params.text_document.uri.to_string();
        let line = params.text_document_position_params.position.line as usize;
        let column = params.text_document_position_params.position.character as usize;

        let hover = analyzer.get_hover(&uri, line, column);

        let lsp_hover = hover.map(|h| Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: h.contents,
            }),
            range: h.range.map(|(sl, sc, el, ec)| Range {
                start: Position { line: sl as u32, character: sc as u32 },
                end: Position { line: el as u32, character: ec as u32 },
            }),
        });

        serde_json::to_value(lsp_hover).map_err(|e| e.to_string())
    }

    /// Handle textDocument/definition request.
    fn handle_definition(
        analyzer: &LanguageAnalyzer,
        req: &Request,
    ) -> Result<Value, String> {
        let params: GotoDefinitionParams = serde_json::from_value(req.params.clone())
            .map_err(|e| e.to_string())?;

        let uri = params.text_document_position_params.text_document.uri.to_string();
        let line = params.text_document_position_params.position.line as usize;
        let column = params.text_document_position_params.position.character as usize;

        let location = analyzer.get_definition(&uri, line, column);

        let lsp_location = location.map(|loc| {
            GotoDefinitionResponse::Scalar(lsp_types::Location {
                uri: loc.uri.parse().unwrap_or_else(|_| Url::parse("file:///").unwrap()),
                range: Range {
                    start: Position { line: loc.line as u32, character: loc.column as u32 },
                    end: Position { line: loc.end_line as u32, character: loc.end_column as u32 },
                },
            })
        });

        serde_json::to_value(lsp_location).map_err(|e| e.to_string())
    }

    /// Handle textDocument/documentSymbol request.
    fn handle_document_symbol(
        analyzer: &LanguageAnalyzer,
        req: &Request,
    ) -> Result<Value, String> {
        let params: DocumentSymbolParams = serde_json::from_value(req.params.clone())
            .map_err(|e| e.to_string())?;

        let uri = params.text_document.uri.to_string();
        let symbols = analyzer.get_document_symbols(&uri);

        let lsp_symbols: Vec<lsp_types::DocumentSymbol> = symbols
            .into_iter()
            .map(|s| convert_document_symbol(s))
            .collect();

        serde_json::to_value(DocumentSymbolResponse::Nested(lsp_symbols))
            .map_err(|e| e.to_string())
    }

    fn convert_document_symbol(sym: DocumentSymbol) -> lsp_types::DocumentSymbol {
        #[allow(deprecated)]
        lsp_types::DocumentSymbol {
            name: sym.name,
            detail: sym.detail,
            kind: match sym.kind {
                SymbolKind::Function => lsp_types::SymbolKind::FUNCTION,
                SymbolKind::Variable => lsp_types::SymbolKind::VARIABLE,
                SymbolKind::Constant => lsp_types::SymbolKind::CONSTANT,
                SymbolKind::Module => lsp_types::SymbolKind::MODULE,
                SymbolKind::Operator => lsp_types::SymbolKind::OPERATOR,
            },
            tags: None,
            deprecated: None,
            range: Range {
                start: Position { line: sym.range.0 as u32, character: sym.range.1 as u32 },
                end: Position { line: sym.range.2 as u32, character: sym.range.3 as u32 },
            },
            selection_range: Range {
                start: Position { line: sym.selection_range.0 as u32, character: sym.selection_range.1 as u32 },
                end: Position { line: sym.selection_range.2 as u32, character: sym.selection_range.3 as u32 },
            },
            children: if sym.children.is_empty() {
                None
            } else {
                Some(sym.children.into_iter().map(convert_document_symbol).collect())
            },
        }
    }

    /// Handle textDocument/semanticTokens/full request.
    fn handle_semantic_tokens(
        analyzer: &LanguageAnalyzer,
        req: &Request,
    ) -> Result<Value, String> {
        let params: SemanticTokensParams = serde_json::from_value(req.params.clone())
            .map_err(|e| e.to_string())?;

        let uri = params.text_document.uri.to_string();
        let tokens = analyzer.get_semantic_tokens(&uri);

        // Convert to LSP delta encoding format
        let mut data: Vec<lsp_types::SemanticToken> = Vec::new();
        let mut prev_line = 0u32;
        let mut prev_char = 0u32;

        for token in tokens {
            let line = token.line as u32;
            let char = token.start_char as u32;

            let delta_line = line - prev_line;
            let delta_start = if delta_line == 0 { char - prev_char } else { char };

            data.push(lsp_types::SemanticToken {
                delta_line,
                delta_start,
                length: token.length as u32,
                token_type: semantic_token_type_index(token.token_type),
                token_modifiers_bitset: token.modifiers,
            });

            prev_line = line;
            prev_char = char;
        }

        let result = SemanticTokensResult::Tokens(SemanticTokens {
            result_id: None,
            data,
        });

        serde_json::to_value(result).map_err(|e| e.to_string())
    }

    fn semantic_token_type_index(tt: SemanticTokenType) -> u32 {
        match tt {
            SemanticTokenType::Keyword => 0,
            SemanticTokenType::Operator => 1,
            SemanticTokenType::Number => 2,
            SemanticTokenType::String => 3,
            SemanticTokenType::Comment => 4,
            SemanticTokenType::Function => 5,
            SemanticTokenType::Variable => 6,
            SemanticTokenType::Parameter => 7,
            SemanticTokenType::Type => 8,
            SemanticTokenType::Macro => 9,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyzer_creation() {
        let analyzer = LanguageAnalyzer::new();
        assert!(analyzer.documents.is_empty());
    }

    #[test]
    fn test_document_lifecycle() {
        let mut analyzer = LanguageAnalyzer::new();

        // Open
        let diags = analyzer.open_document("test://file.ouro", "1 2 ADD", 1);
        assert!(diags.is_empty());
        assert!(analyzer.get_document("test://file.ouro").is_some());

        // Update
        let diags = analyzer.update_document("test://file.ouro", "1 2 ADD OUTPUT", 2);
        assert!(diags.is_empty());

        // Close
        analyzer.close_document("test://file.ouro");
        assert!(analyzer.get_document("test://file.ouro").is_none());
    }

    #[test]
    fn test_parse_error_diagnostic() {
        let mut analyzer = LanguageAnalyzer::new();
        let diags = analyzer.open_document("test://file.ouro", "IF { 1 2", 1);
        assert!(!diags.is_empty());
        assert!(diags[0].severity == Severity::Error);
        assert!(diags[0].message.contains("Parse error"));
    }

    #[test]
    fn test_paradox_warning() {
        let mut analyzer = LanguageAnalyzer::new();
        let diags = analyzer.open_document("test://file.ouro", "PARADOX", 1);
        assert!(!diags.is_empty());
        assert!(diags.iter().any(|d| d.severity == Severity::Warning));
    }

    #[test]
    fn test_completions_non_empty() {
        let mut analyzer = LanguageAnalyzer::new();
        analyzer.open_document("test://file.ouro", "1 2 ", 1);
        let completions = analyzer.get_completions("test://file.ouro", 0, 4);
        assert!(!completions.is_empty());

        // Should have keywords
        assert!(completions.iter().any(|c| c.label == "IF"));
        // Should have opcodes
        assert!(completions.iter().any(|c| c.label == "ADD"));
        // Should have temporal ops
        assert!(completions.iter().any(|c| c.label == "ORACLE"));
    }

    #[test]
    fn test_completions_filtered() {
        let mut analyzer = LanguageAnalyzer::new();
        analyzer.open_document("test://file.ouro", "AD", 1);
        let completions = analyzer.get_completions("test://file.ouro", 0, 2);
        // Should filter to ADD
        assert!(completions.iter().any(|c| c.label == "ADD"));
        // Should not have unrelated
        assert!(!completions.iter().any(|c| c.label == "SUB"));
    }

    #[test]
    fn test_hover_opcode() {
        let mut analyzer = LanguageAnalyzer::new();
        analyzer.open_document("test://file.ouro", "ORACLE", 1);
        let hover = analyzer.get_hover("test://file.ouro", 0, 0);
        assert!(hover.is_some());
        let h = hover.unwrap();
        assert!(h.contents.contains("ORACLE"));
        assert!(h.contents.contains("anamnesis"));
    }

    #[test]
    fn test_hover_keyword() {
        let mut analyzer = LanguageAnalyzer::new();
        analyzer.open_document("test://file.ouro", "IF { 1 }", 1);
        let hover = analyzer.get_hover("test://file.ouro", 0, 0);
        assert!(hover.is_some());
        assert!(hover.unwrap().contents.contains("IF"));
    }

    #[test]
    fn test_document_symbols() {
        let mut analyzer = LanguageAnalyzer::new();
        analyzer.open_document("test://file.ouro", "PROCEDURE foo { 1 }", 1);
        let symbols = analyzer.get_document_symbols("test://file.ouro");
        assert!(!symbols.is_empty());
        assert!(symbols.iter().any(|s| s.name == "foo"));
    }

    #[test]
    fn test_go_to_definition() {
        let mut analyzer = LanguageAnalyzer::new();
        // Use correct Ourochronos syntax: procedure calls don't use parentheses
        let diags = analyzer.open_document("test://file.ouro", "PROCEDURE myproc { 1 }\nmyproc", 1);
        assert!(diags.is_empty(), "Unexpected diagnostics: {:?}", diags.iter().map(|d| &d.message).collect::<Vec<_>>());

        let loc = analyzer.get_definition("test://file.ouro", 1, 0);
        assert!(loc.is_some());
        let l = loc.unwrap();
        assert_eq!(l.line, 0);
    }

    #[test]
    fn test_semantic_tokens() {
        let mut analyzer = LanguageAnalyzer::new();
        analyzer.open_document("test://file.ouro", "IF { 1 ADD }", 1);
        let tokens = analyzer.get_semantic_tokens("test://file.ouro");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_document_position_conversion() {
        let doc = Document::new(
            "test://file.ouro".to_string(),
            "line one\nline two\nline three".to_string(),
            1,
        );

        // First line
        assert_eq!(doc.offset_to_position(0), (0, 0));
        assert_eq!(doc.offset_to_position(4), (0, 4));

        // Second line (after newline at index 8)
        assert_eq!(doc.offset_to_position(9), (1, 0));
        assert_eq!(doc.offset_to_position(13), (1, 4));
    }

    #[test]
    fn test_word_at_position() {
        let doc = Document::new(
            "test://file.ouro".to_string(),
            "ADD SUB MUL".to_string(),
            1,
        );

        assert_eq!(doc.word_at_position(0, 0), Some("ADD"));
        assert_eq!(doc.word_at_position(0, 4), Some("SUB"));
        assert_eq!(doc.word_at_position(0, 8), Some("MUL"));
    }

    #[test]
    fn test_effect_checking() {
        let mut analyzer = LanguageAnalyzer::new();
        // Pure procedure with temporal operation should error
        let diags = analyzer.open_document(
            "test://file.ouro",
            "PROCEDURE test PURE { 0 ORACLE }",
            1,
        );
        assert!(diags.iter().any(|d| d.message.contains("PURE") && d.message.contains("effects")));
    }

    #[test]
    fn test_unknown_procedure_error() {
        let mut analyzer = LanguageAnalyzer::new();
        // Use correct Ourochronos syntax: procedure calls don't use parentheses
        // When an unknown word is used, it should produce a parse error
        let diags = analyzer.open_document("test://file.ouro", "undefined_proc", 1);
        // Should produce some kind of error (parse error for unknown identifier)
        assert!(!diags.is_empty(), "Expected diagnostics for unknown procedure");
        assert!(diags.iter().any(|d| d.message.contains("Unknown") || d.message.contains("Parse error")),
            "Expected unknown/parse error diagnostic, got: {:?}", diags.iter().map(|d| &d.message).collect::<Vec<_>>());
    }

    #[test]
    fn test_oracle_in_loop_warning() {
        let mut analyzer = LanguageAnalyzer::new();
        let diags = analyzer.open_document(
            "test://file.ouro",
            "WHILE { 1 } { 0 ORACLE POP }",
            1,
        );
        assert!(diags.iter().any(|d| d.message.contains("ORACLE inside loop")));
    }

    #[test]
    fn test_stdlib_completion() {
        let mut analyzer = LanguageAnalyzer::new();
        analyzer.open_document("test://file.ouro", "fac", 1);
        let completions = analyzer.get_completions("test://file.ouro", 0, 3);
        assert!(completions.iter().any(|c| c.label == "factorial"));
    }

    #[test]
    fn test_analyze_convenience() {
        let mut analyzer = LanguageAnalyzer::new();
        let diags = analyzer.analyze("test://file.ouro", "1 2 ADD");
        assert!(diags.is_empty());
    }
}
