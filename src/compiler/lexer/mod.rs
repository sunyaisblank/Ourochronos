//! Lexer for OUROCHRONOS.
//!
//! Provides tokenization with full source span tracking for precise error reporting.
//!
//! # Architecture
//!
//! The lexer converts source text to a stream of tokens, each annotated with
//! its source location (line, column, byte offset).
//!
//! ```text
//! Source Text → Character Stream → Tokens with Spans
//! ```
//!
//! # Precision Requirements
//!
//! - Byte offsets are exact for UTF-8 multi-byte characters
//! - Column tracking uses character count (not bytes)
//! - Line numbers are 1-indexed (first line is line 1)

// Re-export span types from parser module
pub use crate::parser::{Span, Token, SpannedToken, tokenize, tokenize_with_spans};

/// Token classification for syntax highlighting and analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    /// Keyword (opcode, control flow).
    Keyword,
    /// Numeric literal.
    Number,
    /// String literal.
    String,
    /// Character literal.
    Char,
    /// Identifier (variable, procedure name).
    Identifier,
    /// Punctuation ({ } ( ) [ ] , = ;).
    Punctuation,
    /// Comment (not tokenized, but useful for highlighting).
    Comment,
    /// Whitespace (not tokenized).
    Whitespace,
    /// Unknown/error token.
    Unknown,
}

impl Token {
    /// Classify a token for syntax purposes.
    pub fn kind(&self) -> TokenKind {
        match self {
            Token::Word(w) => {
                // Check if it's a known keyword
                let upper = w.to_uppercase();
                if is_keyword(&upper) {
                    TokenKind::Keyword
                } else {
                    TokenKind::Identifier
                }
            }
            Token::Number(_) => TokenKind::Number,
            Token::StringLit(_) => TokenKind::String,
            Token::CharLit(_) => TokenKind::Char,
            Token::LBrace | Token::RBrace |
            Token::LParen | Token::RParen |
            Token::LBracket | Token::RBracket |
            Token::Comma | Token::Equals | Token::Semicolon => TokenKind::Punctuation,
        }
    }
}

/// Check if a word is a known keyword.
fn is_keyword(word: &str) -> bool {
    // Stack operations
    matches!(word,
        "NOP" | "HALT" | "POP" | "DUP" | "SWAP" | "OVER" | "ROT" |
        "DEPTH" | "PICK" | "ROLL" | "REVERSE" | "EXEC" | "DIP" | "KEEP" | "BI" | "REC" |
        // Arithmetic
        "ADD" | "SUB" | "MUL" | "DIV" | "MOD" | "NEG" | "ABS" | "MIN" | "MAX" | "SIGN" |
        // Bitwise
        "NOT" | "AND" | "OR" | "XOR" | "SHL" | "SHR" |
        // Comparison
        "EQ" | "NEQ" | "LT" | "GT" | "LTE" | "GTE" | "SLT" | "SGT" | "SLTE" | "SGTE" |
        // Temporal
        "ORACLE" | "PROPHECY" | "PRESENT" | "PARADOX" |
        // Control flow
        "IF" | "ELSE" | "WHILE" | "ASSERT" | "MATCH" |
        // Data structures
        "VEC_NEW" | "VEC_PUSH" | "VEC_POP" | "VEC_GET" | "VEC_SET" | "VEC_LEN" |
        "HASH_NEW" | "HASH_GET" | "HASH_SET" | "HASH_DEL" | "HASH_HAS" |
        "SET_NEW" | "SET_ADD" | "SET_DEL" | "SET_HAS" | "SET_LEN" |
        // I/O
        "INPUT" | "OUTPUT" | "EMIT" |
        // Declarations
        "PROCEDURE" | "TEMPORAL" | "LET" | "CONST" |
        // Boolean literals
        "TRUE" | "FALSE"
    )
}

/// Source map for looking up spans by position.
#[derive(Debug, Clone)]
pub struct SourceMap {
    /// Line start offsets.
    line_starts: Vec<usize>,
    /// Total source length.
    source_len: usize,
}

impl SourceMap {
    /// Build a source map from source text.
    pub fn new(source: &str) -> Self {
        let mut line_starts = vec![0];
        for (i, c) in source.char_indices() {
            if c == '\n' {
                line_starts.push(i + 1);
            }
        }
        Self {
            line_starts,
            source_len: source.len(),
        }
    }

    /// Get line and column from byte offset.
    pub fn position(&self, offset: usize) -> (usize, usize) {
        if offset >= self.source_len {
            let line = self.line_starts.len();
            let col = offset - self.line_starts.last().copied().unwrap_or(0) + 1;
            return (line, col);
        }

        // Binary search for line
        let line = match self.line_starts.binary_search(&offset) {
            Ok(i) => i + 1,
            Err(i) => i,
        };

        let line_start = self.line_starts.get(line.saturating_sub(1)).copied().unwrap_or(0);
        let col = offset - line_start + 1;

        (line, col)
    }

    /// Get byte offset range for a line (1-indexed).
    pub fn line_range(&self, line: usize) -> Option<(usize, usize)> {
        if line == 0 || line > self.line_starts.len() {
            return None;
        }

        let start = self.line_starts[line - 1];
        let end = self.line_starts.get(line).copied().unwrap_or(self.source_len);

        Some((start, end))
    }

    /// Get number of lines.
    pub fn line_count(&self) -> usize {
        self.line_starts.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_kind() {
        assert_eq!(Token::Word("ADD".to_string()).kind(), TokenKind::Keyword);
        assert_eq!(Token::Word("my_var".to_string()).kind(), TokenKind::Identifier);
        assert_eq!(Token::Number(42).kind(), TokenKind::Number);
        assert_eq!(Token::LBrace.kind(), TokenKind::Punctuation);
    }

    #[test]
    fn test_source_map() {
        let source = "line1\nline2\nline3";
        let map = SourceMap::new(source);

        assert_eq!(map.line_count(), 3);
        assert_eq!(map.position(0), (1, 1));   // start of line1
        assert_eq!(map.position(6), (2, 1));   // start of line2
        assert_eq!(map.position(12), (3, 1));  // start of line3
        assert_eq!(map.position(14), (3, 3));  // 'n' in line3
    }

    #[test]
    fn test_source_map_line_range() {
        let source = "abc\ndef\nghi";
        let map = SourceMap::new(source);

        assert_eq!(map.line_range(1), Some((0, 4)));
        assert_eq!(map.line_range(2), Some((4, 8)));
        assert_eq!(map.line_range(3), Some((8, 11)));
        assert_eq!(map.line_range(4), None);
    }
}
