//! Fallible, source-aware lexical analysis.
//!
//! The lexer owns the token vocabulary and ties every token and diagnostic to
//! retained source text through a [`SourceSpan`]. The legacy parser re-exports
//! [`Token`] during the migration. All spans are half-open UTF-8 byte ranges.

use crate::source::{SourceId, SourceSpan, TextRange};
use std::error::Error;
use std::fmt;

/// Maximum number of lexical tokens accepted from one source or one module
/// graph. The instruction ceiling covers executable words and the additional
/// quarter leaves bounded space for declarations and structural delimiters.
pub const MAX_SOURCE_TOKENS: usize =
    crate::bytecode::MAX_INSTRUCTIONS + crate::bytecode::MAX_INSTRUCTIONS / 4;

/// Tokens accepted by the Ourochronos source parser.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Word(String),
    Number(u64),
    StringLit(String),
    CharLit(char),
    LBrace,
    RBrace,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Equals,
    Semicolon,
}

/// A parser-compatible token and its exact retained-source location.
#[derive(Debug, Clone, PartialEq)]
pub struct LocatedToken {
    /// Token understood by the existing parser.
    pub token: Token,
    /// Half-open UTF-8 byte span covering the complete token spelling.
    pub span: SourceSpan,
}

impl LocatedToken {
    fn new(token: Token, span: SourceSpan) -> Self {
        Self { token, span }
    }
}

/// The kind of literal in which an escape occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiteralKind {
    /// A double-quoted string literal.
    String,
    /// A single-quoted character literal.
    Character,
}

impl fmt::Display for LiteralKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String => formatter.write_str("string"),
            Self::Character => formatter.write_str("character"),
        }
    }
}

/// Structured reason that lexical analysis failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LexErrorKind {
    /// A character does not begin any Ourochronos token.
    UnknownCharacter(char),
    /// A double-quoted literal reached end of input without a closing quote.
    UnterminatedString,
    /// A single-quoted literal reached a line boundary or end of input.
    UnterminatedCharacter,
    /// A character literal contained no character.
    EmptyCharacter,
    /// A character literal contained more than one decoded character.
    OverlongCharacter {
        /// Number of decoded characters between the quotes.
        characters: usize,
    },
    /// An escape is not part of the language, or ends at end of input.
    InvalidEscape {
        /// Character following the backslash, or `None` for a trailing slash.
        escape: Option<char>,
        /// Literal syntax in which the escape occurred.
        literal: LiteralKind,
    },
    /// A `0x` or `0b` prefix was not followed by any valid digit.
    MissingRadixDigits {
        /// Expected radix, currently 2 or 16.
        radix: u32,
    },
    /// An alphanumeric character adjacent to a radix literal is not a digit
    /// in that radix.
    InvalidRadixDigit {
        /// Literal radix, currently 2 or 16.
        radix: u32,
        /// Invalid adjacent character.
        digit: char,
    },
    /// A syntactically valid integer does not fit in `u64`.
    IntegerOverflow {
        /// Radix in which the integer was written.
        radix: u32,
    },
    /// Tokenization reached the allocation ceiling before the current token.
    TokenLimitExceeded {
        /// Maximum number of tokens accepted by this lexer invocation.
        limit: usize,
    },
}

impl fmt::Display for LexErrorKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownCharacter(character) => {
                write!(formatter, "unknown character {character:?}")
            }
            Self::UnterminatedString => formatter.write_str("unterminated string literal"),
            Self::UnterminatedCharacter => formatter.write_str("unterminated character literal"),
            Self::EmptyCharacter => formatter.write_str("empty character literal"),
            Self::OverlongCharacter { characters } => write!(
                formatter,
                "character literal contains {characters} decoded characters"
            ),
            Self::InvalidEscape { escape, literal } => match escape {
                Some(character) => {
                    write!(formatter, "invalid {literal} escape \\{character}")
                }
                None => write!(formatter, "incomplete escape at end of {literal} literal"),
            },
            Self::MissingRadixDigits { radix } => {
                write!(formatter, "base-{radix} literal has no valid digits")
            }
            Self::InvalidRadixDigit { radix, digit } => {
                write!(formatter, "{digit:?} is not a base-{radix} digit")
            }
            Self::IntegerOverflow { radix } => {
                write!(formatter, "base-{radix} integer does not fit in u64")
            }
            Self::TokenLimitExceeded { limit } => {
                write!(formatter, "source exceeds the {limit}-token limit")
            }
        }
    }
}

/// A lexical error tied to an exact retained-source span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexError {
    /// Source range responsible for the diagnostic.
    pub span: SourceSpan,
    /// Structured diagnostic reason.
    pub kind: LexErrorKind,
}

impl fmt::Display for LexError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} at bytes {}..{}: {}",
            self.span.source, self.span.range.start, self.span.range.end, self.kind
        )
    }
}

impl Error for LexError {}

/// Lexes `input` using the complete legacy token surface.
///
/// Valid input produces parser-compatible tokens.  Invalid input produces all
/// diagnostics recoverable in a deterministic left-to-right pass.  No token
/// vector is exposed when any diagnostic is present, preventing callers from
/// accidentally compiling a partially lexed program.
pub fn lex(source: SourceId, input: &str) -> Result<Vec<LocatedToken>, Vec<LexError>> {
    lex_with_token_limit(source, input, MAX_SOURCE_TOKENS)
}

/// Lex with a caller-supplied ceiling no larger than the public source limit.
/// Module-graph construction uses the graph's remaining aggregate budget.
pub(crate) fn lex_with_token_limit(
    source: SourceId,
    input: &str,
    token_limit: usize,
) -> Result<Vec<LocatedToken>, Vec<LexError>> {
    let mut lexer = Lexer::new(source, input, token_limit.min(MAX_SOURCE_TOKENS));
    lexer.run();
    if lexer.errors.is_empty() {
        Ok(lexer.tokens)
    } else {
        Err(lexer.errors)
    }
}

struct Lexer<'a> {
    source: SourceId,
    input: &'a str,
    offset: usize,
    tokens: Vec<LocatedToken>,
    errors: Vec<LexError>,
    token_limit: usize,
    halted: bool,
}

impl<'a> Lexer<'a> {
    fn new(source: SourceId, input: &'a str, token_limit: usize) -> Self {
        Self {
            source,
            input,
            offset: 0,
            tokens: Vec::new(),
            errors: Vec::new(),
            token_limit,
            halted: false,
        }
    }

    fn run(&mut self) {
        while !self.halted {
            let Some(character) = self.current() else {
                break;
            };
            match character {
                ' ' | '\t' | '\n' => {
                    self.bump();
                }
                '\r' => {
                    self.bump();
                    if self.current() == Some('\n') {
                        self.bump();
                    }
                }
                '#' => self.skip_line_comment(),
                ';' if self.following() == Some(';') => self.skip_line_comment(),
                '/' if self.following() == Some('/') => self.skip_line_comment(),
                '{' => self.punctuation(Token::LBrace),
                '}' => self.punctuation(Token::RBrace),
                '(' => self.punctuation(Token::LParen),
                ')' => self.punctuation(Token::RParen),
                '[' => self.punctuation(Token::LBracket),
                ']' => self.punctuation(Token::RBracket),
                ',' => self.punctuation(Token::Comma),
                ';' => self.punctuation(Token::Semicolon),
                '"' => self.string_literal(),
                '\'' => self.character_literal(),
                character if character.is_ascii_digit() => self.number(),
                character if character.is_alphabetic() || character == '_' => self.word(),
                '=' if self.following() == Some('=') => self.double_equals(),
                '=' => self.punctuation(Token::Equals),
                '/' => self.operator(false),
                '+' | '-' | '*' | '%' | '&' | '|' | '^' | '~' | '<' | '>' | '!' | '@' | ':' => {
                    self.operator(true);
                }
                unknown => {
                    let start = self.offset;
                    self.bump();
                    self.error(start, self.offset, LexErrorKind::UnknownCharacter(unknown));
                }
            }
        }
    }

    fn current(&self) -> Option<char> {
        self.input[self.offset..].chars().next()
    }

    fn following(&self) -> Option<char> {
        let mut characters = self.input[self.offset..].chars();
        characters.next()?;
        characters.next()
    }

    fn bump(&mut self) -> Option<char> {
        let character = self.current()?;
        self.offset += character.len_utf8();
        Some(character)
    }

    fn span(&self, start: usize, end: usize) -> SourceSpan {
        SourceSpan::new(self.source, TextRange::new(start, end))
    }

    fn emit(&mut self, start: usize, token: Token) {
        if self.tokens.len() >= self.token_limit {
            self.error(
                start,
                self.offset,
                LexErrorKind::TokenLimitExceeded {
                    limit: self.token_limit,
                },
            );
            self.halted = true;
            return;
        }
        self.tokens
            .push(LocatedToken::new(token, self.span(start, self.offset)));
    }

    fn error(&mut self, start: usize, end: usize, kind: LexErrorKind) {
        self.errors.push(LexError {
            span: self.span(start, end),
            kind,
        });
    }

    fn punctuation(&mut self, token: Token) {
        let start = self.offset;
        self.bump();
        self.emit(start, token);
    }

    fn skip_line_comment(&mut self) {
        while !matches!(self.current(), None | Some('\n')) {
            self.bump();
        }
    }

    fn word(&mut self) {
        let start = self.offset;
        while matches!(self.current(), Some(character) if character.is_alphanumeric() || character == '_')
        {
            self.bump();
        }
        self.emit(
            start,
            Token::Word(self.input[start..self.offset].to_owned()),
        );
    }

    fn double_equals(&mut self) {
        let start = self.offset;
        self.bump();
        self.bump();
        self.emit(start, Token::Word("==".to_owned()));
    }

    fn operator(&mut self, allow_double: bool) {
        let start = self.offset;
        let first = self.bump().expect("operator starts at a character");
        if allow_double
            && matches!(
                (first, self.current()),
                ('<', Some('='))
                    | ('>', Some('='))
                    | ('!', Some('='))
                    | ('<', Some('<'))
                    | ('>', Some('>'))
                    | ('&', Some('&'))
                    | ('|', Some('|'))
            )
        {
            self.bump();
        }
        self.emit(
            start,
            Token::Word(self.input[start..self.offset].to_owned()),
        );
    }

    fn string_literal(&mut self) {
        let start = self.offset;
        self.bump();
        let mut value = String::new();
        let mut valid = true;

        while let Some(character) = self.current() {
            match character {
                '"' => {
                    self.bump();
                    if valid {
                        self.emit(start, Token::StringLit(value));
                    }
                    return;
                }
                '\\' => {
                    let escape_start = self.offset;
                    self.bump();
                    match self.bump() {
                        Some('n') => value.push('\n'),
                        Some('t') => value.push('\t'),
                        Some('r') => value.push('\r'),
                        Some('"') => value.push('"'),
                        Some('\\') => value.push('\\'),
                        Some(invalid) => {
                            valid = false;
                            self.error(
                                escape_start,
                                self.offset,
                                LexErrorKind::InvalidEscape {
                                    escape: Some(invalid),
                                    literal: LiteralKind::String,
                                },
                            );
                        }
                        None => {
                            self.error(
                                escape_start,
                                self.offset,
                                LexErrorKind::InvalidEscape {
                                    escape: None,
                                    literal: LiteralKind::String,
                                },
                            );
                        }
                    }
                }
                other => {
                    self.bump();
                    value.push(other);
                }
            }
        }

        self.error(start, self.offset, LexErrorKind::UnterminatedString);
    }

    fn character_literal(&mut self) {
        let start = self.offset;
        self.bump();
        let mut value = None;
        let mut characters = 0usize;
        let mut valid = true;

        loop {
            match self.current() {
                Some('\'') => {
                    self.bump();
                    match characters {
                        0 => self.error(start, self.offset, LexErrorKind::EmptyCharacter),
                        1 if valid => {
                            self.emit(
                                start,
                                Token::CharLit(value.expect("one decoded character has a value")),
                            );
                        }
                        1 => {}
                        _ => self.error(
                            start,
                            self.offset,
                            LexErrorKind::OverlongCharacter { characters },
                        ),
                    }
                    return;
                }
                None | Some('\n' | '\r') => {
                    self.error(start, self.offset, LexErrorKind::UnterminatedCharacter);
                    return;
                }
                Some('\\') => {
                    let escape_start = self.offset;
                    self.bump();
                    characters += 1;
                    match self.current() {
                        Some('n') => {
                            self.bump();
                            value.get_or_insert('\n')
                        }
                        Some('t') => {
                            self.bump();
                            value.get_or_insert('\t')
                        }
                        Some('r') => {
                            self.bump();
                            value.get_or_insert('\r')
                        }
                        Some('\'') => {
                            self.bump();
                            value.get_or_insert('\'')
                        }
                        Some('\\') => {
                            self.bump();
                            value.get_or_insert('\\')
                        }
                        Some(line_break @ ('\n' | '\r')) => {
                            self.error(
                                escape_start,
                                self.offset + line_break.len_utf8(),
                                LexErrorKind::InvalidEscape {
                                    escape: Some(line_break),
                                    literal: LiteralKind::Character,
                                },
                            );
                            self.error(start, self.offset, LexErrorKind::UnterminatedCharacter);
                            return;
                        }
                        Some(invalid) => {
                            self.bump();
                            valid = false;
                            self.error(
                                escape_start,
                                self.offset,
                                LexErrorKind::InvalidEscape {
                                    escape: Some(invalid),
                                    literal: LiteralKind::Character,
                                },
                            );
                            value.get_or_insert(invalid)
                        }
                        None => {
                            self.error(
                                escape_start,
                                self.offset,
                                LexErrorKind::InvalidEscape {
                                    escape: None,
                                    literal: LiteralKind::Character,
                                },
                            );
                            self.error(start, self.offset, LexErrorKind::UnterminatedCharacter);
                            return;
                        }
                    };
                }
                Some(character) => {
                    self.bump();
                    characters += 1;
                    value.get_or_insert(character);
                }
            }
        }
    }

    fn number(&mut self) {
        let start = self.offset;
        if self.current() == Some('0') {
            match self.following() {
                Some('x' | 'X') => {
                    self.radix_number(start, 16);
                    return;
                }
                Some('b' | 'B') => {
                    self.radix_number(start, 2);
                    return;
                }
                _ => {}
            }
        }

        while matches!(self.current(), Some(character) if character.is_ascii_digit()) {
            self.bump();
        }
        match self.input[start..self.offset].parse::<u64>() {
            Ok(number) => self.emit(start, Token::Number(number)),
            Err(_) => self.error(
                start,
                self.offset,
                LexErrorKind::IntegerOverflow { radix: 10 },
            ),
        }
    }

    fn radix_number(&mut self, start: usize, radix: u32) {
        self.bump();
        self.bump();
        let digits_start = self.offset;
        while matches!(self.current(), Some(character) if character.is_digit(radix)) {
            self.bump();
        }
        let digits_end = self.offset;
        let mut malformed = false;

        if digits_start == digits_end {
            malformed = true;
            self.error(
                start,
                digits_end,
                LexErrorKind::MissingRadixDigits { radix },
            );
        }

        while matches!(self.current(), Some(character) if character.is_alphanumeric() || character == '_')
        {
            let invalid_start = self.offset;
            let character = self.bump().expect("adjacent character is present");
            malformed = true;
            if character.to_digit(radix).is_none() {
                self.error(
                    invalid_start,
                    self.offset,
                    LexErrorKind::InvalidRadixDigit {
                        radix,
                        digit: character,
                    },
                );
            }
        }

        if malformed {
            return;
        }

        match u64::from_str_radix(&self.input[digits_start..digits_end], radix) {
            Ok(number) => self.emit(start, Token::Number(number)),
            Err(_) => self.error(start, self.offset, LexErrorKind::IntegerOverflow { radix }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: SourceId = SourceId::new(7);

    fn tokens(input: &str) -> Vec<LocatedToken> {
        lex(SOURCE, input).expect("input should lex")
    }

    fn errors(input: &str) -> Vec<LexError> {
        lex(SOURCE, input).expect_err("input should be rejected")
    }

    fn range(token: &LocatedToken) -> TextRange {
        token.span.range
    }

    #[test]
    fn token_limit_fails_before_retaining_an_over_limit_token() {
        let errors = lex_with_token_limit(SOURCE, "1 2 3 4 5", 4).unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(
            errors[0].kind,
            LexErrorKind::TokenLimitExceeded { limit: 4 }
        );
        assert_eq!(errors[0].span.range, TextRange::new(8, 9));
        assert_eq!(
            MAX_SOURCE_TOKENS,
            crate::bytecode::MAX_INSTRUCTIONS + crate::bytecode::MAX_INSTRUCTIONS / 4
        );
    }

    #[test]
    fn lexes_complete_legacy_punctuation_and_operator_surface() {
        let actual: Vec<Token> =
            tokens("{}()[], = ; / == + - * % & | ^ ~ < > ! @ : <= >= != << >> && ||")
                .into_iter()
                .map(|located| located.token)
                .collect();
        assert_eq!(
            actual,
            vec![
                Token::LBrace,
                Token::RBrace,
                Token::LParen,
                Token::RParen,
                Token::LBracket,
                Token::RBracket,
                Token::Comma,
                Token::Equals,
                Token::Semicolon,
                Token::Word("/".into()),
                Token::Word("==".into()),
                Token::Word("+".into()),
                Token::Word("-".into()),
                Token::Word("*".into()),
                Token::Word("%".into()),
                Token::Word("&".into()),
                Token::Word("|".into()),
                Token::Word("^".into()),
                Token::Word("~".into()),
                Token::Word("<".into()),
                Token::Word(">".into()),
                Token::Word("!".into()),
                Token::Word("@".into()),
                Token::Word(":".into()),
                Token::Word("<=".into()),
                Token::Word(">=".into()),
                Token::Word("!=".into()),
                Token::Word("<<".into()),
                Token::Word(">>".into()),
                Token::Word("&&".into()),
                Token::Word("||".into()),
            ]
        );
    }

    #[test]
    fn skips_exact_legacy_whitespace_newlines_and_comments() {
        let actual: Vec<Token> = tokens("A # one\r\n B ;; two\nC // three\r\nD\tE")
            .into_iter()
            .map(|located| located.token)
            .collect();
        assert_eq!(
            actual,
            vec![
                Token::Word("A".into()),
                Token::Word("B".into()),
                Token::Word("C".into()),
                Token::Word("D".into()),
                Token::Word("E".into()),
            ]
        );
    }

    #[test]
    fn lexes_numbers_words_and_exact_utf8_byte_spans() {
        let actual = tokens("α_2 42 0xFF 0B1010");
        assert_eq!(actual[0].token, Token::Word("α_2".into()));
        assert_eq!(range(&actual[0]), TextRange::new(0, 4));
        assert_eq!(actual[1].token, Token::Number(42));
        assert_eq!(range(&actual[1]), TextRange::new(5, 7));
        assert_eq!(actual[2].token, Token::Number(255));
        assert_eq!(range(&actual[2]), TextRange::new(8, 12));
        assert_eq!(actual[3].token, Token::Number(10));
        assert_eq!(range(&actual[3]), TextRange::new(13, 19));
        assert!(actual.iter().all(|token| token.span.source == SOURCE));
    }

    #[test]
    fn accepts_the_largest_u64_in_every_supported_radix() {
        let actual: Vec<Token> = tokens(
            "18446744073709551615 0xffffffffffffffff 0b1111111111111111111111111111111111111111111111111111111111111111",
        )
        .into_iter()
        .map(|located| located.token)
        .collect();
        assert_eq!(
            actual,
            vec![
                Token::Number(u64::MAX),
                Token::Number(u64::MAX),
                Token::Number(u64::MAX),
            ]
        );
    }

    #[test]
    fn decodes_supported_string_and_character_escapes() {
        let actual = tokens(r#""a\n\t\r\"\\b" '\n' '\t' '\r' '\'' '\\' 'λ'"#);
        assert_eq!(actual[0].token, Token::StringLit("a\n\t\r\"\\b".into()));
        assert_eq!(actual[1].token, Token::CharLit('\n'));
        assert_eq!(actual[2].token, Token::CharLit('\t'));
        assert_eq!(actual[3].token, Token::CharLit('\r'));
        assert_eq!(actual[4].token, Token::CharLit('\''));
        assert_eq!(actual[5].token, Token::CharLit('\\'));
        assert_eq!(actual[6].token, Token::CharLit('λ'));
    }

    #[test]
    fn permits_multiline_strings_with_exact_span() {
        let actual = tokens("\"a\r\nb\"");
        assert_eq!(actual[0].token, Token::StringLit("a\r\nb".into()));
        assert_eq!(range(&actual[0]), TextRange::new(0, 6));
    }

    #[test]
    fn reports_unknown_utf8_characters_and_recovers() {
        let actual = errors("$ ` ? 🙂");
        assert_eq!(actual.len(), 4);
        assert_eq!(actual[0].kind, LexErrorKind::UnknownCharacter('$'));
        assert_eq!(actual[0].span.range, TextRange::new(0, 1));
        assert_eq!(actual[1].span.range, TextRange::new(2, 3));
        assert_eq!(actual[2].span.range, TextRange::new(4, 5));
        assert_eq!(actual[3].span.range, TextRange::new(6, 10));
    }

    #[test]
    fn reports_unterminated_literals() {
        let string_error = errors("\"abc");
        assert_eq!(string_error[0].kind, LexErrorKind::UnterminatedString);
        assert_eq!(string_error[0].span.range, TextRange::new(0, 4));

        let char_error = errors("'x\n42");
        assert_eq!(char_error[0].kind, LexErrorKind::UnterminatedCharacter);
        assert_eq!(char_error[0].span.range, TextRange::new(0, 2));
    }

    #[test]
    fn reports_empty_and_overlong_characters() {
        let empty = errors("''");
        assert_eq!(empty[0].kind, LexErrorKind::EmptyCharacter);
        assert_eq!(empty[0].span.range, TextRange::new(0, 2));

        let overlong = errors("'ab'");
        assert_eq!(
            overlong[0].kind,
            LexErrorKind::OverlongCharacter { characters: 2 }
        );
        assert_eq!(overlong[0].span.range, TextRange::new(0, 4));
    }

    #[test]
    fn reports_invalid_and_incomplete_escapes() {
        let actual = errors("\"\\q\" '\\z' \"tail\\");
        assert_eq!(actual.len(), 4);
        assert_eq!(
            actual[0].kind,
            LexErrorKind::InvalidEscape {
                escape: Some('q'),
                literal: LiteralKind::String,
            }
        );
        assert_eq!(actual[0].span.range, TextRange::new(1, 3));
        assert_eq!(
            actual[1].kind,
            LexErrorKind::InvalidEscape {
                escape: Some('z'),
                literal: LiteralKind::Character,
            }
        );
        assert_eq!(
            actual[2].kind,
            LexErrorKind::InvalidEscape {
                escape: None,
                literal: LiteralKind::String,
            }
        );
        assert_eq!(actual[3].kind, LexErrorKind::UnterminatedString);
    }

    #[test]
    fn character_escape_at_line_end_recovers_on_the_next_line() {
        let actual = errors("'\\\n$'");
        assert_eq!(actual.len(), 4);
        assert!(matches!(
            actual[0].kind,
            LexErrorKind::InvalidEscape {
                escape: Some('\n'),
                literal: LiteralKind::Character,
            }
        ));
        assert_eq!(actual[1].kind, LexErrorKind::UnterminatedCharacter);
        assert_eq!(actual[2].kind, LexErrorKind::UnknownCharacter('$'));
        assert_eq!(actual[3].kind, LexErrorKind::UnterminatedCharacter);
    }

    #[test]
    fn reports_missing_and_invalid_radix_digits_and_recovers() {
        let actual = errors("0x 0b2 0x1G 0b102 @");
        assert_eq!(
            actual.iter().map(|error| &error.kind).collect::<Vec<_>>(),
            vec![
                &LexErrorKind::MissingRadixDigits { radix: 16 },
                &LexErrorKind::MissingRadixDigits { radix: 2 },
                &LexErrorKind::InvalidRadixDigit {
                    radix: 2,
                    digit: '2',
                },
                &LexErrorKind::InvalidRadixDigit {
                    radix: 16,
                    digit: 'G',
                },
                &LexErrorKind::InvalidRadixDigit {
                    radix: 2,
                    digit: '2',
                },
            ]
        );
        assert_eq!(actual[0].span.range, TextRange::new(0, 2));
        assert_eq!(actual[2].span.range, TextRange::new(5, 6));
    }

    #[test]
    fn reports_decimal_hex_and_binary_u64_overflow() {
        let actual = errors(
            "18446744073709551616 0x10000000000000000 0b10000000000000000000000000000000000000000000000000000000000000000",
        );
        assert_eq!(
            actual.iter().map(|error| &error.kind).collect::<Vec<_>>(),
            vec![
                &LexErrorKind::IntegerOverflow { radix: 10 },
                &LexErrorKind::IntegerOverflow { radix: 16 },
                &LexErrorKind::IntegerOverflow { radix: 2 },
            ]
        );
    }

    #[test]
    fn error_display_includes_source_bytes_and_reason() {
        let error = &errors("$")[0];
        assert_eq!(
            error.to_string(),
            "source 7 at bytes 0..1: unknown character '$'"
        );
        let _: &dyn Error = error;
    }
}
