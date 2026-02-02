//! Temporal operations domain parser.
//!
//! Handles temporal keywords: ORACLE, PROPHECY, PRESENT, PARADOX.
//! These are the core CTC (Closed Timelike Curve) operations.

use crate::ast::{OpCode, Stmt};
use super::domain::{DomainParser, ParseContext};

/// Parser for temporal (CTC) operations.
pub struct TemporalParser;

impl TemporalParser {
    pub const KEYWORDS: &'static [&'static str] = &[
        "ORACLE", "READ",      // Read from future (anamnesis)
        "PROPHECY", "WRITE",   // Write to past (anamnesis)
        "PRESENT",             // Read current memory state
        "PARADOX",             // Signal temporal inconsistency
    ];
}

impl DomainParser for TemporalParser {
    fn keywords(&self) -> &'static [&'static str] {
        Self::KEYWORDS
    }

    fn parse<'a>(&self, keyword: &str, ctx: &mut ParseContext<'a>) -> Result<Stmt, String> {
        match keyword {
            "ORACLE" | "READ" => ctx.emit_op(OpCode::Oracle),
            "PROPHECY" | "WRITE" => ctx.emit_op(OpCode::Prophecy),
            "PRESENT" => ctx.emit_op(OpCode::PresentRead),
            "PARADOX" => ctx.emit_op(OpCode::Paradox),
            _ => Err(format!("Unknown temporal operation: {}", keyword)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_keywords() {
        let parser = TemporalParser;
        assert!(parser.keywords().contains(&"ORACLE"));
        assert!(parser.keywords().contains(&"PROPHECY"));
        assert!(parser.keywords().contains(&"PRESENT"));
        assert!(parser.keywords().contains(&"PARADOX"));
    }

    #[test]
    fn test_parse_oracle() {
        let parser = TemporalParser;
        let mut depth = 1;
        let mut ctx = ParseContext::new(&mut depth, super::super::domain::emit_op_helper);
        let result = parser.parse("ORACLE", &mut ctx);
        assert!(result.is_ok());
        // ORACLE: consumes 1 (address), produces 1 (value)
        assert_eq!(depth, 1);
    }

    #[test]
    fn test_parse_prophecy() {
        let parser = TemporalParser;
        let mut depth = 2;
        let mut ctx = ParseContext::new(&mut depth, super::super::domain::emit_op_helper);
        let result = parser.parse("PROPHECY", &mut ctx);
        assert!(result.is_ok());
        // PROPHECY: consumes 2 (value, address), produces 0
        assert_eq!(depth, 0);
    }
}
