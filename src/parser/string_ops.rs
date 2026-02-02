//! String operations domain parser.
//!
//! Handles string manipulation operations: STR_REV, STR_CAT, STR_SPLIT.

use crate::ast::{OpCode, Stmt};
use super::domain::{DomainParser, ParseContext};

/// Parser for string operations.
pub struct StringOpsParser;

impl StringOpsParser {
    pub const KEYWORDS: &'static [&'static str] = &[
        "STR_REV",
        "STR_CAT", "CONCAT",
        "STR_SPLIT", "SPLIT",
    ];
}

impl DomainParser for StringOpsParser {
    fn keywords(&self) -> &'static [&'static str] {
        Self::KEYWORDS
    }

    fn parse<'a>(&self, keyword: &str, ctx: &mut ParseContext<'a>) -> Result<Stmt, String> {
        match keyword {
            "STR_REV" => ctx.emit_op(OpCode::StrRev),
            "STR_CAT" | "CONCAT" => ctx.emit_op(OpCode::StrCat),
            "STR_SPLIT" | "SPLIT" => ctx.emit_op(OpCode::StrSplit),
            _ => Err(format!("Unknown string operation: {}", keyword)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_keywords() {
        let parser = StringOpsParser;
        assert!(parser.keywords().contains(&"STR_REV"));
        assert!(parser.keywords().contains(&"CONCAT"));
    }
}
