//! Stack operations domain parser.
//!
//! Handles stack manipulation keywords: NOP, HALT, POP, DUP, SWAP, OVER,
//! ROT, DEPTH, PICK, ROLL, REVERSE, EXEC, DIP, KEEP, BI, REC.

use crate::ast::{OpCode, Stmt};
use super::domain::{DomainParser, ParseContext};

/// Parser for stack manipulation operations.
pub struct StackOpsParser;

impl StackOpsParser {
    pub const KEYWORDS: &'static [&'static str] = &[
        "NOP", "HALT", "POP", "DROP", "DUP", "SWAP", "OVER", "ROT",
        "DEPTH", "PICK", "PEEK", "ROLL", "REVERSE", "REV",
        "EXEC", "CALL", "DIP", "KEEP", "BI", "CLEAVE", "REC",
    ];
}

impl DomainParser for StackOpsParser {
    fn keywords(&self) -> &'static [&'static str] {
        Self::KEYWORDS
    }

    fn parse<'a>(&self, keyword: &str, ctx: &mut ParseContext<'a>) -> Result<Stmt, String> {
        match keyword {
            "NOP" => ctx.emit_op(OpCode::Nop),
            "HALT" => ctx.emit_op(OpCode::Halt),
            "POP" | "DROP" => ctx.emit_op(OpCode::Pop),
            "DUP" => ctx.emit_op(OpCode::Dup),
            "SWAP" => ctx.emit_op(OpCode::Swap),
            "OVER" => ctx.emit_op(OpCode::Over),
            "ROT" => ctx.emit_op(OpCode::Rot),
            "DEPTH" => ctx.emit_op(OpCode::Depth),
            "PICK" | "PEEK" => ctx.emit_op(OpCode::Pick),
            "ROLL" => ctx.emit_op(OpCode::Roll),
            "REVERSE" | "REV" => ctx.emit_op(OpCode::Reverse),
            "EXEC" | "CALL" => ctx.emit_op(OpCode::Exec),
            "DIP" => ctx.emit_op(OpCode::Dip),
            "KEEP" => ctx.emit_op(OpCode::Keep),
            "BI" | "CLEAVE" => ctx.emit_op(OpCode::Bi),
            "REC" => ctx.emit_op(OpCode::Rec),
            _ => Err(format!("Unknown stack operation: {}", keyword)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_ops_keywords() {
        let parser = StackOpsParser;
        assert!(parser.keywords().contains(&"DUP"));
        assert!(parser.keywords().contains(&"SWAP"));
        assert!(parser.keywords().contains(&"DROP"));
    }

    #[test]
    fn test_parse_dup() {
        let parser = StackOpsParser;
        let mut depth = 1;
        let mut ctx = ParseContext::new(&mut depth, super::super::domain::emit_op_helper);
        let result = parser.parse("DUP", &mut ctx);
        assert!(result.is_ok());
        assert_eq!(depth, 2); // DUP increases stack depth by 1
    }

    #[test]
    fn test_parse_pop() {
        let parser = StackOpsParser;
        let mut depth = 2;
        let mut ctx = ParseContext::new(&mut depth, super::super::domain::emit_op_helper);
        let result = parser.parse("POP", &mut ctx);
        assert!(result.is_ok());
        assert_eq!(depth, 1); // POP decreases stack depth by 1
    }
}
