//! Arithmetic and comparison operations domain parser.
//!
//! Handles arithmetic (ADD, SUB, MUL, DIV, MOD, etc.), bitwise (NOT, AND, OR, etc.),
//! and comparison (EQ, LT, GT, etc.) keywords.

use crate::ast::{OpCode, Stmt};
use super::domain::{DomainParser, ParseContext};

/// Parser for arithmetic, bitwise, and comparison operations.
pub struct ArithmeticParser;

impl ArithmeticParser {
    pub const KEYWORDS: &'static [&'static str] = &[
        // Arithmetic
        "ADD", "+", "SUB", "-", "MUL", "*", "DIV", "/", "MOD", "%",
        "NEG", "ABS", "MIN", "MAX", "SIGN", "SIGNUM",
        // Bitwise
        "NOT", "~", "AND", "&", "OR", "|", "XOR", "^", "SHL", "<<", "SHR", ">>",
        // Comparison (unsigned)
        "EQ", "==", "NEQ", "!=", "LT", "<", "GT", ">", "LTE", "<=", "GTE", ">=",
        // Comparison (signed)
        "SLT", "SGT", "SLTE", "SGTE",
    ];
}

impl DomainParser for ArithmeticParser {
    fn keywords(&self) -> &'static [&'static str] {
        Self::KEYWORDS
    }

    fn parse<'a>(&self, keyword: &str, ctx: &mut ParseContext<'a>) -> Result<Stmt, String> {
        match keyword {
            // Arithmetic
            "ADD" | "+" => ctx.emit_op(OpCode::Add),
            "SUB" | "-" => ctx.emit_op(OpCode::Sub),
            "MUL" | "*" => ctx.emit_op(OpCode::Mul),
            "DIV" | "/" => ctx.emit_op(OpCode::Div),
            "MOD" | "%" => ctx.emit_op(OpCode::Mod),
            "NEG" => ctx.emit_op(OpCode::Neg),
            "ABS" => ctx.emit_op(OpCode::Abs),
            "MIN" => ctx.emit_op(OpCode::Min),
            "MAX" => ctx.emit_op(OpCode::Max),
            "SIGN" | "SIGNUM" => ctx.emit_op(OpCode::Sign),

            // Bitwise
            "NOT" | "~" => ctx.emit_op(OpCode::Not),
            "AND" | "&" => ctx.emit_op(OpCode::And),
            "OR" | "|" => ctx.emit_op(OpCode::Or),
            "XOR" | "^" => ctx.emit_op(OpCode::Xor),
            "SHL" | "<<" => ctx.emit_op(OpCode::Shl),
            "SHR" | ">>" => ctx.emit_op(OpCode::Shr),

            // Comparison (unsigned)
            "EQ" | "==" => ctx.emit_op(OpCode::Eq),
            "NEQ" | "!=" => ctx.emit_op(OpCode::Neq),
            "LT" | "<" => ctx.emit_op(OpCode::Lt),
            "GT" | ">" => ctx.emit_op(OpCode::Gt),
            "LTE" | "<=" => ctx.emit_op(OpCode::Lte),
            "GTE" | ">=" => ctx.emit_op(OpCode::Gte),

            // Comparison (signed)
            "SLT" => ctx.emit_op(OpCode::Slt),
            "SGT" => ctx.emit_op(OpCode::Sgt),
            "SLTE" => ctx.emit_op(OpCode::Slte),
            "SGTE" => ctx.emit_op(OpCode::Sgte),

            _ => Err(format!("Unknown arithmetic operation: {}", keyword)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arithmetic_keywords() {
        let parser = ArithmeticParser;
        assert!(parser.keywords().contains(&"ADD"));
        assert!(parser.keywords().contains(&"+"));
        assert!(parser.keywords().contains(&"EQ"));
    }

    #[test]
    fn test_parse_add() {
        let parser = ArithmeticParser;
        let mut depth = 2;
        let mut ctx = ParseContext::new(&mut depth, super::super::domain::emit_op_helper);
        let result = parser.parse("ADD", &mut ctx);
        assert!(result.is_ok());
        assert_eq!(depth, 1); // ADD consumes 2, produces 1
    }

    #[test]
    fn test_parse_neg() {
        let parser = ArithmeticParser;
        let mut depth = 1;
        let mut ctx = ParseContext::new(&mut depth, super::super::domain::emit_op_helper);
        let result = parser.parse("NEG", &mut ctx);
        assert!(result.is_ok());
        assert_eq!(depth, 1); // NEG consumes 1, produces 1
    }
}
