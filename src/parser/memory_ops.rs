//! Memory operations domain parser.
//!
//! Handles memory operation keywords: INDEX, STORE, PACK, UNPACK.
//! These provide computed memory access with bounds checking.

use super::domain::{DomainParser, ParseContext};
use crate::ast::{OpCode, Stmt};

/// Parser for memory operations.
pub struct MemoryOpsParser;

impl MemoryOpsParser {
    pub const KEYWORDS: &'static [&'static str] = &[
        "INDEX",  // Read from present[base + offset]
        "STORE",  // Write value to present[base + offset]
        "PACK",   // Pack N stack values into contiguous memory
        "UNPACK", // Read N contiguous memory cells onto stack
    ];
}

impl DomainParser for MemoryOpsParser {
    fn keywords(&self) -> &'static [&'static str] {
        Self::KEYWORDS
    }

    fn parse(&self, keyword: &str, ctx: &mut ParseContext<'_>) -> Result<Stmt, String> {
        match keyword {
            "INDEX" => ctx.emit_op(OpCode::Index),
            "STORE" => ctx.emit_op(OpCode::Store),
            "PACK" => ctx.emit_op(OpCode::Pack),
            "UNPACK" => ctx.emit_op(OpCode::Unpack),
            _ => Err(format!("Unknown memory operation: {}", keyword)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_ops_keywords() {
        let parser = MemoryOpsParser;
        assert!(parser.keywords().contains(&"INDEX"));
        assert!(parser.keywords().contains(&"STORE"));
        assert!(parser.keywords().contains(&"PACK"));
        assert!(parser.keywords().contains(&"UNPACK"));
    }

    #[test]
    fn test_parse_index() {
        let parser = MemoryOpsParser;
        let mut depth = 2;
        let mut ctx = ParseContext::new(&mut depth, super::super::domain::emit_op_helper);
        let result = parser.parse("INDEX", &mut ctx);
        assert!(result.is_ok());
        // INDEX: consumes 2 (base, index), produces 1 (value)
        assert_eq!(depth, 1);
    }

    #[test]
    fn test_parse_store() {
        let parser = MemoryOpsParser;
        let mut depth = 3;
        let mut ctx = ParseContext::new(&mut depth, super::super::domain::emit_op_helper);
        let result = parser.parse("STORE", &mut ctx);
        assert!(result.is_ok());
        // STORE: consumes 3 (value, base, index), produces 0
        assert_eq!(depth, 0);
    }
}
