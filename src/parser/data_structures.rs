//! Data structures domain parser.
//!
//! Handles vector, hash table, and set operations.

use crate::ast::{OpCode, Stmt};
use super::domain::{DomainParser, ParseContext};

/// Parser for data structure operations (Vec, Hash, Set).
pub struct DataStructuresParser;

impl DataStructuresParser {
    pub const KEYWORDS: &'static [&'static str] = &[
        // Vector operations
        "VEC_NEW", "VEC_PUSH", "VEC_POP", "VEC_GET", "VEC_SET", "VEC_LEN",
        // Hash table operations
        "HASH_NEW", "HASH_PUT", "HASH_GET", "HASH_DEL", "HASH_HAS", "HASH_LEN",
        // Set operations
        "SET_NEW", "SET_ADD", "SET_HAS", "SET_DEL", "SET_LEN",
    ];
}

impl DomainParser for DataStructuresParser {
    fn keywords(&self) -> &'static [&'static str] {
        Self::KEYWORDS
    }

    fn parse<'a>(&self, keyword: &str, ctx: &mut ParseContext<'a>) -> Result<Stmt, String> {
        match keyword {
            // Vector operations
            "VEC_NEW" => ctx.emit_op(OpCode::VecNew),
            "VEC_PUSH" => ctx.emit_op(OpCode::VecPush),
            "VEC_POP" => ctx.emit_op(OpCode::VecPop),
            "VEC_GET" => ctx.emit_op(OpCode::VecGet),
            "VEC_SET" => ctx.emit_op(OpCode::VecSet),
            "VEC_LEN" => ctx.emit_op(OpCode::VecLen),

            // Hash table operations
            "HASH_NEW" => ctx.emit_op(OpCode::HashNew),
            "HASH_PUT" => ctx.emit_op(OpCode::HashPut),
            "HASH_GET" => ctx.emit_op(OpCode::HashGet),
            "HASH_DEL" => ctx.emit_op(OpCode::HashDel),
            "HASH_HAS" => ctx.emit_op(OpCode::HashHas),
            "HASH_LEN" => ctx.emit_op(OpCode::HashLen),

            // Set operations
            "SET_NEW" => ctx.emit_op(OpCode::SetNew),
            "SET_ADD" => ctx.emit_op(OpCode::SetAdd),
            "SET_HAS" => ctx.emit_op(OpCode::SetHas),
            "SET_DEL" => ctx.emit_op(OpCode::SetDel),
            "SET_LEN" => ctx.emit_op(OpCode::SetLen),

            _ => Err(format!("Unknown data structure operation: {}", keyword)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_structure_keywords() {
        let parser = DataStructuresParser;
        assert!(parser.keywords().contains(&"VEC_NEW"));
        assert!(parser.keywords().contains(&"HASH_GET"));
        assert!(parser.keywords().contains(&"SET_ADD"));
    }

    #[test]
    fn test_parse_vec_new() {
        let parser = DataStructuresParser;
        let mut depth = 0;
        let mut ctx = ParseContext::new(&mut depth, super::super::domain::emit_op_helper);
        let result = parser.parse("VEC_NEW", &mut ctx);
        assert!(result.is_ok());
        assert_eq!(depth, 1); // VEC_NEW produces a handle
    }
}
