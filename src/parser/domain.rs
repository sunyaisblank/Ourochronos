//! Domain parser trait for modular keyword handling.
//!
//! Each domain parser handles a specific category of keywords (stack ops,
//! arithmetic, temporal, etc.) to decompose the monolithic parse_word().

use crate::ast::{OpCode, Stmt};

/// A domain-specific parser that handles a category of keywords.
pub trait DomainParser {
    /// Returns the list of keywords this parser handles (uppercase).
    fn keywords(&self) -> &'static [&'static str];

    /// Parse the keyword and return the appropriate statement.
    /// The parser context provides access to tokens, stack depth, etc.
    fn parse<'a>(&self, keyword: &str, ctx: &mut ParseContext<'a>) -> Result<Stmt, String>;
}

/// Context passed to domain parsers for accessing parser state.
pub struct ParseContext<'a> {
    /// Current stack depth for tracking.
    pub stack_depth: &'a mut usize,
    /// Emit a simple opcode statement.
    emit_op_fn: fn(&mut usize, OpCode) -> Result<Stmt, String>,
}

impl<'a> ParseContext<'a> {
    pub fn new(
        stack_depth: &'a mut usize,
        emit_op_fn: fn(&mut usize, OpCode) -> Result<Stmt, String>,
    ) -> Self {
        Self {
            stack_depth,
            emit_op_fn,
        }
    }

    /// Emit an opcode and track its stack effect.
    pub fn emit_op(&mut self, op: OpCode) -> Result<Stmt, String> {
        (self.emit_op_fn)(self.stack_depth, op)
    }
}

/// Helper function for emit_op that can be passed to ParseContext.
pub fn emit_op_helper(stack_depth: &mut usize, op: OpCode) -> Result<Stmt, String> {
    let (inputs, outputs) = op.stack_effect();
    *stack_depth = stack_depth.saturating_sub(inputs);
    *stack_depth += outputs;
    Ok(Stmt::Op(op))
}
