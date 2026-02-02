//! Abstract Syntax Tree for OUROCHRONOS.
//!
//! This module provides the AST representation and utilities for traversal.
//!
//! # Types
//!
//! - `OpCode`: All 79+ operations available in OUROCHRONOS
//! - `Stmt`: Statement variants (operations, control flow, etc.)
//! - `Program`: Complete program with statements and procedures
//! - `Procedure`: Named callable unit
//! - `Effect`: Declared side effects for FFI
//!
//! # Visitor Pattern
//!
//! The `Visitor` trait enables AST traversal for analysis and transformation.

// Re-export all AST types from the main ast module
pub use crate::ast::{OpCode, Stmt, Program, Procedure, Effect};

/// AST visitor trait for traversing and analyzing programs.
///
/// Implement this trait to walk the AST and perform analysis or transformation.
pub trait Visitor {
    /// Result type for visit operations.
    type Result: Default;

    /// Visit a program (entry point).
    fn visit_program(&mut self, program: &Program) -> Self::Result {
        for stmt in &program.body {
            self.visit_stmt(stmt);
        }
        for proc in &program.procedures {
            self.visit_procedure(proc);
        }
        Self::Result::default()
    }

    /// Visit a statement.
    fn visit_stmt(&mut self, stmt: &Stmt) -> Self::Result {
        match stmt {
            Stmt::Push(_) => self.visit_push(stmt),
            Stmt::Op(op) => self.visit_op(*op),
            Stmt::If { then_branch, else_branch } => {
                for s in then_branch {
                    self.visit_stmt(s);
                }
                if let Some(else_stmts) = else_branch {
                    for s in else_stmts {
                        self.visit_stmt(s);
                    }
                }
                Self::Result::default()
            }
            Stmt::While { cond, body } => {
                for s in cond {
                    self.visit_stmt(s);
                }
                for s in body {
                    self.visit_stmt(s);
                }
                Self::Result::default()
            }
            Stmt::Block(stmts) => {
                for s in stmts {
                    self.visit_stmt(s);
                }
                Self::Result::default()
            }
            Stmt::Call { .. } => Self::Result::default(),
            Stmt::Match { cases, default } => {
                for (_, body) in cases {
                    for s in body {
                        self.visit_stmt(s);
                    }
                }
                if let Some(default_stmts) = default {
                    for s in default_stmts {
                        self.visit_stmt(s);
                    }
                }
                Self::Result::default()
            }
            Stmt::TemporalScope { body, .. } => {
                for s in body {
                    self.visit_stmt(s);
                }
                Self::Result::default()
            }
        }
    }

    /// Visit a push statement.
    fn visit_push(&mut self, _stmt: &Stmt) -> Self::Result {
        Self::Result::default()
    }

    /// Visit an operation.
    fn visit_op(&mut self, _op: OpCode) -> Self::Result {
        Self::Result::default()
    }

    /// Visit a procedure definition.
    fn visit_procedure(&mut self, _proc: &Procedure) -> Self::Result {
        Self::Result::default()
    }
}

/// OpCode metadata for analysis.
impl OpCode {
    /// Get the number of stack values consumed (popped).
    pub fn pops(&self) -> usize {
        use OpCode::*;
        match self {
            Nop | Depth | Input | Halt => 0,
            Pop | Not | Neg | Abs | Sign | Dup | Output | Emit | Assert |
            VecNew | HashNew | SetNew | VecLen | SetLen => 1,
            Add | Sub | Mul | Div | Mod | And | Or | Xor | Shl | Shr |
            Eq | Neq | Lt | Gt | Lte | Gte | Slt | Sgt | Slte | Sgte |
            Min | Max | Swap | Over | Store | VecPush | VecPop | VecGet |
            HashGet | HashDel | HashHas | SetAdd | SetDel | SetHas => 2,
            Rot | VecSet | HashPut | Pick | Roll | Reverse => 3,
            _ => 0, // Conservative default
        }
    }

    /// Get the number of stack values produced (pushed).
    pub fn pushes(&self) -> usize {
        use OpCode::*;
        match self {
            Nop | Pop | Halt | Assert | Store | Output | Emit |
            VecPush | VecSet | HashPut | HashDel | SetAdd | SetDel => 0,
            Add | Sub | Mul | Div | Mod | And | Or | Xor | Shl | Shr |
            Not | Neg | Abs | Sign | Min | Max |
            Eq | Neq | Lt | Gt | Lte | Gte | Slt | Sgt | Slte | Sgte |
            Depth | Index | Input | Oracle | PresentRead |
            VecNew | HashNew | SetNew | VecPop | VecGet | VecLen |
            HashGet | HashHas | SetHas | SetLen | Prophecy => 1,
            Dup | Over | Swap => 2, // Conceptually preserves/reorders
            Rot => 3, // Reorders 3
            _ => 0, // Conservative default
        }
    }

    /// Check if this operation is temporal (ORACLE/PROPHECY/PRESENT_READ).
    pub fn is_temporal(&self) -> bool {
        matches!(self, OpCode::Oracle | OpCode::Prophecy | OpCode::PresentRead | OpCode::Paradox)
    }

    /// Check if this operation has side effects.
    pub fn has_side_effects(&self) -> bool {
        use OpCode::*;
        matches!(self,
            Output | Emit | Assert | Store | Prophecy |
            FileOpen | FileClose | FileRead | FileWrite | FileSeek | FileFlush |
            TcpConnect | SocketSend | SocketRecv | SocketClose |
            BufferNew | BufferWriteByte | BufferFree
        )
    }

    /// Get a short description of this operation.
    pub fn description(&self) -> &'static str {
        use OpCode::*;
        match self {
            Nop => "no operation",
            Halt => "halt execution",
            Pop => "discard top of stack",
            Dup => "duplicate top of stack",
            Swap => "swap top two elements",
            Over => "copy second element to top",
            Rot => "rotate top three elements",
            Add => "add top two elements",
            Sub => "subtract top from second",
            Mul => "multiply top two elements",
            Div => "divide second by top",
            Mod => "modulo second by top",
            Oracle => "read from temporal memory",
            Prophecy => "write to temporal memory",
            PresentRead => "read from present memory",
            _ => "operation",
        }
    }
}

/// Count operations in a program.
pub fn count_ops(program: &Program) -> usize {
    struct OpCounter(usize);
    impl Visitor for OpCounter {
        type Result = ();
        fn visit_op(&mut self, _op: OpCode) {
            self.0 += 1;
        }
    }
    let mut counter = OpCounter(0);
    counter.visit_program(program);
    counter.0
}

/// Check if a program contains temporal operations.
pub fn has_temporal_ops(program: &Program) -> bool {
    struct TemporalChecker(bool);
    impl Visitor for TemporalChecker {
        type Result = ();
        fn visit_op(&mut self, op: OpCode) {
            if op.is_temporal() {
                self.0 = true;
            }
        }
    }
    let mut checker = TemporalChecker(false);
    checker.visit_program(program);
    checker.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opcode_metadata() {
        assert_eq!(OpCode::Add.pops(), 2);
        assert_eq!(OpCode::Add.pushes(), 1);
        assert!(OpCode::Oracle.is_temporal());
        assert!(!OpCode::Add.is_temporal());
        assert!(OpCode::Output.has_side_effects());
        assert!(!OpCode::Add.has_side_effects());
    }

    #[test]
    fn test_has_temporal_ops() {
        let pure_program = Program {
            body: vec![
                Stmt::Push(crate::core::Value::new(1)),
                Stmt::Push(crate::core::Value::new(2)),
                Stmt::Op(OpCode::Add),
            ],
            procedures: vec![],
            quotes: vec![],
            ffi_declarations: vec![],
        };
        assert!(!has_temporal_ops(&pure_program));

        let temporal_program = Program {
            body: vec![
                Stmt::Push(crate::core::Value::new(0)),
                Stmt::Op(OpCode::Oracle),
            ],
            procedures: vec![],
            quotes: vec![],
            ffi_declarations: vec![],
        };
        assert!(has_temporal_ops(&temporal_program));
    }
}
