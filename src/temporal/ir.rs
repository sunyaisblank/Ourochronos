//! Typed finite temporal intermediate representation.
//!
//! The ordinary VM is intentionally general-purpose.  Global temporal
//! verification needs a smaller contract: one epoch must lower to a total,
//! effect-free transformation over finite words and finite memory.  This IR is
//! that contract.  It is solver-independent and keeps Boolean, word, and
//! memory expressions distinct so an unsupported source construct cannot be
//! smuggled into a verifier as an untyped string.

use crate::ast::{OpCode, Procedure, Program, Stmt};
use crate::core::BoundsPolicy;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Write;

/// Index of an expression in a [`TemporalIr`] arena.
pub type ExprId = usize;

/// The three sorts used by the finite temporal core.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrType {
    Bool,
    Word64,
    Memory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WordUnaryOp {
    Neg,
    AbsSigned,
    SignSigned,
    LogicalNot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WordBinaryOp {
    Add,
    Sub,
    Mul,
    UnsignedDivOrZero,
    UnsignedRemOrZero,
    MinUnsigned,
    MaxUnsigned,
    BitAnd,
    BitOr,
    BitXor,
    ShiftLeftModulo64,
    ShiftRightModulo64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Eq,
    Ne,
    Ult,
    Ugt,
    Ule,
    Uge,
    Slt,
    Sgt,
    Sle,
    Sge,
}

/// A typed expression node.  Children always precede their parent in the IR
/// arena, which makes validation and backend emission deterministic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrExprKind {
    BoolConst(bool),
    WordConst(u64),
    Anamnesis,
    ZeroMemory,
    Select {
        memory: ExprId,
        address: ExprId,
    },
    Store {
        memory: ExprId,
        address: ExprId,
        value: ExprId,
    },
    WordUnary {
        op: WordUnaryOp,
        value: ExprId,
    },
    WordBinary {
        op: WordBinaryOp,
        lhs: ExprId,
        rhs: ExprId,
    },
    Compare {
        op: CompareOp,
        lhs: ExprId,
        rhs: ExprId,
    },
    BoolNot(ExprId),
    BoolAnd(ExprId, ExprId),
    BoolOr(ExprId, ExprId),
    Ite {
        condition: ExprId,
        when_true: ExprId,
        when_false: ExprId,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrExpr {
    pub ty: IrType,
    pub kind: IrExprKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationKind {
    Value,
    Character,
}

/// One buffered observable and the path condition under which it occurs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IrObservation {
    pub guard: ExprId,
    pub value: ExprId,
    pub kind: ObservationKind,
}

/// Whether UNSAT is a proof about the full lowered epoch or only the bounded
/// executions represented by the IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrCompleteness {
    Complete,
    BoundedLoops {
        loop_count: usize,
        unroll_limit: usize,
    },
}

impl IrCompleteness {
    pub fn proves_global_unsat(self) -> bool {
        matches!(self, Self::Complete)
    }
}

/// Solver-independent finite representation of one Ourochronos epoch.
#[derive(Debug, Clone)]
pub struct TemporalIr {
    pub expressions: Vec<IrExpr>,
    pub anamnesis: ExprId,
    pub final_memory: ExprId,
    pub valid: ExprId,
    pub observations: Vec<IrObservation>,
    pub memory_cells: usize,
    pub bounds_policy: BoundsPolicy,
    pub completeness: IrCompleteness,
}

impl TemporalIr {
    pub fn expression(&self, id: ExprId) -> &IrExpr {
        &self.expressions[id]
    }

    /// Emit the IR as self-contained QF_ABV.  Backends may omit commands and
    /// feed the declarations/assertions to an in-process solver.
    pub fn to_smt2(&self, include_commands: bool) -> String {
        let mut out = String::new();
        writeln!(out, "; Ourochronos typed temporal IR").unwrap();
        writeln!(out, "; completeness: {:?}", self.completeness).unwrap();
        writeln!(out, "(set-logic QF_ABV)").unwrap();
        writeln!(out, "(set-option :produce-models true)").unwrap();
        writeln!(
            out,
            "(declare-const anamnesis {})",
            smt_sort(IrType::Memory)
        )
        .unwrap();

        for (id, expr) in self.expressions.iter().enumerate() {
            writeln!(
                out,
                "(define-fun e{} () {} {})",
                id,
                smt_sort(expr.ty),
                self.smt_expr(&expr.kind)
            )
            .unwrap();
        }

        writeln!(out, "(assert e{})", self.valid).unwrap();
        writeln!(
            out,
            "(assert (= e{} e{}))",
            self.final_memory, self.anamnesis
        )
        .unwrap();
        if include_commands {
            writeln!(out, "(check-sat)").unwrap();
            writeln!(out, "(get-model)").unwrap();
        }
        out
    }

    fn smt_expr(&self, kind: &IrExprKind) -> String {
        let e = |id: ExprId| format!("e{}", id);
        match *kind {
            IrExprKind::BoolConst(value) => value.to_string(),
            IrExprKind::WordConst(value) => format!("(_ bv{} 64)", value),
            IrExprKind::Anamnesis => "anamnesis".to_string(),
            IrExprKind::ZeroMemory => {
                "((as const (Array (_ BitVec 64) (_ BitVec 64))) (_ bv0 64))".to_string()
            }
            IrExprKind::Select { memory, address } => format!(
                "(select {} {})",
                e(memory),
                self.normalized_address(&e(address))
            ),
            IrExprKind::Store { memory, address, value } => format!(
                "(store {} {} {})",
                e(memory),
                self.normalized_address(&e(address)),
                e(value)
            ),
            IrExprKind::WordUnary { op, value } => match op {
                WordUnaryOp::Neg => format!("(bvneg {})", e(value)),
                WordUnaryOp::AbsSigned => format!(
                    "(ite (bvslt {} (_ bv0 64)) (bvneg {}) {})",
                    e(value), e(value), e(value)
                ),
                WordUnaryOp::SignSigned => format!(
                    "(ite (= {} (_ bv0 64)) (_ bv0 64) (ite (bvslt {} (_ bv0 64)) (_ bv{} 64) (_ bv1 64)))",
                    e(value), e(value), u64::MAX
                ),
                WordUnaryOp::LogicalNot => format!(
                    "(ite (= {} (_ bv0 64)) (_ bv1 64) (_ bv0 64))",
                    e(value)
                ),
            },
            IrExprKind::WordBinary { op, lhs, rhs } => {
                let lhs = e(lhs);
                let rhs = e(rhs);
                match op {
                    WordBinaryOp::Add => format!("(bvadd {} {})", lhs, rhs),
                    WordBinaryOp::Sub => format!("(bvsub {} {})", lhs, rhs),
                    WordBinaryOp::Mul => format!("(bvmul {} {})", lhs, rhs),
                    WordBinaryOp::UnsignedDivOrZero => format!(
                        "(ite (= {} (_ bv0 64)) (_ bv0 64) (bvudiv {} {}))",
                        rhs, lhs, rhs
                    ),
                    WordBinaryOp::UnsignedRemOrZero => format!(
                        "(ite (= {} (_ bv0 64)) (_ bv0 64) (bvurem {} {}))",
                        rhs, lhs, rhs
                    ),
                    WordBinaryOp::MinUnsigned => {
                        format!("(ite (bvult {} {}) {} {})", lhs, rhs, lhs, rhs)
                    }
                    WordBinaryOp::MaxUnsigned => {
                        format!("(ite (bvugt {} {}) {} {})", lhs, rhs, lhs, rhs)
                    }
                    WordBinaryOp::BitAnd => format!("(bvand {} {})", lhs, rhs),
                    WordBinaryOp::BitOr => format!("(bvor {} {})", lhs, rhs),
                    WordBinaryOp::BitXor => format!("(bvxor {} {})", lhs, rhs),
                    WordBinaryOp::ShiftLeftModulo64 => format!(
                        "(bvshl {} (bvurem {} (_ bv64 64)))",
                        lhs, rhs
                    ),
                    WordBinaryOp::ShiftRightModulo64 => format!(
                        "(bvlshr {} (bvurem {} (_ bv64 64)))",
                        lhs, rhs
                    ),
                }
            }
            IrExprKind::Compare { op, lhs, rhs } => {
                let lhs = e(lhs);
                let rhs = e(rhs);
                match op {
                    CompareOp::Eq => format!("(= {} {})", lhs, rhs),
                    CompareOp::Ne => format!("(not (= {} {}))", lhs, rhs),
                    CompareOp::Ult => format!("(bvult {} {})", lhs, rhs),
                    CompareOp::Ugt => format!("(bvugt {} {})", lhs, rhs),
                    CompareOp::Ule => format!("(bvule {} {})", lhs, rhs),
                    CompareOp::Uge => format!("(bvuge {} {})", lhs, rhs),
                    CompareOp::Slt => format!("(bvslt {} {})", lhs, rhs),
                    CompareOp::Sgt => format!("(bvsgt {} {})", lhs, rhs),
                    CompareOp::Sle => format!("(bvsle {} {})", lhs, rhs),
                    CompareOp::Sge => format!("(bvsge {} {})", lhs, rhs),
                }
            }
            IrExprKind::BoolNot(value) => format!("(not {})", e(value)),
            IrExprKind::BoolAnd(lhs, rhs) => format!("(and {} {})", e(lhs), e(rhs)),
            IrExprKind::BoolOr(lhs, rhs) => format!("(or {} {})", e(lhs), e(rhs)),
            IrExprKind::Ite { condition, when_true, when_false } => format!(
                "(ite {} {} {})",
                e(condition), e(when_true), e(when_false)
            ),
        }
    }

    fn normalized_address(&self, address: &str) -> String {
        let cells = self.memory_cells as u64;
        match self.bounds_policy {
            BoundsPolicy::Wrap => format!("(bvurem {} (_ bv{} 64))", address, cells),
            BoundsPolicy::Clamp => format!(
                "(ite (bvult {} (_ bv{} 64)) {} (_ bv{} 64))",
                address,
                cells,
                address,
                cells - 1
            ),
            BoundsPolicy::Error => address.to_string(),
        }
    }
}

fn smt_sort(ty: IrType) -> &'static str {
    match ty {
        IrType::Bool => "Bool",
        IrType::Word64 => "(_ BitVec 64)",
        IrType::Memory => "(Array (_ BitVec 64) (_ BitVec 64))",
    }
}

/// Why a program cannot enter the finite temporal verification core.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemporalIrError {
    pub message: String,
}

impl TemporalIrError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for TemporalIrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for TemporalIrError {}

#[derive(Debug, Clone, Copy)]
pub struct TemporalIrConfig {
    pub memory_cells: usize,
    pub loop_unroll_limit: usize,
    pub bounds_policy: BoundsPolicy,
}

impl Default for TemporalIrConfig {
    fn default() -> Self {
        Self {
            memory_cells: crate::core::MEMORY_SIZE,
            loop_unroll_limit: 10,
            bounds_policy: BoundsPolicy::Wrap,
        }
    }
}

#[derive(Clone)]
struct SymbolicState {
    stack: Vec<ExprId>,
    present: ExprId,
    valid: ExprId,
    stopped: bool,
}

#[derive(Clone, Copy)]
struct ActiveTemporalScope {
    base: u64,
    size: u64,
    cell_bits: u8,
}

/// Lowers supported source programs into the finite typed IR.
pub struct TemporalIrCompiler<'a> {
    config: TemporalIrConfig,
    expressions: Vec<IrExpr>,
    observations: Vec<IrObservation>,
    procedures: HashMap<&'a str, &'a Procedure>,
    call_stack: Vec<String>,
    loop_count: usize,
    anamnesis: ExprId,
    active_scope: Option<ActiveTemporalScope>,
}

impl<'a> TemporalIrCompiler<'a> {
    /// Whether an opcode has total deterministic lowering in the finite
    /// temporal IR. Structured control and procedure calls are checked
    /// separately by the region validator/compiler.
    pub fn supports_opcode(op: OpCode) -> bool {
        matches!(
            op,
            OpCode::Nop
                | OpCode::Halt
                | OpCode::Pop
                | OpCode::Dup
                | OpCode::Swap
                | OpCode::Over
                | OpCode::Rot
                | OpCode::Depth
                | OpCode::Add
                | OpCode::Sub
                | OpCode::Mul
                | OpCode::Div
                | OpCode::Mod
                | OpCode::Neg
                | OpCode::Abs
                | OpCode::Min
                | OpCode::Max
                | OpCode::Sign
                | OpCode::Not
                | OpCode::And
                | OpCode::Or
                | OpCode::Xor
                | OpCode::Shl
                | OpCode::Shr
                | OpCode::Eq
                | OpCode::Neq
                | OpCode::Lt
                | OpCode::Gt
                | OpCode::Lte
                | OpCode::Gte
                | OpCode::Slt
                | OpCode::Sgt
                | OpCode::Slte
                | OpCode::Sgte
                | OpCode::Oracle
                | OpCode::Prophecy
                | OpCode::PresentRead
                | OpCode::Paradox
                | OpCode::Index
                | OpCode::Store
                | OpCode::Output
                | OpCode::Emit
        )
    }

    pub fn compile(
        program: &'a Program,
        config: TemporalIrConfig,
    ) -> Result<TemporalIr, TemporalIrError> {
        if config.memory_cells == 0 {
            return Err(TemporalIrError::new(
                "temporal IR requires at least one memory cell",
            ));
        }
        if config.memory_cells > crate::core::MAX_DENSE_MEMORY_CELLS {
            return Err(TemporalIrError::new(format!(
                "temporal IR memory width {} exceeds dense result limit {}",
                config.memory_cells,
                crate::core::MAX_DENSE_MEMORY_CELLS
            )));
        }
        let procedures = program
            .procedures
            .iter()
            .map(|procedure| (procedure.name.as_str(), procedure))
            .collect();
        let mut compiler = Self {
            config,
            expressions: Vec::new(),
            observations: Vec::new(),
            procedures,
            call_stack: Vec::new(),
            loop_count: 0,
            anamnesis: 0,
            active_scope: None,
        };
        compiler.anamnesis = compiler.add(IrType::Memory, IrExprKind::Anamnesis);
        let zero_memory = compiler.add(IrType::Memory, IrExprKind::ZeroMemory);
        let valid = compiler.bool_const(true);
        let mut state = SymbolicState {
            stack: Vec::new(),
            present: zero_memory,
            valid,
            stopped: false,
        };
        compiler.lower_block(&program.body, &mut state)?;

        let completeness = if compiler.loop_count == 0 {
            IrCompleteness::Complete
        } else {
            IrCompleteness::BoundedLoops {
                loop_count: compiler.loop_count,
                unroll_limit: compiler.config.loop_unroll_limit,
            }
        };
        compiler.validate()?;
        Ok(TemporalIr {
            expressions: compiler.expressions,
            anamnesis: compiler.anamnesis,
            final_memory: state.present,
            valid: state.valid,
            observations: compiler.observations,
            memory_cells: compiler.config.memory_cells,
            bounds_policy: compiler.config.bounds_policy,
            completeness,
        })
    }

    fn add(&mut self, ty: IrType, kind: IrExprKind) -> ExprId {
        let id = self.expressions.len();
        self.expressions.push(IrExpr { ty, kind });
        id
    }

    fn bool_const(&mut self, value: bool) -> ExprId {
        self.add(IrType::Bool, IrExprKind::BoolConst(value))
    }

    fn word_const(&mut self, value: u64) -> ExprId {
        self.add(IrType::Word64, IrExprKind::WordConst(value))
    }

    fn is_bool_const(&self, id: ExprId, expected: bool) -> bool {
        matches!(self.expressions[id].kind, IrExprKind::BoolConst(value) if value == expected)
    }

    fn bool_not(&mut self, value: ExprId) -> ExprId {
        if self.is_bool_const(value, true) {
            return self.bool_const(false);
        }
        if self.is_bool_const(value, false) {
            return self.bool_const(true);
        }
        self.add(IrType::Bool, IrExprKind::BoolNot(value))
    }

    fn bool_and(&mut self, lhs: ExprId, rhs: ExprId) -> ExprId {
        if self.is_bool_const(lhs, false) || self.is_bool_const(rhs, false) {
            return self.bool_const(false);
        }
        if self.is_bool_const(lhs, true) {
            return rhs;
        }
        if self.is_bool_const(rhs, true) {
            return lhs;
        }
        self.add(IrType::Bool, IrExprKind::BoolAnd(lhs, rhs))
    }

    fn bool_or(&mut self, lhs: ExprId, rhs: ExprId) -> ExprId {
        if self.is_bool_const(lhs, true) || self.is_bool_const(rhs, true) {
            return self.bool_const(true);
        }
        if self.is_bool_const(lhs, false) {
            return rhs;
        }
        if self.is_bool_const(rhs, false) {
            return lhs;
        }
        self.add(IrType::Bool, IrExprKind::BoolOr(lhs, rhs))
    }

    fn ite(&mut self, condition: ExprId, when_true: ExprId, when_false: ExprId) -> ExprId {
        if self.is_bool_const(condition, true) {
            return when_true;
        }
        if self.is_bool_const(condition, false) {
            return when_false;
        }
        if when_true == when_false {
            return when_true;
        }
        let ty = self.expressions[when_true].ty;
        self.add(
            ty,
            IrExprKind::Ite {
                condition,
                when_true,
                when_false,
            },
        )
    }

    fn word_to_bool(&mut self, word: ExprId) -> ExprId {
        let zero = self.word_const(0);
        self.add(
            IrType::Bool,
            IrExprKind::Compare {
                op: CompareOp::Ne,
                lhs: word,
                rhs: zero,
            },
        )
    }

    fn lower_block(
        &mut self,
        stmts: &[Stmt],
        state: &mut SymbolicState,
    ) -> Result<(), TemporalIrError> {
        for stmt in stmts {
            if state.stopped {
                break;
            }
            self.lower_stmt(stmt, state)?;
        }
        Ok(())
    }

    fn lower_stmt(
        &mut self,
        stmt: &Stmt,
        state: &mut SymbolicState,
    ) -> Result<(), TemporalIrError> {
        match stmt {
            Stmt::Push(value) => state.stack.push(self.word_const(value.val)),
            Stmt::PushConstant { value, .. } => state.stack.push(self.word_const(*value)),
            Stmt::ReadTemporal { address, .. } => {
                let address = self.word_const(*address);
                state.stack.push(address);
                self.lower_op(OpCode::Oracle, state)?;
            }
            Stmt::PushQuote(id) => {
                return Err(TemporalIrError::new(format!(
                    "quotation reference {} has no typed lowering in the finite temporal IR",
                    id.as_u64()
                )));
            }
            Stmt::Op(op) => self.lower_op(*op, state)?,
            Stmt::Block(body) => self.lower_block(body, state)?,
            Stmt::Call { name } => self.lower_call(name, state)?,
            Stmt::If {
                then_branch,
                else_branch,
            } => {
                let condition_word = self.pop(state, "IF")?;
                let condition = self.word_to_bool(condition_word);
                let parent_valid = state.valid;

                let mut then_state = state.clone();
                then_state.valid = self.bool_and(parent_valid, condition);
                self.lower_block(then_branch, &mut then_state)?;

                let mut else_state = state.clone();
                let not_condition = self.bool_not(condition);
                else_state.valid = self.bool_and(parent_valid, not_condition);
                if let Some(else_branch) = else_branch {
                    self.lower_block(else_branch, &mut else_state)?;
                }
                *state = self.merge_states(condition, then_state, else_state, "IF")?;
            }
            Stmt::While { cond, body } => {
                self.loop_count += 1;
                *state =
                    self.lower_while(state.clone(), cond, body, self.config.loop_unroll_limit)?;
            }
            Stmt::TemporalScope {
                base,
                size,
                cell_bits,
                body,
            } => {
                if self.active_scope.is_some() {
                    return Err(TemporalIrError::new(
                        "nested TEMPORAL scopes are not supported by the finite temporal IR",
                    ));
                }
                validate_scope(*base, *size, *cell_bits, self.config.memory_cells)?;
                self.active_scope = Some(ActiveTemporalScope {
                    base: *base,
                    size: *size,
                    cell_bits: *cell_bits,
                });
                let result = self.lower_block(body, state);
                self.active_scope = None;
                result?;
            }
        }
        Ok(())
    }

    fn lower_call(&mut self, name: &str, state: &mut SymbolicState) -> Result<(), TemporalIrError> {
        if self.call_stack.iter().any(|active| active == name) {
            return Err(TemporalIrError::new(format!(
                "recursive procedure '{}' is not total in the finite temporal IR",
                name
            )));
        }
        let body = self
            .procedures
            .get(name)
            .ok_or_else(|| TemporalIrError::new(format!("unknown procedure '{}'", name)))?
            .body
            .clone();
        self.call_stack.push(name.to_string());
        let result = self.lower_block(&body, state);
        self.call_stack.pop();
        result
    }

    fn lower_while(
        &mut self,
        mut state: SymbolicState,
        cond: &[Stmt],
        body: &[Stmt],
        remaining: usize,
    ) -> Result<SymbolicState, TemporalIrError> {
        self.lower_block(cond, &mut state)?;
        if state.stopped {
            return Err(TemporalIrError::new(
                "path-dependent HALT in WHILE condition is unsupported",
            ));
        }
        let condition_word = self.pop(&mut state, "WHILE condition")?;
        let condition = self.word_to_bool(condition_word);

        let mut exit_state = state.clone();
        let not_condition = self.bool_not(condition);
        exit_state.valid = self.bool_and(exit_state.valid, not_condition);

        if remaining == 0 {
            // The path is represented only when the next condition is false.
            // Therefore SAT is a replayable witness, while UNSAT remains a
            // bounded result (recorded in IrCompleteness).
            return Ok(exit_state);
        }

        let mut body_state = state;
        body_state.valid = self.bool_and(body_state.valid, condition);
        self.lower_block(body, &mut body_state)?;
        if body_state.stopped {
            return Err(TemporalIrError::new(
                "HALT in a symbolic WHILE body is unsupported",
            ));
        }
        let continue_state = self.lower_while(body_state, cond, body, remaining - 1)?;
        self.merge_states(condition, continue_state, exit_state, "WHILE")
    }

    fn merge_states(
        &mut self,
        condition: ExprId,
        when_true: SymbolicState,
        when_false: SymbolicState,
        construct: &str,
    ) -> Result<SymbolicState, TemporalIrError> {
        let true_impossible = self.is_bool_const(when_true.valid, false);
        let false_impossible = self.is_bool_const(when_false.valid, false);
        let stack = if true_impossible {
            when_false.stack.clone()
        } else if false_impossible {
            when_true.stack.clone()
        } else {
            if when_true.stack.len() != when_false.stack.len() {
                return Err(TemporalIrError::new(format!(
                    "{} branches have different stack heights ({} and {})",
                    construct,
                    when_true.stack.len(),
                    when_false.stack.len()
                )));
            }
            when_true
                .stack
                .iter()
                .zip(&when_false.stack)
                .map(|(&left, &right)| self.ite(condition, left, right))
                .collect()
        };
        if when_true.stopped != when_false.stopped && !true_impossible && !false_impossible {
            return Err(TemporalIrError::new(format!(
                "{} contains path-dependent HALT",
                construct
            )));
        }
        let present = self.ite(condition, when_true.present, when_false.present);
        let valid = self.bool_or(when_true.valid, when_false.valid);
        Ok(SymbolicState {
            stack,
            present,
            valid,
            stopped: if true_impossible {
                when_false.stopped
            } else {
                when_true.stopped
            },
        })
    }

    fn pop(&self, state: &mut SymbolicState, operation: &str) -> Result<ExprId, TemporalIrError> {
        state.stack.pop().ok_or_else(|| {
            TemporalIrError::new(format!("stack underflow while lowering {}", operation))
        })
    }

    fn lower_op(&mut self, op: OpCode, state: &mut SymbolicState) -> Result<(), TemporalIrError> {
        match op {
            OpCode::Nop => {}
            OpCode::Halt => state.stopped = true,
            OpCode::Pop => {
                self.pop(state, op.name())?;
            }
            OpCode::Dup => {
                let value = self.pop(state, op.name())?;
                state.stack.push(value);
                state.stack.push(value);
            }
            OpCode::Swap => {
                let right = self.pop(state, op.name())?;
                let left = self.pop(state, op.name())?;
                state.stack.push(right);
                state.stack.push(left);
            }
            OpCode::Over => {
                if state.stack.len() < 2 {
                    return Err(TemporalIrError::new("stack underflow while lowering OVER"));
                }
                state.stack.push(state.stack[state.stack.len() - 2]);
            }
            OpCode::Rot => {
                if state.stack.len() < 3 {
                    return Err(TemporalIrError::new("stack underflow while lowering ROT"));
                }
                let index = state.stack.len() - 3;
                let value = state.stack.remove(index);
                state.stack.push(value);
            }
            OpCode::Depth => {
                let depth = self.word_const(state.stack.len() as u64);
                state.stack.push(depth);
            }
            OpCode::Add
            | OpCode::Sub
            | OpCode::Mul
            | OpCode::Div
            | OpCode::Mod
            | OpCode::Min
            | OpCode::Max
            | OpCode::And
            | OpCode::Or
            | OpCode::Xor
            | OpCode::Shl
            | OpCode::Shr => {
                let rhs = self.pop(state, op.name())?;
                let lhs = self.pop(state, op.name())?;
                let operation = match op {
                    OpCode::Add => WordBinaryOp::Add,
                    OpCode::Sub => WordBinaryOp::Sub,
                    OpCode::Mul => WordBinaryOp::Mul,
                    OpCode::Div => WordBinaryOp::UnsignedDivOrZero,
                    OpCode::Mod => WordBinaryOp::UnsignedRemOrZero,
                    OpCode::Min => WordBinaryOp::MinUnsigned,
                    OpCode::Max => WordBinaryOp::MaxUnsigned,
                    OpCode::And => WordBinaryOp::BitAnd,
                    OpCode::Or => WordBinaryOp::BitOr,
                    OpCode::Xor => WordBinaryOp::BitXor,
                    OpCode::Shl => WordBinaryOp::ShiftLeftModulo64,
                    OpCode::Shr => WordBinaryOp::ShiftRightModulo64,
                    _ => unreachable!(),
                };
                let result = self.add(
                    IrType::Word64,
                    IrExprKind::WordBinary {
                        op: operation,
                        lhs,
                        rhs,
                    },
                );
                state.stack.push(result);
            }
            OpCode::Neg | OpCode::Abs | OpCode::Sign | OpCode::Not => {
                let value = self.pop(state, op.name())?;
                let operation = match op {
                    OpCode::Neg => WordUnaryOp::Neg,
                    OpCode::Abs => WordUnaryOp::AbsSigned,
                    OpCode::Sign => WordUnaryOp::SignSigned,
                    OpCode::Not => WordUnaryOp::LogicalNot,
                    _ => unreachable!(),
                };
                let result = self.add(
                    IrType::Word64,
                    IrExprKind::WordUnary {
                        op: operation,
                        value,
                    },
                );
                state.stack.push(result);
            }
            OpCode::Eq
            | OpCode::Neq
            | OpCode::Lt
            | OpCode::Gt
            | OpCode::Lte
            | OpCode::Gte
            | OpCode::Slt
            | OpCode::Sgt
            | OpCode::Slte
            | OpCode::Sgte => {
                let rhs = self.pop(state, op.name())?;
                let lhs = self.pop(state, op.name())?;
                let operation = match op {
                    OpCode::Eq => CompareOp::Eq,
                    OpCode::Neq => CompareOp::Ne,
                    OpCode::Lt => CompareOp::Ult,
                    OpCode::Gt => CompareOp::Ugt,
                    OpCode::Lte => CompareOp::Ule,
                    OpCode::Gte => CompareOp::Uge,
                    OpCode::Slt => CompareOp::Slt,
                    OpCode::Sgt => CompareOp::Sgt,
                    OpCode::Slte => CompareOp::Sle,
                    OpCode::Sgte => CompareOp::Sge,
                    _ => unreachable!(),
                };
                let boolean = self.add(
                    IrType::Bool,
                    IrExprKind::Compare {
                        op: operation,
                        lhs,
                        rhs,
                    },
                );
                let one = self.word_const(1);
                let zero = self.word_const(0);
                let result = self.ite(boolean, one, zero);
                state.stack.push(result);
            }
            OpCode::Oracle => {
                let raw_address = self.pop(state, op.name())?;
                let address = self.resolve_address(state, raw_address);
                let value = self.add(
                    IrType::Word64,
                    IrExprKind::Select {
                        memory: self.anamnesis,
                        address,
                    },
                );
                state.stack.push(value);
            }
            OpCode::PresentRead => {
                let raw_address = self.pop(state, op.name())?;
                let address = self.resolve_address(state, raw_address);
                let value = self.add(
                    IrType::Word64,
                    IrExprKind::Select {
                        memory: state.present,
                        address,
                    },
                );
                state.stack.push(value);
            }
            OpCode::Prophecy => {
                let raw_address = self.pop(state, op.name())?;
                let value = self.pop(state, op.name())?;
                let address = self.resolve_address(state, raw_address);
                let value = self.truncate_to_scope(value);
                state.present = self.add(
                    IrType::Memory,
                    IrExprKind::Store {
                        memory: state.present,
                        address,
                        value,
                    },
                );
            }
            OpCode::Index => {
                let index = self.pop(state, op.name())?;
                let base = self.pop(state, op.name())?;
                let address = self.add(
                    IrType::Word64,
                    IrExprKind::WordBinary {
                        op: WordBinaryOp::Add,
                        lhs: base,
                        rhs: index,
                    },
                );
                let address = self.resolve_address(state, address);
                let value = self.add(
                    IrType::Word64,
                    IrExprKind::Select {
                        memory: state.present,
                        address,
                    },
                );
                state.stack.push(value);
            }
            OpCode::Store => {
                let index = self.pop(state, op.name())?;
                let base = self.pop(state, op.name())?;
                let value = self.pop(state, op.name())?;
                let address = self.add(
                    IrType::Word64,
                    IrExprKind::WordBinary {
                        op: WordBinaryOp::Add,
                        lhs: base,
                        rhs: index,
                    },
                );
                let address = self.resolve_address(state, address);
                let value = self.truncate_to_scope(value);
                state.present = self.add(
                    IrType::Memory,
                    IrExprKind::Store {
                        memory: state.present,
                        address,
                        value,
                    },
                );
            }
            OpCode::Output | OpCode::Emit => {
                let value = self.pop(state, op.name())?;
                self.observations.push(IrObservation {
                    guard: state.valid,
                    value,
                    kind: if op == OpCode::Emit {
                        ObservationKind::Character
                    } else {
                        ObservationKind::Value
                    },
                });
            }
            OpCode::Paradox => {
                state.valid = self.bool_const(false);
                state.stopped = true;
            }
            OpCode::Input => {
                return Err(TemporalIrError::new(
                    "INPUT requires an explicitly frozen input stream in global verification",
                ));
            }
            OpCode::Pick
            | OpCode::Roll
            | OpCode::Reverse
            | OpCode::Exec
            | OpCode::Dip
            | OpCode::Keep
            | OpCode::Bi
            | OpCode::Rec
            | OpCode::StrRev
            | OpCode::StrCat
            | OpCode::StrSplit
            | OpCode::Assert
            | OpCode::Pack
            | OpCode::Unpack
            | OpCode::VecNew
            | OpCode::VecPush
            | OpCode::VecPop
            | OpCode::VecGet
            | OpCode::VecSet
            | OpCode::VecLen
            | OpCode::HashNew
            | OpCode::HashPut
            | OpCode::HashGet
            | OpCode::HashDel
            | OpCode::HashHas
            | OpCode::HashLen
            | OpCode::SetNew
            | OpCode::SetAdd
            | OpCode::SetHas
            | OpCode::SetDel
            | OpCode::SetLen
            | OpCode::FFICall
            | OpCode::FFICallNamed
            | OpCode::FileOpen
            | OpCode::FileRead
            | OpCode::FileWrite
            | OpCode::FileSeek
            | OpCode::FileFlush
            | OpCode::FileClose
            | OpCode::FileExists
            | OpCode::FileSize
            | OpCode::BufferNew
            | OpCode::BufferFromStack
            | OpCode::BufferToStack
            | OpCode::BufferLen
            | OpCode::BufferReadByte
            | OpCode::BufferWriteByte
            | OpCode::BufferFree
            | OpCode::TcpConnect
            | OpCode::SocketSend
            | OpCode::SocketRecv
            | OpCode::SocketClose
            | OpCode::ProcExec
            | OpCode::Clock
            | OpCode::Sleep
            | OpCode::Random => {
                return Err(TemporalIrError::new(format!(
                    "{} is outside the finite temporal IR",
                    op.name()
                )));
            }
        }
        Ok(())
    }

    fn resolve_address(&mut self, state: &mut SymbolicState, address: ExprId) -> ExprId {
        let Some(scope) = self.active_scope else {
            self.require_address_in_bounds(state, address, self.config.memory_cells as u64);
            return address;
        };

        let local = match self.config.bounds_policy {
            BoundsPolicy::Wrap => {
                let size = self.word_const(scope.size);
                self.add(
                    IrType::Word64,
                    IrExprKind::WordBinary {
                        op: WordBinaryOp::UnsignedRemOrZero,
                        lhs: address,
                        rhs: size,
                    },
                )
            }
            BoundsPolicy::Clamp => {
                let size = self.word_const(scope.size);
                let in_bounds = self.add(
                    IrType::Bool,
                    IrExprKind::Compare {
                        op: CompareOp::Ult,
                        lhs: address,
                        rhs: size,
                    },
                );
                let last = self.word_const(scope.size - 1);
                self.ite(in_bounds, address, last)
            }
            BoundsPolicy::Error => {
                self.require_address_in_bounds(state, address, scope.size);
                address
            }
        };
        if scope.base == 0 {
            local
        } else {
            let base = self.word_const(scope.base);
            self.add(
                IrType::Word64,
                IrExprKind::WordBinary {
                    op: WordBinaryOp::Add,
                    lhs: base,
                    rhs: local,
                },
            )
        }
    }

    fn truncate_to_scope(&mut self, value: ExprId) -> ExprId {
        let Some(scope) = self.active_scope else {
            return value;
        };
        if scope.cell_bits == 64 {
            return value;
        }
        let mask = self.word_const((1u64 << scope.cell_bits) - 1);
        self.add(
            IrType::Word64,
            IrExprKind::WordBinary {
                op: WordBinaryOp::BitAnd,
                lhs: value,
                rhs: mask,
            },
        )
    }

    fn require_address_in_bounds(
        &mut self,
        state: &mut SymbolicState,
        address: ExprId,
        limit_value: u64,
    ) {
        if self.config.bounds_policy != BoundsPolicy::Error {
            return;
        }
        let limit = self.word_const(limit_value);
        let in_bounds = self.add(
            IrType::Bool,
            IrExprKind::Compare {
                op: CompareOp::Ult,
                lhs: address,
                rhs: limit,
            },
        );
        state.valid = self.bool_and(state.valid, in_bounds);
    }

    fn validate(&self) -> Result<(), TemporalIrError> {
        for (id, expression) in self.expressions.iter().enumerate() {
            let child = |child: ExprId, expected: IrType| -> Result<(), TemporalIrError> {
                if child >= id {
                    return Err(TemporalIrError::new(format!(
                        "IR expression e{} has non-prior child e{}",
                        id, child
                    )));
                }
                if self.expressions[child].ty != expected {
                    return Err(TemporalIrError::new(format!(
                        "IR expression e{} expected {:?} child e{}, found {:?}",
                        id, expected, child, self.expressions[child].ty
                    )));
                }
                Ok(())
            };
            match expression.kind {
                IrExprKind::BoolConst(_)
                | IrExprKind::WordConst(_)
                | IrExprKind::Anamnesis
                | IrExprKind::ZeroMemory => {}
                IrExprKind::Select { memory, address } => {
                    child(memory, IrType::Memory)?;
                    child(address, IrType::Word64)?;
                }
                IrExprKind::Store {
                    memory,
                    address,
                    value,
                } => {
                    child(memory, IrType::Memory)?;
                    child(address, IrType::Word64)?;
                    child(value, IrType::Word64)?;
                }
                IrExprKind::WordUnary { value, .. } => child(value, IrType::Word64)?,
                IrExprKind::WordBinary { lhs, rhs, .. } | IrExprKind::Compare { lhs, rhs, .. } => {
                    child(lhs, IrType::Word64)?;
                    child(rhs, IrType::Word64)?;
                }
                IrExprKind::BoolNot(value) => child(value, IrType::Bool)?,
                IrExprKind::BoolAnd(lhs, rhs) | IrExprKind::BoolOr(lhs, rhs) => {
                    child(lhs, IrType::Bool)?;
                    child(rhs, IrType::Bool)?;
                }
                IrExprKind::Ite {
                    condition,
                    when_true,
                    when_false,
                } => {
                    child(condition, IrType::Bool)?;
                    child(when_true, expression.ty)?;
                    child(when_false, expression.ty)?;
                }
            }
        }
        Ok(())
    }
}

fn validate_scope(
    base: u64,
    size: u64,
    cell_bits: u8,
    memory_cells: usize,
) -> Result<(), TemporalIrError> {
    if size == 0 {
        return Err(TemporalIrError::new(
            "TEMPORAL scope size must be greater than zero",
        ));
    }
    if !(1..=64).contains(&cell_bits) {
        return Err(TemporalIrError::new(
            "TEMPORAL scope bit width must be in 1..=64",
        ));
    }
    let end = base
        .checked_add(size)
        .ok_or_else(|| TemporalIrError::new("TEMPORAL scope address range overflows u64"))?;
    if end > memory_cells as u64 {
        return Err(TemporalIrError::new(format!(
            "TEMPORAL scope [{}..{}) exceeds configured memory width {}",
            base, end, memory_cells
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn compile(source: &str) -> TemporalIr {
        let program = parse(source).unwrap();
        TemporalIrCompiler::compile(&program, TemporalIrConfig::default()).unwrap()
    }

    #[test]
    fn logical_not_matches_vm_semantics() {
        let ir = compile("0 ORACLE NOT 0 PROPHECY");
        let smt = ir.to_smt2(true);
        assert!(smt.contains("(ite (= e"));
        assert!(!smt.contains("bvnot"));
    }

    #[test]
    fn explicit_paradox_becomes_an_unsatisfied_path() {
        let ir = compile("0 ORACLE DUP 42 EQ IF { 0 PROPHECY } ELSE { POP PARADOX }");
        assert!(!ir.expressions.is_empty());
        assert!(ir.to_smt2(false).contains("(assert e"));
    }

    #[test]
    fn nonrecursive_procedures_lower_without_ast_rewrite() {
        let ir = compile("PROCEDURE save_answer { 0 PROPHECY } 42 save_answer");
        assert_eq!(ir.completeness, IrCompleteness::Complete);
    }

    #[test]
    fn loops_are_explicitly_bounded_not_claimed_complete() {
        let ir = compile("0 ORACLE WHILE { DUP 3 LT } { 1 ADD } 0 PROPHECY");
        assert!(matches!(
            ir.completeness,
            IrCompleteness::BoundedLoops { .. }
        ));
    }

    #[test]
    fn stack_underflow_is_a_compile_error() {
        let program = parse("ADD").unwrap();
        let error = TemporalIrCompiler::compile(&program, TemporalIrConfig::default()).unwrap_err();
        assert!(error.to_string().contains("stack underflow"));
    }
}
