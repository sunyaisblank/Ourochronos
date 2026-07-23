//! Pure-program compatibility facade over the validated bytecode VM.
//!
//! The former fast path recursively interpreted source AST and duplicated a
//! large opcode switch. That was not an acceptable optimized backend: its
//! stack, quotation, gas, and input behavior could drift from the executable
//! authority. `FastExecutor` now resolves and lowers the program to the same
//! validated [`BytecodeProgram`] used by normal execution, then delegates to
//! [`BytecodeVm`] through its immutable prevalidated dispatch path. That
//! removes repeated artifact scans while leaving instruction dispatch, gas,
//! errors, and observations identical. The public facade and experimental
//! [`FastStack`] remain for compatibility and benchmarking, but there is no
//! second language runtime in this module.

use super::executor::{EpochResult, EpochStatus, ExecutorConfig};
use crate::ast::{OpCode, Program, Stmt};
use crate::bytecode::BytecodeProgram;
use crate::bytecode_vm::{
    BytecodeVm, BytecodeVmConfig, BytecodeVmError, BytecodeVmStatus, PreparedBytecode,
};
use crate::core::{Memory, OutputItem, PagedMemory, Value};
use crate::hir::HirProgram;

// ═══════════════════════════════════════════════════════════════════════════
// Stack Register Cache
// ═══════════════════════════════════════════════════════════════════════════

/// A stack with the top elements cached in registers for fast access.
///
/// This avoids Vec operations for the most common stack manipulations.
/// The top 4 elements are cached; overflow spills to the backing vector.
#[derive(Debug, Clone)]
pub struct FastStack {
    /// Register 0: Top of stack (if present)
    r0: Option<u64>,
    /// Register 1: Second element (if present)
    r1: Option<u64>,
    /// Register 2: Third element (if present)
    r2: Option<u64>,
    /// Register 3: Fourth element (if present)
    r3: Option<u64>,
    /// Backing storage for elements beyond the cache
    spill: Vec<u64>,
    /// Total number of elements (cached + spilled)
    count: usize,
}

impl FastStack {
    /// Create an empty fast stack.
    #[inline]
    pub fn new() -> Self {
        Self {
            r0: None,
            r1: None,
            r2: None,
            r3: None,
            spill: Vec::new(),
            count: 0,
        }
    }

    /// Create a fast stack with pre-allocated spill capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            r0: None,
            r1: None,
            r2: None,
            r3: None,
            spill: Vec::with_capacity(capacity.saturating_sub(4)),
            count: 0,
        }
    }

    /// Push a value onto the stack.
    #[inline(always)]
    pub fn push(&mut self, val: u64) {
        // Shift existing values down and put new value in r0
        if let Some(v3) = self.r3 {
            self.spill.push(v3);
        }
        self.r3 = self.r2;
        self.r2 = self.r1;
        self.r1 = self.r0;
        self.r0 = Some(val);
        self.count += 1;
    }

    /// Pop a value from the stack.
    #[inline(always)]
    pub fn pop(&mut self) -> Option<u64> {
        if self.count == 0 {
            return None;
        }
        let result = self.r0;
        self.r0 = self.r1;
        self.r1 = self.r2;
        self.r2 = self.r3;
        self.r3 = self.spill.pop();
        self.count -= 1;
        result
    }

    /// Peek at the top of the stack without removing.
    #[inline(always)]
    pub fn peek(&self) -> Option<u64> {
        self.r0
    }

    /// Peek at the second element.
    #[inline(always)]
    pub fn peek_second(&self) -> Option<u64> {
        self.r1
    }

    /// Get the stack depth.
    #[inline(always)]
    pub fn depth(&self) -> usize {
        self.count
    }

    /// Check if stack is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Swap top two elements, returning false on stack underflow.
    #[inline(always)]
    pub fn swap(&mut self) -> bool {
        if self.count < 2 {
            return false;
        }
        std::mem::swap(&mut self.r0, &mut self.r1);
        true
    }

    /// Duplicate the top element.
    #[inline(always)]
    pub fn dup(&mut self) -> bool {
        if let Some(v) = self.r0 {
            self.push(v);
            true
        } else {
            false
        }
    }

    /// OVER: Copy second element to top.
    #[inline(always)]
    pub fn over(&mut self) -> bool {
        if let Some(v) = self.r1 {
            self.push(v);
            true
        } else {
            false
        }
    }

    /// ROT: Rotate top three elements (a b c -- b c a).
    #[inline(always)]
    pub fn rot(&mut self) -> bool {
        if self.count < 3 {
            return false;
        }
        let a = self.r2;
        self.r2 = self.r1;
        self.r1 = self.r0;
        self.r0 = a;
        true
    }

    /// Fused FIB_STEP: (a b -- b (a+b))
    /// This is the hot operation in Fibonacci computation.
    #[inline(always)]
    pub fn fib_step(&mut self) -> bool {
        match (self.r0, self.r1) {
            (Some(b), Some(a)) => {
                self.r1 = Some(b);
                self.r0 = Some(a.wrapping_add(b));
                true
            }
            _ => false,
        }
    }

    /// ADD: Pop two, push sum.
    #[inline(always)]
    pub fn add(&mut self) -> bool {
        match (self.r0, self.r1) {
            (Some(b), Some(a)) => {
                // Pop b (r0), result goes to where a was (r1 becomes new r0)
                self.r0 = self.r1;
                self.r1 = self.r2;
                self.r2 = self.r3;
                self.r3 = self.spill.pop();
                self.count -= 1;
                // Now add: pop a, push result
                let result = a.wrapping_add(b);
                self.r0 = Some(result);
                true
            }
            _ => false,
        }
    }

    /// SUB: Pop two, push difference.
    #[inline(always)]
    pub fn sub(&mut self) -> bool {
        match (self.r0, self.r1) {
            (Some(b), Some(a)) => {
                self.r0 = self.r1;
                self.r1 = self.r2;
                self.r2 = self.r3;
                self.r3 = self.spill.pop();
                self.count -= 1;
                let result = a.wrapping_sub(b);
                self.r0 = Some(result);
                true
            }
            _ => false,
        }
    }

    /// MUL: Pop two, push product.
    #[inline(always)]
    pub fn mul(&mut self) -> bool {
        match (self.r0, self.r1) {
            (Some(b), Some(a)) => {
                self.r0 = self.r1;
                self.r1 = self.r2;
                self.r2 = self.r3;
                self.r3 = self.spill.pop();
                self.count -= 1;
                let result = a.wrapping_mul(b);
                self.r0 = Some(result);
                true
            }
            _ => false,
        }
    }

    /// LT: Less than comparison.
    #[inline(always)]
    pub fn lt(&mut self) -> bool {
        match (self.r0, self.r1) {
            (Some(b), Some(a)) => {
                self.r0 = self.r1;
                self.r1 = self.r2;
                self.r2 = self.r3;
                self.r3 = self.spill.pop();
                self.count -= 1;
                let result = if a < b { 1 } else { 0 };
                self.r0 = Some(result);
                true
            }
            _ => false,
        }
    }

    /// Convert to a `Vec<Value>` with pure provenance.
    pub fn to_value_vec(&self) -> Vec<Value> {
        let mut result = Vec::with_capacity(self.count);

        // Add spill elements first (bottom of stack)
        for &v in &self.spill {
            result.push(Value::new(v));
        }

        // Add cached elements in order (r3, r2, r1, r0)
        if let Some(v) = self.r3 {
            result.push(Value::new(v));
        }
        if let Some(v) = self.r2 {
            result.push(Value::new(v));
        }
        if let Some(v) = self.r1 {
            result.push(Value::new(v));
        }
        if let Some(v) = self.r0 {
            result.push(Value::new(v));
        }

        result
    }

    /// Create from a `Vec<Value>`, discarding provenance.
    pub fn from_value_vec(values: &[Value]) -> Self {
        let mut stack = Self::new();
        for v in values {
            stack.push(v.val);
        }
        stack
    }
}

impl Default for FastStack {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Fast Executor
// ═══════════════════════════════════════════════════════════════════════════

/// Compatibility executor for pure (non-temporal) code regions.
///
/// Execution is authoritative validated bytecode; fields mirror the historic
/// API so callers can inspect stack, output, gas, and status.
pub struct FastExecutor {
    /// Stack with register caching
    pub stack: FastStack,
    /// Present memory (writes still need to be tracked for epoch results)
    pub present: Memory,
    /// Output buffer
    pub output: Vec<OutputItem>,
    /// Instruction count
    pub instructions_executed: u64,
    /// Maximum instructions allowed
    pub max_instructions: u64,
    /// Execution status
    pub status: EpochStatus,
    /// Scripted input values, consumed before falling back to stdin
    /// (mirrors ExecutorConfig::input).
    pub input: Vec<u64>,
    /// Position in the scripted input.
    input_cursor: usize,
    /// Exact input tape consumed by the bytecode authority.
    inputs_consumed: Vec<u64>,
}

impl FastExecutor {
    /// Create a new fast executor.
    pub fn new(max_instructions: u64) -> Self {
        Self {
            stack: FastStack::with_capacity(64),
            present: Memory::new(),
            output: Vec::new(),
            instructions_executed: 0,
            max_instructions,
            status: EpochStatus::Running,
            input: Vec::new(),
            input_cursor: 0,
            inputs_consumed: Vec::new(),
        }
    }

    /// Provide scripted input values, consumed by INPUT before stdin.
    pub fn with_input(mut self, input: Vec<u64>) -> Self {
        self.input = input;
        self.input_cursor = 0;
        self.inputs_consumed.clear();
        self
    }

    /// Execute a pure program (no ORACLE/PROPHECY).
    ///
    /// Returns Err if the program contains temporal operations that require
    /// the full VM.
    pub fn execute_pure(&mut self, program: &Program, quotes: &[Vec<Stmt>]) -> Result<(), String> {
        let _ = quotes;
        if !is_program_pure(program) {
            return Err("program requires the full temporal/effect runtime".to_string());
        }
        self.execute_bytecode(program)
    }

    fn execute_bytecode(&mut self, program: &Program) -> Result<(), String> {
        let hir = HirProgram::resolve(program)
            .map_err(|errors| format!("fast bytecode name resolution failed: {errors:?}"))?;
        let bytecode = BytecodeProgram::compile(&hir)
            .map_err(|error| format!("fast bytecode lowering failed: {error}"))?;
        let prepared = PreparedBytecode::new(bytecode)
            .map_err(|error| format!("fast bytecode preparation failed: {error}"))?;
        let anamnesis = PagedMemory::with_size(self.present.len())
            .map_err(|error| format!("fast bytecode memory failed: {error}"))?;
        let result = BytecodeVm::with_config(BytecodeVmConfig {
            max_instructions: self.max_instructions,
            input: self.input.clone(),
            allow_interactive_input: true,
            ..BytecodeVmConfig::default()
        })
        .run_prepared(&prepared, &anamnesis)
        .map_err(format_bytecode_error)?;

        self.stack = FastStack::from_value_vec(&result.stack);
        self.present = Memory::with_size(result.present.len());
        for (address, value) in result.present.iter_sparse() {
            self.present.write(address, value.clone());
        }
        self.output = result.output;
        self.instructions_executed = result.instructions_executed;
        self.input_cursor = result.inputs_consumed.len();
        self.inputs_consumed = result.inputs_consumed;
        self.status = match result.status {
            BytecodeVmStatus::Finished | BytecodeVmStatus::Halted => EpochStatus::Finished,
            BytecodeVmStatus::Paradox => EpochStatus::Paradox,
        };
        Ok(())
    }

    /// Convert to EpochResult for compatibility with standard VM.
    pub fn to_epoch_result(self) -> EpochResult {
        EpochResult {
            present: self.present,
            output: self.output,
            status: self.status,
            instructions_executed: self.instructions_executed,
            inputs_consumed: self.inputs_consumed,
        }
    }
}

fn format_bytecode_error(error: BytecodeVmError) -> String {
    match error {
        BytecodeVmError::StackUnderflow {
            operation,
            needed,
            available,
        } => format!(
            "Stack underflow: {operation} requires {needed} values, only {available} available"
        ),
        other => other.to_string(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Program Purity Analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Analyse a program to determine if it can be executed in fast (pure) mode.
///
/// A program is pure if it contains no temporal operations (ORACLE, PROPHECY,
/// PRESENT, PARADOX) and no operations that require full VM support (data
/// structures, FFI, file I/O, etc.).
pub fn is_program_pure(program: &Program) -> bool {
    is_stmts_pure(&program.body)
        && program.quotes.iter().all(|q| is_stmts_pure(q))
        && program
            .procedures
            .iter()
            .all(|procedure| is_stmts_pure(&procedure.body))
}

/// Check if a block of statements is pure.
fn is_stmts_pure(stmts: &[Stmt]) -> bool {
    stmts.iter().all(is_stmt_pure)
}

/// Check if a single statement is pure. Per-variant verdicts stay here;
/// the recursion into nested blocks comes from Stmt::child_blocks, so a
/// future statement kind cannot be walked incorrectly by this analysis.
fn is_stmt_pure(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Op(op) => is_op_pure(*op),
        Stmt::Push(_) => true,
        Stmt::PushConstant { .. } => true,
        Stmt::ReadTemporal { .. } => false,
        Stmt::PushQuote(_) => true,
        // Procedure bodies are checked once at the Program boundary above.
        Stmt::Call { .. } => true,
        Stmt::TemporalScope { .. } => false, // Temporal scopes are never pure
        Stmt::If { .. } | Stmt::While { .. } | Stmt::Block(_) => {
            stmt.child_blocks().into_iter().all(is_stmts_pure)
        }
    }
}

/// Check if an opcode is pure (can be executed in fast mode).
fn is_op_pure(op: OpCode) -> bool {
    match op {
        // Temporal operations
        OpCode::Oracle | OpCode::Prophecy | OpCode::PresentRead | OpCode::Paradox => false,

        // Operations requiring full VM
        OpCode::Pack
        | OpCode::Unpack
        | OpCode::Index
        | OpCode::Store
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
        | OpCode::Random
        | OpCode::StrRev
        | OpCode::StrCat
        | OpCode::StrSplit
        | OpCode::Assert => false,

        // All other operations are pure
        _ => true,
    }
}

/// Execute through the validated bytecode authority and prepared dispatcher.
///
/// This compatibility facade never falls back to the legacy AST executor: a
/// bytecode compile, preparation, capability, or runtime failure is returned
/// as an [`EpochStatus::Error`].
pub fn execute_with_fast_path(program: &Program, config: &ExecutorConfig) -> EpochResult {
    let mut fast_exec = FastExecutor::new(config.max_instructions).with_input(config.input.clone());
    if let Err(message) = fast_exec.execute_bytecode(program) {
        fast_exec.status = EpochStatus::Error(message);
    } else if fast_exec.status == EpochStatus::Running {
        fast_exec.status = EpochStatus::Finished;
    }
    fast_exec.to_epoch_result()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_stack_push_pop() {
        let mut stack = FastStack::new();
        stack.push(1);
        stack.push(2);
        stack.push(3);
        assert_eq!(stack.depth(), 3);
        assert_eq!(stack.pop(), Some(3));
        assert_eq!(stack.pop(), Some(2));
        assert_eq!(stack.pop(), Some(1));
        assert_eq!(stack.pop(), None);
    }

    #[test]
    fn test_fast_stack_swap() {
        let mut stack = FastStack::new();
        stack.push(1);
        stack.push(2);
        assert!(stack.swap());
        assert_eq!(stack.pop(), Some(1));
        assert_eq!(stack.pop(), Some(2));
    }

    #[test]
    fn test_fast_stack_swap_rejects_underflow_without_mutating() {
        let mut stack = FastStack::new();
        assert!(!stack.swap());
        assert!(stack.is_empty());

        stack.push(7);
        assert!(!stack.swap());
        assert_eq!(stack.depth(), 1);
        assert_eq!(stack.pop(), Some(7));
    }

    #[test]
    fn test_fast_stack_over() {
        let mut stack = FastStack::new();
        stack.push(1);
        stack.push(2);
        stack.over();
        assert_eq!(stack.depth(), 3);
        assert_eq!(stack.pop(), Some(1));
        assert_eq!(stack.pop(), Some(2));
        assert_eq!(stack.pop(), Some(1));
    }

    #[test]
    fn test_fast_stack_rot() {
        let mut stack = FastStack::new();
        stack.push(1); // a
        stack.push(2); // b
        stack.push(3); // c
                       // Stack: [1, 2, 3] (3 on top)
                       // ROT: a b c -> b c a
        stack.rot();
        assert_eq!(stack.pop(), Some(1)); // a
        assert_eq!(stack.pop(), Some(3)); // c
        assert_eq!(stack.pop(), Some(2)); // b
    }

    #[test]
    fn test_fast_stack_fib_step() {
        let mut stack = FastStack::new();
        // Start with fib(0)=0, fib(1)=1
        stack.push(0);
        stack.push(1);
        // fib_step: (a b -- b (a+b))
        assert!(stack.fib_step());
        // Should be: 1, 1
        assert_eq!(stack.peek(), Some(1)); // 0+1
        assert_eq!(stack.peek_second(), Some(1));

        assert!(stack.fib_step());
        // Should be: 1, 2
        assert_eq!(stack.peek(), Some(2)); // 1+1
        assert_eq!(stack.peek_second(), Some(1));

        assert!(stack.fib_step());
        // Should be: 2, 3
        assert_eq!(stack.peek(), Some(3)); // 1+2
        assert_eq!(stack.peek_second(), Some(2));
    }

    #[test]
    fn test_fast_stack_add() {
        let mut stack = FastStack::new();
        stack.push(10);
        stack.push(20);
        assert!(stack.add());
        assert_eq!(stack.pop(), Some(30));
    }

    #[test]
    fn test_fast_stack_spill() {
        let mut stack = FastStack::new();
        // Push more than 4 elements to test spill
        for i in 1..=10 {
            stack.push(i);
        }
        assert_eq!(stack.depth(), 10);

        // Pop and verify
        for i in (1..=10).rev() {
            assert_eq!(stack.pop(), Some(i));
        }
        assert_eq!(stack.depth(), 0);
    }

    #[test]
    fn test_is_op_pure() {
        assert!(is_op_pure(OpCode::Add));
        assert!(is_op_pure(OpCode::Swap));
        assert!(is_op_pure(OpCode::Dup));
        assert!(!is_op_pure(OpCode::Oracle));
        assert!(!is_op_pure(OpCode::Prophecy));
        assert!(!is_op_pure(OpCode::VecNew));
    }

    #[test]
    fn fast_facade_never_falls_back_after_bytecode_capability_denial() {
        let program = Program {
            body: vec![Stmt::Push(Value::new(10)), Stmt::Op(OpCode::Sleep)],
            ..Program::default()
        };
        let result = execute_with_fast_path(&program, &ExecutorConfig::default());
        assert!(matches!(
            result.status,
            EpochStatus::Error(ref message) if message.contains("sleep capability denied")
        ));
    }

    #[test]
    fn purity_includes_procedure_bodies() {
        let tokens = crate::parser::tokenize("PROCEDURE leak { 0 ORACLE } leak");
        let program = crate::parser::Parser::new(&tokens)
            .parse_program()
            .expect("procedure program parses");
        assert!(!is_program_pure(&program));

        let error = FastExecutor::new(10_000)
            .execute_pure(&program, &program.quotes)
            .expect_err("temporal procedure must not be admitted as pure");
        assert!(error.contains("full temporal/effect runtime"), "{error}");
    }
}
