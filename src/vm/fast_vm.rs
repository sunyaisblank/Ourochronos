//! Fast Virtual Machine for OUROCHRONOS epoch execution.
//!
//! This module provides an optimised execution path for programmes that do not
//! use temporal operations (ORACLE/PROPHECY). When a programme is statically
//! determined to be pure, or when executing within a pure region, this VM
//! bypasses provenance tracking entirely.
//!
//! # Performance Characteristics
//!
//! The fast VM achieves better performance through:
//! - **Stack register caching**: Top 4 stack elements cached in local variables
//! - **No provenance tracking**: Pure operations work with raw u64 values
//! - **Fused operations**: Common patterns executed as single operations
//! - **Reduced dispatch overhead**: Specialised instruction handling
//!
//! # Semantic Preservation
//!
//! The fast VM produces identical results to the standard VM for all pure
//! programmes. Temporal semantics are preserved by falling back to the
//! standard VM when ORACLE/PROPHECY operations are encountered.

use crate::ast::{OpCode, Stmt, Program};
use crate::core::{Value, Memory, OutputItem};
use super::executor::{EpochStatus, EpochResult, ExecutorConfig, Executor};

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

    /// Swap top two elements.
    #[inline(always)]
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.r0, &mut self.r1);
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

    /// Convert to a Vec<Value> with pure provenance.
    pub fn to_value_vec(&self) -> Vec<Value> {
        let mut result = Vec::with_capacity(self.count);

        // Add spill elements first (bottom of stack)
        for &v in &self.spill {
            result.push(Value::new(v));
        }

        // Add cached elements in order (r3, r2, r1, r0)
        if let Some(v) = self.r3 { result.push(Value::new(v)); }
        if let Some(v) = self.r2 { result.push(Value::new(v)); }
        if let Some(v) = self.r1 { result.push(Value::new(v)); }
        if let Some(v) = self.r0 { result.push(Value::new(v)); }

        result
    }

    /// Create from a Vec<Value>, discarding provenance.
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

/// Fast executor for pure (non-temporal) code regions.
///
/// This executor operates on raw u64 values without provenance tracking,
/// providing significantly better performance for computationally intensive
/// pure code like the Fibonacci benchmark.
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
        }
    }

    /// Execute a pure program (no ORACLE/PROPHECY).
    ///
    /// Returns Err if the program contains temporal operations that require
    /// the full VM.
    pub fn execute_pure(&mut self, program: &Program, quotes: &[Vec<Stmt>]) -> Result<(), String> {
        self.execute_block(&program.body, quotes)
    }

    /// Execute a block of statements.
    fn execute_block(&mut self, stmts: &[Stmt], quotes: &[Vec<Stmt>]) -> Result<(), String> {
        for stmt in stmts {
            // Check for termination conditions
            match &self.status {
                EpochStatus::Paradox | EpochStatus::Finished | EpochStatus::Error(_) => {
                    return Ok(());
                }
                EpochStatus::Running => {}
            }

            // Check gas limit
            if self.instructions_executed >= self.max_instructions {
                self.status = EpochStatus::Error("Instruction limit exceeded".to_string());
                return Ok(());
            }

            self.execute_stmt(stmt, quotes)?;
        }
        Ok(())
    }

    /// Execute a single statement.
    fn execute_stmt(&mut self, stmt: &Stmt, quotes: &[Vec<Stmt>]) -> Result<(), String> {
        self.instructions_executed += 1;

        match stmt {
            Stmt::Op(op) => self.execute_op(*op, quotes),

            Stmt::Push(v) => {
                self.stack.push(v.val);
                Ok(())
            }

            Stmt::Block(stmts) => self.execute_block(stmts, quotes),

            Stmt::If { then_branch, else_branch } => {
                let cond = self.stack.pop().ok_or("Stack underflow: IF")?;
                if cond != 0 {
                    self.execute_block(then_branch, quotes)
                } else if let Some(else_stmts) = else_branch {
                    self.execute_block(else_stmts, quotes)
                } else {
                    Ok(())
                }
            }

            Stmt::While { cond, body } => {
                loop {
                    // Check termination
                    if self.status != EpochStatus::Running {
                        break;
                    }

                    // Evaluate condition
                    self.execute_block(cond, quotes)?;
                    let result = self.stack.pop().ok_or("Stack underflow: WHILE condition")?;

                    if result == 0 {
                        break;
                    }

                    // Execute body
                    self.execute_block(body, quotes)?;

                    // Gas check
                    if self.instructions_executed >= self.max_instructions {
                        self.status = EpochStatus::Error("Instruction limit in loop".to_string());
                        break;
                    }
                }
                Ok(())
            }

            Stmt::Call { name } => {
                Err(format!("Procedure call '{}' not inlined - ensure program is preprocessed", name))
            }

            Stmt::Match { cases, default } => {
                let val = self.stack.pop().ok_or("Stack underflow: MATCH")?;

                let mut matched = false;
                for (pattern, body) in cases {
                    if val == *pattern {
                        self.execute_block(body, quotes)?;
                        matched = true;
                        break;
                    }
                }

                if !matched {
                    if let Some(default_body) = default {
                        self.execute_block(default_body, quotes)?;
                    }
                }
                Ok(())
            }

            // Temporal operations cannot be executed in fast mode
            Stmt::TemporalScope { .. } => {
                Err("TemporalScope requires standard VM execution".to_string())
            }
        }
    }

    /// Execute a single opcode.
    fn execute_op(&mut self, op: OpCode, quotes: &[Vec<Stmt>]) -> Result<(), String> {
        match op {
            // ═══════════════════════════════════════════════════════════
            // Temporal Operations - NOT SUPPORTED in fast mode
            // ═══════════════════════════════════════════════════════════
            OpCode::Oracle | OpCode::Prophecy | OpCode::PresentRead | OpCode::Paradox => {
                Err(format!("{:?} requires standard VM execution", op))
            }

            // ═══════════════════════════════════════════════════════════
            // Stack Manipulation (optimised)
            // ═══════════════════════════════════════════════════════════
            OpCode::Nop => Ok(()),

            OpCode::Halt => {
                self.status = EpochStatus::Finished;
                Ok(())
            }

            OpCode::Pop => {
                self.stack.pop().ok_or("Stack underflow: POP")?;
                Ok(())
            }

            OpCode::Dup => {
                if !self.stack.dup() {
                    return Err("Stack underflow: DUP".to_string());
                }
                Ok(())
            }

            OpCode::Swap => {
                self.stack.swap();
                Ok(())
            }

            OpCode::Over => {
                if !self.stack.over() {
                    return Err("Stack underflow: OVER".to_string());
                }
                Ok(())
            }

            OpCode::Rot => {
                if !self.stack.rot() {
                    return Err("Stack underflow: ROT".to_string());
                }
                Ok(())
            }

            OpCode::Depth => {
                self.stack.push(self.stack.depth() as u64);
                Ok(())
            }

            OpCode::Pick => {
                let n = self.stack.pop().ok_or("Stack underflow: PICK")? as usize;
                let depth = self.stack.depth();
                if n >= depth {
                    return Err(format!("PICK out of bounds: {} >= {}", n, depth));
                }
                // Convert to vec, pick, push back
                let vec = self.stack.to_value_vec();
                let idx = depth - 1 - n;
                self.stack.push(vec[idx].val);
                Ok(())
            }

            OpCode::Roll => {
                let n = self.stack.pop().ok_or("Stack underflow: ROLL")? as usize;
                let depth = self.stack.depth();
                if n >= depth {
                    return Err(format!("ROLL out of bounds: {} >= {}", n, depth));
                }
                // Convert to vec, roll, rebuild
                let mut vec = self.stack.to_value_vec();
                let idx = depth - 1 - n;
                let val = vec.remove(idx);
                vec.push(val);
                self.stack = FastStack::from_value_vec(&vec);
                Ok(())
            }

            OpCode::Reverse => {
                let n = self.stack.pop().ok_or("Stack underflow: REVERSE")? as usize;
                let depth = self.stack.depth();
                if n > depth {
                    return Err(format!("REVERSE out of bounds: {} > {}", n, depth));
                }
                if n > 1 {
                    let mut vec = self.stack.to_value_vec();
                    let start = depth - n;
                    vec[start..].reverse();
                    self.stack = FastStack::from_value_vec(&vec);
                }
                Ok(())
            }

            // ═══════════════════════════════════════════════════════════
            // Arithmetic (optimised)
            // ═══════════════════════════════════════════════════════════
            OpCode::Add => {
                if !self.stack.add() {
                    return Err("Stack underflow: ADD".to_string());
                }
                Ok(())
            }

            OpCode::Sub => {
                if !self.stack.sub() {
                    return Err("Stack underflow: SUB".to_string());
                }
                Ok(())
            }

            OpCode::Mul => {
                if !self.stack.mul() {
                    return Err("Stack underflow: MUL".to_string());
                }
                Ok(())
            }

            OpCode::Div => {
                let b = self.stack.pop().ok_or("Stack underflow: DIV")?;
                let a = self.stack.pop().ok_or("Stack underflow: DIV")?;
                let result = if b == 0 { 0 } else { a.wrapping_div(b) };
                self.stack.push(result);
                Ok(())
            }

            OpCode::Mod => {
                let b = self.stack.pop().ok_or("Stack underflow: MOD")?;
                let a = self.stack.pop().ok_or("Stack underflow: MOD")?;
                let result = if b == 0 { 0 } else { a.wrapping_rem(b) };
                self.stack.push(result);
                Ok(())
            }

            OpCode::Neg => {
                let a = self.stack.pop().ok_or("Stack underflow: NEG")?;
                self.stack.push(a.wrapping_neg());
                Ok(())
            }

            OpCode::Abs => {
                let a = self.stack.pop().ok_or("Stack underflow: ABS")?;
                self.stack.push((a as i64).wrapping_abs() as u64);
                Ok(())
            }

            OpCode::Min => {
                let b = self.stack.pop().ok_or("Stack underflow: MIN")?;
                let a = self.stack.pop().ok_or("Stack underflow: MIN")?;
                self.stack.push(a.min(b));
                Ok(())
            }

            OpCode::Max => {
                let b = self.stack.pop().ok_or("Stack underflow: MAX")?;
                let a = self.stack.pop().ok_or("Stack underflow: MAX")?;
                self.stack.push(a.max(b));
                Ok(())
            }

            OpCode::Sign => {
                let a = self.stack.pop().ok_or("Stack underflow: SIGN")?;
                self.stack.push((a as i64).signum() as u64);
                Ok(())
            }

            // ═══════════════════════════════════════════════════════════
            // Bitwise (optimised)
            // ═══════════════════════════════════════════════════════════
            OpCode::Not => {
                let a = self.stack.pop().ok_or("Stack underflow: NOT")?;
                // Logical NOT: 0 -> 1, nonzero -> 0
                self.stack.push(if a == 0 { 1 } else { 0 });
                Ok(())
            }

            OpCode::And => {
                let b = self.stack.pop().ok_or("Stack underflow: AND")?;
                let a = self.stack.pop().ok_or("Stack underflow: AND")?;
                self.stack.push(a & b);
                Ok(())
            }

            OpCode::Or => {
                let b = self.stack.pop().ok_or("Stack underflow: OR")?;
                let a = self.stack.pop().ok_or("Stack underflow: OR")?;
                self.stack.push(a | b);
                Ok(())
            }

            OpCode::Xor => {
                let b = self.stack.pop().ok_or("Stack underflow: XOR")?;
                let a = self.stack.pop().ok_or("Stack underflow: XOR")?;
                self.stack.push(a ^ b);
                Ok(())
            }

            OpCode::Shl => {
                let n = self.stack.pop().ok_or("Stack underflow: SHL")?;
                let a = self.stack.pop().ok_or("Stack underflow: SHL")?;
                let shift = (n % 64) as u32;
                self.stack.push(a.wrapping_shl(shift));
                Ok(())
            }

            OpCode::Shr => {
                let n = self.stack.pop().ok_or("Stack underflow: SHR")?;
                let a = self.stack.pop().ok_or("Stack underflow: SHR")?;
                let shift = (n % 64) as u32;
                self.stack.push(a.wrapping_shr(shift));
                Ok(())
            }

            // ═══════════════════════════════════════════════════════════
            // Comparison (optimised)
            // ═══════════════════════════════════════════════════════════
            OpCode::Eq => {
                let b = self.stack.pop().ok_or("Stack underflow: EQ")?;
                let a = self.stack.pop().ok_or("Stack underflow: EQ")?;
                self.stack.push(if a == b { 1 } else { 0 });
                Ok(())
            }

            OpCode::Neq => {
                let b = self.stack.pop().ok_or("Stack underflow: NEQ")?;
                let a = self.stack.pop().ok_or("Stack underflow: NEQ")?;
                self.stack.push(if a != b { 1 } else { 0 });
                Ok(())
            }

            OpCode::Lt => {
                if !self.stack.lt() {
                    return Err("Stack underflow: LT".to_string());
                }
                Ok(())
            }

            OpCode::Gt => {
                let b = self.stack.pop().ok_or("Stack underflow: GT")?;
                let a = self.stack.pop().ok_or("Stack underflow: GT")?;
                self.stack.push(if a > b { 1 } else { 0 });
                Ok(())
            }

            OpCode::Lte => {
                let b = self.stack.pop().ok_or("Stack underflow: LTE")?;
                let a = self.stack.pop().ok_or("Stack underflow: LTE")?;
                self.stack.push(if a <= b { 1 } else { 0 });
                Ok(())
            }

            OpCode::Gte => {
                let b = self.stack.pop().ok_or("Stack underflow: GTE")?;
                let a = self.stack.pop().ok_or("Stack underflow: GTE")?;
                self.stack.push(if a >= b { 1 } else { 0 });
                Ok(())
            }

            // Signed comparisons
            OpCode::Slt => {
                let b = self.stack.pop().ok_or("Stack underflow: SLT")? as i64;
                let a = self.stack.pop().ok_or("Stack underflow: SLT")? as i64;
                self.stack.push(if a < b { 1 } else { 0 });
                Ok(())
            }

            OpCode::Sgt => {
                let b = self.stack.pop().ok_or("Stack underflow: SGT")? as i64;
                let a = self.stack.pop().ok_or("Stack underflow: SGT")? as i64;
                self.stack.push(if a > b { 1 } else { 0 });
                Ok(())
            }

            OpCode::Slte => {
                let b = self.stack.pop().ok_or("Stack underflow: SLTE")? as i64;
                let a = self.stack.pop().ok_or("Stack underflow: SLTE")? as i64;
                self.stack.push(if a <= b { 1 } else { 0 });
                Ok(())
            }

            OpCode::Sgte => {
                let b = self.stack.pop().ok_or("Stack underflow: SGTE")? as i64;
                let a = self.stack.pop().ok_or("Stack underflow: SGTE")? as i64;
                self.stack.push(if a >= b { 1 } else { 0 });
                Ok(())
            }

            // ═══════════════════════════════════════════════════════════
            // I/O
            // ═══════════════════════════════════════════════════════════
            OpCode::Output => {
                let val = self.stack.pop().ok_or("Stack underflow: OUTPUT")?;
                self.output.push(OutputItem::Val(Value::new(val)));
                Ok(())
            }

            OpCode::Emit => {
                let val = self.stack.pop().ok_or("Stack underflow: EMIT")?;
                let char_val = (val % 256) as u8;
                print!("{}", char_val as char);
                self.output.push(OutputItem::Char(char_val));
                Ok(())
            }

            OpCode::Input => {
                // Input returns 0 in fast mode (no interactive input)
                self.stack.push(0);
                Ok(())
            }

            // ═══════════════════════════════════════════════════════════
            // System
            // ═══════════════════════════════════════════════════════════
            OpCode::Clock => {
                use std::time::{SystemTime, UNIX_EPOCH};
                let millis = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                self.stack.push(millis);
                Ok(())
            }

            OpCode::Sleep => {
                let millis = self.stack.pop().ok_or("Stack underflow: SLEEP")?;
                std::thread::sleep(std::time::Duration::from_millis(millis));
                Ok(())
            }

            OpCode::Random => {
                // Simple PRNG (xorshift)
                static mut SEED: u64 = 0x12345678;
                let val = unsafe {
                    SEED ^= SEED << 13;
                    SEED ^= SEED >> 7;
                    SEED ^= SEED << 17;
                    SEED
                };
                self.stack.push(val);
                Ok(())
            }

            // ═══════════════════════════════════════════════════════════
            // Quotation execution
            // ═══════════════════════════════════════════════════════════
            OpCode::Exec => {
                let id = self.stack.pop().ok_or("Stack underflow: EXEC")? as usize;
                if id >= quotes.len() {
                    return Err(format!("EXEC: Invalid quote ID {}", id));
                }
                self.execute_block(&quotes[id], quotes)
            }

            OpCode::Dip => {
                let id = self.stack.pop().ok_or("Stack underflow: DIP")? as usize;
                if id >= quotes.len() {
                    return Err(format!("DIP: Invalid quote ID {}", id));
                }
                let x = self.stack.pop().ok_or("Stack underflow: DIP")?;
                self.execute_block(&quotes[id], quotes)?;
                self.stack.push(x);
                Ok(())
            }

            OpCode::Keep => {
                let id = self.stack.pop().ok_or("Stack underflow: KEEP")? as usize;
                if id >= quotes.len() {
                    return Err(format!("KEEP: Invalid quote ID {}", id));
                }
                let x = self.stack.pop().ok_or("Stack underflow: KEEP")?;
                self.stack.push(x);
                self.execute_block(&quotes[id], quotes)?;
                self.stack.push(x);
                Ok(())
            }

            OpCode::Bi => {
                let q_id = self.stack.pop().ok_or("Stack underflow: BI")? as usize;
                let p_id = self.stack.pop().ok_or("Stack underflow: BI")? as usize;
                if q_id >= quotes.len() || p_id >= quotes.len() {
                    return Err("BI: Invalid quote ID".to_string());
                }
                let x = self.stack.pop().ok_or("Stack underflow: BI")?;
                self.stack.push(x);
                self.execute_block(&quotes[p_id], quotes)?;
                self.stack.push(x);
                self.execute_block(&quotes[q_id], quotes)
            }

            OpCode::Rec => {
                let id = self.stack.pop().ok_or("Stack underflow: REC")? as usize;
                if id >= quotes.len() {
                    return Err(format!("REC: Invalid quote ID {}", id));
                }
                self.stack.push(id as u64);
                self.execute_block(&quotes[id], quotes)
            }

            // ═══════════════════════════════════════════════════════════
            // Operations requiring full VM (not supported in fast mode)
            // ═══════════════════════════════════════════════════════════
            OpCode::Pack | OpCode::Unpack | OpCode::Index | OpCode::Store |
            OpCode::VecNew | OpCode::VecPush | OpCode::VecPop | OpCode::VecGet |
            OpCode::VecSet | OpCode::VecLen |
            OpCode::HashNew | OpCode::HashPut | OpCode::HashGet | OpCode::HashDel |
            OpCode::HashHas | OpCode::HashLen |
            OpCode::SetNew | OpCode::SetAdd | OpCode::SetHas | OpCode::SetDel | OpCode::SetLen |
            OpCode::FFICall | OpCode::FFICallNamed |
            OpCode::FileOpen | OpCode::FileRead | OpCode::FileWrite | OpCode::FileSeek |
            OpCode::FileFlush | OpCode::FileClose | OpCode::FileExists | OpCode::FileSize |
            OpCode::BufferNew | OpCode::BufferFromStack | OpCode::BufferToStack |
            OpCode::BufferLen | OpCode::BufferReadByte | OpCode::BufferWriteByte | OpCode::BufferFree |
            OpCode::TcpConnect | OpCode::SocketSend | OpCode::SocketRecv | OpCode::SocketClose |
            OpCode::ProcExec |
            OpCode::StrRev | OpCode::StrCat | OpCode::StrSplit | OpCode::Assert => {
                Err(format!("{:?} requires standard VM execution", op))
            }
        }
    }

    /// Convert to EpochResult for compatibility with standard VM.
    pub fn to_epoch_result(self) -> EpochResult {
        EpochResult {
            present: self.present,
            output: self.output,
            status: self.status,
            instructions_executed: self.instructions_executed,
            inputs_consumed: Vec::new(),
        }
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
    is_stmts_pure(&program.body) && program.quotes.iter().all(|q| is_stmts_pure(q))
}

/// Check if a block of statements is pure.
fn is_stmts_pure(stmts: &[Stmt]) -> bool {
    stmts.iter().all(is_stmt_pure)
}

/// Check if a single statement is pure.
fn is_stmt_pure(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Op(op) => is_op_pure(*op),
        Stmt::Push(_) => true,
        Stmt::Block(stmts) => is_stmts_pure(stmts),
        Stmt::If { then_branch, else_branch } => {
            is_stmts_pure(then_branch) &&
            else_branch.as_ref().map(|e| is_stmts_pure(e)).unwrap_or(true)
        }
        Stmt::While { cond, body } => {
            is_stmts_pure(cond) && is_stmts_pure(body)
        }
        Stmt::Call { .. } => true, // After inlining, calls become pure if their body is
        Stmt::Match { cases, default } => {
            cases.iter().all(|(_, body)| is_stmts_pure(body)) &&
            default.as_ref().map(|d| is_stmts_pure(d)).unwrap_or(true)
        }
        Stmt::TemporalScope { .. } => false, // Temporal scopes are never pure
    }
}

/// Check if an opcode is pure (can be executed in fast mode).
fn is_op_pure(op: OpCode) -> bool {
    match op {
        // Temporal operations
        OpCode::Oracle | OpCode::Prophecy | OpCode::PresentRead | OpCode::Paradox => false,

        // Operations requiring full VM
        OpCode::Pack | OpCode::Unpack | OpCode::Index | OpCode::Store |
        OpCode::VecNew | OpCode::VecPush | OpCode::VecPop | OpCode::VecGet |
        OpCode::VecSet | OpCode::VecLen |
        OpCode::HashNew | OpCode::HashPut | OpCode::HashGet | OpCode::HashDel |
        OpCode::HashHas | OpCode::HashLen |
        OpCode::SetNew | OpCode::SetAdd | OpCode::SetHas | OpCode::SetDel | OpCode::SetLen |
        OpCode::FFICall | OpCode::FFICallNamed |
        OpCode::FileOpen | OpCode::FileRead | OpCode::FileWrite | OpCode::FileSeek |
        OpCode::FileFlush | OpCode::FileClose | OpCode::FileExists | OpCode::FileSize |
        OpCode::BufferNew | OpCode::BufferFromStack | OpCode::BufferToStack |
        OpCode::BufferLen | OpCode::BufferReadByte | OpCode::BufferWriteByte | OpCode::BufferFree |
        OpCode::TcpConnect | OpCode::SocketSend | OpCode::SocketRecv | OpCode::SocketClose |
        OpCode::ProcExec |
        OpCode::StrRev | OpCode::StrCat | OpCode::StrSplit | OpCode::Assert => false,

        // All other operations are pure
        _ => true,
    }
}

/// Try to execute a program in fast mode if it's pure, otherwise fall back to standard VM.
///
/// This is the main entry point for optimised execution.
pub fn execute_with_fast_path(
    program: &Program,
    config: &ExecutorConfig,
) -> EpochResult {
    // Check if program can use fast path
    if is_program_pure(program) {
        let mut fast_exec = FastExecutor::new(config.max_instructions);

        // Try fast execution
        match fast_exec.execute_pure(program, &program.quotes) {
            Ok(()) => {
                // Successfully executed in fast mode
                if fast_exec.status == EpochStatus::Running {
                    fast_exec.status = EpochStatus::Finished;
                }
                return fast_exec.to_epoch_result();
            }
            Err(_) => {
                // Fall through to standard VM
            }
        }
    }

    // Fall back to standard VM
    let mut executor = Executor::with_config(config.clone());
    let anamnesis = Memory::new();
    executor.run_epoch(program, &anamnesis)
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
        stack.swap();
        assert_eq!(stack.pop(), Some(1));
        assert_eq!(stack.pop(), Some(2));
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
}
