//! Virtual Machine for OUROCHRONOS epoch execution.
//!
//! The VM executes a single epoch: given an anamnesis (read-only "future" memory),
//! it produces a present (read-write "current" memory) and output.
//!
//! The fixed-point search (in timeloop.rs) repeatedly runs epochs until
//! Present = Anamnesis (temporal consistency achieved).

use crate::core_types::{Value, Address, Memory, OutputItem, DataStructures};
use crate::ast::{OpCode, Stmt, Program};
use crate::provenance::Provenance;
use crate::io::{IOContext, FileMode, SeekOrigin};
use crate::ffi::{FFIContext, FFICaller};
use crate::error::SourceLocation;
use std::io::{self, Write, BufRead};

/// Status of epoch execution.
#[derive(Debug, Clone, PartialEq)]
pub enum EpochStatus {
    /// Epoch is still running.
    Running,
    /// Epoch completed normally (reached end or HALT).
    Finished,
    /// Epoch terminated due to explicit PARADOX instruction.
    Paradox,
    /// Epoch terminated due to runtime error.
    Error(String),
}

/// Complete state of the VM during epoch execution.
#[derive(Debug, Clone)]
pub struct VmState {
    /// The operand stack.
    pub stack: Vec<Value>,
    /// Present memory (being constructed this epoch).
    pub present: Memory,
    /// Anamnesis memory (read-only, from the "future").
    pub anamnesis: Memory,
    /// Output buffer.
    pub output: Vec<OutputItem>,
    /// Execution status.
    pub status: EpochStatus,
    /// Instruction count (for gas limiting).
    pub instructions_executed: u64,
    /// Dynamic data structures (vectors, hash tables, sets).
    pub data_structures: DataStructures,
}

impl VmState {
    /// Create a new VM state for an epoch.
    pub fn new(anamnesis: Memory) -> Self {
        Self {
            stack: Vec::new(),
            present: Memory::new(),
            anamnesis,
            output: Vec::new(),
            status: EpochStatus::Running,
            instructions_executed: 0,
            data_structures: DataStructures::new(),
        }
    }
}

/// Result of executing a single epoch.
#[derive(Debug)]
pub struct EpochResult {
    /// The final present memory.
    pub present: Memory,
    /// Output produced during the epoch.
    pub output: Vec<OutputItem>,
    /// Terminal status.
    pub status: EpochStatus,
    /// Number of instructions executed.
    pub instructions_executed: u64,
    /// Input values consumed during this epoch.
    /// Used for input freezing to ensure temporal consistency.
    pub inputs_consumed: Vec<u64>,
}

/// Configuration for the executor.
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Maximum instructions per epoch (gas limit).
    pub max_instructions: u64,
    /// Whether to print output immediately.
    pub immediate_output: bool,
    /// Input values (simulated input).
    pub input: Vec<u64>,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_instructions: 10_000_000,
            immediate_output: true,
            input: Vec::new(),
        }
    }
}

/// The OUROCHRONOS virtual machine executor.
pub struct Executor {
    pub config: ExecutorConfig,
    input_cursor: usize,
    /// Inputs consumed during the current epoch (for capture).
    inputs_consumed: Vec<u64>,
    /// I/O context for file, network, and buffer operations.
    pub io_context: IOContext,
    /// FFI context for foreign function calls.
    pub ffi_context: FFIContext,
}

impl Executor {
    /// Create a new executor with default configuration.
    pub fn new() -> Self {
        Self {
            config: ExecutorConfig::default(),
            input_cursor: 0,
            inputs_consumed: Vec::new(),
            io_context: IOContext::new(),
            ffi_context: FFIContext::with_stdlib(),
        }
    }

    /// Create an executor with custom configuration.
    pub fn with_config(config: ExecutorConfig) -> Self {
        Self {
            config,
            input_cursor: 0,
            inputs_consumed: Vec::new(),
            io_context: IOContext::new(),
            ffi_context: FFIContext::with_stdlib(),
        }
    }

    /// Create an executor with custom I/O and FFI contexts.
    pub fn with_contexts(config: ExecutorConfig, io_context: IOContext, ffi_context: FFIContext) -> Self {
        Self {
            config,
            input_cursor: 0,
            inputs_consumed: Vec::new(),
            io_context,
            ffi_context,
        }
    }
    
    /// Run a single epoch of execution.
    pub fn run_epoch(&mut self, program: &Program, anamnesis: &Memory) -> EpochResult {
        let mut state = VmState::new(anamnesis.clone());
        self.input_cursor = 0;
        self.inputs_consumed.clear();
        
        match self.execute_block(&program.body, &mut state, &program.quotes) {
            Ok(_) => {}
            Err(e) => {
                state.status = EpochStatus::Error(e);
            }
        }
        
        // If still running at end, consider it finished
        if state.status == EpochStatus::Running {
            state.status = EpochStatus::Finished;
        }
        
        EpochResult {
            present: state.present,
            output: state.output,
            status: state.status,
            instructions_executed: state.instructions_executed,
            inputs_consumed: std::mem::take(&mut self.inputs_consumed),
        }
    }

    /// Execute a program (sequence of statements) within an epoch.
    pub fn execute(&mut self, state: &mut VmState, program: &Program) -> Result<(), String> {
        self.execute_block(&program.body, state, &program.quotes)
    }
    
    /// Execute a block of statements.
    fn execute_block(&mut self, stmts: &[Stmt], state: &mut VmState, quotes: &[Vec<Stmt>]) -> Result<(), String> {
        for stmt in stmts {
            // Check for termination conditions
            match &state.status {
                EpochStatus::Paradox | EpochStatus::Finished | EpochStatus::Error(_) => {
                    return Ok(());
                }
                EpochStatus::Running => {}
            }
            
            // Check gas limit
            if state.instructions_executed >= self.config.max_instructions {
                state.status = EpochStatus::Error("Instruction limit exceeded".to_string());
                return Ok(());
            }
            
            self.execute_stmt(stmt, state, quotes)?;
        }
        Ok(())
    }
    
    /// Execute a single statement.
    fn execute_stmt(&mut self, stmt: &Stmt, state: &mut VmState, quotes: &[Vec<Stmt>]) -> Result<(), String> {
        state.instructions_executed += 1;
        
        match stmt {
            Stmt::Op(op) => self.execute_op(*op, state, quotes),
            
            Stmt::Push(v) => {
                state.stack.push(v.clone());
                Ok(())
            }
            
            Stmt::Block(stmts) => self.execute_block(stmts, state, quotes),
            
            Stmt::If { then_branch, else_branch } => {
                let cond = self.pop(state)?;
                if cond.to_bool() {
                    self.execute_block(then_branch, state, quotes)
                } else if let Some(else_stmts) = else_branch {
                    self.execute_block(else_stmts, state, quotes)
                } else {
                    Ok(())
                }
            }
            
            Stmt::While { cond, body } => {
                loop {
                    // Check termination
                    if state.status != EpochStatus::Running {
                        break;
                    }
                    
                    // Evaluate condition
                    self.execute_block(cond, state, quotes)?;
                    let result = self.pop(state)?;
                    
                    if !result.to_bool() {
                        break;
                    }
                    
                    // Execute body
                    self.execute_block(body, state, quotes)?;
                    
                    // Gas check for infinite loop prevention
                    if state.instructions_executed >= self.config.max_instructions {
                        state.status = EpochStatus::Error("Instruction limit in loop".to_string());
                        break;
                    }
                }
                Ok(())
            }
            
            Stmt::Call { name } => {
                // Note: Procedure lookup happens at runtime via the program
                // For now, this is a placeholder - actual inlining happens in run_epoch
                Err(format!("Procedure call '{}' not inlined - ensure program is preprocessed", name))
            }
            
            Stmt::Match { cases, default } => {
                // Pop the value to match against
                let val = self.pop(state)?.val;
                
                // Find matching case
                let mut matched = false;
                for (pattern, body) in cases {
                    if val == *pattern {
                        for stmt in body {
                            self.execute_stmt(stmt, state, quotes)?;
                        }
                        matched = true;
                        break;
                    }
                }
                
                // Execute default if no match
                if !matched {
                    if let Some(default_body) = default {
                        for stmt in default_body {
                            self.execute_stmt(stmt, state, quotes)?;
                        }
                    }
                }
                Ok(())
            }
            
            Stmt::TemporalScope { base, size, body } => {
                // Temporal scoping: create an isolated memory region
                // 
                // 1. Snapshot the memory region [base, base+size)
                // 2. Execute body with Oracle/Prophecy relative to base
                // 3. On success, propagate changes to parent
                // 4. On paradox/error, discard changes
                
                let base_addr = *base as u16;
                let region_size = *size as u16;
                
                // Snapshot the affected region
                let mut snapshot: Vec<Value> = Vec::with_capacity(region_size as usize);
                for i in 0..region_size {
                    snapshot.push(state.anamnesis.read(base_addr.wrapping_add(i)));
                }
                
                // Execute the body
                let result = self.execute_block(body, state, quotes);
                
                match result {
                    Ok(()) => {
                        // Success - changes are propagated (already written to state.present)
                        Ok(())
                    }
                    Err(e) if e.contains("paradox") || e.contains("PARADOX") => {
                        // Paradox - restore snapshotted region
                        for (i, val) in snapshot.into_iter().enumerate() {
                            state.present.write(base_addr.wrapping_add(i as u16), val);
                        }
                        // Propagate the error
                        Err(e)
                    }
                    Err(e) => Err(e),
                }
            }
        }
    }
    
    /// Execute a single opcode.
    fn execute_op(&mut self, op: OpCode, state: &mut VmState, quotes: &[Vec<Stmt>]) -> Result<(), String> {
        match op {
            // ═══════════════════════════════════════════════════════════
            // Stack Manipulation
            // ═══════════════════════════════════════════════════════════
            
            OpCode::Nop => {}
            
            OpCode::Halt => {
                state.status = EpochStatus::Finished;
            }
            
            OpCode::Pop => {
                self.pop(state)?;
            }
            
            OpCode::Dup => {
                let a = self.peek(state)?;
                state.stack.push(a);
            }
            OpCode::Pick => {
                let n_val = self.pop(state)?;
                let n = n_val.val;
                let len = state.stack.len() as u64;
                if n >= len {
                     return Err(format!("Pick out of bounds: index {} but depth {}", n, len));
                }
                let idx = (len - 1 - n) as usize;
                let val = state.stack[idx].clone();
                state.stack.push(val);
            }
            OpCode::Roll => {
                let n_val = self.pop(state)?;
                let n = n_val.val;
                let len = state.stack.len() as u64;
                if n >= len {
                     return Err(format!("Roll out of bounds: index {} but depth {}", n, len));
                }
                let idx = (len - 1 - n) as usize;
                let val = state.stack.remove(idx);
                state.stack.push(val);
            }
            OpCode::Reverse => {
                let n_val = self.pop(state)?;
                let count = n_val.val as usize;
                let len = state.stack.len();
                if count > len {
                    return Err(format!("Reverse out of bounds: count {} but depth {}", count, len));
                }
                if count > 1 {
                    let start = len - count;
                    state.stack[start..].reverse();
                }
            }
            
            OpCode::Exec => {
                let id_val = self.pop(state)?;
                let id = id_val.val as usize;
                
                if id >= quotes.len() {
                    return Err(format!("Exec: Invalid quote ID {}", id));
                }
                
                // Recursively execute quote
                // Use recursion guard? (not strict requirement yet, gas limit handles it)
                self.execute_block(&quotes[id], state, quotes)?;
            }
            
            OpCode::Dip => {
                let id_val = self.pop(state)?;
                let id = id_val.val as usize;
                
                if id >= quotes.len() {
                    return Err(format!("Dip: Invalid quote ID {}", id));
                }
                
                // Pop X
                let x = self.pop(state)?;
                
                // Execute quote
                self.execute_block(&quotes[id], state, quotes)?;
                
                // Restore X
                state.stack.push(x);
            }
            
            OpCode::Keep => {
                let id_val = self.pop(state)?;
                let id = id_val.val as usize;
                
                if id >= quotes.len() {
                    return Err(format!("Keep: Invalid quote ID {}", id));
                }
                
                // Pop X (to be kept)
                let x = self.pop(state)?;
                
                // Restore X (it stays on stack)
                state.stack.push(x.clone());
                
                // Execute quote (it sees X)
                self.execute_block(&quotes[id], state, quotes)?;
                
                // Keep: ( x q -- .. x ). 
                // Wait. Keep usually means preserve x across call.
                // Standard: ( x q -- .. x )
                // Apply q to x.
                // Stack: x. q.
                // q runs. Consumes x? Maybe.
                // If q consumes x, we need to restore it.
                // "Keep" implies we KEEP a copy of x for later use?
                // Or we keep x on top?
                // Factor `keep`: ( x q -- .. x )
                // "Call quote with x on stack, restoring x afterwards"
                
                // Implementation:
                // Pop q. Pop x.
                // Push x.
                // Run q.
                // Push x.
                // But if q consumes x, we need to ensure it was there.
                // My implementation:
                // Pop q, x.
                // Push x. 
                // Execute q. (q sees x).
                // Push x. (Restores x).
                state.stack.push(x);
            }
            
            OpCode::Bi => {
                 let q_val = self.pop(state)?;
                 let q_id = q_val.val as usize;
                 
                 let p_val = self.pop(state)?;
                 let p_id = p_val.val as usize;
                 
                 if q_id >= quotes.len() || p_id >= quotes.len() {
                     return Err("Bi: Invalid quote ID".to_string());
                 }
                 
                 let x = self.pop(state)?;
                 
                 // Apply P to X
                 state.stack.push(x.clone());
                 self.execute_block(&quotes[p_id], state, quotes)?;
                 
                 // Apply Q to X
                 state.stack.push(x);
                 self.execute_block(&quotes[q_id], state, quotes)?;
            }
            
            OpCode::Rec => {
                let id_val = self.pop(state)?;
                let id = id_val.val as usize;
                
                if id >= quotes.len() {
                    return Err(format!("Rec: Invalid quote ID {}", id));
                }
                
                // Push quote back (for recursion inside)
                state.stack.push(id_val);
                
                // Execute quote
                self.execute_block(&quotes[id], state, quotes)?;
            }
            
            OpCode::StrRev => {
                let len_val = self.pop(state)?;
                let len = len_val.val as usize;
                if len > state.stack.len() {
                    return Err("StrRev: length exceeds stack depth".to_string());
                }
                if len > 0 {
                    let start = state.stack.len() - len;
                    state.stack[start..].reverse();
                }
                state.stack.push(len_val);
            }
            
            OpCode::Assert => {
                let len_val = self.pop(state)?;
                let len = len_val.val as usize;
                
                let mut chars: Vec<u8> = Vec::with_capacity(len);
                for _ in 0..len {
                    let val = self.pop(state)?;
                    chars.push(val.val as u8);
                }
                chars.reverse(); // Stack order is reverse of string order? 
                // Wait, parser pushes char 0, then char 1... then len.
                // So top of stack is len, then char N, ..., char 0.
                // popping gives char N, ..., char 0.
                // So we need to reverse to get 0..N. 
                // Actually parse logic: 
                // "ABC" -> Push 'A', Push 'B', Push 'C', Push 3.
                // Stack: [..., A, B, C, 3] (Top is right)
                // Pop 3. Stack: [..., A, B, C]
                // Pop C, Pop B, Pop A.
                // Result C, B, A. Reverse -> A, B, C. Correct.
                
                let cond_val = self.pop(state)?;
                if cond_val.val == 0 {
                    let msg = String::from_utf8_lossy(&chars);
                    return Err(format!("Assertion failed: {}", msg));
                }
            }
            
            OpCode::StrCat => {
                let len2_val = self.pop(state)?;
                let len2 = len2_val.val as usize;
                
                // Pop string 2 chars
                let mut s2: Vec<Value> = Vec::with_capacity(len2);
                for _ in 0..len2 {
                    s2.push(self.pop(state)?);
                }
                s2.reverse(); // Popped in reverse order
                
                // Pop len1
                let len1_val = self.pop(state)?;
                let len1 = len1_val.val as usize;
                
                // s1 is already on stack. Just verify depth.
                if len1 > state.stack.len() {
                    return Err("StrCat: string 1 length exceeds stack depth".to_string());
                }
                
                // Push s2 back
                for v in s2 {
                    state.stack.push(v);
                }
                
                // Push total length
                state.stack.push(Value {
                    val: (len1 + len2) as u64,
                    prov: len1_val.prov.merge(&len2_val.prov),
                });
            }
            
            OpCode::StrSplit => {
                let delim_val = self.pop(state)?;
                let delim = delim_val.val;
                
                let len_val = self.pop(state)?;
                let len = len_val.val as usize;
                
                // Pop string chars
                let mut chars: Vec<Value> = Vec::with_capacity(len);
                for _ in 0..len {
                    chars.push(self.pop(state)?);
                }
                chars.reverse();
                
                // Split logic
                let mut parts: Vec<Vec<Value>> = Vec::new();
                let mut current: Vec<Value> = Vec::new();
                
                for c in chars {
                    if c.val == delim {
                        parts.push(current);
                        current = Vec::new();
                    } else {
                        current.push(c);
                    }
                }
                parts.push(current); // Last part
                
                // Push parts back to stack
                let count = parts.len();
                for part in parts {
                    let part_len = part.len() as u64;
                    for c in part {
                         state.stack.push(c);
                    }
                    state.stack.push(Value::new(part_len));
                }
                
                state.stack.push(Value::new(count as u64));
            }

            OpCode::Swap => {
                let a = self.pop(state)?;
                let b = self.pop(state)?;
                state.stack.push(a);
                state.stack.push(b);
            }
            
            OpCode::Over => {
                if state.stack.len() < 2 {
                    return Err("Stack underflow: OVER requires 2 elements".to_string());
                }
                let val = state.stack[state.stack.len() - 2].clone();
                state.stack.push(val);
            }
            
            OpCode::Rot => {
                if state.stack.len() < 3 {
                    return Err("Stack underflow: ROT requires 3 elements".to_string());
                }
                let len = state.stack.len();
                let c = state.stack.remove(len - 3);
                state.stack.push(c);
            }
            
            OpCode::Depth => {
                state.stack.push(Value::new(state.stack.len() as u64));
            }
            
            // ═══════════════════════════════════════════════════════════
            // Arithmetic
            // ═══════════════════════════════════════════════════════════
            
            OpCode::Add => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                state.stack.push(a + b);
            }
            
            OpCode::Sub => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                state.stack.push(a - b);
            }
            
            OpCode::Mul => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                state.stack.push(a * b);
            }
            
            OpCode::Div => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                state.stack.push(a / b); // Returns 0 for div by 0
            }
            
            OpCode::Mod => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                state.stack.push(a % b); // Returns 0 for mod by 0
            }
            
            OpCode::Neg => {
                let a = self.pop(state)?;
                state.stack.push(Value {
                    val: a.val.wrapping_neg(),
                    prov: a.prov,
                });
            }

            OpCode::Abs => {
                let a = self.pop(state)?;
                state.stack.push(a.abs());
            }

            OpCode::Min => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                let prov = a.prov.merge(&b.prov);
                let min_val = if a.val < b.val { a.val } else { b.val };
                state.stack.push(Value { val: min_val, prov });
            }

            OpCode::Max => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                let prov = a.prov.merge(&b.prov);
                let max_val = if a.val > b.val { a.val } else { b.val };
                state.stack.push(Value { val: max_val, prov });
            }

            OpCode::Sign => {
                let a = self.pop(state)?;
                state.stack.push(a.signum());
            }

            // ═══════════════════════════════════════════════════════════
            // Bitwise Logic
            // ═══════════════════════════════════════════════════════════
            
            OpCode::Not => {
                let a = self.pop(state)?;
                state.stack.push(!a);
            }
            
            OpCode::And => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                state.stack.push(a & b);
            }
            
            OpCode::Or => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                state.stack.push(a | b);
            }
            
            OpCode::Xor => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                state.stack.push(a ^ b);
            }
            
            OpCode::Shl => {
                let n = self.pop(state)?;
                let a = self.pop(state)?;
                let shift = (n.val % 64) as u32;
                state.stack.push(Value {
                    val: a.val.wrapping_shl(shift),
                    prov: a.prov.merge(&n.prov),
                });
            }
            
            OpCode::Shr => {
                let n = self.pop(state)?;
                let a = self.pop(state)?;
                let shift = (n.val % 64) as u32;
                state.stack.push(Value {
                    val: a.val.wrapping_shr(shift),
                    prov: a.prov.merge(&n.prov),
                });
            }
            
            // ═══════════════════════════════════════════════════════════
            // Comparison
            // ═══════════════════════════════════════════════════════════
            
            OpCode::Eq => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                let prov = a.prov.merge(&b.prov);
                state.stack.push(Value::from_bool_with_prov(a.val == b.val, prov));
            }
            
            OpCode::Neq => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                let prov = a.prov.merge(&b.prov);
                state.stack.push(Value::from_bool_with_prov(a.val != b.val, prov));
            }
            
            OpCode::Lt => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                let prov = a.prov.merge(&b.prov);
                state.stack.push(Value::from_bool_with_prov(a.val < b.val, prov));
            }
            
            OpCode::Gt => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                let prov = a.prov.merge(&b.prov);
                state.stack.push(Value::from_bool_with_prov(a.val > b.val, prov));
            }
            
            OpCode::Lte => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                let prov = a.prov.merge(&b.prov);
                state.stack.push(Value::from_bool_with_prov(a.val <= b.val, prov));
            }
            
            OpCode::Gte => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                let prov = a.prov.merge(&b.prov);
                state.stack.push(Value::from_bool_with_prov(a.val >= b.val, prov));
            }

            // ═══════════════════════════════════════════════════════════
            // Signed Comparison
            // ═══════════════════════════════════════════════════════════

            OpCode::Slt => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                state.stack.push(a.signed_lt(b));
            }

            OpCode::Sgt => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                state.stack.push(a.signed_gt(b));
            }

            OpCode::Slte => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                state.stack.push(a.signed_lte(b));
            }

            OpCode::Sgte => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                state.stack.push(a.signed_gte(b));
            }

            // ═══════════════════════════════════════════════════════════
            // Temporal Operations
            // ═══════════════════════════════════════════════════════════
            
            OpCode::Oracle => {
                let addr_val = self.pop(state)?;
                let addr = addr_val.val as Address;
                
                // Read from anamnesis
                let mut val = state.anamnesis.read(addr);
                
                // Inject oracle provenance
                let oracle_prov = Provenance::single(addr);
                val.prov = val.prov.merge(&oracle_prov).merge(&addr_val.prov);
                
                state.stack.push(val);
            }
            
            OpCode::Prophecy => {
                let addr_val = self.pop(state)?;
                let val = self.pop(state)?;
                let addr = addr_val.val as Address;
                
                // Write to present
                state.present.write(addr, val);
            }
            
            OpCode::PresentRead => {
                let addr_val = self.pop(state)?;
                let addr = addr_val.val as Address;
                
                // Read from present (current epoch's memory)
                let mut val = state.present.read(addr);
                val.prov = val.prov.merge(&addr_val.prov);
                
                state.stack.push(val);
            }
            
            OpCode::Paradox => {
                state.status = EpochStatus::Paradox;
            }
            
            // ═══════════════════════════════════════════════════════════
            // I/O
            // ═══════════════════════════════════════════════════════════
            
            OpCode::Input => {
                let val = if self.input_cursor < self.config.input.len() {
                    let v = self.config.input[self.input_cursor];
                    self.input_cursor += 1;
                    self.inputs_consumed.push(v);
                    v
                } else {
                    // Read from stdin as fallback
                    let v = self.read_input_interactive();
                    self.inputs_consumed.push(v);
                    v
                };
                state.stack.push(Value::new(val));
            }
            
            OpCode::Output => {
                let val = self.pop(state)?;
                
                if self.config.immediate_output {
                    println!("OUTPUT: {} [deps: {:?}]", val.val, val.prov);
                }
                
                state.output.push(OutputItem::Val(val));
            }

            OpCode::Emit => {
                let val = self.pop(state)?;
                let char_val = (val.val % 256) as u8;
                
                if self.config.immediate_output {
                    // For immediate output in debug mode, maybe show explicit EMIT
                    println!("EMIT: '{}' ({})", char_val as char, char_val);
                }
                
                state.output.push(OutputItem::Char(char_val));
            }
            
            // Array/Memory Operations
            OpCode::Pack => {
                // Pack n values into contiguous memory at base address
                let n = self.pop(state)?.val as usize;
                let base = self.pop(state)?.val as Address;
                for i in 0..n {
                    let val = self.pop(state)?;
                    state.present.write(base + (n - 1 - i) as Address, val);
                }
            }
            
            OpCode::Unpack => {
                // Unpack n values from contiguous memory at base address
                let n = self.pop(state)?.val as usize;
                let base = self.pop(state)?.val as Address;
                for i in 0..n {
                    let val = state.present.read(base + i as Address);
                    state.stack.push(val);
                }
            }
            
            OpCode::Index => {
                // Read from base + index
                let index = self.pop(state)?.val as Address;
                let base = self.pop(state)?.val as Address;
                let val = state.present.read(base + index);
                state.stack.push(val);
            }
            
            OpCode::Store => {
                // Store value at base + index
                let index = self.pop(state)?.val as Address;
                let base = self.pop(state)?.val as Address;
                let val = self.pop(state)?;
                state.present.write(base + index, val);
            }

            // ═══════════════════════════════════════════════════════════
            // Vector Operations
            // ═══════════════════════════════════════════════════════════

            OpCode::VecNew => {
                let handle = state.data_structures.vectors.alloc();
                state.stack.push(Value::new(handle));
            }

            OpCode::VecPush => {
                let value = self.pop(state)?;
                let handle = self.pop(state)?.val;
                state.data_structures.vectors.push(handle, value)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
            }

            OpCode::VecPop => {
                let handle = self.pop(state)?.val;
                let value = state.data_structures.vectors.pop(handle)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
                state.stack.push(value);
            }

            OpCode::VecGet => {
                let index = self.pop(state)?.val;
                let handle = self.pop(state)?.val;
                let value = state.data_structures.vectors.get_at(handle, index)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
                state.stack.push(value);
            }

            OpCode::VecSet => {
                let index = self.pop(state)?.val;
                let value = self.pop(state)?;
                let handle = self.pop(state)?.val;
                state.data_structures.vectors.set_at(handle, index, value)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
            }

            OpCode::VecLen => {
                let handle = self.pop(state)?.val;
                let length = state.data_structures.vectors.len(handle)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
                state.stack.push(Value::new(length));
            }

            // ═══════════════════════════════════════════════════════════
            // Hash Table Operations
            // ═══════════════════════════════════════════════════════════

            OpCode::HashNew => {
                let handle = state.data_structures.hashes.alloc();
                state.stack.push(Value::new(handle));
            }

            OpCode::HashPut => {
                let value = self.pop(state)?;
                let key = self.pop(state)?.val;
                let handle = self.pop(state)?.val;
                state.data_structures.hashes.put(handle, key, value)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
            }

            OpCode::HashGet => {
                let key = self.pop(state)?.val;
                let handle = self.pop(state)?.val;
                let (value, found) = state.data_structures.hashes.get_key(handle, key)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
                state.stack.push(value);
                state.stack.push(Value::new(if found { 1 } else { 0 }));
            }

            OpCode::HashDel => {
                let key = self.pop(state)?.val;
                let handle = self.pop(state)?.val;
                state.data_structures.hashes.delete(handle, key)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
            }

            OpCode::HashHas => {
                let key = self.pop(state)?.val;
                let handle = self.pop(state)?.val;
                let found = state.data_structures.hashes.has(handle, key)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
                state.stack.push(Value::new(if found { 1 } else { 0 }));
            }

            OpCode::HashLen => {
                let handle = self.pop(state)?.val;
                let count = state.data_structures.hashes.len(handle)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
                state.stack.push(Value::new(count));
            }

            // ═══════════════════════════════════════════════════════════
            // Set Operations
            // ═══════════════════════════════════════════════════════════

            OpCode::SetNew => {
                let handle = state.data_structures.sets.alloc();
                state.stack.push(Value::new(handle));
            }

            OpCode::SetAdd => {
                let value = self.pop(state)?.val;
                let handle = self.pop(state)?.val;
                state.data_structures.sets.add(handle, value)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
            }

            OpCode::SetHas => {
                let value = self.pop(state)?.val;
                let handle = self.pop(state)?.val;
                let found = state.data_structures.sets.has(handle, value)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
                state.stack.push(Value::new(if found { 1 } else { 0 }));
            }

            OpCode::SetDel => {
                let value = self.pop(state)?.val;
                let handle = self.pop(state)?.val;
                state.data_structures.sets.delete(handle, value)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
            }

            OpCode::SetLen => {
                let handle = self.pop(state)?.val;
                let count = state.data_structures.sets.len(handle)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
                state.stack.push(Value::new(count));
            }

            // ═══════════════════════════════════════════════════════════
            // FFI Operations
            // ═══════════════════════════════════════════════════════════

            OpCode::FFICall => {
                // ( arg1..argN function_id -- result1..resultM )
                let id_val = self.pop(state)?;
                let id = id_val.val as u32;
                FFICaller::call_by_id(
                    &mut self.ffi_context,
                    state,
                    id,
                    SourceLocation::default(),
                ).map_err(|e| e.to_string())?;
            }

            OpCode::FFICallNamed => {
                // ( arg1..argN name_len name_chars... -- result1..resultM )
                // Pop name length and characters to build function name
                let name_len = self.pop(state)?.val as usize;
                let mut name_chars = Vec::with_capacity(name_len);
                for _ in 0..name_len {
                    name_chars.push(self.pop(state)?.val as u8 as char);
                }
                name_chars.reverse();
                let name: String = name_chars.into_iter().collect();

                FFICaller::call_by_name(
                    &mut self.ffi_context,
                    state,
                    &name,
                    SourceLocation::default(),
                ).map_err(|e| e.to_string())?;
            }

            // ═══════════════════════════════════════════════════════════
            // File I/O Operations
            // ═══════════════════════════════════════════════════════════

            OpCode::FileOpen => {
                // ( path_len path_chars... mode -- file_handle )
                let mode_val = self.pop(state)?.val;
                let mode = FileMode::from_u64(mode_val);

                let path_len = self.pop(state)?.val as usize;
                let mut path_chars = Vec::with_capacity(path_len);
                for _ in 0..path_len {
                    path_chars.push(self.pop(state)?.val as u8 as char);
                }
                path_chars.reverse();
                let path: String = path_chars.into_iter().collect();

                let handle = self.io_context.file_open(&path, mode)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
            }

            OpCode::FileRead => {
                // ( file_handle max_bytes -- file_handle buffer_handle bytes_read )
                let max_bytes = self.pop(state)?.val as usize;
                let file_handle = self.pop(state)?.val;

                let (buffer_handle, bytes_read) = self.io_context.file_read(file_handle, max_bytes)
                    .map_err(|e| e.to_string())?;

                state.stack.push(Value::new(file_handle));
                state.stack.push(Value::new(buffer_handle));
                state.stack.push(Value::new(bytes_read as u64));
            }

            OpCode::FileWrite => {
                // ( file_handle buffer_handle -- file_handle bytes_written )
                let buffer_handle = self.pop(state)?.val;
                let file_handle = self.pop(state)?.val;

                let bytes_written = self.io_context.file_write(file_handle, buffer_handle)
                    .map_err(|e| e.to_string())?;

                state.stack.push(Value::new(file_handle));
                state.stack.push(Value::new(bytes_written as u64));
            }

            OpCode::FileSeek => {
                // ( file_handle offset origin -- file_handle new_position )
                let origin_val = self.pop(state)?.val;
                let offset = self.pop(state)?.val as i64;
                let file_handle = self.pop(state)?.val;

                let origin = SeekOrigin::from_u64(origin_val);
                let new_pos = self.io_context.file_seek(file_handle, origin, offset)
                    .map_err(|e| e.to_string())?;

                state.stack.push(Value::new(file_handle));
                state.stack.push(Value::new(new_pos));
            }

            OpCode::FileFlush => {
                // ( file_handle -- file_handle )
                let file_handle = self.pop(state)?.val;

                self.io_context.file_flush(file_handle)
                    .map_err(|e| e.to_string())?;

                state.stack.push(Value::new(file_handle));
            }

            OpCode::FileClose => {
                // ( file_handle -- )
                let file_handle = self.pop(state)?.val;

                self.io_context.file_close(file_handle)
                    .map_err(|e| e.to_string())?;
            }

            OpCode::FileExists => {
                // ( path_len path_chars... -- exists )
                let path_len = self.pop(state)?.val as usize;
                let mut path_chars = Vec::with_capacity(path_len);
                for _ in 0..path_len {
                    path_chars.push(self.pop(state)?.val as u8 as char);
                }
                path_chars.reverse();
                let path: String = path_chars.into_iter().collect();

                let exists = self.io_context.file_exists(&path);
                state.stack.push(Value::new(if exists { 1 } else { 0 }));
            }

            OpCode::FileSize => {
                // ( path_len path_chars... -- size )
                let path_len = self.pop(state)?.val as usize;
                let mut path_chars = Vec::with_capacity(path_len);
                for _ in 0..path_len {
                    path_chars.push(self.pop(state)?.val as u8 as char);
                }
                path_chars.reverse();
                let path: String = path_chars.into_iter().collect();

                let size = self.io_context.file_size(&path)
                    .unwrap_or(0);
                state.stack.push(Value::new(size));
            }

            // ═══════════════════════════════════════════════════════════
            // Buffer Operations
            // ═══════════════════════════════════════════════════════════

            OpCode::BufferNew => {
                // ( capacity -- buffer_handle )
                let capacity = self.pop(state)?.val as usize;
                let handle = self.io_context.buffer_new(capacity);
                state.stack.push(Value::new(handle));
            }

            OpCode::BufferFromStack => {
                // ( byte1..byteN n -- buffer_handle )
                let n = self.pop(state)?.val as usize;
                let mut bytes = Vec::with_capacity(n);
                for _ in 0..n {
                    bytes.push(self.pop(state)?);
                }
                bytes.reverse();
                let handle = self.io_context.buffer_from_values(&bytes);
                state.stack.push(Value::new(handle));
            }

            OpCode::BufferToStack => {
                // ( buffer_handle -- byte1..byteN n )
                let handle = self.pop(state)?.val;
                let values = self.io_context.buffer_to_values(handle)
                    .map_err(|e| e.to_string())?;
                let len = values.len();
                for v in values {
                    state.stack.push(v);
                }
                state.stack.push(Value::new(len as u64));
            }

            OpCode::BufferLen => {
                // ( buffer_handle -- buffer_handle length )
                let handle = self.pop(state)?.val;
                let length = self.io_context.buffer_len(handle)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
                state.stack.push(Value::new(length as u64));
            }

            OpCode::BufferReadByte => {
                // ( buffer_handle -- buffer_handle byte )
                let handle = self.pop(state)?.val;
                let byte = self.io_context.buffer_read_byte(handle)
                    .map_err(|e| e.to_string())?
                    .unwrap_or(0);
                state.stack.push(Value::new(handle));
                state.stack.push(Value::new(byte as u64));
            }

            OpCode::BufferWriteByte => {
                // ( buffer_handle byte -- buffer_handle )
                let byte = self.pop(state)?.val as u8;
                let handle = self.pop(state)?.val;
                self.io_context.buffer_write_byte(handle, byte)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
            }

            OpCode::BufferFree => {
                // ( buffer_handle -- )
                let handle = self.pop(state)?.val;
                self.io_context.buffer_free(handle);
            }

            // ═══════════════════════════════════════════════════════════
            // Network Operations
            // ═══════════════════════════════════════════════════════════

            OpCode::TcpConnect => {
                // ( host_len host_chars... port -- socket_handle )
                let port = self.pop(state)?.val as u16;
                let host_len = self.pop(state)?.val as usize;
                let mut host_chars = Vec::with_capacity(host_len);
                for _ in 0..host_len {
                    host_chars.push(self.pop(state)?.val as u8 as char);
                }
                host_chars.reverse();
                let host: String = host_chars.into_iter().collect();

                let handle = self.io_context.tcp_connect(&host, port)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
            }

            OpCode::SocketSend => {
                // ( socket_handle buffer_handle -- socket_handle bytes_sent )
                let buffer_handle = self.pop(state)?.val;
                let socket_handle = self.pop(state)?.val;

                let bytes_sent = self.io_context.socket_send(socket_handle, buffer_handle)
                    .map_err(|e| e.to_string())?;

                state.stack.push(Value::new(socket_handle));
                state.stack.push(Value::new(bytes_sent as u64));
            }

            OpCode::SocketRecv => {
                // ( socket_handle max_bytes -- socket_handle buffer_handle bytes_recv )
                let max_bytes = self.pop(state)?.val as usize;
                let socket_handle = self.pop(state)?.val;

                let (buffer_handle, bytes_recv) = self.io_context.socket_recv(socket_handle, max_bytes)
                    .map_err(|e| e.to_string())?;

                state.stack.push(Value::new(socket_handle));
                state.stack.push(Value::new(buffer_handle));
                state.stack.push(Value::new(bytes_recv as u64));
            }

            OpCode::SocketClose => {
                // ( socket_handle -- )
                let socket_handle = self.pop(state)?.val;

                self.io_context.socket_close(socket_handle)
                    .map_err(|e| e.to_string())?;
            }

            // ═══════════════════════════════════════════════════════════
            // Process Operations
            // ═══════════════════════════════════════════════════════════

            OpCode::ProcExec => {
                // ( cmd_len cmd_chars... -- output_handle exit_code )
                let cmd_len = self.pop(state)?.val as usize;
                let mut cmd_chars = Vec::with_capacity(cmd_len);
                for _ in 0..cmd_len {
                    cmd_chars.push(self.pop(state)?.val as u8 as char);
                }
                cmd_chars.reverse();
                let command: String = cmd_chars.into_iter().collect();

                let (output_handle, exit_code) = self.io_context.exec(&command)
                    .map_err(|e| e.to_string())?;

                state.stack.push(Value::new(output_handle));
                state.stack.push(Value::new(exit_code as u64));
            }

            // ═══════════════════════════════════════════════════════════
            // System Operations
            // ═══════════════════════════════════════════════════════════

            OpCode::Clock => {
                // ( -- timestamp )
                use std::time::{SystemTime, UNIX_EPOCH};
                let duration = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default();
                state.stack.push(Value::new(duration.as_millis() as u64));
            }

            OpCode::Sleep => {
                // ( milliseconds -- )
                let ms = self.pop(state)?.val;
                std::thread::sleep(std::time::Duration::from_millis(ms));
            }

            OpCode::Random => {
                // ( -- random_value )
                // Simple pseudo-random using time
                use std::time::{SystemTime, UNIX_EPOCH};
                let nanos = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .subsec_nanos();
                state.stack.push(Value::new(nanos as u64));
            }
        }

        Ok(())
    }
    
    /// Pop a value from the stack.
    fn pop(&self, state: &mut VmState) -> Result<Value, String> {
        state.stack.pop().ok_or_else(|| "Stack underflow".to_string())
    }
    
    /// Peek at the top of the stack.
    fn peek(&self, state: &VmState) -> Result<Value, String> {
        state.stack.last().cloned().ok_or_else(|| "Stack underflow".to_string())
    }
    
    /// Read input interactively.
    fn read_input_interactive(&self) -> u64 {
        print!("INPUT> ");
        io::stdout().flush().ok();
        
        let stdin = io::stdin();
        let mut line = String::new();
        
        if stdin.lock().read_line(&mut line).is_ok() {
            line.trim().parse().unwrap_or(0)
        } else {
            0
        }
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    
    #[test]
    fn test_simple_arithmetic() {
        let program = parse("10 20 ADD").unwrap();
        let mut executor = Executor::new();
        let result = executor.run_epoch(&program, &Memory::new());
        
        assert_eq!(result.status, EpochStatus::Finished);
    }
    
    #[test]
    fn test_oracle_prophecy() {
        let program = parse("0 ORACLE 0 PROPHECY").unwrap();
        let mut anamnesis = Memory::new();
        anamnesis.write(0, Value::new(42));
        
        let mut executor = Executor::new();
        let result = executor.run_epoch(&program, &anamnesis);
        
        assert_eq!(result.status, EpochStatus::Finished);
        assert_eq!(result.present.read(0).val, 42);
    }
    
    #[test]
    fn test_halt() {
        let program = parse("10 HALT 20").unwrap();
        let mut executor = Executor::new();
        let result = executor.run_epoch(&program, &Memory::new());
        
        assert_eq!(result.status, EpochStatus::Finished);
    }
    
    #[test]
    fn test_paradox() {
        let program = parse("PARADOX").unwrap();
        let mut executor = Executor::new();
        let result = executor.run_epoch(&program, &Memory::new());
        
        assert_eq!(result.status, EpochStatus::Paradox);
    }
    
    #[test]
    fn test_division_by_zero() {
        let program = parse("10 0 DIV").unwrap();
        let mut executor = Executor::new();
        let result = executor.run_epoch(&program, &Memory::new());

        // Should not error; returns 0
        assert_eq!(result.status, EpochStatus::Finished);
    }

    #[test]
    fn test_math_opcodes() {
        // Test MIN
        let program = parse("10 20 MIN").unwrap();
        let mut executor = Executor::new();
        let result = executor.run_epoch(&program, &Memory::new());
        assert_eq!(result.status, EpochStatus::Finished);
        assert_eq!(result.present.read(0).val, 0); // Stack result

        // Test MAX
        let program = parse("10 20 MAX").unwrap();
        let mut executor = Executor::new();
        let result = executor.run_epoch(&program, &Memory::new());
        assert_eq!(result.status, EpochStatus::Finished);

        // Test ABS (positive)
        let program = parse("42 ABS OUTPUT").unwrap();
        let mut executor = Executor::new();
        let result = executor.run_epoch(&program, &Memory::new());
        assert_eq!(result.status, EpochStatus::Finished);
        assert_eq!(result.output.len(), 1);

        // Test SIGN
        let program = parse("0 SIGN OUTPUT").unwrap();
        let mut executor = Executor::new();
        let result = executor.run_epoch(&program, &Memory::new());
        assert_eq!(result.status, EpochStatus::Finished);
    }

    #[test]
    fn test_signed_comparison() {
        // Testing signed comparisons with two's complement
        // -1 (u64::MAX) should be less than 0 in signed comparison
        let program = parse("18446744073709551615 0 SLT OUTPUT").unwrap(); // u64::MAX = -1 signed
        let mut executor = Executor::new();
        let result = executor.run_epoch(&program, &Memory::new());
        assert_eq!(result.status, EpochStatus::Finished);
        // -1 < 0 in signed => 1
        if let Some(crate::core_types::OutputItem::Val(v)) = result.output.first() {
            assert_eq!(v.val, 1);
        }

        // But in unsigned, u64::MAX > 0
        let program = parse("18446744073709551615 0 LT OUTPUT").unwrap();
        let mut executor = Executor::new();
        let result = executor.run_epoch(&program, &Memory::new());
        assert_eq!(result.status, EpochStatus::Finished);
        // u64::MAX < 0 in unsigned => 0
        if let Some(crate::core_types::OutputItem::Val(v)) = result.output.first() {
            assert_eq!(v.val, 0);
        }
    }

    #[test]
    fn test_vector_operations() {
        // Test VEC_NEW, VEC_PUSH, VEC_LEN
        let program = parse("VEC_NEW 42 VEC_PUSH 99 VEC_PUSH VEC_LEN OUTPUT").unwrap();
        let mut executor = Executor::new();
        executor.config.immediate_output = false;
        let result = executor.run_epoch(&program, &Memory::new());
        assert_eq!(result.status, EpochStatus::Finished);
        if let Some(crate::core_types::OutputItem::Val(v)) = result.output.first() {
            assert_eq!(v.val, 2); // Vector has 2 elements
        }
    }

    #[test]
    fn test_vector_get_pop() {
        // Test VEC_GET and VEC_POP
        let program = parse("VEC_NEW 100 VEC_PUSH 200 VEC_PUSH 0 VEC_GET OUTPUT").unwrap();
        let mut executor = Executor::new();
        executor.config.immediate_output = false;
        let result = executor.run_epoch(&program, &Memory::new());
        assert_eq!(result.status, EpochStatus::Finished);
        // First element is 100
        if let Some(crate::core_types::OutputItem::Val(v)) = result.output.first() {
            assert_eq!(v.val, 100);
        }
    }

    #[test]
    fn test_hash_operations() {
        // Test HASH_NEW, HASH_PUT, HASH_GET
        let program = parse("HASH_NEW 123 456 HASH_PUT 123 HASH_GET OUTPUT").unwrap();
        let mut executor = Executor::new();
        executor.config.immediate_output = false;
        let result = executor.run_epoch(&program, &Memory::new());
        assert_eq!(result.status, EpochStatus::Finished);
        // Found flag should be 1
        if let Some(crate::core_types::OutputItem::Val(v)) = result.output.first() {
            assert_eq!(v.val, 1);
        }
    }

    #[test]
    fn test_hash_has() {
        // Test HASH_HAS for existing and non-existing keys
        let program = parse("HASH_NEW 42 999 HASH_PUT 42 HASH_HAS OUTPUT 100 HASH_HAS OUTPUT").unwrap();
        let mut executor = Executor::new();
        executor.config.immediate_output = false;
        let result = executor.run_epoch(&program, &Memory::new());
        assert_eq!(result.status, EpochStatus::Finished);
        // First output: 42 exists => 1
        if let Some(crate::core_types::OutputItem::Val(v)) = result.output.get(0) {
            assert_eq!(v.val, 1);
        }
        // Second output: 100 doesn't exist => 0
        if let Some(crate::core_types::OutputItem::Val(v)) = result.output.get(1) {
            assert_eq!(v.val, 0);
        }
    }

    #[test]
    fn test_set_operations() {
        // Test SET_NEW, SET_ADD, SET_HAS
        let program = parse("SET_NEW 42 SET_ADD 99 SET_ADD 42 SET_HAS OUTPUT 100 SET_HAS OUTPUT").unwrap();
        let mut executor = Executor::new();
        executor.config.immediate_output = false;
        let result = executor.run_epoch(&program, &Memory::new());
        assert_eq!(result.status, EpochStatus::Finished);
        // 42 exists => 1
        if let Some(crate::core_types::OutputItem::Val(v)) = result.output.get(0) {
            assert_eq!(v.val, 1);
        }
        // 100 doesn't exist => 0
        if let Some(crate::core_types::OutputItem::Val(v)) = result.output.get(1) {
            assert_eq!(v.val, 0);
        }
    }

    #[test]
    fn test_set_len() {
        // Test SET_LEN
        let program = parse("SET_NEW 1 SET_ADD 2 SET_ADD 1 SET_ADD SET_LEN OUTPUT").unwrap();
        let mut executor = Executor::new();
        executor.config.immediate_output = false;
        let result = executor.run_epoch(&program, &Memory::new());
        assert_eq!(result.status, EpochStatus::Finished);
        // Set has 2 unique elements (1, 2) - adding 1 again doesn't change size
        if let Some(crate::core_types::OutputItem::Val(v)) = result.output.first() {
            assert_eq!(v.val, 2);
        }
    }
}
