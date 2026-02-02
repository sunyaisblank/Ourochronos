//! SMT-LIB2 encoder for OUROCHRONOS programs.
//!
//! Compiles the fixed-point constraint A = F(A) to SMT-LIB2 format,
//! allowing industrial solvers (Z3, CVC5) to find fixed points or
//! prove paradoxes (UNSAT).

use crate::ast::{Program, Stmt, OpCode};
use std::fmt::Write;

/// SMT-LIB2 encoder for OUROCHRONOS.
pub struct SmtEncoder {
    /// Generated SMT code.
    output: String,
    /// Counter for fresh variable names.
    var_counter: usize,
    /// Maximum loop unrolling depth.
    max_unroll: usize,
}

impl SmtEncoder {
    /// Create a new encoder.
    pub fn new() -> Self {
        Self {
            output: String::new(),
            var_counter: 0,
            max_unroll: 10,
        }
    }
    
    /// Create an encoder with custom unroll limit.
    pub fn with_unroll_limit(max_unroll: usize) -> Self {
        Self {
            output: String::new(),
            var_counter: 0,
            max_unroll,
        }
    }
    
    /// Encode a program to SMT-LIB2.
    pub fn encode(&mut self, program: &Program) -> String {
        self.output.clear();
        self.var_counter = 0;
        
        // Header
        writeln!(self.output, "; OUROCHRONOS SMT-LIB2 Encoding").unwrap();
        writeln!(self.output, "; Fixed-point constraint: A = F(A)").unwrap();
        writeln!(self.output).unwrap();
        
        // Logic: Quantifier-Free Arrays and Bit-Vectors
        writeln!(self.output, "(set-logic QF_ABV)").unwrap();
        writeln!(self.output, "(set-option :produce-models true)").unwrap();
        writeln!(self.output).unwrap();
        
        // Declare Anamnesis (the unknown we're solving for)
        writeln!(self.output, "; Anamnesis: the memory state from the 'future'").unwrap();
        writeln!(self.output, "(declare-const anamnesis (Array (_ BitVec 16) (_ BitVec 64)))").unwrap();
        writeln!(self.output).unwrap();
        
        // Initialize Present to all zeros
        writeln!(self.output, "; Present: starts as all zeros").unwrap();
        writeln!(self.output, "(declare-const present_init (Array (_ BitVec 16) (_ BitVec 64)))").unwrap();
        writeln!(self.output, "(assert (= present_init ((as const (Array (_ BitVec 16) (_ BitVec 64))) (_ bv0 64))))").unwrap();
        writeln!(self.output).unwrap();
        
        // Symbolic execution
        writeln!(self.output, "; Symbolic execution of program").unwrap();
        let mut stack: Vec<String> = Vec::new();
        let mut present = "present_init".to_string();
        
        self.encode_block(&program.body, &mut stack, &mut present, 0);
        
        writeln!(self.output).unwrap();
        
        // Fixed-point constraint
        writeln!(self.output, "; Fixed-point constraint: Present = Anamnesis").unwrap();
        writeln!(self.output, "(assert (= {} anamnesis))", present).unwrap();
        writeln!(self.output).unwrap();
        
        // Check and get model
        writeln!(self.output, "(check-sat)").unwrap();
        writeln!(self.output, "(get-model)").unwrap();
        
        self.output.clone()
    }
    
    /// Generate a fresh variable name.
    fn fresh_var(&mut self, prefix: &str) -> String {
        let name = format!("{}_{}", prefix, self.var_counter);
        self.var_counter += 1;
        name
    }
    
    /// Encode a block of statements.
    fn encode_block(&mut self, stmts: &[Stmt], 
                    stack: &mut Vec<String>, 
                    present: &mut String,
                    depth: usize) {
        for stmt in stmts {
            self.encode_stmt(stmt, stack, present, depth);
        }
    }
    
    /// Encode a single statement.
    fn encode_stmt(&mut self, stmt: &Stmt, 
                   stack: &mut Vec<String>, 
                   present: &mut String,
                   depth: usize) {
        match stmt {
            Stmt::Push(v) => {
                stack.push(format!("(_ bv{} 64)", v.val));
            }
            
            Stmt::Op(op) => {
                self.encode_op(*op, stack, present);
            }
            
            Stmt::Block(stmts) => {
                self.encode_block(stmts, stack, present, depth);
            }
            
            Stmt::If { then_branch, else_branch } => {
                self.encode_if(then_branch, else_branch.as_deref(), stack, present, depth);
            }
            
            Stmt::While { cond, body } => {
                self.encode_while(cond, body, stack, present, depth);
            }
            
            Stmt::Call { name } => {
                panic!("SMT encoding does not support procedure calls - inline procedures first: {}", name);
            }
            
            Stmt::Match { cases: _, default: _ } => {
                // Pattern matching requires complex ITE chains
                // For now, skip - would need full stack simulation
            }
            
            Stmt::TemporalScope { body, .. } => {
                // Temporal scoping: encode body, but note isolation semantics
                // are not fully expressible in SMT without memory regions
                self.encode_block(body, stack, present, depth);
            }
        }
    }
    
    /// Encode an opcode.
    fn encode_op(&mut self, op: OpCode, stack: &mut Vec<String>, present: &mut String) {
        match op {
            OpCode::Nop | OpCode::Halt | OpCode::Paradox => {}
            
            OpCode::Pop => { stack.pop(); }
            
            OpCode::Dup => {
                if let Some(top) = stack.last().cloned() {
                    stack.push(top);
                }
            }
            
            OpCode::Swap => {
                if stack.len() >= 2 {
                    let len = stack.len();
                    stack.swap(len - 1, len - 2);
                }
            }
            
            OpCode::Over => {
                if stack.len() >= 2 {
                    let val = stack[stack.len() - 2].clone();
                    stack.push(val);
                }
            }
            
            OpCode::Rot => {
                if stack.len() >= 3 {
                    let len = stack.len();
                    let c = stack.remove(len - 3);
                    stack.push(c);
                }
            }
            
            OpCode::Depth => {
                stack.push(format!("(_ bv{} 64)", stack.len()));
            }
            OpCode::Pick => {
                 panic!("SMT encoding for PICK not implemented");
            }
            OpCode::Add => {
                if stack.len() >= 2 {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(format!("(bvadd {} {})", a, b));
                }
            }
            
            OpCode::Sub => {
                if stack.len() >= 2 {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(format!("(bvsub {} {})", a, b));
                }
            }
            
            OpCode::Mul => {
                if stack.len() >= 2 {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(format!("(bvmul {} {})", a, b));
                }
            }
            
            OpCode::Div => {
                if stack.len() >= 2 {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    // Handle div by zero: (ite (= b 0) 0 (bvudiv a b))
                    stack.push(format!(
                        "(ite (= {} (_ bv0 64)) (_ bv0 64) (bvudiv {} {}))",
                        b, a, b
                    ));
                }
            }
            
            OpCode::Mod => {
                if stack.len() >= 2 {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(format!(
                        "(ite (= {} (_ bv0 64)) (_ bv0 64) (bvurem {} {}))",
                        b, a, b
                    ));
                }
            }
            
            OpCode::Neg => {
                if let Some(a) = stack.pop() {
                    stack.push(format!("(bvneg {})", a));
                }
            }

            OpCode::Abs => {
                if let Some(a) = stack.pop() {
                    // Signed absolute value: if negative, negate
                    stack.push(format!("(ite (bvslt {} #x0000000000000000) (bvneg {}) {})", a, a, a));
                }
            }

            OpCode::Min => {
                if stack.len() >= 2 {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(format!("(ite (bvult {} {}) {} {})", a, b, a, b));
                }
            }

            OpCode::Max => {
                if stack.len() >= 2 {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(format!("(ite (bvugt {} {}) {} {})", a, b, a, b));
                }
            }

            OpCode::Sign => {
                if let Some(a) = stack.pop() {
                    // Signed signum: -1, 0, or 1
                    stack.push(format!(
                        "(ite (= {} #x0000000000000000) #x0000000000000000 (ite (bvslt {} #x0000000000000000) #xFFFFFFFFFFFFFFFF #x0000000000000001))",
                        a, a
                    ));
                }
            }
            
            OpCode::Not => {
                if let Some(a) = stack.pop() {
                    stack.push(format!("(bvnot {})", a));
                }
            }
            
            OpCode::And => {
                if stack.len() >= 2 {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(format!("(bvand {} {})", a, b));
                }
            }
            
            OpCode::Or => {
                if stack.len() >= 2 {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(format!("(bvor {} {})", a, b));
                }
            }
            
            OpCode::Xor => {
                if stack.len() >= 2 {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(format!("(bvxor {} {})", a, b));
                }
            }
            
            OpCode::Shl => {
                if stack.len() >= 2 {
                    let n = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(format!("(bvshl {} {})", a, n));
                }
            }
            
            OpCode::Shr => {
                if stack.len() >= 2 {
                    let n = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(format!("(bvlshr {} {})", a, n));
                }
            }
            
            OpCode::Eq => {
                if stack.len() >= 2 {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(format!("(ite (= {} {}) (_ bv1 64) (_ bv0 64))", a, b));
                }
            }
            
            OpCode::Neq => {
                if stack.len() >= 2 {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(format!("(ite (not (= {} {})) (_ bv1 64) (_ bv0 64))", a, b));
                }
            }
            
            OpCode::Lt => {
                if stack.len() >= 2 {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(format!("(ite (bvult {} {}) (_ bv1 64) (_ bv0 64))", a, b));
                }
            }
            
            OpCode::Gt => {
                if stack.len() >= 2 {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(format!("(ite (bvugt {} {}) (_ bv1 64) (_ bv0 64))", a, b));
                }
            }
            
            OpCode::Lte => {
                if stack.len() >= 2 {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(format!("(ite (bvule {} {}) (_ bv1 64) (_ bv0 64))", a, b));
                }
            }
            
            OpCode::Gte => {
                if stack.len() >= 2 {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(format!("(ite (bvuge {} {}) (_ bv1 64) (_ bv0 64))", a, b));
                }
            }

            // Signed comparison
            OpCode::Slt => {
                if stack.len() >= 2 {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(format!("(ite (bvslt {} {}) (_ bv1 64) (_ bv0 64))", a, b));
                }
            }

            OpCode::Sgt => {
                if stack.len() >= 2 {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(format!("(ite (bvsgt {} {}) (_ bv1 64) (_ bv0 64))", a, b));
                }
            }

            OpCode::Slte => {
                if stack.len() >= 2 {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(format!("(ite (bvsle {} {}) (_ bv1 64) (_ bv0 64))", a, b));
                }
            }

            OpCode::Sgte => {
                if stack.len() >= 2 {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(format!("(ite (bvsge {} {}) (_ bv1 64) (_ bv0 64))", a, b));
                }
            }

            OpCode::Oracle => {
                if let Some(addr) = stack.pop() {
                    let addr16 = format!("((_ extract 15 0) {})", addr);
                    stack.push(format!("(select anamnesis {})", addr16));
                }
            }
            
            OpCode::Prophecy => {
                if stack.len() >= 2 {
                    let addr = stack.pop().unwrap();
                    let val = stack.pop().unwrap();
                    let addr16 = format!("((_ extract 15 0) {})", addr);
                    
                    let new_present = self.fresh_var("present");
                    writeln!(self.output, "(declare-const {} (Array (_ BitVec 16) (_ BitVec 64)))", new_present).unwrap();
                    writeln!(self.output, "(assert (= {} (store {} {} {})))", new_present, present, addr16, val).unwrap();
                    *present = new_present;
                }
            }
            
            OpCode::PresentRead => {
                if let Some(addr) = stack.pop() {
                    let addr16 = format!("((_ extract 15 0) {})", addr);
                    stack.push(format!("(select {} {})", present, addr16));
                }
            }
            
            OpCode::Input => {
                // Fresh symbolic input
                let input_var = self.fresh_var("input");
                writeln!(self.output, "(declare-const {} (_ BitVec 64))", input_var).unwrap();
                stack.push(input_var);
            }
            
            OpCode::Output | OpCode::Emit => {
                stack.pop();
            }
            
            // Array opcodes - not fully supported in SMT encoding
            OpCode::Pack | OpCode::Unpack | OpCode::Index | OpCode::Store => {
                // Array operations require complex memory theory
                // Dynamic stack structure changes cannot be statically typed easily
            }

            // Data structure opcodes - not fully supported in SMT encoding
            // These require complex heap/reference semantics
            OpCode::VecNew | OpCode::VecPush | OpCode::VecPop |
            OpCode::VecGet | OpCode::VecSet | OpCode::VecLen |
            OpCode::HashNew | OpCode::HashPut | OpCode::HashGet |
            OpCode::HashDel | OpCode::HashHas | OpCode::HashLen |
            OpCode::SetNew | OpCode::SetAdd | OpCode::SetHas |
            OpCode::SetDel | OpCode::SetLen => {
                // Data structure operations require heap theory
                // For now, treat as uninterpreted operations
            }

            OpCode::Roll | OpCode::Reverse | OpCode::StrRev | OpCode::StrCat | OpCode::StrSplit | OpCode::Assert | OpCode::Exec | OpCode::Dip | OpCode::Keep | OpCode::Bi | OpCode::Rec => {
                // Conservative approach: do nothing or invalidate stack?
                // For this prototype, we ignore their effect on type stack structure
            }

            // FFI and I/O operations - not supported in SMT encoding
            // These require modeling external state and side effects
            OpCode::FFICall | OpCode::FFICallNamed |
            OpCode::FileOpen | OpCode::FileRead | OpCode::FileWrite |
            OpCode::FileSeek | OpCode::FileFlush | OpCode::FileClose |
            OpCode::FileExists | OpCode::FileSize |
            OpCode::BufferNew | OpCode::BufferFromStack | OpCode::BufferToStack |
            OpCode::BufferLen | OpCode::BufferReadByte | OpCode::BufferWriteByte |
            OpCode::BufferFree |
            OpCode::TcpConnect | OpCode::SocketSend | OpCode::SocketRecv |
            OpCode::SocketClose |
            OpCode::ProcExec |
            OpCode::Clock | OpCode::Sleep | OpCode::Random => {
                // External operations cannot be encoded in SMT
                // These inherently have side effects not captured by the logic
            }
        }
    }
    
    /// Encode IF statement using ITE.
    fn encode_if(&mut self, 
                 then_branch: &[Stmt],
                 else_branch: Option<&[Stmt]>,
                 stack: &mut Vec<String>,
                 present: &mut String,
                 depth: usize) {
        // Pop condition
        let cond = stack.pop().unwrap_or("(_ bv0 64)".to_string());
        let cond_bool = format!("(not (= {} (_ bv0 64)))", cond);
        
        // Save state
        let stack_before = stack.clone();
        let present_before = present.clone();
        
        // Encode then branch
        self.encode_block(then_branch, stack, present, depth + 1);
        let stack_then = stack.clone();
        let present_then = present.clone();
        
        // Encode else branch
        *stack = stack_before;
        *present = present_before.clone();
        
        if let Some(else_stmts) = else_branch {
            self.encode_block(else_stmts, stack, present, depth + 1);
        }
        let stack_else = stack.clone();
        let present_else = present.clone();
        
        // Merge present using ITE
        let merged_present = self.fresh_var("present");
        writeln!(self.output, "(declare-const {} (Array (_ BitVec 16) (_ BitVec 64)))", merged_present).unwrap();
        writeln!(self.output, "(assert (= {} (ite {} {} {})))", 
                 merged_present, cond_bool, present_then, present_else).unwrap();
        *present = merged_present;
        
        // Merge stacks (take minimum length, merge with ITE)
        let min_len = stack_then.len().min(stack_else.len());
        stack.clear();
        for i in 0..min_len {
            let merged_val = format!("(ite {} {} {})", cond_bool, stack_then[i], stack_else[i]);
            stack.push(merged_val);
        }
    }
    
    /// Encode WHILE statement via bounded unrolling.
    fn encode_while(&mut self,
                    cond: &[Stmt],
                    body: &[Stmt],
                    stack: &mut Vec<String>,
                    present: &mut String,
                    depth: usize) {
        if depth >= self.max_unroll {
            writeln!(self.output, "; Loop unroll limit reached").unwrap();
            return;
        }
        
        // Unroll: for i in 0..max_unroll: if cond then body
        for _ in 0..self.max_unroll {
            // Evaluate condition
            self.encode_block(cond, stack, present, depth + 1);
            
            if stack.is_empty() {
                break;
            }
            
            // Encode as conditional body
            let cond_val = stack.pop().unwrap();
            let cond_bool = format!("(not (= {} (_ bv0 64)))", cond_val);
            
            // Save state
            let stack_before = stack.clone();
            let present_before = present.clone();
            
            // Encode body
            self.encode_block(body, stack, present, depth + 1);
            
            // Merge: if cond was true, use new state; else keep old
            let merged_present = self.fresh_var("present");
            writeln!(self.output, "(declare-const {} (Array (_ BitVec 16) (_ BitVec 64)))", merged_present).unwrap();
            writeln!(self.output, "(assert (= {} (ite {} {} {})))", 
                     merged_present, cond_bool, present, present_before).unwrap();
            *present = merged_present;
            
            // Merge stack
            let min_len = stack.len().min(stack_before.len());
            let new_stack: Vec<String> = (0..min_len)
                .map(|i| format!("(ite {} {} {})", cond_bool, stack[i], stack_before[i]))
                .collect();
            *stack = new_stack;
        }
    }
}

impl Default for SmtEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    
    #[test]
    fn test_encode_simple() {
        let program = parse("0 ORACLE 0 PROPHECY").unwrap();
        let mut encoder = SmtEncoder::new();
        let smt = encoder.encode(&program);
        
        assert!(smt.contains("anamnesis"));
        assert!(smt.contains("check-sat"));
    }
    
    #[test]
    fn test_encode_arithmetic() {
        let program = parse("10 20 ADD 0 PROPHECY").unwrap();
        let mut encoder = SmtEncoder::new();
        let smt = encoder.encode(&program);
        
        assert!(smt.contains("bvadd"));
    }
}
