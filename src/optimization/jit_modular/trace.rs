//! Trace recording for the tracing JIT.
//!
//! A trace is a recorded sequence of operations through a hot loop.
//! Traces are compiled to native code or specialized handlers for
//! faster execution.

use crate::ast::OpCode;

/// A recorded trace of operations through a hot loop.
#[derive(Debug, Clone)]
pub struct Trace {
    /// The hash identifying this trace's entry point.
    pub entry_hash: u64,
    /// Recorded operations in execution order.
    pub ops: Vec<TraceOp>,
    /// Number of times this trace has been executed.
    pub execution_count: u64,
    /// Whether this trace has been compiled.
    pub compiled: bool,
}

impl Trace {
    /// Create a new trace with the given entry hash.
    pub fn new(entry_hash: u64) -> Self {
        Self {
            entry_hash,
            ops: Vec::new(),
            execution_count: 0,
            compiled: false,
        }
    }

    /// Record an operation to the trace.
    pub fn record(&mut self, op: TraceOp) {
        self.ops.push(op);
    }

    /// Get the number of operations in this trace.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Check if the trace is empty.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Mark this trace as having been executed.
    pub fn mark_executed(&mut self) {
        self.execution_count += 1;
    }

    /// Check if this trace is hot (executed many times).
    pub fn is_hot(&self, threshold: u64) -> bool {
        self.execution_count >= threshold
    }

    /// Analyze the trace to extract statistics.
    pub fn analyze(&self) -> TraceAnalysis {
        let mut analysis = TraceAnalysis::default();

        for op in &self.ops {
            match op {
                TraceOp::Push(_) => analysis.push_count += 1,
                TraceOp::Op(opcode) => {
                    analysis.op_count += 1;
                    match opcode {
                        OpCode::Add | OpCode::Sub | OpCode::Mul |
                        OpCode::Div | OpCode::Mod => analysis.arithmetic_count += 1,
                        OpCode::Oracle => analysis.oracle_count += 1,
                        OpCode::Prophecy => analysis.prophecy_count += 1,
                        OpCode::Dup | OpCode::Swap | OpCode::Over |
                        OpCode::Rot | OpCode::Pop => analysis.stack_count += 1,
                        OpCode::Eq | OpCode::Neq | OpCode::Lt |
                        OpCode::Gt | OpCode::Lte | OpCode::Gte => analysis.comparison_count += 1,
                        _ => {}
                    }
                }
                TraceOp::Guard { .. } => analysis.guard_count += 1,
                TraceOp::LoopBack => analysis.has_loop = true,
                TraceOp::Exit => analysis.has_exit = true,
            }
        }

        analysis.total_ops = self.ops.len();
        analysis.is_temporal = analysis.oracle_count > 0 || analysis.prophecy_count > 0;
        analysis
    }
}

/// A single operation in a trace.
#[derive(Debug, Clone, PartialEq)]
pub enum TraceOp {
    /// Push a constant value.
    Push(u64),
    /// Simple opcode (no control flow).
    Op(OpCode),
    /// Guard: check condition, deoptimize if false.
    Guard { expected: bool },
    /// Loop back to start (tail of trace).
    LoopBack,
    /// Exit the trace and return to interpreter.
    Exit,
}

/// Analysis results for a trace.
#[derive(Debug, Clone, Default)]
pub struct TraceAnalysis {
    /// Total number of operations.
    pub total_ops: usize,
    /// Number of push operations.
    pub push_count: usize,
    /// Number of opcode operations.
    pub op_count: usize,
    /// Number of arithmetic operations.
    pub arithmetic_count: usize,
    /// Number of stack manipulation operations.
    pub stack_count: usize,
    /// Number of comparison operations.
    pub comparison_count: usize,
    /// Number of ORACLE operations.
    pub oracle_count: usize,
    /// Number of PROPHECY operations.
    pub prophecy_count: usize,
    /// Number of guard operations.
    pub guard_count: usize,
    /// Whether the trace loops back.
    pub has_loop: bool,
    /// Whether the trace has an exit.
    pub has_exit: bool,
    /// Whether the trace is temporal (has ORACLE/PROPHECY).
    pub is_temporal: bool,
}

impl TraceAnalysis {
    /// Get the ratio of arithmetic to total operations.
    pub fn arithmetic_density(&self) -> f64 {
        if self.total_ops == 0 {
            0.0
        } else {
            self.arithmetic_count as f64 / self.total_ops as f64
        }
    }

    /// Get the ratio of temporal operations.
    pub fn temporal_density(&self) -> f64 {
        if self.total_ops == 0 {
            0.0
        } else {
            (self.oracle_count + self.prophecy_count) as f64 / self.total_ops as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_creation() {
        let trace = Trace::new(12345);
        assert_eq!(trace.entry_hash, 12345);
        assert!(trace.is_empty());
        assert_eq!(trace.execution_count, 0);
    }

    #[test]
    fn test_trace_recording() {
        let mut trace = Trace::new(0);
        trace.record(TraceOp::Push(42));
        trace.record(TraceOp::Op(OpCode::Dup));
        trace.record(TraceOp::Op(OpCode::Add));

        assert_eq!(trace.len(), 3);
        assert!(!trace.is_empty());
    }

    #[test]
    fn test_trace_analysis() {
        let mut trace = Trace::new(0);
        trace.record(TraceOp::Push(1));
        trace.record(TraceOp::Push(2));
        trace.record(TraceOp::Op(OpCode::Add));
        trace.record(TraceOp::Op(OpCode::Oracle));
        trace.record(TraceOp::LoopBack);

        let analysis = trace.analyze();
        assert_eq!(analysis.push_count, 2);
        assert_eq!(analysis.arithmetic_count, 1);
        assert_eq!(analysis.oracle_count, 1);
        assert!(analysis.has_loop);
        assert!(analysis.is_temporal);
    }

    #[test]
    fn test_trace_hotness() {
        let mut trace = Trace::new(0);
        assert!(!trace.is_hot(10));

        for _ in 0..10 {
            trace.mark_executed();
        }
        assert!(trace.is_hot(10));
    }
}
