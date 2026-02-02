//! Trace execution and specialization.
//!
//! This module handles executing compiled traces, either through
//! specialized Rust handlers or (optionally) native code.

use super::trace::{Trace, TraceOp};
use super::patterns::TracePattern;

/// Result of trace execution.
#[derive(Debug, Clone)]
pub enum ExecutionResult {
    /// Trace completed successfully.
    Success {
        /// Number of iterations executed.
        iterations: u64,
    },
    /// Trace guard failed, need to deoptimize.
    GuardFailure {
        /// Index of the failed guard.
        guard_index: usize,
    },
    /// Trace execution error.
    Error(String),
}

/// Statistics about trace execution.
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    /// Total traces executed.
    pub traces_executed: u64,
    /// Total iterations across all traces.
    pub total_iterations: u64,
    /// Guard failures (deoptimizations).
    pub guard_failures: u64,
    /// Successful trace completions.
    pub successes: u64,
}

impl ExecutionStats {
    /// Get the success rate.
    pub fn success_rate(&self) -> f64 {
        if self.traces_executed == 0 {
            1.0
        } else {
            self.successes as f64 / self.traces_executed as f64
        }
    }

    /// Record a successful execution.
    pub fn record_success(&mut self, iterations: u64) {
        self.traces_executed += 1;
        self.successes += 1;
        self.total_iterations += iterations;
    }

    /// Record a guard failure.
    pub fn record_guard_failure(&mut self) {
        self.traces_executed += 1;
        self.guard_failures += 1;
    }
}

/// Compiled trace representation.
#[derive(Debug, Clone)]
pub struct CompiledTrace {
    /// The pattern this trace was compiled as.
    pub pattern: TracePattern,
    /// The original trace (for deoptimization).
    pub source: Trace,
}

impl CompiledTrace {
    /// Create a new compiled trace.
    pub fn new(pattern: TracePattern, source: Trace) -> Self {
        Self { pattern, source }
    }

    /// Get the expected speedup for this compiled trace.
    pub fn expected_speedup(&self) -> f64 {
        self.pattern.expected_speedup()
    }
}

/// Trait for trace executors.
pub trait TraceExecutor {
    /// Execute a compiled trace.
    fn execute(&mut self, trace: &CompiledTrace, iterations: u64) -> ExecutionResult;

    /// Check if this executor can handle the given pattern.
    fn can_execute(&self, pattern: TracePattern) -> bool;
}

/// A simple interpreter-based trace executor.
#[derive(Debug, Default)]
pub struct InterpretedExecutor {
    /// Execution statistics.
    pub stats: ExecutionStats,
}

impl InterpretedExecutor {
    pub fn new() -> Self {
        Self::default()
    }
}

impl TraceExecutor for InterpretedExecutor {
    fn execute(&mut self, trace: &CompiledTrace, iterations: u64) -> ExecutionResult {
        // Simple loop execution
        for _ in 0..iterations {
            for (idx, op) in trace.source.ops.iter().enumerate() {
                match op {
                    TraceOp::Guard { expected: _ } => {
                        // In interpreted mode, we don't actually check guards
                        // This would require full VM state
                    }
                    TraceOp::Exit => {
                        self.stats.record_success(iterations);
                        return ExecutionResult::Success { iterations };
                    }
                    TraceOp::LoopBack => {
                        // Continue to next iteration
                        break;
                    }
                    _ => {
                        // Other ops would be interpreted here
                        let _ = idx; // Suppress warning
                    }
                }
            }
        }

        self.stats.record_success(iterations);
        ExecutionResult::Success { iterations }
    }

    fn can_execute(&self, _pattern: TracePattern) -> bool {
        // Interpreter can execute any pattern (slowly)
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_stats() {
        let mut stats = ExecutionStats::default();
        assert_eq!(stats.success_rate(), 1.0);

        stats.record_success(10);
        stats.record_success(20);
        stats.record_guard_failure();

        assert_eq!(stats.traces_executed, 3);
        assert_eq!(stats.successes, 2);
        assert_eq!(stats.guard_failures, 1);
        assert!((stats.success_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_compiled_trace() {
        let trace = Trace::new(0);
        let compiled = CompiledTrace::new(TracePattern::FibonacciStep, trace);

        assert_eq!(compiled.pattern, TracePattern::FibonacciStep);
        assert!(compiled.expected_speedup() > 1.0);
    }
}
