//! Modular Tracing JIT compiler for OUROCHRONOS.
//!
//! This module is organized into focused submodules:
//!
//! - **trace**: Trace recording and analysis
//! - **patterns**: Pattern detection and classification
//! - **execution**: Trace execution and specialization
//!
//! # Architecture
//!
//! The JIT uses a two-tier approach:
//! - **Tier 0 (Interpreter)**: Fast VM executes all code initially
//! - **Tier 1 (JIT)**: Hot loops are compiled after threshold executions
//!
//! # Trace Recording
//!
//! When a loop header is detected as hot, we record a "trace" - the sequence
//! of operations executed through one iteration. This trace is then compiled
//! to specialized handlers for faster execution.
//!
//! # Pattern Detection
//!
//! Traces are analyzed to detect known patterns:
//! - **Pure patterns**: Fibonacci, counter, sum, factorial, etc.
//! - **Temporal patterns**: Read-modify-write, convergence, causal chain, etc.
//!
//! Each pattern has a specialized handler that executes faster than
//! interpreting the trace.

pub mod trace;
pub mod patterns;
pub mod execution;

// Re-export key types for convenience
pub use trace::{Trace, TraceOp, TraceAnalysis};
pub use patterns::{TracePattern, PatternCategory, detect_pattern, score_pattern};
pub use execution::{
    ExecutionResult, ExecutionStats, CompiledTrace,
    TraceExecutor, InterpretedExecutor
};

/// JIT compiler statistics.
#[derive(Debug, Clone, Default)]
pub struct JitStats {
    /// Number of traces recorded.
    pub traces_recorded: u64,
    /// Number of traces compiled.
    pub traces_compiled: u64,
    /// Number of hot trace executions.
    pub hot_executions: u64,
    /// Pattern detection counts.
    pub pattern_counts: std::collections::HashMap<String, u64>,
    /// Execution statistics.
    pub execution_stats: ExecutionStats,
}

impl JitStats {
    /// Record a new trace.
    pub fn record_trace(&mut self) {
        self.traces_recorded += 1;
    }

    /// Record trace compilation.
    pub fn record_compilation(&mut self, pattern: TracePattern) {
        self.traces_compiled += 1;
        *self.pattern_counts
            .entry(pattern.name().to_string())
            .or_insert(0) += 1;
    }

    /// Get the compilation rate.
    pub fn compilation_rate(&self) -> f64 {
        if self.traces_recorded == 0 {
            0.0
        } else {
            self.traces_compiled as f64 / self.traces_recorded as f64
        }
    }
}

/// Configuration for the JIT compiler.
#[derive(Debug, Clone)]
pub struct JitConfig {
    /// Number of executions before a loop is considered hot.
    pub hot_threshold: u64,
    /// Maximum trace length to record.
    pub max_trace_length: usize,
    /// Whether to enable pattern detection.
    pub enable_patterns: bool,
    /// Whether to enable temporal pattern optimization.
    pub enable_temporal_patterns: bool,
}

impl Default for JitConfig {
    fn default() -> Self {
        Self {
            hot_threshold: 10,
            max_trace_length: 1000,
            enable_patterns: true,
            enable_temporal_patterns: true,
        }
    }
}

impl JitConfig {
    /// Create a configuration for aggressive optimization.
    pub fn aggressive() -> Self {
        Self {
            hot_threshold: 5,
            max_trace_length: 2000,
            enable_patterns: true,
            enable_temporal_patterns: true,
        }
    }

    /// Create a configuration for conservative optimization.
    pub fn conservative() -> Self {
        Self {
            hot_threshold: 50,
            max_trace_length: 500,
            enable_patterns: true,
            enable_temporal_patterns: false,
        }
    }

    /// Create a configuration with JIT disabled.
    pub fn disabled() -> Self {
        Self {
            hot_threshold: u64::MAX,
            max_trace_length: 0,
            enable_patterns: false,
            enable_temporal_patterns: false,
        }
    }
}

/// Simple JIT compiler that manages trace recording and compilation.
#[derive(Debug)]
pub struct SimpleJit {
    /// JIT configuration.
    pub config: JitConfig,
    /// Recorded traces indexed by entry hash.
    traces: std::collections::HashMap<u64, Trace>,
    /// Compiled traces.
    compiled: std::collections::HashMap<u64, CompiledTrace>,
    /// JIT statistics.
    pub stats: JitStats,
}

impl SimpleJit {
    /// Create a new JIT compiler with default configuration.
    pub fn new() -> Self {
        Self::with_config(JitConfig::default())
    }

    /// Create a JIT compiler with custom configuration.
    pub fn with_config(config: JitConfig) -> Self {
        Self {
            config,
            traces: std::collections::HashMap::new(),
            compiled: std::collections::HashMap::new(),
            stats: JitStats::default(),
        }
    }

    /// Record entry to a loop with the given hash.
    pub fn record_entry(&mut self, entry_hash: u64) {
        let trace = self.traces
            .entry(entry_hash)
            .or_insert_with(|| {
                self.stats.record_trace();
                Trace::new(entry_hash)
            });

        trace.mark_executed();

        // Check if trace should be compiled
        if !trace.compiled && trace.is_hot(self.config.hot_threshold) {
            self.compile_trace(entry_hash);
        }
    }

    /// Record an operation to the current trace.
    pub fn record_op(&mut self, entry_hash: u64, op: TraceOp) {
        if let Some(trace) = self.traces.get_mut(&entry_hash) {
            if trace.len() < self.config.max_trace_length {
                trace.record(op);
            }
        }
    }

    /// Compile a hot trace.
    fn compile_trace(&mut self, entry_hash: u64) {
        if let Some(trace) = self.traces.get_mut(&entry_hash) {
            if trace.compiled {
                return;
            }

            let analysis = trace.analyze();
            let pattern = if self.config.enable_patterns {
                if self.config.enable_temporal_patterns || !analysis.is_temporal {
                    detect_pattern(&analysis)
                } else {
                    TracePattern::Generic
                }
            } else {
                TracePattern::Generic
            };

            let compiled = CompiledTrace::new(pattern, trace.clone());
            self.compiled.insert(entry_hash, compiled);
            trace.compiled = true;

            self.stats.record_compilation(pattern);
        }
    }

    /// Get a compiled trace if available.
    pub fn get_compiled(&self, entry_hash: u64) -> Option<&CompiledTrace> {
        self.compiled.get(&entry_hash)
    }

    /// Check if a trace is compiled and ready.
    pub fn is_compiled(&self, entry_hash: u64) -> bool {
        self.compiled.contains_key(&entry_hash)
    }

    /// Get the number of recorded traces.
    pub fn trace_count(&self) -> usize {
        self.traces.len()
    }

    /// Get the number of compiled traces.
    pub fn compiled_count(&self) -> usize {
        self.compiled.len()
    }
}

impl Default for SimpleJit {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::OpCode;
#[allow(unused_imports)]

    #[test]
    fn test_jit_stats() {
        let mut stats = JitStats::default();
        stats.record_trace();
        stats.record_trace();
        stats.record_compilation(TracePattern::FibonacciStep);

        assert_eq!(stats.traces_recorded, 2);
        assert_eq!(stats.traces_compiled, 1);
        assert_eq!(stats.compilation_rate(), 0.5);
    }

    #[test]
    fn test_jit_config() {
        let config = JitConfig::default();
        assert_eq!(config.hot_threshold, 10);
        assert!(config.enable_patterns);

        let aggressive = JitConfig::aggressive();
        assert!(aggressive.hot_threshold < config.hot_threshold);

        let disabled = JitConfig::disabled();
        assert_eq!(disabled.hot_threshold, u64::MAX);
    }

    #[test]
    fn test_simple_jit() {
        let mut jit = SimpleJit::new();

        // Record a trace that becomes hot
        for _ in 0..15 {
            jit.record_entry(12345);
        }

        // Should be compiled after hot threshold
        assert!(jit.is_compiled(12345));
        assert_eq!(jit.compiled_count(), 1);
    }

    #[test]
    fn test_jit_record_ops() {
        let mut jit = SimpleJit::new();
        jit.record_entry(100);

        jit.record_op(100, TraceOp::Push(42));
        jit.record_op(100, TraceOp::Op(OpCode::Dup));
        jit.record_op(100, TraceOp::Op(OpCode::Add));
        jit.record_op(100, TraceOp::LoopBack);

        // Make it hot
        for _ in 0..20 {
            jit.record_entry(100);
        }

        assert!(jit.is_compiled(100));
        let compiled = jit.get_compiled(100).unwrap();
        assert!(!compiled.source.is_empty());
    }
}
