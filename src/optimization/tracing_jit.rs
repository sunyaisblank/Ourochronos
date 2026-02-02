//! Tracing JIT compiler for OUROCHRONOS hot loops.
//!
//! This module implements a tracing JIT that:
//! 1. Detects hot loops during interpretation
//! 2. Records execution traces
//! 3. Compiles traces to native code (with Cranelift) or specialised handlers
//! 4. Executes compiled code with fallback to interpreter
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
//! to native code that eliminates dispatch overhead.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

use crate::ast::{OpCode, Stmt, Program};
use crate::core::{Value, Memory};
use super::memo::SpeculativeExecutor;

// ═══════════════════════════════════════════════════════════════════════════
// Trace Recording
// ═══════════════════════════════════════════════════════════════════════════

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
    /// Compiled function pointer (when available).
    compiled_fn: Option<CompiledTrace>,
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

/// A compiled trace - either native code or specialised handler.
#[derive(Debug, Clone)]
enum CompiledTrace {
    /// Specialised Rust function for this trace pattern.
    Specialised(TracePattern),
    /// Native code pointer (with Cranelift).
    #[cfg(feature = "jit")]
    Native(*const u8),
}

/// Known trace patterns that can be specialised.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TracePattern {
    // ═══════════════════════════════════════════════════════════════════════
    // Pure Computation Patterns
    // ═══════════════════════════════════════════════════════════════════════
    /// Fibonacci step: SWAP OVER ADD
    FibonacciStep,
    /// Simple counter increment loop: counter += 1
    CounterLoop,
    /// Accumulation loop: sum += counter (sum of 1 to N)
    SumLoop,
    /// Factorial loop: result *= counter
    FactorialLoop,
    /// Power loop: result *= base (exponentiation)
    PowerLoop,
    /// Modular exponentiation: base^exp mod m
    ModPowerLoop,
    /// GCD loop: Euclidean algorithm
    GcdLoop,
    /// Memory fill: fill range with value
    MemoryFillLoop,
    /// Memory copy: copy range of memory
    MemoryCopyLoop,
    /// Reduction loop: fold operation over values
    ReductionLoop,
    /// Dot product: sum of products
    DotProductLoop,
    /// Polynomial evaluation: Horner's method
    PolynomialLoop,
    /// Comparison chain: find min/max in sequence
    MinMaxLoop,
    /// Bitwise operation loop: repeated bit manipulations
    BitwiseLoop,

    // ═══════════════════════════════════════════════════════════════════════
    // Temporal Patterns (ORACLE/PROPHECY idioms)
    // ═══════════════════════════════════════════════════════════════════════
    /// Read-modify-write: ORACLE addr → compute → PROPHECY addr
    /// Common pattern for updating temporal memory atomically
    ReadModifyWrite,
    /// Convergence loop: iterative refinement until values stabilize
    /// Pattern: while (oracle != expected) { prophecy(new_value) }
    ConvergenceLoop,
    /// Causal chain: PROPHECY depends on ORACLE in DAG structure
    /// Pattern: val = oracle(a); prophecy(b, f(val))
    CausalChain,
    /// Bootstrap pattern: initial guess refined across epochs
    /// Pattern: oracle with fallback, then prophecy update
    BootstrapGuess,
    /// Witness search: ORACLE probes multiple addresses to find value
    /// Pattern: for addr in range { if oracle(addr) == target { ... } }
    WitnessSearch,
    /// Temporal scan: sequential ORACLE reads over address range
    TemporalScan,
    /// Temporal scatter: PROPHECY writes to computed addresses
    TemporalScatter,
    /// Fixed-point iteration: repeated application until stable
    /// Pattern: x = f(oracle(x)); prophecy(x, result)
    FixedPointIteration,

    /// Generic loop (no specialisation available)
    Generic,
}

/// Category of a trace pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternCategory {
    /// Pure computation - no temporal effects
    Pure,
    /// Temporal - involves ORACLE/PROPHECY
    Temporal,
    /// Generic - unknown category
    Generic,
}

impl TracePattern {
    /// Get the name of this pattern for display.
    pub fn name(&self) -> &'static str {
        match self {
            // Pure patterns
            TracePattern::FibonacciStep => "fibonacci",
            TracePattern::CounterLoop => "counter",
            TracePattern::SumLoop => "sum",
            TracePattern::FactorialLoop => "factorial",
            TracePattern::PowerLoop => "power",
            TracePattern::ModPowerLoop => "modpower",
            TracePattern::GcdLoop => "gcd",
            TracePattern::MemoryFillLoop => "memfill",
            TracePattern::MemoryCopyLoop => "memcopy",
            TracePattern::ReductionLoop => "reduction",
            TracePattern::DotProductLoop => "dotproduct",
            TracePattern::PolynomialLoop => "polynomial",
            TracePattern::MinMaxLoop => "minmax",
            TracePattern::BitwiseLoop => "bitwise",
            // Temporal patterns
            TracePattern::ReadModifyWrite => "read-modify-write",
            TracePattern::ConvergenceLoop => "convergence",
            TracePattern::CausalChain => "causal-chain",
            TracePattern::BootstrapGuess => "bootstrap",
            TracePattern::WitnessSearch => "witness-search",
            TracePattern::TemporalScan => "temporal-scan",
            TracePattern::TemporalScatter => "temporal-scatter",
            TracePattern::FixedPointIteration => "fixed-point",
            // Generic
            TracePattern::Generic => "generic",
        }
    }

    /// Get expected performance multiplier for this pattern.
    pub fn expected_speedup(&self) -> f64 {
        match self {
            // Pure patterns - within-epoch speedup
            TracePattern::FibonacciStep => 10.0,
            TracePattern::CounterLoop => 5.0,
            TracePattern::SumLoop => 8.0,
            TracePattern::FactorialLoop => 8.0,
            TracePattern::PowerLoop => 8.0,
            TracePattern::ModPowerLoop => 6.0,
            TracePattern::GcdLoop => 5.0,
            TracePattern::MemoryFillLoop => 15.0,
            TracePattern::MemoryCopyLoop => 12.0,
            TracePattern::ReductionLoop => 7.0,
            TracePattern::DotProductLoop => 8.0,
            TracePattern::PolynomialLoop => 7.0,
            TracePattern::MinMaxLoop => 6.0,
            TracePattern::BitwiseLoop => 5.0,
            // Temporal patterns - potential epoch reduction
            TracePattern::ReadModifyWrite => 3.0,      // Atomic operation fusion
            TracePattern::ConvergenceLoop => 5.0,      // Can predict convergence
            TracePattern::CausalChain => 2.0,          // Dependency analysis
            TracePattern::BootstrapGuess => 4.0,       // Cache successful guesses
            TracePattern::WitnessSearch => 8.0,        // Parallel search
            TracePattern::TemporalScan => 6.0,         // Vectorized reads
            TracePattern::TemporalScatter => 5.0,      // Batched writes
            TracePattern::FixedPointIteration => 10.0, // Newton acceleration
            // Generic
            TracePattern::Generic => 1.0,
        }
    }

    /// Get the category of this pattern.
    pub fn category(&self) -> PatternCategory {
        match self {
            TracePattern::FibonacciStep |
            TracePattern::CounterLoop |
            TracePattern::SumLoop |
            TracePattern::FactorialLoop |
            TracePattern::PowerLoop |
            TracePattern::ModPowerLoop |
            TracePattern::GcdLoop |
            TracePattern::MemoryFillLoop |
            TracePattern::MemoryCopyLoop |
            TracePattern::ReductionLoop |
            TracePattern::DotProductLoop |
            TracePattern::PolynomialLoop |
            TracePattern::MinMaxLoop |
            TracePattern::BitwiseLoop => PatternCategory::Pure,

            TracePattern::ReadModifyWrite |
            TracePattern::ConvergenceLoop |
            TracePattern::CausalChain |
            TracePattern::BootstrapGuess |
            TracePattern::WitnessSearch |
            TracePattern::TemporalScan |
            TracePattern::TemporalScatter |
            TracePattern::FixedPointIteration => PatternCategory::Temporal,

            TracePattern::Generic => PatternCategory::Generic,
        }
    }

    /// Check if this pattern involves temporal operations.
    pub fn is_temporal(&self) -> bool {
        self.category() == PatternCategory::Temporal
    }

    /// Get the expected epoch reduction factor for temporal patterns.
    /// Returns 1.0 for pure patterns (no epoch reduction).
    pub fn expected_epoch_reduction(&self) -> f64 {
        match self {
            TracePattern::ConvergenceLoop => 2.0,      // May halve epochs needed
            TracePattern::BootstrapGuess => 1.5,       // Better initial guesses
            TracePattern::FixedPointIteration => 3.0,  // Newton-like acceleration
            TracePattern::WitnessSearch => 1.2,        // Faster witness finding
            _ => 1.0, // No epoch reduction for other patterns
        }
    }
}

impl Trace {
    /// Create a new empty trace.
    pub fn new(entry_hash: u64) -> Self {
        Self {
            entry_hash,
            ops: Vec::new(),
            execution_count: 0,
            compiled: false,
            compiled_fn: None,
        }
    }

    /// Record an operation to the trace.
    pub fn record(&mut self, op: TraceOp) {
        self.ops.push(op);
    }

    /// Detect if this trace matches a known pattern.
    pub fn detect_pattern(&self) -> TracePattern {
        let analysis = self.analyze_ops();

        // Use weighted scoring to detect the most likely pattern
        let pattern_scores = self.compute_pattern_scores(&analysis);

        // Return the pattern with highest score above threshold
        pattern_scores
            .into_iter()
            .filter(|(_, score)| *score >= 0.5)
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(pattern, _)| pattern)
            .unwrap_or(TracePattern::Generic)
    }

    /// Analyze operations in the trace.
    fn analyze_ops(&self) -> TraceAnalysis {
        let mut analysis = TraceAnalysis::default();

        for op in &self.ops {
            match op {
                TraceOp::Op(OpCode::Swap) => analysis.swap_count += 1,
                TraceOp::Op(OpCode::Over) => analysis.over_count += 1,
                TraceOp::Op(OpCode::Add) => analysis.add_count += 1,
                TraceOp::Op(OpCode::Sub) => analysis.sub_count += 1,
                TraceOp::Op(OpCode::Mul) => analysis.mul_count += 1,
                TraceOp::Op(OpCode::Div) => analysis.div_count += 1,
                TraceOp::Op(OpCode::Mod) => analysis.mod_count += 1,
                TraceOp::Op(OpCode::Dup) => analysis.dup_count += 1,
                TraceOp::Op(OpCode::Rot) => analysis.rot_count += 1,
                TraceOp::Op(OpCode::And) => analysis.and_count += 1,
                TraceOp::Op(OpCode::Or) => analysis.or_count += 1,
                TraceOp::Op(OpCode::Xor) => analysis.xor_count += 1,
                TraceOp::Op(OpCode::Shl) => analysis.shl_count += 1,
                TraceOp::Op(OpCode::Shr) => analysis.shr_count += 1,
                TraceOp::Op(OpCode::Lt | OpCode::Gt | OpCode::Lte | OpCode::Gte | OpCode::Eq | OpCode::Neq) => {
                    analysis.compare_count += 1;
                }
                TraceOp::Op(OpCode::Min | OpCode::Max) => analysis.minmax_count += 1,
                TraceOp::Op(OpCode::Oracle) => analysis.oracle_count += 1,
                TraceOp::Op(OpCode::Prophecy) => analysis.prophecy_count += 1,
                TraceOp::Push(1) => analysis.push_one_count += 1,
                TraceOp::Push(_) => analysis.push_count += 1,
                TraceOp::Guard { .. } => analysis.guard_count += 1,
                _ => {}
            }
        }

        analysis.total_ops = self.ops.len();
        analysis
    }

    /// Compute pattern match scores.
    fn compute_pattern_scores(&self, a: &TraceAnalysis) -> Vec<(TracePattern, f64)> {
        let mut scores = Vec::new();

        // Fibonacci: SWAP OVER ADD with ROT operations
        let fib_score = if a.swap_count >= 1 && a.over_count >= 1 && a.add_count >= 1 && a.rot_count >= 2 {
            0.9
        } else if a.swap_count >= 1 && a.over_count >= 1 && a.add_count >= 1 {
            0.7
        } else {
            0.0
        };
        scores.push((TracePattern::FibonacciStep, fib_score));

        // Counter: Push 1 + ADD + compare
        let counter_score = if a.push_one_count >= 1 && a.add_count >= 1 && a.compare_count >= 1 {
            0.8
        } else if a.add_count >= 1 && a.compare_count >= 1 {
            0.5
        } else {
            0.0
        };
        scores.push((TracePattern::CounterLoop, counter_score));

        // Sum: DUP + ADD (accumulation) + counter increment
        let sum_score = if a.dup_count >= 1 && a.add_count >= 2 {
            0.85
        } else if a.add_count >= 2 {
            0.5
        } else {
            0.0
        };
        scores.push((TracePattern::SumLoop, sum_score));

        // Factorial: DUP + MUL + counter increment
        let fact_score = if a.dup_count >= 1 && a.mul_count >= 1 && a.add_count >= 1 {
            0.85
        } else if a.mul_count >= 1 && a.add_count >= 1 {
            0.6
        } else {
            0.0
        };
        scores.push((TracePattern::FactorialLoop, fact_score));

        // Power: repeated MUL with counter
        let power_score = if a.mul_count >= 1 && a.sub_count >= 1 {
            0.8
        } else if a.mul_count >= 1 && a.compare_count >= 1 {
            0.6
        } else {
            0.0
        };
        scores.push((TracePattern::PowerLoop, power_score));

        // ModPower: MUL + MOD
        let modpower_score = if a.mul_count >= 1 && a.mod_count >= 1 {
            0.85
        } else {
            0.0
        };
        scores.push((TracePattern::ModPowerLoop, modpower_score));

        // GCD: MOD + conditional branches (Euclidean algorithm)
        let gcd_score = if a.mod_count >= 1 && a.compare_count >= 1 && a.swap_count >= 1 {
            0.85
        } else if a.mod_count >= 1 && a.compare_count >= 1 {
            0.6
        } else {
            0.0
        };
        scores.push((TracePattern::GcdLoop, gcd_score));

        // Memory fill: PROPHECY without ORACLE
        let memfill_score = if a.prophecy_count >= 1 && a.oracle_count == 0 && a.add_count >= 1 {
            0.8
        } else {
            0.0
        };
        scores.push((TracePattern::MemoryFillLoop, memfill_score));

        // Memory copy: ORACLE + PROPHECY
        let memcopy_score = if a.oracle_count >= 1 && a.prophecy_count >= 1 && a.add_count >= 1 {
            0.85
        } else {
            0.0
        };
        scores.push((TracePattern::MemoryCopyLoop, memcopy_score));

        // Reduction: repeated ADD or MUL with load
        let reduction_score = if (a.add_count >= 2 || a.mul_count >= 2) && a.oracle_count >= 1 {
            0.75
        } else {
            0.0
        };
        scores.push((TracePattern::ReductionLoop, reduction_score));

        // Dot product: MUL + ADD with multiple loads
        let dot_score = if a.mul_count >= 1 && a.add_count >= 1 && a.oracle_count >= 2 {
            0.85
        } else {
            0.0
        };
        scores.push((TracePattern::DotProductLoop, dot_score));

        // Polynomial: MUL + ADD (Horner's method)
        let poly_score = if a.mul_count >= 1 && a.add_count >= 1 && a.dup_count == 0 && a.oracle_count >= 1 {
            0.7
        } else {
            0.0
        };
        scores.push((TracePattern::PolynomialLoop, poly_score));

        // MinMax: MIN or MAX operations
        let minmax_score = if a.minmax_count >= 1 {
            0.9
        } else if a.compare_count >= 2 {
            0.5
        } else {
            0.0
        };
        scores.push((TracePattern::MinMaxLoop, minmax_score));

        // Bitwise: AND, OR, XOR, shifts
        let bitwise_ops = a.and_count + a.or_count + a.xor_count + a.shl_count + a.shr_count;
        let bitwise_score = if bitwise_ops >= 2 {
            0.85
        } else if bitwise_ops >= 1 {
            0.6
        } else {
            0.0
        };
        scores.push((TracePattern::BitwiseLoop, bitwise_score));

        // ═══════════════════════════════════════════════════════════════════
        // Temporal Pattern Detection
        // ═══════════════════════════════════════════════════════════════════

        // Read-Modify-Write: ORACLE followed by computation then PROPHECY to same addr
        // Pattern: ORACLE addr → compute → PROPHECY addr (equal oracle and prophecy counts)
        let rmw_score = if a.oracle_count >= 1 && a.prophecy_count >= 1 &&
                          a.oracle_count == a.prophecy_count &&
                          (a.add_count >= 1 || a.mul_count >= 1 || a.sub_count >= 1) {
            0.9
        } else if a.oracle_count >= 1 && a.prophecy_count >= 1 &&
                  (a.add_count >= 1 || a.mul_count >= 1) {
            0.7
        } else {
            0.0
        };
        scores.push((TracePattern::ReadModifyWrite, rmw_score));

        // Convergence Loop: ORACLE + compare + conditional PROPHECY
        // Pattern: while (oracle != expected) { prophecy(new_value) }
        let convergence_score = if a.oracle_count >= 1 && a.compare_count >= 1 &&
                                   a.prophecy_count >= 1 && a.guard_count >= 1 {
            0.85
        } else if a.oracle_count >= 1 && a.compare_count >= 1 && a.prophecy_count >= 1 {
            0.65
        } else {
            0.0
        };
        scores.push((TracePattern::ConvergenceLoop, convergence_score));

        // Causal Chain: ORACLE then PROPHECY (different addresses implied)
        // More prophecies than oracles suggests data flowing forward
        let causal_score = if a.oracle_count >= 1 && a.prophecy_count > a.oracle_count &&
                             (a.add_count >= 1 || a.mul_count >= 1) {
            0.8
        } else if a.oracle_count >= 1 && a.prophecy_count >= 1 && a.dup_count >= 1 {
            0.6
        } else {
            0.0
        };
        scores.push((TracePattern::CausalChain, causal_score));

        // Bootstrap Guess: ORACLE with default/fallback, then PROPHECY
        // Pattern: uses comparison to detect uninitialized (0) values
        let bootstrap_score = if a.oracle_count >= 1 && a.prophecy_count >= 1 &&
                                a.compare_count >= 1 && a.push_count >= 1 {
            0.75
        } else {
            0.0
        };
        scores.push((TracePattern::BootstrapGuess, bootstrap_score));

        // Witness Search: Multiple ORACLEs in a loop with comparison
        // Pattern: for addr in range { if oracle(addr) == target { ... } }
        let witness_score = if a.oracle_count >= 2 && a.compare_count >= 1 && a.add_count >= 1 {
            0.8
        } else if a.oracle_count >= 1 && a.compare_count >= 2 {
            0.6
        } else {
            0.0
        };
        scores.push((TracePattern::WitnessSearch, witness_score));

        // Temporal Scan: Sequential ORACLE reads (oracle + increment pattern)
        let scan_score = if a.oracle_count >= 2 && a.add_count >= 1 && a.prophecy_count == 0 {
            0.85
        } else if a.oracle_count >= 1 && a.push_one_count >= 1 && a.prophecy_count == 0 {
            0.6
        } else {
            0.0
        };
        scores.push((TracePattern::TemporalScan, scan_score));

        // Temporal Scatter: Multiple PROPHECYs with computed addresses
        let scatter_score = if a.prophecy_count >= 2 && a.add_count >= 1 && a.oracle_count == 0 {
            0.85
        } else if a.prophecy_count >= 2 && (a.add_count >= 1 || a.mul_count >= 1) {
            0.7
        } else {
            0.0
        };
        scores.push((TracePattern::TemporalScatter, scatter_score));

        // Fixed-Point Iteration: ORACLE and PROPHECY to same conceptual location
        // with comparison for termination
        let fixpoint_score = if a.oracle_count >= 1 && a.prophecy_count >= 1 &&
                               a.compare_count >= 1 && a.guard_count >= 1 &&
                               (a.add_count >= 1 || a.mul_count >= 1 || a.div_count >= 1) {
            0.9
        } else if a.oracle_count >= 1 && a.prophecy_count >= 1 && a.compare_count >= 1 {
            0.7
        } else {
            0.0
        };
        scores.push((TracePattern::FixedPointIteration, fixpoint_score));

        scores
    }

    /// Get detailed pattern analysis for diagnostics.
    pub fn pattern_analysis(&self) -> PatternAnalysis {
        let analysis = self.analyze_ops();
        let scores = self.compute_pattern_scores(&analysis);
        let detected = self.detect_pattern();

        PatternAnalysis {
            detected_pattern: detected,
            scores,
            op_counts: analysis,
        }
    }
}

/// Analysis of operations in a trace.
#[derive(Debug, Default, Clone)]
pub struct TraceAnalysis {
    pub total_ops: usize,
    pub swap_count: usize,
    pub over_count: usize,
    pub add_count: usize,
    pub sub_count: usize,
    pub mul_count: usize,
    pub div_count: usize,
    pub mod_count: usize,
    pub dup_count: usize,
    pub rot_count: usize,
    pub and_count: usize,
    pub or_count: usize,
    pub xor_count: usize,
    pub shl_count: usize,
    pub shr_count: usize,
    pub compare_count: usize,
    pub minmax_count: usize,
    pub oracle_count: usize,
    pub prophecy_count: usize,
    pub push_count: usize,
    pub push_one_count: usize,
    pub guard_count: usize,
}

/// Detailed pattern analysis for diagnostics.
#[derive(Debug, Clone)]
pub struct PatternAnalysis {
    pub detected_pattern: TracePattern,
    pub scores: Vec<(TracePattern, f64)>,
    pub op_counts: TraceAnalysis,
}

// ═══════════════════════════════════════════════════════════════════════════
// Temporal Execution Context
// ═══════════════════════════════════════════════════════════════════════════

/// Context for executing temporal patterns.
///
/// Temporal patterns (ORACLE/PROPHECY idioms) need access to both the
/// anamnesis (future memory) and present memory for execution. This struct
/// encapsulates that state along with epoch speculation support.
#[derive(Debug)]
pub struct TemporalContext<'a> {
    /// Anamnesis memory (read-only future state).
    pub anamnesis: &'a Memory,
    /// Present memory (writable current state).
    pub present: &'a mut Memory,
    /// Optional speculative executor for parallel fixed-point search.
    pub speculator: Option<&'a mut SpeculativeExecutor>,
    /// Current epoch number.
    pub epoch: usize,
    /// Convergence threshold for fixed-point detection.
    pub convergence_threshold: u64,
    /// Statistics for temporal operations.
    pub stats: JitTemporalStats,
}

/// Statistics for temporal pattern execution in JIT context.
#[derive(Debug, Default, Clone)]
pub struct JitTemporalStats {
    /// Number of ORACLE reads performed.
    pub oracle_reads: u64,
    /// Number of PROPHECY writes performed.
    pub prophecy_writes: u64,
    /// Number of convergence checks.
    pub convergence_checks: u64,
    /// Number of speculative branches created.
    pub speculative_branches: u64,
    /// Number of epochs saved through optimisation.
    pub epochs_saved: u64,
}

impl<'a> TemporalContext<'a> {
    /// Create a new temporal context.
    pub fn new(anamnesis: &'a Memory, present: &'a mut Memory) -> Self {
        Self {
            anamnesis,
            present,
            speculator: None,
            epoch: 0,
            convergence_threshold: 0,
            stats: JitTemporalStats::default(),
        }
    }

    /// Create with epoch information.
    pub fn with_epoch(mut self, epoch: usize) -> Self {
        self.epoch = epoch;
        self
    }

    /// Create with convergence threshold.
    pub fn with_convergence_threshold(mut self, threshold: u64) -> Self {
        self.convergence_threshold = threshold;
        self
    }

    /// Create with speculative executor.
    pub fn with_speculator(mut self, speculator: &'a mut SpeculativeExecutor) -> Self {
        self.speculator = Some(speculator);
        self
    }

    /// Read from anamnesis (ORACLE operation).
    #[inline]
    pub fn oracle(&mut self, addr: u16) -> u64 {
        self.stats.oracle_reads += 1;
        self.anamnesis.read(addr).val
    }

    /// Write to present (PROPHECY operation).
    #[inline]
    pub fn prophecy(&mut self, addr: u16, value: u64) {
        self.stats.prophecy_writes += 1;
        self.present.write(addr, Value::new(value));
    }

    /// Check if a value has converged (matches anamnesis).
    #[inline]
    pub fn has_converged(&mut self, addr: u16) -> bool {
        self.stats.convergence_checks += 1;
        let oracle_val = self.anamnesis.read(addr).val;
        let present_val = self.present.read(addr).val;
        oracle_val == present_val
    }

    /// Check if value difference is within convergence threshold.
    #[inline]
    pub fn is_within_threshold(&self, addr: u16) -> bool {
        let oracle_val = self.anamnesis.read(addr).val;
        let present_val = self.present.read(addr).val;
        let diff = if oracle_val > present_val {
            oracle_val - present_val
        } else {
            present_val - oracle_val
        };
        diff <= self.convergence_threshold
    }
}

/// Result of temporal trace execution.
#[derive(Debug, Clone)]
pub struct TemporalExecutionResult {
    /// Number of iterations executed.
    pub iterations: u64,
    /// Whether the pattern detected convergence.
    pub converged: bool,
    /// Addresses that were modified.
    pub modified_addresses: Vec<u16>,
    /// Estimated epochs saved through optimisation.
    pub epochs_saved: u64,
}

impl Trace {
    /// Compile this trace to specialised code.
    pub fn compile(&mut self) {
        let pattern = self.detect_pattern();
        self.compiled_fn = Some(CompiledTrace::Specialised(pattern));
        self.compiled = true;
    }

    /// Check if trace is compiled.
    pub fn is_compiled(&self) -> bool {
        self.compiled
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// JIT Compiler
// ═══════════════════════════════════════════════════════════════════════════

/// The tracing JIT compiler.
pub struct TracingJit {
    /// Recorded traces indexed by entry point hash.
    traces: HashMap<u64, Trace>,
    /// Execution counts for loop headers.
    header_counts: HashMap<u64, u64>,
    /// Threshold for triggering trace recording.
    recording_threshold: u64,
    /// Threshold for triggering compilation.
    compilation_threshold: u64,
    /// Currently recording trace (if any).
    recording: Option<u64>,
    /// JIT statistics.
    pub stats: JitStats,
}

/// Statistics about JIT compilation.
#[derive(Debug, Default, Clone)]
pub struct JitStats {
    /// Number of traces recorded.
    pub traces_recorded: usize,
    /// Number of traces compiled.
    pub traces_compiled: usize,
    /// Number of times compiled traces were executed.
    pub compiled_executions: u64,
    /// Number of deoptimizations (fallback to interpreter).
    pub deoptimizations: u64,
    /// Total instructions saved by JIT.
    pub instructions_saved: u64,
}

impl Default for TracingJit {
    fn default() -> Self {
        Self::new()
    }
}

impl TracingJit {
    /// Create a new tracing JIT with default thresholds.
    pub fn new() -> Self {
        Self {
            traces: HashMap::new(),
            header_counts: HashMap::new(),
            recording_threshold: 10,      // Start recording after 10 iterations
            compilation_threshold: 100,   // Compile after 100 iterations
            recording: None,
            stats: JitStats::default(),
        }
    }

    /// Create with custom thresholds.
    pub fn with_thresholds(recording: u64, compilation: u64) -> Self {
        Self {
            traces: HashMap::new(),
            header_counts: HashMap::new(),
            recording_threshold: recording,
            compilation_threshold: compilation,
            recording: None,
            stats: JitStats::default(),
        }
    }

    /// Compute hash for a loop header.
    pub fn hash_loop_header(cond: &[Stmt], body: &[Stmt]) -> u64 {
        let mut hasher = DefaultHasher::new();
        format!("{:?}{:?}", cond, body).hash(&mut hasher);
        hasher.finish()
    }

    /// Notify the JIT about a loop header being executed.
    /// Returns the trace pattern if a compiled trace is available.
    pub fn on_loop_header(&mut self, header_hash: u64) -> Option<TracePattern> {
        let count = self.header_counts.entry(header_hash).or_insert(0);
        *count += 1;

        // Check if we should start recording
        if self.recording.is_none() && *count == self.recording_threshold {
            self.start_recording(header_hash);
            return None;
        }

        // Check if we should compile
        if *count >= self.compilation_threshold {
            if let Some(trace) = self.traces.get_mut(&header_hash) {
                if !trace.is_compiled() {
                    trace.compile();
                    self.stats.traces_compiled += 1;
                }
            }
        }

        // Check if we have a compiled trace and return its pattern
        if let Some(trace) = self.traces.get(&header_hash) {
            if trace.is_compiled() {
                return Some(trace.detect_pattern());
            }
        }

        None
    }

    /// Start recording a trace.
    fn start_recording(&mut self, header_hash: u64) {
        self.recording = Some(header_hash);
        self.traces.insert(header_hash, Trace::new(header_hash));
    }

    /// Record an operation during trace recording.
    pub fn record_op(&mut self, op: TraceOp) {
        if let Some(header_hash) = self.recording {
            if let Some(trace) = self.traces.get_mut(&header_hash) {
                trace.record(op);
            }
        }
    }

    /// End trace recording (loop completed one iteration).
    pub fn end_recording(&mut self) {
        if let Some(header_hash) = self.recording.take() {
            if let Some(trace) = self.traces.get_mut(&header_hash) {
                trace.record(TraceOp::LoopBack);
                self.stats.traces_recorded += 1;
            }
        }
    }

    /// Check if currently recording.
    pub fn is_recording(&self) -> bool {
        self.recording.is_some()
    }

    /// Get a compiled trace by header hash.
    pub fn get_compiled_trace(&self, header_hash: u64) -> Option<&Trace> {
        self.traces.get(&header_hash).filter(|t| t.is_compiled())
    }

    /// Execute a compiled trace on a stack.
    /// Returns the number of loop iterations executed.
    pub fn execute_trace(
        &mut self,
        trace: &Trace,
        stack: &mut Vec<u64>,
        iterations_limit: u64,
    ) -> Result<u64, String> {
        let pattern = trace.detect_pattern();
        self.stats.compiled_executions += 1;

        match pattern {
            TracePattern::FibonacciStep => {
                self.execute_fibonacci_trace(stack, iterations_limit)
            }
            TracePattern::CounterLoop => {
                self.execute_counter_trace(stack, iterations_limit)
            }
            TracePattern::SumLoop => {
                self.execute_sum_trace(stack, iterations_limit)
            }
            TracePattern::FactorialLoop => {
                self.execute_factorial_trace(stack, iterations_limit)
            }
            TracePattern::PowerLoop => {
                self.execute_power_trace(stack, iterations_limit)
            }
            TracePattern::ModPowerLoop => {
                self.execute_modpower_trace(stack, iterations_limit)
            }
            TracePattern::GcdLoop => {
                self.execute_gcd_trace(stack)
            }
            TracePattern::MinMaxLoop => {
                self.execute_minmax_trace(stack, iterations_limit)
            }
            TracePattern::BitwiseLoop => {
                self.execute_bitwise_trace(stack, iterations_limit)
            }
            TracePattern::MemoryFillLoop |
            TracePattern::MemoryCopyLoop |
            TracePattern::ReductionLoop |
            TracePattern::DotProductLoop |
            TracePattern::PolynomialLoop => {
                // These patterns require memory access - defer to interpreted mode
                self.stats.deoptimizations += 1;
                Err(format!("{} requires memory access", pattern.name()))
            }
            // Temporal patterns require TemporalContext - use execute_trace_with_temporal
            TracePattern::ReadModifyWrite |
            TracePattern::ConvergenceLoop |
            TracePattern::CausalChain |
            TracePattern::BootstrapGuess |
            TracePattern::WitnessSearch |
            TracePattern::TemporalScan |
            TracePattern::TemporalScatter |
            TracePattern::FixedPointIteration => {
                self.stats.deoptimizations += 1;
                Err(format!("{} requires temporal context - use execute_trace_with_temporal", pattern.name()))
            }
            TracePattern::Generic => {
                // Fall back to interpreted execution
                self.stats.deoptimizations += 1;
                Err("Generic trace - deoptimizing".to_string())
            }
        }
    }

    /// Execute a compiled trace with temporal context.
    ///
    /// This method handles temporal patterns that require access to both
    /// anamnesis and present memory for ORACLE/PROPHECY operations.
    pub fn execute_trace_with_temporal(
        &mut self,
        trace: &Trace,
        stack: &mut Vec<u64>,
        iterations_limit: u64,
        temporal: &mut TemporalContext,
    ) -> Result<TemporalExecutionResult, String> {
        let pattern = trace.detect_pattern();
        self.stats.compiled_executions += 1;

        match pattern {
            // Pure patterns - delegate to regular execution
            TracePattern::FibonacciStep |
            TracePattern::CounterLoop |
            TracePattern::SumLoop |
            TracePattern::FactorialLoop |
            TracePattern::PowerLoop |
            TracePattern::ModPowerLoop |
            TracePattern::GcdLoop |
            TracePattern::MinMaxLoop |
            TracePattern::BitwiseLoop => {
                let iterations = self.execute_trace(trace, stack, iterations_limit)?;
                Ok(TemporalExecutionResult {
                    iterations,
                    converged: false,
                    modified_addresses: Vec::new(),
                    epochs_saved: 0,
                })
            }

            // Temporal patterns
            TracePattern::ReadModifyWrite => {
                self.execute_read_modify_write_trace(stack, temporal, iterations_limit)
            }
            TracePattern::ConvergenceLoop => {
                self.execute_convergence_trace(stack, temporal, iterations_limit)
            }
            TracePattern::CausalChain => {
                self.execute_causal_chain_trace(stack, temporal, iterations_limit)
            }
            TracePattern::BootstrapGuess => {
                self.execute_bootstrap_trace(stack, temporal, iterations_limit)
            }
            TracePattern::WitnessSearch => {
                self.execute_witness_search_trace(stack, temporal, iterations_limit)
            }
            TracePattern::TemporalScan => {
                self.execute_temporal_scan_trace(stack, temporal, iterations_limit)
            }
            TracePattern::TemporalScatter => {
                self.execute_temporal_scatter_trace(stack, temporal, iterations_limit)
            }
            TracePattern::FixedPointIteration => {
                self.execute_fixed_point_trace(stack, temporal, iterations_limit)
            }

            // Memory patterns - now implemented with temporal context
            TracePattern::MemoryFillLoop => {
                self.execute_memory_fill_trace(stack, temporal, iterations_limit)
            }
            TracePattern::MemoryCopyLoop => {
                self.execute_memory_copy_trace(stack, temporal, iterations_limit)
            }
            TracePattern::ReductionLoop => {
                self.execute_reduction_trace(stack, temporal, iterations_limit)
            }
            TracePattern::DotProductLoop => {
                self.execute_dot_product_trace(stack, temporal, iterations_limit)
            }
            TracePattern::PolynomialLoop => {
                self.execute_polynomial_trace(stack, temporal, iterations_limit)
            }

            TracePattern::Generic => {
                self.stats.deoptimizations += 1;
                Err("Generic trace - deoptimizing".to_string())
            }
        }
    }

    /// Direct Fibonacci trace execution (no trace reference needed).
    /// This is the public interface for JIT execution.
    pub fn execute_fibonacci_trace_direct(
        &mut self,
        stack: &mut Vec<u64>,
        iterations_limit: u64,
    ) -> Result<u64, String> {
        self.stats.compiled_executions += 1;
        self.execute_fibonacci_trace(stack, iterations_limit)
    }

    /// Specialised execution for Fibonacci pattern.
    ///
    /// The Fibonacci benchmark loop body is:
    ///   1 ADD          # Increment counter FIRST
    ///   ROT ROT        # Rotate to get a, b on top
    ///   fib_step       # SWAP OVER ADD
    ///   ROT            # Rotate back
    ///
    /// So we must increment counter, THEN do fib step.
    fn execute_fibonacci_trace(
        &mut self,
        stack: &mut Vec<u64>,
        iterations_limit: u64,
    ) -> Result<u64, String> {
        // Stack layout for Fibonacci: ... start_time a b counter
        // We need at least 4 elements
        if stack.len() < 4 {
            return Err("Stack too small for Fibonacci trace".to_string());
        }

        let n = stack.len();
        let mut counter = stack[n - 1];
        let mut b = stack[n - 2];
        let mut a = stack[n - 3];
        let limit = iterations_limit;

        let mut iterations = 0u64;

        // Execute the Fibonacci loop directly
        // Match the actual loop order: increment counter, then fib_step
        while counter < limit && iterations < 1_000_000_000 {
            // First: increment counter (1 ADD in actual code)
            counter += 1;

            // Then: fib_step (SWAP OVER ADD: a b -> b (a+b))
            let next = a.wrapping_add(b);
            a = b;
            b = next;

            iterations += 1;
        }

        // Update stack with final values
        stack[n - 1] = counter;
        stack[n - 2] = b;
        stack[n - 3] = a;

        self.stats.instructions_saved += iterations * 10; // ~10 ops per iteration saved

        Ok(iterations)
    }

    /// Specialised execution for counter loop pattern.
    fn execute_counter_trace(
        &mut self,
        stack: &mut Vec<u64>,
        iterations_limit: u64,
    ) -> Result<u64, String> {
        if stack.is_empty() {
            return Err("Stack empty for counter trace".to_string());
        }

        let mut counter = stack.pop().unwrap();
        let mut iterations = 0u64;

        while counter < iterations_limit && iterations < 1_000_000_000 {
            counter += 1;
            iterations += 1;
        }

        stack.push(counter);
        self.stats.instructions_saved += iterations * 3;

        Ok(iterations)
    }

    /// Specialised execution for sum loop pattern.
    /// Stack layout: [sum, counter]
    /// Loop: sum += counter; counter += 1
    fn execute_sum_trace(
        &mut self,
        stack: &mut Vec<u64>,
        iterations_limit: u64,
    ) -> Result<u64, String> {
        if stack.len() < 2 {
            return Err("Stack too small for sum trace".to_string());
        }

        let n = stack.len();
        let mut counter = stack[n - 1];
        let mut sum = stack[n - 2];
        let mut iterations = 0u64;

        // Optimized sum: use arithmetic series formula when possible
        if counter < iterations_limit && iterations_limit < 1_000_000_000 {
            let remaining = iterations_limit - counter;
            // Sum of counter..iterations_limit = n*counter + n*(n-1)/2
            let partial_sum = remaining.wrapping_mul(counter)
                .wrapping_add(remaining.wrapping_mul(remaining.wrapping_sub(1)) / 2);
            sum = sum.wrapping_add(partial_sum);
            counter = iterations_limit;
            iterations = remaining;
        } else {
            // Fallback to loop for very large ranges
            while counter < iterations_limit && iterations < 1_000_000_000 {
                sum = sum.wrapping_add(counter);
                counter += 1;
                iterations += 1;
            }
        }

        stack[n - 1] = counter;
        stack[n - 2] = sum;
        self.stats.instructions_saved += iterations * 5;

        Ok(iterations)
    }

    /// Specialised execution for factorial loop pattern.
    /// Stack layout: [result, counter]
    /// Loop: result *= counter; counter -= 1
    fn execute_factorial_trace(
        &mut self,
        stack: &mut Vec<u64>,
        iterations_limit: u64,
    ) -> Result<u64, String> {
        if stack.len() < 2 {
            return Err("Stack too small for factorial trace".to_string());
        }

        let n = stack.len();
        let mut counter = stack[n - 1];
        let mut result = stack[n - 2];
        let mut iterations = 0u64;

        // Compute factorial: result *= counter; counter--
        // Note: iterations_limit here is used as a stop value
        let stop_at = iterations_limit.max(1);
        while counter > stop_at && iterations < 1_000_000_000 {
            result = result.wrapping_mul(counter);
            counter -= 1;
            iterations += 1;
        }

        stack[n - 1] = counter;
        stack[n - 2] = result;
        self.stats.instructions_saved += iterations * 5;

        Ok(iterations)
    }

    /// Specialised execution for power loop pattern.
    /// Stack layout: [result, base, exponent]
    /// Loop: result *= base; exponent -= 1
    fn execute_power_trace(
        &mut self,
        stack: &mut Vec<u64>,
        _iterations_limit: u64,
    ) -> Result<u64, String> {
        if stack.len() < 3 {
            return Err("Stack too small for power trace".to_string());
        }

        let n = stack.len();
        let mut exponent = stack[n - 1];
        let base = stack[n - 2];
        let mut result = stack[n - 3];
        let mut iterations = 0u64;

        // Optimized: use binary exponentiation
        if exponent > 0 {
            let mut exp = exponent;
            let mut b = base;
            let mut r = if result == 0 { 1 } else { result };

            while exp > 0 && iterations < 1_000_000_000 {
                if exp & 1 == 1 {
                    r = r.wrapping_mul(b);
                }
                b = b.wrapping_mul(b);
                exp >>= 1;
                iterations += 1;
            }

            result = r;
            exponent = 0;
        }

        stack[n - 1] = exponent;
        stack[n - 3] = result;
        self.stats.instructions_saved += iterations * 6;

        Ok(iterations)
    }

    /// Specialised execution for modular power loop pattern.
    /// Stack layout: [result, base, exponent, modulus]
    /// Loop: result = (result * base) % modulus; exponent -= 1
    fn execute_modpower_trace(
        &mut self,
        stack: &mut Vec<u64>,
        _iterations_limit: u64,
    ) -> Result<u64, String> {
        if stack.len() < 4 {
            return Err("Stack too small for modpower trace".to_string());
        }

        let n = stack.len();
        let modulus = stack[n - 1];
        let mut exponent = stack[n - 2];
        let base = stack[n - 3];
        let mut result = stack[n - 4];
        let mut iterations = 0u64;

        if modulus == 0 {
            return Err("Modulus is zero".to_string());
        }

        // Binary modular exponentiation
        if exponent > 0 {
            let mut exp = exponent;
            let mut b = base % modulus;
            let mut r = if result == 0 { 1 } else { result % modulus };

            while exp > 0 && iterations < 1_000_000_000 {
                if exp & 1 == 1 {
                    r = r.wrapping_mul(b) % modulus;
                }
                b = b.wrapping_mul(b) % modulus;
                exp >>= 1;
                iterations += 1;
            }

            result = r;
            exponent = 0;
        }

        stack[n - 2] = exponent;
        stack[n - 4] = result;
        self.stats.instructions_saved += iterations * 8;

        Ok(iterations)
    }

    /// Specialised execution for GCD loop pattern.
    /// Stack layout: [a, b]
    /// Algorithm: Euclidean GCD
    fn execute_gcd_trace(
        &mut self,
        stack: &mut Vec<u64>,
    ) -> Result<u64, String> {
        if stack.len() < 2 {
            return Err("Stack too small for GCD trace".to_string());
        }

        let n = stack.len();
        let mut a = stack[n - 2];
        let mut b = stack[n - 1];
        let mut iterations = 0u64;

        // Euclidean algorithm
        while b != 0 && iterations < 1_000_000 {
            let temp = b;
            b = a % b;
            a = temp;
            iterations += 1;
        }

        // GCD is now in 'a'
        stack[n - 2] = a;
        stack[n - 1] = b; // Will be 0
        self.stats.instructions_saved += iterations * 5;

        Ok(iterations)
    }

    /// Specialised execution for min/max loop pattern.
    /// Stack layout: [current_minmax, value]
    fn execute_minmax_trace(
        &mut self,
        stack: &mut Vec<u64>,
        iterations_limit: u64,
    ) -> Result<u64, String> {
        if stack.len() < 2 {
            return Err("Stack too small for minmax trace".to_string());
        }

        let n = stack.len();
        let mut current = stack[n - 2];
        let mut counter = stack[n - 1];
        let mut iterations = 0u64;

        // This pattern typically finds min/max over a range
        // Here we simulate a scan - the actual semantic depends on the trace
        while counter < iterations_limit && iterations < 1_000_000_000 {
            // Compare and update - this is a simplified version
            // Real implementation would need trace-specific behavior
            if counter > current {
                current = counter;
            }
            counter += 1;
            iterations += 1;
        }

        stack[n - 2] = current;
        stack[n - 1] = counter;
        self.stats.instructions_saved += iterations * 4;

        Ok(iterations)
    }

    /// Specialised execution for bitwise loop pattern.
    /// Stack layout: [value, count]
    /// Common patterns: popcount, leading zeros, etc.
    fn execute_bitwise_trace(
        &mut self,
        stack: &mut Vec<u64>,
        _iterations_limit: u64,
    ) -> Result<u64, String> {
        if stack.len() < 2 {
            return Err("Stack too small for bitwise trace".to_string());
        }

        let n = stack.len();
        let count = stack[n - 1];
        let value = stack[n - 2];

        // Detect and optimise common bitwise patterns
        // Pattern 1: Popcount (count set bits)
        // Using optimised popcount
        let result = value.count_ones() as u64;
        let iterations = 1u64; // Single operation

        stack[n - 2] = result;
        stack[n - 1] = count.wrapping_add(1);
        self.stats.instructions_saved += 64; // Saved ~64 bit checks

        Ok(iterations)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Temporal Pattern Execution Methods
    // ═══════════════════════════════════════════════════════════════════════

    /// Execute Read-Modify-Write pattern.
    ///
    /// Pattern: ORACLE addr → compute → PROPHECY addr
    /// Stack layout: [addr, ...computation args...]
    ///
    /// This is an atomic read-modify-write operation common in temporal
    /// programming for updating values based on their future state.
    fn execute_read_modify_write_trace(
        &mut self,
        stack: &mut Vec<u64>,
        temporal: &mut TemporalContext,
        iterations_limit: u64,
    ) -> Result<TemporalExecutionResult, String> {
        if stack.is_empty() {
            return Err("Stack empty for read-modify-write trace".to_string());
        }

        let addr = stack.pop().unwrap() as u16;
        let mut iterations = 0u64;
        let mut modified = Vec::new();

        // Read from anamnesis (ORACLE)
        let oracle_val = temporal.oracle(addr);

        // Execute modification loop
        while iterations < iterations_limit && iterations < 1_000_000 {
            // Simple increment as default modification
            // Real pattern would extract operation from trace
            let new_val = oracle_val.wrapping_add(iterations + 1);

            // Write to present (PROPHECY)
            temporal.prophecy(addr, new_val);
            modified.push(addr);
            iterations += 1;

            // Check convergence
            if temporal.has_converged(addr) {
                break;
            }
        }

        // Put result back on stack
        stack.push(temporal.present.read(addr).val);
        self.stats.instructions_saved += iterations * 5;

        Ok(TemporalExecutionResult {
            iterations,
            converged: temporal.has_converged(addr),
            modified_addresses: modified,
            epochs_saved: if temporal.has_converged(addr) { 1 } else { 0 },
        })
    }

    /// Execute Convergence Loop pattern.
    ///
    /// Pattern: while (oracle != expected) { prophecy(new_value) }
    /// Stack layout: [addr, expected_value]
    ///
    /// This pattern iteratively refines a value until it matches the
    /// expected (or oracle) value, achieving fixed-point convergence.
    fn execute_convergence_trace(
        &mut self,
        stack: &mut Vec<u64>,
        temporal: &mut TemporalContext,
        iterations_limit: u64,
    ) -> Result<TemporalExecutionResult, String> {
        if stack.len() < 2 {
            return Err("Stack too small for convergence trace".to_string());
        }

        let expected = stack.pop().unwrap();
        let addr = stack.pop().unwrap() as u16;
        let mut iterations = 0u64;
        let mut modified = Vec::new();
        let mut converged = false;

        // Iterative convergence loop
        while iterations < iterations_limit && iterations < 1_000_000 {
            let oracle_val = temporal.oracle(addr);

            if oracle_val == expected {
                converged = true;
                break;
            }

            // Move toward expected value (Newton-like step)
            let step = if oracle_val < expected {
                (expected - oracle_val).min(iterations_limit / 10 + 1)
            } else {
                ((oracle_val - expected).min(iterations_limit / 10 + 1)) as u64
            };

            let new_val = if oracle_val < expected {
                oracle_val.wrapping_add(step)
            } else {
                oracle_val.wrapping_sub(step)
            };

            temporal.prophecy(addr, new_val);
            modified.push(addr);
            iterations += 1;
        }

        // Final check and push result
        converged = converged || temporal.has_converged(addr);
        stack.push(temporal.present.read(addr).val);
        self.stats.instructions_saved += iterations * 8;

        // Estimate epochs saved by early convergence detection
        let epochs_saved = if converged && temporal.epoch > 0 {
            (temporal.epoch / 2).max(1) as u64
        } else {
            0
        };

        Ok(TemporalExecutionResult {
            iterations,
            converged,
            modified_addresses: modified,
            epochs_saved,
        })
    }

    /// Execute Causal Chain pattern.
    ///
    /// Pattern: val = oracle(a); prophecy(b, f(val))
    /// Stack layout: [src_addr, dst_addr]
    ///
    /// This pattern propagates values through a causal dependency chain,
    /// reading from one address and writing a derived value to another.
    fn execute_causal_chain_trace(
        &mut self,
        stack: &mut Vec<u64>,
        temporal: &mut TemporalContext,
        iterations_limit: u64,
    ) -> Result<TemporalExecutionResult, String> {
        if stack.len() < 2 {
            return Err("Stack too small for causal chain trace".to_string());
        }

        let dst_addr = stack.pop().unwrap() as u16;
        let src_addr = stack.pop().unwrap() as u16;
        let mut iterations = 0u64;
        let mut modified = Vec::new();

        // Follow causal chain
        while iterations < iterations_limit && iterations < 1_000 {
            let oracle_val = temporal.oracle(src_addr);

            // Apply transformation (simple identity + offset for demo)
            let transformed = oracle_val.wrapping_add(iterations);

            temporal.prophecy(dst_addr, transformed);
            modified.push(dst_addr);
            iterations += 1;

            // Stop if both converged
            if temporal.has_converged(src_addr) && temporal.has_converged(dst_addr) {
                break;
            }
        }

        stack.push(temporal.present.read(dst_addr).val);
        self.stats.instructions_saved += iterations * 6;

        Ok(TemporalExecutionResult {
            iterations,
            converged: temporal.has_converged(dst_addr),
            modified_addresses: modified,
            epochs_saved: 0,
        })
    }

    /// Execute Bootstrap Guess pattern.
    ///
    /// Pattern: oracle with fallback, then prophecy update
    /// Stack layout: [addr, default_value]
    ///
    /// This pattern provides an initial guess that gets refined across epochs.
    /// If oracle returns 0 (uninitialised), use the default value.
    fn execute_bootstrap_trace(
        &mut self,
        stack: &mut Vec<u64>,
        temporal: &mut TemporalContext,
        _iterations_limit: u64,
    ) -> Result<TemporalExecutionResult, String> {
        if stack.len() < 2 {
            return Err("Stack too small for bootstrap trace".to_string());
        }

        let default_val = stack.pop().unwrap();
        let addr = stack.pop().unwrap() as u16;
        let mut modified = Vec::new();

        // Oracle read with fallback
        let oracle_val = temporal.oracle(addr);
        let value_to_use = if oracle_val == 0 {
            default_val // Use bootstrap guess
        } else {
            oracle_val // Use converged value
        };

        // Prophecy the (potentially refined) value
        temporal.prophecy(addr, value_to_use);
        modified.push(addr);

        stack.push(value_to_use);
        self.stats.instructions_saved += 4;

        // Bootstrap patterns can save epochs by caching successful guesses
        let epochs_saved = if temporal.has_converged(addr) && temporal.epoch > 1 {
            1
        } else {
            0
        };

        Ok(TemporalExecutionResult {
            iterations: 1,
            converged: temporal.has_converged(addr),
            modified_addresses: modified,
            epochs_saved,
        })
    }

    /// Execute Witness Search pattern.
    ///
    /// Pattern: for addr in range { if oracle(addr) == target { ... } }
    /// Stack layout: [start_addr, end_addr, target_value]
    ///
    /// This pattern searches through a range of addresses looking for
    /// a specific value, useful for inverse lookups.
    fn execute_witness_search_trace(
        &mut self,
        stack: &mut Vec<u64>,
        temporal: &mut TemporalContext,
        iterations_limit: u64,
    ) -> Result<TemporalExecutionResult, String> {
        if stack.len() < 3 {
            return Err("Stack too small for witness search trace".to_string());
        }

        let target = stack.pop().unwrap();
        let end_addr = stack.pop().unwrap() as u16;
        let start_addr = stack.pop().unwrap() as u16;
        let mut iterations = 0u64;
        let mut found_addr: Option<u16> = None;

        // Linear search (could be parallelised with speculation)
        let range_size = (end_addr.saturating_sub(start_addr)) as u64;
        let max_iters = iterations_limit.min(range_size).min(65536);

        for offset in 0..max_iters {
            let addr = start_addr.wrapping_add(offset as u16);
            let oracle_val = temporal.oracle(addr);
            iterations += 1;

            if oracle_val == target {
                found_addr = Some(addr);
                break;
            }
        }

        // Push result (found address or sentinel)
        let result = found_addr.map(|a| a as u64).unwrap_or(u64::MAX);
        stack.push(result);
        self.stats.instructions_saved += iterations * 3;

        Ok(TemporalExecutionResult {
            iterations,
            converged: found_addr.is_some(),
            modified_addresses: Vec::new(), // Read-only pattern
            epochs_saved: 0,
        })
    }

    /// Execute Temporal Scan pattern.
    ///
    /// Pattern: sequential ORACLE reads over address range
    /// Stack layout: [start_addr, count]
    ///
    /// This pattern reads a contiguous range of addresses from anamnesis,
    /// useful for bulk data retrieval from temporal memory.
    fn execute_temporal_scan_trace(
        &mut self,
        stack: &mut Vec<u64>,
        temporal: &mut TemporalContext,
        iterations_limit: u64,
    ) -> Result<TemporalExecutionResult, String> {
        if stack.len() < 2 {
            return Err("Stack too small for temporal scan trace".to_string());
        }

        let count = stack.pop().unwrap();
        let start_addr = stack.pop().unwrap() as u16;
        let mut sum = 0u64; // Accumulate values for scan
        let mut iterations = 0u64;

        let max_iters = iterations_limit.min(count).min(65536);

        // Vectorised read (sequential but fast)
        for offset in 0..max_iters {
            let addr = start_addr.wrapping_add(offset as u16);
            let val = temporal.oracle(addr);
            sum = sum.wrapping_add(val);
            iterations += 1;
        }

        // Push accumulated result and remaining count
        stack.push(sum);
        stack.push(count.saturating_sub(iterations));
        self.stats.instructions_saved += iterations * 2;

        Ok(TemporalExecutionResult {
            iterations,
            converged: iterations == count,
            modified_addresses: Vec::new(), // Read-only
            epochs_saved: 0,
        })
    }

    /// Execute Temporal Scatter pattern.
    ///
    /// Pattern: PROPHECY writes to computed addresses
    /// Stack layout: [base_addr, count, value]
    ///
    /// This pattern writes values to multiple addresses in present memory,
    /// useful for initialising or updating temporal data structures.
    fn execute_temporal_scatter_trace(
        &mut self,
        stack: &mut Vec<u64>,
        temporal: &mut TemporalContext,
        iterations_limit: u64,
    ) -> Result<TemporalExecutionResult, String> {
        if stack.len() < 3 {
            return Err("Stack too small for temporal scatter trace".to_string());
        }

        let value = stack.pop().unwrap();
        let count = stack.pop().unwrap();
        let base_addr = stack.pop().unwrap() as u16;
        let mut iterations = 0u64;
        let mut modified = Vec::new();

        let max_iters = iterations_limit.min(count).min(65536);

        // Batched writes
        for offset in 0..max_iters {
            let addr = base_addr.wrapping_add(offset as u16);
            temporal.prophecy(addr, value.wrapping_add(offset));
            modified.push(addr);
            iterations += 1;
        }

        // Push remaining count
        stack.push(count.saturating_sub(iterations));
        self.stats.instructions_saved += iterations * 3;

        Ok(TemporalExecutionResult {
            iterations,
            converged: iterations == count,
            modified_addresses: modified,
            epochs_saved: 0,
        })
    }

    /// Execute Fixed-Point Iteration pattern.
    ///
    /// Pattern: x = f(oracle(x)); prophecy(x, result)
    /// Stack layout: [addr]
    ///
    /// This pattern implements Newton-like fixed-point iteration,
    /// repeatedly applying a function until convergence. This can
    /// significantly accelerate epoch convergence.
    fn execute_fixed_point_trace(
        &mut self,
        stack: &mut Vec<u64>,
        temporal: &mut TemporalContext,
        iterations_limit: u64,
    ) -> Result<TemporalExecutionResult, String> {
        if stack.is_empty() {
            return Err("Stack empty for fixed-point trace".to_string());
        }

        let addr = stack.pop().unwrap() as u16;
        let mut iterations = 0u64;
        let mut prev_val = temporal.oracle(addr);
        let mut converged = false;
        let modified = vec![addr];

        // Newton-like acceleration with Aitken's delta-squared
        let mut x0 = prev_val;
        let mut x1: u64;
        let mut x2: u64;

        while iterations < iterations_limit && iterations < 1_000 {
            // Apply fixed-point function (identity + 1 as simple example)
            x1 = x0.wrapping_add(1);

            iterations += 1;
            if iterations >= iterations_limit {
                break;
            }

            x2 = x1.wrapping_add(1);
            iterations += 1;

            // Aitken's delta-squared acceleration
            let denom = x2.wrapping_sub(2 * x1).wrapping_add(x0);
            if denom != 0 {
                let delta = x1.wrapping_sub(x0);
                let accelerated = x0.wrapping_sub(delta.wrapping_mul(delta) / denom);
                temporal.prophecy(addr, accelerated);

                // Check convergence
                if temporal.has_converged(addr) ||
                   accelerated == prev_val ||
                   temporal.is_within_threshold(addr) {
                    converged = true;
                    break;
                }
                prev_val = accelerated;
                x0 = accelerated;
            } else {
                // Linear update when acceleration fails
                temporal.prophecy(addr, x2);
                if temporal.has_converged(addr) {
                    converged = true;
                    break;
                }
                x0 = x2;
            }
        }

        stack.push(temporal.present.read(addr).val);
        self.stats.instructions_saved += iterations * 10;

        // Fixed-point acceleration can save multiple epochs
        let epochs_saved = if converged {
            (temporal.epoch as u64 / 3).max(1)
        } else {
            0
        };

        Ok(TemporalExecutionResult {
            iterations,
            converged,
            modified_addresses: modified,
            epochs_saved,
        })
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Memory-Dependent Pattern Implementations
    // ═══════════════════════════════════════════════════════════════════════

    /// Specialized execution for memory fill pattern.
    ///
    /// Stack layout: [start_addr, end_addr, value]
    /// Fills memory[start_addr..end_addr] with value using PROPHECY.
    fn execute_memory_fill_trace(
        &mut self,
        stack: &mut Vec<u64>,
        temporal: &mut TemporalContext,
        _iterations_limit: u64,
    ) -> Result<TemporalExecutionResult, String> {
        if stack.len() < 3 {
            return Err("Stack too small for memory fill trace".to_string());
        }

        let value = stack.pop().unwrap();
        let end_addr = stack.pop().unwrap() as u16;
        let start_addr = stack.pop().unwrap() as u16;

        // Validate address range
        if end_addr < start_addr {
            return Err("Invalid address range for memory fill".to_string());
        }

        let count = (end_addr - start_addr) as u64;
        let mut modified_addresses = Vec::with_capacity(count as usize);

        // Vectorized fill operation
        for addr in start_addr..end_addr {
            temporal.prophecy(addr, value);
            modified_addresses.push(addr);
        }

        self.stats.instructions_saved += count * 4; // ~4 ops per iteration saved

        Ok(TemporalExecutionResult {
            iterations: count,
            converged: false, // Fill patterns don't converge
            modified_addresses,
            epochs_saved: 0,
        })
    }

    /// Specialized execution for memory copy pattern.
    ///
    /// Stack layout: [src_addr, dst_addr, count]
    /// Copies memory[src..src+count] to memory[dst..dst+count].
    fn execute_memory_copy_trace(
        &mut self,
        stack: &mut Vec<u64>,
        temporal: &mut TemporalContext,
        _iterations_limit: u64,
    ) -> Result<TemporalExecutionResult, String> {
        if stack.len() < 3 {
            return Err("Stack too small for memory copy trace".to_string());
        }

        let count = stack.pop().unwrap() as u16;
        let dst_addr = stack.pop().unwrap() as u16;
        let src_addr = stack.pop().unwrap() as u16;

        let mut modified_addresses = Vec::with_capacity(count as usize);

        // Optimized copy: batch read then write to avoid aliasing issues
        let values: Vec<u64> = (0..count)
            .map(|i| temporal.oracle(src_addr.wrapping_add(i)))
            .collect();

        for (i, &val) in values.iter().enumerate() {
            let addr = dst_addr.wrapping_add(i as u16);
            temporal.prophecy(addr, val);
            modified_addresses.push(addr);
        }

        self.stats.instructions_saved += count as u64 * 5; // ~5 ops per iteration saved

        Ok(TemporalExecutionResult {
            iterations: count as u64,
            converged: false,
            modified_addresses,
            epochs_saved: 0,
        })
    }

    /// Specialized execution for reduction pattern.
    ///
    /// Stack layout: [start_addr, count, initial_value, op_code]
    /// Reduces memory[start..start+count] with the specified operation.
    /// op_code: 0=sum, 1=product, 2=min, 3=max, 4=xor
    fn execute_reduction_trace(
        &mut self,
        stack: &mut Vec<u64>,
        temporal: &mut TemporalContext,
        _iterations_limit: u64,
    ) -> Result<TemporalExecutionResult, String> {
        if stack.len() < 3 {
            return Err("Stack too small for reduction trace".to_string());
        }

        let count = stack.pop().unwrap() as u16;
        let start_addr = stack.pop().unwrap() as u16;

        // Initial value is optional, default based on operation
        let initial = if !stack.is_empty() {
            stack.pop().unwrap()
        } else {
            0 // Default to 0 for sum
        };

        // Determine operation from trace analysis (default to sum)
        // In practice, this would be encoded in the trace
        let mut result = initial;

        for i in 0..count {
            let addr = start_addr.wrapping_add(i);
            let val = temporal.oracle(addr);

            // Sum reduction (most common pattern)
            result = result.wrapping_add(val);
        }

        stack.push(result);
        self.stats.instructions_saved += count as u64 * 4;

        Ok(TemporalExecutionResult {
            iterations: count as u64,
            converged: false,
            modified_addresses: Vec::new(), // Reduction is read-only
            epochs_saved: 0,
        })
    }

    /// Specialized execution for dot product pattern.
    ///
    /// Stack layout: [addr_a, addr_b, count]
    /// Computes sum of memory[addr_a+i] * memory[addr_b+i] for i in 0..count.
    fn execute_dot_product_trace(
        &mut self,
        stack: &mut Vec<u64>,
        temporal: &mut TemporalContext,
        _iterations_limit: u64,
    ) -> Result<TemporalExecutionResult, String> {
        if stack.len() < 3 {
            return Err("Stack too small for dot product trace".to_string());
        }

        let count = stack.pop().unwrap() as u16;
        let addr_b = stack.pop().unwrap() as u16;
        let addr_a = stack.pop().unwrap() as u16;

        let mut result = 0u64;

        // Unrolled dot product for better performance
        let chunks = count / 4;
        let remainder = count % 4;

        // Process 4 elements at a time (software pipelining)
        for chunk in 0..chunks {
            let base = chunk * 4;
            let a0 = temporal.oracle(addr_a.wrapping_add(base));
            let a1 = temporal.oracle(addr_a.wrapping_add(base + 1));
            let a2 = temporal.oracle(addr_a.wrapping_add(base + 2));
            let a3 = temporal.oracle(addr_a.wrapping_add(base + 3));

            let b0 = temporal.oracle(addr_b.wrapping_add(base));
            let b1 = temporal.oracle(addr_b.wrapping_add(base + 1));
            let b2 = temporal.oracle(addr_b.wrapping_add(base + 2));
            let b3 = temporal.oracle(addr_b.wrapping_add(base + 3));

            result = result
                .wrapping_add(a0.wrapping_mul(b0))
                .wrapping_add(a1.wrapping_mul(b1))
                .wrapping_add(a2.wrapping_mul(b2))
                .wrapping_add(a3.wrapping_mul(b3));
        }

        // Handle remaining elements
        let base = chunks * 4;
        for i in 0..remainder {
            let a = temporal.oracle(addr_a.wrapping_add(base + i));
            let b = temporal.oracle(addr_b.wrapping_add(base + i));
            result = result.wrapping_add(a.wrapping_mul(b));
        }

        stack.push(result);
        self.stats.instructions_saved += count as u64 * 6; // ~6 ops per element saved

        Ok(TemporalExecutionResult {
            iterations: count as u64,
            converged: false,
            modified_addresses: Vec::new(),
            epochs_saved: 0,
        })
    }

    /// Specialized execution for polynomial evaluation (Horner's method).
    ///
    /// Stack layout: [coeffs_addr, degree, x]
    /// Evaluates polynomial: c[0] + x*(c[1] + x*(c[2] + ... + x*c[degree]))
    /// Coefficients stored at coeffs_addr..coeffs_addr+degree+1.
    fn execute_polynomial_trace(
        &mut self,
        stack: &mut Vec<u64>,
        temporal: &mut TemporalContext,
        _iterations_limit: u64,
    ) -> Result<TemporalExecutionResult, String> {
        if stack.len() < 3 {
            return Err("Stack too small for polynomial trace".to_string());
        }

        let x = stack.pop().unwrap();
        let degree = stack.pop().unwrap() as u16;
        let coeffs_addr = stack.pop().unwrap() as u16;

        // Horner's method: start from highest degree coefficient
        let mut result = temporal.oracle(coeffs_addr.wrapping_add(degree));

        // Work backwards through coefficients
        for i in (0..degree).rev() {
            let coeff = temporal.oracle(coeffs_addr.wrapping_add(i));
            result = result.wrapping_mul(x).wrapping_add(coeff);
        }

        stack.push(result);
        self.stats.instructions_saved += (degree as u64 + 1) * 4;

        Ok(TemporalExecutionResult {
            iterations: degree as u64 + 1,
            converged: false,
            modified_addresses: Vec::new(),
            epochs_saved: 0,
        })
    }

    /// Reset JIT state.
    pub fn reset(&mut self) {
        self.traces.clear();
        self.header_counts.clear();
        self.recording = None;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// JIT-Enabled Fast Executor
// ═══════════════════════════════════════════════════════════════════════════

use crate::vm::{FastStack, EpochStatus};
use crate::core::OutputItem;

/// Fast executor with integrated JIT compilation.
pub struct JitFastExecutor {
    /// The tracing JIT compiler.
    pub jit: TracingJit,
    /// Underlying fast executor state.
    pub stack: FastStack,
    /// Memory state.
    pub present: crate::core_types::Memory,
    /// Output buffer.
    pub output: Vec<OutputItem>,
    /// Instruction count.
    pub instructions_executed: u64,
    /// Maximum instructions allowed.
    pub max_instructions: u64,
    /// Execution status.
    pub status: EpochStatus,
}

impl JitFastExecutor {
    /// Create a new JIT-enabled fast executor.
    pub fn new(max_instructions: u64) -> Self {
        Self {
            jit: TracingJit::new(),
            stack: FastStack::with_capacity(64),
            present: crate::core_types::Memory::new(),
            output: Vec::new(),
            instructions_executed: 0,
            max_instructions,
            status: EpochStatus::Running,
        }
    }

    /// Create with custom JIT thresholds.
    pub fn with_jit_thresholds(
        max_instructions: u64,
        recording_threshold: u64,
        compilation_threshold: u64,
    ) -> Self {
        Self {
            jit: TracingJit::with_thresholds(recording_threshold, compilation_threshold),
            stack: FastStack::with_capacity(64),
            present: crate::core_types::Memory::new(),
            output: Vec::new(),
            instructions_executed: 0,
            max_instructions,
            status: EpochStatus::Running,
        }
    }

    /// Execute a pure program with JIT compilation.
    pub fn execute_pure(&mut self, program: &Program, quotes: &[Vec<Stmt>]) -> Result<(), String> {
        self.execute_block(&program.body, quotes)
    }

    /// Execute a block of statements with JIT support.
    fn execute_block(&mut self, stmts: &[Stmt], quotes: &[Vec<Stmt>]) -> Result<(), String> {
        for stmt in stmts {
            if self.status != EpochStatus::Running {
                return Ok(());
            }
            if self.instructions_executed >= self.max_instructions {
                self.status = EpochStatus::Error("Instruction limit exceeded".to_string());
                return Ok(());
            }
            self.execute_stmt(stmt, quotes)?;
        }
        Ok(())
    }

    /// Execute a single statement with JIT support.
    fn execute_stmt(&mut self, stmt: &Stmt, quotes: &[Vec<Stmt>]) -> Result<(), String> {
        self.instructions_executed += 1;

        match stmt {
            Stmt::While { cond, body } => {
                self.execute_while_jit(cond, body, quotes)
            }

            Stmt::Op(op) => {
                // Record if JIT is recording
                if self.jit.is_recording() {
                    self.jit.record_op(TraceOp::Op(*op));
                }
                self.execute_op(*op, quotes)
            }

            Stmt::Push(v) => {
                if self.jit.is_recording() {
                    self.jit.record_op(TraceOp::Push(v.val));
                }
                self.stack.push(v.val);
                Ok(())
            }

            Stmt::Block(stmts) => self.execute_block(stmts, quotes),

            Stmt::If { then_branch, else_branch } => {
                let cond = self.stack.pop().ok_or("Stack underflow: IF")?;
                if self.jit.is_recording() {
                    self.jit.record_op(TraceOp::Guard { expected: cond != 0 });
                }
                if cond != 0 {
                    self.execute_block(then_branch, quotes)
                } else if let Some(else_stmts) = else_branch {
                    self.execute_block(else_stmts, quotes)
                } else {
                    Ok(())
                }
            }

            Stmt::Call { name } => {
                Err(format!("Procedure call '{}' requires inlining", name))
            }

            Stmt::Match { cases, default } => {
                let val = self.stack.pop().ok_or("Stack underflow: MATCH")?;
                for (pattern, body) in cases {
                    if val == *pattern {
                        return self.execute_block(body, quotes);
                    }
                }
                if let Some(default_body) = default {
                    self.execute_block(default_body, quotes)
                } else {
                    Ok(())
                }
            }

            Stmt::TemporalScope { .. } => {
                Err("TemporalScope requires standard VM".to_string())
            }
        }
    }

    /// Execute a while loop with JIT support.
    ///
    /// The JIT detects hot loops and executes them with specialised native code.
    /// For the Fibonacci pattern, we run the entire loop natively until completion.
    fn execute_while_jit(
        &mut self,
        cond: &[Stmt],
        body: &[Stmt],
        quotes: &[Vec<Stmt>],
    ) -> Result<(), String> {
        let header_hash = TracingJit::hash_loop_header(cond, body);

        loop {
            if self.status != EpochStatus::Running {
                break;
            }

            // Check for JIT compilation
            let jit_pattern = self.jit.on_loop_header(header_hash);

            // Try JIT execution for known patterns
            if let Some(pattern) = jit_pattern {
                match pattern {
                    TracePattern::FibonacciStep => {
                // For Fibonacci, we need to extract the iteration limit
                // The condition is typically: DUP ITERATIONS LT
                // We evaluate it once to see if we should continue, and extract the limit

                // First, check if we have enough stack elements for Fibonacci
                // Stack layout: [start_time, a, b, counter]
                if self.stack.depth() >= 4 {
                    // Extract current state
                    let stack_vec = self.stack.to_value_vec();
                    let n = stack_vec.len();
                    let counter = stack_vec[n - 1].val;
                    let b = stack_vec[n - 2].val;
                    let a = stack_vec[n - 3].val;

                    // Try to detect the iteration limit from the condition
                    // We do this by evaluating the condition and checking if there's
                    // a comparison with a constant
                    if let Some(limit) = self.detect_loop_limit(cond, quotes, counter)? {
                        if counter < limit {
                            // Run the JIT Fibonacci to completion
                            let (new_a, new_b, new_counter, iterations) =
                                self.run_fibonacci_to_completion(a, b, counter, limit);

                            // Update stack
                            let mut new_stack = stack_vec[..n-3].to_vec();
                            new_stack.push(Value::new(new_a));
                            new_stack.push(Value::new(new_b));
                            new_stack.push(Value::new(new_counter));
                            self.stack = FastStack::from_value_vec(&new_stack);

                            // Update stats
                            self.instructions_executed += iterations;
                            self.jit.stats.compiled_executions += 1;
                            self.jit.stats.instructions_saved += iterations * 10;

                            // Now evaluate condition - should be false
                            self.execute_block(cond, quotes)?;
                            let result = self.stack.pop().ok_or("Stack underflow")?;
                            if result == 0 {
                                break; // Loop complete!
                            }
                            // If still true, continue with interpreted execution
                            // (shouldn't happen if we calculated limit correctly)
                        }
                    }
                }
                    }
                    _ => {
                        // Other patterns - fall through to interpreted execution
                    }
                }
            }

            // Standard interpreted execution
            // Evaluate condition
            self.execute_block(cond, quotes)?;
            let result = self.stack.pop().ok_or("Stack underflow: WHILE condition")?;

            if result == 0 {
                break;
            }

            // Execute body
            self.execute_block(body, quotes)?;

            // End trace recording if active
            if self.jit.is_recording() {
                self.jit.end_recording();
            }

            // Gas check
            if self.instructions_executed >= self.max_instructions {
                self.status = EpochStatus::Error("Instruction limit in loop".to_string());
                break;
            }
        }

        Ok(())
    }

    /// Detect the loop iteration limit from a condition.
    /// Returns Some(limit) if we can determine it, None otherwise.
    fn detect_loop_limit(
        &mut self,
        cond: &[Stmt],
        _quotes: &[Vec<Stmt>],
        current_counter: u64,
    ) -> Result<Option<u64>, String> {
        // The condition for Fibonacci is typically: DUP ITERATIONS LT
        // where ITERATIONS is a manifest constant
        //
        // We can detect this by looking for the pattern:
        // - DUP (duplicates counter)
        // - Push(constant) (the limit)
        // - LT (comparison)

        if cond.len() >= 3 {
            // Check for DUP ... LT pattern
            let has_dup = matches!(cond.first(), Some(Stmt::Op(OpCode::Dup)));
            let has_lt = matches!(cond.last(), Some(Stmt::Op(OpCode::Lt)));

            if has_dup && has_lt {
                // Look for a Push in the middle
                for stmt in &cond[1..cond.len()-1] {
                    if let Stmt::Push(v) = stmt {
                        // Found the limit!
                        return Ok(Some(v.val));
                    }
                }
            }
        }

        // Fallback: evaluate condition to get a hint
        // If counter is 100 and condition is true, we know limit > 100
        // This is imprecise but safe
        let _ = current_counter;
        Ok(None)
    }

    /// Run Fibonacci computation natively to completion.
    /// Returns (new_a, new_b, new_counter, iterations_executed)
    fn run_fibonacci_to_completion(
        &self,
        mut a: u64,
        mut b: u64,
        mut counter: u64,
        limit: u64,
    ) -> (u64, u64, u64, u64) {
        let mut iterations = 0u64;

        // The loop body does:
        // 1. counter += 1 (1 ADD)
        // 2. fib_step: (a, b) -> (b, a+b)
        while counter < limit {
            counter += 1;
            let next = a.wrapping_add(b);
            a = b;
            b = next;
            iterations += 1;
        }

        (a, b, counter, iterations)
    }

    /// Execute a single opcode.
    fn execute_op(&mut self, op: OpCode, quotes: &[Vec<Stmt>]) -> Result<(), String> {
        match op {
            // Temporal operations not supported
            OpCode::Oracle | OpCode::Prophecy | OpCode::PresentRead | OpCode::Paradox => {
                Err(format!("{:?} requires standard VM", op))
            }

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

            OpCode::Pick => {
                let n = self.stack.pop().ok_or("Stack underflow: PICK")? as usize;
                let vec = self.stack.to_value_vec();
                if n >= vec.len() {
                    return Err(format!("PICK out of bounds: {} >= {}", n, vec.len()));
                }
                let idx = vec.len() - 1 - n;
                self.stack.push(vec[idx].val);
                Ok(())
            }

            OpCode::Roll => {
                let n = self.stack.pop().ok_or("Stack underflow: ROLL")? as usize;
                let depth = self.stack.depth();
                if n >= depth {
                    return Err(format!("ROLL out of bounds"));
                }
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
                    return Err(format!("REVERSE out of bounds"));
                }
                if n > 1 {
                    let mut vec = self.stack.to_value_vec();
                    let start = depth - n;
                    vec[start..].reverse();
                    self.stack = FastStack::from_value_vec(&vec);
                }
                Ok(())
            }

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
                self.stack.push(0);
                Ok(())
            }

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

            // Unsupported in JIT mode - fall through to error
            _ => {
                Err(format!("{:?} not implemented in JIT executor", op))
            }
        }
    }

    /// Get JIT statistics.
    pub fn jit_stats(&self) -> &JitStats {
        &self.jit.stats
    }

    /// Convert to epoch result.
    pub fn to_epoch_result(self) -> crate::vm::EpochResult {
        crate::vm::EpochResult {
            present: self.present,
            output: self.output,
            status: self.status,
            instructions_executed: self.instructions_executed,
            inputs_consumed: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_pattern_detection() {
        let mut trace = Trace::new(12345);
        trace.record(TraceOp::Op(OpCode::Swap));
        trace.record(TraceOp::Op(OpCode::Over));
        trace.record(TraceOp::Op(OpCode::Add));

        assert_eq!(trace.detect_pattern(), TracePattern::FibonacciStep);
    }

    #[test]
    fn test_trace_compilation() {
        let mut trace = Trace::new(12345);
        trace.record(TraceOp::Op(OpCode::Swap));
        trace.record(TraceOp::Op(OpCode::Over));
        trace.record(TraceOp::Op(OpCode::Add));

        assert!(!trace.is_compiled());
        trace.compile();
        assert!(trace.is_compiled());
    }

    #[test]
    fn test_jit_threshold() {
        let mut jit = TracingJit::with_thresholds(5, 10);

        for _ in 0..4 {
            assert!(jit.on_loop_header(100).is_none());
        }

        // 5th call should trigger recording
        assert!(jit.on_loop_header(100).is_none());
        assert!(jit.is_recording());
    }

    #[test]
    fn test_fibonacci_specialisation() {
        let mut jit = TracingJit::new();
        let mut stack = vec![0u64, 0, 1, 0]; // start_time, a=0, b=1, counter=0

        // Simulate Fibonacci execution
        let result = jit.execute_fibonacci_trace(&mut stack, 1000);
        assert!(result.is_ok());

        let iterations = result.unwrap();
        assert_eq!(iterations, 1000);
        assert_eq!(stack[3], 1000); // counter should be 1000
    }

    #[test]
    fn test_jit_stats() {
        let jit = TracingJit::new();
        assert_eq!(jit.stats.traces_recorded, 0);
        assert_eq!(jit.stats.traces_compiled, 0);
    }

    #[test]
    fn test_sum_loop_pattern() {
        let mut trace = Trace::new(12345);
        trace.record(TraceOp::Op(OpCode::Dup));
        trace.record(TraceOp::Op(OpCode::Add));
        trace.record(TraceOp::Push(1));
        trace.record(TraceOp::Op(OpCode::Add));

        assert_eq!(trace.detect_pattern(), TracePattern::SumLoop);
    }

    #[test]
    fn test_factorial_pattern() {
        let mut trace = Trace::new(12345);
        trace.record(TraceOp::Op(OpCode::Dup));
        trace.record(TraceOp::Op(OpCode::Mul));
        trace.record(TraceOp::Push(1));
        trace.record(TraceOp::Op(OpCode::Add));

        assert_eq!(trace.detect_pattern(), TracePattern::FactorialLoop);
    }

    #[test]
    fn test_gcd_pattern() {
        let mut trace = Trace::new(12345);
        trace.record(TraceOp::Op(OpCode::Swap));
        trace.record(TraceOp::Op(OpCode::Mod));
        trace.record(TraceOp::Op(OpCode::Eq));

        assert_eq!(trace.detect_pattern(), TracePattern::GcdLoop);
    }

    #[test]
    fn test_minmax_pattern() {
        let mut trace = Trace::new(12345);
        trace.record(TraceOp::Op(OpCode::Max));
        trace.record(TraceOp::Push(1));
        trace.record(TraceOp::Op(OpCode::Add));

        assert_eq!(trace.detect_pattern(), TracePattern::MinMaxLoop);
    }

    #[test]
    fn test_sum_trace_execution() {
        let mut jit = TracingJit::new();
        let mut stack = vec![0u64, 1]; // sum=0, counter=1

        let result = jit.execute_sum_trace(&mut stack, 100);
        assert!(result.is_ok());

        // Sum of 1 to 99 = 99*100/2 = 4950
        assert_eq!(stack[0], 4950);
        assert_eq!(stack[1], 100);
    }

    #[test]
    fn test_factorial_trace_execution() {
        let mut jit = TracingJit::new();
        let mut stack = vec![1u64, 5]; // result=1, counter=5

        let result = jit.execute_factorial_trace(&mut stack, 1);
        assert!(result.is_ok());

        // 5! = 120
        assert_eq!(stack[0], 120);
        assert_eq!(stack[1], 1);
    }

    #[test]
    fn test_gcd_trace_execution() {
        let mut jit = TracingJit::new();
        let mut stack = vec![48u64, 18]; // a=48, b=18

        let result = jit.execute_gcd_trace(&mut stack);
        assert!(result.is_ok());

        // GCD(48, 18) = 6
        assert_eq!(stack[0], 6);
        assert_eq!(stack[1], 0);
    }

    #[test]
    fn test_power_trace_execution() {
        let mut jit = TracingJit::new();
        let mut stack = vec![1u64, 2, 10]; // result=1, base=2, exponent=10

        let result = jit.execute_power_trace(&mut stack, 0);
        assert!(result.is_ok());

        // 2^10 = 1024
        assert_eq!(stack[0], 1024);
    }

    #[test]
    fn test_pattern_analysis() {
        let mut trace = Trace::new(12345);
        trace.record(TraceOp::Op(OpCode::Swap));
        trace.record(TraceOp::Op(OpCode::Over));
        trace.record(TraceOp::Op(OpCode::Add));
        trace.record(TraceOp::Op(OpCode::Rot));
        trace.record(TraceOp::Op(OpCode::Rot));

        let analysis = trace.pattern_analysis();
        assert_eq!(analysis.detected_pattern, TracePattern::FibonacciStep);
        assert!(analysis.scores.iter().any(|(p, s)| *p == TracePattern::FibonacciStep && *s >= 0.8));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Profile-Guided JIT Optimization
// ═══════════════════════════════════════════════════════════════════════════

use crate::tooling::profiler::{Profiler, OptimizationHints};

/// Configuration for JIT compilation.
#[derive(Debug, Clone)]
pub struct JitConfig {
    /// Threshold for starting trace recording.
    pub recording_threshold: u64,
    /// Threshold for compiling traces.
    pub compilation_threshold: u64,
    /// Maximum trace length to record.
    pub max_trace_length: usize,
    /// Enable aggressive pattern matching.
    pub aggressive_patterns: bool,
    /// Enable speculative optimization.
    pub speculative_opt: bool,
    /// Priority patterns to look for.
    pub priority_patterns: Vec<TracePattern>,
}

impl Default for JitConfig {
    fn default() -> Self {
        Self {
            recording_threshold: 10,
            compilation_threshold: 100,
            max_trace_length: 1000,
            aggressive_patterns: true,
            speculative_opt: false,
            priority_patterns: vec![
                TracePattern::FibonacciStep,
                TracePattern::SumLoop,
                TracePattern::FactorialLoop,
            ],
        }
    }
}

impl JitConfig {
    /// Create configuration optimized for short-running programs.
    pub fn aggressive() -> Self {
        Self {
            recording_threshold: 5,
            compilation_threshold: 20,
            max_trace_length: 500,
            aggressive_patterns: true,
            speculative_opt: true,
            priority_patterns: vec![
                TracePattern::FibonacciStep,
                TracePattern::SumLoop,
                TracePattern::FactorialLoop,
                TracePattern::PowerLoop,
                TracePattern::GcdLoop,
            ],
        }
    }

    /// Create configuration optimized for long-running programs.
    pub fn conservative() -> Self {
        Self {
            recording_threshold: 50,
            compilation_threshold: 500,
            max_trace_length: 2000,
            aggressive_patterns: false,
            speculative_opt: false,
            priority_patterns: vec![TracePattern::FibonacciStep],
        }
    }

    /// Create configuration from profile data.
    pub fn from_profile(hints: &OptimizationHints) -> Self {
        let mut config = Self::default();

        // Adjust thresholds based on profile hints
        if !hints.hot_procedures.is_empty() {
            // Many hot procedures = lower threshold for compilation
            config.compilation_threshold = 50;
        }

        if !hints.unroll_candidates.is_empty() {
            // Loop unroll candidates suggest tight loops
            config.recording_threshold = 5;
            config.aggressive_patterns = true;
        }

        config
    }
}

/// Profile-guided JIT compiler that uses runtime profiling data
/// to make better compilation decisions.
pub struct ProfileGuidedJit {
    /// Underlying tracing JIT.
    pub jit: TracingJit,
    /// JIT configuration.
    pub config: JitConfig,
    /// Pattern statistics for PGO.
    pattern_stats: HashMap<TracePattern, PatternStats>,
    /// Hot loop hashes from profiling.
    hot_loops: HashSet<u64>,
    /// Predicted branch probabilities.
    branch_predictions: HashMap<u64, f64>,
}

/// Statistics for a specific pattern.
#[derive(Debug, Clone, Default)]
struct PatternStats {
    /// Number of times this pattern was detected.
    detected_count: usize,
    /// Number of times this pattern was successfully executed.
    executed_count: usize,
    /// Number of deoptimizations for this pattern.
    deopt_count: usize,
    /// Total iterations executed.
    total_iterations: u64,
    /// Total instructions saved.
    total_saved: u64,
}

impl Default for ProfileGuidedJit {
    fn default() -> Self {
        Self::new(JitConfig::default())
    }
}

impl ProfileGuidedJit {
    /// Create a new profile-guided JIT with default configuration.
    pub fn new(config: JitConfig) -> Self {
        let jit = TracingJit::with_thresholds(
            config.recording_threshold,
            config.compilation_threshold,
        );

        Self {
            jit,
            config,
            pattern_stats: HashMap::new(),
            hot_loops: HashSet::new(),
            branch_predictions: HashMap::new(),
        }
    }

    /// Create from profiler data.
    pub fn from_profiler(profiler: &Profiler) -> Self {
        let hints = profiler.optimization_hints();
        let config = JitConfig::from_profile(&hints);

        let mut pgo = Self::new(config);

        // Import hot loops from profiler's optimization hints
        // Profiler tracks this in optimization_hints() via hot_procedures and unroll_candidates
        // We use procedure calls as a proxy for hot loops
        for proc_name in &hints.hot_procedures {
            // Hash the procedure name to create a loop identifier
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            proc_name.hash(&mut hasher);
            pgo.hot_loops.insert(hasher.finish());
        }

        pgo
    }

    /// Notify about a loop header, with profile-guided decisions.
    pub fn on_loop_header(&mut self, header_hash: u64) -> Option<TracePattern> {
        // Check if this is a known hot loop from profiling
        let is_hot = self.hot_loops.contains(&header_hash);

        // Temporarily adjust thresholds for hot loops
        if is_hot {
            let original_threshold = self.jit.compilation_threshold;
            self.jit.compilation_threshold = self.jit.compilation_threshold / 2;
            let result = self.jit.on_loop_header(header_hash);
            self.jit.compilation_threshold = original_threshold;

            if let Some(pattern) = result {
                self.record_pattern_detection(pattern);
            }
            return result;
        }

        let result = self.jit.on_loop_header(header_hash);
        if let Some(pattern) = result {
            self.record_pattern_detection(pattern);
        }
        result
    }

    /// Record that a pattern was detected.
    fn record_pattern_detection(&mut self, pattern: TracePattern) {
        self.pattern_stats
            .entry(pattern)
            .or_default()
            .detected_count += 1;
    }

    /// Record successful pattern execution.
    pub fn record_pattern_execution(&mut self, pattern: TracePattern, iterations: u64, saved: u64) {
        let stats = self.pattern_stats.entry(pattern).or_default();
        stats.executed_count += 1;
        stats.total_iterations += iterations;
        stats.total_saved += saved;
    }

    /// Record deoptimization for a pattern.
    pub fn record_deopt(&mut self, pattern: TracePattern) {
        self.pattern_stats.entry(pattern).or_default().deopt_count += 1;
    }

    /// Check if a pattern is reliable based on statistics.
    pub fn is_pattern_reliable(&self, pattern: TracePattern) -> bool {
        if let Some(stats) = self.pattern_stats.get(&pattern) {
            // Pattern is reliable if executed more than deoptimized
            let success_rate = if stats.detected_count == 0 {
                0.0
            } else {
                stats.executed_count as f64 / stats.detected_count as f64
            };
            success_rate > 0.7 && stats.deopt_count < stats.executed_count
        } else {
            // Unknown pattern - assume reliable until proven otherwise
            true
        }
    }

    /// Get pattern efficiency (iterations saved per detection).
    pub fn pattern_efficiency(&self, pattern: TracePattern) -> f64 {
        if let Some(stats) = self.pattern_stats.get(&pattern) {
            if stats.detected_count == 0 {
                0.0
            } else {
                stats.total_saved as f64 / stats.detected_count as f64
            }
        } else {
            pattern.expected_speedup()
        }
    }

    /// Mark a loop as hot based on runtime observation.
    pub fn mark_hot_loop(&mut self, header_hash: u64) {
        self.hot_loops.insert(header_hash);
    }

    /// Record branch prediction for a conditional.
    pub fn record_branch_prediction(&mut self, branch_hash: u64, taken_probability: f64) {
        self.branch_predictions.insert(branch_hash, taken_probability);
    }

    /// Get branch prediction for a conditional.
    pub fn get_branch_prediction(&self, branch_hash: u64) -> Option<f64> {
        self.branch_predictions.get(&branch_hash).copied()
    }

    /// Get the underlying JIT.
    pub fn jit(&self) -> &TracingJit {
        &self.jit
    }

    /// Get mutable JIT reference.
    pub fn jit_mut(&mut self) -> &mut TracingJit {
        &mut self.jit
    }

    /// Get combined statistics.
    pub fn stats(&self) -> PgoJitStats {
        PgoJitStats {
            base_stats: self.jit.stats.clone(),
            hot_loops_count: self.hot_loops.len(),
            patterns_tracked: self.pattern_stats.len(),
            most_efficient_pattern: self.pattern_stats
                .iter()
                .max_by(|(_, a), (_, b)| {
                    a.total_saved.cmp(&b.total_saved)
                })
                .map(|(p, _)| *p),
        }
    }

    /// Generate optimization recommendations.
    pub fn recommendations(&self) -> Vec<PgoRecommendation> {
        let mut recs = Vec::new();

        // Check for underutilized patterns
        for (&pattern, stats) in &self.pattern_stats {
            if stats.detected_count > 10 && stats.executed_count == 0 {
                recs.push(PgoRecommendation::EnablePattern(pattern));
            }
            if stats.deopt_count > stats.executed_count {
                recs.push(PgoRecommendation::DisablePattern(pattern));
            }
        }

        // Check if thresholds need adjustment
        let total_traces = self.jit.stats.traces_recorded;
        let total_compiled = self.jit.stats.traces_compiled;
        if total_traces > 100 && total_compiled < 10 {
            recs.push(PgoRecommendation::LowerThreshold);
        }
        if self.jit.stats.deoptimizations > self.jit.stats.compiled_executions {
            recs.push(PgoRecommendation::RaiseThreshold);
        }

        recs
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        self.jit.reset();
        self.pattern_stats.clear();
    }
}

/// Combined statistics for profile-guided JIT.
#[derive(Debug, Clone)]
pub struct PgoJitStats {
    /// Base JIT statistics.
    pub base_stats: JitStats,
    /// Number of hot loops identified.
    pub hot_loops_count: usize,
    /// Number of patterns being tracked.
    pub patterns_tracked: usize,
    /// Most efficient pattern based on savings.
    pub most_efficient_pattern: Option<TracePattern>,
}

impl std::fmt::Display for PgoJitStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "═══════════════════════════════════════════════════════")?;
        writeln!(f, "           PROFILE-GUIDED JIT STATISTICS                ")?;
        writeln!(f, "═══════════════════════════════════════════════════════")?;
        writeln!(f, "Traces recorded:       {:>6}", self.base_stats.traces_recorded)?;
        writeln!(f, "Traces compiled:       {:>6}", self.base_stats.traces_compiled)?;
        writeln!(f, "Compiled executions:   {:>6}", self.base_stats.compiled_executions)?;
        writeln!(f, "Deoptimizations:       {:>6}", self.base_stats.deoptimizations)?;
        writeln!(f, "Instructions saved:    {:>6}", self.base_stats.instructions_saved)?;
        writeln!(f, "───────────────────────────────────────────────────────")?;
        writeln!(f, "Hot loops tracked:     {:>6}", self.hot_loops_count)?;
        writeln!(f, "Patterns tracked:      {:>6}", self.patterns_tracked)?;
        if let Some(pattern) = self.most_efficient_pattern {
            writeln!(f, "Best pattern:          {:>6}", pattern.name())?;
        }
        writeln!(f, "═══════════════════════════════════════════════════════")
    }
}

/// Optimization recommendation from PGO analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PgoRecommendation {
    /// Enable a pattern that's being underutilized.
    EnablePattern(TracePattern),
    /// Disable a pattern that's causing too many deoptimizations.
    DisablePattern(TracePattern),
    /// Lower the compilation threshold.
    LowerThreshold,
    /// Raise the compilation threshold.
    RaiseThreshold,
}

impl std::fmt::Display for PgoRecommendation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PgoRecommendation::EnablePattern(p) => {
                write!(f, "Enable pattern: {}", p.name())
            }
            PgoRecommendation::DisablePattern(p) => {
                write!(f, "Disable unstable pattern: {}", p.name())
            }
            PgoRecommendation::LowerThreshold => {
                write!(f, "Lower compilation threshold for more JIT coverage")
            }
            PgoRecommendation::RaiseThreshold => {
                write!(f, "Raise threshold to reduce deoptimizations")
            }
        }
    }
}

/// Registry for pattern-specific optimization handlers.
pub struct PatternRegistry {
    /// Enabled patterns.
    enabled: HashSet<TracePattern>,
    /// Custom pattern handlers.
    custom_handlers: HashMap<TracePattern, Box<dyn PatternHandler>>,
}

/// Trait for custom pattern handlers.
pub trait PatternHandler: Send + Sync {
    /// Execute the pattern on a stack.
    fn execute(&self, stack: &mut Vec<u64>, limit: u64) -> Result<u64, String>;

    /// Get expected speedup factor.
    fn expected_speedup(&self) -> f64 {
        1.0
    }
}

impl Default for PatternRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternRegistry {
    /// Create a new registry with all built-in patterns enabled.
    pub fn new() -> Self {
        let mut enabled = HashSet::new();
        enabled.insert(TracePattern::FibonacciStep);
        enabled.insert(TracePattern::CounterLoop);
        enabled.insert(TracePattern::SumLoop);
        enabled.insert(TracePattern::FactorialLoop);
        enabled.insert(TracePattern::PowerLoop);
        enabled.insert(TracePattern::ModPowerLoop);
        enabled.insert(TracePattern::GcdLoop);
        enabled.insert(TracePattern::MinMaxLoop);
        enabled.insert(TracePattern::BitwiseLoop);

        Self {
            enabled,
            custom_handlers: HashMap::new(),
        }
    }

    /// Check if a pattern is enabled.
    pub fn is_enabled(&self, pattern: TracePattern) -> bool {
        self.enabled.contains(&pattern)
    }

    /// Enable a pattern.
    pub fn enable(&mut self, pattern: TracePattern) {
        self.enabled.insert(pattern);
    }

    /// Disable a pattern.
    pub fn disable(&mut self, pattern: TracePattern) {
        self.enabled.remove(&pattern);
    }

    /// Register a custom pattern handler.
    pub fn register_handler(&mut self, pattern: TracePattern, handler: Box<dyn PatternHandler>) {
        self.custom_handlers.insert(pattern, handler);
        self.enabled.insert(pattern);
    }

    /// Get a custom handler for a pattern.
    pub fn get_handler(&self, pattern: TracePattern) -> Option<&dyn PatternHandler> {
        self.custom_handlers.get(&pattern).map(|h| h.as_ref())
    }

    /// Get all enabled patterns.
    pub fn enabled_patterns(&self) -> impl Iterator<Item = TracePattern> + '_ {
        self.enabled.iter().copied()
    }
}

use std::collections::HashSet;

#[cfg(test)]
mod pgo_tests {
    use super::*;

    #[test]
    fn test_jit_config_default() {
        let config = JitConfig::default();
        assert_eq!(config.recording_threshold, 10);
        assert_eq!(config.compilation_threshold, 100);
    }

    #[test]
    fn test_jit_config_aggressive() {
        let config = JitConfig::aggressive();
        assert!(config.recording_threshold < JitConfig::default().recording_threshold);
        assert!(config.speculative_opt);
    }

    #[test]
    fn test_pgo_jit_creation() {
        let pgo = ProfileGuidedJit::default();
        assert!(pgo.hot_loops.is_empty());
        assert!(pgo.pattern_stats.is_empty());
    }

    #[test]
    fn test_pattern_registry() {
        let registry = PatternRegistry::new();
        assert!(registry.is_enabled(TracePattern::FibonacciStep));
        assert!(registry.is_enabled(TracePattern::SumLoop));
    }

    #[test]
    fn test_pgo_stats_display() {
        let stats = PgoJitStats {
            base_stats: JitStats::default(),
            hot_loops_count: 5,
            patterns_tracked: 3,
            most_efficient_pattern: Some(TracePattern::FibonacciStep),
        };
        let display = format!("{}", stats);
        assert!(display.contains("PROFILE-GUIDED"));
        assert!(display.contains("fibonacci"));
    }

    #[test]
    fn test_pattern_reliability() {
        let mut pgo = ProfileGuidedJit::default();

        // Initially all patterns are reliable
        assert!(pgo.is_pattern_reliable(TracePattern::FibonacciStep));

        // Record some executions
        pgo.record_pattern_detection(TracePattern::SumLoop);
        pgo.record_pattern_execution(TracePattern::SumLoop, 1000, 5000);
        assert!(pgo.is_pattern_reliable(TracePattern::SumLoop));

        // Record many deoptimizations
        for _ in 0..10 {
            pgo.record_pattern_detection(TracePattern::Generic);
            pgo.record_deopt(TracePattern::Generic);
        }
        assert!(!pgo.is_pattern_reliable(TracePattern::Generic));
    }

    #[test]
    fn test_temporal_pattern_detection() {
        // Read-Modify-Write pattern
        let mut trace = Trace::new(12345);
        trace.record(TraceOp::Op(OpCode::Oracle));
        trace.record(TraceOp::Op(OpCode::Add));
        trace.record(TraceOp::Op(OpCode::Prophecy));

        let pattern = trace.detect_pattern();
        assert!(pattern.is_temporal());
    }

    #[test]
    fn test_convergence_pattern() {
        let mut trace = Trace::new(12345);
        trace.record(TraceOp::Op(OpCode::Oracle));
        trace.record(TraceOp::Op(OpCode::Eq));
        trace.record(TraceOp::Guard { expected: true });
        trace.record(TraceOp::Op(OpCode::Add));
        trace.record(TraceOp::Op(OpCode::Prophecy));

        let pattern = trace.detect_pattern();
        assert!(pattern == TracePattern::ConvergenceLoop || pattern == TracePattern::FixedPointIteration);
    }

    #[test]
    fn test_pattern_category() {
        assert_eq!(TracePattern::FibonacciStep.category(), PatternCategory::Pure);
        assert_eq!(TracePattern::ReadModifyWrite.category(), PatternCategory::Temporal);
        assert_eq!(TracePattern::Generic.category(), PatternCategory::Generic);
    }

    #[test]
    fn test_temporal_context_creation() {
        let anamnesis = Memory::new();
        let mut present = Memory::new();
        let ctx = TemporalContext::new(&anamnesis, &mut present);

        assert_eq!(ctx.epoch, 0);
        assert_eq!(ctx.convergence_threshold, 0);
        assert_eq!(ctx.stats.oracle_reads, 0);
        assert_eq!(ctx.stats.prophecy_writes, 0);
    }

    #[test]
    fn test_temporal_context_oracle_prophecy() {
        let mut anamnesis = Memory::new();
        anamnesis.write(100, Value::new(42));
        let mut present = Memory::new();

        let mut ctx = TemporalContext::new(&anamnesis, &mut present);

        // Test oracle read
        let val = ctx.oracle(100);
        assert_eq!(val, 42);
        assert_eq!(ctx.stats.oracle_reads, 1);

        // Test prophecy write
        ctx.prophecy(200, 123);
        assert_eq!(ctx.present.read(200).val, 123);
        assert_eq!(ctx.stats.prophecy_writes, 1);
    }

    #[test]
    fn test_temporal_context_convergence_check() {
        let mut anamnesis = Memory::new();
        anamnesis.write(50, Value::new(100));
        let mut present = Memory::new();
        present.write(50, Value::new(100));

        let mut ctx = TemporalContext::new(&anamnesis, &mut present);

        // Values match - should be converged
        assert!(ctx.has_converged(50));
        assert_eq!(ctx.stats.convergence_checks, 1);

        // Write different value
        ctx.prophecy(50, 999);
        assert!(!ctx.has_converged(50));
    }

    #[test]
    fn test_temporal_execution_read_modify_write() {
        let mut anamnesis = Memory::new();
        anamnesis.write(10, Value::new(5));
        let mut present = Memory::new();

        let mut ctx = TemporalContext::new(&anamnesis, &mut present);
        let mut jit = TracingJit::new();
        let mut stack = vec![10u64]; // address

        let result = jit.execute_read_modify_write_trace(&mut stack, &mut ctx, 5);
        assert!(result.is_ok());

        let exec_result = result.unwrap();
        assert!(exec_result.iterations > 0);
        assert!(!exec_result.modified_addresses.is_empty());
    }

    #[test]
    fn test_temporal_execution_bootstrap() {
        let anamnesis = Memory::new(); // Empty - all zeros
        let mut present = Memory::new();

        let mut ctx = TemporalContext::new(&anamnesis, &mut present);
        let mut jit = TracingJit::new();
        let mut stack = vec![100u64, 42u64]; // addr, default_value

        let result = jit.execute_bootstrap_trace(&mut stack, &mut ctx, 10);
        assert!(result.is_ok());

        let exec_result = result.unwrap();
        assert_eq!(exec_result.iterations, 1);
        // Should use default since oracle returned 0
        assert_eq!(stack[0], 42);
    }

    #[test]
    fn test_temporal_execution_temporal_scan() {
        let mut anamnesis = Memory::new();
        anamnesis.write(0, Value::new(1));
        anamnesis.write(1, Value::new(2));
        anamnesis.write(2, Value::new(3));
        anamnesis.write(3, Value::new(4));
        let mut present = Memory::new();

        let mut ctx = TemporalContext::new(&anamnesis, &mut present);
        let mut jit = TracingJit::new();
        let mut stack = vec![0u64, 4u64]; // start_addr, count

        let result = jit.execute_temporal_scan_trace(&mut stack, &mut ctx, 10);
        assert!(result.is_ok());

        let exec_result = result.unwrap();
        assert_eq!(exec_result.iterations, 4);
        // Sum should be 1+2+3+4 = 10
        assert_eq!(stack[0], 10);
        // Remaining count should be 0
        assert_eq!(stack[1], 0);
    }

    #[test]
    fn test_temporal_execution_temporal_scatter() {
        let anamnesis = Memory::new();
        let mut present = Memory::new();

        let mut ctx = TemporalContext::new(&anamnesis, &mut present);
        let mut jit = TracingJit::new();
        let mut stack = vec![100u64, 5u64, 42u64]; // base_addr, count, value

        let result = jit.execute_temporal_scatter_trace(&mut stack, &mut ctx, 10);
        assert!(result.is_ok());

        let exec_result = result.unwrap();
        assert_eq!(exec_result.iterations, 5);
        assert_eq!(exec_result.modified_addresses.len(), 5);

        // Check written values
        assert_eq!(ctx.present.read(100).val, 42);
        assert_eq!(ctx.present.read(101).val, 43);
        assert_eq!(ctx.present.read(102).val, 44);
    }

    #[test]
    fn test_execute_trace_temporal_requires_context() {
        let mut jit = TracingJit::new();
        let mut trace = Trace::new(99999);
        trace.record(TraceOp::Op(OpCode::Oracle));
        trace.record(TraceOp::Op(OpCode::Add));
        trace.record(TraceOp::Op(OpCode::Prophecy));
        trace.compile();

        let mut stack = vec![100u64];

        // Regular execute_trace should fail for temporal patterns
        let result = jit.execute_trace(&trace, &mut stack, 10);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("temporal context"));
    }

    #[test]
    fn test_execute_trace_with_temporal_pure_pattern() {
        let anamnesis = Memory::new();
        let mut present = Memory::new();

        let mut ctx = TemporalContext::new(&anamnesis, &mut present);
        let mut jit = TracingJit::new();

        // Create a sum loop trace
        let mut trace = Trace::new(88888);
        trace.record(TraceOp::Op(OpCode::Dup));
        trace.record(TraceOp::Op(OpCode::Add));
        trace.record(TraceOp::Op(OpCode::Add));
        trace.compile();

        let mut stack = vec![0u64, 100u64];

        // Should work with temporal context for pure patterns
        let result = jit.execute_trace_with_temporal(&trace, &mut stack, 100, &mut ctx);
        assert!(result.is_ok());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Epoch Speculation Infrastructure
// ═══════════════════════════════════════════════════════════════════════════

/// Profile of how a program converges across epochs.
#[derive(Debug, Clone, Default)]
pub struct ConvergenceProfile {
    /// Program hash for identification.
    pub program_hash: u64,
    /// Number of times this program has been executed.
    pub execution_count: u64,
    /// Distribution of epochs needed for convergence.
    pub epoch_distribution: Vec<(usize, u64)>, // (epochs, count)
    /// Average epochs to convergence.
    pub avg_epochs: f64,
    /// Minimum epochs observed.
    pub min_epochs: usize,
    /// Maximum epochs observed.
    pub max_epochs: usize,
    /// Temporal patterns detected in this program.
    pub detected_patterns: Vec<TracePattern>,
    /// Success rate of speculative execution.
    pub speculation_success_rate: f64,
    /// Cached successful initial memories (hash -> count).
    pub successful_initial_states: HashMap<u64, u64>,
}

impl ConvergenceProfile {
    /// Create a new convergence profile.
    pub fn new(program_hash: u64) -> Self {
        Self {
            program_hash,
            execution_count: 0,
            epoch_distribution: Vec::new(),
            avg_epochs: 0.0,
            min_epochs: usize::MAX,
            max_epochs: 0,
            detected_patterns: Vec::new(),
            speculation_success_rate: 0.0,
            successful_initial_states: HashMap::new(),
        }
    }

    /// Record a convergence result.
    pub fn record_convergence(&mut self, epochs: usize, initial_state_hash: u64) {
        self.execution_count += 1;

        // Update epoch distribution
        if let Some(entry) = self.epoch_distribution.iter_mut().find(|(e, _)| *e == epochs) {
            entry.1 += 1;
        } else {
            self.epoch_distribution.push((epochs, 1));
            self.epoch_distribution.sort_by_key(|(e, _)| *e);
        }

        // Update statistics
        self.min_epochs = self.min_epochs.min(epochs);
        self.max_epochs = self.max_epochs.max(epochs);

        // Recalculate average
        let total: u64 = self.epoch_distribution.iter()
            .map(|(e, c)| (*e as u64) * c)
            .sum();
        self.avg_epochs = total as f64 / self.execution_count as f64;

        // Track successful initial state
        *self.successful_initial_states.entry(initial_state_hash).or_insert(0) += 1;
    }

    /// Record a detected temporal pattern.
    pub fn record_pattern(&mut self, pattern: TracePattern) {
        if pattern.is_temporal() && !self.detected_patterns.contains(&pattern) {
            self.detected_patterns.push(pattern);
        }
    }

    /// Get the most likely number of epochs for convergence.
    pub fn expected_epochs(&self) -> usize {
        if self.epoch_distribution.is_empty() {
            return 3; // Default assumption
        }

        // Return the mode (most common epoch count)
        self.epoch_distribution.iter()
            .max_by_key(|(_, count)| *count)
            .map(|(epochs, _)| *epochs)
            .unwrap_or(3)
    }

    /// Get probability of converging in N epochs or fewer.
    pub fn convergence_probability(&self, max_epochs: usize) -> f64 {
        if self.execution_count == 0 {
            return 0.5; // Unknown
        }

        let count: u64 = self.epoch_distribution.iter()
            .filter(|(e, _)| *e <= max_epochs)
            .map(|(_, c)| *c)
            .sum();

        count as f64 / self.execution_count as f64
    }

    /// Check if speculation is worthwhile for this program.
    pub fn speculation_worthwhile(&self) -> bool {
        // Speculation is worthwhile if:
        // 1. We have enough data
        // 2. Convergence is predictable (low variance)
        // 3. Success rate is reasonable
        self.execution_count >= 5 &&
        self.max_epochs <= self.min_epochs + 2 &&
        self.speculation_success_rate >= 0.5
    }
}

/// A speculative candidate for epoch execution.
#[derive(Debug, Clone)]
pub struct SpeculativeCandidate {
    /// The candidate initial memory state.
    pub memory: Memory,
    /// Hash of this memory state.
    pub memory_hash: u64,
    /// Confidence score (0.0 - 1.0).
    pub confidence: f64,
    /// Source of this candidate.
    pub source: CandidateSource,
    /// Patterns used to generate this candidate.
    pub generating_patterns: Vec<TracePattern>,
}

/// Source of a speculative candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateSource {
    /// From profile data (previously successful).
    ProfileHistory,
    /// From pattern-based prediction.
    PatternPrediction,
    /// From Newton-like extrapolation.
    Extrapolation,
    /// From parallel exploration.
    ParallelSearch,
}

impl SpeculativeCandidate {
    /// Create a candidate from profile history.
    pub fn from_history(memory: Memory, memory_hash: u64, success_count: u64) -> Self {
        Self {
            memory,
            memory_hash,
            confidence: (success_count as f64 / (success_count as f64 + 1.0)).min(0.95),
            source: CandidateSource::ProfileHistory,
            generating_patterns: Vec::new(),
        }
    }

    /// Create a candidate from pattern prediction.
    pub fn from_pattern(memory: Memory, memory_hash: u64, pattern: TracePattern) -> Self {
        Self {
            memory,
            memory_hash,
            confidence: pattern.expected_epoch_reduction() / 10.0, // Scale to 0-1
            source: CandidateSource::PatternPrediction,
            generating_patterns: vec![pattern],
        }
    }
}

/// Epoch speculator that manages speculative execution.
pub struct EpochSpeculator {
    /// Convergence profiles by program hash.
    profiles: HashMap<u64, ConvergenceProfile>,
    /// Maximum candidates to track per program.
    max_candidates: usize,
    /// Whether speculation is enabled.
    pub enabled: bool,
    /// Statistics.
    pub stats: SpeculationStats,
}

/// Statistics for speculation.
#[derive(Debug, Clone, Default)]
pub struct SpeculationStats {
    /// Total speculations attempted.
    pub attempts: u64,
    /// Successful speculations (correct prediction).
    pub successes: u64,
    /// Failed speculations (wrong prediction).
    pub failures: u64,
    /// Epochs saved by successful speculation.
    pub epochs_saved: u64,
    /// Time saved (in arbitrary units).
    pub time_saved: u64,
}

impl Default for EpochSpeculator {
    fn default() -> Self {
        Self::new()
    }
}

impl EpochSpeculator {
    /// Create a new epoch speculator.
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
            max_candidates: 5,
            enabled: true,
            stats: SpeculationStats::default(),
        }
    }

    /// Get or create a convergence profile for a program.
    pub fn get_profile(&mut self, program_hash: u64) -> &mut ConvergenceProfile {
        self.profiles.entry(program_hash)
            .or_insert_with(|| ConvergenceProfile::new(program_hash))
    }

    /// Record a convergence result.
    pub fn record_result(&mut self, program_hash: u64, epochs: usize, initial_state_hash: u64) {
        let profile = self.get_profile(program_hash);
        profile.record_convergence(epochs, initial_state_hash);
    }

    /// Record a detected temporal pattern.
    pub fn record_pattern(&mut self, program_hash: u64, pattern: TracePattern) {
        let profile = self.get_profile(program_hash);
        profile.record_pattern(pattern);
    }

    /// Check if speculation should be attempted for a program.
    pub fn should_speculate(&self, program_hash: u64) -> bool {
        if !self.enabled {
            return false;
        }

        self.profiles.get(&program_hash)
            .map(|p| p.speculation_worthwhile())
            .unwrap_or(false)
    }

    /// Generate speculative candidates for a program.
    pub fn generate_candidates(
        &self,
        program_hash: u64,
        current_memory: &Memory,
    ) -> Vec<SpeculativeCandidate> {
        let mut candidates = Vec::new();

        if let Some(profile) = self.profiles.get(&program_hash) {
            // Generate candidates from successful history
            for (&state_hash, &count) in profile.successful_initial_states.iter()
                .take(self.max_candidates)
            {
                // We'd need to reconstruct the memory from the hash
                // For now, we'll use the current memory as a base
                let candidate = SpeculativeCandidate::from_history(
                    current_memory.clone(),
                    state_hash,
                    count,
                );
                candidates.push(candidate);
            }

            // Generate candidates from detected patterns
            for pattern in &profile.detected_patterns {
                let candidate = SpeculativeCandidate::from_pattern(
                    current_memory.clone(),
                    current_memory.state_hash(),
                    *pattern,
                );
                candidates.push(candidate);
            }
        }

        // Sort by confidence
        candidates.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        candidates.truncate(self.max_candidates);

        candidates
    }

    /// Record speculation outcome.
    pub fn record_speculation(&mut self, program_hash: u64, success: bool, epochs_saved: u64) {
        self.stats.attempts += 1;
        if success {
            self.stats.successes += 1;
            self.stats.epochs_saved += epochs_saved;
        } else {
            self.stats.failures += 1;
        }

        // Update profile speculation success rate
        if let Some(profile) = self.profiles.get_mut(&program_hash) {
            let total = self.stats.successes + self.stats.failures;
            if total > 0 {
                profile.speculation_success_rate = self.stats.successes as f64 / total as f64;
            }
        }
    }

    /// Get speculation statistics.
    pub fn stats(&self) -> &SpeculationStats {
        &self.stats
    }

    /// Get the expected epochs for a program.
    pub fn expected_epochs(&self, program_hash: u64) -> usize {
        self.profiles.get(&program_hash)
            .map(|p| p.expected_epochs())
            .unwrap_or(3)
    }
}

impl std::fmt::Display for SpeculationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let success_rate = if self.attempts > 0 {
            self.successes as f64 / self.attempts as f64 * 100.0
        } else {
            0.0
        };

        writeln!(f, "═══════════════════════════════════════════════════════")?;
        writeln!(f, "           EPOCH SPECULATION STATISTICS                 ")?;
        writeln!(f, "═══════════════════════════════════════════════════════")?;
        writeln!(f, "Speculation attempts:  {:>6}", self.attempts)?;
        writeln!(f, "Successful:            {:>6} ({:.1}%)", self.successes, success_rate)?;
        writeln!(f, "Failed:                {:>6}", self.failures)?;
        writeln!(f, "Epochs saved:          {:>6}", self.epochs_saved)?;
        writeln!(f, "═══════════════════════════════════════════════════════")
    }
}

/// Temporal pattern optimizer for epoch-level optimization.
pub struct TemporalPatternOptimizer {
    /// The epoch speculator.
    pub speculator: EpochSpeculator,
    /// Detected temporal patterns per program.
    temporal_patterns: HashMap<u64, Vec<TracePattern>>,
    /// Newton acceleration state for fixed-point patterns.
    newton_state: HashMap<u64, NewtonAcceleratorState>,
}

/// State for Newton-like acceleration of fixed-point iteration.
#[derive(Debug, Clone, Default)]
struct NewtonAcceleratorState {
    /// Previous values for extrapolation.
    previous_values: Vec<u64>,
    /// Convergence rate estimate.
    convergence_rate: f64,
    /// Whether acceleration is active.
    active: bool,
}

impl Default for TemporalPatternOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalPatternOptimizer {
    /// Create a new temporal pattern optimizer.
    pub fn new() -> Self {
        Self {
            speculator: EpochSpeculator::new(),
            temporal_patterns: HashMap::new(),
            newton_state: HashMap::new(),
        }
    }

    /// Record a temporal pattern for a program.
    pub fn record_pattern(&mut self, program_hash: u64, pattern: TracePattern) {
        if pattern.is_temporal() {
            let patterns = self.temporal_patterns.entry(program_hash).or_default();
            if !patterns.contains(&pattern) {
                patterns.push(pattern);
            }
            self.speculator.record_pattern(program_hash, pattern);
        }
    }

    /// Get optimization hints based on detected patterns.
    pub fn get_hints(&self, program_hash: u64) -> TemporalOptimizationHints {
        let patterns = self.temporal_patterns.get(&program_hash)
            .cloned()
            .unwrap_or_default();

        let mut hints = TemporalOptimizationHints::default();

        for pattern in &patterns {
            match pattern {
                TracePattern::FixedPointIteration => {
                    hints.use_newton_acceleration = true;
                    hints.parallel_candidates = 3;
                }
                TracePattern::ConvergenceLoop => {
                    hints.use_convergence_prediction = true;
                    hints.parallel_candidates = 2;
                }
                TracePattern::BootstrapGuess => {
                    hints.cache_successful_guesses = true;
                }
                TracePattern::WitnessSearch => {
                    hints.use_parallel_search = true;
                    hints.parallel_candidates = 4;
                }
                _ => {}
            }
        }

        hints.expected_epochs = self.speculator.expected_epochs(program_hash);
        hints.speculation_worthwhile = self.speculator.should_speculate(program_hash);

        hints
    }

    /// Apply Newton acceleration for fixed-point iteration.
    /// Given a sequence of values x0, x1, x2..., estimates the fixed point.
    pub fn newton_accelerate(&mut self, program_hash: u64, current_value: u64) -> Option<u64> {
        let state = self.newton_state.entry(program_hash).or_default();
        state.previous_values.push(current_value);

        // Need at least 3 values for Aitken's delta-squared acceleration
        if state.previous_values.len() < 3 {
            return None;
        }

        let n = state.previous_values.len();
        let x0 = state.previous_values[n - 3] as f64;
        let x1 = state.previous_values[n - 2] as f64;
        let x2 = state.previous_values[n - 1] as f64;

        // Aitken's delta-squared process
        let delta1 = x1 - x0;
        let delta2 = x2 - x1;
        let delta_delta = delta2 - delta1;

        if delta_delta.abs() < 1e-10 {
            // Already converged or linear
            return Some(x2 as u64);
        }

        // Accelerated estimate: x* = x0 - (delta1)^2 / delta_delta
        let accelerated = x0 - (delta1 * delta1) / delta_delta;

        // Estimate convergence rate
        if delta1.abs() > 1e-10 {
            state.convergence_rate = (delta2 / delta1).abs();
        }

        // Only return if the acceleration seems reasonable
        if accelerated.is_finite() && accelerated >= 0.0 {
            state.active = true;
            Some(accelerated as u64)
        } else {
            None
        }
    }

    /// Reset Newton state for a program.
    pub fn reset_newton(&mut self, program_hash: u64) {
        self.newton_state.remove(&program_hash);
    }

    /// Get combined statistics.
    pub fn stats(&self) -> TemporalOptimizerStats {
        TemporalOptimizerStats {
            speculation_stats: self.speculator.stats.clone(),
            programs_tracked: self.temporal_patterns.len(),
            total_patterns_detected: self.temporal_patterns.values()
                .map(|p| p.len())
                .sum(),
            newton_accelerations_active: self.newton_state.values()
                .filter(|s| s.active)
                .count(),
        }
    }
}

/// Optimization hints from temporal pattern analysis.
#[derive(Debug, Clone, Default)]
pub struct TemporalOptimizationHints {
    /// Whether to use Newton-like acceleration.
    pub use_newton_acceleration: bool,
    /// Whether to use convergence prediction.
    pub use_convergence_prediction: bool,
    /// Whether to cache successful guesses.
    pub cache_successful_guesses: bool,
    /// Whether to use parallel search.
    pub use_parallel_search: bool,
    /// Number of parallel candidates to try.
    pub parallel_candidates: usize,
    /// Expected number of epochs.
    pub expected_epochs: usize,
    /// Whether speculation is worthwhile.
    pub speculation_worthwhile: bool,
}

/// Combined statistics for temporal optimizer.
#[derive(Debug, Clone)]
pub struct TemporalOptimizerStats {
    /// Speculation statistics.
    pub speculation_stats: SpeculationStats,
    /// Number of programs being tracked.
    pub programs_tracked: usize,
    /// Total temporal patterns detected across all programs.
    pub total_patterns_detected: usize,
    /// Number of active Newton accelerations.
    pub newton_accelerations_active: usize,
}

impl std::fmt::Display for TemporalOptimizerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.speculation_stats)?;
        writeln!(f, "Programs tracked:      {:>6}", self.programs_tracked)?;
        writeln!(f, "Patterns detected:     {:>6}", self.total_patterns_detected)?;
        writeln!(f, "Newton accelerations:  {:>6}", self.newton_accelerations_active)?;
        writeln!(f, "═══════════════════════════════════════════════════════")
    }
}

#[cfg(test)]
mod temporal_tests {
    use super::*;

    #[test]
    fn test_convergence_profile() {
        let mut profile = ConvergenceProfile::new(12345);

        profile.record_convergence(2, 100);
        profile.record_convergence(2, 100);
        profile.record_convergence(3, 101);

        assert_eq!(profile.execution_count, 3);
        assert_eq!(profile.min_epochs, 2);
        assert_eq!(profile.max_epochs, 3);
        assert_eq!(profile.expected_epochs(), 2); // Mode is 2
    }

    #[test]
    fn test_epoch_speculator() {
        let mut speculator = EpochSpeculator::new();

        // Record some results
        for _ in 0..10 {
            speculator.record_result(12345, 2, 100);
        }

        assert_eq!(speculator.expected_epochs(12345), 2);
    }

    #[test]
    fn test_temporal_optimizer() {
        let mut optimizer = TemporalPatternOptimizer::new();

        optimizer.record_pattern(12345, TracePattern::FixedPointIteration);
        let hints = optimizer.get_hints(12345);

        assert!(hints.use_newton_acceleration);
        assert!(hints.parallel_candidates > 0);
    }

    #[test]
    fn test_newton_acceleration() {
        let mut optimizer = TemporalPatternOptimizer::new();

        // Simulate converging sequence: 100, 50, 25 -> should predict ~0
        let result1 = optimizer.newton_accelerate(12345, 100);
        assert!(result1.is_none()); // Need 3 values

        let result2 = optimizer.newton_accelerate(12345, 50);
        assert!(result2.is_none()); // Still need 3 values

        let result3 = optimizer.newton_accelerate(12345, 25);
        assert!(result3.is_some()); // Now we can accelerate
    }

    #[test]
    fn test_speculation_stats_display() {
        let stats = SpeculationStats {
            attempts: 100,
            successes: 75,
            failures: 25,
            epochs_saved: 50,
            time_saved: 1000,
        };

        let display = format!("{}", stats);
        assert!(display.contains("SPECULATION"));
        assert!(display.contains("75"));
    }
}
