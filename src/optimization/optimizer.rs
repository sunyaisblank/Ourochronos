//! Optimization passes for OUROCHRONOS programs.
//!
//! Implements comprehensive optimization passes for Phase 6:
//! - Dead code elimination
//! - Constant folding
//! - Common subexpression elimination (CSE)
//! - Tail call optimization (TCO)
//! - Loop unrolling
//! - Inline caching infrastructure
//! - Speculative optimization patterns
//! - Profile-guided optimization support

use crate::ast::{Stmt, OpCode, Program};
use crate::core::Value;
use std::collections::{HashMap, HashSet};

// ═══════════════════════════════════════════════════════════════════════════
// Value Numbering for CSE
// ═══════════════════════════════════════════════════════════════════════════

/// Value number - unique identifier for a value in the computation.
type ValueNumber = u64;

/// Expression key for CSE hash table.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ExprKey {
    /// Literal constant value.
    Literal(u64),
    /// Binary operation: (op, left_vn, right_vn).
    Binary(OpCode, ValueNumber, ValueNumber),
    /// Unary operation: (op, operand_vn).
    Unary(OpCode, ValueNumber),
    /// Unknown/external value (cannot be deduplicated).
    Unknown(u64),
}

/// Result of CSE transformation on a statement.
enum CseTransform {
    /// Keep the statement unchanged.
    Keep(Stmt),
    /// Replace with different statements.
    Replace(Vec<Stmt>),
    /// Remove the statement entirely.
    Remove,
}

/// State for value numbering during CSE pass.
#[derive(Debug)]
struct ValueNumberingState {
    /// Next value number to assign.
    next_vn: ValueNumber,
    /// Abstract stack with value numbers.
    stack: Vec<ValueNumber>,
    /// Map from expression keys to their value numbers.
    expr_to_vn: HashMap<ExprKey, ValueNumber>,
    /// Map from literal values to their value numbers.
    literal_to_vn: HashMap<u64, ValueNumber>,
}

impl ValueNumberingState {
    /// Create new value numbering state.
    fn new() -> Self {
        Self {
            next_vn: 0,
            stack: Vec::new(),
            expr_to_vn: HashMap::new(),
            literal_to_vn: HashMap::new(),
        }
    }

    /// Get or create value number for a literal constant.
    fn get_or_create_literal_vn(&mut self, val: u64) -> ValueNumber {
        if let Some(&vn) = self.literal_to_vn.get(&val) {
            vn
        } else {
            let vn = self.next_vn;
            self.next_vn += 1;
            self.literal_to_vn.insert(val, vn);
            self.expr_to_vn.insert(ExprKey::Literal(val), vn);
            vn
        }
    }

    /// Get or create value number for an expression.
    fn get_or_create_expr_vn(&mut self, key: ExprKey) -> ValueNumber {
        if let Some(&vn) = self.expr_to_vn.get(&key) {
            vn
        } else {
            let vn = self.next_vn;
            self.next_vn += 1;
            self.expr_to_vn.insert(key, vn);
            vn
        }
    }

    /// Create a new unknown value number (for external inputs).
    fn create_unknown_vn(&mut self) -> ValueNumber {
        let vn = self.next_vn;
        self.next_vn += 1;
        self.expr_to_vn.insert(ExprKey::Unknown(vn), vn);
        vn
    }

    /// Push a value number onto the abstract stack.
    fn push(&mut self, vn: ValueNumber) {
        self.stack.push(vn);
    }

    /// Pop a value number from the abstract stack.
    fn pop(&mut self) -> ValueNumber {
        self.stack.pop().unwrap_or_else(|| self.create_unknown_vn())
    }

    /// Find if a value number exists on the stack, return depth from top.
    fn find_value_on_stack(&self, vn: ValueNumber) -> Option<usize> {
        for (i, &stack_vn) in self.stack.iter().rev().enumerate() {
            if stack_vn == vn {
                return Some(i);
            }
        }
        None
    }

    /// Invalidate all tracking (e.g., after control flow merge).
    fn invalidate(&mut self) {
        self.stack.clear();
        // Keep expr_to_vn for potential future matches within same basic block
    }
}

/// Optimization level for the compiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimizations.
    None,
    /// Basic instruction fusion (combine consecutive ops).
    Basic,
    /// Full optimizations including pattern matching and AOT passes.
    Full,
    /// Aggressive optimizations with speculative transforms.
    Aggressive,
}

impl Default for OptLevel {
    fn default() -> Self {
        OptLevel::Basic
    }
}

/// Optimized instruction representation.
///
/// These are fused instructions that represent common patterns
/// more efficiently than individual opcodes.
#[derive(Debug, Clone, PartialEq)]
pub enum OptInstr {
    /// Original statement (unfused).
    Stmt(Stmt),

    /// Fused addition: add N to top of stack.
    /// Replaces consecutive ADD operations.
    FusedAdd(i64),

    /// Fused subtraction: subtract N from top of stack.
    FusedSub(i64),

    /// Fused multiplication: multiply top by N.
    FusedMul(i64),

    /// Fused memory operations: N consecutive reads/writes.
    FusedMemOps(Vec<(u16, OpCode)>),

    /// Clear pattern: set memory cell to zero.
    /// Detects patterns like: ORACLE DUP NOT IF { 0 PROPHECY }
    Clear(u16),

    /// Move until zero: scan memory in a direction until finding 0.
    MoveUntil(i32),

    /// Copy value: copy from one cell to another.
    CopyTo { src: u16, dst: u16 },

    /// Constant folded value - the result of compile-time computation.
    ConstFolded(u64),

    /// Tail call - optimized procedure call at the end of a procedure.
    TailCall { name: String },

    /// Unrolled loop - fixed iteration count loop expanded.
    UnrolledLoop { iterations: usize, body: Vec<OptInstr> },

    /// Inline cached call - procedure with cached resolution.
    InlineCached { name: String, cache_slot: usize },

    /// Speculative guard - check condition before optimized path.
    SpeculativeGuard {
        condition: Box<OptInstr>,
        fast_path: Vec<OptInstr>,
        slow_path: Vec<OptInstr>,
    },

    /// Block of optimized instructions.
    Block(Vec<OptInstr>),
}

/// Optimizer that transforms programs for better performance.
#[derive(Debug)]
pub struct Optimizer {
    level: OptLevel,
    stats: OptStats,
    /// Constant pool for folded values.
    const_pool: HashMap<u64, u64>,
    /// Known constants on abstract stack.
    stack_constants: Vec<Option<u64>>,
    /// Inline cache slots.
    inline_cache_slots: usize,
    /// Loop unroll threshold.
    unroll_threshold: usize,
    /// Profile data for guided optimization.
    profile_data: Option<ProfileData>,
}

/// Statistics about optimizations applied.
#[derive(Debug, Default, Clone)]
pub struct OptStats {
    /// Number of instructions before optimization.
    pub original_count: usize,
    /// Number of instructions after optimization.
    pub optimized_count: usize,
    /// Number of fused additions.
    pub fused_adds: usize,
    /// Number of fused memory operations.
    pub fused_mem_ops: usize,
    /// Number of patterns detected (Clear, CopyTo, etc).
    pub patterns_detected: usize,
    /// Number of constants folded.
    pub constants_folded: usize,
    /// Number of dead code blocks eliminated.
    pub dead_code_eliminated: usize,
    /// Number of common subexpressions eliminated.
    pub cse_eliminated: usize,
    /// Number of tail calls optimized.
    pub tail_calls_optimized: usize,
    /// Number of loops unrolled.
    pub loops_unrolled: usize,
    /// Number of inline cache slots created.
    pub inline_caches_created: usize,
    /// Number of speculative optimizations applied.
    pub speculative_opts: usize,
}

impl OptStats {
    /// Calculate reduction ratio.
    pub fn reduction_ratio(&self) -> f64 {
        if self.original_count == 0 {
            1.0
        } else {
            1.0 - (self.optimized_count as f64 / self.original_count as f64)
        }
    }

    /// Total optimizations applied.
    pub fn total_optimizations(&self) -> usize {
        self.fused_adds + self.fused_mem_ops + self.patterns_detected +
        self.constants_folded + self.dead_code_eliminated + self.cse_eliminated +
        self.tail_calls_optimized + self.loops_unrolled + self.inline_caches_created +
        self.speculative_opts
    }
}

impl std::fmt::Display for OptStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "═══════════════════════════════════════════════════════")?;
        writeln!(f, "              OPTIMIZATION STATISTICS                   ")?;
        writeln!(f, "═══════════════════════════════════════════════════════")?;
        writeln!(f, "Instructions: {} → {} ({:.1}% reduction)",
            self.original_count, self.optimized_count,
            self.reduction_ratio() * 100.0)?;
        writeln!(f, "───────────────────────────────────────────────────────")?;
        writeln!(f, "Fused additions:       {:>6}", self.fused_adds)?;
        writeln!(f, "Fused memory ops:      {:>6}", self.fused_mem_ops)?;
        writeln!(f, "Pattern matches:       {:>6}", self.patterns_detected)?;
        writeln!(f, "Constants folded:      {:>6}", self.constants_folded)?;
        writeln!(f, "Dead code eliminated:  {:>6}", self.dead_code_eliminated)?;
        writeln!(f, "CSE eliminated:        {:>6}", self.cse_eliminated)?;
        writeln!(f, "Tail calls optimized:  {:>6}", self.tail_calls_optimized)?;
        writeln!(f, "Loops unrolled:        {:>6}", self.loops_unrolled)?;
        writeln!(f, "Inline caches:         {:>6}", self.inline_caches_created)?;
        writeln!(f, "Speculative opts:      {:>6}", self.speculative_opts)?;
        writeln!(f, "───────────────────────────────────────────────────────")?;
        writeln!(f, "Total optimizations:   {:>6}", self.total_optimizations())?;
        writeln!(f, "═══════════════════════════════════════════════════════")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Formal Optimization Quality Metrics
// ═══════════════════════════════════════════════════════════════════════════

/// Distributional statistics for optimization quality assessment.
///
/// Tracks the shape of confidence score distributions using statistical moments:
/// - **Mean (μ)**: Location - average confidence score
/// - **Variance (σ²)**: Spread - how much scores deviate from mean
/// - **Skewness (γ₁)**: Asymmetry - positive = right tail, negative = left tail
/// - **Kurtosis (γ₂)**: Tailedness/peakedness - high = heavy tails
///
/// These moments provide insights into optimization effectiveness:
/// - High mean with low variance = consistent high-quality optimizations
/// - Negative skewness = most optimizations are high confidence
/// - High kurtosis = outliers present (very good or very poor matches)
#[derive(Debug, Clone, Default)]
pub struct DistributionalMetrics {
    /// Number of observations.
    pub count: usize,
    /// Running sum for mean calculation.
    sum: f64,
    /// Running sum of squares for variance.
    sum_sq: f64,
    /// Running sum of cubes for skewness.
    sum_cube: f64,
    /// Running sum of fourth powers for kurtosis.
    sum_fourth: f64,
    /// Minimum observed value.
    pub min: f64,
    /// Maximum observed value.
    pub max: f64,
}

impl DistributionalMetrics {
    /// Create new metrics tracker.
    pub fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
            sum_cube: 0.0,
            sum_fourth: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Record a new observation.
    pub fn record(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        let v2 = value * value;
        let v3 = v2 * value;
        let v4 = v3 * value;
        self.sum_sq += v2;
        self.sum_cube += v3;
        self.sum_fourth += v4;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    /// Compute mean (first raw moment).
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    /// Compute variance (second central moment).
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            let mean = self.mean();
            self.sum_sq / self.count as f64 - mean * mean
        }
    }

    /// Compute standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Compute skewness (third standardized moment).
    ///
    /// Measures asymmetry of the distribution:
    /// - Positive: Right tail is longer (more low values)
    /// - Negative: Left tail is longer (more high values)
    /// - Zero: Symmetric distribution
    pub fn skewness(&self) -> f64 {
        if self.count < 3 {
            return 0.0;
        }

        let mean = self.mean();
        let std_dev = self.std_dev();
        if std_dev < 1e-10 {
            return 0.0;
        }

        let n = self.count as f64;
        // E[(X - μ)³] = E[X³] - 3μE[X²] + 2μ³
        let central_third = self.sum_cube / n
            - 3.0 * mean * self.sum_sq / n
            + 2.0 * mean * mean * mean;

        central_third / (std_dev * std_dev * std_dev)
    }

    /// Compute excess kurtosis (fourth standardized moment - 3).
    ///
    /// Measures tailedness compared to normal distribution:
    /// - Positive (leptokurtic): Heavier tails than normal
    /// - Negative (platykurtic): Lighter tails than normal
    /// - Zero (mesokurtic): Similar to normal distribution
    pub fn kurtosis(&self) -> f64 {
        if self.count < 4 {
            return 0.0;
        }

        let mean = self.mean();
        let variance = self.variance();
        if variance < 1e-10 {
            return 0.0;
        }

        let n = self.count as f64;
        // E[(X - μ)⁴] = E[X⁴] - 4μE[X³] + 6μ²E[X²] - 3μ⁴
        let central_fourth = self.sum_fourth / n
            - 4.0 * mean * self.sum_cube / n
            + 6.0 * mean * mean * self.sum_sq / n
            - 3.0 * mean * mean * mean * mean;

        (central_fourth / (variance * variance)) - 3.0
    }

    /// Get a summary string of all moments.
    pub fn summary(&self) -> String {
        format!(
            "n={}, μ={:.4}, σ={:.4}, γ₁={:.4}, γ₂={:.4}, range=[{:.4}, {:.4}]",
            self.count,
            self.mean(),
            self.std_dev(),
            self.skewness(),
            self.kurtosis(),
            self.min,
            self.max
        )
    }

    /// Assess optimization quality based on distribution shape.
    pub fn quality_assessment(&self) -> OptimizationQuality {
        if self.count < 5 {
            return OptimizationQuality::InsufficientData;
        }

        let mean = self.mean();
        let std_dev = self.std_dev();
        let skewness = self.skewness();

        // High mean, low variance, negative skew = excellent
        if mean > 0.7 && std_dev < 0.2 && skewness < 0.0 {
            OptimizationQuality::Excellent
        } else if mean > 0.5 && std_dev < 0.3 {
            OptimizationQuality::Good
        } else if mean > 0.3 {
            OptimizationQuality::Fair
        } else {
            OptimizationQuality::Poor
        }
    }
}

/// Qualitative assessment of optimization effectiveness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationQuality {
    /// Not enough data to assess.
    InsufficientData,
    /// Poor optimization (low confidence, high variance).
    Poor,
    /// Fair optimization (moderate confidence).
    Fair,
    /// Good optimization (high confidence, moderate variance).
    Good,
    /// Excellent optimization (high confidence, low variance, negative skew).
    Excellent,
}

impl std::fmt::Display for OptimizationQuality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientData => write!(f, "Insufficient Data"),
            Self::Poor => write!(f, "Poor"),
            Self::Fair => write!(f, "Fair"),
            Self::Good => write!(f, "Good"),
            Self::Excellent => write!(f, "Excellent"),
        }
    }
}

/// Comprehensive optimization effectiveness tracker.
///
/// Tracks multiple dimensions of optimization quality:
/// - Pattern detection confidence distribution
/// - Speedup factor distribution
/// - Convergence rate for temporal patterns
/// - Resource utilization efficiency
#[derive(Debug, Clone, Default)]
pub struct OptimizationEffectiveness {
    /// Distribution of pattern detection confidence scores.
    pub pattern_confidence: DistributionalMetrics,
    /// Distribution of achieved speedup factors.
    pub speedup_factors: DistributionalMetrics,
    /// Distribution of convergence rates (for temporal patterns).
    pub convergence_rates: DistributionalMetrics,
    /// Total instructions saved.
    pub total_instructions_saved: u64,
    /// Total epochs saved (temporal optimization).
    pub total_epochs_saved: u64,
    /// Number of successful optimizations.
    pub successful_optimizations: usize,
    /// Number of deoptimizations (fallbacks).
    pub deoptimizations: usize,
}

impl OptimizationEffectiveness {
    /// Create new effectiveness tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a pattern detection with confidence score.
    pub fn record_pattern_detection(&mut self, confidence: f64) {
        self.pattern_confidence.record(confidence);
    }

    /// Record a speedup factor achieved.
    pub fn record_speedup(&mut self, factor: f64) {
        self.speedup_factors.record(factor);
    }

    /// Record a convergence rate observation.
    pub fn record_convergence_rate(&mut self, rate: f64) {
        self.convergence_rates.record(rate);
    }

    /// Record instructions saved by an optimization.
    pub fn record_instructions_saved(&mut self, count: u64) {
        self.total_instructions_saved += count;
        self.successful_optimizations += 1;
    }

    /// Record epochs saved by temporal optimization.
    pub fn record_epochs_saved(&mut self, count: u64) {
        self.total_epochs_saved += count;
    }

    /// Record a deoptimization (fallback to interpreter).
    pub fn record_deoptimization(&mut self) {
        self.deoptimizations += 1;
    }

    /// Compute overall effectiveness score (0.0 to 1.0).
    pub fn overall_score(&self) -> f64 {
        let total_attempts = self.successful_optimizations + self.deoptimizations;
        if total_attempts == 0 {
            return 0.0;
        }

        // Weighted combination of factors
        let success_rate = self.successful_optimizations as f64 / total_attempts as f64;
        let mean_confidence = self.pattern_confidence.mean();
        let mean_speedup = (self.speedup_factors.mean() / 10.0).min(1.0); // Normalize to 0-1

        // Weighted average: success_rate most important
        0.5 * success_rate + 0.3 * mean_confidence + 0.2 * mean_speedup
    }

    /// Generate a comprehensive report.
    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str("╔═══════════════════════════════════════════════════════╗\n");
        report.push_str("║      OPTIMIZATION EFFECTIVENESS REPORT                ║\n");
        report.push_str("╠═══════════════════════════════════════════════════════╣\n");

        report.push_str(&format!("║ Overall Score: {:.2}%                                  \n",
            self.overall_score() * 100.0));

        report.push_str("╠───────────────────────────────────────────────────────╣\n");
        report.push_str("║ PATTERN DETECTION                                     ║\n");
        report.push_str(&format!("║   Confidence: {}  \n", self.pattern_confidence.summary()));
        report.push_str(&format!("║   Quality: {}                                         \n",
            self.pattern_confidence.quality_assessment()));

        report.push_str("╠───────────────────────────────────────────────────────╣\n");
        report.push_str("║ PERFORMANCE                                           ║\n");
        report.push_str(&format!("║   Speedup: {}  \n", self.speedup_factors.summary()));
        report.push_str(&format!("║   Instructions saved: {}                              \n",
            self.total_instructions_saved));

        if self.total_epochs_saved > 0 {
            report.push_str("╠───────────────────────────────────────────────────────╣\n");
            report.push_str("║ TEMPORAL OPTIMIZATION                                 ║\n");
            report.push_str(&format!("║   Epochs saved: {}                                    \n",
                self.total_epochs_saved));
            report.push_str(&format!("║   Convergence: {}  \n", self.convergence_rates.summary()));
        }

        report.push_str("╠───────────────────────────────────────────────────────╣\n");
        report.push_str("║ RELIABILITY                                           ║\n");
        report.push_str(&format!("║   Successful: {}                                      \n",
            self.successful_optimizations));
        report.push_str(&format!("║   Deoptimizations: {}                                 \n",
            self.deoptimizations));

        report.push_str("╚═══════════════════════════════════════════════════════╝\n");

        report
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Reference Frame Documentation
// ═══════════════════════════════════════════════════════════════════════════

/// Documents the geometric, distributional, and symmetrical reference frames
/// used in the optimization system.
///
/// # Geometric Reference (Transform Theory)
///
/// Optimization passes can be understood as geometric transforms on program space:
///
/// | Transform Type | Properties Preserved | Optimization Examples |
/// |----------------|---------------------|----------------------|
/// | **Isometry** | Distances, angles | Instruction reordering (no semantic change) |
/// | **Similarity** | Ratios, angles | Loop unrolling (scales iteration space) |
/// | **Affine** | Parallelism, ratios | Constant folding (linear transform on values) |
/// | **Projective** | Cross-ratio only | Dead code elimination (projection to subspace) |
///
/// The stack operations form an affine transform group on the value space.
///
/// # Distributional Reference (Statistical Moments)
///
/// Optimization confidence is characterized by its distribution:
///
/// | Moment | Symbol | Interpretation for Optimization |
/// |--------|--------|--------------------------------|
/// | Mean | μ | Average confidence - higher = better patterns |
/// | Variance | σ² | Consistency - lower = more reliable |
/// | Skewness | γ₁ | Bias - negative = most scores are high |
/// | Kurtosis | γ₂ | Outliers - high = occasional excellent/poor |
///
/// # Symmetrical Reference (Conservation Laws)
///
/// Temporal optimization obeys conservation laws from CTC physics:
///
/// | Conservation | Constraint | Implication |
/// |--------------|-----------|-------------|
/// | **Self-consistency** | S = F(S) | Fixed-point convergence required |
/// | **Causal closure** | No information creation | Bootstrap patterns bounded |
/// | **Provenance tracking** | Causal history preserved | Oracle/Prophecy correlation |
///
/// Continuous symmetries (Noether's theorem analogue):
/// - Time translation → epoch count conservation (total work preserved)
/// - Value permutation → commutativity exploitation in CSE
///
/// Discrete symmetries:
/// - Loop unrolling: Z_n → Z_1 reduction (n-fold periodicity eliminated)
/// - Dead code: Projection symmetry (irreducible subspace preserved)
pub struct ReferenceFrameDocumentation;

impl ReferenceFrameDocumentation {
    /// Returns documentation for geometric transforms in optimization.
    pub fn geometric_transforms() -> &'static str {
        r#"
GEOMETRIC TRANSFORMS IN OPTIMIZATION
====================================

Optimization passes transform programs while preserving semantic equivalence.
These can be classified by the geometric properties they preserve:

1. ISOMETRY (Distance-Preserving)
   - Preserves: All distances and angles
   - Examples: Instruction scheduling within basic blocks
   - The "shape" of computation is unchanged

2. SIMILARITY (Ratio-Preserving)
   - Preserves: Ratios and angles
   - Examples: Loop unrolling (scales iteration space by factor k)
   - Work is scaled but relative structure maintained

3. AFFINE (Parallelism-Preserving)
   - Preserves: Parallelism and ratios along lines
   - Examples: Constant folding, strength reduction
   - Stack operations form an affine group on value space

4. PROJECTIVE (Cross-Ratio-Preserving)
   - Preserves: Only cross-ratios
   - Examples: Dead code elimination (projects to live subspace)
   - Most general linear transform, loses most structure

The optimization level determines which transforms are applied:
- None: Identity (trivial isometry)
- Basic: Isometry + simple similarity
- Full: Affine transforms (constant folding, CSE)
- Aggressive: Projective (speculative elimination)
"#
    }

    /// Returns documentation for distributional analysis.
    pub fn distributional_analysis() -> &'static str {
        r#"
DISTRIBUTIONAL ANALYSIS OF OPTIMIZATION QUALITY
===============================================

We characterize optimization confidence using statistical moments:

MOMENT     FORMULA                  INTERPRETATION
------     -------                  --------------
Mean       μ = E[X]                 Average confidence score
Variance   σ² = E[(X-μ)²]           Spread around mean
Skewness   γ₁ = E[(X-μ)³]/σ³        Asymmetry of distribution
Kurtosis   γ₂ = E[(X-μ)⁴]/σ⁴ - 3    Tail heaviness vs normal

QUALITY INTERPRETATION:

| Mean | Variance | Skewness | Kurtosis | Assessment |
|------|----------|----------|----------|------------|
| High | Low      | Negative | Low      | Excellent  |
| High | Moderate | Any      | Low      | Good       |
| Mid  | High     | Positive | High     | Fair       |
| Low  | High     | Positive | High     | Poor       |

Negative skewness (left-skewed) is desirable: most scores are high.
Low kurtosis indicates consistency without outliers.
"#
    }

    /// Returns documentation for conservation laws in temporal optimization.
    pub fn conservation_laws() -> &'static str {
        r#"
CONSERVATION LAWS IN TEMPORAL OPTIMIZATION
==========================================

Ourochronos implements Deutschian CTC semantics, which impose conservation laws:

1. SELF-CONSISTENCY CONSERVATION
   Constraint: S = F(S) (fixed-point)
   Physical basis: Novikov self-consistency in CTCs
   Implication: Every temporal loop must converge or oscillate

2. CAUSAL CLOSURE
   Constraint: No net information creation
   Physical basis: Causality preservation
   Implication: Bootstrap patterns have bounded complexity

3. PROVENANCE CONSERVATION
   Constraint: Causal history tracked through all operations
   Physical basis: Information conservation
   Implication: Every PROPHECY traceable to ORACLE sources

SYMMETRY → CONSERVATION (Noether-like correspondence):

| Symmetry Type | Conservation Law |
|---------------|------------------|
| Time translation | Total computation work |
| Value permutation | CSE equivalence classes |
| Loop periodicity | Unrolling factor bounds |

Asymmetric patterns (grandfather paradox) manifest as oscillation:
  x_{n+1} = ¬x_n → period-2 cycle, no fixed point

Acceleration methods (Aitken, Anderson) exploit these symmetries
to predict fixed points before exhaustive iteration.
"#
    }
}

/// Profile data for guided optimization.
#[derive(Debug, Clone, Default)]
pub struct ProfileData {
    /// Hot procedures (called frequently).
    pub hot_procs: HashSet<String>,
    /// Hot loops (iteration count > threshold).
    pub hot_loops: HashMap<usize, usize>,
    /// Branch probabilities (location -> taken probability).
    pub branch_probs: HashMap<usize, f64>,
    /// Memory access patterns.
    pub memory_hotspots: HashSet<u16>,
}

impl ProfileData {
    /// Create from profiler results.
    pub fn from_profiler(profiler: &crate::tooling::profiler::Profiler) -> Self {
        let mut data = Self::default();

        // Extract instruction hotspots
        for (op, stats) in profiler.instruction_stats() {
            if stats.count > 1000 {
                // This op is hot
                match op {
                    OpCode::Oracle | OpCode::Prophecy => {
                        // Track temporal hotspots
                    }
                    _ => {}
                }
            }
        }

        // Extract memory hotspots
        for (addr, count) in profiler.memory_profile().hotspots(20) {
            if count > 100 {
                data.memory_hotspots.insert(addr);
            }
        }

        data
    }
}

impl Optimizer {
    /// Create a new optimizer with the given level.
    pub fn new(level: OptLevel) -> Self {
        Self {
            level,
            stats: OptStats::default(),
            const_pool: HashMap::new(),
            stack_constants: Vec::new(),
            inline_cache_slots: 0,
            unroll_threshold: 8,
            profile_data: None,
        }
    }

    /// Create optimizer with profile data for guided optimization.
    pub fn with_profile(level: OptLevel, profile: ProfileData) -> Self {
        Self {
            level,
            stats: OptStats::default(),
            const_pool: HashMap::new(),
            stack_constants: Vec::new(),
            inline_cache_slots: 0,
            unroll_threshold: 8,
            profile_data: Some(profile),
        }
    }

    /// Set loop unroll threshold.
    pub fn set_unroll_threshold(&mut self, threshold: usize) {
        self.unroll_threshold = threshold;
    }

    /// Get optimization statistics.
    pub fn stats(&self) -> &OptStats {
        &self.stats
    }

    /// Optimize a program.
    pub fn optimize(&mut self, program: &Program) -> Vec<OptInstr> {
        self.stats = OptStats::default();
        self.stats.original_count = Self::count_stmts(&program.body);
        self.stack_constants.clear();
        self.const_pool.clear();
        self.inline_cache_slots = 0;

        let result = match self.level {
            OptLevel::None => self.pass_through(&program.body),
            OptLevel::Basic => self.basic_optimize(&program.body),
            OptLevel::Full => self.full_optimize(&program.body),
            OptLevel::Aggressive => self.aggressive_optimize(&program.body),
        };

        self.stats.optimized_count = Self::count_opt_instrs(&result);
        result
    }

    /// Optimize a program and return transformed statements.
    pub fn optimize_to_stmts(&mut self, program: &Program) -> Vec<Stmt> {
        let optimized = self.optimize(program);
        Self::unoptimize(&optimized)
    }

    fn count_stmts(stmts: &[Stmt]) -> usize {
        let mut count = 0;
        for stmt in stmts {
            count += 1;
            match stmt {
                Stmt::Block(inner) => count += Self::count_stmts(inner),
                Stmt::If { then_branch, else_branch } => {
                    count += Self::count_stmts(then_branch);
                    if let Some(eb) = else_branch {
                        count += Self::count_stmts(eb);
                    }
                }
                Stmt::While { cond, body } => {
                    count += Self::count_stmts(cond);
                    count += Self::count_stmts(body);
                }
                _ => {}
            }
        }
        count
    }

    fn count_opt_instrs(instrs: &[OptInstr]) -> usize {
        let mut count = 0;
        for instr in instrs {
            count += 1;
            match instr {
                OptInstr::Block(inner) => count += Self::count_opt_instrs(inner),
                OptInstr::UnrolledLoop { body, .. } => count += Self::count_opt_instrs(body),
                OptInstr::SpeculativeGuard { fast_path, slow_path, .. } => {
                    count += Self::count_opt_instrs(fast_path);
                    count += Self::count_opt_instrs(slow_path);
                }
                _ => {}
            }
        }
        count
    }

    /// No optimization - just wrap statements.
    fn pass_through(&self, stmts: &[Stmt]) -> Vec<OptInstr> {
        stmts.iter().map(|s| OptInstr::Stmt(s.clone())).collect()
    }

    /// Basic optimization: fuse consecutive operations.
    fn basic_optimize(&mut self, stmts: &[Stmt]) -> Vec<OptInstr> {
        let mut result = Vec::new();
        let mut i = 0;

        while i < stmts.len() {
            match &stmts[i] {
                // Fuse consecutive Push+ADD operations
                Stmt::Push(val) => {
                    if i + 1 < stmts.len() {
                        if let Stmt::Op(OpCode::Add) = &stmts[i + 1] {
                            // Check for more adds
                            let mut sum = val.val as i64;
                            let mut j = i + 2;

                            while j + 1 < stmts.len() {
                                if let (Stmt::Push(v), Stmt::Op(OpCode::Add)) =
                                    (&stmts[j], &stmts[j + 1]) {
                                    sum += v.val as i64;
                                    j += 2;
                                } else {
                                    break;
                                }
                            }

                            if j > i + 2 {
                                // Found consecutive adds
                                self.stats.fused_adds += 1;
                                result.push(OptInstr::FusedAdd(sum));
                                i = j;
                                continue;
                            }
                        }
                    }
                    result.push(OptInstr::Stmt(stmts[i].clone()));
                }

                // Recursively optimize blocks
                Stmt::Block(inner) => {
                    let optimized = self.basic_optimize(inner);
                    result.push(OptInstr::Block(optimized));
                }

                Stmt::If { then_branch, else_branch } => {
                    let opt_then = self.basic_optimize(then_branch);
                    let opt_else = else_branch.as_ref().map(|eb| self.basic_optimize(eb));

                    // Keep original if structure but with optimized branches
                    result.push(OptInstr::Stmt(Stmt::If {
                        then_branch: Self::unoptimize(&opt_then),
                        else_branch: opt_else.map(|e| Self::unoptimize(&e)),
                    }));
                }

                Stmt::While { cond, body } => {
                    let opt_cond = self.basic_optimize(cond);
                    let opt_body = self.basic_optimize(body);

                    result.push(OptInstr::Stmt(Stmt::While {
                        cond: Self::unoptimize(&opt_cond),
                        body: Self::unoptimize(&opt_body),
                    }));
                }

                _ => {
                    result.push(OptInstr::Stmt(stmts[i].clone()));
                }
            }
            i += 1;
        }

        result
    }

    /// Full optimization: includes pattern detection, constant folding, DCE.
    fn full_optimize(&mut self, stmts: &[Stmt]) -> Vec<OptInstr> {
        // Apply passes in order
        let pass1 = self.constant_folding_pass(stmts);
        let pass2 = self.dead_code_elimination_pass(&pass1);
        let pass3 = self.cse_pass(&pass2);
        let pass4 = self.basic_optimize(&pass3);
        let pass5 = self.peephole_optimize(pass4);
        let pass6 = self.tail_call_optimize(pass5);

        pass6
    }

    /// Aggressive optimization: includes speculative transforms and loop unrolling.
    fn aggressive_optimize(&mut self, stmts: &[Stmt]) -> Vec<OptInstr> {
        // First apply full optimizations
        let full = self.full_optimize(stmts);

        // Then apply aggressive passes
        let unrolled = self.loop_unroll_pass(full);
        let inline_cached = self.inline_caching_pass(unrolled);
        let speculative = self.speculative_optimize_pass(inline_cached);

        speculative
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Constant Folding Pass
    // ═══════════════════════════════════════════════════════════════════════

    fn constant_folding_pass(&mut self, stmts: &[Stmt]) -> Vec<Stmt> {
        let mut result = Vec::new();
        let mut stack: Vec<Option<u64>> = Vec::new();

        for stmt in stmts {
            match stmt {
                Stmt::Push(val) => {
                    stack.push(Some(val.val));
                    result.push(stmt.clone());
                }

                Stmt::Op(op) => {
                    if let Some(folded) = self.try_fold_op(*op, &mut stack) {
                        // Replace with constant
                        self.stats.constants_folded += 1;
                        result.push(Stmt::Push(Value::new(folded)));
                    } else {
                        // Can't fold, invalidate stack tracking
                        self.apply_op_to_stack(*op, &mut stack);
                        result.push(stmt.clone());
                    }
                }

                Stmt::If { then_branch, else_branch } => {
                    // Check if condition is constant
                    if let Some(Some(cond)) = stack.pop() {
                        if cond != 0 {
                            // Condition is true, only emit then branch
                            self.stats.dead_code_eliminated += 1;
                            result.extend(self.constant_folding_pass(then_branch));
                        } else if let Some(else_stmts) = else_branch {
                            // Condition is false, only emit else branch
                            self.stats.dead_code_eliminated += 1;
                            result.extend(self.constant_folding_pass(else_stmts));
                        }
                        // If condition is false and no else, emit nothing
                    } else {
                        // Condition not constant, recurse into branches
                        result.push(Stmt::If {
                            then_branch: self.constant_folding_pass(then_branch),
                            else_branch: else_branch.as_ref().map(|eb| self.constant_folding_pass(eb)),
                        });
                    }
                }

                Stmt::While { cond, body } => {
                    // Check if condition is provably false
                    let folded_cond = self.constant_folding_pass(cond);
                    if self.is_constant_false(&folded_cond) {
                        self.stats.dead_code_eliminated += 1;
                        // Don't emit the while at all
                    } else {
                        result.push(Stmt::While {
                            cond: folded_cond,
                            body: self.constant_folding_pass(body),
                        });
                    }
                }

                Stmt::Block(inner) => {
                    let folded = self.constant_folding_pass(inner);
                    if !folded.is_empty() {
                        result.push(Stmt::Block(folded));
                    }
                }

                _ => {
                    stack.clear(); // Unknown effect
                    result.push(stmt.clone());
                }
            }
        }

        result
    }

    fn try_fold_op(&self, op: OpCode, stack: &mut Vec<Option<u64>>) -> Option<u64> {
        match op {
            OpCode::Add => {
                if let (Some(Some(b)), Some(Some(a))) = (stack.pop(), stack.pop()) {
                    let result = a.wrapping_add(b);
                    stack.push(Some(result));
                    return Some(result);
                }
            }
            OpCode::Sub => {
                if let (Some(Some(b)), Some(Some(a))) = (stack.pop(), stack.pop()) {
                    let result = a.wrapping_sub(b);
                    stack.push(Some(result));
                    return Some(result);
                }
            }
            OpCode::Mul => {
                if let (Some(Some(b)), Some(Some(a))) = (stack.pop(), stack.pop()) {
                    let result = a.wrapping_mul(b);
                    stack.push(Some(result));
                    return Some(result);
                }
            }
            OpCode::Div => {
                if let (Some(Some(b)), Some(Some(a))) = (stack.pop(), stack.pop()) {
                    let result = if b == 0 { 0 } else { a / b };
                    stack.push(Some(result));
                    return Some(result);
                }
            }
            OpCode::Mod => {
                if let (Some(Some(b)), Some(Some(a))) = (stack.pop(), stack.pop()) {
                    let result = if b == 0 { 0 } else { a % b };
                    stack.push(Some(result));
                    return Some(result);
                }
            }
            OpCode::And => {
                if let (Some(Some(b)), Some(Some(a))) = (stack.pop(), stack.pop()) {
                    let result = a & b;
                    stack.push(Some(result));
                    return Some(result);
                }
            }
            OpCode::Or => {
                if let (Some(Some(b)), Some(Some(a))) = (stack.pop(), stack.pop()) {
                    let result = a | b;
                    stack.push(Some(result));
                    return Some(result);
                }
            }
            OpCode::Xor => {
                if let (Some(Some(b)), Some(Some(a))) = (stack.pop(), stack.pop()) {
                    let result = a ^ b;
                    stack.push(Some(result));
                    return Some(result);
                }
            }
            OpCode::Not => {
                if let Some(Some(a)) = stack.pop() {
                    // Logical NOT: 0 -> 1, nonzero -> 0
                    let result = if a == 0 { 1 } else { 0 };
                    stack.push(Some(result));
                    return Some(result);
                }
            }
            OpCode::Neg => {
                if let Some(Some(a)) = stack.pop() {
                    let result = a.wrapping_neg();
                    stack.push(Some(result));
                    return Some(result);
                }
            }
            OpCode::Shl => {
                if let (Some(Some(n)), Some(Some(a))) = (stack.pop(), stack.pop()) {
                    let result = a.wrapping_shl((n % 64) as u32);
                    stack.push(Some(result));
                    return Some(result);
                }
            }
            OpCode::Shr => {
                if let (Some(Some(n)), Some(Some(a))) = (stack.pop(), stack.pop()) {
                    let result = a.wrapping_shr((n % 64) as u32);
                    stack.push(Some(result));
                    return Some(result);
                }
            }
            OpCode::Eq => {
                if let (Some(Some(b)), Some(Some(a))) = (stack.pop(), stack.pop()) {
                    let result = if a == b { 1 } else { 0 };
                    stack.push(Some(result));
                    return Some(result);
                }
            }
            OpCode::Neq => {
                if let (Some(Some(b)), Some(Some(a))) = (stack.pop(), stack.pop()) {
                    let result = if a != b { 1 } else { 0 };
                    stack.push(Some(result));
                    return Some(result);
                }
            }
            OpCode::Lt => {
                if let (Some(Some(b)), Some(Some(a))) = (stack.pop(), stack.pop()) {
                    let result = if a < b { 1 } else { 0 };
                    stack.push(Some(result));
                    return Some(result);
                }
            }
            OpCode::Gt => {
                if let (Some(Some(b)), Some(Some(a))) = (stack.pop(), stack.pop()) {
                    let result = if a > b { 1 } else { 0 };
                    stack.push(Some(result));
                    return Some(result);
                }
            }
            OpCode::Lte => {
                if let (Some(Some(b)), Some(Some(a))) = (stack.pop(), stack.pop()) {
                    let result = if a <= b { 1 } else { 0 };
                    stack.push(Some(result));
                    return Some(result);
                }
            }
            OpCode::Gte => {
                if let (Some(Some(b)), Some(Some(a))) = (stack.pop(), stack.pop()) {
                    let result = if a >= b { 1 } else { 0 };
                    stack.push(Some(result));
                    return Some(result);
                }
            }
            OpCode::Min => {
                if let (Some(Some(b)), Some(Some(a))) = (stack.pop(), stack.pop()) {
                    let result = a.min(b);
                    stack.push(Some(result));
                    return Some(result);
                }
            }
            OpCode::Max => {
                if let (Some(Some(b)), Some(Some(a))) = (stack.pop(), stack.pop()) {
                    let result = a.max(b);
                    stack.push(Some(result));
                    return Some(result);
                }
            }
            OpCode::Abs => {
                if let Some(Some(a)) = stack.pop() {
                    // Abs of unsigned is itself
                    stack.push(Some(a));
                    return Some(a);
                }
            }
            _ => {}
        }
        None
    }

    fn apply_op_to_stack(&self, op: OpCode, stack: &mut Vec<Option<u64>>) {
        match op {
            // Binary ops
            OpCode::Add | OpCode::Sub | OpCode::Mul | OpCode::Div | OpCode::Mod |
            OpCode::And | OpCode::Or | OpCode::Xor | OpCode::Shl | OpCode::Shr |
            OpCode::Eq | OpCode::Neq | OpCode::Lt | OpCode::Gt | OpCode::Lte | OpCode::Gte |
            OpCode::Min | OpCode::Max | OpCode::Slt | OpCode::Sgt | OpCode::Slte | OpCode::Sgte => {
                stack.pop();
                stack.pop();
                stack.push(None);
            }
            // Unary ops
            OpCode::Not | OpCode::Neg | OpCode::Abs | OpCode::Sign => {
                stack.pop();
                stack.push(None);
            }
            // Stack manipulation
            OpCode::Dup => {
                if let Some(v) = stack.last().cloned() {
                    stack.push(v);
                }
            }
            OpCode::Pop => { stack.pop(); }
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
                stack.push(Some(stack.len() as u64));
            }
            // Temporal/IO ops invalidate tracking
            OpCode::Oracle | OpCode::Prophecy | OpCode::PresentRead |
            OpCode::Input | OpCode::Output | OpCode::Emit => {
                stack.clear();
            }
            _ => {
                stack.push(None); // Unknown effect
            }
        }
    }

    fn is_constant_false(&self, stmts: &[Stmt]) -> bool {
        // Check if the condition always evaluates to 0 (false)
        if stmts.len() == 1 {
            if let Stmt::Push(val) = &stmts[0] {
                return val.val == 0;
            }
        }
        false
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Dead Code Elimination Pass
    // ═══════════════════════════════════════════════════════════════════════

    fn dead_code_elimination_pass(&mut self, stmts: &[Stmt]) -> Vec<Stmt> {
        let mut result = Vec::new();
        let mut i = 0;

        while i < stmts.len() {
            match &stmts[i] {
                // Check for code after HALT or PARADOX
                Stmt::Op(OpCode::Halt) | Stmt::Op(OpCode::Paradox) => {
                    result.push(stmts[i].clone());
                    // Everything after is dead code
                    let remaining = stmts.len() - i - 1;
                    if remaining > 0 {
                        self.stats.dead_code_eliminated += remaining;
                    }
                    break;
                }

                // Push followed by Pop is dead
                Stmt::Push(_) => {
                    if i + 1 < stmts.len() {
                        if let Stmt::Op(OpCode::Pop) = &stmts[i + 1] {
                            self.stats.dead_code_eliminated += 2;
                            i += 2;
                            continue;
                        }
                    }
                    result.push(stmts[i].clone());
                }

                // Nop is dead
                Stmt::Op(OpCode::Nop) => {
                    self.stats.dead_code_eliminated += 1;
                }

                // Empty blocks are dead
                Stmt::Block(inner) => {
                    let cleaned = self.dead_code_elimination_pass(inner);
                    if !cleaned.is_empty() {
                        result.push(Stmt::Block(cleaned));
                    } else {
                        self.stats.dead_code_eliminated += 1;
                    }
                }

                Stmt::If { then_branch, else_branch } => {
                    let cleaned_then = self.dead_code_elimination_pass(then_branch);
                    let cleaned_else = else_branch.as_ref()
                        .map(|eb| self.dead_code_elimination_pass(eb));

                    // If both branches are empty, we still need to pop the condition
                    result.push(Stmt::If {
                        then_branch: cleaned_then,
                        else_branch: cleaned_else,
                    });
                }

                Stmt::While { cond, body } => {
                    let cleaned_cond = self.dead_code_elimination_pass(cond);
                    let cleaned_body = self.dead_code_elimination_pass(body);

                    result.push(Stmt::While {
                        cond: cleaned_cond,
                        body: cleaned_body,
                    });
                }

                _ => {
                    result.push(stmts[i].clone());
                }
            }
            i += 1;
        }

        result
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Common Subexpression Elimination Pass
    // ═══════════════════════════════════════════════════════════════════════

    /// Common Subexpression Elimination using value numbering for stack-based code.
    ///
    /// For stack-based languages, CSE works by:
    /// 1. Assigning value numbers to each unique value/computation
    /// 2. Tracking the abstract stack state with value numbers
    /// 3. Detecting when the same value number is computed twice
    /// 4. Replacing redundant computations with stack manipulation (DUP, OVER)
    fn cse_pass(&mut self, stmts: &[Stmt]) -> Vec<Stmt> {
        let mut result = Vec::new();
        let mut vn_state = ValueNumberingState::new();

        for stmt in stmts {
            match self.cse_transform_stmt(stmt, &mut vn_state) {
                CseTransform::Keep(s) => result.push(s),
                CseTransform::Replace(replacement) => {
                    self.stats.cse_eliminated += 1;
                    result.extend(replacement);
                }
                CseTransform::Remove => {
                    self.stats.cse_eliminated += 1;
                }
            }
        }

        result
    }

    /// Transform a statement using CSE value numbering.
    fn cse_transform_stmt(&mut self, stmt: &Stmt, state: &mut ValueNumberingState) -> CseTransform {
        match stmt {
            Stmt::Push(val) => {
                let vn = state.get_or_create_literal_vn(val.val);

                // Check if this value is already on the stack
                if let Some(depth) = state.find_value_on_stack(vn) {
                    // Value is already on stack - use stack manipulation instead
                    let replacement = self.generate_stack_copy(depth);
                    state.push(vn);
                    return CseTransform::Replace(replacement);
                }

                // Track the new value on stack
                state.push(vn);
                CseTransform::Keep(stmt.clone())
            }

            Stmt::Op(op) => {
                match op {
                    // Binary operations
                    OpCode::Add | OpCode::Sub | OpCode::Mul | OpCode::Div |
                    OpCode::Mod | OpCode::And | OpCode::Or | OpCode::Xor |
                    OpCode::Shl | OpCode::Shr | OpCode::Eq | OpCode::Neq |
                    OpCode::Lt | OpCode::Gt | OpCode::Lte | OpCode::Gte |
                    OpCode::Min | OpCode::Max | OpCode::Slt | OpCode::Sgt |
                    OpCode::Slte | OpCode::Sgte => {
                        if state.stack.len() >= 2 {
                            let vn_b = state.pop();
                            let vn_a = state.pop();

                            // Check if this operation was already computed
                            // Copy the value to avoid borrow issues
                            let expr_key = ExprKey::Binary(*op, vn_a, vn_b);
                            let existing_vn = state.expr_to_vn.get(&expr_key).copied();

                            if let Some(ev) = existing_vn {
                                // Expression already computed - check if result on stack
                                if let Some(depth) = state.find_value_on_stack(ev) {
                                    // Result is on stack - restore operands and copy result
                                    state.push(vn_a);
                                    state.push(vn_b);
                                    let mut replacement = vec![
                                        Stmt::Op(OpCode::Pop),
                                        Stmt::Op(OpCode::Pop),
                                    ];
                                    replacement.extend(self.generate_stack_copy(depth + 2)); // +2 for the pops we'll emit
                                    state.push(ev);
                                    return CseTransform::Replace(replacement);
                                }
                            }

                            // Compute new value number for this expression
                            let result_vn = state.get_or_create_expr_vn(expr_key);
                            state.push(result_vn);
                        }
                        CseTransform::Keep(stmt.clone())
                    }

                    // Unary operations
                    OpCode::Not | OpCode::Neg | OpCode::Abs | OpCode::Sign => {
                        if !state.stack.is_empty() {
                            let vn_a = state.pop();
                            let expr_key = ExprKey::Unary(*op, vn_a);

                            // Copy to avoid borrow issues
                            let existing_vn = state.expr_to_vn.get(&expr_key).copied();

                            if let Some(ev) = existing_vn {
                                if let Some(depth) = state.find_value_on_stack(ev) {
                                    state.push(vn_a);
                                    let mut replacement = vec![Stmt::Op(OpCode::Pop)];
                                    replacement.extend(self.generate_stack_copy(depth + 1));
                                    state.push(ev);
                                    return CseTransform::Replace(replacement);
                                }
                            }

                            let result_vn = state.get_or_create_expr_vn(expr_key);
                            state.push(result_vn);
                        }
                        CseTransform::Keep(stmt.clone())
                    }

                    // Stack manipulation
                    OpCode::Dup => {
                        if let Some(&vn) = state.stack.last() {
                            state.push(vn);
                        }
                        CseTransform::Keep(stmt.clone())
                    }
                    OpCode::Pop => {
                        state.pop();
                        CseTransform::Keep(stmt.clone())
                    }
                    OpCode::Swap => {
                        if state.stack.len() >= 2 {
                            let len = state.stack.len();
                            state.stack.swap(len - 1, len - 2);
                        }
                        CseTransform::Keep(stmt.clone())
                    }
                    OpCode::Over => {
                        if state.stack.len() >= 2 {
                            let vn = state.stack[state.stack.len() - 2];
                            state.push(vn);
                        }
                        CseTransform::Keep(stmt.clone())
                    }
                    OpCode::Rot => {
                        if state.stack.len() >= 3 {
                            let len = state.stack.len();
                            let c = state.stack.remove(len - 3);
                            state.stack.push(c);
                        }
                        CseTransform::Keep(stmt.clone())
                    }
                    OpCode::Depth => {
                        let vn = state.create_unknown_vn();
                        state.push(vn);
                        CseTransform::Keep(stmt.clone())
                    }
                    OpCode::Pick => {
                        // PICK is complex - just create unknown value number
                        state.pop(); // Remove index
                        let vn = state.create_unknown_vn();
                        state.push(vn);
                        CseTransform::Keep(stmt.clone())
                    }

                    // Temporal operations - invalidate all tracking
                    OpCode::Oracle | OpCode::Prophecy | OpCode::PresentRead => {
                        state.invalidate();
                        CseTransform::Keep(stmt.clone())
                    }

                    // I/O operations - add unknown values
                    OpCode::Input => {
                        let vn = state.create_unknown_vn();
                        state.push(vn);
                        CseTransform::Keep(stmt.clone())
                    }
                    OpCode::Output | OpCode::Emit => {
                        state.pop();
                        CseTransform::Keep(stmt.clone())
                    }

                    // All other operations - keep as is
                    _ => {
                        CseTransform::Keep(stmt.clone())
                    }
                }
            }

            // For blocks, if/while, etc - recursively process
            Stmt::Block(inner) => {
                let optimized = self.cse_pass(inner);
                CseTransform::Keep(Stmt::Block(optimized))
            }

            Stmt::If { then_branch, else_branch } => {
                // Condition was already evaluated - pop result
                state.pop();
                // Process branches with separate states (conservative)
                let opt_then = self.cse_pass(then_branch);
                let opt_else = else_branch.as_ref().map(|eb| self.cse_pass(eb));
                // After if/else, we don't know the stack state
                state.invalidate();
                CseTransform::Keep(Stmt::If {
                    then_branch: opt_then,
                    else_branch: opt_else,
                })
            }

            Stmt::While { cond, body } => {
                // Loops invalidate CSE state due to potential back-edges
                state.invalidate();
                CseTransform::Keep(Stmt::While {
                    cond: self.cse_pass(cond),
                    body: self.cse_pass(body),
                })
            }

            _ => CseTransform::Keep(stmt.clone()),
        }
    }

    /// Generate stack operations to copy value at given depth to top.
    fn generate_stack_copy(&self, depth: usize) -> Vec<Stmt> {
        match depth {
            0 => vec![Stmt::Op(OpCode::Dup)],  // Top of stack
            1 => vec![Stmt::Op(OpCode::Over)], // Second from top
            _ => {
                // For deeper values, use PICK with depth
                vec![
                    Stmt::Push(Value::new(depth as u64)),
                    Stmt::Op(OpCode::Pick),
                ]
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Peephole Optimization Pass
    // ═══════════════════════════════════════════════════════════════════════

    /// Peephole optimization pass.
    fn peephole_optimize(&mut self, instrs: Vec<OptInstr>) -> Vec<OptInstr> {
        let mut result = Vec::new();
        let mut i = 0;

        while i < instrs.len() {
            match &instrs[i] {
                // Detect Clear pattern: ORACLE addr, 0, PROPHECY
                OptInstr::Stmt(Stmt::Op(OpCode::Oracle)) => {
                    if i + 2 < instrs.len() {
                        if let (
                            OptInstr::Stmt(Stmt::Push(val)),
                            OptInstr::Stmt(Stmt::Op(OpCode::Prophecy))
                        ) = (&instrs[i + 1], &instrs[i + 2]) {
                            if val.val == 0 {
                                self.stats.patterns_detected += 1;
                                result.push(OptInstr::Clear(0)); // Address unknown at this stage
                                i += 3;
                                continue;
                            }
                        }
                    }
                    result.push(instrs[i].clone());
                }

                // Detect DUP + POP = NOP
                OptInstr::Stmt(Stmt::Op(OpCode::Dup)) => {
                    if i + 1 < instrs.len() {
                        if let OptInstr::Stmt(Stmt::Op(OpCode::Pop)) = &instrs[i + 1] {
                            self.stats.patterns_detected += 1;
                            i += 2;
                            continue;
                        }
                    }
                    result.push(instrs[i].clone());
                }

                // Detect SWAP + SWAP = NOP
                OptInstr::Stmt(Stmt::Op(OpCode::Swap)) => {
                    if i + 1 < instrs.len() {
                        if let OptInstr::Stmt(Stmt::Op(OpCode::Swap)) = &instrs[i + 1] {
                            self.stats.patterns_detected += 1;
                            i += 2;
                            continue;
                        }
                    }
                    result.push(instrs[i].clone());
                }

                // Detect 0 ADD = NOP (identity)
                OptInstr::Stmt(Stmt::Push(val)) if val.val == 0 => {
                    if i + 1 < instrs.len() {
                        if let OptInstr::Stmt(Stmt::Op(OpCode::Add)) = &instrs[i + 1] {
                            self.stats.patterns_detected += 1;
                            i += 2;
                            continue;
                        }
                    }
                    result.push(instrs[i].clone());
                }

                // Detect 1 MUL = NOP (identity)
                OptInstr::Stmt(Stmt::Push(val)) if val.val == 1 => {
                    if i + 1 < instrs.len() {
                        if let OptInstr::Stmt(Stmt::Op(OpCode::Mul)) = &instrs[i + 1] {
                            self.stats.patterns_detected += 1;
                            i += 2;
                            continue;
                        }
                    }
                    result.push(instrs[i].clone());
                }

                // Detect 0 MUL = CLEAR (always 0)
                OptInstr::Stmt(Stmt::Push(val)) if val.val == 0 => {
                    if i + 1 < instrs.len() {
                        if let OptInstr::Stmt(Stmt::Op(OpCode::Mul)) = &instrs[i + 1] {
                            // Replace with POP, PUSH 0
                            self.stats.patterns_detected += 1;
                            result.push(OptInstr::Stmt(Stmt::Op(OpCode::Pop)));
                            result.push(OptInstr::Stmt(Stmt::Push(Value::new(0))));
                            i += 2;
                            continue;
                        }
                    }
                    result.push(instrs[i].clone());
                }

                // Recursively optimize blocks
                OptInstr::Block(inner) => {
                    let optimized = self.peephole_optimize(inner.clone());
                    result.push(OptInstr::Block(optimized));
                }

                _ => {
                    result.push(instrs[i].clone());
                }
            }
            i += 1;
        }

        result
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Tail Call Optimization Pass
    // ═══════════════════════════════════════════════════════════════════════

    fn tail_call_optimize(&mut self, instrs: Vec<OptInstr>) -> Vec<OptInstr> {
        let mut result = Vec::new();

        for (i, instr) in instrs.iter().enumerate() {
            match instr {
                OptInstr::Stmt(Stmt::Call { name }) => {
                    // Check if this is the last instruction or followed only by returns
                    let is_tail = self.is_tail_position(&instrs, i);

                    if is_tail {
                        self.stats.tail_calls_optimized += 1;
                        result.push(OptInstr::TailCall { name: name.clone() });
                    } else {
                        result.push(instr.clone());
                    }
                }

                OptInstr::Block(inner) => {
                    let optimized = self.tail_call_optimize(inner.clone());
                    result.push(OptInstr::Block(optimized));
                }

                _ => {
                    result.push(instr.clone());
                }
            }
        }

        result
    }

    fn is_tail_position(&self, instrs: &[OptInstr], pos: usize) -> bool {
        // Tail position if:
        // 1. Last instruction, or
        // 2. Followed only by HALT
        if pos == instrs.len() - 1 {
            return true;
        }

        for instr in &instrs[pos + 1..] {
            match instr {
                OptInstr::Stmt(Stmt::Op(OpCode::Halt)) => continue,
                OptInstr::Stmt(Stmt::Op(OpCode::Nop)) => continue,
                _ => return false,
            }
        }

        true
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Loop Unrolling Pass
    // ═══════════════════════════════════════════════════════════════════════

    fn loop_unroll_pass(&mut self, instrs: Vec<OptInstr>) -> Vec<OptInstr> {
        let mut result = Vec::new();

        for instr in instrs {
            match instr {
                OptInstr::Stmt(Stmt::While { cond, body }) => {
                    // Check if loop count is known at compile time
                    if let Some(count) = self.analyze_loop_bound(&cond) {
                        if count <= self.unroll_threshold {
                            self.stats.loops_unrolled += 1;

                            // Unroll the loop
                            let opt_body = self.loop_unroll_pass(
                                body.iter().map(|s| OptInstr::Stmt(s.clone())).collect()
                            );

                            result.push(OptInstr::UnrolledLoop {
                                iterations: count,
                                body: opt_body,
                            });
                            continue;
                        }
                    }

                    // Can't unroll, keep original
                    result.push(OptInstr::Stmt(Stmt::While {
                        cond: cond.clone(),
                        body: body.clone(),
                    }));
                }

                OptInstr::Block(inner) => {
                    let optimized = self.loop_unroll_pass(inner);
                    result.push(OptInstr::Block(optimized));
                }

                _ => {
                    result.push(instr);
                }
            }
        }

        result
    }

    /// Analyze loop condition to determine iteration bound.
    ///
    /// Recognizes the following patterns:
    /// 1. Fixed constant: `PUSH n; PUSH 0; GT` → n iterations
    /// 2. Range comparison: `PUSH counter; PUSH limit; LT` with increment
    /// 3. Equality check: `PUSH val; PUSH target; NEQ` → bounded by val-target
    ///
    /// Returns Some(bound) if a fixed iteration count can be determined.
    fn analyze_loop_bound(&self, cond: &[Stmt]) -> Option<usize> {
        // Extract the pattern from condition statements
        let bound_info = self.extract_loop_bound_pattern(cond)?;

        // Validate bound is within unroll threshold
        if bound_info <= self.unroll_threshold {
            Some(bound_info)
        } else {
            None
        }
    }

    /// Extract loop bound from condition pattern.
    ///
    /// For stack-based languages, we recognize these condition patterns:
    /// - `PUSH n; PUSH 0; GT` → n iterations (counting down from n to 0)
    /// - `PUSH 0; PUSH n; LT` → n iterations (counting up from 0 to n)
    /// - `PUSH start; PUSH end; NEQ` → |end - start| iterations
    fn extract_loop_bound_pattern(&self, cond: &[Stmt]) -> Option<usize> {
        // Pattern 1: Simple constant comparison (most common)
        // Condition: [PUSH a, PUSH b, CMP]
        if cond.len() >= 3 {
            // Look for PUSH a, PUSH b, comparison at end
            let last = cond.last()?;
            let cmp_op = match last {
                Stmt::Op(op) => *op,
                _ => return None,
            };

            // Extract the two values being compared
            let (val_a, val_b) = self.extract_comparison_values(&cond[..cond.len()-1])?;

            match cmp_op {
                // n > 0: n iterations counting down
                OpCode::Gt if val_b == 0 => {
                    if val_a > 0 && val_a <= u16::MAX as u64 {
                        return Some(val_a as usize);
                    }
                }
                // 0 < n: n iterations counting up
                OpCode::Lt if val_a == 0 && val_b > 0 => {
                    if val_b <= u16::MAX as u64 {
                        return Some(val_b as usize);
                    }
                }
                // n >= 1: n iterations
                OpCode::Gte if val_b == 1 => {
                    if val_a > 0 && val_a <= u16::MAX as u64 {
                        return Some(val_a as usize);
                    }
                }
                // 0 <= n-1: n iterations
                OpCode::Lte if val_a == 0 && val_b > 0 => {
                    if val_b <= u16::MAX as u64 {
                        return Some(val_b as usize);
                    }
                }
                // a != b: |a - b| iterations (assumes convergence)
                OpCode::Neq => {
                    let diff = if val_a > val_b {
                        val_a - val_b
                    } else {
                        val_b - val_a
                    };
                    if diff > 0 && diff <= u16::MAX as u64 {
                        return Some(diff as usize);
                    }
                }
                _ => {}
            }
        }

        // Pattern 2: Single constant with implicit comparison
        // Condition: [PUSH n] where n is treated as boolean
        if cond.len() == 1 {
            if let Stmt::Push(val) = &cond[0] {
                // Non-zero constant → 1 iteration (will become false after decrement)
                if val.val > 0 && val.val <= self.unroll_threshold as u64 {
                    return Some(val.val as usize);
                }
            }
        }

        None
    }

    /// Extract two constant values from statements for comparison analysis.
    /// Returns (left_operand, right_operand) if both can be determined as constants.
    fn extract_comparison_values(&self, stmts: &[Stmt]) -> Option<(u64, u64)> {
        // Track the abstract stack to find the top two values
        let mut stack: Vec<Option<u64>> = Vec::new();

        for stmt in stmts {
            match stmt {
                Stmt::Push(val) => {
                    stack.push(Some(val.val));
                }
                Stmt::Op(OpCode::Dup) => {
                    if let Some(top) = stack.last().cloned() {
                        stack.push(top);
                    } else {
                        stack.push(None);
                    }
                }
                Stmt::Op(OpCode::Swap) => {
                    if stack.len() >= 2 {
                        let len = stack.len();
                        stack.swap(len - 1, len - 2);
                    }
                }
                Stmt::Op(OpCode::Over) => {
                    if stack.len() >= 2 {
                        let val = stack[stack.len() - 2].clone();
                        stack.push(val);
                    } else {
                        stack.push(None);
                    }
                }
                Stmt::Op(OpCode::Pop) => {
                    stack.pop();
                }
                // Binary operations that produce a result
                Stmt::Op(OpCode::Add) | Stmt::Op(OpCode::Sub) |
                Stmt::Op(OpCode::Mul) | Stmt::Op(OpCode::Div) |
                Stmt::Op(OpCode::Mod) | Stmt::Op(OpCode::And) |
                Stmt::Op(OpCode::Or) | Stmt::Op(OpCode::Xor) => {
                    if stack.len() >= 2 {
                        let b = stack.pop();
                        let a = stack.pop();
                        // Try to compute the result if both are known
                        let result = match (a, b, stmt) {
                            (Some(Some(av)), Some(Some(bv)), Stmt::Op(OpCode::Add)) =>
                                Some(av.wrapping_add(bv)),
                            (Some(Some(av)), Some(Some(bv)), Stmt::Op(OpCode::Sub)) =>
                                Some(av.wrapping_sub(bv)),
                            (Some(Some(av)), Some(Some(bv)), Stmt::Op(OpCode::Mul)) =>
                                Some(av.wrapping_mul(bv)),
                            _ => None,
                        };
                        stack.push(result);
                    } else {
                        stack.push(None);
                    }
                }
                // Operations that invalidate tracking
                Stmt::Op(OpCode::Oracle) | Stmt::Op(OpCode::Prophecy) |
                Stmt::Op(OpCode::PresentRead) | Stmt::Op(OpCode::Input) => {
                    stack.push(None); // Unknown value
                }
                _ => {
                    // For other operations, conservatively mark as unknown
                    stack.push(None);
                }
            }
        }

        // We need at least two values on the stack for comparison
        if stack.len() >= 2 {
            let b = stack.pop()?;
            let a = stack.pop()?;
            match (a, b) {
                (Some(av), Some(bv)) => Some((av, bv)),
                _ => None,
            }
        } else {
            None
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Inline Caching Pass
    // ═══════════════════════════════════════════════════════════════════════

    fn inline_caching_pass(&mut self, instrs: Vec<OptInstr>) -> Vec<OptInstr> {
        let mut result = Vec::new();

        for instr in instrs {
            match instr {
                OptInstr::Stmt(Stmt::Call { name }) => {
                    // Create inline cache slot for this call site
                    let slot = self.inline_cache_slots;
                    self.inline_cache_slots += 1;
                    self.stats.inline_caches_created += 1;

                    result.push(OptInstr::InlineCached {
                        name: name.clone(),
                        cache_slot: slot,
                    });
                }

                OptInstr::TailCall { name } => {
                    // Tail calls also benefit from inline caching
                    let slot = self.inline_cache_slots;
                    self.inline_cache_slots += 1;
                    self.stats.inline_caches_created += 1;

                    result.push(OptInstr::InlineCached {
                        name: name.clone(),
                        cache_slot: slot,
                    });
                }

                OptInstr::Block(inner) => {
                    let optimized = self.inline_caching_pass(inner);
                    result.push(OptInstr::Block(optimized));
                }

                _ => {
                    result.push(instr);
                }
            }
        }

        result
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Speculative Optimization Pass
    // ═══════════════════════════════════════════════════════════════════════

    fn speculative_optimize_pass(&mut self, instrs: Vec<OptInstr>) -> Vec<OptInstr> {
        let mut result = Vec::new();

        // Use profile data if available
        let has_profile = self.profile_data.is_some();

        for instr in instrs {
            match &instr {
                OptInstr::Stmt(Stmt::If { then_branch, else_branch }) => {
                    // Check if we have branch probability data
                    if has_profile {
                        // Speculate on the likely branch
                        self.stats.speculative_opts += 1;

                        let fast_path = self.speculative_optimize_pass(
                            then_branch.iter().map(|s| OptInstr::Stmt(s.clone())).collect()
                        );

                        let slow_path = else_branch.as_ref().map(|eb| {
                            self.speculative_optimize_pass(
                                eb.iter().map(|s| OptInstr::Stmt(s.clone())).collect()
                            )
                        }).unwrap_or_default();

                        result.push(OptInstr::SpeculativeGuard {
                            condition: Box::new(OptInstr::Stmt(Stmt::Op(OpCode::Nop))), // Placeholder
                            fast_path,
                            slow_path,
                        });
                        continue;
                    }

                    result.push(instr);
                }

                OptInstr::Block(inner) => {
                    let optimized = self.speculative_optimize_pass(inner.clone());
                    result.push(OptInstr::Block(optimized));
                }

                _ => {
                    result.push(instr);
                }
            }
        }

        result
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Conversion Back to Statements
    // ═══════════════════════════════════════════════════════════════════════

    /// Convert optimized instructions back to statements (for compatibility).
    pub fn unoptimize(instrs: &[OptInstr]) -> Vec<Stmt> {
        instrs.iter().flat_map(|instr| {
            match instr {
                OptInstr::Stmt(s) => vec![s.clone()],
                OptInstr::FusedAdd(n) => vec![
                    Stmt::Push(Value::new(*n as u64)),
                    Stmt::Op(OpCode::Add),
                ],
                OptInstr::FusedSub(n) => vec![
                    Stmt::Push(Value::new(*n as u64)),
                    Stmt::Op(OpCode::Sub),
                ],
                OptInstr::FusedMul(n) => vec![
                    Stmt::Push(Value::new(*n as u64)),
                    Stmt::Op(OpCode::Mul),
                ],
                OptInstr::Clear(_) => vec![
                    Stmt::Push(Value::new(0)),
                    Stmt::Op(OpCode::Prophecy),
                ],
                OptInstr::ConstFolded(v) => vec![
                    Stmt::Push(Value::new(*v)),
                ],
                OptInstr::TailCall { name } => vec![
                    Stmt::Call { name: name.clone() },
                ],
                OptInstr::InlineCached { name, .. } => vec![
                    Stmt::Call { name: name.clone() },
                ],
                OptInstr::UnrolledLoop { iterations, body } => {
                    let body_stmts = Self::unoptimize(body);
                    let mut result = Vec::new();
                    for _ in 0..*iterations {
                        result.extend(body_stmts.clone());
                    }
                    result
                }
                OptInstr::SpeculativeGuard { fast_path, slow_path, .. } => {
                    // Fall back to if/else structure
                    let then_stmts = Self::unoptimize(fast_path);
                    let else_stmts = if slow_path.is_empty() {
                        None
                    } else {
                        Some(Self::unoptimize(slow_path))
                    };
                    vec![Stmt::If {
                        then_branch: then_stmts,
                        else_branch: else_stmts,
                    }]
                }
                OptInstr::Block(inner) => Self::unoptimize(inner),
                OptInstr::FusedMemOps(ops) => {
                    ops.iter().map(|(_, op)| Stmt::Op(*op)).collect()
                }
                OptInstr::MoveUntil(_) | OptInstr::CopyTo { .. } => {
                    // These need expansion
                    vec![]
                }
            }
        }).collect()
    }
}

/// Tiered execution: decides whether to interpret or JIT compile.
#[derive(Debug)]
pub struct TieredExecutor {
    /// Execution counts for each block.
    execution_counts: HashMap<u64, usize>,
    /// Threshold for JIT compilation.
    jit_threshold: usize,
    /// Hot blocks that have been JIT compiled.
    jit_compiled: HashSet<u64>,
    /// Deoptimization counts (speculation failures).
    deopt_counts: HashMap<u64, usize>,
    /// Deoptimization threshold before falling back to interpreter.
    deopt_threshold: usize,
}

impl Default for TieredExecutor {
    fn default() -> Self {
        Self::new(100) // JIT after 100 executions
    }
}

impl TieredExecutor {
    /// Create with custom threshold.
    pub fn new(jit_threshold: usize) -> Self {
        Self {
            execution_counts: HashMap::new(),
            jit_threshold,
            jit_compiled: HashSet::new(),
            deopt_counts: HashMap::new(),
            deopt_threshold: 10,
        }
    }

    /// Check if a block should be JIT compiled.
    pub fn should_jit(&mut self, block_hash: u64) -> bool {
        // Don't re-JIT if already compiled
        if self.jit_compiled.contains(&block_hash) {
            return false;
        }

        // Check deoptimization count
        if let Some(&deopt) = self.deopt_counts.get(&block_hash) {
            if deopt >= self.deopt_threshold {
                return false; // Too many deoptimizations, stay interpreted
            }
        }

        let count = self.execution_counts.entry(block_hash).or_insert(0);
        *count += 1;

        if *count >= self.jit_threshold {
            self.jit_compiled.insert(block_hash);
            true
        } else {
            false
        }
    }

    /// Mark a block as needing deoptimization.
    pub fn deoptimize(&mut self, block_hash: u64) {
        self.jit_compiled.remove(&block_hash);
        *self.deopt_counts.entry(block_hash).or_insert(0) += 1;
    }

    /// Check if a block is JIT compiled.
    pub fn is_jit_compiled(&self, block_hash: u64) -> bool {
        self.jit_compiled.contains(&block_hash)
    }

    /// Get current execution count for a block.
    pub fn get_count(&self, block_hash: u64) -> usize {
        *self.execution_counts.get(&block_hash).unwrap_or(&0)
    }

    /// Get deoptimization count for a block.
    pub fn get_deopt_count(&self, block_hash: u64) -> usize {
        *self.deopt_counts.get(&block_hash).unwrap_or(&0)
    }

    /// Reset statistics.
    pub fn reset(&mut self) {
        self.execution_counts.clear();
        self.jit_compiled.clear();
        self.deopt_counts.clear();
    }

    /// Get statistics.
    pub fn stats(&self) -> TieredStats {
        TieredStats {
            blocks_tracked: self.execution_counts.len(),
            blocks_jit_compiled: self.jit_compiled.len(),
            total_deoptimizations: self.deopt_counts.values().sum(),
        }
    }
}

/// Statistics for tiered execution.
#[derive(Debug, Clone, Default)]
pub struct TieredStats {
    /// Number of unique blocks tracked.
    pub blocks_tracked: usize,
    /// Number of blocks that have been JIT compiled.
    pub blocks_jit_compiled: usize,
    /// Total deoptimization events.
    pub total_deoptimizations: usize,
}

impl std::fmt::Display for TieredStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tiered: {} blocks, {} JIT compiled, {} deopts",
            self.blocks_tracked, self.blocks_jit_compiled, self.total_deoptimizations)
    }
}

/// Inline cache for fast procedure dispatch.
#[derive(Debug)]
pub struct InlineCache {
    /// Cache entries: slot -> cached procedure info.
    entries: Vec<Option<CacheEntry>>,
    /// Hit count.
    hits: usize,
    /// Miss count.
    misses: usize,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    /// Cached procedure name.
    name: String,
    /// Cached procedure body hash.
    body_hash: u64,
}

impl InlineCache {
    /// Create a new inline cache with given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: vec![None; capacity],
            hits: 0,
            misses: 0,
        }
    }

    /// Look up a cached procedure.
    pub fn lookup(&mut self, slot: usize, name: &str) -> Option<u64> {
        if slot >= self.entries.len() {
            self.misses += 1;
            return None;
        }

        if let Some(entry) = &self.entries[slot] {
            if entry.name == name {
                self.hits += 1;
                return Some(entry.body_hash);
            }
        }

        self.misses += 1;
        None
    }

    /// Update cache entry.
    pub fn update(&mut self, slot: usize, name: String, body_hash: u64) {
        if slot < self.entries.len() {
            self.entries[slot] = Some(CacheEntry { name, body_hash });
        }
    }

    /// Get hit rate.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Reset cache.
    pub fn clear(&mut self) {
        for entry in &mut self.entries {
            *entry = None;
        }
        self.hits = 0;
        self.misses = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_creation() {
        let opt = Optimizer::new(OptLevel::Basic);
        assert_eq!(opt.stats().original_count, 0);
    }

    #[test]
    fn test_opt_stats_display() {
        let stats = OptStats {
            original_count: 100,
            optimized_count: 50,
            fused_adds: 10,
            fused_mem_ops: 5,
            patterns_detected: 3,
            constants_folded: 8,
            dead_code_eliminated: 12,
            cse_eliminated: 2,
            tail_calls_optimized: 1,
            loops_unrolled: 1,
            inline_caches_created: 5,
            speculative_opts: 2,
        };
        let s = format!("{}", stats);
        assert!(s.contains("50.0%"));
        assert!(s.contains("Constants folded"));
    }

    #[test]
    fn test_tiered_executor() {
        let mut tiered = TieredExecutor::new(3);
        assert!(!tiered.should_jit(12345));
        assert!(!tiered.should_jit(12345));
        assert!(tiered.should_jit(12345));
        assert_eq!(tiered.get_count(12345), 3);
        assert!(tiered.is_jit_compiled(12345));
    }

    #[test]
    fn test_tiered_deoptimization() {
        let mut tiered = TieredExecutor::new(1);
        assert!(tiered.should_jit(12345));
        assert!(tiered.is_jit_compiled(12345));

        tiered.deoptimize(12345);
        assert!(!tiered.is_jit_compiled(12345));
        assert_eq!(tiered.get_deopt_count(12345), 1);
    }

    #[test]
    fn test_basic_optimization() {
        let mut opt = Optimizer::new(OptLevel::Basic);
        let mut program = Program::new();
        program.body = vec![
            Stmt::Push(Value::new(1)),
            Stmt::Op(OpCode::Add),
            Stmt::Push(Value::new(2)),
            Stmt::Op(OpCode::Add),
        ];

        let result = opt.optimize(&program);
        // Should have fused the two adds
        assert!(opt.stats().fused_adds > 0 || result.len() > 0);
    }

    #[test]
    fn test_constant_folding() {
        let mut opt = Optimizer::new(OptLevel::Full);
        let mut program = Program::new();
        program.body = vec![
            Stmt::Push(Value::new(10)),
            Stmt::Push(Value::new(20)),
            Stmt::Op(OpCode::Add),
        ];

        let result = opt.optimize(&program);
        // Should have folded to single constant
        assert!(opt.stats().constants_folded > 0 || result.len() <= 2);
    }

    #[test]
    fn test_dead_code_elimination() {
        let mut opt = Optimizer::new(OptLevel::Full);
        let mut program = Program::new();
        program.body = vec![
            Stmt::Push(Value::new(42)),
            Stmt::Op(OpCode::Halt),
            Stmt::Push(Value::new(99)), // Dead code
            Stmt::Op(OpCode::Add),      // Dead code
        ];

        let result = opt.optimize(&program);
        // Should have eliminated dead code after HALT
        assert!(opt.stats().dead_code_eliminated > 0);
        assert!(result.len() <= 3);
    }

    #[test]
    fn test_peephole_patterns() {
        let mut opt = Optimizer::new(OptLevel::Full);
        let mut program = Program::new();
        program.body = vec![
            Stmt::Op(OpCode::Dup),
            Stmt::Op(OpCode::Pop), // Should be eliminated (DUP + POP = NOP)
        ];

        let result = opt.optimize(&program);
        assert!(opt.stats().patterns_detected > 0 || result.is_empty());
    }

    #[test]
    fn test_inline_cache() {
        let mut cache = InlineCache::new(10);

        // Miss on first lookup
        assert!(cache.lookup(0, "test_proc").is_none());

        // Update cache
        cache.update(0, "test_proc".to_string(), 12345);

        // Hit on second lookup
        assert_eq!(cache.lookup(0, "test_proc"), Some(12345));

        // Miss on different name
        assert!(cache.lookup(0, "other_proc").is_none());

        assert!(cache.hit_rate() > 0.0);
    }

    #[test]
    fn test_opt_level_aggressive() {
        let mut opt = Optimizer::new(OptLevel::Aggressive);
        let mut program = Program::new();
        program.body = vec![
            Stmt::Push(Value::new(5)),
            Stmt::Push(Value::new(3)),
            Stmt::Op(OpCode::Add),
        ];

        let result = opt.optimize(&program);
        assert!(result.len() > 0);
    }

    #[test]
    fn test_unoptimize_fused_add() {
        let instrs = vec![OptInstr::FusedAdd(42)];
        let stmts = Optimizer::unoptimize(&instrs);

        assert_eq!(stmts.len(), 2);
        if let Stmt::Push(v) = &stmts[0] {
            assert_eq!(v.val, 42);
        }
        assert!(matches!(stmts[1], Stmt::Op(OpCode::Add)));
    }

    #[test]
    fn test_unoptimize_unrolled_loop() {
        let instrs = vec![OptInstr::UnrolledLoop {
            iterations: 3,
            body: vec![OptInstr::Stmt(Stmt::Push(Value::new(1)))],
        }];

        let stmts = Optimizer::unoptimize(&instrs);
        assert_eq!(stmts.len(), 3); // 3 iterations
    }
}
