//! Statistical metrics collection for convergence analysis.
//!
//! This module provides numerically stable statistical computations following
//! first-principle approaches. All calculations use analytic solutions where
//! possible, with documented tolerances for any approximations.
//!
//! # Numerical Stability
//!
//! - Mean/Variance: Welford's online algorithm (numerically stable)
//! - Higher moments: Two-pass algorithm for skewness/kurtosis
//! - Error tracking: Kahan summation for reduced accumulation error
//!
//! # Precision Guarantees
//!
//! - All floating-point operations use f64 (double precision, ~15-16 significant digits)
//! - Relative error bounds documented for each computation
//! - Catastrophic cancellation avoided through stable formulations

use std::collections::VecDeque;

// =============================================================================
// Online Statistics (Welford's Algorithm)
// =============================================================================

/// Online statistics using Welford's algorithm.
///
/// Computes mean and variance in a single pass with numerical stability.
/// The algorithm maintains running aggregates without storing all values,
/// avoiding catastrophic cancellation that can occur with naive formulas.
///
/// # Numerical Properties
///
/// - Mean: Computed exactly (no approximation)
/// - Variance: Welford's formula guarantees stability for large n
/// - Relative error: O(ε) where ε ≈ 2.2e-16 (machine epsilon for f64)
#[derive(Debug, Clone)]
pub struct OnlineStatistics {
    /// Count of observations.
    n: u64,
    /// Running mean.
    mean: f64,
    /// Running sum of squared deviations from mean (M2 in Welford's notation).
    m2: f64,
    /// Minimum observed value.
    min: f64,
    /// Maximum observed value.
    max: f64,
    /// Sum (with Kahan compensation for reduced error).
    sum: f64,
    /// Kahan compensation term.
    compensation: f64,
}

impl Default for OnlineStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl OnlineStatistics {
    /// Create a new statistics accumulator.
    pub const fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            sum: 0.0,
            compensation: 0.0,
        }
    }

    /// Add a new observation.
    ///
    /// Uses Welford's online algorithm for numerical stability.
    #[inline]
    pub fn push(&mut self, x: f64) {
        self.n += 1;
        let n = self.n as f64;

        // Welford's algorithm for mean and variance
        let delta = x - self.mean;
        self.mean += delta / n;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;

        // Track min/max
        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }

        // Kahan summation for accurate sum
        let y = x - self.compensation;
        let t = self.sum + y;
        self.compensation = (t - self.sum) - y;
        self.sum = t;
    }

    /// Add an integer observation.
    #[inline]
    pub fn push_u64(&mut self, x: u64) {
        self.push(x as f64);
    }

    /// Add an integer observation (i64).
    #[inline]
    pub fn push_i64(&mut self, x: i64) {
        self.push(x as f64);
    }

    /// Number of observations.
    #[inline]
    pub fn count(&self) -> u64 {
        self.n
    }

    /// Arithmetic mean.
    ///
    /// Returns NaN if no observations.
    #[inline]
    pub fn mean(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            self.mean
        }
    }

    /// Sum of all observations (Kahan-compensated).
    #[inline]
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Population variance (σ²).
    ///
    /// Uses n as divisor. Returns NaN if n < 1.
    #[inline]
    pub fn variance_population(&self) -> f64 {
        if self.n < 1 {
            f64::NAN
        } else {
            self.m2 / self.n as f64
        }
    }

    /// Sample variance (s²).
    ///
    /// Uses n-1 as divisor (Bessel's correction). Returns NaN if n < 2.
    #[inline]
    pub fn variance_sample(&self) -> f64 {
        if self.n < 2 {
            f64::NAN
        } else {
            self.m2 / (self.n - 1) as f64
        }
    }

    /// Population standard deviation (σ).
    #[inline]
    pub fn std_dev_population(&self) -> f64 {
        self.variance_population().sqrt()
    }

    /// Sample standard deviation (s).
    #[inline]
    pub fn std_dev_sample(&self) -> f64 {
        self.variance_sample().sqrt()
    }

    /// Minimum observed value.
    #[inline]
    pub fn min(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            self.min
        }
    }

    /// Maximum observed value.
    #[inline]
    pub fn max(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            self.max
        }
    }

    /// Range (max - min).
    #[inline]
    pub fn range(&self) -> f64 {
        self.max - self.min
    }

    /// Coefficient of variation (CV = σ/μ).
    ///
    /// Undefined if mean is zero.
    #[inline]
    pub fn coefficient_of_variation(&self) -> f64 {
        if self.mean.abs() < f64::EPSILON {
            f64::NAN
        } else {
            self.std_dev_population() / self.mean.abs()
        }
    }

    /// Merge another statistics accumulator into this one.
    ///
    /// Uses parallel algorithm for combining two Welford accumulators.
    /// This is exact (no approximation error beyond floating-point).
    pub fn merge(&mut self, other: &Self) {
        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = other.clone();
            return;
        }

        let n_a = self.n as f64;
        let n_b = other.n as f64;
        let n_combined = n_a + n_b;

        let delta = other.mean - self.mean;
        let mean_combined = self.mean + delta * n_b / n_combined;

        // Parallel variance combination formula
        let m2_combined = self.m2 + other.m2 + delta * delta * n_a * n_b / n_combined;

        self.n += other.n;
        self.mean = mean_combined;
        self.m2 = m2_combined;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);

        // Merge sums with Kahan
        let y = other.sum - self.compensation;
        let t = self.sum + y;
        self.compensation = (t - self.sum) - y;
        self.sum = t;
    }
}

// =============================================================================
// Distributional Measures
// =============================================================================

/// Complete distributional measures including higher moments.
///
/// For skewness and kurtosis, we use the standard formulas after collecting
/// all values, as online algorithms for higher moments have poor numerical
/// properties.
///
/// # Moment Definitions
///
/// - Raw moment k: E[X^k]
/// - Central moment k: E[(X - μ)^k]
/// - Standardized moment k: E[(X - μ)^k] / σ^k
///
/// # Skewness (γ₁)
///
/// - γ₁ > 0: Right-skewed (tail extends right)
/// - γ₁ < 0: Left-skewed (tail extends left)
/// - γ₁ = 0: Symmetric
///
/// # Kurtosis (γ₂)
///
/// We report excess kurtosis (kurtosis - 3), so normal distribution = 0.
/// - γ₂ > 0: Leptokurtic (heavy tails, sharp peak)
/// - γ₂ < 0: Platykurtic (light tails, flat peak)
/// - γ₂ = 0: Mesokurtic (normal-like tails)
#[derive(Debug, Clone)]
pub struct DistributionalMeasures {
    /// Online statistics for mean/variance.
    online: OnlineStatistics,
    /// Stored values for higher moment calculation.
    values: Vec<f64>,
    /// Maximum values to store (for memory bounds).
    max_values: usize,
}

impl DistributionalMeasures {
    /// Create with default capacity.
    pub fn new() -> Self {
        Self::with_capacity(10_000)
    }

    /// Create with specified capacity.
    pub fn with_capacity(max_values: usize) -> Self {
        Self {
            online: OnlineStatistics::new(),
            values: Vec::with_capacity(max_values.min(1000)),
            max_values,
        }
    }

    /// Add an observation.
    pub fn push(&mut self, x: f64) {
        self.online.push(x);
        if self.values.len() < self.max_values {
            self.values.push(x);
        }
    }

    /// Add an integer observation.
    pub fn push_u64(&mut self, x: u64) {
        self.push(x as f64);
    }

    /// Number of observations.
    pub fn count(&self) -> u64 {
        self.online.count()
    }

    /// Mean (first raw moment / location).
    pub fn mean(&self) -> f64 {
        self.online.mean()
    }

    /// Population variance (second central moment / spread).
    pub fn variance(&self) -> f64 {
        self.online.variance_population()
    }

    /// Standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.online.std_dev_population()
    }

    /// Skewness (third standardized moment / asymmetry).
    ///
    /// Requires at least 3 observations. Uses Fisher's definition.
    /// Returns NaN if insufficient data or zero variance.
    pub fn skewness(&self) -> f64 {
        let n = self.values.len();
        if n < 3 {
            return f64::NAN;
        }

        let mean = self.online.mean();
        let std_dev = self.online.std_dev_population();

        if std_dev.abs() < f64::EPSILON {
            return f64::NAN;
        }

        let n_f = n as f64;
        let m3: f64 = self.values.iter()
            .map(|x| ((x - mean) / std_dev).powi(3))
            .sum();

        // Fisher's adjustment for sample skewness
        let adjustment = (n_f * (n_f - 1.0)).sqrt() / (n_f - 2.0);
        adjustment * m3 / n_f
    }

    /// Excess kurtosis (fourth standardized moment - 3 / tailedness).
    ///
    /// Requires at least 4 observations. Returns excess kurtosis so that
    /// normal distribution = 0.
    /// Returns NaN if insufficient data or zero variance.
    pub fn kurtosis(&self) -> f64 {
        let n = self.values.len();
        if n < 4 {
            return f64::NAN;
        }

        let mean = self.online.mean();
        let std_dev = self.online.std_dev_population();

        if std_dev.abs() < f64::EPSILON {
            return f64::NAN;
        }

        let n_f = n as f64;
        let m4: f64 = self.values.iter()
            .map(|x| ((x - mean) / std_dev).powi(4))
            .sum();

        // Fisher's adjustment for sample kurtosis
        let g2 = m4 / n_f - 3.0;
        let adjustment = (n_f - 1.0) / ((n_f - 2.0) * (n_f - 3.0));
        adjustment * ((n_f + 1.0) * g2 + 6.0) - 3.0 * (n_f - 1.0).powi(2) / ((n_f - 2.0) * (n_f - 3.0))
    }

    /// Median (requires stored values).
    pub fn median(&self) -> f64 {
        if self.values.is_empty() {
            return f64::NAN;
        }

        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        }
    }

    /// Percentile (0-100).
    pub fn percentile(&self, p: f64) -> f64 {
        if self.values.is_empty() || p < 0.0 || p > 100.0 {
            return f64::NAN;
        }

        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let rank = (p / 100.0) * (sorted.len() - 1) as f64;
        let lower = rank.floor() as usize;
        let upper = rank.ceil() as usize;
        let fraction = rank - lower as f64;

        if lower == upper {
            sorted[lower]
        } else {
            sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
        }
    }

    /// Get online statistics reference.
    pub fn online(&self) -> &OnlineStatistics {
        &self.online
    }
}

impl Default for DistributionalMeasures {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Convergence Metrics
// =============================================================================

/// Metrics specific to temporal convergence analysis.
#[derive(Debug, Clone)]
pub struct ConvergenceMetrics {
    /// Epoch durations in microseconds.
    pub epoch_durations: OnlineStatistics,
    /// Instructions per epoch.
    pub instructions_per_epoch: OnlineStatistics,
    /// Oracle operations per epoch.
    pub oracle_ops_per_epoch: OnlineStatistics,
    /// Prophecy operations per epoch.
    pub prophecy_ops_per_epoch: OnlineStatistics,
    /// Changed cells per epoch.
    pub changes_per_epoch: OnlineStatistics,
    /// Total delta (L1 distance) per epoch.
    pub delta_per_epoch: OnlineStatistics,
    /// Convergence rate per epoch (0.0-1.0).
    pub convergence_rate: OnlineStatistics,
    /// State hash history for cycle detection.
    state_hashes: VecDeque<u64>,
    /// Maximum state history to maintain.
    max_history: usize,
    /// Detected oscillation period (None if not oscillating).
    pub detected_period: Option<usize>,
}

impl Default for ConvergenceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl ConvergenceMetrics {
    /// Create new convergence metrics.
    pub fn new() -> Self {
        Self::with_history(1000)
    }

    /// Create with specified history size for cycle detection.
    pub fn with_history(max_history: usize) -> Self {
        Self {
            epoch_durations: OnlineStatistics::new(),
            instructions_per_epoch: OnlineStatistics::new(),
            oracle_ops_per_epoch: OnlineStatistics::new(),
            prophecy_ops_per_epoch: OnlineStatistics::new(),
            changes_per_epoch: OnlineStatistics::new(),
            delta_per_epoch: OnlineStatistics::new(),
            convergence_rate: OnlineStatistics::new(),
            state_hashes: VecDeque::with_capacity(max_history),
            max_history,
            detected_period: None,
        }
    }

    /// Record an epoch's metrics.
    pub fn record_epoch(
        &mut self,
        duration_us: u64,
        instructions: u64,
        oracle_ops: u64,
        prophecy_ops: u64,
        changed_cells: usize,
        total_delta: u64,
        state_hash: u64,
    ) {
        self.epoch_durations.push_u64(duration_us);
        self.instructions_per_epoch.push_u64(instructions);
        self.oracle_ops_per_epoch.push_u64(oracle_ops);
        self.prophecy_ops_per_epoch.push_u64(prophecy_ops);
        self.changes_per_epoch.push_u64(changed_cells as u64);
        self.delta_per_epoch.push_u64(total_delta);

        // Calculate convergence rate
        let total_ops = oracle_ops + prophecy_ops;
        let rate = if total_ops == 0 {
            1.0
        } else {
            1.0 - (changed_cells as f64 / total_ops as f64).min(1.0)
        };
        self.convergence_rate.push(rate);

        // Cycle detection
        self.add_state_hash(state_hash);
    }

    /// Add a state hash and check for cycles.
    fn add_state_hash(&mut self, hash: u64) {
        // Check if this hash already exists in history
        for (i, &h) in self.state_hashes.iter().enumerate() {
            if h == hash {
                let period = self.state_hashes.len() - i;
                self.detected_period = Some(period);
                return;
            }
        }

        // Add to history
        if self.state_hashes.len() >= self.max_history {
            self.state_hashes.pop_front();
        }
        self.state_hashes.push_back(hash);
    }

    /// Number of epochs recorded.
    pub fn epoch_count(&self) -> u64 {
        self.epoch_durations.count()
    }

    /// Average epoch duration in microseconds.
    pub fn avg_epoch_duration_us(&self) -> f64 {
        self.epoch_durations.mean()
    }

    /// Average instructions per epoch.
    pub fn avg_instructions(&self) -> f64 {
        self.instructions_per_epoch.mean()
    }

    /// Average convergence rate.
    pub fn avg_convergence_rate(&self) -> f64 {
        self.convergence_rate.mean()
    }

    /// Estimate remaining epochs to convergence based on trend.
    ///
    /// Uses linear extrapolation of convergence rate.
    /// Returns None if trend is flat or negative.
    pub fn estimate_remaining_epochs(&self) -> Option<usize> {
        if self.convergence_rate.count() < 3 {
            return None;
        }

        let current_rate = self.avg_convergence_rate();
        if current_rate >= 1.0 {
            return Some(0); // Already converged
        }

        // Simple heuristic: estimate based on rate approaching 1.0
        let remaining = (1.0 - current_rate) / (current_rate + f64::EPSILON);
        if remaining > 0.0 && remaining < 1_000_000.0 {
            Some(remaining.ceil() as usize)
        } else {
            None
        }
    }

    /// Format as a summary string.
    pub fn format_summary(&self) -> String {
        let duration_str = if self.avg_epoch_duration_us() >= 1_000_000.0 {
            format!("{:.2}s", self.avg_epoch_duration_us() / 1_000_000.0)
        } else if self.avg_epoch_duration_us() >= 1_000.0 {
            format!("{:.2}ms", self.avg_epoch_duration_us() / 1_000.0)
        } else {
            format!("{:.0}us", self.avg_epoch_duration_us())
        };

        format!(
            "epochs={} avg_duration={} avg_inst={:.0} avg_rate={:.4} period={:?}",
            self.epoch_count(),
            duration_str,
            self.avg_instructions(),
            self.avg_convergence_rate(),
            self.detected_period
        )
    }
}

// =============================================================================
// Provenance Metrics
// =============================================================================

/// Metrics for provenance/causal dependency analysis.
#[derive(Debug, Clone, Default)]
pub struct ProvenanceMetrics {
    /// Count of pure (non-temporal) values.
    pub pure_count: u64,
    /// Count of temporal (oracle-dependent) values.
    pub temporal_count: u64,
    /// Dependency counts per temporal value.
    pub dependency_counts: OnlineStatistics,
    /// Causal depth statistics.
    pub causal_depths: OnlineStatistics,
    /// Addresses that are oracle sources.
    pub oracle_sources: u64,
    /// Addresses that are prophecy targets.
    pub prophecy_targets: u64,
}

impl ProvenanceMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a pure value.
    pub fn record_pure(&mut self) {
        self.pure_count += 1;
    }

    /// Record a temporal value with its dependencies.
    pub fn record_temporal(&mut self, dependency_count: usize, causal_depth: usize) {
        self.temporal_count += 1;
        self.dependency_counts.push_u64(dependency_count as u64);
        self.causal_depths.push_u64(causal_depth as u64);
    }

    /// Record oracle source.
    pub fn record_oracle_source(&mut self) {
        self.oracle_sources += 1;
    }

    /// Record prophecy target.
    pub fn record_prophecy_target(&mut self) {
        self.prophecy_targets += 1;
    }

    /// Ratio of temporal to total values.
    pub fn temporal_ratio(&self) -> f64 {
        let total = self.pure_count + self.temporal_count;
        if total == 0 {
            0.0
        } else {
            self.temporal_count as f64 / total as f64
        }
    }

    /// Average dependency count for temporal values.
    pub fn avg_dependencies(&self) -> f64 {
        self.dependency_counts.mean()
    }

    /// Average causal depth.
    pub fn avg_causal_depth(&self) -> f64 {
        self.causal_depths.mean()
    }

    /// Format as summary string.
    pub fn format_summary(&self) -> String {
        format!(
            "pure={} temporal={} ({:.1}%) avg_deps={:.1} avg_depth={:.1}",
            self.pure_count,
            self.temporal_count,
            self.temporal_ratio() * 100.0,
            self.avg_dependencies(),
            self.avg_causal_depth()
        )
    }
}

// =============================================================================
// Error Accumulation Tracking
// =============================================================================

/// Tracks error accumulation with tight tolerances.
///
/// Used to monitor numerical stability during long-running computations.
/// Reports when accumulated error exceeds specified tolerances.
#[derive(Debug, Clone)]
pub struct ErrorAccumulator {
    /// Running sum of absolute errors.
    total_absolute_error: f64,
    /// Running sum of relative errors.
    total_relative_error: f64,
    /// Maximum observed absolute error.
    max_absolute_error: f64,
    /// Maximum observed relative error.
    max_relative_error: f64,
    /// Count of error measurements.
    count: u64,
    /// Absolute error tolerance.
    abs_tolerance: f64,
    /// Relative error tolerance.
    rel_tolerance: f64,
    /// Whether tolerance has been exceeded.
    tolerance_exceeded: bool,
}

impl ErrorAccumulator {
    /// Create with specified tolerances.
    ///
    /// # Arguments
    /// - `abs_tolerance`: Maximum acceptable absolute error
    /// - `rel_tolerance`: Maximum acceptable relative error (e.g., 1e-10 for tight tolerance)
    pub fn new(abs_tolerance: f64, rel_tolerance: f64) -> Self {
        Self {
            total_absolute_error: 0.0,
            total_relative_error: 0.0,
            max_absolute_error: 0.0,
            max_relative_error: 0.0,
            count: 0,
            abs_tolerance,
            rel_tolerance,
            tolerance_exceeded: false,
        }
    }

    /// Create with single-precision tolerance (~1e-7).
    pub fn single_precision() -> Self {
        Self::new(1e-6, 1e-7)
    }

    /// Create with double-precision tolerance (~1e-15).
    pub fn double_precision() -> Self {
        Self::new(1e-14, 1e-15)
    }

    /// Record an error measurement.
    ///
    /// # Arguments
    /// - `computed`: The computed value
    /// - `reference`: The reference (true) value
    pub fn record(&mut self, computed: f64, reference: f64) {
        let abs_error = (computed - reference).abs();
        let rel_error = if reference.abs() > f64::EPSILON {
            abs_error / reference.abs()
        } else {
            abs_error
        };

        self.total_absolute_error += abs_error;
        self.total_relative_error += rel_error;
        self.max_absolute_error = self.max_absolute_error.max(abs_error);
        self.max_relative_error = self.max_relative_error.max(rel_error);
        self.count += 1;

        // Check tolerance
        if abs_error > self.abs_tolerance || rel_error > self.rel_tolerance {
            self.tolerance_exceeded = true;
        }
    }

    /// Record absolute error directly.
    pub fn record_absolute(&mut self, error: f64) {
        self.total_absolute_error += error;
        self.max_absolute_error = self.max_absolute_error.max(error);
        self.count += 1;

        if error > self.abs_tolerance {
            self.tolerance_exceeded = true;
        }
    }

    /// Check if tolerance has been exceeded.
    pub fn is_exceeded(&self) -> bool {
        self.tolerance_exceeded
    }

    /// Mean absolute error.
    pub fn mean_absolute_error(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_absolute_error / self.count as f64
        }
    }

    /// Mean relative error.
    pub fn mean_relative_error(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_relative_error / self.count as f64
        }
    }

    /// Maximum absolute error observed.
    pub fn max_absolute(&self) -> f64 {
        self.max_absolute_error
    }

    /// Maximum relative error observed.
    pub fn max_relative(&self) -> f64 {
        self.max_relative_error
    }

    /// Format as summary string.
    pub fn format_summary(&self) -> String {
        let status = if self.tolerance_exceeded { "EXCEEDED" } else { "OK" };
        format!(
            "errors={} mae={:.2e} mre={:.2e} max_abs={:.2e} max_rel={:.2e} [{}]",
            self.count,
            self.mean_absolute_error(),
            self.mean_relative_error(),
            self.max_absolute_error,
            self.max_relative_error,
            status
        )
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_online_statistics_basic() {
        let mut stats = OnlineStatistics::new();
        stats.push(1.0);
        stats.push(2.0);
        stats.push(3.0);
        stats.push(4.0);
        stats.push(5.0);

        assert_eq!(stats.count(), 5);
        assert!((stats.mean() - 3.0).abs() < 1e-10);
        assert!((stats.variance_population() - 2.0).abs() < 1e-10);
        assert_eq!(stats.min(), 1.0);
        assert_eq!(stats.max(), 5.0);
    }

    #[test]
    fn test_online_statistics_welford_stability() {
        // Test numerical stability with large offset
        let mut stats = OnlineStatistics::new();
        let offset = 1e10;
        for i in 0..1000 {
            stats.push(offset + i as f64);
        }

        // Mean should be offset + 499.5
        let expected_mean = offset + 499.5;
        assert!((stats.mean() - expected_mean).abs() < 1e-6);

        // Variance of 0..999 (population) = E[X^2] - E[X]^2
        // E[X] = 499.5, E[X^2] = sum(i^2)/1000 for i=0..999
        // Sum of i^2 from 0 to n-1 = (n-1)*n*(2n-1)/6 = 999*1000*1999/6 = 332833500
        // E[X^2] = 332833500 / 1000 = 332833.5
        // Var = 332833.5 - 499.5^2 = 332833.5 - 249500.25 = 83333.25
        let expected_var = 83333.25;
        assert!(
            (stats.variance_population() - expected_var).abs() / expected_var < 1e-6,
            "Expected variance {}, got {}",
            expected_var,
            stats.variance_population()
        );
    }

    #[test]
    fn test_online_statistics_merge() {
        let mut stats1 = OnlineStatistics::new();
        let mut stats2 = OnlineStatistics::new();

        for i in 0..50 {
            stats1.push(i as f64);
        }
        for i in 50..100 {
            stats2.push(i as f64);
        }

        stats1.merge(&stats2);

        assert_eq!(stats1.count(), 100);
        assert!((stats1.mean() - 49.5).abs() < 1e-10);
    }

    #[test]
    fn test_distributional_measures() {
        let mut dm = DistributionalMeasures::new();

        // Normal-ish distribution
        for x in [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0] {
            dm.push(x);
        }

        assert_eq!(dm.count(), 10);
        assert!((dm.mean() - 3.0).abs() < 1e-10);
        assert!((dm.median() - 3.0).abs() < 1e-10);

        // Skewness should be near 0 for symmetric distribution
        let skew = dm.skewness();
        assert!(skew.abs() < 0.5, "Expected near-zero skewness, got {}", skew);
    }

    #[test]
    fn test_convergence_metrics() {
        let mut cm = ConvergenceMetrics::new();

        // Record some epochs
        for i in 0..5 {
            cm.record_epoch(
                1000 + i * 100,  // duration
                500,             // instructions
                10,              // oracle ops
                10,              // prophecy ops
                5 - i as usize,  // changed cells (decreasing)
                50,              // delta
                i as u64,        // unique hashes
            );
        }

        assert_eq!(cm.epoch_count(), 5);
        assert!(cm.avg_convergence_rate() > 0.5);
    }

    #[test]
    fn test_convergence_cycle_detection() {
        let mut cm = ConvergenceMetrics::new();

        // Create a cycle: hashes 1, 2, 3, 1, 2, 3...
        for hash in [1u64, 2, 3, 1] {
            cm.record_epoch(1000, 100, 10, 10, 1, 10, hash);
        }

        assert_eq!(cm.detected_period, Some(3));
    }

    #[test]
    fn test_provenance_metrics() {
        let mut pm = ProvenanceMetrics::new();

        pm.record_pure();
        pm.record_pure();
        pm.record_temporal(3, 2);
        pm.record_temporal(5, 4);

        assert_eq!(pm.pure_count, 2);
        assert_eq!(pm.temporal_count, 2);
        assert!((pm.temporal_ratio() - 0.5).abs() < 1e-10);
        assert!((pm.avg_dependencies() - 4.0).abs() < 1e-10);
        assert!((pm.avg_causal_depth() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_error_accumulator() {
        // Use custom tolerances for clarity
        let mut ea = ErrorAccumulator::new(1e-5, 1e-5);

        // Small errors within tolerance
        ea.record(1.000001, 1.0); // 1e-6 error, below 1e-5 tolerance
        ea.record(2.000001, 2.0); // 5e-7 relative error

        assert!(!ea.is_exceeded(), "Small errors should be within tolerance");

        // Large error exceeds tolerance
        ea.record(1.1, 1.0); // 10% error, way beyond 1e-5
        assert!(ea.is_exceeded(), "Large error should exceed tolerance");
    }

    #[test]
    fn test_error_accumulator_double_precision() {
        let mut ea = ErrorAccumulator::double_precision();

        // Errors within double-precision tolerance (1e-15)
        ea.record(1.0 + 1e-16, 1.0);
        ea.record(2.0 + 1e-16, 2.0);

        assert!(!ea.is_exceeded(), "Tiny errors should be within tolerance");

        // Error exceeding double-precision tolerance
        ea.record(1.0 + 1e-14, 1.0);
        assert!(ea.is_exceeded(), "1e-14 error should exceed 1e-15 tolerance");
    }

    #[test]
    fn test_kahan_summation() {
        let mut stats = OnlineStatistics::new();

        // Add many small numbers that would lose precision with naive summation
        for _ in 0..1_000_000 {
            stats.push(0.1);
        }

        // Expected sum is 100000.0
        // Naive summation would accumulate error; Kahan should be accurate
        let expected = 100_000.0;
        let actual = stats.sum();
        let error = (actual - expected).abs();

        // Error should be very small (much less than 0.01)
        assert!(error < 0.01, "Kahan sum error too large: {}", error);
    }
}
