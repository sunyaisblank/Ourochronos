//! Trace pattern detection and classification.
//!
//! This module defines known trace patterns that can be specialized
//! for faster execution. Patterns are categorized as:
//! - Pure: Computation without temporal effects
//! - Temporal: Patterns involving ORACLE/PROPHECY

use super::trace::TraceAnalysis;

/// Known trace patterns that can be specialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TracePattern {
    // Pure Computation Patterns
    /// Fibonacci step: SWAP OVER ADD
    FibonacciStep,
    /// Simple counter increment loop
    CounterLoop,
    /// Accumulation loop: sum += counter
    SumLoop,
    /// Factorial loop: result *= counter
    FactorialLoop,
    /// Power loop: result *= base
    PowerLoop,
    /// Modular exponentiation
    ModPowerLoop,
    /// GCD loop: Euclidean algorithm
    GcdLoop,
    /// Memory fill: fill range with value
    MemoryFillLoop,
    /// Memory copy: copy range
    MemoryCopyLoop,
    /// Reduction loop: fold operation
    ReductionLoop,
    /// Dot product: sum of products
    DotProductLoop,
    /// Polynomial evaluation: Horner's method
    PolynomialLoop,
    /// Find min/max in sequence
    MinMaxLoop,
    /// Repeated bit manipulations
    BitwiseLoop,

    // Temporal Patterns
    /// Read-modify-write: ORACLE → compute → PROPHECY
    ReadModifyWrite,
    /// Convergence loop: iterate until stable
    ConvergenceLoop,
    /// Causal chain: PROPHECY depends on ORACLE
    CausalChain,
    /// Bootstrap pattern: initial guess refined
    BootstrapGuess,
    /// Witness search: probe addresses
    WitnessSearch,
    /// Sequential ORACLE reads
    TemporalScan,
    /// PROPHECY writes to computed addresses
    TemporalScatter,
    /// Repeated application until stable
    FixedPointIteration,

    /// Generic loop (no specialization available)
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
    /// Get the display name of this pattern.
    pub fn name(&self) -> &'static str {
        match self {
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
            TracePattern::ReadModifyWrite => "read-modify-write",
            TracePattern::ConvergenceLoop => "convergence",
            TracePattern::CausalChain => "causal-chain",
            TracePattern::BootstrapGuess => "bootstrap",
            TracePattern::WitnessSearch => "witness-search",
            TracePattern::TemporalScan => "temporal-scan",
            TracePattern::TemporalScatter => "temporal-scatter",
            TracePattern::FixedPointIteration => "fixed-point",
            TracePattern::Generic => "generic",
        }
    }

    /// Get expected performance multiplier.
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
            // Temporal patterns - epoch reduction potential
            TracePattern::ReadModifyWrite => 2.0,
            TracePattern::ConvergenceLoop => 3.0,
            TracePattern::CausalChain => 1.5,
            TracePattern::BootstrapGuess => 2.0,
            TracePattern::WitnessSearch => 4.0,
            TracePattern::TemporalScan => 2.5,
            TracePattern::TemporalScatter => 2.0,
            TracePattern::FixedPointIteration => 3.0,
            // Generic - minimal speedup from compilation
            TracePattern::Generic => 1.2,
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
}

/// Detect the pattern from a trace analysis.
pub fn detect_pattern(analysis: &TraceAnalysis) -> TracePattern {
    // Temporal patterns take precedence
    if analysis.is_temporal {
        return detect_temporal_pattern(analysis);
    }

    // Pure pattern detection
    detect_pure_pattern(analysis)
}

/// Detect temporal patterns based on ORACLE/PROPHECY usage.
fn detect_temporal_pattern(analysis: &TraceAnalysis) -> TracePattern {
    let has_oracle = analysis.oracle_count > 0;
    let has_prophecy = analysis.prophecy_count > 0;

    match (has_oracle, has_prophecy) {
        (true, true) => {
            // Both ORACLE and PROPHECY - could be various patterns
            if analysis.oracle_count == 1 && analysis.prophecy_count == 1 {
                TracePattern::ReadModifyWrite
            } else if analysis.has_loop {
                if analysis.comparison_count > 0 {
                    TracePattern::ConvergenceLoop
                } else {
                    TracePattern::FixedPointIteration
                }
            } else {
                TracePattern::CausalChain
            }
        }
        (true, false) => {
            // Only ORACLE - reading patterns
            if analysis.oracle_count > 2 {
                TracePattern::TemporalScan
            } else if analysis.comparison_count > 0 {
                TracePattern::WitnessSearch
            } else {
                TracePattern::BootstrapGuess
            }
        }
        (false, true) => {
            // Only PROPHECY - writing patterns
            if analysis.prophecy_count > 2 {
                TracePattern::TemporalScatter
            } else {
                TracePattern::Generic
            }
        }
        (false, false) => TracePattern::Generic,
    }
}

/// Detect pure computational patterns.
fn detect_pure_pattern(analysis: &TraceAnalysis) -> TracePattern {
    // Check for Fibonacci signature: SWAP OVER ADD (stack_count >= 2, arithmetic == 1)
    if analysis.stack_count >= 2 && analysis.arithmetic_count == 1 {
        return TracePattern::FibonacciStep;
    }

    // Check for specific signatures based on operation density
    let density = analysis.arithmetic_density();

    if density > 0.5 {
        // Arithmetic-heavy loop
        if analysis.arithmetic_count >= 2 {
            if analysis.comparison_count > 0 {
                TracePattern::MinMaxLoop
            } else {
                TracePattern::ReductionLoop
            }
        } else {
            TracePattern::CounterLoop
        }
    } else if analysis.has_loop {
        TracePattern::Generic
    } else {
        TracePattern::Generic
    }
}

/// Score a pattern match (0.0 to 1.0, higher is better match).
pub fn score_pattern(pattern: TracePattern, analysis: &TraceAnalysis) -> f64 {
    match pattern {
        TracePattern::FibonacciStep => {
            // Fibonacci: expects SWAP OVER ADD with few other ops
            if analysis.stack_count >= 2 && analysis.arithmetic_count == 1 {
                0.9
            } else {
                0.1
            }
        }
        TracePattern::CounterLoop => {
            if analysis.arithmetic_count == 1 && analysis.push_count <= 2 {
                0.8
            } else {
                0.3
            }
        }
        TracePattern::ReadModifyWrite => {
            if analysis.oracle_count == 1 && analysis.prophecy_count == 1 {
                0.9
            } else {
                0.2
            }
        }
        TracePattern::ConvergenceLoop => {
            if analysis.has_loop && analysis.oracle_count > 0 && analysis.comparison_count > 0 {
                0.7
            } else {
                0.2
            }
        }
        _ => 0.5, // Default medium score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_names() {
        assert_eq!(TracePattern::FibonacciStep.name(), "fibonacci");
        assert_eq!(TracePattern::ReadModifyWrite.name(), "read-modify-write");
    }

    #[test]
    fn test_pattern_categories() {
        assert_eq!(TracePattern::FibonacciStep.category(), PatternCategory::Pure);
        assert_eq!(TracePattern::ReadModifyWrite.category(), PatternCategory::Temporal);
        assert_eq!(TracePattern::Generic.category(), PatternCategory::Generic);
    }

    #[test]
    fn test_pattern_speedup() {
        assert!(TracePattern::FibonacciStep.expected_speedup() > 1.0);
        assert!(TracePattern::Generic.expected_speedup() >= 1.0);
    }

    #[test]
    fn test_detect_pure_pattern() {
        let mut analysis = TraceAnalysis::default();
        analysis.total_ops = 4;
        analysis.stack_count = 2;
        analysis.arithmetic_count = 1;

        let pattern = detect_pure_pattern(&analysis);
        assert_eq!(pattern, TracePattern::FibonacciStep);
    }

    #[test]
    fn test_detect_temporal_pattern() {
        let mut analysis = TraceAnalysis::default();
        analysis.total_ops = 5;
        analysis.oracle_count = 1;
        analysis.prophecy_count = 1;
        analysis.is_temporal = true;

        let pattern = detect_pattern(&analysis);
        assert_eq!(pattern, TracePattern::ReadModifyWrite);
    }
}
