//! Shared test utilities for Ourochronos integration tests.
//!
//! This module provides:
//! - Program parsing helpers
//! - Configuration builders
//! - Assertion helpers for convergence results
//! - Test data generators
//!
//! ## AAA Pattern
//!
//! All tests follow the Arrange-Act-Assert pattern:
//! - Arrange: Set up test fixtures and configuration
//! - Act: Execute the operation under test
//! - Assert: Verify the expected outcome

use ourochronos::*;
use ourochronos::timeloop::{ConvergenceStatus, ParadoxDiagnosis};

// =============================================================================
// Program Parsing Utilities
// =============================================================================

/// Parse Ourochronos source code into a Program.
///
/// # Panics
/// Panics if parsing fails, which is appropriate for test code.
pub fn parse(code: &str) -> Program {
    let tokens = tokenize(code);
    let mut parser = Parser::new(&tokens);
    parser.parse_program().expect("Failed to parse program")
}

/// Parse and validate a program, returning parsing errors if any.
pub fn try_parse(code: &str) -> Result<Program, String> {
    let tokens = tokenize(code);
    let mut parser = Parser::new(&tokens);
    parser.parse_program().map_err(|e| format!("{:?}", e))
}

// =============================================================================
// Configuration Builders
// =============================================================================

/// Create a default configuration for standard execution.
pub fn default_config() -> Config {
    Config {
        max_epochs: 100,
        mode: ExecutionMode::Standard,
        seed: 0,
        verbose: false,
        frozen_inputs: Vec::new(),
        max_instructions: 10_000_000,
    }
}

/// Create a configuration with custom epoch limit.
pub fn config_with_epochs(max_epochs: usize) -> Config {
    Config {
        max_epochs,
        ..default_config()
    }
}

/// Create a configuration for action-guided execution.
pub fn action_config() -> Config {
    Config {
        max_epochs: 100,
        mode: ExecutionMode::ActionGuided {
            config: ActionConfig::anti_trivial(),
            num_seeds: 4,
        },
        seed: 0,
        verbose: false,
        frozen_inputs: Vec::new(),
        max_instructions: 10_000_000,
    }
}

/// Create a configuration with custom number of seeds for action-guided mode.
pub fn action_config_with_seeds(num_seeds: usize) -> Config {
    Config {
        max_epochs: 100,
        mode: ExecutionMode::ActionGuided {
            config: ActionConfig::anti_trivial(),
            num_seeds,
        },
        seed: 0,
        verbose: false,
        frozen_inputs: Vec::new(),
        max_instructions: 10_000_000,
    }
}

/// Create a configuration for diagnostic mode.
pub fn diagnostic_config() -> Config {
    Config {
        mode: ExecutionMode::Diagnostic,
        max_epochs: 100,
        seed: 0,
        verbose: false,
        frozen_inputs: Vec::new(),
        max_instructions: 10_000_000,
    }
}

/// Create a configuration with frozen inputs.
pub fn config_with_frozen_inputs(inputs: Vec<u64>) -> Config {
    Config {
        frozen_inputs: inputs,
        ..default_config()
    }
}

// =============================================================================
// Assertion Helpers
// =============================================================================

/// Assert that a program converges to a consistent state.
pub fn assert_consistent(result: &ConvergenceStatus) {
    assert!(
        matches!(result, ConvergenceStatus::Consistent { .. }),
        "Expected Consistent, got {:?}",
        result
    );
}

/// Assert that a program converges in a specific number of epochs.
pub fn assert_consistent_in_epochs(result: &ConvergenceStatus, expected_epochs: usize) {
    match result {
        ConvergenceStatus::Consistent { epochs, .. } => {
            assert_eq!(
                *epochs, expected_epochs,
                "Expected {} epochs, got {}",
                expected_epochs, epochs
            );
        }
        _ => panic!("Expected Consistent, got {:?}", result),
    }
}

/// Assert that a program oscillates with a specific period.
pub fn assert_oscillation_with_period(result: &ConvergenceStatus, expected_period: usize) {
    match result {
        ConvergenceStatus::Oscillation { period, .. } => {
            assert_eq!(
                *period, expected_period,
                "Expected oscillation period {}, got {}",
                expected_period, period
            );
        }
        _ => panic!("Expected Oscillation, got {:?}", result),
    }
}

/// Assert that a program oscillates (any period).
pub fn assert_oscillation(result: &ConvergenceStatus) {
    assert!(
        matches!(result, ConvergenceStatus::Oscillation { .. }),
        "Expected Oscillation, got {:?}",
        result
    );
}

/// Assert that a program times out or diverges.
pub fn assert_timeout_or_divergence(result: &ConvergenceStatus) {
    assert!(
        matches!(
            result,
            ConvergenceStatus::Timeout { .. } | ConvergenceStatus::Divergence { .. }
        ),
        "Expected Timeout or Divergence, got {:?}",
        result
    );
}

/// Assert that a program hits a paradox.
pub fn assert_paradox(result: &ConvergenceStatus) {
    assert!(
        matches!(result, ConvergenceStatus::Paradox { .. }),
        "Expected Paradox, got {:?}",
        result
    );
}

/// Extract memory from a consistent result, panicking if not consistent.
pub fn extract_memory(result: &ConvergenceStatus) -> &Memory {
    match result {
        ConvergenceStatus::Consistent { memory, .. } => memory,
        _ => panic!("Expected Consistent, got {:?}", result),
    }
}

/// Extract output from a consistent result, panicking if not consistent.
pub fn extract_output(result: &ConvergenceStatus) -> &Vec<OutputItem> {
    match result {
        ConvergenceStatus::Consistent { output, .. } => output,
        _ => panic!("Expected Consistent, got {:?}", result),
    }
}

/// Assert that a specific memory address has a given value.
pub fn assert_memory_value(result: &ConvergenceStatus, address: u16, expected: u64) {
    let memory = extract_memory(result);
    let actual = memory.read(address).val;
    assert_eq!(
        actual, expected,
        "Memory[{}] = {}, expected {}",
        address, actual, expected
    );
}

/// Assert that output contains a specific value.
pub fn assert_output_contains(result: &ConvergenceStatus, expected: u64) {
    let output = extract_output(result);
    let values: Vec<u64> = output.iter().filter_map(|o| {
        if let OutputItem::Val(v) = o { Some(v.val) } else { None }
    }).collect();
    assert!(
        values.contains(&expected),
        "Output {:?} does not contain {}",
        values,
        expected
    );
}

/// Assert that output is non-empty.
pub fn assert_has_output(result: &ConvergenceStatus) {
    let output = extract_output(result);
    assert!(!output.is_empty(), "Expected non-empty output");
}

/// Assert that output is empty.
pub fn assert_no_output(result: &ConvergenceStatus) {
    let output = extract_output(result);
    assert!(output.is_empty(), "Expected empty output, got {:?}", output);
}

// =============================================================================
// Program Execution Helpers
// =============================================================================

/// Execute a program with default configuration.
pub fn run(code: &str) -> ConvergenceStatus {
    let program = parse(code);
    TimeLoop::new(default_config()).run(&program)
}

/// Execute a program with a specific configuration.
pub fn run_with_config(code: &str, config: Config) -> ConvergenceStatus {
    let program = parse(code);
    TimeLoop::new(config).run(&program)
}

/// Execute a program in action-guided mode.
pub fn run_action_guided(code: &str) -> ConvergenceStatus {
    let program = parse(code);
    TimeLoop::new(action_config()).run(&program)
}

/// Execute a program in diagnostic mode.
pub fn run_diagnostic(code: &str) -> ConvergenceStatus {
    let program = parse(code);
    TimeLoop::new(diagnostic_config()).run(&program)
}

// =============================================================================
// Determinism Verification
// =============================================================================

/// Verify that a program produces deterministic results across multiple runs.
pub fn verify_determinism(code: &str, runs: usize) -> bool {
    let program = parse(code);
    let config = default_config();

    let mut results: Vec<Vec<u64>> = Vec::new();

    for _ in 0..runs {
        let result = TimeLoop::new(config.clone()).run(&program);
        if let ConvergenceStatus::Consistent { memory, .. } = result {
            // Collect all non-zero memory values
            let mut values: Vec<u64> = (0..256u16)
                .map(|addr| memory.read(addr).val)
                .collect();
            results.push(values);
        } else {
            return false;
        }
    }

    // All runs must produce identical memory
    results.windows(2).all(|w| w[0] == w[1])
}

// =============================================================================
// Test Programs
// =============================================================================

/// Collection of canonical test programs for various scenarios.
pub mod programs {
    /// Trivial program that always converges in one epoch.
    pub const TRIVIAL: &str = "10 20 ADD OUTPUT";

    /// Self-fulfilling prophecy (immediate fixed point).
    pub const SELF_FULFILLING: &str = "0 ORACLE 0 PROPHECY";

    /// Grandfather paradox (period-2 oscillation).
    pub const GRANDFATHER_PARADOX: &str = "0 ORACLE NOT 0 PROPHECY";

    /// Divergence (unbounded growth).
    pub const DIVERGENCE: &str = "0 ORACLE 1 ADD 0 PROPHECY";

    /// Conditional convergence.
    pub const CONDITIONAL: &str = "0 ORACLE DUP 3 EQ IF { DUP 0 PROPHECY } ELSE { POP 3 0 PROPHECY }";

    /// Factor finder (for action-guided testing).
    pub const FACTOR_WITNESS: &str = r#"
        0 ORACLE
        DUP 1 GT IF {
            DUP 15 LT IF {
                DUP 15 SWAP MOD 0 EQ IF {
                    DUP 0 PROPHECY
                    OUTPUT
                } ELSE {
                    1 ADD 0 PROPHECY
                }
            } ELSE {
                1 ADD 0 PROPHECY
            }
        } ELSE {
            1 ADD 0 PROPHECY
        }
    "#;

    /// Arithmetic chain for determinism testing.
    pub const ARITHMETIC_CHAIN: &str = "5 3 MUL 2 ADD 7 SUB 0 PROPHECY";

    /// Division by zero handling.
    pub const DIV_BY_ZERO: &str = "10 0 DIV 0 PROPHECY";

    /// Stack manipulation test.
    pub const STACK_OPS: &str = "1 2 3 ROT DEPTH 0 PROPHECY";
}
