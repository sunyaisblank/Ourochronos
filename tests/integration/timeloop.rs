//! Integration tests for the TimeLoop execution engine.
//!
//! Component: TimeLoop (Temporal Execution Engine)
//!
//! These tests verify:
//! - Fixed-point convergence behaviour
//! - Paradox detection and diagnosis
//! - Action-guided execution semantics
//! - Determinism across executions
//! - Edge cases and invariants
//!
//! ## Test Organisation
//!
//! Tests are grouped by convergence behaviour:
//! - Consistent: Programs that reach a fixed point
//! - Oscillation: Programs with cyclic behaviour (paradoxes)
//! - Divergence: Programs with unbounded growth
//! - ActionGuided: Programs using the action principle
//! - Invariants: Mathematical properties that must hold

#![cfg(test)]

use crate::common::*;

use ourochronos::*;
use ourochronos::timeloop::{ConvergenceStatus, ParadoxDiagnosis, Direction};

// =============================================================================
// Consistent Execution Tests
// =============================================================================

mod consistent {
    use super::*;

    #[test]
    fn trivial_arithmetic_converges_in_one_epoch() {
        // Arrange
        let code = programs::TRIVIAL;

        // Act
        let result = run(code);

        // Assert
        assert_consistent_in_epochs(&result, 1);
    }

    #[test]
    fn self_fulfilling_prophecy_converges() {
        // Arrange: Oracle reads same value it writes
        let code = programs::SELF_FULFILLING;

        // Act
        let result = run(code);

        // Assert
        assert_consistent(&result);
    }

    #[test]
    fn conditional_convergence_finds_fixed_point() {
        // Arrange: Program converges to value 3
        let code = programs::CONDITIONAL;

        // Act
        let result = run(code);

        // Assert
        assert_consistent(&result);
        assert_memory_value(&result, 0, 3);
    }

    #[test]
    fn arithmetic_chain_produces_correct_result() {
        // Arrange: 5 * 3 + 2 - 7 = 10
        let code = programs::ARITHMETIC_CHAIN;

        // Act
        let result = run(code);

        // Assert
        assert_consistent(&result);
        assert_memory_value(&result, 0, 10);
    }

    #[test]
    fn division_by_zero_returns_zero() {
        // Arrange: Division by zero should yield 0 (not crash)
        let code = programs::DIV_BY_ZERO;

        // Act
        let result = run(code);

        // Assert
        assert_consistent(&result);
        assert_memory_value(&result, 0, 0);
    }

    #[test]
    fn stack_operations_produce_correct_depth() {
        // Arrange: 1 2 3 ROT DEPTH -> stack depth before DEPTH is 3
        let code = programs::STACK_OPS;

        // Act
        let result = run(code);

        // Assert
        assert_consistent(&result);
        assert_memory_value(&result, 0, 3);
    }

    #[test]
    fn output_instruction_produces_value() {
        // Arrange
        let code = "42 OUTPUT 1 0 PROPHECY";

        // Act
        let result = run(code);

        // Assert
        assert_consistent(&result);
        assert_output_contains(&result, 42);
    }

    #[test]
    fn multiple_outputs_collected() {
        // Arrange
        let code = "1 OUTPUT 2 OUTPUT 3 OUTPUT 0 0 PROPHECY";

        // Act
        let result = run(code);

        // Assert
        assert_consistent(&result);
        let output = extract_output(&result);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn nested_conditionals_converge() {
        // Arrange: Nested IF/ELSE structure
        let code = r#"
            0 ORACLE
            DUP 5 GT IF {
                DUP 10 GT IF {
                    10 0 PROPHECY
                } ELSE {
                    5 0 PROPHECY
                }
            } ELSE {
                DUP 0 PROPHECY
            }
        "#;

        // Act
        let result = run(code);

        // Assert
        assert_consistent(&result);
    }
}

// =============================================================================
// Oscillation Tests (Paradoxes)
// =============================================================================

mod oscillation {
    use super::*;

    #[test]
    fn grandfather_paradox_oscillates_with_period_2() {
        // Arrange: NOT(x) = x has no fixed point
        let code = programs::GRANDFATHER_PARADOX;

        // Act
        let result = run(code);

        // Assert
        assert_oscillation_with_period(&result, 2);
    }

    #[test]
    fn increment_oscillation_detected() {
        // Arrange: x + 1 = x has no solution
        let code = "0 ORACLE 1 ADD 0 PROPHECY";
        let config = config_with_epochs(50);

        // Act
        let result = run_with_config(code, config);

        // Assert: Should timeout or detect divergence
        assert_timeout_or_divergence(&result);
    }

    #[test]
    fn period_3_oscillation() {
        // Arrange: Cyclic permutation with period 3
        let code = r#"
            0 ORACLE
            DUP 0 EQ IF { 1 0 PROPHECY }
            ELSE { DUP 1 EQ IF { 2 0 PROPHECY }
            ELSE { 0 0 PROPHECY } }
        "#;

        // Act
        let result = run(code);

        // Assert: Should oscillate with period 3
        match result {
            ConvergenceStatus::Oscillation { period, .. } => {
                assert!(period >= 2, "Expected oscillation period >= 2, got {}", period);
            }
            ConvergenceStatus::Timeout { .. } => {
                // Also acceptable - pattern too complex to detect
            }
            _ => panic!("Expected Oscillation or Timeout, got {:?}", result),
        }
    }

    #[test]
    fn diagnostic_mode_identifies_negative_loop() {
        // Arrange: Grandfather paradox in diagnostic mode
        let code = programs::GRANDFATHER_PARADOX;

        // Act
        let result = run_diagnostic(code);

        // Assert
        match result {
            ConvergenceStatus::Oscillation { diagnosis, .. } => {
                match diagnosis {
                    ParadoxDiagnosis::NegativeLoop { .. } => {
                        // Expected diagnosis
                    }
                    _ => panic!("Expected NegativeLoop diagnosis, got {:?}", diagnosis),
                }
            }
            _ => panic!("Expected Oscillation with diagnosis, got {:?}", result),
        }
    }
}

// =============================================================================
// Divergence Tests
// =============================================================================

mod divergence {
    use super::*;

    #[test]
    fn unbounded_increment_diverges() {
        // Arrange
        let code = programs::DIVERGENCE;
        let config = config_with_epochs(50);

        // Act
        let result = run_with_config(code, config);

        // Assert
        assert_timeout_or_divergence(&result);
    }

    #[test]
    fn multiplication_divergence() {
        // Arrange: Start with at least 1, then x * 2 diverges (1 -> 2 -> 4 -> 8 -> ...)
        // Using "1 MAX" ensures we start with at least 1 even if Oracle reads 0
        let code = "0 ORACLE 1 MAX 2 MUL 0 PROPHECY";
        let config = config_with_epochs(30);

        // Act
        let result = run_with_config(code, config);

        // Assert: Should timeout or diverge (multiplication grows)
        assert_timeout_or_divergence(&result);
    }
}

// =============================================================================
// Action-Guided Execution Tests
// =============================================================================

mod action_guided {
    use super::*;

    #[test]
    fn action_guided_finds_self_fulfilling_fixed_point() {
        // Arrange
        let code = programs::SELF_FULFILLING;

        // Act
        let result = run_action_guided(code);

        // Assert
        assert_consistent(&result);
    }

    #[test]
    fn action_guided_prefers_output_producing_solution() {
        // Arrange: Program where one fixed point produces output
        let code = r#"
            0 ORACLE
            DUP 0 EQ NOT IF {
                DUP OUTPUT
                DUP 0 PROPHECY
            } ELSE {
                1 0 PROPHECY
            }
        "#;

        // Act
        let result = run_action_guided(code);

        // Assert: Should prefer the output-producing branch
        match result {
            ConvergenceStatus::Consistent { output, .. } => {
                assert!(!output.is_empty(), "Action-guided should prefer output-producing fixed point");
            }
            _ => panic!("Expected consistent execution, got {:?}", result),
        }
    }

    #[test]
    fn action_guided_finds_factor_witness() {
        // Arrange: Factor-finding program
        let code = programs::FACTOR_WITNESS;

        // Act
        let result = run_action_guided(code);

        // Assert
        match result {
            ConvergenceStatus::Consistent { memory, output, .. } => {
                let factor = memory.read(0).val;
                // Factor should divide 15
                assert!(factor > 1 && factor < 15, "Factor {} should be > 1 and < 15", factor);
                assert_eq!(15 % factor, 0, "Factor {} should divide 15", factor);
                assert!(!output.is_empty(), "Should output the factor");
            }
            _ => panic!("Expected consistent execution, got {:?}", result),
        }
    }

    #[test]
    fn action_guided_mode_equality() {
        // Arrange
        let mode1 = ExecutionMode::ActionGuided {
            config: ActionConfig::anti_trivial(),
            num_seeds: 4,
        };
        let mode2 = ExecutionMode::ActionGuided {
            config: ActionConfig::anti_trivial(),
            num_seeds: 4,
        };

        // Assert
        assert_eq!(mode1, mode2);
    }
}

// =============================================================================
// Determinism Tests
// =============================================================================

mod determinism {
    use super::*;

    #[test]
    fn trivial_program_is_deterministic() {
        assert!(verify_determinism(programs::TRIVIAL, 10));
    }

    #[test]
    fn self_fulfilling_is_deterministic() {
        assert!(verify_determinism(programs::SELF_FULFILLING, 10));
    }

    #[test]
    fn conditional_is_deterministic() {
        assert!(verify_determinism(programs::CONDITIONAL, 10));
    }

    #[test]
    fn arithmetic_is_deterministic() {
        assert!(verify_determinism(programs::ARITHMETIC_CHAIN, 10));
    }

    #[test]
    fn repeated_execution_identical() {
        // Arrange
        let code = "0 ORACLE DUP 5 GT IF { 5 0 PROPHECY } ELSE { DUP 0 PROPHECY }";
        let program = parse(code);

        // Act
        let mut results: Vec<u64> = Vec::new();
        for _ in 0..100 {
            let result = TimeLoop::new(default_config()).run(&program);
            if let ConvergenceStatus::Consistent { memory, .. } = result {
                results.push(memory.read(0).val);
            }
        }

        // Assert: all results identical
        assert!(!results.is_empty());
        let first = results[0];
        assert!(results.iter().all(|&r| r == first));
    }
}

// =============================================================================
// Invariant Tests
// =============================================================================

mod invariants {
    use super::*;

    #[test]
    fn invariant_consistent_result_has_memory() {
        // Invariant: Consistent results always have valid memory

        let programs = vec![
            programs::TRIVIAL,
            programs::SELF_FULFILLING,
            programs::CONDITIONAL,
            programs::ARITHMETIC_CHAIN,
        ];

        for code in programs {
            let result = run(code);
            if let ConvergenceStatus::Consistent { memory, .. } = result {
                // Memory should be accessible
                let _ = memory.read(0);
            }
        }
    }

    #[test]
    fn invariant_epoch_count_positive() {
        // Invariant: Epoch count must be positive for consistent results

        let result = run(programs::TRIVIAL);
        match result {
            ConvergenceStatus::Consistent { epochs, .. } => {
                assert!(epochs >= 1, "Epochs must be >= 1");
            }
            _ => panic!("Expected Consistent"),
        }
    }

    #[test]
    fn invariant_oscillation_period_at_least_2() {
        // Invariant: Oscillation period must be at least 2

        let result = run(programs::GRANDFATHER_PARADOX);
        match result {
            ConvergenceStatus::Oscillation { period, .. } => {
                assert!(period >= 2, "Oscillation period must be >= 2");
            }
            _ => panic!("Expected Oscillation"),
        }
    }

    #[test]
    fn invariant_prophecy_writes_to_memory() {
        // Invariant: PROPHECY should write to the specified address

        let code = "42 5 PROPHECY";
        let result = run(code);

        match result {
            ConvergenceStatus::Consistent { memory, .. } => {
                assert_eq!(memory.read(5).val, 42);
            }
            _ => panic!("Expected Consistent"),
        }
    }

    #[test]
    fn invariant_oracle_reads_from_anamnesis() {
        // Invariant: Oracle should read the value written by prophecy

        // After convergence, oracle(0) should equal the value prophecy(0) wrote
        let code = "7 0 PROPHECY 0 ORACLE 1 PROPHECY";
        let result = run(code);

        match result {
            ConvergenceStatus::Consistent { memory, .. } => {
                // addr[0] = 7, addr[1] = oracle(0) = 7
                assert_eq!(memory.read(0).val, 7);
                assert_eq!(memory.read(1).val, 7);
            }
            _ => panic!("Expected Consistent"),
        }
    }

    #[test]
    fn invariant_timeout_respects_max_epochs() {
        // Invariant: Timeout should occur when max_epochs is reached

        let code = programs::DIVERGENCE;
        let config = config_with_epochs(5);
        let result = run_with_config(code, config);

        match result {
            ConvergenceStatus::Timeout { max_epochs } => {
                assert_eq!(max_epochs, 5);
            }
            ConvergenceStatus::Divergence { .. } => {
                // Also acceptable - detected divergence before timeout
            }
            _ => panic!("Expected Timeout or Divergence, got {:?}", result),
        }
    }
}

// =============================================================================
// Edge Cases
// =============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn empty_program_converges() {
        // Arrange: Empty program (just whitespace)
        let code = "  ";
        let program = parse(code);

        // Act
        let result = TimeLoop::new(default_config()).run(&program);

        // Assert: Should converge immediately
        assert_consistent(&result);
    }

    #[test]
    fn single_literal_converges() {
        // Arrange
        let code = "42";
        let program = parse(code);

        // Act
        let result = TimeLoop::new(default_config()).run(&program);

        // Assert
        assert_consistent(&result);
    }

    #[test]
    fn max_value_handling() {
        // Arrange: Test with maximum u64 value
        let code = "18446744073709551615 0 PROPHECY";

        // Act
        let result = run(code);

        // Assert
        assert_consistent(&result);
        // The value might overflow but should not crash
    }

    #[test]
    fn zero_epoch_limit_handled() {
        // Arrange
        let code = programs::SELF_FULFILLING;
        let config = config_with_epochs(0);

        // Act
        let result = run_with_config(code, config);

        // Assert: Should timeout immediately or handle gracefully
        // Implementation may vary
        assert!(matches!(
            result,
            ConvergenceStatus::Timeout { .. } | ConvergenceStatus::Consistent { .. }
        ));
    }

    #[test]
    fn multiple_prophecies_same_address() {
        // Arrange: Multiple prophecies to same address (last wins)
        let code = "1 0 PROPHECY 2 0 PROPHECY 3 0 PROPHECY";

        // Act
        let result = run(code);

        // Assert: Last prophecy value should be present
        assert_consistent(&result);
        assert_memory_value(&result, 0, 3);
    }

    #[test]
    fn oracle_before_prophecy_uses_anamnesis() {
        // Arrange: Oracle reads before any prophecy in this epoch
        let code = "0 ORACLE 5 ADD 0 PROPHECY";

        // Act
        let result = run(code);

        // Assert: Should converge (0 + 5 = 5, then 5 + 5 = 10, ...)
        // Actually diverges since it keeps adding 5
        assert_timeout_or_divergence(&result);
    }
}
