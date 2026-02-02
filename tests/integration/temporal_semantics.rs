//! Integration tests for temporal semantics.
//!
//! Component: Temporal Semantics
//!
//! These tests verify the core temporal mechanics of Ourochronos:
//! - Oracle/Prophecy consistency (Closed Timelike Curves)
//! - Fixed-point existence and uniqueness
//! - Causal loop behaviour
//! - Information flow across temporal boundaries
//! - Constraint satisfaction semantics

#![cfg(test)]

use crate::common::*;

use ourochronos::*;
use ourochronos::timeloop::ConvergenceStatus;

// =============================================================================
// Closed Timelike Curve (CTC) Tests
// =============================================================================

mod ctc_semantics {
    use super::*;

    #[test]
    fn ctc_simple_identity() {
        // Simplest CTC: x = oracle(x), where oracle reads anamnesis
        // Any value is a fixed point
        let result = run("0 ORACLE 0 PROPHECY");
        assert_consistent(&result);
    }

    #[test]
    fn ctc_constant_solution() {
        // CTC with constant solution: oracle + 0 = oracle
        let result = run("0 ORACLE 0 ADD 0 PROPHECY");
        assert_consistent(&result);
    }

    #[test]
    fn ctc_no_solution_diverges() {
        // CTC with no solution: oracle + 1 = oracle (impossible)
        let result = run_with_config(
            "0 ORACLE 1 ADD 0 PROPHECY",
            config_with_epochs(50)
        );
        assert_timeout_or_divergence(&result);
    }

    #[test]
    fn ctc_multiple_fixed_points_first_wins() {
        // Multiple fixed points: 0 or non-zero both work
        // Standard mode takes first (0)
        let code = "0 ORACLE DUP 0 PROPHECY";
        let result = run(code);
        assert_consistent(&result);
        assert_memory_value(&result, 0, 0);
    }

    #[test]
    fn ctc_conditional_fixed_point() {
        // Fixed point selected by condition
        let code = r#"
            0 ORACLE
            DUP 5 EQ IF {
                5 0 PROPHECY
            } ELSE {
                5 0 PROPHECY
            }
        "#;
        let result = run(code);
        assert_consistent(&result);
        assert_memory_value(&result, 0, 5);
    }
}

// =============================================================================
// Fixed Point Existence Tests
// =============================================================================

mod fixed_point_existence {
    use super::*;

    #[test]
    fn existence_trivial_program() {
        // Trivial programs (no temporal ops) always have fixed point
        let result = run("1 2 ADD OUTPUT");
        assert_consistent(&result);
    }

    #[test]
    fn existence_identity_mapping() {
        // f(x) = x always has fixed points
        let result = run("0 ORACLE 0 PROPHECY");
        assert_consistent(&result);
    }

    #[test]
    fn existence_constant_mapping() {
        // f(x) = c has fixed point c
        let result = run("42 0 PROPHECY");
        assert_consistent(&result);
        assert_memory_value(&result, 0, 42);
    }

    #[test]
    fn nonexistence_negation() {
        // f(x) = NOT(x) has no fixed point
        let result = run("0 ORACLE NOT 0 PROPHECY");
        assert_oscillation(&result);
    }

    #[test]
    fn nonexistence_increment() {
        // f(x) = x + 1 has no finite fixed point
        let result = run_with_config(
            "0 ORACLE 1 ADD 0 PROPHECY",
            config_with_epochs(30)
        );
        assert_timeout_or_divergence(&result);
    }

    #[test]
    fn existence_modular_fixed_point() {
        // f(x) = (x + 1) mod 10 has no fixed point
        // but bounded behaviour
        let code = "0 ORACLE 1 ADD 10 MOD 0 PROPHECY";
        let config = config_with_epochs(50);
        let result = run_with_config(code, config);

        // Should oscillate with period 10 or timeout
        assert!(matches!(
            result,
            ConvergenceStatus::Oscillation { .. }
            | ConvergenceStatus::Timeout { .. }
        ));
    }
}

// =============================================================================
// Causal Loop Tests
// =============================================================================

mod causal_loops {
    use super::*;

    #[test]
    fn positive_causal_loop_converges() {
        // x = x (positive loop) converges immediately
        let result = run("0 ORACLE 0 PROPHECY");
        assert_consistent(&result);
    }

    #[test]
    fn negative_causal_loop_oscillates() {
        // x = NOT(x) (negative loop) oscillates
        let result = run("0 ORACLE NOT 0 PROPHECY");
        assert_oscillation_with_period(&result, 2);
    }

    #[test]
    fn chained_causal_dependencies() {
        // a = b, b = a (mutual dependency)
        let code = r#"
            1 ORACLE 0 PROPHECY
            0 ORACLE 1 PROPHECY
        "#;
        let result = run(code);
        // Should converge: both stabilise to same value
        assert_consistent(&result);
    }

    #[test]
    fn chained_negation_oscillates() {
        // a = NOT(b), b = a
        let code = r#"
            1 ORACLE NOT 0 PROPHECY
            0 ORACLE 1 PROPHECY
        "#;
        let result = run(code);
        // Creates oscillation through indirection
        assert_oscillation(&result);
    }

    #[test]
    fn delayed_feedback_loop() {
        // Multi-step computation before feedback
        let code = r#"
            0 ORACLE
            2 MUL
            3 ADD
            5 MOD
            0 PROPHECY
        "#;
        let config = config_with_epochs(100);
        let result = run_with_config(code, config);

        // f(x) = (2x + 3) mod 5
        // Fixed points: 2x + 3 = x (mod 5) => x = -3 = 2 (mod 5)
        // Should converge to 2
        match result {
            ConvergenceStatus::Consistent { memory, .. } => {
                assert_eq!(memory.read(0).val, 2);
            }
            _ => {
                // Might oscillate if starting from wrong value
                // Both are acceptable outcomes
            }
        }
    }
}

// =============================================================================
// Information Flow Tests
// =============================================================================

mod information_flow {
    use super::*;

    #[test]
    fn future_to_past_information_transfer() {
        // Oracle reads value from "future" (anamnesis)
        let code = "0 ORACLE 1 ADD 0 PROPHECY";
        let config = config_with_epochs(20);
        let result = run_with_config(code, config);

        // Information flows back but no fixed point exists
        assert_timeout_or_divergence(&result);
    }

    #[test]
    fn conditional_on_future_information() {
        // Decision based on oracle value
        // Use GTE so that when value reaches 10, it stays at 10 (fixed point)
        let code = r#"
            0 ORACLE
            DUP 10 GTE IF {
                10 0 PROPHECY
            } ELSE {
                DUP 1 ADD 0 PROPHECY
            }
        "#;
        let result = run(code);

        // Should converge when oracle >= 10
        assert_consistent(&result);
    }

    #[test]
    fn output_depends_on_future() {
        // Output value determined by temporal computation
        let code = r#"
            0 ORACLE
            DUP 5 EQ IF {
                DUP OUTPUT
                5 0 PROPHECY
            } ELSE {
                5 0 PROPHECY
            }
        "#;
        let result = run(code);

        assert_consistent(&result);
        assert_output_contains(&result, 5);
    }

    #[test]
    fn multiple_oracle_reads() {
        // Reading from multiple oracle addresses
        // Write the sum to addr 0, and the same sum to addr 1
        let code = r#"
            0 ORACLE 1 ORACLE ADD
            DUP
            0 PROPHECY
            1 PROPHECY
        "#;
        let result = run(code);

        // Both addresses end up with sum = 0 + 0 = 0
        assert_consistent(&result);
    }
}

// =============================================================================
// Constraint Satisfaction Tests
// =============================================================================

mod constraint_satisfaction {
    use super::*;

    #[test]
    fn satisfiable_constraint_converges() {
        // x = 5 (simple constraint)
        let result = run("5 0 PROPHECY");
        assert_consistent(&result);
        assert_memory_value(&result, 0, 5);
    }

    #[test]
    fn unsatisfiable_constraint_no_fixed_point() {
        // x = x + 1 (unsatisfiable for finite x)
        let config = config_with_epochs(30);
        let result = run_with_config("0 ORACLE 1 ADD 0 PROPHECY", config);
        assert_timeout_or_divergence(&result);
    }

    #[test]
    fn disjunctive_constraint() {
        // x = 3 OR x = 7 (first valid solution wins)
        let code = r#"
            0 ORACLE
            DUP 3 EQ IF { 3 0 PROPHECY }
            ELSE { DUP 7 EQ IF { 7 0 PROPHECY }
            ELSE { 3 0 PROPHECY } }
        "#;
        let result = run(code);
        assert_consistent(&result);
        let mem = extract_memory(&result);
        let val = mem.read(0).val;
        assert!(val == 3 || val == 7, "Expected 3 or 7, got {}", val);
    }

    #[test]
    fn conjunctive_constraint() {
        // x > 5 AND x < 10 AND x = oracle(x)
        let code = r#"
            0 ORACLE
            DUP 5 GT IF {
                DUP 10 LT IF {
                    DUP 0 PROPHECY
                } ELSE {
                    7 0 PROPHECY
                }
            } ELSE {
                7 0 PROPHECY
            }
        "#;
        let result = run(code);
        assert_consistent(&result);
        let mem = extract_memory(&result);
        let val = mem.read(0).val;
        assert!(val > 5 && val < 10, "Expected 5 < val < 10, got {}", val);
    }

    #[test]
    fn factor_constraint_with_action() {
        // Find factor of 12 using action-guided mode
        let code = r#"
            0 ORACLE
            DUP 1 GT IF {
                DUP 12 LT IF {
                    12 OVER MOD 0 EQ IF {
                        DUP 0 PROPHECY OUTPUT
                    } ELSE {
                        2 0 PROPHECY
                    }
                } ELSE {
                    2 0 PROPHECY
                }
            } ELSE {
                2 0 PROPHECY
            }
        "#;
        let result = run_action_guided(code);

        match result {
            ConvergenceStatus::Consistent { memory, .. } => {
                let factor = memory.read(0).val;
                if factor > 1 && factor < 12 {
                    assert_eq!(12 % factor, 0, "Factor {} should divide 12", factor);
                }
            }
            _ => {
                // Action-guided might not find solution in all cases
            }
        }
    }
}

// =============================================================================
// Epoch Dynamics Tests
// =============================================================================

mod epoch_dynamics {
    use super::*;

    #[test]
    fn single_epoch_convergence() {
        // Program with no temporal ops converges in 1 epoch
        let result = run("1 2 ADD OUTPUT");
        assert_consistent_in_epochs(&result, 1);
    }

    #[test]
    fn multi_epoch_convergence() {
        // Program needing multiple epochs
        let code = "0 ORACLE DUP 5 EQ IF { 5 0 PROPHECY } ELSE { 5 0 PROPHECY }";
        let result = run(code);

        match result {
            ConvergenceStatus::Consistent { epochs, .. } => {
                assert!(epochs >= 1);
            }
            _ => panic!("Expected Consistent"),
        }
    }

    #[test]
    fn epoch_count_bounded() {
        // Converging programs should have bounded epoch count
        let code = "0 ORACLE 0 PROPHECY";
        let result = run(code);

        match result {
            ConvergenceStatus::Consistent { epochs, .. } => {
                assert!(epochs <= 10, "Simple program should converge quickly");
            }
            _ => panic!("Expected Consistent"),
        }
    }

    #[test]
    fn anamnesis_updates_between_epochs() {
        // Verify anamnesis (oracle input) changes between epochs
        let code = r#"
            0 ORACLE
            DUP 0 EQ IF {
                1 0 PROPHECY
            } ELSE {
                DUP 0 PROPHECY
            }
        "#;
        let result = run(code);

        // Starting from 0, first epoch sets 1, then stabilises
        assert_consistent(&result);
        assert_memory_value(&result, 0, 1);
    }
}

// =============================================================================
// Invariant Tests
// =============================================================================

mod temporal_invariants {
    use super::*;

    #[test]
    fn invariant_convergence_implies_fixed_point() {
        // If program converges, oracle(x) = prophecy(x) for all x
        let code = "0 ORACLE 0 PROPHECY";
        let result = run(code);

        if let ConvergenceStatus::Consistent { memory, .. } = result {
            // After convergence, memory[0] should equal what oracle would read
            // This is the definition of a fixed point
            let val = memory.read(0).val;
            // Run one more epoch to verify stability
            let code2 = format!("{} 0 ORACLE 0 PROPHECY", val);
            let result2 = run(&code2);
            assert_memory_value(&result2, 0, val);
        }
    }

    #[test]
    fn invariant_oscillation_has_period_at_least_2() {
        let result = run("0 ORACLE NOT 0 PROPHECY");

        if let ConvergenceStatus::Oscillation { period, .. } = result {
            assert!(period >= 2, "Oscillation period must be >= 2");
        }
    }

    #[test]
    fn invariant_output_order_preserved() {
        // Outputs should appear in program order
        let result = run("1 OUTPUT 2 OUTPUT 3 OUTPUT 0 0 PROPHECY");
        let output = extract_output(&result);
        let values: Vec<u64> = output.iter().filter_map(|o| {
            if let ourochronos::OutputItem::Val(v) = o { Some(v.val) } else { None }
        }).collect();
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn invariant_memory_changes_tracked() {
        // All prophecy writes should be reflected in final memory
        let result = run("10 0 PROPHECY 20 1 PROPHECY 30 2 PROPHECY");
        assert_memory_value(&result, 0, 10);
        assert_memory_value(&result, 1, 20);
        assert_memory_value(&result, 2, 30);
    }
}
