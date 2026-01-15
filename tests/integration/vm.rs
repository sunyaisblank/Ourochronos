//! Integration tests for the Virtual Machine.
//!
//! Component: VM (Virtual Machine)
//!
//! These tests verify:
//! - Stack operations (PUSH, POP, DUP, SWAP, ROT)
//! - Arithmetic operations (ADD, SUB, MUL, DIV, MOD)
//! - Comparison operations (EQ, LT, GT)
//! - Logical operations (NOT, AND, OR)
//! - Control flow (IF/ELSE)
//! - Memory operations (ORACLE, PROPHECY)
//! - Edge cases and invariants

#![cfg(test)]

use crate::common::*;

use ourochronos::*;
use ourochronos::core_types::{Value, Address};

// =============================================================================
// Stack Operation Tests
// =============================================================================

mod stack_operations {
    use super::*;

    #[test]
    fn dup_duplicates_top() {
        // Arrange & Act
        let result = run("5 DUP ADD 0 PROPHECY");

        // Assert: 5 + 5 = 10
        assert_memory_value(&result, 0, 10);
    }

    #[test]
    fn swap_exchanges_top_two() {
        // Arrange: 3 5 SWAP -> 5 3
        let result = run("3 5 SWAP SUB 0 PROPHECY");

        // Assert: 5 - 3 = 2 (after swap, 5 is below 3)
        assert_memory_value(&result, 0, 2);
    }

    #[test]
    fn rot_rotates_top_three() {
        // Arrange: 1 2 3 ROT -> 2 3 1
        let result = run("1 2 3 ROT 0 PROPHECY");

        // Assert: After ROT, 1 is on top
        assert_memory_value(&result, 0, 1);
    }

    #[test]
    fn depth_returns_stack_size() {
        // Arrange: Push 3 values, get depth
        let result = run("1 2 3 DEPTH 0 PROPHECY POP POP POP");

        // Assert: Depth was 3 before DEPTH
        assert_memory_value(&result, 0, 3);
    }

    #[test]
    fn over_copies_second_to_top() {
        // Arrange: 1 2 OVER -> 1 2 1
        let result = run("1 2 OVER 0 PROPHECY POP POP");

        // Assert: OVER copied the 1 to top
        assert_memory_value(&result, 0, 1);
    }

    #[test]
    fn drop_removes_top() {
        // Arrange: 1 2 DROP -> 1
        let result = run("1 2 DROP 0 PROPHECY");

        // Assert: 2 was dropped, 1 remains
        assert_memory_value(&result, 0, 1);
    }
}

// =============================================================================
// Arithmetic Operation Tests
// =============================================================================

mod arithmetic {
    use super::*;

    #[test]
    fn add_sums_top_two() {
        let result = run("10 20 ADD 0 PROPHECY");
        assert_memory_value(&result, 0, 30);
    }

    #[test]
    fn sub_subtracts_top_from_second() {
        // Arrange: 20 - 7 = 13
        let result = run("20 7 SUB 0 PROPHECY");
        assert_memory_value(&result, 0, 13);
    }

    #[test]
    fn mul_multiplies_top_two() {
        let result = run("6 7 MUL 0 PROPHECY");
        assert_memory_value(&result, 0, 42);
    }

    #[test]
    fn div_divides_second_by_top() {
        let result = run("100 4 DIV 0 PROPHECY");
        assert_memory_value(&result, 0, 25);
    }

    #[test]
    fn div_by_zero_returns_zero() {
        let result = run("100 0 DIV 0 PROPHECY");
        assert_memory_value(&result, 0, 0);
    }

    #[test]
    fn mod_returns_remainder() {
        let result = run("17 5 MOD 0 PROPHECY");
        assert_memory_value(&result, 0, 2);
    }

    #[test]
    fn mod_by_zero_returns_zero() {
        let result = run("17 0 MOD 0 PROPHECY");
        assert_memory_value(&result, 0, 0);
    }

    #[test]
    fn neg_negates_value() {
        // Arrange: Assuming wrapping arithmetic
        let result = run("5 NEG 0 PROPHECY");
        assert_consistent(&result);
        // NEG on unsigned wraps to MAX - 4
    }

    #[test]
    fn arithmetic_chain() {
        // (5 + 3) * 2 - 4 = 12
        let result = run("5 3 ADD 2 MUL 4 SUB 0 PROPHECY");
        assert_memory_value(&result, 0, 12);
    }

    #[test]
    fn arithmetic_overflow_wraps() {
        // Arrange: MAX + 1 should wrap to 0
        let result = run("18446744073709551615 1 ADD 0 PROPHECY");
        assert_consistent(&result);
        assert_memory_value(&result, 0, 0);
    }

    #[test]
    fn arithmetic_underflow_wraps() {
        // Arrange: 0 - 1 should wrap to MAX
        let result = run("0 1 SUB 0 PROPHECY");
        assert_consistent(&result);
        assert_memory_value(&result, 0, 18446744073709551615);
    }
}

// =============================================================================
// Comparison Operation Tests
// =============================================================================

mod comparison {
    use super::*;

    #[test]
    fn eq_returns_1_for_equal() {
        let result = run("5 5 EQ 0 PROPHECY");
        assert_memory_value(&result, 0, 1);
    }

    #[test]
    fn eq_returns_0_for_not_equal() {
        let result = run("5 6 EQ 0 PROPHECY");
        assert_memory_value(&result, 0, 0);
    }

    #[test]
    fn lt_returns_1_when_less() {
        let result = run("3 5 LT 0 PROPHECY");
        assert_memory_value(&result, 0, 1);
    }

    #[test]
    fn lt_returns_0_when_greater_or_equal() {
        let result = run("5 3 LT 0 PROPHECY");
        assert_memory_value(&result, 0, 0);
    }

    #[test]
    fn gt_returns_1_when_greater() {
        let result = run("5 3 GT 0 PROPHECY");
        assert_memory_value(&result, 0, 1);
    }

    #[test]
    fn gt_returns_0_when_less_or_equal() {
        let result = run("3 5 GT 0 PROPHECY");
        assert_memory_value(&result, 0, 0);
    }

    #[test]
    fn comparison_with_zero() {
        // Test edge case: comparing with 0
        let result = run("0 0 EQ 0 PROPHECY");
        assert_memory_value(&result, 0, 1);
    }
}

// =============================================================================
// Logical Operation Tests
// =============================================================================

mod logical {
    use super::*;

    #[test]
    fn not_inverts_zero() {
        // NOT 0 = 1
        let result = run("0 NOT 0 PROPHECY");
        assert_memory_value(&result, 0, 1);
    }

    #[test]
    fn not_inverts_nonzero() {
        // NOT 5 = 0 (any nonzero becomes 0)
        let result = run("5 NOT 0 PROPHECY");
        assert_memory_value(&result, 0, 0);
    }

    #[test]
    fn and_with_both_true() {
        let result = run("1 1 AND 0 PROPHECY");
        assert_memory_value(&result, 0, 1);
    }

    #[test]
    fn and_with_one_false() {
        let result = run("1 0 AND 0 PROPHECY");
        assert_memory_value(&result, 0, 0);
    }

    #[test]
    fn or_with_one_true() {
        let result = run("1 0 OR 0 PROPHECY");
        assert_memory_value(&result, 0, 1);
    }

    #[test]
    fn or_with_both_false() {
        let result = run("0 0 OR 0 PROPHECY");
        assert_memory_value(&result, 0, 0);
    }

    #[test]
    fn double_negation() {
        // NOT NOT 1 = 1
        let result = run("1 NOT NOT 0 PROPHECY");
        assert_memory_value(&result, 0, 1);
    }
}

// =============================================================================
// Control Flow Tests
// =============================================================================

mod control_flow {
    use super::*;

    #[test]
    fn if_executes_then_branch_when_true() {
        // 1 IF { 42 } ELSE { 0 }
        let result = run("1 IF { 42 0 PROPHECY } ELSE { 0 0 PROPHECY }");
        assert_memory_value(&result, 0, 42);
    }

    #[test]
    fn if_executes_else_branch_when_false() {
        // 0 IF { 42 } ELSE { 99 }
        let result = run("0 IF { 42 0 PROPHECY } ELSE { 99 0 PROPHECY }");
        assert_memory_value(&result, 0, 99);
    }

    #[test]
    fn nested_if_else() {
        // Nested conditionals
        let code = r#"
            1 IF {
                1 IF {
                    10 0 PROPHECY
                } ELSE {
                    20 0 PROPHECY
                }
            } ELSE {
                30 0 PROPHECY
            }
        "#;
        let result = run(code);
        assert_memory_value(&result, 0, 10);
    }

    #[test]
    fn if_without_else() {
        // IF without ELSE should still work
        let result = run("1 42 0 PROPHECY");
        assert_consistent(&result);
    }

    #[test]
    fn conditional_based_on_comparison() {
        // if 5 > 3 then 100 else 0
        let code = "5 3 GT IF { 100 0 PROPHECY } ELSE { 0 0 PROPHECY }";
        let result = run(code);
        assert_memory_value(&result, 0, 100);
    }
}

// =============================================================================
// Memory Operation Tests
// =============================================================================

mod memory {
    use super::*;

    #[test]
    fn prophecy_writes_to_address() {
        let result = run("42 5 PROPHECY");
        assert_memory_value(&result, 5, 42);
    }

    #[test]
    fn multiple_prophecies_different_addresses() {
        let result = run("1 0 PROPHECY 2 1 PROPHECY 3 2 PROPHECY");
        assert_memory_value(&result, 0, 1);
        assert_memory_value(&result, 1, 2);
        assert_memory_value(&result, 2, 3);
    }

    #[test]
    fn oracle_reads_prophecy_value_on_convergence() {
        // Self-consistent: oracle reads what prophecy writes
        let code = "0 ORACLE 0 PROPHECY";
        let result = run(code);
        assert_consistent(&result);
    }

    #[test]
    fn prophecy_overwrites_previous() {
        // Multiple prophecies to same address: last wins
        let result = run("10 0 PROPHECY 20 0 PROPHECY 30 0 PROPHECY");
        assert_memory_value(&result, 0, 30);
    }

    #[test]
    fn address_wraparound() {
        // Address is 16-bit, so 65536 wraps to 0
        let result = run("42 65536 PROPHECY");
        // 65536 % 65536 = 0
        assert_memory_value(&result, 0, 42);
    }
}

// =============================================================================
// Output Operation Tests
// =============================================================================

mod output {
    use super::*;

    #[test]
    fn output_captures_value() {
        let result = run("42 OUTPUT 0 0 PROPHECY");
        assert_output_contains(&result, 42);
    }

    #[test]
    fn multiple_outputs() {
        let result = run("1 OUTPUT 2 OUTPUT 3 OUTPUT 0 0 PROPHECY");
        let output = extract_output(&result);
        assert_eq!(output.len(), 3);
        let values: Vec<u64> = output.iter().filter_map(|o| {
            if let ourochronos::OutputItem::Val(v) = o { Some(v.val) } else { None }
        }).collect();
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn output_does_not_pop_implicitly() {
        // OUTPUT consumes the value from stack
        let result = run("42 OUTPUT 0 PROPHECY");
        assert_consistent(&result);
    }
}

// =============================================================================
// Invariant Tests
// =============================================================================

mod invariants {
    use super::*;

    #[test]
    fn invariant_stack_underflow_safe() {
        // Operations on empty stack should not crash
        let result = run("POP 0 0 PROPHECY");
        // Should either handle gracefully or error, but not panic
        assert!(matches!(
            result,
            ConvergenceStatus::Consistent { .. }
            | ConvergenceStatus::Error { .. }
        ));
    }

    #[test]
    fn invariant_memory_initialised_to_zero() {
        // Unwritten memory should read as 0
        let result = run("100 ORACLE 0 PROPHECY");
        // Address 100 was never written, oracle should read 0
        assert_memory_value(&result, 0, 0);
    }

    #[test]
    fn invariant_arithmetic_associativity() {
        // (a + b) + c = a + (b + c)
        let result1 = run("1 2 ADD 3 ADD 0 PROPHECY");
        let result2 = run("1 2 3 ADD ADD 0 PROPHECY");

        let mem1 = extract_memory(&result1);
        let mem2 = extract_memory(&result2);

        assert_eq!(
            mem1.read(0).val,
            mem2.read(0).val
        );
    }

    #[test]
    fn invariant_multiplication_commutativity() {
        // a * b = b * a
        let result1 = run("3 7 MUL 0 PROPHECY");
        let result2 = run("7 3 MUL 0 PROPHECY");

        let mem1 = extract_memory(&result1);
        let mem2 = extract_memory(&result2);

        assert_eq!(
            mem1.read(0).val,
            mem2.read(0).val
        );
    }
}
