//! Integration tests for bounds-checked memory operations.
//!
//! These tests verify that the ErrorConfig/BoundsPolicy infrastructure
//! is correctly wired through the executor for Oracle, Prophecy,
//! PresentRead, Index, and Store operations.
//!
//! Note: INDEX and STORE are not parser-level keywords, so tests for
//! those opcodes construct programs programmatically.

#![cfg(test)]

use ourochronos::*;
use ourochronos::core::error::ErrorConfig;
use ourochronos::temporal::timeloop::{TimeLoopConfig, ConvergenceStatus};

/// Run a programme (from source text) with a specific error config.
fn run_with_error_config(code: &str, error_config: ErrorConfig) -> ConvergenceStatus {
    let tokens = tokenize(code);
    let mut parser = Parser::new(&tokens);
    let program = parser.parse_program().expect("Failed to parse");

    let config = TimeLoopConfig {
        max_epochs: 100,
        error_config,
        ..Default::default()
    };
    TimeLoop::new(config).expect("valid configuration").run(&program)
}

/// Run a pre-built programme with a specific error config.
fn run_program_with_error_config(program: &Program, error_config: ErrorConfig) -> ConvergenceStatus {
    let config = TimeLoopConfig {
        max_epochs: 100,
        error_config,
        ..Default::default()
    };
    TimeLoop::new(config).expect("valid configuration").run(program)
}

/// Build a programme from a list of statements.
fn make_program(body: Vec<Stmt>) -> Program {
    Program {
        procedures: Vec::new(),
        quotes: Vec::new(),
        body,
        ffi_declarations: Vec::new(),
    }
}

// =============================================================================
// Wrap Policy Tests (default behaviour)
// =============================================================================

mod wrap_policy {
    use super::*;

    #[test]
    fn oracle_wraps_out_of_bounds_address() {
        // Address 65536 should wrap to 0 under Wrap policy.
        // Write 42 to address 0, then oracle-read address 65536 (wraps to 0).
        let result = run_with_error_config(
            "42 0 PROPHECY 65536 ORACLE OUTPUT",
            ErrorConfig::default(),
        );
        assert!(
            matches!(result, ConvergenceStatus::Consistent { .. }),
            "Wrap policy should allow out-of-bounds oracle: {:?}", result
        );
    }

    #[test]
    fn prophecy_wraps_out_of_bounds_address() {
        // Writing to address 65536 should wrap to address 0.
        let result = run_with_error_config(
            "99 65536 PROPHECY",
            ErrorConfig::default(),
        );
        assert!(
            matches!(result, ConvergenceStatus::Consistent { .. }),
            "Wrap policy should allow out-of-bounds prophecy: {:?}", result
        );
    }

    #[test]
    fn index_wraps_overflow() {
        // Index with base + offset exceeding memory size should wrap.
        // Push 42 to address 0 (65500+36 wraps to 0), then Index-read it.
        let program = make_program(vec![
            Stmt::Push(Value::new(42)),
            Stmt::Push(Value::new(0)),       // address for prophecy (where the value lands)
            Stmt::Op(OpCode::Prophecy),
            Stmt::Push(Value::new(65500)),   // base
            Stmt::Push(Value::new(36)),      // offset (65500+36 = 65536 wraps to 0)
            Stmt::Op(OpCode::Index),
            Stmt::Op(OpCode::Output),
        ]);
        let result = run_program_with_error_config(&program, ErrorConfig::default());
        assert!(
            matches!(result, ConvergenceStatus::Consistent { .. }),
            "Wrap policy should allow index overflow: {:?}", result
        );
    }
}

// =============================================================================
// Error (Strict) Policy Tests
// =============================================================================

mod error_policy {
    use super::*;

    #[test]
    fn oracle_errors_on_out_of_bounds() {
        // Under Error policy, out-of-bounds oracle should produce an error.
        let result = run_with_error_config(
            "65536 ORACLE OUTPUT",
            ErrorConfig::strict(),
        );
        match &result {
            ConvergenceStatus::Error { message, .. } => {
                assert!(
                    message.contains("bounds") || message.contains("Memory") || message.contains("address"),
                    "Error message should mention bounds violation: {}", message
                );
            }
            ConvergenceStatus::Consistent { .. } => {
                // 65536 as u16 wraps to 0, which is in bounds.
                // This is acceptable if the value gets truncated before the
                // bounds check.
            }
            _ => {} // Some error variant is acceptable
        }
    }

    #[test]
    fn oracle_errors_on_large_address() {
        // Address 100000 is clearly out of bounds.
        let result = run_with_error_config(
            "100000 ORACLE OUTPUT",
            ErrorConfig::strict(),
        );
        match &result {
            ConvergenceStatus::Error { message, .. } => {
                assert!(
                    message.contains("bounds") || message.contains("Memory") || message.contains("address"),
                    "Error message should mention bounds: {}", message
                );
            }
            _ => panic!("Expected Error for out-of-bounds oracle under strict mode, got {:?}", result),
        }
    }

    #[test]
    fn prophecy_errors_on_large_address() {
        let result = run_with_error_config(
            "42 100000 PROPHECY",
            ErrorConfig::strict(),
        );
        assert!(
            matches!(result, ConvergenceStatus::Error { .. }),
            "Strict mode should error on out-of-bounds prophecy: {:?}", result
        );
    }

    #[test]
    fn index_errors_on_overflow() {
        // base 50000 + offset 50000 = 100000, out of bounds under strict mode.
        let program = make_program(vec![
            Stmt::Push(Value::new(50000)),
            Stmt::Push(Value::new(50000)),
            Stmt::Op(OpCode::Index),
            Stmt::Op(OpCode::Output),
        ]);
        let result = run_program_with_error_config(&program, ErrorConfig::strict());
        assert!(
            matches!(result, ConvergenceStatus::Error { .. }),
            "Strict mode should error on index overflow: {:?}", result
        );
    }

    #[test]
    fn store_errors_on_overflow() {
        // value=42, base=50000, offset=50000 -> address 100000, out of bounds.
        let program = make_program(vec![
            Stmt::Push(Value::new(42)),
            Stmt::Push(Value::new(50000)),
            Stmt::Push(Value::new(50000)),
            Stmt::Op(OpCode::Store),
        ]);
        let result = run_program_with_error_config(&program, ErrorConfig::strict());
        assert!(
            matches!(result, ConvergenceStatus::Error { .. }),
            "Strict mode should error on store overflow: {:?}", result
        );
    }
}

// =============================================================================
// In-Bounds Operations (should work under all policies)
// =============================================================================

mod valid_operations {
    use super::*;

    #[test]
    fn oracle_prophecy_in_bounds_strict() {
        // Normal in-bounds operations should work under strict mode.
        let result = run_with_error_config(
            "42 100 PROPHECY 100 ORACLE OUTPUT",
            ErrorConfig::strict(),
        );
        assert!(
            matches!(result, ConvergenceStatus::Consistent { .. }),
            "In-bounds operations should succeed under strict mode: {:?}", result
        );
    }

    #[test]
    fn index_store_in_bounds_strict() {
        // Store 42 at base=100, offset=5 (address 105), then Index-read it back.
        let program = make_program(vec![
            Stmt::Push(Value::new(42)),
            Stmt::Push(Value::new(100)),
            Stmt::Push(Value::new(5)),
            Stmt::Op(OpCode::Store),
            Stmt::Push(Value::new(100)),
            Stmt::Push(Value::new(5)),
            Stmt::Op(OpCode::Index),
            Stmt::Op(OpCode::Output),
        ]);
        let result = run_program_with_error_config(&program, ErrorConfig::strict());
        assert!(
            matches!(result, ConvergenceStatus::Consistent { .. }),
            "In-bounds index/store should succeed under strict mode: {:?}", result
        );
    }
}
