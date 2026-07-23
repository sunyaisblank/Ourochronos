//! Benchmark Tests for Ourochronos VM.
//!
//! This module provides performance comparison tests between:
//! - Standard VM (with provenance tracking)
//! - Fast VM (register-cached, no provenance)
//!
//! ## Benchmark Categories
//!
//! - **Arithmetic**: Pure computational workloads
//! - **Stack Operations**: Stack manipulation performance
//! - **Control Flow**: Loop and conditional overhead
//! - **Temporal**: ORACLE/PROPHECY operation cost
//!
//! ## Running Benchmarks
//!
//! ```bash
//! # Run all benchmark tests
//! cargo test benchmark --release -- --nocapture
//!
//! # Run specific benchmark
//! cargo test benchmark::fibonacci --release -- --nocapture
//! ```

use ourochronos::temporal::timeloop::TimeLoop;
use ourochronos::vm::fast_vm::{is_program_pure, FastExecutor};
use ourochronos::vm::{EpochStatus, Executor, ExecutorConfig};
use ourochronos::*;
use std::time::{Duration, Instant};

/// Minimum iterations for stable timing.
const MIN_ITERATIONS: u32 = 10;
/// Target benchmark duration in milliseconds.
const TARGET_DURATION_MS: u64 = 100;

// =============================================================================
// Benchmark Infrastructure
// =============================================================================

/// Result of a single benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Name of the benchmark.
    pub name: String,
    /// Total time for all iterations.
    pub total_time: Duration,
    /// Number of iterations.
    pub iterations: u32,
    /// Total instructions executed.
    pub total_instructions: u64,
    /// Whether execution succeeded.
    pub success: bool,
}

impl BenchmarkResult {
    /// Average time per iteration.
    pub fn avg_time(&self) -> Duration {
        self.total_time / self.iterations
    }

    /// Instructions per second.
    pub fn instructions_per_second(&self) -> f64 {
        let secs = self.total_time.as_secs_f64();
        if secs > 0.0 {
            self.total_instructions as f64 / secs
        } else {
            0.0
        }
    }

    /// Print benchmark result.
    pub fn print(&self) {
        println!(
            "{}: {:?}/iter ({} iters, {:.2}M inst/sec)",
            self.name,
            self.avg_time(),
            self.iterations,
            self.instructions_per_second() / 1_000_000.0
        );
    }
}

/// Compare two benchmark results.
pub fn compare_results(baseline: &BenchmarkResult, optimized: &BenchmarkResult) {
    let speedup = baseline.avg_time().as_nanos() as f64 / optimized.avg_time().as_nanos() as f64;

    println!(
        "  {} vs {}: {:.2}x speedup",
        baseline.name, optimized.name, speedup
    );
}

/// Parse a program from source code.
fn parse(code: &str) -> Program {
    let tokens = tokenize(code);
    let mut parser = Parser::new(&tokens);
    parser.parse_program().expect("Failed to parse program")
}

// =============================================================================
// VM Benchmark Functions
// =============================================================================

/// Benchmark standard VM execution.
fn benchmark_vm(name: &str, program: &Program, max_instructions: u64) -> BenchmarkResult {
    let config = ExecutorConfig {
        max_instructions,
        immediate_output: false,
        input: Vec::new(),
        ..Default::default()
    };

    let start = Instant::now();
    let mut iterations = 0u32;
    let mut total_instructions = 0u64;
    let mut success = true;

    // Run until we hit target duration or minimum iterations
    while iterations < MIN_ITERATIONS || start.elapsed().as_millis() < TARGET_DURATION_MS as u128 {
        let mut executor = Executor::with_config(config.clone());
        let anamnesis = Memory::new();
        let result = executor.run_epoch(program, &anamnesis);

        match result.status {
            EpochStatus::Finished => {}
            EpochStatus::Error(ref e) if !e.contains("requires standard") => {
                success = false;
            }
            _ => {}
        }

        total_instructions += result.instructions_executed;
        iterations += 1;

        // Safety limit
        if iterations > 10000 {
            break;
        }
    }

    BenchmarkResult {
        name: format!("VM:{}", name),
        total_time: start.elapsed(),
        iterations,
        total_instructions,
        success,
    }
}

/// Benchmark fast VM execution.
fn benchmark_fast_vm(name: &str, program: &Program, max_instructions: u64) -> BenchmarkResult {
    let start = Instant::now();
    let mut iterations = 0u32;
    let mut total_instructions = 0u64;
    let mut success = true;

    // Check if program is pure
    if !is_program_pure(program) {
        return BenchmarkResult {
            name: format!("FastVM:{}", name),
            total_time: Duration::ZERO,
            iterations: 0,
            total_instructions: 0,
            success: false,
        };
    }

    // Run until we hit target duration or minimum iterations
    while iterations < MIN_ITERATIONS || start.elapsed().as_millis() < TARGET_DURATION_MS as u128 {
        let mut executor = FastExecutor::new(max_instructions);
        match executor.execute_pure(program, &program.quotes) {
            Ok(()) => {}
            Err(_) => {
                success = false;
            }
        }

        total_instructions += executor.instructions_executed;
        iterations += 1;

        // Safety limit
        if iterations > 10000 {
            break;
        }
    }

    BenchmarkResult {
        name: format!("FastVM:{}", name),
        total_time: start.elapsed(),
        iterations,
        total_instructions,
        success,
    }
}

// =============================================================================
// Benchmark Programs
// =============================================================================

/// Fibonacci computation (pure, stack-intensive).
const FIBONACCI_N: &str = r#"
    1 1
    0 WHILE { DUP 100000 LT } {
        OVER ADD
    }
    OUTPUT
"#;

/// Factorial computation (pure, multiplication-heavy).
const FACTORIAL: &str = r#"
    1 1
    0 WHILE { DUP 20 LT } {
        SWAP OVER MUL SWAP 1 ADD
    }
    POP OUTPUT
"#;

/// Tight arithmetic loop.
const ARITHMETIC_LOOP: &str = r#"
    0
    0 WHILE { DUP 10000 LT } {
        1 ADD DUP 3 MUL 2 ADD 7 MOD SWAP 1 ADD
    }
    POP OUTPUT
"#;

/// Stack manipulation stress test.
const STACK_STRESS: &str = r#"
    1 2 3 4 5
    0 WHILE { DUP 1000 LT } {
        SWAP OVER ROT SWAP OVER ROT
        1 ADD
    }
    DEPTH OUTPUT
"#;

/// Comparison operations.
const COMPARISON_LOOP: &str = r#"
    0 1
    0 WHILE { DUP 10000 LT } {
        OVER 5000 GT IF { OVER 1 ADD SWAP POP SWAP }
        OVER 2500 LT IF { OVER 1 ADD SWAP POP SWAP }
        1 ADD
    }
    OUTPUT OUTPUT
"#;

/// Bitwise operations.
const BITWISE_LOOP: &str = r#"
    1
    0 WHILE { DUP 10000 LT } {
        OVER 3 SHL OVER XOR OVER AND SWAP 1 ADD
    }
    OUTPUT OUTPUT
"#;

/// Nested loops.
const NESTED_LOOPS: &str = r#"
    0
    0 WHILE { DUP 100 LT } {
        0 WHILE { DUP 100 LT } {
            OVER 1 ADD SWAP POP SWAP
            1 ADD
        }
        POP 1 ADD
    }
    OUTPUT OUTPUT
"#;

/// Simple temporal program (tests temporal overhead).
const TEMPORAL_SIMPLE: &str = r#"
    0 ORACLE 1 ADD 0 PROPHECY
"#;

/// Self-consistent temporal (converges quickly).
const TEMPORAL_CONSISTENT: &str = r#"
    0 ORACLE DUP 0 PROPHECY
"#;

// =============================================================================
// Benchmark Tests
// =============================================================================

#[test]
fn benchmark_fibonacci() {
    println!("\n=== Fibonacci Benchmark ===");

    let program = parse(FIBONACCI_N);
    let max_instructions = 10_000_000;

    let vm_result = benchmark_vm("fibonacci", &program, max_instructions);
    vm_result.print();

    let fast_result = benchmark_fast_vm("fibonacci", &program, max_instructions);
    if fast_result.success {
        fast_result.print();
        compare_results(&vm_result, &fast_result);
    }
}

#[test]
fn benchmark_factorial() {
    println!("\n=== Factorial Benchmark ===");

    let program = parse(FACTORIAL);
    let max_instructions = 10_000_000;

    let vm_result = benchmark_vm("factorial", &program, max_instructions);
    vm_result.print();

    let fast_result = benchmark_fast_vm("factorial", &program, max_instructions);
    if fast_result.success {
        fast_result.print();
        compare_results(&vm_result, &fast_result);
    }
}

#[test]
fn benchmark_arithmetic_loop() {
    println!("\n=== Arithmetic Loop Benchmark ===");

    let program = parse(ARITHMETIC_LOOP);
    let max_instructions = 10_000_000;

    let vm_result = benchmark_vm("arithmetic", &program, max_instructions);
    vm_result.print();

    let fast_result = benchmark_fast_vm("arithmetic", &program, max_instructions);
    if fast_result.success {
        fast_result.print();
        compare_results(&vm_result, &fast_result);
    }
}

#[test]
fn benchmark_stack_operations() {
    println!("\n=== Stack Operations Benchmark ===");

    let program = parse(STACK_STRESS);
    let max_instructions = 10_000_000;

    let vm_result = benchmark_vm("stack", &program, max_instructions);
    vm_result.print();

    let fast_result = benchmark_fast_vm("stack", &program, max_instructions);
    if fast_result.success {
        fast_result.print();
        compare_results(&vm_result, &fast_result);
    }
}

#[test]
fn benchmark_comparisons() {
    println!("\n=== Comparison Operations Benchmark ===");

    let program = parse(COMPARISON_LOOP);
    let max_instructions = 10_000_000;

    let vm_result = benchmark_vm("comparison", &program, max_instructions);
    vm_result.print();

    let fast_result = benchmark_fast_vm("comparison", &program, max_instructions);
    if fast_result.success {
        fast_result.print();
        compare_results(&vm_result, &fast_result);
    }
}

#[test]
fn benchmark_bitwise() {
    println!("\n=== Bitwise Operations Benchmark ===");

    let program = parse(BITWISE_LOOP);
    let max_instructions = 10_000_000;

    let vm_result = benchmark_vm("bitwise", &program, max_instructions);
    vm_result.print();

    let fast_result = benchmark_fast_vm("bitwise", &program, max_instructions);
    if fast_result.success {
        fast_result.print();
        compare_results(&vm_result, &fast_result);
    }
}

#[test]
fn benchmark_nested_loops() {
    println!("\n=== Nested Loops Benchmark ===");

    let program = parse(NESTED_LOOPS);
    let max_instructions = 10_000_000;

    let vm_result = benchmark_vm("nested", &program, max_instructions);
    vm_result.print();

    let fast_result = benchmark_fast_vm("nested", &program, max_instructions);
    if fast_result.success {
        fast_result.print();
        compare_results(&vm_result, &fast_result);
    }
}

#[test]
fn benchmark_temporal_overhead() {
    println!("\n=== Temporal Operations Benchmark ===");

    // Temporal programs cannot use FastVM
    let program = parse(TEMPORAL_SIMPLE);
    let max_instructions = 10_000_000;

    let vm_result = benchmark_vm("temporal_simple", &program, max_instructions);
    vm_result.print();

    // This should fail (temporal ops not supported in fast VM)
    let fast_result = benchmark_fast_vm("temporal_simple", &program, max_instructions);
    if !fast_result.success {
        println!("  FastVM: Not applicable (temporal operations)");
    }
}

#[test]
fn benchmark_timeloop_convergence() {
    println!("\n=== TimeLoop Convergence Benchmark ===");

    let program = parse(TEMPORAL_CONSISTENT);

    let config = Config {
        max_epochs: 100,
        mode: ExecutionMode::Standard,
        seed: 0,
        verbose: false,
        frozen_inputs: Vec::new(),
        max_instructions: 10_000_000,
        ..Default::default()
    };

    let start = Instant::now();
    let mut iterations = 0u32;

    while iterations < MIN_ITERATIONS || start.elapsed().as_millis() < TARGET_DURATION_MS as u128 {
        let _result = TimeLoop::new(config.clone())
            .expect("valid configuration")
            .run(&program);
        iterations += 1;

        if iterations > 1000 {
            break;
        }
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;

    println!("TimeLoop:consistent: {:?}/iter ({} iters)", avg, iterations);
}

// =============================================================================
// Invariant Tests (Benchmark-Related)
// =============================================================================

/// Verify that FastVM is observably indistinguishable from the VM for pure
/// programmes: identical output buffers and successful completion. This is
/// the differential gate behind fast_vm's semantic-preservation claim.
#[test]
fn invariant_fastvm_matches_vm() {
    let pure_programs = [
        "10 20 ADD OUTPUT",
        "1 2 3 ROT OUTPUT OUTPUT OUTPUT",
        "5 DUP MUL OUTPUT",
        "100 50 SUB 25 ADD OUTPUT",
        "7 3 MOD OUTPUT",
        "72 EMIT 73 EMIT",
        "1 IF { 42 OUTPUT } ELSE { 7 OUTPUT }",
        "3 WHILE { DUP 0 GT } { DUP OUTPUT 1 SUB } POP",
        "INPUT OUTPUT INPUT OUTPUT",
    ];
    let scripted_input = vec![11u64, 22u64];

    for code in &pure_programs {
        let program = parse(code);
        assert!(is_program_pure(&program), "expected pure: {}", code);

        let config = ExecutorConfig {
            max_instructions: 10_000,
            immediate_output: false,
            input: scripted_input.clone(),
            ..Default::default()
        };
        let mut vm_exec = Executor::with_config(config);
        let anamnesis = Memory::new();
        let vm_result = vm_exec.run_epoch(&program, &anamnesis);
        assert_eq!(
            vm_result.status,
            EpochStatus::Finished,
            "VM failed for: {}",
            code
        );

        let mut fast_exec = FastExecutor::new(10_000).with_input(scripted_input.clone());
        fast_exec
            .execute_pure(&program, &program.quotes)
            .unwrap_or_else(|e| panic!("FastVM failed for {}: {}", code, e));

        let vm_out: Vec<String> = vm_result
            .output
            .iter()
            .map(crate::common::render_output_item)
            .collect();
        let fast_out: Vec<String> = fast_exec
            .output
            .iter()
            .map(crate::common::render_output_item)
            .collect();
        assert_eq!(vm_out, fast_out, "output diverged for: {}", code);
    }
}

/// Rejection behavior is part of the optimized-runtime identity contract. In
/// particular, register caching must not turn a statically invalid stack
/// operation into a no-op when the cached registers are empty.
#[test]
fn invariant_fastvm_and_vm_reject_stack_underflow() {
    for code in ["SWAP", "1 SWAP", "DUP", "1 OVER", "1 2 ROT"] {
        let program = parse(code);
        assert!(is_program_pure(&program), "expected pure: {}", code);

        let mut vm_exec = Executor::with_config(ExecutorConfig::default());
        let vm_result = vm_exec.run_epoch(&program, &Memory::new());
        let vm_error = match vm_result.status {
            EpochStatus::Error(message) => message,
            status => panic!(
                "reference VM unexpectedly returned {:?} for {}",
                status, code
            ),
        };

        let mut fast_exec = FastExecutor::new(10_000);
        let fast_error = fast_exec
            .execute_pure(&program, &program.quotes)
            .expect_err("FastVM must report the same underflow");
        assert!(!vm_error.is_empty(), "{}: missing VM rejection", code);
        assert!(!fast_error.is_empty(), "{}: missing FastVM rejection", code);
    }
}

/// Verify that purity analysis is sound (pure programs don't use temporal ops).
#[test]
fn invariant_purity_analysis_sound() {
    let pure_programs = [
        "10 20 ADD OUTPUT",
        "1 2 3 ROT DEPTH OUTPUT",
        "100 0 WHILE { DUP 0 GT } { 1 SUB } OUTPUT",
    ];

    let impure_programs = [
        "0 ORACLE OUTPUT",
        "42 0 PROPHECY",
        "PARADOX",
        "0 PRESENT OUTPUT",
    ];

    for code in &pure_programs {
        let program = parse(code);
        assert!(is_program_pure(&program), "Expected pure: {}", code);
    }

    for code in &impure_programs {
        let program = parse(code);
        assert!(!is_program_pure(&program), "Expected impure: {}", code);
    }
}

/// Verify that temporal programs fall back to VM correctly.
#[test]
fn invariant_temporal_fallback() {
    let code = "0 ORACLE 1 ADD 0 PROPHECY";
    let program = parse(code);

    // Should be impure
    assert!(!is_program_pure(&program));

    // FastVM should fail gracefully
    let mut fast_exec = FastExecutor::new(10_000);
    let result = fast_exec.execute_pure(&program, &program.quotes);

    assert!(result.is_err(), "FastVM should reject temporal operations");
}
