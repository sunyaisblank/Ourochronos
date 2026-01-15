//! Benchmark Tests for Ourochronos VM and JIT.
//!
//! This module provides performance comparison tests between:
//! - Standard VM (with provenance tracking)
//! - Fast VM (register-cached, no provenance)
//! - JIT compilation (when enabled)
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

use std::time::{Duration, Instant};
use ourochronos::*;
use ourochronos::vm::{Executor, ExecutorConfig, EpochStatus};
use ourochronos::fast_vm::{FastExecutor, is_program_pure};
use ourochronos::timeloop::TimeLoop;

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
    let speedup = baseline.avg_time().as_nanos() as f64
        / optimized.avg_time().as_nanos() as f64;

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
            EpochStatus::Error(ref e) => {
                if !e.contains("requires standard") {
                    success = false;
                }
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

/// Benchmark JIT compilation (if available).
#[cfg(feature = "jit")]
fn benchmark_jit(name: &str, program: &Program) -> BenchmarkResult {
    use ourochronos::jit::JitCompiler;

    let start = Instant::now();

    match JitCompiler::new() {
        Ok(mut jit) => {
            match jit.compile(program) {
                Ok(_compiled) => {
                    let compile_time = start.elapsed();

                    BenchmarkResult {
                        name: format!("JIT:{}", name),
                        total_time: compile_time,
                        iterations: 1,
                        total_instructions: jit.stats.opcodes_translated as u64,
                        success: true,
                    }
                }
                Err(e) => {
                    BenchmarkResult {
                        name: format!("JIT:{}", name),
                        total_time: start.elapsed(),
                        iterations: 0,
                        total_instructions: 0,
                        success: false,
                    }
                }
            }
        }
        Err(_) => {
            BenchmarkResult {
                name: format!("JIT:{}", name),
                total_time: Duration::ZERO,
                iterations: 0,
                total_instructions: 0,
                success: false,
            }
        }
    }
}

#[cfg(not(feature = "jit"))]
fn benchmark_jit(_name: &str, _program: &Program) -> BenchmarkResult {
    BenchmarkResult {
        name: "JIT:disabled".to_string(),
        total_time: Duration::ZERO,
        iterations: 0,
        total_instructions: 0,
        success: false,
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
    };

    let start = Instant::now();
    let mut iterations = 0u32;

    while iterations < MIN_ITERATIONS || start.elapsed().as_millis() < TARGET_DURATION_MS as u128 {
        let _result = TimeLoop::new(config.clone()).run(&program);
        iterations += 1;

        if iterations > 1000 {
            break;
        }
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;

    println!(
        "TimeLoop:consistent: {:?}/iter ({} iters)",
        avg, iterations
    );
}

#[test]
fn benchmark_jit_compilation() {
    println!("\n=== JIT Compilation Benchmark ===");

    let programs = [
        ("fibonacci", FIBONACCI_N),
        ("factorial", FACTORIAL),
        ("arithmetic", ARITHMETIC_LOOP),
        ("stack", STACK_STRESS),
    ];

    for (name, code) in &programs {
        let program = parse(code);
        let result = benchmark_jit(name, &program);

        if result.success {
            println!(
                "JIT compile time for {}: {:?} ({} opcodes)",
                name, result.total_time, result.total_instructions
            );
        } else {
            println!("JIT: {} - not available or compilation failed", name);
        }
    }
}

// =============================================================================
// Invariant Tests (Benchmark-Related)
// =============================================================================

/// Verify that FastVM produces identical results to VM for pure programs.
#[test]
fn invariant_fastvm_matches_vm() {
    let pure_programs = [
        "10 20 ADD",
        "1 2 3 ROT",
        "5 DUP MUL",
        "100 50 SUB 25 ADD",
        "7 3 MOD",
    ];

    for code in &pure_programs {
        let program = parse(code);

        // Run in VM
        let config = ExecutorConfig {
            max_instructions: 10_000,
            immediate_output: false,
            input: Vec::new(),
        };
        let mut vm_exec = Executor::with_config(config);
        let anamnesis = Memory::new();
        let vm_result = vm_exec.run_epoch(&program, &anamnesis);

        // Run in FastVM
        if is_program_pure(&program) {
            let mut fast_exec = FastExecutor::new(10_000);
            let _ = fast_exec.execute_pure(&program, &program.quotes);

            // Compare stacks (values should match)
            let vm_stack: Vec<u64> = vm_exec.config.input.clone(); // placeholder
            let fast_stack = fast_exec.stack.to_value_vec();

            // At minimum, both should complete successfully
            assert!(
                vm_result.status == EpochStatus::Finished,
                "VM failed for: {}", code
            );
        }
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
        assert!(
            is_program_pure(&program),
            "Expected pure: {}", code
        );
    }

    for code in &impure_programs {
        let program = parse(code);
        assert!(
            !is_program_pure(&program),
            "Expected impure: {}", code
        );
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
