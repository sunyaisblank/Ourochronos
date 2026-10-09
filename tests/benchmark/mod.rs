//! Benchmark Tests for Ourochronos VM.
//!
//! This module compares the standard and pure execution facades. Both execute
//! validated bytecode, so these timings include admission and facade overhead.
//! Every measured run must complete with independently expected output.
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
fn benchmark_vm(
    name: &str,
    program: &Program,
    max_instructions: u64,
    expected_output: &[OutputItem],
) -> BenchmarkResult {
    let config = ExecutorConfig {
        max_instructions,
        immediate_output: false,
        input: Vec::new(),
        ..Default::default()
    };

    let start = Instant::now();
    let mut iterations = 0u32;
    let mut total_instructions = 0u64;

    // Run until we hit target duration or minimum iterations
    while iterations < MIN_ITERATIONS || start.elapsed().as_millis() < TARGET_DURATION_MS as u128 {
        let mut executor = Executor::with_config(config.clone());
        let anamnesis = Memory::new();
        let result = executor.run_epoch(program, &anamnesis);

        assert_eq!(result.status, EpochStatus::Finished, "VM failed for {name}");
        assert_eq!(
            result.output.as_slice(),
            expected_output,
            "VM output for {name}"
        );

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
    }
}

/// Benchmark fast VM execution.
fn benchmark_fast_vm(
    name: &str,
    program: &Program,
    max_instructions: u64,
    expected_output: &[OutputItem],
) -> BenchmarkResult {
    let start = Instant::now();
    let mut iterations = 0u32;
    let mut total_instructions = 0u64;
    assert!(
        is_program_pure(program),
        "FastVM benchmark must be pure: {name}"
    );

    // Run until we hit target duration or minimum iterations
    while iterations < MIN_ITERATIONS || start.elapsed().as_millis() < TARGET_DURATION_MS as u128 {
        let mut executor = FastExecutor::new(max_instructions);
        executor
            .execute_pure(program, &program.quotes)
            .unwrap_or_else(|error| panic!("FastVM failed for {name}: {error}"));
        assert_eq!(
            executor.status,
            EpochStatus::Finished,
            "FastVM failed for {name}"
        );
        assert_eq!(
            executor.output.as_slice(),
            expected_output,
            "FastVM output for {name}"
        );

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
    }
}

fn benchmark_pure(name: &str, code: &str, expected: &[u64]) {
    let program = parse(code);
    let output = numeric_output(expected);
    let vm_result = benchmark_vm(name, &program, 10_000_000, &output);
    let fast_result = benchmark_fast_vm(name, &program, 10_000_000, &output);
    vm_result.print();
    fast_result.print();
    compare_results(&vm_result, &fast_result);
}

fn numeric_output(words: &[u64]) -> Vec<OutputItem> {
    words
        .iter()
        .map(|&word| OutputItem::Val(Value::new(word)))
        .collect()
}

// =============================================================================
// Benchmark Programs
// =============================================================================

/// Fibonacci computation (pure, stack-intensive).
const FIBONACCI_N: &str = r#"
    0 0 1
    WHILE { 2 PICK 100000 LT } {
        SWAP OVER ADD
        ROT 1 ADD ROT ROT
    }
    OUTPUT OUTPUT OUTPUT
"#;

/// Factorial computation (pure, multiplication-heavy).
const FACTORIAL: &str = r#"
    1 1
    WHILE { DUP 20 LTE } {
        SWAP OVER MUL SWAP 1 ADD
    }
    POP OUTPUT
"#;

/// Tight arithmetic loop.
const ARITHMETIC_LOOP: &str = r#"
    0 0
    WHILE { DUP 10000 LT } {
        SWAP 3 MUL 2 ADD 7 MOD SWAP 1 ADD
    }
    OUTPUT OUTPUT
"#;

/// Stack manipulation stress test.
const STACK_STRESS: &str = r#"
    1 2 3 4 5
    0 WHILE { DUP 1000 LT } {
        SWAP OVER ROT POP SWAP OVER ROT POP
        1 ADD
    }
    POP DEPTH OUTPUT OUTPUT OUTPUT OUTPUT OUTPUT OUTPUT
"#;

/// Comparison operations.
const COMPARISON_LOOP: &str = r#"
    0 0
    WHILE { DUP 10000 LT } {
        DUP 5000 GT IF { SWAP 1 ADD SWAP }
        DUP 2500 LT IF { SWAP 1 ADD SWAP }
        1 ADD
    }
    OUTPUT OUTPUT
"#;

/// Bitwise operations.
const BITWISE_LOOP: &str = r#"
    1 0
    WHILE { DUP 10000 LT } {
        SWAP DUP 3 SHL XOR OVER XOR 65535 AND SWAP 1 ADD
    }
    OUTPUT OUTPUT
"#;

/// Nested loops.
const NESTED_LOOPS: &str = r#"
    0 0
    WHILE { DUP 100 LT } {
        0 WHILE { DUP 100 LT } {
            ROT 1 ADD ROT ROT
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
    0 ORACLE DUP OUTPUT 0 PROPHECY
"#;

// =============================================================================
// Benchmark Tests
// =============================================================================

#[test]
fn benchmark_fibonacci() {
    let (mut previous, mut current) = (0u64, 1u64);
    for _ in 0..100_000 {
        (previous, current) = (current, previous.wrapping_add(current));
    }
    benchmark_pure("fibonacci", FIBONACCI_N, &[current, previous, 100_000]);
}

#[test]
fn benchmark_factorial() {
    let expected = (1u64..=20).product();
    benchmark_pure("factorial", FACTORIAL, &[expected]);
}

#[test]
fn benchmark_arithmetic_loop() {
    let mut accumulator = 0u64;
    for _ in 0..10_000 {
        accumulator = (accumulator * 3 + 2) % 7;
    }
    benchmark_pure("arithmetic", ARITHMETIC_LOOP, &[10_000, accumulator]);
}

#[test]
fn benchmark_stack_operations() {
    // SWAP OVER ROT POP restores its two operands. Each iteration preserves
    // the five data words and advances only the counter.
    let words: Vec<u64> = (1..=5).collect();
    let expected: Vec<_> = std::iter::once(words.len() as u64)
        .chain(words.iter().rev().copied())
        .collect();
    benchmark_pure("stack", STACK_STRESS, &expected);
}

#[test]
fn benchmark_comparisons() {
    let mut matches = 0;
    for index in 0..10_000 {
        matches += u64::from(index > 5000) + u64::from(index < 2500);
    }
    benchmark_pure("comparison", COMPARISON_LOOP, &[10_000, matches]);
}

#[test]
fn benchmark_bitwise() {
    let mut accumulator = 1u64;
    for index in 0..10_000 {
        accumulator = (accumulator ^ (accumulator << 3) ^ index) & 0xffff;
    }
    benchmark_pure("bitwise", BITWISE_LOOP, &[10_000, accumulator]);
}

#[test]
fn benchmark_nested_loops() {
    let mut visits = 0;
    for _ in 0..100 {
        for _ in 0..100 {
            visits += 1;
        }
    }
    benchmark_pure("nested", NESTED_LOOPS, &[100, visits]);
}

#[test]
fn benchmark_temporal_overhead() {
    println!("\n=== Temporal Operations Benchmark ===");

    // Temporal programs cannot use FastVM
    let program = parse(TEMPORAL_SIMPLE);
    let max_instructions = 10_000_000;

    let vm_result = benchmark_vm("temporal_simple", &program, max_instructions, &[]);
    vm_result.print();

    assert!(!is_program_pure(&program));
    let error = FastExecutor::new(max_instructions)
        .execute_pure(&program, &program.quotes)
        .expect_err("FastVM must reject temporal operations");
    assert!(error.contains("full temporal/effect runtime"), "{error}");
    println!("  FastVM: Not applicable (temporal operations)");
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
        let result = TimeLoop::new(config.clone())
            .expect("valid configuration")
            .run(&program);
        match result {
            ConvergenceStatus::Consistent { epochs, output, .. } => {
                assert_eq!(epochs, 1);
                assert_eq!(
                    output,
                    vec![OutputItem::Val(Value::with_provenance(
                        0,
                        ourochronos::core::Provenance::single(0),
                    ))],
                );
            }
            status => panic!("consistent temporal benchmark failed: {status:?}"),
        }
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

/// Failed runs and changed observations must invalidate a timing campaign.
/// These are real program mutations, including gas exhaustion and a value
/// changed to a character; both facades must expose the disagreement.
#[test]
fn benchmark_checks_reject_failed_and_incorrect_runs() {
    let cases = [
        ("POP", 100, vec![]),
        ("WHILE { 1 } { NOP }", 20, vec![]),
        ("PARADOX", 100, vec![]),
        ("41 OUTPUT", 100, numeric_output(&[42])),
        ("42 EMIT", 100, numeric_output(&[42])),
        ("42 OUTPUT 42 OUTPUT", 100, numeric_output(&[42])),
        ("NOP", 100, numeric_output(&[42])),
    ];
    for (code, gas, expected) in cases {
        let program = parse(code);
        assert!(
            std::panic::catch_unwind(|| benchmark_vm(code, &program, gas, &expected)).is_err(),
            "VM benchmark accepted {code}",
        );
        if is_program_pure(&program) {
            assert!(
                std::panic::catch_unwind(|| benchmark_fast_vm(code, &program, gas, &expected))
                    .is_err(),
                "FastVM benchmark accepted {code}",
            );
        }
    }
}

/// Both facades must match independently expected typed output. They share
/// bytecode dispatch, so agreement between them alone is insufficient.
#[test]
fn invariant_fastvm_matches_vm() {
    let pure_programs = [
        ("10 20 ADD OUTPUT", numeric_output(&[30])),
        ("1 2 3 ROT OUTPUT OUTPUT OUTPUT", numeric_output(&[1, 3, 2])),
        ("5 DUP MUL OUTPUT", numeric_output(&[25])),
        ("100 50 SUB 25 ADD OUTPUT", numeric_output(&[75])),
        ("7 3 MOD OUTPUT", numeric_output(&[1])),
        (
            "72 EMIT 73 EMIT",
            vec![OutputItem::Char(b'H'), OutputItem::Char(b'I')],
        ),
        (
            "1 IF { 42 OUTPUT } ELSE { 7 OUTPUT }",
            numeric_output(&[42]),
        ),
        (
            "3 WHILE { DUP 0 GT } { DUP OUTPUT 1 SUB } POP",
            numeric_output(&[3, 2, 1]),
        ),
        ("INPUT OUTPUT INPUT OUTPUT", numeric_output(&[11, 22])),
    ];
    let scripted_input = vec![11u64, 22u64];

    for (code, expected) in &pure_programs {
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

        assert_eq!(&vm_result.output, expected, "VM output for: {code}");
        assert_eq!(
            fast_exec.status,
            EpochStatus::Finished,
            "FastVM status for: {code}"
        );
        assert_eq!(&fast_exec.output, expected, "FastVM output for: {code}");
    }
}

/// Rejection behavior is part of the optimized-runtime identity contract. In
/// particular, neither facade may turn a statically invalid stack operation
/// into a successful no-op.
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
