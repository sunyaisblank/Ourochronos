//! Executable conformance for the independently defined numeric core in oracle/.
//!
//! Compilation, linking, package loading, and solver decoding belong to the
//! system under test. Expected stores/stacks/observations come solely from the
//! independent text parser and continuation machine. Numeric snapshots omit
//! provenance deliberately; this target is not a proof of the complete language.

mod oracle;

use oracle::{Bounds, Environment, Fault, Observation, Snapshot, Stop};
use ourochronos::vm::fast_vm::{is_program_pure, FastExecutor};
use ourochronos::vm::{EpochStatus, Executor, ExecutorConfig, VmState};
use ourochronos::{
    admit_program, link, AdmissionConfig, AdmissionPhase, BoundsPolicy, BytecodeExecution,
    BytecodeProgram, BytecodeVm, BytecodeVmConfig, BytecodeVmError, BytecodeVmStatus,
    GlobalFixedPointSolver, GlobalSolveConfig, GlobalSolveResult, ModuleGraph, ObjectModule,
    OutputItem, PackageError, PackageManifest, PackageWitness, PagedMemory, PortablePackage, Value,
};
use proptest::prelude::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

const BYTECODE_GAS: u64 = 100_000;

fn bounds(policy: Bounds) -> BoundsPolicy {
    match policy {
        Bounds::Error => BoundsPolicy::Error,
        Bounds::Wrap => BoundsPolicy::Wrap,
        Bounds::Clamp => BoundsPolicy::Clamp,
    }
}

fn config(environment: &Environment, gas: u64) -> BytecodeVmConfig {
    BytecodeVmConfig {
        max_instructions: gas,
        max_stack_depth: environment.stack,
        max_call_depth: environment.calls,
        max_output_items: environment.output,
        memory_bounds: bounds(environment.bounds),
        input: environment.input.clone(),
        random_input: environment.random.clone(),
        // All live observations and effects retain their denied defaults.
        ..BytecodeVmConfig::default()
    }
}

fn anamnesis(environment: &Environment) -> PagedMemory {
    let mut memory = PagedMemory::with_size(environment.anamnesis.len()).unwrap();
    for (address, &word) in environment.anamnesis.iter().enumerate() {
        memory.write(address as u64, Value::new(word)).unwrap();
    }
    memory
}

fn words(memory: &PagedMemory) -> Vec<u64> {
    (0..memory.len())
        .map(|address| {
            memory
                .read_checked(address as u64, BoundsPolicy::Error, Default::default())
                .unwrap()
                .val
        })
        .collect()
}

fn observations(output: &[OutputItem]) -> Vec<Observation> {
    output
        .iter()
        .map(|item| match item {
            OutputItem::Val(word) => Observation::Number(word.val),
            OutputItem::Char(byte) => Observation::Byte(*byte),
        })
        .collect()
}

fn snapshot(execution: &BytecodeExecution) -> Snapshot {
    assert!(
        execution.effects.is_empty(),
        "core execution staged an effect"
    );
    Snapshot {
        stack: execution.stack.iter().map(|word| word.val).collect(),
        present: words(&execution.present),
        output: observations(&execution.output),
        consumed: execution.inputs_consumed.clone(),
        stop: match execution.status {
            BytecodeVmStatus::Finished => Stop::Finished,
            BytecodeVmStatus::Halted => Stop::Halted,
            BytecodeVmStatus::Paradox => Stop::Paradox,
        },
    }
}

fn bytecode(source: &str, width: usize) -> BytecodeProgram {
    let parsed = ourochronos::parser::parse(source)
        .unwrap_or_else(|error| panic!("production parse rejected {source:?}: {error}"));
    admit_program(
        &parsed,
        AdmissionConfig {
            memory_cells: width,
        },
    )
    .unwrap_or_else(|error| panic!("production admission rejected {source:?}: {error}"))
    .into_program()
}

fn expected(source: &str, environment: &Environment) -> Snapshot {
    let program = oracle::parse(source)
        .unwrap_or_else(|error| panic!("independent core parse rejected {source:?}: {error}"));
    let epoch = oracle::evaluate(&program, environment).unwrap_or_else(|error| {
        panic!("independent core execution rejected {source:?}: {error:?}")
    });
    assert!(epoch.steps <= environment.steps);
    epoch.snapshot
}

fn run(program: &BytecodeProgram, environment: &Environment) -> Snapshot {
    let memory = anamnesis(environment);
    let execution = BytecodeVm::with_config(config(environment, BYTECODE_GAS))
        .run(program, &memory)
        .unwrap_or_else(|error| panic!("production runtime rejected an expected success: {error}"));
    assert_eq!(words(&memory), environment.anamnesis, "anamnesis changed");
    snapshot(&execution)
}

fn compare(source: &str, environment: &Environment) -> Snapshot {
    let expected = expected(source, environment);
    let program = bytecode(source, environment.anamnesis.len());
    assert_eq!(run(&program, environment), expected, "source: {source}");
    let prepared = ourochronos::PreparedBytecode::new(program).unwrap();
    let execution = BytecodeVm::with_config(config(environment, BYTECODE_GAS))
        .run_prepared(&prepared, &anamnesis(environment))
        .unwrap();
    assert_eq!(snapshot(&execution), expected, "prepared source: {source}");
    expected
}

/// Compatibility APIs expose fewer distinctions: HALT becomes Finished, and
/// VmState lacks consumed-input history. Compare each observable they expose;
/// the direct bytecode checks above separately cover exact stop and input tape.
fn compare_facades(source: &str, environment: &Environment, pure: bool) {
    let expected = compare(source, environment);
    let parsed = ourochronos::parser::parse(source).unwrap();
    let mut memory = ourochronos::Memory::with_size(environment.anamnesis.len());
    for (address, &word) in environment.anamnesis.iter().enumerate() {
        memory.write(address as u64, Value::new(word));
    }
    let mut state = VmState::new(memory.clone());
    let mut executor = Executor::with_config(ExecutorConfig {
        max_instructions: BYTECODE_GAS,
        immediate_output: false,
        input: environment.input.clone(),
        ..ExecutorConfig::default()
    });
    executor.execute(&mut state, &parsed).unwrap();
    let expected_status = match expected.stop {
        Stop::Finished | Stop::Halted => EpochStatus::Finished,
        Stop::Paradox => EpochStatus::Paradox,
    };
    assert_eq!(state.status, expected_status, "facade stop for {source}");
    assert_eq!(
        state.stack.iter().map(|word| word.val).collect::<Vec<_>>(),
        expected.stack
    );
    assert_eq!(
        (0..memory.len())
            .map(|address| state.present.read(address as u64).val)
            .collect::<Vec<_>>(),
        expected.present
    );
    assert_eq!(observations(&state.output), expected.output);
    let epoch = executor.run_epoch(&parsed, &memory);
    assert_eq!(epoch.status, expected_status);
    assert_eq!(epoch.inputs_consumed, expected.consumed);
    if pure {
        assert!(
            is_program_pure(&parsed),
            "declared pure case rejected: {source}"
        );
        let mut fast = FastExecutor::new(BYTECODE_GAS).with_input(environment.input.clone());
        fast.present = ourochronos::Memory::with_size(environment.anamnesis.len());
        fast.execute_pure(&parsed, &parsed.quotes).unwrap();
        assert_eq!(fast.status, expected_status);
        assert_eq!(
            fast.stack
                .to_value_vec()
                .iter()
                .map(|word| word.val)
                .collect::<Vec<_>>(),
            expected.stack
        );
        assert_eq!(observations(&fast.output), expected.output);
        assert_eq!(
            (0..memory.len())
                .map(|address| fast.present.read(address as u64).val)
                .collect::<Vec<_>>(),
            expected.present
        );
    }
}

#[test]
fn declared_core_examples_cover_stacks_control_calls_and_quotes() {
    let environment = Environment {
        input: vec![11, 22],
        ..Environment::default()
    };
    for source in [
        "",
        "5 2 SUB DUP OUTPUT",
        "1 2 3 ROT DEPTH OUTPUT",
        "10 20 30 1 PICK 2 ROLL 3 REVERSE",
        "1 DUP SWAP OVER POP",
        "0 REVERSE DEPTH OUTPUT",
        "18446744073709551615 NEG OUTPUT",
        "9223372036854775808 ABS DUP SIGN OUTPUT OUTPUT",
        "0 NOT OUTPUT 2 NOT OUTPUT 0 SIGN OUTPUT",
        "65 EMIT 321 EMIT 18446744073709551615 EMIT",
        "INPUT INPUT SUB DUP OUTPUT",
        "0 IF { 41 } ELSE { 42 } OUTPUT",
        "9 IF { 41 } ELSE { 42 } OUTPUT",
        "0 IF { 9 OUTPUT } 7",
        "5 WHILE { DUP 0 GT } { DUP OUTPUT 1 SUB } POP",
        "0 WHILE { DUP 0 GT } { 1 SUB }",
        "PROCEDURE square PURE { DUP MUL } 7 square OUTPUT",
        "PROCEDURE inc PURE { 1 ADD } PROCEDURE twice PURE { inc inc } 40 twice",
        "PROCEDURE countdown PURE { DUP 0 GT IF { 1 SUB countdown } } 7 countdown",
        "5 [ 1 ADD ] EXEC OUTPUT",
        "10 20 [ 1 ADD ] DIP",
        "5 [ 1 ADD ] KEEP",
        "[ [ 42 ] EXEC ] EXEC OUTPUT",
        "[ 3 WHILE { DUP 0 GT } { 1 SUB } ] EXEC",
        "5 HALT 7 OUTPUT",
    ] {
        compare_facades(source, &environment, true);
    }
    compare_facades("7 OUTPUT PARADOX", &environment, false);
}

#[test]
fn word_domain_boundaries_are_checked_without_discarding_cases() {
    let values = [0, 1, 2, 63, 64, 127, 1u64 << 63, u64::MAX];
    let operations = [
        "ADD", "SUB", "MUL", "DIV", "MOD", "AND", "OR", "XOR", "SHL", "SHR", "EQ", "NEQ", "LT",
        "GT", "LTE", "GTE", "SLT", "SGT", "SLTE", "SGTE", "MIN", "MAX",
    ];
    let environment = Environment::default();
    for left in values {
        for right in values {
            for operation in operations {
                compare(
                    &format!("{left} {right} {operation} DUP OUTPUT"),
                    &environment,
                );
            }
        }
    }
}

#[test]
fn finite_nested_loops_match_independent_iteration_counts() {
    let environment = Environment::default();
    for outer in 0..=5 {
        for inner in 0..=5 {
            let source = format!(
                "0 {outer} WHILE {{ DUP 0 GT }} {{
                    {inner} WHILE {{ DUP 0 GT }} {{ ROT 1 ADD ROT ROT 1 SUB }}
                    POP 1 SUB
                }} POP OUTPUT"
            );
            let observed = compare(&source, &environment);
            assert!(observed.stack.is_empty());
            assert_eq!(observed.output, vec![Observation::Number(outer * inner)]);
        }
    }
}

#[test]
fn present_is_fresh_and_anamnesis_and_frozen_tapes_are_immutable() {
    let environment = Environment {
        anamnesis: vec![99, 22, 33, 44],
        input: vec![65, 7, 999],
        ..Environment::default()
    };
    compare_facades(
        "0 PRESENT OUTPUT 0 ORACLE DUP 0 PROPHECY 0 PRESENT OUTPUT 0 ORACLE OUTPUT",
        &environment,
        false,
    );
    compare_facades(
        "42 1 2 STORE 3 PRESENT DUP OUTPUT 1 2 INDEX OUTPUT",
        &environment,
        false,
    );
    compare_facades(
        "18446744073709551615 1 ADD 0 PROPHECY 0 PRESENT",
        &environment,
        false,
    );
    compare_facades("INPUT EMIT INPUT DUP OUTPUT", &environment, true);
    let source = "0 PRESENT 1 ADD DUP OUTPUT 0 PROPHECY";
    let wanted = expected(source, &environment);
    assert_eq!(wanted.present, vec![1, 0, 0, 0]);
    assert_eq!(wanted.output, vec![Observation::Number(1)]);
    let program = bytecode(source, environment.anamnesis.len());
    let memory = anamnesis(&environment);
    let vm = BytecodeVm::with_config(config(&environment, BYTECODE_GAS));
    for _ in 0..3 {
        assert_eq!(snapshot(&vm.run(&program, &memory).unwrap()), wanted);
        assert_eq!(words(&memory), environment.anamnesis);
    }
    for policy in [Bounds::Wrap, Bounds::Clamp] {
        let configured = Environment {
            bounds: policy,
            ..environment.clone()
        };
        compare(
            "7 ORACLE OUTPUT 42 7 PROPHECY 7 PRESENT OUTPUT",
            &configured,
        );
        // Base+offset wraps as a word before address policy is applied.
        compare(
            "77 18446744073709551615 1 STORE 0 PRESENT OUTPUT",
            &configured,
        );
    }
}

fn fault(error: BytecodeVmError, environment: &Environment) -> Fault {
    match error {
        BytecodeVmError::StackUnderflow { .. } => Fault::Underflow,
        BytecodeVmError::MemoryOutOfBounds {
            address,
            memory_cells,
        } => Fault::Address {
            address,
            width: memory_cells,
        },
        BytecodeVmError::InputExhausted { consumed } => Fault::InputExhausted { consumed },
        BytecodeVmError::RandomInputExhausted { consumed } => Fault::RandomExhausted { consumed },
        BytecodeVmError::InvalidQuote { .. } => Fault::InvalidCode,
        BytecodeVmError::StackLimitExceeded { limit } => {
            assert_eq!(limit, environment.stack);
            Fault::StackLimit
        }
        BytecodeVmError::CallDepthExceeded { limit } => {
            assert_eq!(limit, environment.calls);
            Fault::CallLimit
        }
        BytecodeVmError::AllocationLimit {
            what: "output",
            limit,
        } => {
            assert_eq!(limit, environment.output);
            Fault::OutputLimit
        }
        other => panic!("unexpected production failure layer: {other:?}"),
    }
}

#[test]
fn admitted_dynamic_failures_match_the_independent_core() {
    let base = Environment::default();
    let cases = [
        (
            "INPUT INPUT ADD",
            Environment {
                input: vec![1],
                ..base.clone()
            },
            Fault::InputExhausted { consumed: 1 },
        ),
        (
            "1 INPUT PICK",
            Environment {
                input: vec![7],
                ..base.clone()
            },
            Fault::Underflow,
        ),
        (
            "1 [ NOP ] POP INPUT EXEC",
            Environment {
                input: vec![99],
                ..base.clone()
            },
            Fault::InvalidCode,
        ),
        (
            "INPUT ORACLE",
            Environment {
                input: vec![7],
                ..base.clone()
            },
            Fault::Address {
                address: 7,
                width: 4,
            },
        ),
        (
            "1 2 3",
            Environment {
                stack: 2,
                ..base.clone()
            },
            Fault::StackLimit,
        ),
        (
            "1 OUTPUT 2 OUTPUT",
            Environment {
                output: 1,
                ..base.clone()
            },
            Fault::OutputLimit,
        ),
        (
            "PROCEDURE countdown PURE { DUP 0 GT IF { 1 SUB countdown } } 7 countdown",
            Environment { calls: 2, ..base },
            Fault::CallLimit,
        ),
    ];
    for (source, environment, wanted) in cases {
        let reference = oracle::parse(source).unwrap();
        assert_eq!(
            oracle::evaluate(&reference, &environment).unwrap_err(),
            wanted
        );
        // Admission is required to succeed; a compiler rejection cannot hide a
        // runtime counterexample and is reported by bytecode() as its own layer.
        let program = bytecode(source, environment.anamnesis.len());
        let error = BytecodeVm::with_config(config(&environment, BYTECODE_GAS))
            .run(&program, &anamnesis(&environment))
            .unwrap_err();
        assert_eq!(
            fault(error, &environment),
            wanted,
            "dynamic failure for {source}"
        );
    }
}

#[test]
fn source_admission_is_distinct_from_dynamic_failure() {
    for source in ["SWAP", "1 ADD", "INPUT IF { 1 } ELSE { }"] {
        let program = oracle::parse(source).unwrap();
        let environment = Environment {
            input: vec![0],
            ..Environment::default()
        };
        if source.starts_with("INPUT") {
            assert!(oracle::evaluate(&program, &environment).is_ok());
        } else {
            assert_eq!(
                oracle::evaluate(&program, &environment).unwrap_err(),
                Fault::Underflow
            );
        }
        let parsed = ourochronos::parser::parse(source).unwrap();
        let error = admit_program(&parsed, AdmissionConfig { memory_cells: 4 }).unwrap_err();
        assert!(
            matches!(
                error.phase,
                AdmissionPhase::Types | AdmissionPhase::Semantics
            ),
            "{error}"
        );
    }
    for source in ["1 IF { 2", "[ 1", "18446744073709551616"] {
        assert!(oracle::parse(source).is_err());
        assert!(ourochronos::parser::parse(source).is_err());
    }
}

#[test]
fn source_fuel_and_public_bytecode_gas_have_separate_units() {
    let source = "5 2 ADD OUTPUT";
    let environment = Environment::default();
    let reference = oracle::parse(source).unwrap();
    let model = oracle::evaluate(&reference, &environment).unwrap();
    let program = bytecode(source, 4);
    // Four literal/primitive records and one implicit main Return: this count
    // is derived from the declared straight-line lowering contract, not read
    // from instructions.len() or copied from an observed execution count.
    for gas in 0..=5 {
        let result = BytecodeVm::with_config(config(&environment, gas))
            .run(&program, &anamnesis(&environment));
        if gas < 5 {
            assert!(matches!(result, Err(BytecodeVmError::GasExhausted { limit }) if limit == gas));
        } else {
            let result = result.unwrap();
            assert_eq!(result.instructions_executed, 5);
            assert_eq!(snapshot(&result), model.snapshot);
        }
    }
    let infinite = "WHILE { 1 } { NOP }";
    let environment = Environment {
        steps: 25,
        ..environment
    };
    assert_eq!(
        oracle::evaluate(&oracle::parse(infinite).unwrap(), &environment).unwrap_err(),
        Fault::StepLimit
    );
    assert!(matches!(
        BytecodeVm::with_config(config(&environment, 40))
            .run(&bytecode(infinite, 4), &anamnesis(&environment)),
        Err(BytecodeVmError::GasExhausted { limit: 40 })
    ));
}

#[test]
fn supported_artifact_paths_preserve_independent_observations() {
    let environment = Environment::default();
    for source in [
        "18446744073709551615 1 ADD DUP OUTPUT",
        "3 WHILE { DUP 0 GT } { DUP OUTPUT 1 SUB } POP",
        "PROCEDURE square PURE { DUP MUL } 7 square OUTPUT [ 5 1 ADD ] EXEC",
        "42 1 2 STORE 3 PRESENT OUTPUT 0 ORACLE DUP 0 PROPHECY",
    ] {
        let wanted = expected(source, &environment);
        let original = bytecode(source, 4);
        let encoded = original.to_bytes().unwrap();
        let decoded = BytecodeProgram::from_bytes(&encoded).unwrap();
        assert_eq!(decoded.to_bytes().unwrap(), encoded);
        assert_eq!(run(&decoded, &environment), wanted);
        let object = ObjectModule::new("conformance", original);
        let bytes = object.to_bytes().unwrap();
        let restored = ObjectModule::from_bytes(&bytes).unwrap();
        assert_eq!(restored.to_bytes().unwrap(), bytes);
        let linked = link(&[restored]).unwrap();
        assert_eq!(run(&linked, &environment), wanted);
        let manifest = PackageManifest::with_runtime("core", 4, BYTECODE_GAS, BoundsPolicy::Error);
        let package = PortablePackage::new(manifest, linked).unwrap();
        let bytes = package.to_bytes().unwrap();
        let restored = PortablePackage::from_bytes(&bytes).unwrap();
        assert_eq!(restored.to_bytes().unwrap(), bytes);
        assert_eq!(run(&restored.program, &environment), wanted);
    }
    let program = bytecode("INPUT OUTPUT", 4);
    assert!(matches!(
        PortablePackage::new(PackageManifest::with_memory("frozen-input", 4), program),
        Err(PackageError::UnsupportedPrimitive(
            ourochronos::OpCode::Input
        ))
    ));
}

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "ourochronos-core-conformance-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn real_module_objects_link_and_package_deterministically() {
    let fixture = Fixture::new();
    std::fs::write(
        fixture.0.join("lib.ouro"),
        "PROCEDURE square PURE { DUP MUL }",
    )
    .unwrap();
    let main = "IMPORT \"lib.ouro\" 7 square OUTPUT [ 5 1 ADD ] EXEC OUTPUT";
    std::fs::write(fixture.0.join("main.ouro"), main).unwrap();
    // Imports are outside the oracle grammar. The independently specified
    // expected module composition has a declaration-only dependency and the
    // root initializer; it does not inspect or translate the production AST.
    let wanted = expected(
        "PROCEDURE square PURE { DUP MUL } 7 square OUTPUT [ 5 1 ADD ] EXEC OUTPUT",
        &Environment::default(),
    );
    let graph = ModuleGraph::load(fixture.0.join("main.ouro"), vec![]).unwrap();
    let objects = graph.compile_objects().unwrap();
    assert_eq!(objects.len(), 2);
    let mut decoded = Vec::new();
    for object in objects {
        assert!(!object.metadata.source_files.is_empty());
        let bytes = object.to_bytes().unwrap();
        let object = ObjectModule::from_bytes(&bytes).unwrap();
        assert_eq!(object.to_bytes().unwrap(), bytes);
        decoded.push(object);
    }
    let first = link(&decoded).unwrap();
    decoded.reverse();
    let second = link(&decoded).unwrap();
    assert_eq!(first.to_bytes().unwrap(), second.to_bytes().unwrap());
    assert_eq!(run(&first, &Environment::default()), wanted);
    let package = PortablePackage::new(PackageManifest::with_memory("modules", 4), second).unwrap();
    let restored = PortablePackage::from_bytes(&package.to_bytes().unwrap()).unwrap();
    assert_eq!(run(&restored.program, &Environment::default()), wanted);
}

#[test]
fn solver_witnesses_are_checked_by_independent_finite_transitions() {
    for (source, width, radix) in [
        ("0 ORACLE 1 AND DUP OUTPUT 0 PROPHECY", 1, 2usize),
        (
            "1 ORACLE 1 ADD 7 AND DUP 0 PROPHECY OUTPUT 0 ORACLE 2 MUL 7 AND DUP 1 PROPHECY OUTPUT",
            2,
            8,
        ),
        ("0 ORACLE NOT DUP OUTPUT 0 PROPHECY", 1, 2),
    ] {
        let core = oracle::parse(source).unwrap();
        let mut fixed = Vec::new();
        for index in 0..radix.pow(width as u32) {
            let mut remainder = index;
            let state: Vec<_> = (0..width)
                .map(|_| {
                    let digit = (remainder % radix) as u64;
                    remainder /= radix;
                    digit
                })
                .collect();
            let environment = Environment {
                anamnesis: state.clone(),
                ..Environment::default()
            };
            let observed = oracle::evaluate(&core, &environment).unwrap().snapshot;
            // Every transition in the declared finite domain remains closed.
            assert!(observed.present.iter().all(|&word| word < radix as u64));
            if observed.present == state {
                fixed.push(state);
            }
        }
        let program = bytecode(source, width);
        let result = GlobalFixedPointSolver::solve_bytecode(
            &program,
            GlobalSolveConfig {
                memory_cells: width,
                bounds_policy: BoundsPolicy::Error,
                solver_timeout_ms: 5000,
                max_instructions: BYTECODE_GAS,
                ..GlobalSolveConfig::default()
            },
        );
        match result {
            GlobalSolveResult::Found(witness) => {
                let state: Vec<_> = (0..width)
                    .map(|address| witness.memory.read(address as u64).val)
                    .collect();
                assert!(
                    fixed.contains(&state),
                    "solver state absent from independent fixed states"
                );
                let environment = Environment {
                    anamnesis: state.clone(),
                    ..Environment::default()
                };
                let wanted = oracle::evaluate(&core, &environment).unwrap().snapshot;
                assert_eq!(wanted.present, state);
                assert_eq!(observations(&witness.output), wanted.output);
                assert!(witness.completeness.proves_global_unsat());
                assert_eq!(run(&program, &environment), wanted);
                let manifest = PackageManifest::with_runtime(
                    "solver-core",
                    width,
                    BYTECODE_GAS,
                    BoundsPolicy::Error,
                );
                let state = state
                    .into_iter()
                    .enumerate()
                    .filter(|(_, word)| *word != 0)
                    .map(|(address, word)| (address as u64, word))
                    .collect();
                let embedded = PackageWitness::replay_bound(
                    &manifest,
                    &program,
                    state,
                    witness.instructions_executed,
                )
                .unwrap();
                let package =
                    PortablePackage::with_replay_witness(manifest, program, embedded).unwrap();
                let restored = PortablePackage::from_bytes(&package.to_bytes().unwrap()).unwrap();
                assert_eq!(run(&restored.program, &environment), wanted);
            }
            GlobalSolveResult::ProvenNoFixedPoint(certificate) => {
                assert!(fixed.is_empty());
                assert!(certificate.completeness.proves_global_unsat());
                // Agreement on this exhaustive finite domain does not turn
                // the backend evidence into an independent UNSAT certificate.
            }
            other => panic!("finite solver case was not decided: {other:?}"),
        }
    }
    let program = bytecode("42 0 PROPHECY 0 PRESENT OUTPUT", 1);
    assert!(
        matches!(
            GlobalFixedPointSolver::solve_bytecode(
                &program,
                GlobalSolveConfig {
                    memory_cells: 1,
                    max_instructions: 1,
                    bounds_policy: BoundsPolicy::Error,
                    solver_timeout_ms: 5000,
                    ..GlobalSolveConfig::default()
                }
            ),
            GlobalSolveResult::Unknown { .. }
        ),
        "gas-truncated solver witness became eligible"
    );
}

#[test]
fn changed_oracle_expectations_detect_consequential_disagreements() {
    let source = "INPUT DUP 0 PROPHECY 0 PRESENT OUTPUT 65 EMIT 7";
    let environment = Environment {
        input: vec![42],
        ..Environment::default()
    };
    let wanted = expected(source, &environment);
    let actual = run(&bytecode(source, 4), &environment);
    assert_eq!(actual, wanted);
    let mut mutants = vec![wanted.clone(); 6];
    mutants[0].stack[0] = 8;
    mutants[1].present[0] = 0;
    mutants[2].output[0] = Observation::Number(43);
    mutants[3].output[1] = Observation::Number(65);
    mutants[4].consumed[0] = 41;
    mutants[5].stop = Stop::Halted;
    for mutant in mutants {
        assert_ne!(
            actual, mutant,
            "a changed observation escaped the comparison"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]
    #[test]
    fn generated_wrapping_expressions_and_metamorphic_identities(
        a in any::<u64>(), b in any::<u64>(), c in any::<u64>(), count in any::<u64>(),
    ) {
        let environment = Environment::default();
        let left = compare(&format!("{a} {b} ADD {c} ADD DUP OUTPUT"), &environment);
        let right = compare(&format!("{a} {b} {c} ADD ADD DUP OUTPUT"), &environment);
        prop_assert_eq!(left, right);
        let original = compare(&format!("{a} DUP OUTPUT"), &environment);
        let restored = compare(&format!("{a} {b} ADD {b} SUB DUP OUTPUT"), &environment);
        prop_assert_eq!(original, restored);
        let shifted = compare(&format!("{a} {count} SHL DUP OUTPUT"), &environment);
        let reduced = compare(&format!("{a} {} SHL DUP OUTPUT", count % 64), &environment);
        prop_assert_eq!(shifted, reduced);
    }
}
