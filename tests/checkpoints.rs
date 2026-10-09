//! Saved-state continuation qualified against the independent numeric oracle.
//! Oracle source steps and fetched VM records use distinct gas conventions;
//! terminal stores/stacks/output/input/faults are compared without equating them.
#[allow(dead_code)]
mod oracle;

use oracle::{Bounds, Environment, Fault, Observation, Snapshot, Stop};
use ourochronos::checkpoint::{
    CheckpointError, CheckpointOutcome, CheckpointPauseReason, ClassicalCheckpoint,
    ClassicalCheckpointConfig, UnverifiedCheckpoint, MAX_CHECKPOINT_BYTES,
    MAX_CHECKPOINT_INSTRUCTIONS, MAX_CHECKPOINT_MEMORY_CELLS,
};
use ourochronos::halting::{BoundedHaltingAnalyzer, ResumableHaltingResult};
use ourochronos::{
    admit_program, AdmissionConfig, BoundsPolicy, BytecodeExecution, BytecodeProgram, BytecodeVm,
    BytecodeVmConfig, BytecodeVmError, BytecodeVmStatus, OutputItem, PagedMemory,
};
use proptest::prelude::*;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicBool, Ordering};

fn config() -> ClassicalCheckpointConfig {
    ClassicalCheckpointConfig {
        memory_cells: 4,
        max_instructions: 10_000,
        input: vec![65, 7],
        max_stack_depth: 512,
        max_call_depth: 128,
        max_output_items: 512,
        ..ClassicalCheckpointConfig::default()
    }
}

#[test]
fn unused_procedures_and_dynamic_quote_capability_closure_are_distinct() {
    let config = config();
    let ordinary = bytecode("PROCEDURE latent { 0 ORACLE POP } 42 OUTPUT", &config);
    let result = complete(
        ClassicalCheckpoint::start(&ordinary, config.clone())
            .unwrap()
            .run_slice(100, None)
            .unwrap(),
    );
    assert!(matches!(&result.output[..], [OutputItem::Val(value)] if value.val == 42));
    let reachable = bytecode("PROCEDURE latent { 0 ORACLE POP } latent", &config);
    assert!(matches!(
        ClassicalCheckpoint::start(&reachable, config.clone()),
        Err(CheckpointError::UnsupportedOpcode("ORACLE"))
    ));
    // A numeric word can select quotation zero, so a reachable combinator must
    // conservatively inspect every quote, including later literal definitions.
    let parsed = ourochronos::parser::parse("INPUT EXEC [ 0 ORACLE POP ] POP").unwrap();
    let dynamic =
        BytecodeProgram::compile(&ourochronos::hir::HirProgram::resolve(&parsed).unwrap()).unwrap();
    assert!(matches!(
        ClassicalCheckpoint::start(&dynamic, config),
        Err(CheckpointError::UnsupportedOpcode("ORACLE"))
    ));
}

fn bytecode(source: &str, config: &ClassicalCheckpointConfig) -> BytecodeProgram {
    admit_program(
        &ourochronos::parser::parse(source).unwrap(),
        AdmissionConfig {
            memory_cells: config.memory_cells,
        },
    )
    .unwrap_or_else(|error| panic!("source {source:?}: {error}"))
    .into_program()
}

fn vm_config(config: &ClassicalCheckpointConfig) -> BytecodeVmConfig {
    BytecodeVmConfig {
        max_instructions: config.max_instructions,
        max_call_depth: config.max_call_depth,
        max_stack_depth: config.max_stack_depth,
        max_output_items: config.max_output_items,
        max_output_bytes: config.max_output_bytes,
        memory_bounds: config.memory_bounds,
        input: config.input.clone(),
        ..BytecodeVmConfig::default()
    }
}

fn snapshot(execution: &BytecodeExecution) -> Snapshot {
    assert!(execution.effects.is_empty());
    assert!(execution.stack.iter().all(|value| value.prov.is_pure()));
    Snapshot {
        stack: execution.stack.iter().map(|v| v.val).collect(),
        present: execution.present.iter().map(|(_, v)| v.val).collect(),
        output: execution
            .output
            .iter()
            .map(|item| match item {
                OutputItem::Val(value) => Observation::Number(value.val),
                OutputItem::Char(byte) => Observation::Byte(*byte),
            })
            .collect(),
        consumed: execution.inputs_consumed.clone(),
        stop: match execution.status {
            BytecodeVmStatus::Finished => Stop::Finished,
            BytecodeVmStatus::Halted => Stop::Halted,
            BytecodeVmStatus::Paradox => panic!("outside checkpoint core"),
        },
    }
}

fn environment(config: &ClassicalCheckpointConfig) -> Environment {
    Environment {
        anamnesis: vec![0; config.memory_cells],
        input: config.input.clone(),
        stack: config.max_stack_depth,
        calls: config.max_call_depth,
        output: config.max_output_items,
        bounds: match config.memory_bounds {
            BoundsPolicy::Error => Bounds::Error,
            BoundsPolicy::Wrap => Bounds::Wrap,
            BoundsPolicy::Clamp => Bounds::Clamp,
        },
        ..Environment::default()
    }
}

fn paused(outcome: CheckpointOutcome) -> ClassicalCheckpoint {
    match outcome {
        CheckpointOutcome::Paused { checkpoint, .. } => checkpoint,
        other => panic!("expected pause, got {other:?}"),
    }
}

fn complete(outcome: CheckpointOutcome) -> BytecodeExecution {
    match outcome {
        CheckpointOutcome::Complete(result) => result,
        other => panic!("expected completion, got {other:?}"),
    }
}

fn reload(
    checkpoint: ClassicalCheckpoint,
    program: &BytecodeProgram,
    config: &ClassicalCheckpointConfig,
) -> ClassicalCheckpoint {
    let bytes = checkpoint.to_bytes().unwrap();
    let restored = UnverifiedCheckpoint::from_bytes(&bytes)
        .unwrap()
        .validate(program, config, None)
        .unwrap();
    assert_eq!(
        restored.to_bytes().unwrap(),
        bytes,
        "restoration changed saved state"
    );
    restored
}

fn assert_every_boundary(source: &str, config: &ClassicalCheckpointConfig, independent: bool) {
    let program = bytecode(source, config);
    let direct = BytecodeVm::with_config(vm_config(config))
        .run(
            &program,
            &PagedMemory::with_size(config.memory_cells).unwrap(),
        )
        .unwrap();
    let wanted = snapshot(&direct);
    if independent {
        let expected = oracle::evaluate(&oracle::parse(source).unwrap(), &environment(config))
            .unwrap()
            .snapshot;
        assert_eq!(wanted, expected, "independent expectation: {source}");
    }
    for cut in 0..direct.instructions_executed {
        let saved = paused(
            ClassicalCheckpoint::start(&program, config.clone())
                .unwrap()
                .run_slice(cut, None)
                .unwrap(),
        );
        assert_eq!(saved.instructions_executed(), cut);
        let before_cancel = saved.to_bytes().unwrap();
        let cancelled = AtomicBool::new(true);
        let saved = paused(saved.run_slice(1, Some(&cancelled)).unwrap());
        assert_eq!(
            saved.to_bytes().unwrap(),
            before_cancel,
            "cancelled at {cut}"
        );
        let saved = reload(saved, &program, config);
        let result = complete(saved.run_slice(u64::MAX, None).unwrap());
        assert_eq!(snapshot(&result), wanted, "cut {cut}: {source}");
        assert_eq!(result.instructions_executed, direct.instructions_executed);
        assert_eq!(result.maximum_call_depth, direct.maximum_call_depth);
        assert_eq!(result.maximum_stack_depth, direct.maximum_stack_depth);
    }
    let mut saved = ClassicalCheckpoint::start(&program, config.clone()).unwrap();
    let mut fetched = 0;
    loop {
        match saved.run_slice(1, None).unwrap() {
            CheckpointOutcome::Paused { checkpoint, reason } => {
                fetched += 1;
                assert_eq!(reason, CheckpointPauseReason::SliceLimit);
                assert_eq!(checkpoint.instructions_executed(), fetched);
                saved = reload(checkpoint, &program, config);
            }
            CheckpointOutcome::Complete(result) => {
                assert_eq!(snapshot(&result), wanted);
                assert_eq!(result.instructions_executed, direct.instructions_executed);
                break;
            }
            other => panic!("unexpected {other:?}"),
        }
    }
}

#[test]
fn every_fetched_boundary_preserves_independent_classical_observations() {
    let config = config();
    for source in [
        "",
        "18446744073709551615 1 ADD DUP OUTPUT 65 EMIT",
        "PROCEDURE inc PURE { 1 ADD } PROCEDURE twice PURE { inc inc } 40 twice OUTPUT",
        "PROCEDURE countdown PURE { DUP 0 GT IF { 1 SUB countdown } } 4 countdown OUTPUT",
        "[ [ 42 ] EXEC ] EXEC OUTPUT",
        "10 20 [ 1 ADD ] DIP OUTPUT OUTPUT",
        "5 [ 1 ADD ] KEEP OUTPUT OUTPUT",
        "INPUT EMIT INPUT DUP OUTPUT 1 2 STORE 3 PRESENT OUTPUT",
        "4 WHILE { DUP 0 GT } { DUP OUTPUT 1 SUB } POP",
        "0 IF { 41 } ELSE { 42 } OUTPUT",
        "5 HALT 7 OUTPUT",
        "18446744073709551615 0 SLT OUTPUT",
    ] {
        assert_every_boundary(source, &config, true);
    }
    for policy in [BoundsPolicy::Wrap, BoundsPolicy::Clamp] {
        let mut configured = config.clone();
        configured.memory_bounds = policy;
        assert_every_boundary("77 0 7 STORE 7 PRESENT OUTPUT", &configured, true);
    }
}

#[test]
fn bi_second_and_rec_completion_frames_resume_without_repeating_calls() {
    for source in [
        "5 [ 1 ADD ] [ 2 MUL ] BI OUTPUT OUTPUT",
        "9 [ POP 1 ADD ] REC OUTPUT",
    ] {
        assert_every_boundary(source, &config(), false);
    }
}

#[test]
fn packed_memory_and_stack_strings_keep_the_shared_dispatch_contract() {
    // These operators are outside the independent oracle's grammar; qualify
    // continuation against authoritative uninterrupted dispatch explicitly.
    for source in [
        "11 22 0 2 PACK 0 2 UNPACK OUTPUT OUTPUT",
        "65 66 2 STR_REV OUTPUT OUTPUT OUTPUT",
        "65 1 66 1 STR_CAT OUTPUT OUTPUT OUTPUT",
        "65 44 66 3 44 STR_SPLIT OUTPUT OUTPUT OUTPUT OUTPUT OUTPUT",
    ] {
        assert_every_boundary(source, &config(), false);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]
    #[test]
    fn generated_wrapping_words_remain_equal_after_arbitrary_saved_cut(
        left in any::<u64>(), right in any::<u64>(), cut in 0u64..6,
    ) {
        let config = config();
        let source = format!("{left} {right} ADD DUP OUTPUT");
        let program = bytecode(&source, &config);
        let expected = oracle::evaluate(&oracle::parse(&source).unwrap(), &environment(&config)).unwrap().snapshot;
        let saved = paused(ClassicalCheckpoint::start(&program, config.clone()).unwrap().run_slice(cut, None).unwrap());
        let saved = reload(saved, &program, &config);
        let result = complete(saved.run_slice(u64::MAX, None).unwrap());
        prop_assert_eq!(snapshot(&result), expected);
        prop_assert_eq!(result.instructions_executed, 6);
    }
}

fn expected_fault(error: &BytecodeVmError) -> Fault {
    match error {
        BytecodeVmError::InputExhausted { consumed } => Fault::InputExhausted {
            consumed: *consumed,
        },
        BytecodeVmError::StackUnderflow { .. } => Fault::Underflow,
        BytecodeVmError::InvalidQuote { .. } => Fault::InvalidCode,
        BytecodeVmError::MemoryOutOfBounds {
            address,
            memory_cells,
        } => Fault::Address {
            address: *address,
            width: *memory_cells,
        },
        BytecodeVmError::StackLimitExceeded { .. } => Fault::StackLimit,
        BytecodeVmError::CallDepthExceeded { .. } => Fault::CallLimit,
        BytecodeVmError::AllocationLimit { what: "output", .. } => Fault::OutputLimit,
        other => panic!("unexpected fault: {other:?}"),
    }
}

#[test]
fn faults_match_uninterrupted_dispatch_and_independent_oracle_after_recovery() {
    for (source, mut config, input, stack, calls, output) in [
        ("INPUT INPUT ADD", config(), vec![1], 512, 128, 512),
        ("1 INPUT PICK", config(), vec![7], 512, 128, 512),
        (
            "1 [ NOP ] POP INPUT EXEC",
            config(),
            vec![99],
            512,
            128,
            512,
        ),
        ("INPUT PRESENT", config(), vec![7], 512, 128, 512),
        ("1 2 3", config(), vec![], 2, 128, 512),
        ("1 OUTPUT 2 OUTPUT", config(), vec![], 512, 128, 1),
        (
            "PROCEDURE countdown PURE { DUP 0 GT IF { 1 SUB countdown } } 7 countdown",
            config(),
            vec![],
            512,
            2,
            512,
        ),
    ] {
        config.input = input;
        config.max_stack_depth = stack;
        config.max_call_depth = calls;
        config.max_output_items = output;
        let program = bytecode(source, &config);
        let direct = BytecodeVm::with_config(vm_config(&config))
            .run(
                &program,
                &PagedMemory::with_size(config.memory_cells).unwrap(),
            )
            .unwrap_err();
        assert_eq!(
            oracle::evaluate(&oracle::parse(source).unwrap(), &environment(&config)).unwrap_err(),
            expected_fault(&direct)
        );
        let CheckpointOutcome::Fault {
            error,
            instructions,
        } = ClassicalCheckpoint::start(&program, config.clone())
            .unwrap()
            .run_slice(u64::MAX, None)
            .unwrap()
        else {
            panic!("expected terminal fault")
        };
        assert_eq!(error, direct);
        assert!(instructions > 0);
        let mut checkpoint = ClassicalCheckpoint::start(&program, config.clone()).unwrap();
        loop {
            match checkpoint.run_slice(1, None).unwrap() {
                CheckpointOutcome::Paused {
                    checkpoint: next, ..
                } => checkpoint = reload(next, &program, &config),
                CheckpointOutcome::Fault {
                    error,
                    instructions: split_count,
                } => {
                    assert_eq!(error, direct);
                    assert_eq!(split_count, instructions);
                    break;
                }
                other => panic!("expected fault, got {other:?}"),
            }
        }
    }
}

#[test]
fn cumulative_bound_and_cancellation_remain_unknown_and_cannot_reset() {
    let mut config = config();
    config.max_instructions = 37;
    let program = bytecode("WHILE { 1 } { NOP }", &config);
    let cancellation = AtomicBool::new(true);
    let checkpoint = ClassicalCheckpoint::start(&program, config.clone()).unwrap();
    let CheckpointOutcome::Paused { checkpoint, reason } =
        checkpoint.run_slice(100, Some(&cancellation)).unwrap()
    else {
        panic!("expected cancellation")
    };
    assert_eq!(reason, CheckpointPauseReason::Cancelled);
    assert_eq!(checkpoint.instructions_executed(), 0);
    cancellation.store(false, Ordering::Relaxed);
    let checkpoint = paused(checkpoint.run_slice(13, Some(&cancellation)).unwrap());
    cancellation.store(true, Ordering::Relaxed);
    let checkpoint = paused(checkpoint.run_slice(13, Some(&cancellation)).unwrap());
    assert_eq!(checkpoint.instructions_executed(), 13);
    let bytes = checkpoint.to_bytes().unwrap();
    assert!(matches!(
        UnverifiedCheckpoint::from_bytes(&bytes).unwrap().validate(
            &program,
            &config,
            Some(&cancellation)
        ),
        Err(CheckpointError::ValidationCancelled {
            instructions_verified: 0
        })
    ));
    cancellation.store(false, Ordering::Relaxed);
    let checkpoint = UnverifiedCheckpoint::from_bytes(&bytes)
        .unwrap()
        .validate(&program, &config, Some(&cancellation))
        .unwrap();
    let checkpoint = paused(checkpoint.run_slice(100, None).unwrap());
    assert_eq!(checkpoint.instructions_executed(), 37);
    let checkpoint = reload(checkpoint, &program, &config);
    let ResumableHaltingResult::Unknown {
        instructions,
        reason,
        checkpoint,
    } = BoundedHaltingAnalyzer::analyze_checkpoint(checkpoint, u64::MAX, None).unwrap()
    else {
        panic!("cutoff must stay UNKNOWN")
    };
    assert_eq!(instructions, 37);
    assert_eq!(reason, CheckpointPauseReason::InstructionBound);
    assert_eq!(checkpoint.instructions_executed(), 37);
    assert_eq!(
        BytecodeVm::with_config(vm_config(&config))
            .run(&program, &PagedMemory::with_size(4).unwrap())
            .unwrap_err(),
        BytecodeVmError::GasExhausted { limit: 37 }
    );
    let mut expanded = config.clone();
    expanded.max_instructions += 1;
    assert!(matches!(
        UnverifiedCheckpoint::from_bytes(&checkpoint.to_bytes().unwrap())
            .unwrap()
            .validate(&program, &expanded, None),
        Err(CheckpointError::ConfigurationMismatch)
    ));
}

#[test]
fn finishing_record_at_exact_ceiling_is_a_finite_halt() {
    for (source, bound, stop) in [
        ("", 1, BytecodeVmStatus::Finished),
        ("7 HALT", 2, BytecodeVmStatus::Halted),
    ] {
        let mut config = config();
        config.max_instructions = bound;
        let program = bytecode(source, &config);
        let execution = complete(
            ClassicalCheckpoint::start(&program, config.clone())
                .unwrap()
                .run_slice(bound, None)
                .unwrap(),
        );
        assert_eq!(execution.status, stop);
        assert_eq!(execution.instructions_executed, bound);
        let result = BoundedHaltingAnalyzer::analyze_checkpoint(
            ClassicalCheckpoint::start(&program, config).unwrap(),
            bound,
            None,
        )
        .unwrap();
        assert!(
            matches!(result, ResumableHaltingResult::Halted { instructions, .. } if instructions == bound)
        );
    }
}

fn reseal(bytes: &mut [u8]) {
    let end = bytes.len() - 32;
    let mut hash = Sha256::new();
    hash.update(b"ourochronos.classical-checkpoint.image/v1\0");
    hash.update(&bytes[..end]);
    bytes[end..].copy_from_slice(&hash.finalize());
}

#[test]
fn every_single_byte_corruption_and_truncation_is_rejected() {
    let config = config();
    let program = bytecode("INPUT DUP OUTPUT 1 ADD", &config);
    let checkpoint = paused(
        ClassicalCheckpoint::start(&program, config)
            .unwrap()
            .run_slice(3, None)
            .unwrap(),
    );
    let bytes = checkpoint.to_bytes().unwrap();
    assert_eq!(checkpoint.to_bytes().unwrap(), bytes);
    for i in 0..bytes.len() {
        let mut damaged = bytes.clone();
        damaged[i] ^= 1;
        assert!(
            UnverifiedCheckpoint::from_bytes(&damaged).is_err(),
            "corrupt byte {i}"
        );
        assert!(
            UnverifiedCheckpoint::from_bytes(&bytes[..i]).is_err(),
            "truncated at {i}"
        );
    }
    let mut trailing = bytes;
    trailing.push(0);
    assert!(UnverifiedCheckpoint::from_bytes(&trailing).is_err());
}

#[test]
fn attacker_resealed_state_is_not_upgraded_by_checksum_or_valid_control_ranges() {
    let config = config();
    let program = bytecode("INPUT DUP OUTPUT 1 ADD", &config);
    let checkpoint = paused(
        ClassicalCheckpoint::start(&program, config.clone())
            .unwrap()
            .run_slice(3, None)
            .unwrap(),
    );
    let original = checkpoint.to_bytes().unwrap();
    // Header16 + code32 + config(48+1+4+input16) = state offset117.
    let state = 16 + 32 + 48 + 1 + 4 + config.input.len() * 8;
    // State scalars48, no frames (count4), stack count4, one stack word.
    for offset in [state, state + 8, state + 24, state + 32, state + 48 + 4 + 4] {
        let mut changed = original.clone();
        changed[offset] ^= 1;
        reseal(&mut changed);
        if let Ok(image) = UnverifiedCheckpoint::from_bytes(&changed) {
            assert!(
                image.validate(&program, &config, None).is_err(),
                "resealed state offset {offset} accepted"
            );
        }
    }
    // A syntactically valid replacement word, not an invalid checksum, must
    // be rejected specifically by canonical reachability equality.
    let mut changed = original;
    let word = state + 48 + 4 + 4;
    changed[word..word + 8].copy_from_slice(&99u64.to_le_bytes());
    reseal(&mut changed);
    assert!(matches!(
        UnverifiedCheckpoint::from_bytes(&changed)
            .unwrap()
            .validate(&program, &config, None),
        Err(CheckpointError::UnreachableState)
    ));
    for (source, cut, relative_word) in [
        // Scalar state48 + frame count4 + frame caller8 + completion tag1.
        ("5 [ 1 ADD ] [ 2 MUL ] BI OUTPUT OUTPUT", 4, 61),
        // Scalar48 + frame count4 + stack count4 + memory count4 + address8.
        ("77 0 0 STORE 0 PRESENT OUTPUT", 4, 68),
    ] {
        let program = bytecode(source, &config);
        let checkpoint = paused(
            ClassicalCheckpoint::start(&program, config.clone())
                .unwrap()
                .run_slice(cut, None)
                .unwrap(),
        );
        let mut changed = checkpoint.to_bytes().unwrap();
        let word = state + relative_word;
        changed[word..word + 8].copy_from_slice(&99u64.to_le_bytes());
        reseal(&mut changed);
        assert!(
            matches!(
                UnverifiedCheckpoint::from_bytes(&changed)
                    .unwrap()
                    .validate(&program, &config, None),
                Err(CheckpointError::UnreachableState)
            ),
            "resealed memory/completion {source}"
        );
    }
}

#[test]
fn exact_executable_frozen_input_and_each_resource_policy_are_bound() {
    let config = config();
    let program = bytecode("INPUT OUTPUT", &config);
    let bytes = ClassicalCheckpoint::start(&program, config.clone())
        .unwrap()
        .to_bytes()
        .unwrap();
    let changed_program = bytecode("INPUT DUP OUTPUT", &config);
    assert!(matches!(
        UnverifiedCheckpoint::from_bytes(&bytes)
            .unwrap()
            .validate(&changed_program, &config, None),
        Err(CheckpointError::ProgramMismatch)
    ));
    for field in 0..8 {
        let mut changed = config.clone();
        match field {
            0 => changed.memory_cells += 1,
            1 => changed.max_instructions += 1,
            2 => changed.max_call_depth += 1,
            3 => changed.max_stack_depth += 1,
            4 => changed.max_output_items += 1,
            5 => changed.max_output_bytes -= 1,
            6 => changed.memory_bounds = BoundsPolicy::Wrap,
            _ => changed.input[0] += 1,
        }
        assert!(
            matches!(
                UnverifiedCheckpoint::from_bytes(&bytes)
                    .unwrap()
                    .validate(&program, &changed, None),
                Err(CheckpointError::ConfigurationMismatch)
            ),
            "field {field}"
        );
    }
}

#[test]
fn declared_decoder_and_work_bounds_reject_adjacent_invalid_inputs() {
    let config = config();
    let program = bytecode("7 OUTPUT", &config);
    let bytes = ClassicalCheckpoint::start(&program, config.clone())
        .unwrap()
        .to_bytes()
        .unwrap();
    // Attacker supplies a recomputed checksum plus oversized input count.
    let mut count = bytes.clone();
    count[97..101].copy_from_slice(&u32::MAX.to_le_bytes());
    reseal(&mut count);
    assert!(UnverifiedCheckpoint::from_bytes(&count).is_err());
    let mut semantics = bytes;
    semantics[10..12].copy_from_slice(&2u16.to_le_bytes());
    reseal(&mut semantics);
    assert!(matches!(
        UnverifiedCheckpoint::from_bytes(&semantics),
        Err(CheckpointError::UnsupportedVersion { .. })
    ));
    for field in 0..3 {
        let mut changed = config.clone();
        match field {
            0 => changed.max_instructions = MAX_CHECKPOINT_INSTRUCTIONS + 1,
            1 => changed.memory_cells = MAX_CHECKPOINT_MEMORY_CELLS + 1,
            _ => changed.max_output_bytes = MAX_CHECKPOINT_BYTES + 1,
        }
        assert!(matches!(
            ClassicalCheckpoint::start(&program, changed),
            Err(CheckpointError::InvalidConfiguration)
        ));
    }
    assert!(matches!(
        UnverifiedCheckpoint::from_bytes(&vec![0; MAX_CHECKPOINT_BYTES + 1]),
        Err(CheckpointError::TooLarge { .. })
    ));
    let minimal = ClassicalCheckpointConfig {
        memory_cells: 1,
        max_instructions: 1,
        max_stack_depth: 0,
        max_call_depth: 0,
        max_output_items: 0,
        max_output_bytes: 0,
        input: vec![],
        ..config
    };
    let empty = bytecode("", &minimal);
    let checkpoint = ClassicalCheckpoint::start(&empty, minimal.clone()).unwrap();
    let restored = reload(checkpoint, &empty, &minimal);
    assert_eq!(
        complete(restored.run_slice(1, None).unwrap()).instructions_executed,
        1
    );
}

#[test]
fn checkpoint_boundary_does_not_narrow_existing_one_shot_analysis() {
    let config = config();
    for source in [
        "0 ORACLE POP",
        "CLOCK POP",
        "RANDOM POP",
        "VEC_NEW POP",
        "PARADOX",
    ] {
        let program = bytecode(source, &config);
        assert!(matches!(
            ClassicalCheckpoint::start(&program, config.clone()),
            Err(CheckpointError::UnsupportedOpcode(_))
        ));
    }
    let source = ourochronos::parser::parse("VEC_NEW POP").unwrap();
    assert!(matches!(
        BoundedHaltingAnalyzer::analyze(&source, 100, 4),
        ourochronos::BoundedHaltingResult::Halted { .. }
    ));
}
