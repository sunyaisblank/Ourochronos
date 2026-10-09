//! Exhaustive small graph oracles, coupled systems, and trusted extraction
//! conformance for the independent symbolic affine recurrence checker.

use ourochronos::hir::HirProgram;
use ourochronos::parser::parse;
use ourochronos::stdlib::StdLib;
use ourochronos::temporal::affine_recurrence::{
    certify_affine_recurrence, check_affine_bytecode_binding, check_affine_recurrence_certificate,
    extract_affine_bytecode, AffineBooleanSystem, AffineClassCount, AffineExtractionConfig,
    AffineReadout, AffineReadoutOutcome, AffineRecurrenceCertificate, AffineRecurrenceError,
    AFFINE_RECURRENCE_SEMANTICS_VERSION,
};
use ourochronos::{
    BoundsPolicy, BytecodeProgram, BytecodeVm, BytecodeVmConfig, BytecodeVmStatus, PagedMemory,
    Value,
};
use std::collections::BTreeSet;

fn query(mask: u64) -> AffineReadout {
    AffineReadout {
        mask,
        constant: false,
    }
}

fn model(rows: &[u64], offset: u64) -> AffineBooleanSystem {
    AffineBooleanSystem {
        bits: rows.len() as u8,
        rows: rows.to_vec(),
        offset,
    }
}

// A third implementation used only as a finite graph oracle: expand row bits
// and XOR ordinary booleans, rather than compose matrices or count set bits.
fn oracle_step(system: &AffineBooleanSystem, state: u64) -> u64 {
    let mut output = 0;
    for bit in 0..system.bits as usize {
        let mut value = ((system.offset >> bit) & 1) != 0;
        for input in 0..system.bits {
            if ((system.rows[bit] >> input) & 1) != 0 {
                value ^= ((state >> input) & 1) != 0;
            }
        }
        if value {
            output |= 1u64 << bit;
        }
    }
    output
}

fn oracle_readout(readout: AffineReadout, state: u64) -> bool {
    let mut value = readout.constant;
    for bit in 0..64 {
        if ((readout.mask >> bit) & 1) != 0 {
            value ^= ((state >> bit) & 1) != 0;
        }
    }
    value
}

fn graph_cycles(system: &AffineBooleanSystem) -> BTreeSet<Vec<u64>> {
    let count = 1usize << system.bits;
    let successors: Vec<_> = (0..count)
        .map(|state| oracle_step(system, state as u64) as usize)
        .collect();
    let mut cycles = BTreeSet::new();
    for seed in 0..count {
        let mut visited = vec![usize::MAX; count];
        let mut path = Vec::new();
        let mut state = seed;
        while visited[state] == usize::MAX {
            visited[state] = path.len();
            path.push(state as u64);
            state = successors[state];
        }
        let mut cycle = path[visited[state]..].to_vec();
        cycle.sort_unstable();
        cycles.insert(cycle);
    }
    cycles
}

fn assert_graph_claim(
    system: &AffineBooleanSystem,
    readout: AffineReadout,
    cycles: &BTreeSet<Vec<u64>>,
) {
    let certificate = certify_affine_recurrence(system, readout);
    if cycles.iter().any(|cycle| cycle.len() != 1) {
        assert!(
            matches!(
                certificate,
                Err(AffineRecurrenceError::StabilizationFailed { .. })
            ),
            "periodic model {system:?}"
        );
        return;
    }
    let certificate = certificate.unwrap();
    let verified = check_affine_recurrence_certificate(&certificate, system, readout).unwrap();
    assert_eq!(
        verified.class_count.as_u128().unwrap(),
        cycles.len() as u128,
        "{system:?}"
    );
    let values: BTreeSet<_> = cycles
        .iter()
        .map(|cycle| oracle_readout(readout, cycle[0]))
        .collect();
    match verified.outcome {
        AffineReadoutOutcome::Uniform { value } => assert_eq!(values, BTreeSet::from([value])),
        AffineReadoutOutcome::Disagreement {
            zero_state,
            one_state,
        } => {
            assert_eq!(values, BTreeSet::from([false, true]));
            assert!(cycles.contains(&vec![zero_state]) && cycles.contains(&vec![one_state]));
            assert_ne!(zero_state, one_state);
            assert!(!oracle_readout(readout, zero_state));
            assert!(oracle_readout(readout, one_state));
        }
    }
    // Every state reaches one of the reported singleton classes by n steps.
    for seed in 0..1u64 << system.bits {
        let mut reached = seed;
        for _ in 0..system.bits {
            reached = oracle_step(system, reached);
        }
        assert_eq!(oracle_step(system, reached), reached);
        assert!(cycles.contains(&vec![reached]));
    }
}

#[test]
fn exhaustive_all_affine_models_through_three_bits_and_all_readouts() {
    for bits in 1..=3usize {
        let mask = (1u64 << bits) - 1;
        for packed in 0..1u64 << (bits * bits + bits) {
            let rows: Vec<_> = (0..bits)
                .map(|bit| (packed >> (bits * bit)) & mask)
                .collect();
            let system = model(&rows, packed >> (bits * bits));
            let cycles = graph_cycles(&system);
            for readout_mask in 0..=mask {
                for constant in [false, true] {
                    assert_graph_claim(
                        &system,
                        AffineReadout {
                            mask: readout_mask,
                            constant,
                        },
                        &cycles,
                    );
                }
            }
        }
    }
}

#[test]
fn complete_graphs_through_eight_bits_cover_coupled_stable_and_periodic_models() {
    let mut random = 0x4288_16c5_1234_abcd_u64;
    for bits in 4..=8usize {
        let mask = (1u64 << bits) - 1;
        for case in 0..48 {
            let mut next_word = || {
                random = random
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                random
            };
            let rank = case % bits;
            let rows: Vec<_> = (0..bits)
                .map(|bit| {
                    if case % 2 == 0 {
                        if bit < rank {
                            1u64 << bit
                        } else {
                            next_word() & ((1u64 << bit) - 1)
                        }
                    } else {
                        next_word() & mask
                    }
                })
                .collect();
            let offset = if case % 2 == 0 {
                next_word() & mask & !((1u64 << rank) - 1)
            } else {
                next_word() & mask
            };
            let system = model(&rows, offset);
            let cycles = graph_cycles(&system);
            for readout in [
                query(0),
                query(mask),
                AffineReadout {
                    mask: next_word() & mask,
                    constant: true,
                },
            ] {
                assert_graph_claim(&system, readout, &cycles);
            }
        }
    }
}

#[test]
fn coupled_xor_feedback_and_affine_offset_have_two_step_transients() {
    // Both coordinates depend on both old coordinates. Cancellation gives A^2=0;
    // this is genuinely coupled XOR feedback, beyond copying/routing bits.
    let system = model(&[3, 3], 1);
    let c = certify_affine_recurrence(&system, query(3)).unwrap();
    assert_eq!(c.stabilized_power, model(&[0, 0], 2));
    assert_eq!(c.rank, 0);
    assert_eq!(c.class_count.as_u128(), Some(1));
    assert_eq!(c.outcome, AffineReadoutOutcome::Uniform { value: true });
    assert_ne!(system.apply(0).unwrap(), 2);
    assert_eq!(system.apply(system.apply(0).unwrap()).unwrap(), 2);
    assert_graph_claim(&system, query(3), &graph_cycles(&system));
}

#[test]
fn coupled_feedforward_transient_reaches_full_dimension_sixty_four() {
    let rows: Vec<_> = (0..64)
        .map(|bit| match bit {
            0 => 0,
            1 => 1,
            _ => (1u64 << (bit - 1)) | (1u64 << (bit - 2)),
        })
        .collect();
    let system = model(&rows, 1);
    let c = certify_affine_recurrence(&system, query(u64::MAX)).unwrap();
    assert!(c.stabilized_power.rows.iter().all(|&row| row == 0));
    assert_eq!(c.class_count.as_u128(), Some(1));
    // A^63 e0 is nonzero while A^64 e0 is zero, exhibiting a true 64-step
    // transient (the affine offset cancels in the difference of two runs).
    let mut zero = 0;
    let mut unit = 1;
    for _ in 0..63 {
        zero = oracle_step(&system, zero);
        unit = oracle_step(&system, unit);
    }
    assert_ne!(zero, unit);
    assert_eq!(oracle_step(&system, zero), oracle_step(&system, unit));
    assert_eq!(
        system.apply(c.stabilized_power.offset).unwrap(),
        c.stabilized_power.offset
    );
}

#[test]
fn sixty_four_bits_count_classes_without_enumeration_and_produce_large_witnesses() {
    let mut rows: Vec<_> = (0..64).map(|bit| 1u64 << bit).collect();
    rows[2] = 3;
    for (bit, row) in rows.iter_mut().enumerate().skip(3) {
        *row = 1u64 << (bit - 1);
    }
    let system = model(&rows, 0);
    let c = certify_affine_recurrence(&system, query(1u64 << 63)).unwrap();
    assert_eq!(c.rank, 2);
    assert_eq!(c.class_count.as_u128(), Some(4));
    assert_eq!(
        c.outcome,
        AffineReadoutOutcome::Disagreement {
            zero_state: 0,
            one_state: u64::MAX ^ 2
        }
    );
    let uniform = certify_affine_recurrence(&system, query((1u64 << 2) | (1u64 << 63))).unwrap();
    assert_eq!(
        uniform.outcome,
        AffineReadoutOutcome::Uniform { value: false }
    );
    let identity = model(&(0..64).map(|bit| 1u64 << bit).collect::<Vec<_>>(), 0);
    let c = certify_affine_recurrence(&identity, query(1u64 << 63)).unwrap();
    assert_eq!(c.rank, 64);
    assert_eq!(c.class_count.as_u128(), Some(18_446_744_073_709_551_616));
    assert_eq!(
        c.outcome,
        AffineReadoutOutcome::Disagreement {
            zero_state: 0,
            one_state: 1u64 << 63
        }
    );
    assert_eq!(AffineClassCount { exponent: 65 }.as_u128(), None);
}

#[test]
fn periodic_swap_not_unipotent_and_rotations_are_rejected() {
    for system in [
        model(&[2, 1], 0),
        model(&[1], 1),
        model(&[1, 3], 0),
        model(&[2, 4, 1], 0),
    ] {
        assert!(graph_cycles(&system).iter().any(|cycle| cycle.len() > 1));
        assert!(matches!(
            certify_affine_recurrence(&system, query(1)),
            Err(AffineRecurrenceError::StabilizationFailed { .. })
        ));
    }
    // Explicitly forge the tempting F^(2n)=F^n argument for an involution.
    let swap = model(&[2, 1], 0);
    let identity = model(&[1, 2], 0);
    let mut forged = certify_affine_recurrence(&identity, query(1)).unwrap();
    forged.model = swap.clone();
    assert_eq!(forged.stabilized_power, identity); // F^2 really is identity.
    assert!(matches!(
        check_affine_recurrence_certificate(&forged, &swap, query(1)),
        Err(AffineRecurrenceError::StabilizationFailed { .. })
    ));
}

#[test]
fn independently_supplied_certificate_binds_model_query_version_and_all_claims() {
    let system = model(&[1, 1], 0);
    let readout = query(1);
    let hand = AffineRecurrenceCertificate {
        semantics_version: AFFINE_RECURRENCE_SEMANTICS_VERSION,
        model: system.clone(),
        query: readout,
        stabilized_power: system.clone(),
        image_basis: vec![3],
        rank: 1,
        class_count: AffineClassCount { exponent: 1 },
        outcome: AffineReadoutOutcome::Disagreement {
            zero_state: 0,
            one_state: 3,
        },
    };
    check_affine_recurrence_certificate(&hand, &system, readout).unwrap();
    assert_eq!(certify_affine_recurrence(&system, readout).unwrap(), hand);
    for field in 0..9 {
        let mut changed = hand.clone();
        match field {
            0 => changed.semantics_version += 1,
            1 => changed.model.offset = 3,
            2 => changed.query.constant = true,
            3 => changed.stabilized_power.rows[0] = 0,
            4 => changed.stabilized_power.offset = 3,
            5 => changed.image_basis[0] = 1,
            6 => changed.rank = 0,
            7 => changed.class_count.exponent = 64,
            8 => changed.outcome = AffineReadoutOutcome::Uniform { value: false },
            _ => unreachable!(),
        }
        assert!(
            check_affine_recurrence_certificate(&changed, &system, readout).is_err(),
            "field {field}"
        );
    }
    let mut forged_witness = hand.clone();
    forged_witness.outcome = AffineReadoutOutcome::Disagreement {
        zero_state: 0,
        one_state: 1,
    };
    assert!(check_affine_recurrence_certificate(&forged_witness, &system, readout).is_err());
    assert!(check_affine_recurrence_certificate(&hand, &model(&[1, 2], 0), readout).is_err());
    assert!(check_affine_recurrence_certificate(&hand, &system, query(2)).is_err());
    let mut excessive = hand;
    excessive.image_basis = vec![3; 65];
    assert!(check_affine_recurrence_certificate(&excessive, &system, readout).is_err());
}

#[test]
fn malformed_dimensions_coefficients_offsets_states_and_readouts_reject() {
    for invalid in [
        AffineBooleanSystem {
            bits: 0,
            rows: vec![],
            offset: 0,
        },
        AffineBooleanSystem {
            bits: 65,
            rows: vec![0; 65],
            offset: 0,
        },
        AffineBooleanSystem {
            bits: 2,
            rows: vec![1],
            offset: 0,
        },
        model(&[4, 1], 0),
        model(&[1, 2], 4),
    ] {
        assert!(invalid.validate().is_err());
        assert!(certify_affine_recurrence(&invalid, query(0)).is_err());
    }
    assert!(model(&[1, 2], 0).apply(4).is_err());
    assert!(certify_affine_recurrence(&model(&[1, 2], 0), query(4)).is_err());
}

fn compile(source: &str, prelude: bool) -> BytecodeProgram {
    let mut parsed = parse(source).unwrap();
    if prelude {
        parsed.procedures.extend(StdLib::procedures());
    }
    BytecodeProgram::compile(&HirProgram::resolve(&parsed).unwrap()).unwrap()
}

fn source_for(system: &AffineBooleanSystem) -> String {
    let mut source = format!("TEMPORAL 0 {} BITS 1 {{", system.bits);
    for (cell, &row) in system.rows.iter().enumerate() {
        source.push_str(&format!(" {}", (system.offset >> cell) & 1));
        for dependency in 0..system.bits {
            if ((row >> dependency) & 1) != 0 {
                source.push_str(&format!(" {dependency} ORACLE XOR"));
            }
        }
        source.push_str(&format!(" {cell} PROPHECY"));
    }
    source.push_str(" }");
    source
}

fn compare_adapter_vm(
    program: &BytecodeProgram,
    config: &AffineExtractionConfig,
) -> AffineBooleanSystem {
    let extracted = extract_affine_bytecode(program, config).unwrap();
    check_affine_bytecode_binding(&extracted, program, config).unwrap();
    let system = extracted.model;
    let vm = BytecodeVm::with_config(BytecodeVmConfig {
        max_instructions: config.max_instructions,
        max_stack_depth: config.max_stack_depth,
        memory_bounds: config.memory_bounds,
        ..BytecodeVmConfig::default()
    });
    for state in 0..1u64 << system.bits {
        let mut input = PagedMemory::with_size(config.memory_cells).unwrap();
        for cell in 0..system.bits {
            input
                .write(cell as u64, Value::new((state >> cell) & 1))
                .unwrap();
        }
        // Nonzero input outside the scope cannot enter fresh output frame.
        for cell in system.bits as usize..config.memory_cells {
            input.write(cell as u64, Value::new(u64::MAX)).unwrap();
        }
        let execution = vm.run(program, &input).unwrap();
        assert_eq!(execution.status, BytecodeVmStatus::Finished);
        assert!(execution.output.is_empty() && execution.effects.is_empty());
        let result = (0..system.bits).fold(0, |value, cell| {
            value | (execution.present.get(cell as u64).unwrap().val << cell)
        });
        assert_eq!(result, oracle_step(&system, state));
        for cell in system.bits as usize..config.memory_cells {
            assert_eq!(execution.present.get(cell as u64).unwrap().val, 0);
        }
        assert_eq!(
            execution.instructions_executed,
            (program.main.end - program.main.start) as u64
        );
    }
    system
}

#[test]
fn coupled_adapter_models_match_complete_vm_graphs_including_unused_prelude() {
    let cfg = AffineExtractionConfig {
        memory_cells: 16,
        ..AffineExtractionConfig::default()
    };
    for original in [
        model(&[3, 3], 1),
        model(&[1, 2, 3, 4], 8),
        model(&[0, 1, 3, 6, 12, 24, 48, 96], 1),
    ] {
        let program = compile(&source_for(&original), true);
        assert!(!program.procedures.is_empty());
        let extracted = compare_adapter_vm(&program, &cfg);
        assert_eq!(extracted, original);
        assert_graph_claim(
            &extracted,
            query((1u64 << extracted.bits) - 1),
            &graph_cycles(&extracted),
        );
    }
    let present = compile(
        "TEMPORAL 0 2 BITS 1 { 0 ORACLE 1 ORACLE XOR 0 PROPHECY 0 PRESENT 1 XOR 1 PROPHECY }",
        true,
    );
    assert_eq!(compare_adapter_vm(&present, &cfg), model(&[3, 3], 2));
}

#[test]
fn sixty_four_cell_coupled_bytecode_extracts_without_domain_enumeration() {
    let mut rows = vec![1, 2, 3];
    rows.extend((3..64).map(|bit| 1u64 << (bit - 1)));
    let original = model(&rows, 0);
    let program = compile(&source_for(&original), true);
    let cfg = AffineExtractionConfig {
        memory_cells: 128,
        ..AffineExtractionConfig::default()
    };
    let extracted = extract_affine_bytecode(&program, &cfg).unwrap();
    assert_eq!(extracted.model, original);
    let certificate = certify_affine_recurrence(&extracted.model, query(1u64 << 63)).unwrap();
    assert_eq!(certificate.class_count.as_u128(), Some(4));
    assert_eq!(
        certificate.outcome,
        AffineReadoutOutcome::Disagreement {
            zero_state: 0,
            one_state: u64::MAX ^ 2
        }
    );
    let vm = BytecodeVm::with_config(BytecodeVmConfig {
        max_instructions: cfg.max_instructions,
        max_stack_depth: cfg.max_stack_depth,
        ..BytecodeVmConfig::default()
    });
    for state in [0, 1, 2, 1u64 << 63, u64::MAX] {
        let mut input = PagedMemory::with_size(cfg.memory_cells).unwrap();
        for cell in 0..64 {
            input.write(cell, Value::new((state >> cell) & 1)).unwrap();
        }
        let observed = vm.run(&program, &input).unwrap();
        let actual = (0..64).fold(0, |value, bit| {
            value | (observed.present.get(bit).unwrap().val << bit)
        });
        assert_eq!(actual, oracle_step(&original, state));
    }
}

#[test]
fn full_word_symbolic_constants_addresses_masking_and_bounds_are_preserved() {
    let mut cfg = AffineExtractionConfig {
        memory_cells: 8,
        ..AffineExtractionConfig::default()
    };
    let high = compile(
        "TEMPORAL 0 2 BITS 1 { 18446744073709551615 0 ORACLE XOR 0 PROPHECY 0 PRESENT 1 PROPHECY }",
        false,
    );
    assert_eq!(compare_adapter_vm(&high, &cfg), model(&[1, 1], 3));
    let cancel = compile(
        "TEMPORAL 0 2 BITS 1 { 18446744073709551615 18446744073709551615 XOR ORACLE 0 PROPHECY }",
        false,
    );
    assert_eq!(compare_adapter_vm(&cancel, &cfg), model(&[1, 0], 0));
    let outside = compile(
        "TEMPORAL 0 2 BITS 1 { 18446744073709551615 ORACLE 18446744073709551615 PROPHECY }",
        false,
    );
    assert!(extract_affine_bytecode(&outside, &cfg).is_err());
    cfg.memory_bounds = BoundsPolicy::Wrap;
    assert_eq!(compare_adapter_vm(&outside, &cfg), model(&[0, 2], 0));
    cfg.memory_bounds = BoundsPolicy::Clamp;
    assert_eq!(compare_adapter_vm(&outside, &cfg), model(&[0, 2], 0));
    let wrap_zero = compile("TEMPORAL 0 2 BITS 1 { 2 ORACLE 2 PROPHECY }", false);
    cfg.memory_bounds = BoundsPolicy::Wrap;
    assert_eq!(compare_adapter_vm(&wrap_zero, &cfg), model(&[1, 0], 0));
}

#[test]
fn adapter_identity_resource_and_unsupported_boundaries_are_explicit() {
    let program = compile(&source_for(&model(&[3, 3], 1)), true);
    let cfg = AffineExtractionConfig {
        memory_cells: 8,
        ..AffineExtractionConfig::default()
    };
    let extracted = extract_affine_bytecode(&program, &cfg).unwrap();
    let mut different = program.clone();
    let unused = different.procedures[0].range.start as usize;
    different.instructions[unused] = ourochronos::Instruction::Primitive(ourochronos::OpCode::Nop);
    assert!(check_affine_bytecode_binding(&extracted, &different, &cfg).is_err());
    for field in 0..4 {
        let mut changed = cfg.clone();
        match field {
            0 => changed.memory_cells += 1,
            1 => changed.memory_bounds = BoundsPolicy::Wrap,
            2 => changed.max_instructions += 1,
            3 => changed.max_stack_depth += 1,
            _ => unreachable!(),
        }
        assert!(check_affine_bytecode_binding(&extracted, &program, &changed).is_err());
    }
    let mut changed = extracted.clone();
    changed.model.offset ^= 1;
    assert!(check_affine_bytecode_binding(&changed, &program, &cfg).is_err());
    changed = extracted;
    changed.semantics_version += 1;
    assert!(check_affine_bytecode_binding(&changed, &program, &cfg).is_err());
    let mut exact = cfg.clone();
    exact.max_instructions = (program.main.end - program.main.start) as u64;
    extract_affine_bytecode(&program, &exact).unwrap();
    exact.max_instructions -= 1;
    assert!(matches!(
        extract_affine_bytecode(&program, &exact),
        Err(AffineRecurrenceError::AdapterResourceLimit(_))
    ));
    exact = cfg;
    exact.max_stack_depth = 0;
    assert!(extract_affine_bytecode(&program, &exact).is_err());
    for source in [
        "TEMPORAL 0 2 BITS 1 { 0 ORACLE ORACLE 0 PROPHECY }",
        "TEMPORAL 0 2 BITS 1 { 0 ORACLE 1 ADD 0 PROPHECY }",
        "TEMPORAL 0 2 BITS 1 { 0 ORACLE OUTPUT }",
        "TEMPORAL 0 2 BITS 1 { 0 ORACLE IF { 1 0 PROPHECY } }",
        "TEMPORAL 0 2 BITS 1 { WHILE { 0 } { NOP } }",
        "TEMPORAL 0 2 BITS 1 { TEMPORAL 0 1 BITS 1 { NOP } }",
        "PROCEDURE helper { 0 ORACLE 0 PROPHECY } TEMPORAL 0 2 BITS 1 { helper }",
        "TEMPORAL 0 2 BITS 2 { 0 ORACLE 0 PROPHECY }",
        "TEMPORAL 1 2 BITS 1 { 0 ORACLE 0 PROPHECY }",
        "TEMPORAL 0 2 BITS 1 { [ 0 ] POP }",
        "TEMPORAL 0 2 BITS 1 { INPUT POP }",
    ] {
        assert!(
            extract_affine_bytecode(&compile(source, false), &AffineExtractionConfig::default())
                .is_err(),
            "{source}"
        );
    }
}
