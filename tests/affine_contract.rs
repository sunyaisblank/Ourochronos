//! Independent finite oracles and source/VM pipelines for affine contracts.

use ourochronos::hir::HirProgram;
use ourochronos::parser::parse;
use ourochronos::stdlib::StdLib;
use ourochronos::temporal::affine_contract::*;
use ourochronos::temporal::affine_recurrence::{
    certify_affine_recurrence, AffineBooleanSystem, AffineExtractionConfig, AffineReadout,
};
use ourochronos::{
    BoundsPolicy, BytecodeProgram, BytecodeVm, BytecodeVmConfig, PagedMemory, Value,
};

fn equation(mask: u64, value: bool) -> AffineParityEquation {
    AffineParityEquation { mask, value }
}
fn relation(before_mask: u64, after_mask: u64, value: bool) -> AffineTemporalRelation {
    AffineTemporalRelation {
        before_mask,
        after_mask,
        value,
    }
}
fn component(
    rows: &[u64],
    offset: u64,
    assumptions: &[AffineParityEquation],
    guarantees: &[AffineTemporalRelation],
    protected_bits: u64,
) -> AffineContractComponent {
    AffineContractComponent {
        model: AffineBooleanSystem {
            bits: rows.len() as u8,
            rows: rows.to_vec(),
            offset,
        },
        contract: AffineTemporalContract {
            assumptions: assumptions.to_vec(),
            guarantees: guarantees.to_vec(),
            protected_bits,
        },
        source: None,
    }
}
fn bit_parity(mask: u64, state: u64) -> bool {
    let mut result = false;
    for bit in 0..64 {
        if mask & (1u64 << bit) != 0 {
            result ^= state & (1u64 << bit) != 0;
        }
    }
    result
}
fn oracle_step(model: &AffineBooleanSystem, state: u64) -> u64 {
    let mut result = model.offset;
    for (bit, &row) in model.rows.iter().enumerate() {
        if bit_parity(row, state) {
            result ^= 1u64 << bit;
        }
    }
    result
}
fn meets_pre(contract: &AffineTemporalContract, state: u64) -> bool {
    contract
        .assumptions
        .iter()
        .all(|eq| bit_parity(eq.mask, state) == eq.value)
}
fn meets_post(contract: &AffineTemporalContract, initial: u64, next: u64) -> bool {
    (initial ^ next) & contract.protected_bits == 0
        && contract.guarantees.iter().all(|row| {
            bit_parity(row.before_mask, initial) ^ bit_parity(row.after_mask, next) == row.value
        })
}
fn proven(component: &AffineContractComponent) -> AffineContractCertificate {
    match certify_affine_contract(component).unwrap() {
        AffineContractAnalysis::Proven(c) => c,
        other => panic!("expected proven contract: {other:?}"),
    }
}
fn composed(
    left: &AffineContractComponent,
    right: &AffineContractComponent,
) -> AffineCompositionCertificate {
    match compose_affine_contracts(left, right).unwrap() {
        AffineCompositionAnalysis::Proven(c) => *c,
        other => panic!("expected composed contract: {other:?}"),
    }
}

fn assert_oracle(component: &AffineContractComponent) {
    let satisfying: Vec<_> = (0..1u64 << component.model.bits)
        .filter(|&state| meets_pre(&component.contract, state))
        .collect();
    let true_contract = satisfying.iter().all(|&state| {
        meets_post(
            &component.contract,
            state,
            oracle_step(&component.model, state),
        )
    });
    match certify_affine_contract(component).unwrap() {
        AffineContractAnalysis::UnsatisfiableAssumptions => assert!(satisfying.is_empty()),
        AffineContractAnalysis::Proven(certificate) => {
            assert!(!satisfying.is_empty() && true_contract);
            let checked = check_affine_contract_certificate(&certificate, component).unwrap();
            assert_eq!(
                checked.satisfying_states.as_u128(),
                Some(satisfying.len() as u128)
            );
        }
        AffineContractAnalysis::Counterexample(example) => {
            assert!(!satisfying.is_empty() && !true_contract);
            assert!(satisfying.contains(&example.initial));
            assert_eq!(
                example.successor,
                oracle_step(&component.model, example.initial)
            );
            match example.failure {
                AffineContractFailure::Guarantee { index } => {
                    let row = component.contract.guarantees[index];
                    assert_ne!(
                        bit_parity(row.before_mask, example.initial)
                            ^ bit_parity(row.after_mask, example.successor),
                        row.value
                    );
                }
                AffineContractFailure::ProtectedBit { bit } => {
                    assert_ne!((example.initial ^ example.successor) & (1u64 << bit), 0)
                }
            }
        }
    }
}

#[test]
fn exhaustive_small_models_parity_assumptions_relations_and_frames_match_oracle() {
    for bits in 1..=2usize {
        let domain = (1u64 << bits) - 1;
        for packed in 0..1u64 << (bits * bits + bits) {
            let rows: Vec<_> = (0..bits)
                .map(|bit| (packed >> (bit * bits)) & domain)
                .collect();
            let offset = packed >> (bits * bits);
            for assumption_mask in 0..=domain {
                for assumption_value in [false, true] {
                    for before in 0..=domain {
                        for after in 0..=domain {
                            for value in [false, true] {
                                assert_oracle(&component(
                                    &rows,
                                    offset,
                                    &[equation(assumption_mask, assumption_value)],
                                    &[relation(before, after, value)],
                                    0,
                                ));
                            }
                        }
                    }
                    for frame in 0..=domain {
                        assert_oracle(&component(
                            &rows,
                            offset,
                            &[equation(assumption_mask, assumption_value)],
                            &[],
                            frame,
                        ));
                    }
                }
            }
        }
    }
}

#[test]
fn complete_eight_bit_oracles_cover_multiple_coupled_equations_and_counterexamples() {
    let mut random = 0x91ce_ef02_9dd0_3801_u64;
    for bits in 3..=8usize {
        let domain = (1u64 << bits) - 1;
        for case in 0..48 {
            let mut next = || {
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                random
            };
            let rows: Vec<_> = (0..bits).map(|_| next() & domain).collect();
            let offset = next() & domain;
            let assumptions: Vec<_> = (0..case % 5)
                .map(|_| equation(next() & domain, next() & 1 != 0))
                .collect();
            let guarantees: Vec<_> = (0..case % 4)
                .map(|_| relation(next() & domain, next() & domain, next() & 1 != 0))
                .collect();
            let protected = next() & domain;
            assert_oracle(&component(
                &rows,
                offset,
                &assumptions,
                &guarantees,
                protected,
            ));
        }
    }
}

#[test]
fn nonvacuity_counts_contradictions_tautologies_and_frame_failures_are_explicit() {
    let empty = component(&[1, 2], 0, &[], &[], 3);
    assert_eq!(proven(&empty).summary.satisfying_states.as_u128(), Some(4));
    let duplicates = component(
        &[1, 2],
        0,
        &[equation(3, true), equation(3, true), equation(0, false)],
        &[relation(3, 0, true)],
        3,
    );
    let c = proven(&duplicates);
    assert_eq!(c.summary.assumption_rank, 1);
    assert_eq!(c.summary.satisfying_states.as_u128(), Some(2));
    for pre in [
        vec![equation(0, true)],
        vec![equation(3, false), equation(3, true)],
    ] {
        let impossible = component(&[1, 2], 3, &pre, &[relation(0, 0, true)], 3);
        assert_eq!(
            certify_affine_contract(&impossible).unwrap(),
            AffineContractAnalysis::UnsatisfiableAssumptions
        );
        let mut forged = proven(&empty);
        forged.component = impossible.clone();
        assert_eq!(
            check_affine_contract_certificate(&forged, &impossible),
            Err(AffineContractError::VacuousAssumptions)
        );
    }
    let flips = component(&[1, 2], 1, &[], &[], 1);
    match certify_affine_contract(&flips).unwrap() {
        AffineContractAnalysis::Counterexample(example) => {
            assert_eq!(
                example.failure,
                AffineContractFailure::ProtectedBit { bit: 0 }
            );
            assert_eq!(example.successor, example.initial ^ 1);
        }
        other => panic!("{other:?}"),
    }
}

fn realistic_components() -> (AffineContractComponent, AffineContractComponent) {
    let left = component(
        &[1, 2, 3, 8],
        0,
        &[equation(3, false)],
        &[
            relation(1, 1, false),
            relation(2, 2, false),
            relation(3, 4, false),
            relation(8, 8, false),
        ],
        11,
    );
    let right = component(
        &[5, 6, 4, 8],
        0,
        &[equation(4, false)],
        &[
            relation(5, 1, false),
            relation(6, 2, false),
            relation(4, 4, false),
            relation(8, 8, false),
        ],
        12,
    );
    (left, right)
}

#[test]
fn coupled_components_compose_relations_preserve_frame_and_bind_the_composed_map() {
    let (left, right) = realistic_components();
    // G has two-cycles outside its precondition. Contract verification requires
    // no recurrence-stabilization theorem for general affine transfers.
    assert!(certify_affine_recurrence(
        &right.model,
        AffineReadout {
            mask: 1,
            constant: false
        }
    )
    .is_err());
    let c = composed(&left, &right);
    let checked = check_affine_composition_certificate(&c, &left, &right).unwrap();
    assert_eq!(checked.model.rows, [2, 1, 3, 8]);
    assert_eq!(checked.contract.protected_bits, 8);
    assert_eq!(checked.contract.assumptions, left.contract.assumptions);
    assert!(checked.contract.guarantees.contains(&relation(3, 4, false))); // Retained: G frames all AFTER bits.
    assert!(checked.contract.guarantees.contains(&relation(2, 1, false))); // G BEFORE pulled through F.
    assert!(!checked.contract.guarantees.contains(&relation(1, 1, false))); // Dropped: G does not frame bit0.
    for state in 0..16 {
        assert_eq!(
            oracle_step(&checked.model, state),
            oracle_step(&right.model, oracle_step(&left.model, state))
        );
        if meets_pre(&left.contract, state) {
            assert!(meets_post(
                &checked.contract,
                state,
                oracle_step(&checked.model, state)
            ));
        }
    }
    for field in 0..6 {
        let mut bad = c.clone();
        match field {
            0 => bad.semantics_version += 1,
            1 => bad.composed.component.model.rows[0] = 1,
            2 => bad.composed.component.contract.protected_bits = 0,
            3 => bad.canonical_handoff[0].value ^= true,
            4 => bad.left.component.contract.guarantees.clear(),
            5 => bad.right.component.model.offset ^= 1,
            _ => unreachable!(),
        }
        assert!(
            check_affine_composition_certificate(&bad, &left, &right).is_err(),
            "field {field}"
        );
    }
}

#[test]
fn composition_pullback_includes_affine_offset() {
    let left = component(&[1, 1], 2, &[], &[relation(0, 3, true)], 1);
    let right = component(
        &[1, 2],
        3,
        &[equation(3, true)],
        &[relation(1, 1, true), relation(2, 2, true)],
        0,
    );
    let c = composed(&left, &right);
    assert_eq!(c.composed.component.model.rows, [1, 1]);
    assert_eq!(c.composed.component.model.offset, 1);
    assert_eq!(
        c.composed.component.contract.guarantees,
        [relation(1, 1, true), relation(1, 2, false)]
    );
    for state in 0..4 {
        assert!(meets_post(
            &c.composed.component.contract,
            state,
            oracle_step(&c.composed.component.model, state)
        ));
    }
    let mut bad = c;
    bad.composed.component.contract.guarantees[1].value = true;
    assert!(check_affine_composition_certificate(&bad, &left, &right).is_err());
}

#[test]
fn actual_handoff_failure_differs_from_insufficient_abstract_guarantee() {
    let actual_bad = component(&[1], 0, &[], &[], 1);
    let right = component(&[1], 0, &[equation(1, false)], &[], 1);
    match compose_affine_contracts(&actual_bad, &right).unwrap() {
        AffineCompositionAnalysis::HandoffCounterexample {
            initial,
            intermediate,
            assumption_index,
        } => {
            assert_eq!(intermediate, initial);
            assert_eq!(intermediate, 1);
            assert_eq!(assumption_index, 0);
        }
        other => panic!("expected real failing handoff: {other:?}"),
    }
    let weak = component(&[0], 0, &[], &[], 0); // Actually establishes AFTER=0 but does not declare it.
    match compose_affine_contracts(&weak, &right).unwrap() {
        AffineCompositionAnalysis::InsufficientGuarantee {
            initial,
            permitted_intermediate,
            actual_intermediate,
            assumption_index,
        } => {
            assert!(meets_pre(&weak.contract, initial));
            assert!(meets_post(&weak.contract, initial, permitted_intermediate));
            assert_eq!(permitted_intermediate, 1);
            assert_eq!(actual_intermediate, 0);
            assert_eq!(assumption_index, 0);
            assert!(meets_pre(&right.contract, actual_intermediate));
            assert!(!meets_pre(&right.contract, permitted_intermediate));
        }
        other => panic!("expected insufficient guarantee: {other:?}"),
    }
    let strong = component(&[0], 0, &[], &[relation(0, 1, false)], 0);
    composed(&strong, &right);
    let impossible = component(&[1], 0, &[equation(0, true)], &[], 0);
    assert!(matches!(
        compose_affine_contracts(&impossible, &right).unwrap(),
        AffineCompositionAnalysis::UnsatisfiableAssumptions {
            side: AffineComponentSide::Left
        }
    ));
    assert!(matches!(
        compose_affine_contracts(&strong, &impossible).unwrap(),
        AffineCompositionAnalysis::UnsatisfiableAssumptions {
            side: AffineComponentSide::Right
        }
    ));
}

#[test]
fn exhaustive_two_bit_composition_pairs_match_direct_pipeline_and_handoff_oracle() {
    let models: Vec<_> = (0..64u64)
        .map(|packed| {
            let rows = [packed & 3, (packed >> 2) & 3];
            let offset = packed >> 4;
            let guarantees: Vec<_> = (0..2)
                .map(|bit| relation(rows[bit], 1u64 << bit, offset & (1u64 << bit) != 0))
                .collect();
            component(
                &rows,
                offset,
                &[equation((packed >> 1) & 3, packed & 1 != 0)],
                &guarantees,
                0,
            )
        })
        .collect();
    for left in &models {
        for right in &models {
            let left_states: Vec<_> = (0..4)
                .filter(|&state| meets_pre(&left.contract, state))
                .collect();
            let right_states: Vec<_> = (0..4)
                .filter(|&state| meets_pre(&right.contract, state))
                .collect();
            let valid_handoff = left_states
                .iter()
                .all(|&state| meets_pre(&right.contract, oracle_step(&left.model, state)));
            match compose_affine_contracts(left, right).unwrap() {
                AffineCompositionAnalysis::Proven(certificate) => {
                    assert!(!left_states.is_empty() && !right_states.is_empty() && valid_handoff);
                    let checked =
                        check_affine_composition_certificate(&certificate, left, right).unwrap();
                    for initial in 0..4 {
                        let successor =
                            oracle_step(&right.model, oracle_step(&left.model, initial));
                        assert_eq!(oracle_step(&checked.model, initial), successor);
                        if meets_pre(&left.contract, initial) {
                            assert!(meets_post(&checked.contract, initial, successor));
                        }
                    }
                }
                AffineCompositionAnalysis::UnsatisfiableAssumptions {
                    side: AffineComponentSide::Left,
                } => assert!(left_states.is_empty()),
                AffineCompositionAnalysis::UnsatisfiableAssumptions {
                    side: AffineComponentSide::Right,
                } => assert!(!left_states.is_empty() && right_states.is_empty()),
                AffineCompositionAnalysis::HandoffCounterexample {
                    initial,
                    intermediate,
                    assumption_index,
                } => {
                    assert!(!valid_handoff && left_states.contains(&initial));
                    assert_eq!(intermediate, oracle_step(&left.model, initial));
                    let assumption = right.contract.assumptions[assumption_index];
                    assert_ne!(bit_parity(assumption.mask, intermediate), assumption.value);
                }
                other => panic!(
                    "complete exact left relations should need no stronger guarantee: {other:?}"
                ),
            }
        }
    }
}

#[test]
fn sixty_four_bit_contract_and_joint_handoff_are_proved_without_enumeration() {
    let rows: Vec<_> = (0..64)
        .map(|bit| if bit < 2 { 1u64 << bit } else { 3 })
        .collect();
    let left = component(
        &rows,
        0,
        &[equation(3, false)],
        &[relation(3, 1u64 << 63, false)],
        3,
    );
    let identity: Vec<_> = (0..64).map(|bit| 1u64 << bit).collect();
    let right = component(
        &identity,
        0,
        &[equation(1u64 << 63, false)],
        &[relation(u64::MAX, u64::MAX, false)],
        u64::MAX,
    );
    let c = composed(&left, &right);
    let checked = check_affine_composition_certificate(&c, &left, &right).unwrap();
    assert_eq!(
        checked.summary.satisfying_states.as_u128(),
        Some(1u128 << 63)
    );
    assert_eq!(checked.contract.protected_bits, 3);
    assert_eq!(checked.model, left.model);
    let unconstrained = component(&identity, 0, &[], &[], u64::MAX);
    assert_eq!(
        proven(&unconstrained).summary.satisfying_states.as_u128(),
        Some(1u128 << 64)
    );
}

#[test]
fn certificate_rechecks_truth_and_binds_model_contract_source_and_summary() {
    let good = component(
        &[1, 1],
        0,
        &[equation(1, true)],
        &[relation(0, 3, false)],
        1,
    );
    let c = proven(&good);
    for field in 0..7 {
        let mut bad = c.clone();
        match field {
            0 => bad.semantics_version += 1,
            1 => bad.component.model.offset ^= 1,
            2 => bad.component.contract.assumptions[0].value = false,
            3 => bad.canonical_assumptions[0].value = false,
            4 => bad.summary.assumption_rank = 0,
            5 => bad.summary.satisfying_states.exponent = 64,
            6 => bad.summary.protected_bits = 0,
            _ => unreachable!(),
        }
        assert!(check_affine_contract_certificate(&bad, &good).is_err());
    }
    let wrong = component(
        &[1, 1],
        1,
        &[equation(1, true)],
        &[relation(0, 3, false)],
        1,
    );
    let mut forged = c;
    forged.component = wrong.clone();
    assert!(matches!(
        check_affine_contract_certificate(&forged, &wrong),
        Err(AffineContractError::InvalidCertificate(_))
    ));
    let mut large = proven(&good);
    large.canonical_assumptions = vec![equation(1, true); 65];
    assert!(check_affine_contract_certificate(&large, &good).is_err());
    for invalid in [
        component(&[1, 2], 0, &[equation(4, false)], &[], 0),
        component(&[1, 2], 0, &[], &[relation(0, 4, false)], 0),
        component(&[1, 2], 0, &[], &[], 4),
    ] {
        assert!(certify_affine_contract(&invalid).is_err());
    }
    let mut excessive = good.clone();
    excessive.contract.assumptions = vec![equation(0, false); MAX_AFFINE_ASSUMPTIONS + 1];
    assert!(matches!(
        certify_affine_contract(&excessive),
        Err(AffineContractError::ResourceLimit(_))
    ));
    excessive = good;
    excessive.contract.guarantees = vec![relation(0, 0, false); MAX_AFFINE_GUARANTEES + 1];
    assert!(matches!(
        certify_affine_contract(&excessive),
        Err(AffineContractError::ResourceLimit(_))
    ));
}

fn compile(source: &str) -> BytecodeProgram {
    let mut parsed = parse(source).unwrap();
    parsed.procedures.extend(StdLib::procedures());
    BytecodeProgram::compile(&HirProgram::resolve(&parsed).unwrap()).unwrap()
}
fn source_for(component: &AffineContractComponent) -> String {
    let mut source = format!("TEMPORAL 0 {} BITS 1 {{", component.model.bits);
    for (cell, &row) in component.model.rows.iter().enumerate() {
        source.push_str(&format!(" {}", (component.model.offset >> cell) & 1));
        for bit in 0..component.model.bits {
            if row & (1u64 << bit) != 0 {
                source.push_str(&format!(" {bit} ORACLE XOR"));
            }
        }
        source.push_str(&format!(" {cell} PROPHECY"));
    }
    source.push_str(" }");
    source
}

#[test]
fn coupled_source_components_compose_in_the_explicit_vm_memory_pipeline() {
    let (left, right) = realistic_components();
    let left_program = compile(&source_for(&left));
    let right_program = compile(&source_for(&right));
    let resources = AffineExtractionConfig {
        memory_cells: 16,
        ..AffineExtractionConfig::default()
    };
    let left = affine_component_from_bytecode(&left_program, &resources, left.contract).unwrap();
    let right = affine_component_from_bytecode(&right_program, &resources, right.contract).unwrap();
    check_affine_source_component(&left, &left_program, &resources).unwrap();
    check_affine_source_component(&right, &right_program, &resources).unwrap();
    let certificate = composed(&left, &right);
    let vm = BytecodeVm::with_config(BytecodeVmConfig {
        max_instructions: resources.max_instructions,
        max_stack_depth: resources.max_stack_depth,
        memory_bounds: resources.memory_bounds,
        ..BytecodeVmConfig::default()
    });
    for initial in 0..16 {
        let mut input = PagedMemory::with_size(16).unwrap();
        for bit in 0..4 {
            input.write(bit, Value::new((initial >> bit) & 1)).unwrap();
        }
        let middle = vm.run(&left_program, &input).unwrap();
        let final_state = vm.run(&right_program, &middle.present).unwrap();
        let successor = (0..4).fold(0, |value, bit| {
            value | (final_state.present.get(bit).unwrap().val << bit)
        });
        assert_eq!(
            successor,
            oracle_step(&certificate.composed.component.model, initial)
        );
        if meets_pre(&left.contract, initial) {
            assert!(meets_post(
                &certificate.composed.component.contract,
                initial,
                successor
            ));
        }
    }
    let mut changed = resources.clone();
    changed.max_instructions += 1;
    assert!(check_affine_source_component(&left, &left_program, &changed).is_err());
    let mut bad = certificate.clone();
    bad.left
        .component
        .source
        .as_mut()
        .unwrap()
        .resources
        .max_instructions += 1;
    assert!(check_affine_composition_certificate(&bad, &left, &right).is_err());
    bad = certificate;
    bad.left.component.source.as_mut().unwrap().program_sha256[0] ^= 1;
    assert!(check_affine_composition_certificate(&bad, &left, &right).is_err());
}

#[test]
fn source_aliasing_gas_stack_host_loop_and_wrapping_boundaries_are_explicit() {
    let source = compile("TEMPORAL 0 2 BITS 1 { 0 ORACLE 0 PROPHECY 1 ORACLE 2 PROPHECY }");
    let resources = AffineExtractionConfig {
        memory_cells: 8,
        memory_bounds: BoundsPolicy::Wrap,
        ..AffineExtractionConfig::default()
    };
    let frame = AffineTemporalContract {
        assumptions: vec![],
        guarantees: vec![],
        protected_bits: 1,
    };
    let aliased = affine_component_from_bytecode(&source, &resources, frame).unwrap();
    assert_eq!(aliased.model.rows, [2, 0]);
    match certify_affine_contract(&aliased).unwrap() {
        AffineContractAnalysis::Counterexample(example) => {
            assert_eq!(
                example.failure,
                AffineContractFailure::ProtectedBit { bit: 0 }
            );
            assert!(meets_pre(&aliased.contract, example.initial));
            assert!(!meets_post(
                &aliased.contract,
                example.initial,
                example.successor
            ));
        }
        other => panic!("{other:?}"),
    }
    let mut exact = resources.clone();
    exact.max_instructions = (source.main.end - source.main.start) as u64;
    affine_component_from_bytecode(&source, &exact, aliased.contract.clone()).unwrap();
    exact.max_instructions -= 1;
    assert!(affine_component_from_bytecode(&source, &exact, aliased.contract.clone()).is_err());
    exact = resources.clone();
    exact.max_stack_depth = 0;
    assert!(affine_component_from_bytecode(&source, &exact, aliased.contract.clone()).is_err());
    exact = resources;
    exact.memory_bounds = BoundsPolicy::Error;
    assert!(affine_component_from_bytecode(&source, &exact, aliased.contract.clone()).is_err());
    for body in [
        "0 ORACLE 1 ADD 0 PROPHECY",
        "WHILE { 1 } { NOP }",
        "CLOCK 0 PROPHECY",
        "INPUT 0 PROPHECY",
        "RANDOM 0 PROPHECY",
        "VEC_NEW POP",
        "0 ORACLE OUTPUT",
    ] {
        let source = compile(&format!("TEMPORAL 0 2 BITS 1 {{ {body} }}"));
        assert!(
            affine_component_from_bytecode(
                &source,
                &AffineExtractionConfig::default(),
                aliased.contract.clone()
            )
            .is_err(),
            "{body}"
        );
    }
}
