//! Independent finite qualification of the declared FAMILY fragments.
//! Expected maps use ordinary words and orbit walks, never VM dispatch,
//! temporal lowering, the production graph analyzer, or projection wires.
//! This target proves neither general PSPACE completeness nor an ideal selector.

use ourochronos::admission::{admit_program, AdmissionConfig};
use ourochronos::core::BoundsPolicy;
use ourochronos::parser::parse;
use ourochronos::temporal::transition_graph::ProgramGraphConfig;
use ourochronos::{
    PolynomialBound, ProjectionFamilyCertificate, PspaceFamilyContract, PspaceFamilyVerifier,
    PspaceInstanceCertificate, PspaceInstanceConfig,
    PspaceInstanceVerificationResult as ResultKind, PspaceReadoutEvidence,
    PspaceUniformFamilyGenerator, UniformFamilyError,
};
use std::collections::BTreeSet;

const FLAGS: &str = "UNIFORM; TOTAL; READOUT_INVARIANT; IDEAL_DEUTSCH; EFFECTS_FROZEN;";

fn source(body: &str, cells: usize, flags: &str) -> ourochronos::ast::Program {
    parse(&format!(
        "FAMILY independent {{ CTC_CELLS POLY 0 1 {cells}; \
         CHRONOLOGY_BITS POLY 10000 1 0; TRANSITION_STEPS POLY 1000 1 0; \
         {flags} }} {body}"
    ))
    .expect("every generated fixture must parse")
}

fn width(cells: usize) -> PolynomialBound {
    PolynomialBound {
        coefficient: 0,
        degree: 1,
        additive: cells as u64,
    }
}

fn config(cells: usize, bits: u8, input: &[u8]) -> PspaceInstanceConfig {
    PspaceInstanceConfig {
        input_bits: input.len() as u64,
        input: input.to_vec(),
        graph: ProgramGraphConfig {
            memory_cells: cells,
            cell_bits: bits,
            max_states: 1 << (cells * bits as usize),
            max_instructions: 1000,
            bounds_policy: BoundsPolicy::Error,
        },
    }
}

fn verified(result: ResultKind) -> Box<PspaceInstanceCertificate> {
    match result {
        ResultKind::Verified(certificate) => certificate,
        other => panic!("expected complete verification, got {other:?}"),
    }
}

// Enumerate an orbit from EVERY node. The repeated suffix is its recurrent
// cycle; sorting/deduplication removes different rotations and shared basins.
fn cycles(edges: &[u32]) -> BTreeSet<Vec<usize>> {
    let mut answer = BTreeSet::new();
    for start in 0..edges.len() {
        let mut path = Vec::new();
        let mut current = start;
        loop {
            if let Some(first) = path.iter().position(|&state| state == current) {
                let mut cycle = path[first..].to_vec();
                cycle.sort_unstable();
                answer.insert(cycle);
                break;
            }
            path.push(current);
            current = edges[current] as usize;
        }
    }
    answer
}

fn unanimous(edges: &[u32], readouts: &[u8]) -> Option<u8> {
    let recurrent: Vec<usize> = cycles(edges).into_iter().flatten().collect();
    let first = readouts[recurrent[0]];
    recurrent
        .iter()
        .all(|&state| readouts[state] == first && first <= 1)
        .then_some(first)
}

#[test]
fn every_one_bit_map_and_boolean_readout_is_classified_independently() {
    // All 4 total maps and all 4 Boolean readout functions. Transient outputs
    // may disagree; every cycle, including one unreachable from state zero,
    // must agree. No generated parse/admission/run case can disappear.
    let mut outcomes = [0usize; 2];
    for map in 0..4 {
        let edges = vec![map & 1, (map >> 1) & 1];
        for readout in 0..4 {
            let outputs = vec![(readout & 1) as u8, ((readout >> 1) & 1) as u8];
            let body = format!(
                "0 ORACLE IF {{ {} 0 PROPHECY {} OUTPUT }} ELSE {{ {} 0 PROPHECY {} OUTPUT }}",
                edges[1], outputs[1], edges[0], outputs[0]
            );
            let result = PspaceFamilyVerifier::verify(&source(&body, 1, FLAGS), config(1, 1, &[0]));
            match unanimous(&edges, &outputs) {
                Some(decision) => {
                    outcomes[0] += 1;
                    let certificate = verified(result);
                    assert_eq!(certificate.transition_evidence.successors, edges);
                    assert_eq!(
                        certificate.transition_evidence.readouts,
                        outputs
                            .iter()
                            .copied()
                            .map(PspaceReadoutEvidence::Boolean)
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(certificate.recurrent_class_count, cycles(&edges).len());
                    assert_eq!(certificate.decision, decision);
                    certificate.check_structure().unwrap();
                }
                None => {
                    outcomes[1] += 1;
                    assert!(
                        matches!(result, ResultKind::Refuted { .. }),
                        "{body}: {result:?}"
                    );
                    assert!(result.to_json().contains("\"status\":\"refuted\""));
                }
            }
        }
    }
    assert_eq!(outcomes, [12, 4]);
}

#[test]
fn periodic_and_separately_recurrent_classes_require_more_than_fixed_points() {
    let periodic = source("0 ORACLE 1 XOR 0 PROPHECY 1 OUTPUT", 1, FLAGS);
    let certificate = verified(PspaceFamilyVerifier::verify(&periodic, config(1, 2, &[0])));
    let edges = vec![1, 0, 3, 2];
    assert_eq!(certificate.transition_evidence.successors, edges);
    assert!(edges
        .iter()
        .enumerate()
        .all(|(state, &next)| state != next as usize));
    assert_eq!(cycles(&edges), BTreeSet::from([vec![0, 1], vec![2, 3]]));
    assert_eq!(certificate.recurrent_class_count, 2);
    // Another recurrent class is never visited by the orbit from zero.
    let varying = source("0 ORACLE DUP 0 PROPHECY 1 SHR OUTPUT", 1, FLAGS);
    assert!(matches!(
        PspaceFamilyVerifier::verify(&varying, config(1, 2, &[0])),
        ResultKind::Refuted { .. }
    ));
}

#[derive(Clone, Copy)]
enum WordExpr {
    Constant(u64),
    Copy(usize),
    And(usize, usize),
    Or(usize, usize),
    Xor(usize, usize),
    AndConstant(usize, u64),
    OrConstant(usize, u64),
    XorConstant(usize, u64),
    Shr(usize, u64),
}

impl WordExpr {
    fn value(self, prior: &[u64]) -> u64 {
        match self {
            Self::Constant(value) => value,
            Self::Copy(index) => prior[index],
            Self::And(a, b) => prior[a] & prior[b],
            Self::Or(a, b) => prior[a] | prior[b],
            Self::Xor(a, b) => prior[a] ^ prior[b],
            Self::AndConstant(a, b) => prior[a] & b,
            Self::OrConstant(a, b) => prior[a] | b,
            Self::XorConstant(a, b) => prior[a] ^ b,
            Self::Shr(a, b) => prior[a] >> (b % 64),
        }
    }
    fn text(self) -> String {
        match self {
            Self::Constant(value) => value.to_string(),
            Self::Copy(a) => format!("{a} ORACLE"),
            Self::And(a, b) => format!("{a} ORACLE {b} ORACLE AND"),
            Self::Or(a, b) => format!("{a} ORACLE {b} ORACLE OR"),
            Self::Xor(a, b) => format!("{a} ORACLE {b} ORACLE XOR"),
            Self::AndConstant(a, b) => format!("{a} ORACLE {b} AND"),
            Self::OrConstant(a, b) => format!("{a} ORACLE {b} OR"),
            Self::XorConstant(a, b) => format!("{a} ORACLE {b} XOR"),
            Self::Shr(a, b) => format!("{a} ORACLE {b} SHR"),
        }
    }
}

fn decode(state: usize, cells: usize, bits: usize) -> Vec<u64> {
    (0..cells)
        .map(|cell| ((state >> (cell * bits)) & ((1 << bits) - 1)) as u64)
        .collect()
}

fn encode(words: &[u64], bits: usize) -> u32 {
    words
        .iter()
        .enumerate()
        .map(|(cell, &word)| (word as u32) << (cell * bits))
        .sum()
}

fn route(prior: &[u64], assignments: &[(usize, WordExpr)]) -> Vec<u64> {
    let mut present = vec![0; prior.len()];
    for &(target, expression) in assignments {
        present[target] = expression.value(prior);
    }
    present
}

#[test]
fn routing_gates_fresh_zero_and_last_write_match_word_oracle_and_explicit_circuit() {
    use WordExpr::*;
    // Every admitted expression, an overwritten target, and an unwritten
    // fourth cell. RHS values always read frozen prior state, never a previous
    // write. Shift 65 independently checks the word machine's modulo-64 rule.
    let assignments = [
        (0, Constant(3)),
        (1, Copy(0)),
        (2, And(0, 1)),
        (0, Or(1, 2)),
        (1, Xor(0, 2)),
        (2, AndConstant(1, 2)),
        (0, OrConstant(2, 1)),
        (1, XorConstant(0, 3)),
        (2, Shr(1, 65)),
    ];
    // Qualify EACH gate while its value survives into the final frame. The
    // composite fixture below alone cannot observe its overwritten writes.
    for &(_, expression) in &assignments {
        let isolated = source(
            &format!("INPUT {} 0 PROPHECY OUTPUT", expression.text()),
            3,
            FLAGS,
        );
        let generator =
            PspaceUniformFamilyGenerator::admit(&isolated, width(3), 2, BoundsPolicy::Error)
                .unwrap();
        let theorem = ProjectionFamilyCertificate::prove(generator.clone()).unwrap();
        let circuit = theorem.generate_circuit(1).unwrap();
        for input in [[0], [1]] {
            let certificate = verified(generator.specialize(&input, 64, 100).unwrap().verify());
            let edges: Vec<u32> = (0..64)
                .map(|state| expression.value(&decode(state, 3, 2)) as u32)
                .collect();
            assert_eq!(certificate.transition_evidence.successors, edges);
            assert_eq!(certificate.recurrent_class_count, cycles(&edges).len());
            for (state, &expected) in edges.iter().enumerate() {
                let bits: Vec<u8> = (0..6).map(|bit| ((state >> bit) & 1) as u8).collect();
                let (next, decision) = circuit.evaluate(&theorem, &input, &bits).unwrap();
                let actual: u32 = next
                    .iter()
                    .enumerate()
                    .map(|(bit, &value)| u32::from(value) << bit)
                    .sum();
                assert_eq!(actual, expected);
                assert_eq!(decision, input[0]);
            }
        }
    }
    let body = format!(
        "INPUT {} OUTPUT",
        assignments
            .iter()
            .map(|&(target, expression)| format!("{} {target} PROPHECY", expression.text()))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let program = source(&body, 4, FLAGS);
    let generator =
        PspaceUniformFamilyGenerator::admit(&program, width(4), 2, BoundsPolicy::Error).unwrap();
    let theorem = ProjectionFamilyCertificate::prove(generator.clone()).unwrap();
    for n in 1..=3 {
        let circuit = theorem.generate_circuit(n).unwrap();
        for input_number in 0..1usize << n {
            let input: Vec<u8> = (0..n)
                .map(|bit| ((input_number >> bit) & 1) as u8)
                .collect();
            let certificate = verified(generator.specialize(&input, 256, 1000).unwrap().verify());
            let edges: Vec<u32> = (0..256)
                .map(|state| encode(&route(&decode(state, 4, 2), &assignments), 2))
                .collect();
            assert_eq!(certificate.transition_evidence.successors, edges);
            assert_eq!(certificate.recurrent_class_count, cycles(&edges).len());
            assert_eq!(certificate.decision, input[0]);
            for (state, &expected_successor) in edges.iter().enumerate() {
                let prior_bits: Vec<u8> = (0..8).map(|bit| ((state >> bit) & 1) as u8).collect();
                let (next, decision) = circuit.evaluate(&theorem, &input, &prior_bits).unwrap();
                let actual: u32 = next
                    .iter()
                    .enumerate()
                    .map(|(bit, &value)| u32::from(value) << bit)
                    .sum();
                assert_eq!(actual, expected_successor);
                assert_eq!(decision, input[0]);
            }
            theorem.cross_check_finite(&circuit, &certificate).unwrap();
        }
    }
}

#[test]
fn polynomial_width_specialization_is_distinct_from_one_finite_input() {
    let program = parse(&format!(
        "FAMILY growth {{ CTC_CELLS POLY 1 1 1; CHRONOLOGY_BITS POLY 1000 1 0; \
         TRANSITION_STEPS POLY 20 1 0; {FLAGS} }} INPUT 0 ORACLE 0 PROPHECY OUTPUT"
    ))
    .unwrap();
    let polynomial = PolynomialBound {
        coefficient: 1,
        degree: 1,
        additive: 1,
    };
    let generator =
        PspaceUniformFamilyGenerator::admit(&program, polynomial, 1, BoundsPolicy::Error).unwrap();
    let theorem = ProjectionFamilyCertificate::prove(generator.clone()).unwrap();
    for n in 1..=3 {
        for number in 0..1usize << n {
            let input: Vec<u8> = (0..n).map(|bit| ((number >> bit) & 1) as u8).collect();
            let instance = generator.specialize(&input, 1 << (n + 1), 20).unwrap();
            assert_eq!(instance.generation.temporal_cells, n + 1);
            assert_eq!(
                instance.generation.generation_work_ceiling,
                generator.generation_work_bound.additive as u128 + n as u128
            );
            let certificate = verified(instance.verify());
            let edges: Vec<u32> = (0..1 << (n + 1)).map(|state| (state & 1) as u32).collect();
            assert_eq!(certificate.transition_evidence.successors, edges);
            assert_eq!(certificate.recurrent_class_count, 2);
            assert_eq!(theorem.decision(&input).unwrap(), input[0]);
            theorem
                .cross_check_finite(&theorem.generate_circuit(n as u64).unwrap(), &certificate)
                .unwrap();
        }
    }
    assert!(matches!(
        generator.specialize(&[], 4, 20),
        Err(UniformFamilyError::InvalidInputLength { .. })
    ));
    assert!(matches!(
        generator.specialize(&[2], 4, 20),
        Err(UniformFamilyError::NonBooleanInput { .. })
    ));
    let crossing = PolynomialBound {
        coefficient: 2,
        degree: 1,
        additive: 0,
    };
    // 2*n equals n+1 at n=1 but violates it for every n>1.
    assert!(matches!(
        PspaceUniformFamilyGenerator::admit(&program, crossing, 1, BoundsPolicy::Error),
        Err(UniformFamilyError::WidthRuleExceedsContract)
    ));
}

#[test]
fn resource_adjacencies_partiality_and_unsupported_theorems_keep_separate_statuses() {
    let program = source("0 ORACLE 0 PROPHECY 1 OUTPUT", 1, FLAGS);
    // Six explicit fetched records plus implicit main Return: 7. Numeric
    // workspace = input1 + decision1 + pc3 + stack(2*64) = 133 bits.
    let admitted = admit_program(&program, AdmissionConfig { memory_cells: 1 }).unwrap();
    let mut contract = PspaceFamilyContract::from(program.family_declaration.as_ref().unwrap());
    contract.transition_steps = width(7);
    contract.chronology_respecting_bits = width(133);
    let mut exact = config(1, 1, &[0]);
    exact.graph.max_instructions = 7;
    let certificate = verified(PspaceFamilyVerifier::verify_bytecode(
        &contract,
        admitted.program(),
        exact.clone(),
    ));
    assert_eq!(certificate.maximum_transition_steps, 7);
    assert_eq!(
        certificate.transition_evidence.instructions_executed,
        vec![7, 7]
    );
    assert_eq!(certificate.chronology_respecting_bits, 133);
    let mut short = exact.clone();
    short.graph.max_instructions = 6;
    assert!(matches!(
        PspaceFamilyVerifier::verify_bytecode(&contract, admitted.program(), short),
        ResultKind::Unknown { .. }
    ));
    let mut short = exact.clone();
    short.graph.max_states = 1;
    assert!(matches!(
        PspaceFamilyVerifier::verify_bytecode(&contract, admitted.program(), short),
        ResultKind::Unknown { .. }
    ));
    for field in 0..2 {
        let mut false_contract = contract.clone();
        if field == 0 {
            false_contract.transition_steps = width(6);
        } else {
            false_contract.chronology_respecting_bits = width(132);
        }
        assert!(matches!(
            PspaceFamilyVerifier::verify_bytecode(
                &false_contract,
                admitted.program(),
                exact.clone()
            ),
            ResultKind::Refuted { .. }
        ));
    }
    for body in [
        "INPUT POP INPUT 0 PROPHECY 1 OUTPUT",
        "0 ORACLE 1 ADD 0 PROPHECY 1 OUTPUT",
        "PARADOX",
        "2 OUTPUT",
    ] {
        assert!(
            matches!(
                PspaceFamilyVerifier::verify(&source(body, 1, FLAGS), config(1, 1, &[0])),
                ResultKind::Refuted { .. }
            ),
            "{body}"
        );
    }
    let general = source("0 ORACLE 0 ADD 0 PROPHECY 1 OUTPUT", 1, FLAGS);
    let generator =
        PspaceUniformFamilyGenerator::admit(&general, width(1), 1, BoundsPolicy::Error).unwrap();
    assert!(matches!(
        ProjectionFamilyCertificate::prove(generator.clone()),
        Err(UniformFamilyError::UnsupportedFamilyTheorem { .. })
    ));
    assert!(generator
        .specialize(&[0], 2, 100)
        .unwrap()
        .verify()
        .is_verified());
    let incomplete = source(
        "0 ORACLE 0 PROPHECY 1 OUTPUT",
        1,
        "UNIFORM; TOTAL; READOUT_INVARIANT; EFFECTS_FROZEN;",
    );
    let unknown = PspaceFamilyVerifier::verify(&incomplete, config(1, 1, &[0]));
    assert!(matches!(unknown, ResultKind::Unknown { .. }));
    assert!(unknown.to_json().contains("\"status\":\"unknown\""));
    let invalid = source(
        "PROCEDURE hidden PURE { 1 OUTPUT } 0 ORACLE 0 PROPHECY 1 OUTPUT",
        1,
        FLAGS,
    );
    let unsupported = PspaceFamilyVerifier::verify(&invalid, config(1, 1, &[0]));
    assert!(matches!(unsupported, ResultKind::Unsupported { .. }));
    assert!(unsupported.to_json().contains("\"status\":\"unsupported\""));
    assert!(matches!(
        PspaceUniformFamilyGenerator::admit(
            &source("1 WHILE { DUP } { 1 SUB } POP 1 OUTPUT", 1, FLAGS),
            width(1),
            1,
            BoundsPolicy::Error
        ),
        Err(UniformFamilyError::NonUniformTemplate { .. })
    ));
}

#[test]
fn retained_certificate_mutations_and_program_substitution_are_detected() {
    let program = source("0 ORACLE 1 XOR 0 PROPHECY 1 OUTPUT", 1, FLAGS);
    let generator =
        PspaceUniformFamilyGenerator::admit(&program, width(1), 1, BoundsPolicy::Error).unwrap();
    let theorem = ProjectionFamilyCertificate::prove(generator.clone()).unwrap();
    let certificate = verified(generator.specialize(&[0], 2, 100).unwrap().verify());
    let mut changed = (*certificate).clone();
    changed.transition_evidence.successors[0] = 0;
    assert!(changed.check_structure().is_err());
    let mut changed = (*certificate).clone();
    changed.transition_evidence.readouts[1] = PspaceReadoutEvidence::Boolean(0);
    assert!(changed.check_structure().is_err());
    let mut changed = generator.clone();
    changed.generation_work_bound.additive -= 1;
    assert!(changed.check_structure().is_err());
    let mut circuit = theorem.generate_circuit(1).unwrap();
    circuit.next_temporal[0] = ourochronos::ProjectionWire::Constant(false);
    assert!(circuit.check_structure(&theorem).is_err());
    let substitute = source("0 ORACLE 0 PROPHECY 1 OUTPUT", 1, FLAGS);
    let admitted = admit_program(&substitute, AdmissionConfig { memory_cells: 1 }).unwrap();
    assert!(certificate
        .recheck_bytecode(
            &certificate.contract,
            admitted.program(),
            config(1, 1, &[0])
        )
        .is_err());
    let finite_json = certificate.to_json();
    assert!(finite_json.contains("\"polynomial_time_uniform_declared\":true"));
    assert!(finite_json.contains("\"ideal_deutsch_selector_declared\":true"));
    let theorem_json = theorem.verification_json(&ResultKind::Verified(certificate));
    assert!(theorem_json.contains("\"proved_for_all_nonempty_inputs\""));
    assert!(theorem_json.contains("\"external_model_assumption\":\"ideal-deutsch-selector\""));
}

#[test]
fn cli_artifacts_preserve_finite_outcomes_and_external_assumptions() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let directory = std::env::temp_dir().join(format!(
        "ouro-family-conformance-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&directory).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(directory.clone());
    for (index, body, flags, state_limit, exit, label, status) in [
        (
            0,
            "0 ORACLE 0 PROPHECY 1 OUTPUT",
            FLAGS,
            "2",
            0,
            "VERIFIED FINITE FAMILY INSTANCE",
            "verified",
        ),
        (
            1,
            "0 ORACLE DUP 0 PROPHECY OUTPUT",
            FLAGS,
            "2",
            2,
            "REFUTED FAMILY INSTANCE",
            "refuted",
        ),
        (
            2,
            "0 ORACLE 0 PROPHECY 1 OUTPUT",
            FLAGS,
            "1",
            3,
            "UNKNOWN FAMILY INSTANCE",
            "unknown",
        ),
        (
            3,
            "0 ORACLE 0 PROPHECY 1 OUTPUT",
            "UNIFORM; TOTAL; READOUT_INVARIANT; EFFECTS_FROZEN;",
            "2",
            3,
            "UNKNOWN FAMILY INSTANCE",
            "unknown",
        ),
    ] {
        let path = directory.join(format!("{index}.ouro"));
        let artifact = directory.join(format!("{index}.json"));
        // Serialize fixtures directly: source() deliberately returns no AST
        // serializer and expectations are independent of compiler output.
        std::fs::write(&path, format!("FAMILY cli {{ CTC_CELLS POLY 0 1 1; CHRONOLOGY_BITS POLY 1000 1 0; TRANSITION_STEPS POLY 20 1 0; {flags} }} {body}")).unwrap();
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_ourochronos"))
            .arg(&path)
            .args([
                "--verify-family",
                "0",
                "--state-limit",
                state_limit,
                "--artifact",
            ])
            .arg(&artifact)
            .output()
            .unwrap();
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.status.code(), Some(exit), "{text}");
        assert!(text.contains(label), "{text}");
        let json = std::fs::read_to_string(artifact).unwrap();
        assert!(json.contains("\"finite_instance\":{"), "{json}");
        assert!(json.contains(&format!("\"status\":\"{status}\"")), "{json}");
        if index == 0 {
            assert!(text
                .contains("Nature's ideal Deutsch selector remains an external model assumption."));
            assert!(json.contains("\"family_theorem\""));
            assert!(json.contains("\"external_model_assumption\":\"ideal-deutsch-selector\""));
        }
    }
}
