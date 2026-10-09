//! Finite proof producer/checker conformance and adversarial certificate cases.

use ourochronos::bytecode::SourceMapEntry;
use ourochronos::finite_proof::{
    check_certificate, generate_certificate, FiniteNoFixedPointCertificate, FiniteProofConfig,
    FiniteProofError, FiniteProofQuery, FiniteProofScope, FiniteProofTerminal, FiniteTransition,
    FINITE_PROOF_SEMANTICS_VERSION, MAX_FINITE_PROOF_BYTES,
};
use ourochronos::hir::HirProgram;
use ourochronos::parser::parse;
use ourochronos::source::{SourceId, SourceSpan, TextRange};
use ourochronos::{
    admit_program, AdmissionConfig, BoundsPolicy, BytecodeProgram, BytecodeVm, Instruction, OpCode,
    PagedMemory, Value,
};
use sha2::{Digest, Sha256};

const QUERY: FiniteProofQuery = FiniteProofQuery::NoFixedPoint;

fn compile(source: &str, _memory: usize) -> BytecodeProgram {
    // Raw compiler output permits testing unsupported fragments before the
    // production source-region gate. The generator still verifies bytecode.
    BytecodeProgram::compile(&HirProgram::resolve(&parse(source).unwrap()).unwrap()).unwrap()
}

fn flip() -> BytecodeProgram {
    compile("TEMPORAL 0 1 BITS 1 { 0 ORACLE NOT 0 PROPHECY }", 16)
}

#[test]
fn unused_procedures_cannot_hide_a_reachable_capability() {
    let program = compile(
        "PROCEDURE display { 65 EMIT } TEMPORAL 0 1 BITS 1 { 0 ORACLE NOT 0 PROPHECY }",
        16,
    );
    let certificate = generate_certificate(&program, &config(), QUERY).unwrap();
    assert_eq!(
        check_certificate(&certificate, &program, &config(), QUERY)
            .unwrap()
            .enumerated_states,
        2
    );
    let reachable = compile(
        "PROCEDURE display { 65 EMIT } TEMPORAL 0 1 BITS 1 { display 0 ORACLE NOT 0 PROPHECY }",
        16,
    );
    assert!(matches!(
        generate_certificate(&reachable, &config(), QUERY),
        Err(FiniteProofError::Unsupported(_))
    ));
}

#[test]
fn portable_proof_claim_requires_semantic_recheck_after_reload() {
    use ourochronos::{PortableArtifact, VerificationArtifactKind};
    let program = flip();
    let certificate = generate_certificate(&program, &config(), QUERY).unwrap();
    let mut artifact = PortableArtifact::from_bytecode(program.clone()).unwrap();
    artifact
        .attach_evidence(
            VerificationArtifactKind::SolverCertificate,
            1,
            certificate.to_bytes().unwrap(),
        )
        .unwrap();
    let restored = PortableArtifact::from_bytes(&artifact.to_bytes().unwrap()).unwrap();
    let payload = &restored
        .provenance
        .as_ref()
        .unwrap()
        .evidence
        .as_ref()
        .unwrap()
        .payload;
    let restored_proof = FiniteNoFixedPointCertificate::from_bytes(payload).unwrap();
    check_certificate(&restored_proof, restored.program(), &config(), QUERY).unwrap();

    let mut false_claim = certificate;
    false_claim.rows[0].present[0] = 0;
    // A producer can serialize and integrity-protect a false claim. Only the
    // independent interpreter can distinguish it from a proof.
    artifact
        .attach_evidence(
            VerificationArtifactKind::SolverCertificate,
            1,
            false_claim.to_bytes().unwrap(),
        )
        .unwrap();
    let restored = PortableArtifact::from_bytes(&artifact.to_bytes().unwrap()).unwrap();
    let payload = &restored
        .provenance
        .as_ref()
        .unwrap()
        .evidence
        .as_ref()
        .unwrap()
        .payload;
    let restored_proof = FiniteNoFixedPointCertificate::from_bytes(payload).unwrap();
    assert_eq!(
        check_certificate(&restored_proof, restored.program(), &config(), QUERY),
        Err(FiniteProofError::ObservationMismatch { row: 0 })
    );
}

fn config() -> FiniteProofConfig {
    FiniteProofConfig::default()
}

// This table is hand-built without generation or any VM execution.
fn hand_flip(program: &BytecodeProgram) -> FiniteNoFixedPointCertificate {
    FiniteNoFixedPointCertificate {
        semantics_version: FINITE_PROOF_SEMANTICS_VERSION,
        program_sha256: Sha256::digest(program.to_bytes().unwrap()).into(),
        config: config(),
        query: QUERY,
        scope: FiniteProofScope {
            cells: 1,
            cell_bits: 1,
        },
        rows: [1, 0]
            .into_iter()
            .map(|word| FiniteTransition {
                present: vec![word],
                stack: vec![],
                output: vec![],
                terminal: FiniteProofTerminal::Finished,
                instructions_executed: 8,
            })
            .collect(),
    }
}

fn rechecksum(bytes: &mut Vec<u8>) {
    bytes.truncate(bytes.len() - 32);
    let checksum: [u8; 32] = Sha256::digest(&*bytes).into();
    bytes.extend_from_slice(&checksum);
}

#[test]
fn independently_supplied_complete_table_checks_and_serializes_deterministically() {
    let program = flip();
    assert_eq!(
        admit_program(
            &parse("TEMPORAL 0 1 BITS 1 { 0 ORACLE NOT 0 PROPHECY }").unwrap(),
            AdmissionConfig { memory_cells: 16 }
        )
        .unwrap()
        .program(),
        &program
    );
    let hand = hand_flip(&program);
    let checked = check_certificate(&hand, &program, &config(), QUERY).unwrap();
    assert_eq!(checked.enumerated_states, 2);
    assert_eq!(
        generate_certificate(&program, &config(), QUERY).unwrap(),
        hand
    );
    let bytes = hand.to_bytes().unwrap();
    let decoded = FiniteNoFixedPointCertificate::from_bytes(&bytes).unwrap();
    assert_eq!(decoded, hand);
    assert_eq!(decoded.to_bytes().unwrap(), bytes);
    check_certificate(&decoded, &program, &config(), QUERY).unwrap();
}

#[test]
fn true_fixed_states_and_forged_no_fixed_state_tables_are_rejected() {
    for body in ["0 ORACLE 0 PROPHECY", "1 0 PROPHECY", "0 0 PROPHECY", "NOP"] {
        let program = compile(&format!("TEMPORAL 0 1 BITS 1 {{ {body} }}"), 16);
        assert!(matches!(
            generate_certificate(&program, &config(), QUERY),
            Err(FiniteProofError::FixedPointFound { .. })
        ));
    }
    let identity = compile("TEMPORAL 0 1 BITS 1 { 0 ORACLE 0 PROPHECY }", 16);
    let mut forged = hand_flip(&identity);
    forged
        .rows
        .iter_mut()
        .for_each(|row| row.instructions_executed = 7);
    assert_eq!(
        check_certificate(&forged, &identity, &config(), QUERY),
        Err(FiniteProofError::ObservationMismatch { row: 0 })
    );
    // An accurate complete transition table still cannot prove a false query.
    forged.rows[0].present[0] = 0;
    forged.rows[1].present[0] = 1;
    assert_eq!(
        check_certificate(&forged, &identity, &config(), QUERY),
        Err(FiniteProofError::FixedPointFound { row: 0 })
    );
}

#[test]
fn every_retained_observation_and_domain_row_is_checked() {
    let program = flip();
    let original = hand_flip(&program);
    for change in 0..6 {
        let mut changed = original.clone();
        match change {
            0 => changed.rows[0].present[0] = 0,
            1 => changed.rows[0].stack.push(7),
            2 => changed.rows[0].output.push(9),
            3 => changed.rows[0].terminal = FiniteProofTerminal::Halted,
            4 => changed.rows[0].instructions_executed -= 1,
            5 => changed.rows.swap(0, 1),
            _ => unreachable!(),
        }
        assert_eq!(
            check_certificate(&changed, &program, &config(), QUERY),
            Err(FiniteProofError::ObservationMismatch { row: 0 })
        );
        let reloaded =
            FiniteNoFixedPointCertificate::from_bytes(&changed.to_bytes().unwrap()).unwrap();
        assert!(check_certificate(&reloaded, &program, &config(), QUERY).is_err());
    }
    let mut incomplete = original.clone();
    incomplete.rows.pop();
    assert!(matches!(
        check_certificate(&incomplete, &program, &config(), QUERY),
        Err(FiniteProofError::InvalidEncoding(_))
    ));
    let mut extra = original.clone();
    extra.rows.push(extra.rows[0].clone());
    assert!(extra.to_bytes().is_err());
}

#[test]
fn program_scope_semantics_bounds_and_all_resource_fields_bind_exactly() {
    let program = flip();
    let certificate = hand_flip(&program);
    let neighbor = compile("TEMPORAL 0 1 BITS 1 { 0 ORACLE 1 XOR 0 PROPHECY }", 16);
    assert!(matches!(
        check_certificate(&certificate, &neighbor, &config(), QUERY),
        Err(FiniteProofError::BindingMismatch(_))
    ));
    let mut source_location = program.clone();
    source_location.source_map.push(SourceMapEntry {
        instruction: 0,
        span: SourceSpan::new(SourceId::new(0), TextRange::new(0, 1)),
    });
    assert!(matches!(
        check_certificate(&certificate, &source_location, &config(), QUERY),
        Err(FiniteProofError::BindingMismatch(_))
    ));
    for field in 0..8 {
        let mut expected = config();
        match field {
            0 => expected.memory_cells += 1,
            1 => expected.memory_bounds = BoundsPolicy::Clamp,
            2 => expected.max_instructions += 1,
            3 => expected.max_call_depth -= 1,
            4 => expected.max_stack_depth += 1,
            5 => expected.max_temporal_depth = 0,
            6 => expected.max_output_items += 1,
            7 => expected.max_output_bytes += 1,
            _ => unreachable!(),
        }
        assert_eq!(
            check_certificate(&certificate, &program, &expected, QUERY),
            Err(FiniteProofError::BindingMismatch("resource configuration"))
        );
        let mut mutated = certificate.clone();
        mutated.config = expected;
        assert_eq!(
            check_certificate(&mutated, &program, &config(), QUERY),
            Err(FiniteProofError::BindingMismatch("resource configuration"))
        );
    }
    let mut changed = certificate.clone();
    changed.scope.cell_bits = 2;
    assert_eq!(
        check_certificate(&changed, &program, &config(), QUERY),
        Err(FiniteProofError::BindingMismatch("scope"))
    );
    changed = certificate;
    changed.semantics_version += 1;
    assert_eq!(
        check_certificate(&changed, &program, &config(), QUERY),
        Err(FiniteProofError::BindingMismatch("semantics version"))
    );
}

#[test]
fn corruption_unknown_query_tags_truncation_counts_and_stale_bindings_reject() {
    let program = flip();
    let bytes = hand_flip(&program).to_bytes().unwrap();
    for index in [0, 8, 10, 12, 14, 46, 50, 83, 85, 89, 98, bytes.len() - 1] {
        let mut bad = bytes.clone();
        bad[index] ^= 1;
        assert_eq!(
            FiniteNoFixedPointCertificate::from_bytes(&bad),
            Err(FiniteProofError::CorruptEncoding)
        );
    }
    // Recompute the checksum to show that integrity is not proof authority.
    for (index, value) in [
        (0, b'X'),
        (8, 2),
        (10, 2),
        (12, 1),
        (13, 1),
        (50, 3),
        (83, 13),
        (84, 13),
        (85, 3),
        (89, 2),
        (106, 255),
        (110, 255),
    ] {
        let mut bad = bytes.clone();
        bad[index] = value;
        rechecksum(&mut bad);
        assert!(
            FiniteNoFixedPointCertificate::from_bytes(&bad).is_err(),
            "field {index}"
        );
    }
    for cut in [0, 8, 31, 32, 89, bytes.len() - 1] {
        assert!(FiniteNoFixedPointCertificate::from_bytes(&bytes[..cut]).is_err());
    }
    let mut missing_rows = bytes.clone();
    missing_rows.drain(114..139); // Remove the second row, preserve/recompute checksum.
    rechecksum(&mut missing_rows);
    assert!(FiniteNoFixedPointCertificate::from_bytes(&missing_rows).is_err());
    let mut trailing = bytes.clone();
    trailing.insert(trailing.len() - 32, 0);
    rechecksum(&mut trailing);
    assert!(FiniteNoFixedPointCertificate::from_bytes(&trailing).is_err());
    for index in [14, 46] {
        let mut stale = bytes.clone();
        stale[index] ^= 1;
        rechecksum(&mut stale);
        let decoded = FiniteNoFixedPointCertificate::from_bytes(&stale).unwrap();
        assert!(matches!(
            check_certificate(&decoded, &program, &config(), QUERY),
            Err(FiniteProofError::BindingMismatch(_))
        ));
    }
    assert!(matches!(
        FiniteNoFixedPointCertificate::from_bytes(&vec![0; MAX_FINITE_PROOF_BYTES + 1]),
        Err(FiniteProofError::ResourceLimit(_))
    ));
}

#[test]
fn numeric_semantics_cover_wrapping_signed_zero_division_shifts_and_stack_controls() {
    let cases: &[(&str, &[u64])] = &[
        ("18446744073709551615 1 ADD OUTPUT", &[0]),
        ("0 1 SUB OUTPUT", &[u64::MAX]),
        ("18446744073709551615 2 MUL OUTPUT", &[u64::MAX - 1]),
        ("7 0 DIV OUTPUT 7 0 MOD OUTPUT 7 3 DIV OUTPUT 7 3 MOD OUTPUT", &[0, 0, 2, 1]),
        ("1 NEG OUTPUT 9223372036854775808 ABS OUTPUT 18446744073709551615 SIGN OUTPUT", &[u64::MAX, 1 << 63, u64::MAX]),
        ("0 SIGN OUTPUT 7 SIGN OUTPUT 9 3 MIN OUTPUT 9 3 MAX OUTPUT", &[0, 1, 3, 9]),
        ("0 NOT OUTPUT 7 NOT OUTPUT 6 3 AND OUTPUT 6 3 OR OUTPUT 6 3 XOR OUTPUT", &[1, 0, 2, 7, 5]),
        ("1 64 SHL OUTPUT 8 65 SHR OUTPUT 9223372036854775808 1 SHL OUTPUT", &[1, 4, 0]),
        ("4 4 EQ OUTPUT 4 5 NEQ OUTPUT 4 5 LT OUTPUT 5 4 GT OUTPUT 5 5 LTE OUTPUT 5 5 GTE OUTPUT", &[1, 1, 1, 1, 1, 1]),
        ("18446744073709551615 0 SLT OUTPUT 0 18446744073709551615 SGT OUTPUT 0 0 SLTE OUTPUT 0 0 SGTE OUTPUT", &[1, 1, 1, 1]),
        ("5 POP NOP 7 DUP OUTPUT OUTPUT", &[7, 7]),
        ("1 2 SWAP OUTPUT OUTPUT 1 2 OVER OUTPUT OUTPUT OUTPUT", &[1, 2, 1, 2, 1]),
        ("1 2 3 ROT OUTPUT OUTPUT OUTPUT 11 22 DEPTH OUTPUT POP POP", &[1, 3, 2, 2]),
        ("11 22 1 PICK OUTPUT OUTPUT OUTPUT 11 22 1 ROLL OUTPUT OUTPUT", &[11, 22, 11, 11, 22]),
        ("1 2 2 REVERSE OUTPUT OUTPUT 1 2 0 REVERSE OUTPUT OUTPUT", &[1, 2, 2, 1]),
        ("18446744073709551615 0 PROPHECY 0 PRESENT OUTPUT", &[1]),
    ];
    for &(body, expected_output) in cases {
        let program = compile(
            &format!("TEMPORAL 0 1 BITS 1 {{ {body} 0 ORACLE NOT 0 PROPHECY }}"),
            16,
        );
        let certificate = generate_certificate(&program, &config(), QUERY).unwrap();
        for row in &certificate.rows {
            assert_eq!(row.output, expected_output, "{body}");
            assert!(row.stack.is_empty(), "{body}");
        }
        check_certificate(&certificate, &program, &config(), QUERY).unwrap();
    }
}

#[test]
fn forward_branches_acyclic_calls_halt_and_instruction_costs_match() {
    let source = "PROCEDURE flip { 0 ORACLE NOT 0 PROPHECY } PROCEDURE via { flip } \
                  TEMPORAL 0 1 BITS 1 { 0 ORACLE IF { 9 OUTPUT } ELSE { 8 OUTPUT } via }";
    let program = compile(source, 16);
    let certificate = generate_certificate(&program, &config(), QUERY).unwrap();
    assert_eq!(certificate.rows[0].output, [8]);
    assert_eq!(certificate.rows[1].output, [9]);
    // Both calls and both procedure RETURNs count; the true arm also fetches JUMP.
    assert_eq!(certificate.rows[0].instructions_executed, 17);
    assert_eq!(certificate.rows[1].instructions_executed, 18);
    let no_else = compile(
        "TEMPORAL 0 1 BITS 1 { 0 ORACLE IF { 7 OUTPUT } 0 ORACLE NOT 0 PROPHECY }",
        16,
    );
    let c = generate_certificate(&no_else, &config(), QUERY).unwrap();
    assert!(c.rows[0].output.is_empty());
    assert_eq!(c.rows[1].output, [7]);
    let halt = compile(
        "TEMPORAL 0 1 BITS 1 { 0 ORACLE NOT 0 PROPHECY HALT NOP }",
        16,
    );
    let c = generate_certificate(&halt, &config(), QUERY).unwrap();
    assert!(c
        .rows
        .iter()
        .all(|r| r.terminal == FiniteProofTerminal::Halted && r.instructions_executed == 7));
    let mut exact = config();
    exact.max_instructions = 8;
    generate_certificate(&flip(), &exact, QUERY).unwrap();
    for gas in [0, 6, 7] {
        exact.max_instructions = gas;
        assert!(matches!(
            generate_certificate(&flip(), &exact, QUERY),
            Err(FiniteProofError::Execution { .. })
        ));
    }
    exact = config();
    exact.max_call_depth = 0;
    assert!(matches!(
        generate_certificate(&program, &exact, QUERY),
        Err(FiniteProofError::Execution { .. })
    ));
    exact = config();
    exact.max_stack_depth = 0;
    assert!(matches!(
        generate_certificate(&flip(), &exact, QUERY),
        Err(FiniteProofError::Execution { .. })
    ));
    exact = config();
    exact.max_output_items = 0;
    assert!(matches!(
        generate_certificate(&program, &exact, QUERY),
        Err(FiniteProofError::Execution { .. })
    ));
}

#[test]
fn scope_masking_zero_frame_and_each_address_policy_are_explicit() {
    for (bounds, address, toggled) in [
        (BoundsPolicy::Wrap, 2u64, 0),
        (BoundsPolicy::Clamp, u64::MAX, 1),
    ] {
        let mut cfg = config();
        cfg.memory_bounds = bounds;
        let program = compile(
            &format!("TEMPORAL 0 2 BITS 1 {{ {address} ORACLE 1 XOR {address} PROPHECY }}"),
            16,
        );
        let c = generate_certificate(&program, &cfg, QUERY).unwrap();
        assert_eq!(c.rows.len(), 4); // The other fourteen cells remain statically zero.
        for (index, row) in c.rows.iter().enumerate() {
            assert_eq!(row.present[toggled], (((index >> toggled) & 1) ^ 1) as u64);
            assert_eq!(row.present[1 - toggled], 0);
        }
        check_certificate(&c, &program, &cfg, QUERY).unwrap();
    }
    for body in ["2 ORACLE NOT 0 PROPHECY", "0 ORACLE NOT 2 PROPHECY"] {
        let program = compile(&format!("TEMPORAL 0 2 BITS 1 {{ {body} }}"), 16);
        assert!(matches!(
            generate_certificate(&program, &config(), QUERY),
            Err(FiniteProofError::Execution { .. })
        ));
    }
    // A full-word fixed state must lie in the masked domain: all writes mask,
    // even when arithmetic/literals range outside the finite cell width.
    let program = compile("TEMPORAL 0 1 BITS 2 { 0 ORACLE 5 ADD 0 PROPHECY }", 16);
    let c = generate_certificate(&program, &config(), QUERY).unwrap();
    assert_eq!(
        c.rows.iter().map(|r| r.present[0]).collect::<Vec<_>>(),
        [1, 2, 3, 0]
    );
    // Finite VM conformance neighbors exercise arbitrary words beyond the
    // enumerated mask and nonzero anamnesis outside the scope. Present still
    // starts fresh, writes mask, and the outside frame remains zero.
    let vm = BytecodeVm::with_config(config().vm_config());
    for scoped in [4, 1 << 63, u64::MAX] {
        let mut input = PagedMemory::with_size(16).unwrap();
        input.write(0, Value::new(scoped)).unwrap();
        for address in 1..16 {
            input.write(address, Value::new(u64::MAX)).unwrap();
        }
        let observed = vm.run(&program, &input).unwrap();
        assert_eq!(
            observed.present.get(0).unwrap().val,
            scoped.wrapping_add(5) & 3
        );
        for address in 1..16 {
            assert_eq!(observed.present.get(address).unwrap().val, 0);
        }
    }
    let mut illegal_frame_claim = c;
    illegal_frame_claim.rows[0].present.push(1);
    assert!(matches!(
        check_certificate(&illegal_frame_claim, &program, &config(), QUERY),
        Err(FiniteProofError::InvalidEncoding(_))
    ));
}

#[test]
fn numeric_output_byte_profile_accounts_for_scoped_dependencies() {
    let program = compile(
        "TEMPORAL 0 2 BITS 1 { 0 ORACLE 1 ORACLE ADD OUTPUT 0 ORACLE NOT 0 PROPHECY }",
        16,
    );
    let mut cfg = config();
    cfg.max_output_items = 1;
    cfg.max_output_bytes = 128;
    let c = generate_certificate(&program, &cfg, QUERY).unwrap();
    assert_eq!(
        c.rows.iter().map(|r| r.output[0]).collect::<Vec<_>>(),
        [0, 1, 1, 2]
    );
    check_certificate(&c, &program, &cfg, QUERY).unwrap();
    cfg.max_output_bytes = 127;
    assert!(matches!(
        generate_certificate(&program, &cfg, QUERY),
        Err(FiniteProofError::Unsupported(_))
    ));
}

#[test]
fn finite_domain_boundary_and_resource_ceiling_neighbors_are_bounded() {
    let program = compile("TEMPORAL 0 1 BITS 12 { 0 ORACLE 1 ADD 0 PROPHECY }", 16);
    let mut cfg = config();
    cfg.max_instructions = 9;
    let c = generate_certificate(&program, &cfg, QUERY).unwrap();
    assert_eq!(c.rows.len(), 4096);
    assert_eq!(c.rows[4095].present, [0]);
    let bytes = c.to_bytes().unwrap();
    check_certificate(
        &FiniteNoFixedPointCertificate::from_bytes(&bytes).unwrap(),
        &program,
        &cfg,
        QUERY,
    )
    .unwrap();
    for source in [
        "TEMPORAL 0 1 BITS 13 { 0 ORACLE 1 ADD 0 PROPHECY }",
        "TEMPORAL 0 13 BITS 1 { 0 ORACLE NOT 0 PROPHECY }",
    ] {
        let p = compile(source, 16);
        assert!(matches!(
            generate_certificate(&p, &cfg, QUERY),
            Err(FiniteProofError::Unsupported(_))
        ));
    }
    let mut costly = cfg.clone();
    costly.max_instructions = 4096;
    assert!(matches!(
        generate_certificate(&program, &costly, QUERY),
        Err(FiniteProofError::ResourceLimit(_))
    ));
    costly = cfg;
    costly.max_stack_depth = 4096;
    costly.max_output_items = 4096;
    assert!(matches!(
        generate_certificate(&program, &costly, QUERY),
        Err(FiniteProofError::ResourceLimit(_))
    ));
}

#[test]
fn unsupported_capabilities_nested_scopes_loops_and_recursion_never_become_proofs() {
    for source in [
        "0 ORACLE NOT 0 PROPHECY",
        "TEMPORAL 0 1 BITS 1 { 0 ORACLE NOT 0 PROPHECY } 1 POP",
        "TEMPORAL 1 1 BITS 1 { 0 ORACLE NOT 0 PROPHECY }",
        "TEMPORAL 0 1 BITS 1 { TEMPORAL 0 1 BITS 1 { 0 ORACLE NOT 0 PROPHECY } }",
        "TEMPORAL 0 1 BITS 1 { WHILE { 0 } { NOP } 0 ORACLE NOT 0 PROPHECY }",
        "TEMPORAL 0 1 BITS 1 { [ 1 ] POP 0 ORACLE NOT 0 PROPHECY }",
        "PROCEDURE recur { recur } TEMPORAL 0 1 BITS 1 { recur }",
    ] {
        let p = compile(source, 16);
        assert!(
            generate_certificate(&p, &config(), QUERY).is_err(),
            "{source}"
        );
    }
    for opcode in [
        OpCode::Input,
        OpCode::Clock,
        OpCode::Random,
        OpCode::Emit,
        OpCode::VecNew,
        OpCode::Assert,
        OpCode::Pack,
        OpCode::Unpack,
        OpCode::Index,
        OpCode::Store,
        OpCode::Paradox,
        OpCode::Exec,
        OpCode::StrRev,
    ] {
        let mut p = flip();
        p.instructions[3] = Instruction::Primitive(opcode);
        assert!(
            matches!(
                generate_certificate(&p, &config(), QUERY),
                Err(FiniteProofError::Unsupported(_))
            ),
            "{opcode:?}"
        );
    }
    let mut malformed = flip();
    malformed.instructions[0] = Instruction::TemporalEnter {
        base: 0,
        size: 1,
        cell_bits: 1,
        exit_target: 7,
    };
    assert!(generate_certificate(&malformed, &config(), QUERY).is_err());
    let mut underflow = flip();
    underflow.instructions[1] = Instruction::Primitive(OpCode::Pop);
    assert!(matches!(
        generate_certificate(&underflow, &config(), QUERY),
        Err(FiniteProofError::Execution { .. })
    ));
}
