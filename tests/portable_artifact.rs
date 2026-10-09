use ourochronos::linker::{
    link_with_metadata, LinkedProgram, ObjectVerificationArtifact, VerificationArtifactKind,
};
use ourochronos::portable_artifact::*;
use ourochronos::source::{SourceSpan, TextRange};
use ourochronos::{
    BytecodeProgram, HirProgram, Instruction, ObjectModule, PackageError, PackageManifest,
    PackageWitness, PortablePackage, SourceManager,
};
use sha2::{Digest, Sha256};

fn linked() -> LinkedProgram {
    let mut sources = SourceManager::new();
    let source = sources.add_virtual("fixture.ouro", "7 OUTPUT\n");
    let parsed = ourochronos::parser::parse("7 OUTPUT").unwrap();
    let hir = HirProgram::resolve(&parsed).unwrap();
    let mut code = BytecodeProgram::compile(&hir).unwrap();
    code.source_map = vec![
        ourochronos::SourceMapEntry {
            instruction: 0,
            span: SourceSpan::new(source, TextRange::new(0, 1)),
        },
        ourochronos::SourceMapEntry {
            instruction: 1,
            span: SourceSpan::new(source, TextRange::new(2, 8)),
        },
    ];
    let object = ObjectModule::from_compiled("fixture", &hir, code, &sources).unwrap();
    link_with_metadata(&[object]).unwrap()
}

fn artifact() -> PortableArtifact {
    PortableArtifact::from_linked(linked()).unwrap()
}

#[test]
fn runtime_diagnostics_retain_exact_record_span_and_gas_without_opening_source() {
    use ourochronos::{BytecodeVm, BytecodeVmConfig, BytecodeVmError, OpCode, PagedMemory};
    let mut linked = linked();
    linked.code.instructions[1] = Instruction::Primitive(OpCode::PresentRead);
    let manifest = linked.metadata.source_files.clone();
    let artifact = PortableArtifact::from_linked(linked).unwrap();
    let decoded = PortableArtifact::from_bytes(&artifact.to_bytes().unwrap()).unwrap();
    let memory = PagedMemory::with_size(4).unwrap();
    let vm = BytecodeVm::with_config(BytecodeVmConfig {
        memory_bounds: ourochronos::BoundsPolicy::Error,
        ..BytecodeVmConfig::default()
    });
    let error = vm.run_diagnostic(decoded.program(), &memory).unwrap_err();
    assert_eq!(error.instruction, Some(1));
    assert_eq!(error.instructions_executed, 2);
    assert_eq!(error.span.unwrap().range, TextRange::new(2, 8));
    assert_eq!(
        *error.error,
        BytecodeVmError::MemoryOutOfBounds {
            address: 7,
            memory_cells: 4
        }
    );
    assert_eq!(
        vm.run(decoded.program(), &memory).unwrap_err(),
        *error.error
    );
    let message = error.format_with_sources(&manifest);
    assert!(
        message.contains("source \"fixture.ouro\" bytes 2..8"),
        "{message}"
    );
    let mut malicious_name = manifest.clone();
    malicious_name[0].name = "fake\nERROR: forged\u{1b}[31m".into();
    let escaped = error.format_with_sources(&malicious_name);
    assert!(!escaped.contains('\n'));
    assert!(!escaped.contains('\u{1b}'));
    assert!(error
        .format_with_sources(&[])
        .contains("source 0 bytes 2..8"));
    let gas = BytecodeVm::with_config(BytecodeVmConfig {
        max_instructions: 1,
        ..BytecodeVmConfig::default()
    })
    .run_diagnostic(decoded.program(), &memory)
    .unwrap_err();
    assert_eq!(gas.instruction, Some(1));
    assert_eq!(gas.instructions_executed, 1);
    assert_eq!(*gas.error, BytecodeVmError::GasExhausted { limit: 1 });
    let mut invalid = decoded.program().clone();
    invalid.main.end = u32::MAX;
    let before = vm.run_diagnostic(&invalid, &memory).unwrap_err();
    assert_eq!(before.instruction, None);
    assert_eq!(before.span, None);
    assert_eq!(before.instructions_executed, 0);
}

fn reseal(bytes: &mut [u8]) {
    let end = bytes.len() - 32;
    let mut hash = Sha256::new();
    hash.update(b"ourochronos.portable-artifact/v1\0");
    hash.update(&bytes[..end]);
    bytes[end..].copy_from_slice(&hash.finalize());
}

fn provenance_start(bytes: &[u8]) -> usize {
    24 + u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize
}

fn first_source_length(bytes: &[u8]) -> usize {
    let start = provenance_start(bytes);
    let name_len = u32::from_le_bytes(bytes[start + 40..start + 44].try_into().unwrap()) as usize;
    start + 44 + name_len
}

#[test]
fn linked_sources_and_final_evidence_round_trip_deterministically() {
    let mut final_linked = linked();
    final_linked.metadata.verification = Some(ObjectVerificationArtifact {
        kind: VerificationArtifactKind::SolverCertificate,
        format_version: 1,
        payload: b"pre-link proof must not survive".to_vec(),
    });
    let mut artifact = PortableArtifact::from_linked(final_linked).unwrap();
    assert!(artifact.provenance.as_ref().unwrap().evidence.is_none());
    assert_eq!(
        artifact.provenance_status(),
        SourceProvenanceStatus::ManifestAvailable
    );
    artifact
        .attach_evidence(
            VerificationArtifactKind::BytecodeReport,
            1,
            b"new final-code claim".to_vec(),
        )
        .unwrap();
    let bytes = artifact.to_bytes().unwrap();
    let decoded = PortableArtifact::from_bytes(&bytes).unwrap();
    assert_eq!(decoded, artifact);
    assert_eq!(decoded.to_bytes().unwrap(), bytes);
    let source = &decoded.provenance.as_ref().unwrap().source_files[0];
    assert_eq!(source.name, "fixture.ouro");
    assert_eq!(source.byte_len, 9);
    assert_ne!(source.content_digest, 0);
    assert_eq!(
        decoded.program().source_map[1].span.range,
        TextRange::new(2, 8)
    );
    assert_eq!(
        decoded.legacy_bytes().unwrap(),
        linked().code.to_bytes().unwrap()
    );
}

#[test]
fn legacy_formats_and_synthetic_sources_remain_explicitly_unavailable() {
    let code = linked().code;
    let bytes = code.to_bytes().unwrap();
    let bare = PortableArtifact::from_bytes(&bytes).unwrap();
    assert_eq!(bare.program(), &code);
    assert!(bare.provenance.is_none());
    assert_eq!(
        bare.provenance_status(),
        SourceProvenanceStatus::Unavailable
    );
    let package = PortablePackage::new(PackageManifest::with_memory("old", 4), code).unwrap();
    let decoded = PortableArtifact::from_bytes(&package.to_bytes().unwrap()).unwrap();
    assert_eq!(decoded.package(), Some(&package));
    assert_eq!(
        decoded.provenance_status(),
        SourceProvenanceStatus::Unavailable
    );

    let mut synthetic = artifact();
    synthetic.provenance.as_mut().unwrap().source_files[0].content_digest = 0;
    assert_eq!(
        synthetic.provenance_status(),
        SourceProvenanceStatus::Unavailable
    );
    let decoded = PortableArtifact::from_bytes(&synthetic.to_bytes().unwrap()).unwrap();
    assert_eq!(
        decoded.provenance_status(),
        SourceProvenanceStatus::Unavailable
    );
    synthetic.provenance.as_mut().unwrap().source_files[0].content_digest = 123;
    synthetic.provenance.as_mut().unwrap().source_files[0].name = "<source:0>".into();
    assert_eq!(
        synthetic.provenance_status(),
        SourceProvenanceStatus::Unavailable
    );
    synthetic.provenance.as_mut().unwrap().source_files.clear();
    assert_eq!(
        synthetic.provenance_status(),
        SourceProvenanceStatus::Unavailable
    );
}

#[test]
fn every_single_byte_corruption_is_rejected() {
    let bytes = artifact().to_bytes().unwrap();
    for index in 0..bytes.len() {
        let mut corrupted = bytes.clone();
        corrupted[index] ^= 1;
        assert!(
            PortableArtifact::from_bytes(&corrupted).is_err(),
            "accepted byte {index}"
        );
    }
    for length in 0..bytes.len() {
        assert!(
            PortableArtifact::from_bytes(&bytes[..length]).is_err(),
            "accepted prefix {length}"
        );
    }
    let mut trailing = bytes;
    trailing.push(0);
    assert_eq!(
        PortableArtifact::from_bytes(&trailing),
        Err(PortableArtifactError::TrailingBytes)
    );
}

#[test]
fn valid_program_changes_cannot_reseal_stale_source_or_evidence_bindings() {
    let mut stale = artifact();
    stale
        .attach_evidence(VerificationArtifactKind::SolverCertificate, 1, vec![1])
        .unwrap();
    let original = stale.to_bytes().unwrap();
    let PortableArtifactPayload::Bytecode(code) = &mut stale.payload else {
        panic!()
    };
    code.instructions[0] = Instruction::PushWord(8);
    assert_eq!(
        stale.provenance_status(),
        SourceProvenanceStatus::Unavailable
    );
    assert_eq!(
        stale.to_bytes(),
        Err(PortableArtifactError::ProgramBindingMismatch)
    );
    let changed = stale.program().to_bytes().unwrap();
    let mut resealed = original;
    let end = provenance_start(&resealed);
    assert_eq!(changed.len(), end - 24);
    resealed[24..end].copy_from_slice(&changed);
    reseal(&mut resealed);
    assert_eq!(
        PortableArtifact::from_bytes(&resealed),
        Err(PortableArtifactError::ProgramBindingMismatch)
    );

    stale.provenance.as_mut().unwrap().program_digest = program_digest(stale.program()).unwrap();
    assert_eq!(
        stale.to_bytes(),
        Err(PortableArtifactError::EvidenceBindingMismatch)
    );
}

#[test]
fn source_ids_ranges_names_and_structural_checks_are_enforced() {
    let valid = artifact();
    let mut bad_id = valid.clone();
    bad_id.provenance.as_mut().unwrap().source_files[0].id = 1;
    assert!(matches!(
        bad_id.to_bytes(),
        Err(PortableArtifactError::InvalidProvenance(_))
    ));
    let mut absent_id = valid.clone();
    let PortableArtifactPayload::Bytecode(code) = &mut absent_id.payload else {
        panic!()
    };
    code.source_map[0].span.source = ourochronos::SourceId::new(1);
    absent_id.provenance.as_mut().unwrap().program_digest =
        program_digest(absent_id.program()).unwrap();
    assert!(matches!(
        absent_id.to_bytes(),
        Err(PortableArtifactError::InvalidProvenance(_))
    ));
    let mut short_source = valid.clone();
    short_source.provenance.as_mut().unwrap().source_files[0].byte_len = 7;
    assert!(matches!(
        short_source.to_bytes(),
        Err(PortableArtifactError::InvalidProvenance(_))
    ));
    for name in ["", "\0", "   "] {
        let mut bad = valid.clone();
        bad.provenance.as_mut().unwrap().source_files[0].name = name.into();
        assert!(matches!(
            bad.to_bytes(),
            Err(PortableArtifactError::InvalidProvenance(_))
        ));
    }
    let mut bad = valid;
    let PortableArtifactPayload::Bytecode(code) = &mut bad.payload else {
        panic!()
    };
    code.instructions[0] = Instruction::Primitive(ourochronos::OpCode::Swap);
    bad.provenance = None;
    assert!(matches!(
        bad.to_bytes(),
        Err(PortableArtifactError::BytecodeVerification(_))
    ));
}

#[test]
fn decoded_metadata_rejects_utf8_and_all_declared_table_bounds() {
    let good = artifact().to_bytes().unwrap();
    let start = provenance_start(&good);
    let cases = [
        (
            start + 32,
            (MAX_PORTABLE_SOURCE_FILES as u32 + 1).to_le_bytes(),
        ),
        (
            start + 40,
            (MAX_PORTABLE_SOURCE_NAME_BYTES as u32 + 1).to_le_bytes(),
        ),
        (20, (MAX_PORTABLE_PROVENANCE_BYTES as u32 + 1).to_le_bytes()),
    ];
    for (offset, value) in cases {
        let mut bad = good.clone();
        bad[offset..offset + 4].copy_from_slice(&value);
        reseal(&mut bad);
        assert!(matches!(
            PortableArtifact::from_bytes(&bad),
            Err(PortableArtifactError::LimitExceeded { .. })
        ));
    }
    let mut bad = good;
    bad[start + 44] = 0xff;
    reseal(&mut bad);
    assert_eq!(
        PortableArtifact::from_bytes(&bad),
        Err(PortableArtifactError::InvalidUtf8)
    );

    let mut oversized = artifact();
    let template = oversized.provenance.as_ref().unwrap().source_files[0].clone();
    oversized.provenance.as_mut().unwrap().source_files = (0..1024)
        .map(|id| {
            let mut source = template.clone();
            source.id = id;
            source.name = "x".repeat(MAX_PORTABLE_SOURCE_NAME_BYTES);
            source
        })
        .collect();
    assert!(matches!(
        oversized.to_bytes(),
        Err(PortableArtifactError::LimitExceeded {
            what: "provenance byte",
            ..
        })
    ));
    assert!(matches!(
        artifact().attach_evidence(
            VerificationArtifactKind::BytecodeReport,
            1,
            vec![0; MAX_PORTABLE_EVIDENCE_BYTES + 1]
        ),
        Err(PortableArtifactError::LimitExceeded {
            what: "evidence byte",
            ..
        })
    ));
}

#[test]
fn package_envelopes_keep_policies_and_check_sources_before_witness_replay() {
    let linked = linked();
    let manifest = PackageManifest::with_memory("packaged", 4);
    let execution = ourochronos::BytecodeVm::default()
        .run(
            &linked.code,
            &ourochronos::PagedMemory::with_size(4).unwrap(),
        )
        .unwrap();
    let witness = PackageWitness::replay_bound(
        &manifest,
        &linked.code,
        Vec::new(),
        execution.instructions_executed,
    )
    .unwrap();
    let package =
        PortablePackage::with_replay_witness(manifest, linked.code.clone(), witness).unwrap();
    let artifact = PortableArtifact::from_linked_package(linked, package.clone()).unwrap();
    let bytes = artifact.to_bytes().unwrap();
    let decoded = PortableArtifact::from_bytes(&bytes).unwrap();
    assert_eq!(decoded.package(), Some(&package));
    assert_eq!(decoded.to_bytes().unwrap(), bytes);

    // The payload now has a stale witness digest. A simultaneous source-map
    // inconsistency must fail at the provenance boundary before replay.
    let mut bad_witness = bytes;
    bad_witness[24 + 48] ^= 1;
    reseal(&mut bad_witness);
    assert!(matches!(
        PortableArtifact::from_bytes(&bad_witness),
        Err(PortableArtifactError::InvalidPackage(
            PackageError::InvalidWitness(_)
        ))
    ));
    let offset = first_source_length(&bad_witness);
    bad_witness[offset..offset + 8].copy_from_slice(&0u64.to_le_bytes());
    reseal(&mut bad_witness);
    assert!(matches!(
        PortableArtifact::from_bytes(&bad_witness),
        Err(PortableArtifactError::InvalidProvenance(_))
    ));
}

#[test]
fn package_constructor_rejects_a_different_linked_program() {
    let mut other = linked().code;
    other.instructions[0] = Instruction::PushWord(8);
    let package =
        PortablePackage::new(PackageManifest::with_memory("different", 4), other).unwrap();
    assert_eq!(
        PortableArtifact::from_linked_package(linked(), package),
        Err(PortableArtifactError::ProgramBindingMismatch)
    );
}
