//! Actual safe-host dispatch, bounded codec/approval and retired-package guards.

use ourochronos::bytecode::{
    CodeRange, ForeignEffects, ForeignEntry, ForeignScalarType, Instruction,
};
use ourochronos::core::OuroResult;
use ourochronos::hir::{ForeignId, HirProgram};
use ourochronos::package::{PackageError, PackageManifest, PortablePackage};
use ourochronos::parser::parse;
use ourochronos::runtime::ffi::{
    DynamicLibraryManager, ExtendedFFIContext, ForeignHostError, ForeignHostTable,
};
use ourochronos::runtime::host_manifest::*;
use ourochronos::{BytecodeProgram, BytecodeVm, BytecodeVmConfig, BytecodeVmError, PagedMemory};
use sha2::{Digest, Sha256};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

fn id(index: usize) -> ForeignId {
    ForeignId::try_from_index(index).unwrap()
}
fn compile(source: &str) -> BytecodeProgram {
    BytecodeProgram::compile(&HirProgram::resolve(&parse(source).unwrap()).unwrap()).unwrap()
}
fn program() -> BytecodeProgram {
    compile("FOREIGN \"embedding\" { PROC combine ( a: u64 b: i64 -- result: i64 ) PURE; PROC observe ( word: u64 -- ) PURE; } INPUT 18446744073709551615 combine DUP observe")
}
fn row(foreign: usize, args: &[u64], result: Option<u64>) -> FrozenScalarObservation {
    FrozenScalarObservation {
        foreign: id(foreign),
        arguments: args.to_vec(),
        result,
    }
}
fn manifest(program: &BytecodeProgram) -> HostManifest {
    HostManifest::new(
        program,
        vec![7],
        vec![row(1, &[6], None), row(0, &[7, u64::MAX], Some(6))],
    )
    .unwrap()
}
fn reseal(bytes: &mut [u8]) {
    let end = bytes.len() - 32;
    let mut digest = Sha256::new();
    digest.update(b"ourochronos.host-manifest/v1\0");
    digest.update(&bytes[..end]);
    bytes[end..].copy_from_slice(&digest.finalize());
}

#[test]
fn roundtrip_approved_snapshot_drives_actual_vm_and_outlives_original_owners() {
    let p = program();
    let m = manifest(&p);
    let approved = m.digest().unwrap();
    let bytes = m.to_bytes().unwrap();
    let decoded = HostManifest::from_bytes(&bytes).unwrap();
    assert_eq!(decoded, m);
    assert_eq!(decoded.to_bytes().unwrap(), bytes);
    let checked = decoded.check(&p, approved).unwrap();
    assert_eq!(checked.approved_digest(), approved);
    assert_eq!(checked.lookup(id(0), &[7, u64::MAX]).unwrap(), Some(6));
    assert_eq!(checked.lookup(id(1), &[6]).unwrap(), None);
    let input = checked.input().to_vec();
    let table = Arc::new(checked.frozen_host_table().unwrap());
    let executable = p.clone();
    drop(checked);
    drop(m);
    drop(p);
    drop(bytes);
    let vm = BytecodeVm::with_config(BytecodeVmConfig {
        input,
        max_instructions: 8,
        ..BytecodeVmConfig::default()
    })
    .with_foreign_host(table);
    let result = vm
        .run(&executable, &PagedMemory::with_size(8).unwrap())
        .unwrap();
    assert_eq!(
        result.stack.iter().map(|word| word.val).collect::<Vec<_>>(),
        [6]
    );
    assert_eq!(result.inputs_consumed, [7]);
    assert_eq!(result.instructions_executed, 6);
    assert!(result.effects.is_empty());
    assert!(matches!(
        PortablePackage::new(PackageManifest::current("host"), executable),
        Err(PackageError::UnsupportedForeignCall)
    ));
}

#[test]
fn arity_and_invalid_host_results_fail_before_or_after_callbacks_as_declared() {
    let p = program();
    let descriptor = p.foreigns[0].clone();
    let calls = Arc::new(AtomicUsize::new(0));
    let count = calls.clone();
    let mut table = ForeignHostTable::new();
    table
        .bind(descriptor.clone(), move |_| {
            count.fetch_add(1, Ordering::SeqCst);
            Ok(Some(3))
        })
        .unwrap();
    for args in [vec![], vec![7], vec![7, 8, 9]] {
        assert!(
            matches!(table.call(&descriptor, &args), Err(ForeignHostError::ArgumentMismatch { expected: 2, got, .. }) if got == args.len())
        );
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(table.call(&descriptor, &[u64::MAX, 1]).unwrap(), Some(3));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let mut wrong = descriptor.clone();
    wrong.parameters[0] = ForeignScalarType::I64;
    assert!(matches!(
        table.call(&wrong, &[1, 2]),
        Err(ForeignHostError::SignatureMismatch { .. })
    ));
    let mut missing = ForeignHostTable::new();
    missing.bind(descriptor.clone(), |_| Ok(None)).unwrap();
    assert!(matches!(
        missing.call(&descriptor, &[1, 2]),
        Err(ForeignHostError::ReturnMismatch {
            expected_value: true,
            ..
        })
    ));
    let void = p.foreigns[1].clone();
    let mut extra = ForeignHostTable::new();
    extra.bind(void.clone(), |_| Ok(Some(u64::MAX))).unwrap();
    assert!(matches!(
        extra.call(&void, &[1]),
        Err(ForeignHostError::ReturnMismatch {
            expected_value: false,
            ..
        })
    ));
    let mut declined = ForeignHostTable::new();
    declined
        .bind(descriptor.clone(), |_| Err("declined".into()))
        .unwrap();
    assert!(matches!(
        declined.call(&descriptor, &[1, 2]),
        Err(ForeignHostError::CallbackFailed { .. })
    ));
}

#[test]
fn binding_rejects_program_descriptor_and_resealed_snapshot_substitutions() {
    let p = program();
    let m = manifest(&p);
    let approved = m.digest().unwrap();
    let mut other = p.clone();
    other.instructions[0] = Instruction::PushWord(7);
    assert!(matches!(
        m.clone().check(&other, approved),
        Err(HostManifestError::ProgramMismatch)
    ));
    let mut changed = m.clone();
    changed.foreigns[0].symbol.push_str("_alternate");
    assert!(matches!(
        changed.check(&p, approved),
        Err(HostManifestError::DescriptorMismatch)
    ));
    let mut missing = m.clone();
    missing.foreigns.pop();
    missing.observations.retain(|row| row.foreign == id(0));
    assert!(matches!(
        missing.check(&p, approved),
        Err(HostManifestError::DescriptorMismatch)
    ));
    for mutate in 0..3 {
        let mut changed = m.clone();
        match mutate {
            0 => changed.input[0] = 8,
            1 => changed.observations[0].arguments[0] = 8,
            _ => changed.observations[0].result = Some(10),
        }
        let bytes = changed.to_bytes().unwrap(); // Valid new checksum is not approval.
        assert!(matches!(
            HostManifest::from_bytes(&bytes)
                .unwrap()
                .check(&p, approved),
            Err(HostManifestError::ApprovalMismatch)
        ));
    }
    let mut changed = m;
    changed.version += 1;
    assert!(matches!(
        changed.check(&p, approved),
        Err(HostManifestError::UnsupportedVersion)
    ));
}

#[test]
fn finite_lookup_misses_and_effectful_dependencies_never_authorize_host_behavior() {
    let p = program();
    let m = manifest(&p);
    let approved = m.digest().unwrap();
    let checked = m.check(&p, approved).unwrap();
    assert!(matches!(
        checked.lookup(id(0), &[8, u64::MAX]),
        Err(HostManifestError::MissingObservation { .. })
    ));
    assert!(matches!(
        checked.lookup(id(0), &[7]),
        Err(HostManifestError::ArgumentMismatch { .. })
    ));
    assert!(matches!(
        checked.lookup(id(9), &[]),
        Err(HostManifestError::UnknownForeign { .. })
    ));
    let vm = BytecodeVm::with_config(BytecodeVmConfig {
        input: vec![8],
        ..BytecodeVmConfig::default()
    })
    .with_foreign_host(Arc::new(checked.frozen_host_table().unwrap()));
    assert!(matches!(
        vm.run(&p, &PagedMemory::with_size(8).unwrap()),
        Err(BytecodeVmError::ForeignCall(
            ForeignHostError::CallbackFailed { .. }
        ))
    ));
    // IO/READS requirements may be described, but cannot produce snapshot callbacks.
    for flags in [
        ForeignEffects::IO,
        ForeignEffects::READS,
        ForeignEffects::WRITES,
        ForeignEffects::ALLOC,
        ForeignEffects::TEMPORAL,
    ] {
        let mut effectful = p.clone();
        effectful.foreigns[0].effects = ForeignEffects::from_bits(flags).unwrap();
        let passive = HostManifest::new(&effectful, vec![], vec![]).unwrap();
        let approved = passive.digest().unwrap();
        let checked = passive.check(&effectful, approved).unwrap();
        assert!(matches!(
            checked.frozen_host_table(),
            Err(HostManifestError::UnsupportedEffect { .. })
        ));
        assert!(matches!(
            HostManifest::new(&effectful, vec![], vec![row(0, &[7, u64::MAX], Some(6))]),
            Err(HostManifestError::UnsupportedEffect { .. })
        ));
    }
}

#[test]
fn codec_rejects_corruption_stale_flags_counts_and_every_truncated_prefix() {
    let m = manifest(&program());
    let bytes = m.to_bytes().unwrap();
    for end in 0..bytes.len() {
        assert!(
            HostManifest::from_bytes(&bytes[..end]).is_err(),
            "prefix {end}"
        );
    }
    for index in 0..bytes.len() {
        let mut bad = bytes.clone();
        bad[index] ^= 0x80;
        assert!(HostManifest::from_bytes(&bad).is_err(), "byte {index}");
    }
    for (offset, data) in [
        (8, 2u16.to_le_bytes().to_vec()),
        (10, 1u16.to_le_bytes().to_vec()),
        (16, 2u16.to_le_bytes().to_vec()),
        (50, 65u32.to_le_bytes().to_vec()),
    ] {
        let mut bad = bytes.clone();
        bad[offset..offset + data.len()].copy_from_slice(&data);
        reseal(&mut bad);
        assert!(
            HostManifest::from_bytes(&bad).is_err(),
            "resealed offset {offset}"
        );
    }
    let mut suffix = bytes;
    suffix.push(0);
    assert!(HostManifest::from_bytes(&suffix).is_err());
    assert!(matches!(
        HostManifest::from_bytes(&vec![0; MAX_HOST_MANIFEST_BYTES + 1]),
        Err(HostManifestError::ResourceLimit(_))
    ));
}

#[test]
fn duplicate_noncanonical_bad_shapes_and_snapshot_debug_do_not_leak_words() {
    let p = program();
    assert!(matches!(
        HostManifest::new(
            &p,
            vec![],
            vec![row(0, &[1, 2], Some(3)), row(0, &[1, 2], Some(4))]
        ),
        Err(HostManifestError::DuplicateObservation)
    ));
    assert!(matches!(
        HostManifest::new(&p, vec![], vec![row(0, &[1], Some(3))]),
        Err(HostManifestError::ArgumentMismatch { .. })
    ));
    assert!(matches!(
        HostManifest::new(&p, vec![], vec![row(0, &[1, 2], None)]),
        Err(HostManifestError::ResultMismatch { .. })
    ));
    assert!(matches!(
        HostManifest::new(&p, vec![], vec![row(1, &[3], Some(3))]),
        Err(HostManifestError::ResultMismatch { .. })
    ));
    let mut m = manifest(&p);
    m.observations.reverse();
    assert!(matches!(
        m.to_bytes(),
        Err(HostManifestError::NoncanonicalObservations)
    ));
    let secret_word = 9876543210123456789u64;
    let m = HostManifest::new(
        &p,
        vec![secret_word],
        vec![row(0, &[secret_word, u64::MAX], Some(secret_word))],
    )
    .unwrap();
    assert!(!format!("{m:?}").contains(&secret_word.to_string()));
    assert!(!format!("{:?}", m.observations[0]).contains(&secret_word.to_string()));
    let approved = m.digest().unwrap();
    let checked = m.check(&p, approved).unwrap();
    assert!(!format!("{checked:?}").contains(&secret_word.to_string()));
}

#[test]
fn full_safe_sixteen_argument_abi_is_distinct_from_unsafe_dynamic_subset() {
    let mut instructions: Vec<_> = (0..16)
        .map(|i| Instruction::PushWord(if i % 2 == 0 { i } else { u64::MAX - i }))
        .collect();
    instructions.extend([Instruction::CallForeign(id(0)), Instruction::Return]);
    let p = BytecodeProgram {
        main: CodeRange {
            start: 0,
            end: instructions.len() as u32,
        },
        instructions,
        procedures: vec![],
        quotations: vec![],
        source_map: vec![],
        foreigns: vec![ForeignEntry {
            id: id(0),
            library: "C:\\passive-namespace\\never-loaded.dll".into(),
            symbol: "scalar16".into(),
            parameters: (0..16)
                .map(|i| {
                    if i % 2 == 0 {
                        ForeignScalarType::U64
                    } else {
                        ForeignScalarType::I64
                    }
                })
                .collect(),
            result: Some(ForeignScalarType::U64),
            effects: ForeignEffects::from_bits(ForeignEffects::PURE).unwrap(),
        }],
    };
    let arguments: Vec<_> = p.instructions[..16]
        .iter()
        .map(|i| match i {
            Instruction::PushWord(word) => *word,
            _ => unreachable!(),
        })
        .collect();
    let m = HostManifest::new(
        &p,
        vec![0; MAX_HOST_INPUT_WORDS],
        vec![row(0, &arguments, Some(u64::MAX))],
    )
    .unwrap();
    let approved = m.digest().unwrap();
    let checked = HostManifest::from_bytes(&m.to_bytes().unwrap())
        .unwrap()
        .check(&p, approved)
        .unwrap();
    let result = BytecodeVm::new()
        .with_foreign_host(Arc::new(checked.frozen_host_table().unwrap()))
        .run(&p, &PagedMemory::with_size(8).unwrap())
        .unwrap();
    assert_eq!(result.stack[0].val, u64::MAX);
    assert!(HostManifest::new(&p, vec![0; MAX_HOST_INPUT_WORDS + 1], vec![]).is_err());
    let mut too_many = p;
    too_many.foreigns[0].parameters.push(ForeignScalarType::U64);
    assert!(matches!(
        HostManifest::new(&too_many, vec![], vec![]),
        Err(HostManifestError::InvalidDescriptor { .. })
    ));
    // Merely name the unsafe functions; no native initialization runs.
    let _loader: unsafe fn(&mut DynamicLibraryManager, &str) -> OuroResult<()> =
        DynamicLibraryManager::load;
    let _extended: unsafe fn(&mut ExtendedFFIContext, &str) -> OuroResult<()> =
        ExtendedFFIContext::load_library;
}

#[test]
fn finite_observation_ceiling_retains_exact_lookup_and_bounds_resealed_counts() {
    let p = program();
    let rows = (0..MAX_HOST_OBSERVATIONS)
        .map(|word| row(0, &[word as u64, u64::MAX], Some(word as u64)))
        .collect();
    let m = HostManifest::new(&p, vec![], rows).unwrap();
    let approved = m.digest().unwrap();
    let bytes = m.to_bytes().unwrap();
    let checked = HostManifest::from_bytes(&bytes)
        .unwrap()
        .check(&p, approved)
        .unwrap();
    for word in [0, MAX_HOST_OBSERVATIONS / 2, MAX_HOST_OBSERVATIONS - 1] {
        assert_eq!(
            checked.lookup(id(0), &[word as u64, u64::MAX]).unwrap(),
            Some(word as u64)
        );
    }
    let mut overflow = m;
    overflow
        .observations
        .push(row(0, &[MAX_HOST_OBSERVATIONS as u64, u64::MAX], Some(0)));
    assert!(matches!(
        overflow.to_bytes(),
        Err(HostManifestError::ResourceLimit(_))
    ));
    assert!(matches!(
        HostManifest::new(&p, vec![], overflow.observations),
        Err(HostManifestError::ResourceLimit(_))
    ));
    // Each retained row has two arguments and one result (30 encoded bytes).
    // A valid checksum cannot authorize a count beyond the finite ceiling.
    let count = bytes.len() - 32 - MAX_HOST_OBSERVATIONS * 30 - 4;
    let mut changed = bytes;
    changed[count..count + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    reseal(&mut changed);
    assert!(matches!(
        HostManifest::from_bytes(&changed),
        Err(HostManifestError::ResourceLimit(_))
    ));
}
