//! Study safeguards and backend/oracle comparisons; no production cache hook.

#[allow(dead_code)]
#[path = "../examples/solver_reuse.rs"]
mod study;

use ourochronos::finite_proof::FiniteProofConfig;
use ourochronos::hir::HirProgram;
use ourochronos::parser::parse;
use ourochronos::{BoundsPolicy, BytecodeProgram};
use study::{Answer, Query, StudyCache, Workload, STUDY_SEMANTICS};

fn compile(source: &str) -> BytecodeProgram {
    BytecodeProgram::compile(&HirProgram::resolve(&parse(source).unwrap()).unwrap()).unwrap()
}
fn workload(source: &str) -> Workload {
    Workload::compile("test", source, &[], FiniteProofConfig::default()).unwrap()
}

#[test]
fn twelve_frozen_fixtures_all_384_queries_match_complete_oracle_and_original_vm() {
    let fixtures = study::fixtures().unwrap();
    assert_eq!(fixtures.len(), 12);
    let expected_fixed = [16, 1, 0, 4, 2, 0, 1, 1, 1, 1, 1, 1];
    for (w, expected) in fixtures.iter().zip(expected_fixed) {
        assert_eq!(w.queries.len(), 32);
        assert_eq!(w.fixed().len(), expected, "{}", w.name);
        assert!(w.scope().states() <= 256);
        let reused = study::solve_reused(w, &w.queries).unwrap();
        for (query, reused) in w.queries.iter().zip(reused) {
            let answer = study::solve_fresh(w, query).unwrap();
            w.validate_answer(query, &answer).unwrap();
            w.validate_answer(query, &reused).unwrap();
            assert_eq!(
                matches!(answer, Answer::Sat(_)),
                w.oracle_sat(query).unwrap()
            );
            assert_eq!(
                matches!(answer, Answer::Sat(_)),
                matches!(reused, Answer::Sat(_))
            );
        }
    }
}

#[test]
fn keys_bind_code_tape_query_domain_resources_and_semantics_revision() {
    let identity = "TEMPORAL 0 1 BITS 2 { 0 ORACLE 0 PROPHECY }";
    let w = workload(identity);
    let key = w.key(&Query::Exists, STUDY_SEMANTICS).unwrap();
    assert_eq!(
        key,
        workload(identity)
            .key(&Query::Exists, STUDY_SEMANTICS)
            .unwrap()
    );
    assert_ne!(key, w.key(&Query::Exists, "next-semantics").unwrap());
    assert_ne!(
        key,
        w.key(&Query::CellEquals { cell: 0, value: 0 }, STUDY_SEMANTICS)
            .unwrap()
    );
    assert_ne!(
        key,
        workload("TEMPORAL 0 1 BITS 2 { 0 ORACLE NOP 0 PROPHECY }")
            .key(&Query::Exists, STUDY_SEMANTICS)
            .unwrap()
    );
    assert_ne!(
        key,
        workload("TEMPORAL 0 1 BITS 3 { 0 ORACLE 0 PROPHECY }")
            .key(&Query::Exists, STUDY_SEMANTICS)
            .unwrap()
    );
    for policy in [BoundsPolicy::Error, BoundsPolicy::Wrap, BoundsPolicy::Clamp] {
        for field in 0..7 {
            let mut config = FiniteProofConfig {
                memory_bounds: policy,
                ..FiniteProofConfig::default()
            };
            match field {
                0 => config.memory_cells -= 1,
                1 => config.max_instructions += 1,
                2 => config.max_stack_depth -= 1,
                3 => config.max_call_depth -= 1,
                4 => config.max_output_items += 1,
                5 => config.max_output_bytes += 1,
                _ => config.memory_cells -= 2,
            }
            let changed = Workload::compile("changed", identity, &[], config).unwrap();
            assert_ne!(key, changed.key(&Query::Exists, STUDY_SEMANTICS).unwrap());
        }
    }
    let source = "TEMPORAL 0 1 BITS 2 { INPUT 0 PROPHECY }";
    let a = Workload::compile("input", source, &[0], FiniteProofConfig::default()).unwrap();
    let b = Workload::compile("input", source, &[1], FiniteProofConfig::default()).unwrap();
    assert_eq!(a.original(), b.original());
    assert_ne!(
        a.key(&Query::Exists, STUDY_SEMANTICS).unwrap(),
        b.key(&Query::Exists, STUDY_SEMANTICS).unwrap()
    );
    // Even identical specialized words retain distinct exact original identity.
    let literal = workload("TEMPORAL 0 1 BITS 2 { 0 0 PROPHECY }");
    assert_ne!(
        a.key(&Query::Exists, STUDY_SEMANTICS).unwrap(),
        literal.key(&Query::Exists, STUDY_SEMANTICS).unwrap()
    );
    assert!(w
        .key(&Query::CellEquals { cell: 1, value: 0 }, STUDY_SEMANTICS)
        .is_err());
}

#[test]
fn cache_admits_only_replayed_sat_or_complete_unsat_and_enforces_caps() {
    let identity = workload("TEMPORAL 0 1 BITS 2 { 0 ORACLE 0 PROPHECY }");
    let flip = workload("TEMPORAL 0 1 BITS 2 { 0 ORACLE 1 XOR 0 PROPHECY }");
    let mut cache = StudyCache::new(2, 1024).unwrap();
    assert!(cache.is_empty());
    assert!(cache
        .insert(&identity, &Query::Exists, Answer::Unknown)
        .is_err());
    assert!(cache
        .insert(&identity, &Query::Exists, Answer::Unsat)
        .is_err());
    assert!(cache
        .insert(&flip, &Query::Exists, Answer::Sat(vec![0; 16]))
        .is_err());
    let mut outside = vec![0; 16];
    outside[15] = 1;
    assert!(cache
        .insert(&identity, &Query::Exists, Answer::Sat(outside))
        .is_err());
    let mut oversized_word = vec![0; 16];
    oversized_word[0] = 4;
    assert!(cache
        .insert(&identity, &Query::Exists, Answer::Sat(oversized_word))
        .is_err());
    let sat = Answer::Sat(identity.fixed()[0].clone());
    cache
        .insert(&identity, &Query::Exists, sat.clone())
        .unwrap();
    assert_eq!(cache.get(&identity, &Query::Exists).unwrap(), Some(&sat));
    assert_eq!(cache.get(&flip, &Query::Exists).unwrap(), None);
    cache.insert(&flip, &Query::Exists, Answer::Unsat).unwrap();
    let constrained = Query::CellEquals { cell: 0, value: 9 };
    cache
        .insert(&identity, &constrained, Answer::Unsat)
        .unwrap();
    assert_eq!(cache.len(), 2);
    assert!(cache.bytes() <= 1024);
    assert_eq!(cache.get(&identity, &Query::Exists).unwrap(), None);
    assert_eq!(
        cache.get(&identity, &constrained).unwrap(),
        Some(&Answer::Unsat)
    );
    assert!(StudyCache::new(1001, 1024).is_err());
    assert!(StudyCache::new(1, study::MAX_CACHE_BYTES + 1).is_err());
    let mut tiny = StudyCache::new(1, 255).unwrap();
    assert!(tiny.insert(&flip, &Query::Exists, Answer::Unsat).is_err());
    assert!(tiny.is_empty());
}

#[test]
fn input_specialization_is_linear_exact_and_preserves_fetched_gas() {
    let source = "TEMPORAL 0 1 BITS 2 { INPUT 3 XOR 0 PROPHECY }";
    let original = compile(source);
    assert!(study::specialize_input(&original, &[]).is_err());
    assert!(study::specialize_input(&original, &[1, 2]).is_err());
    for word in [0, 1, 3, u64::MAX] {
        let w = Workload::from_program(
            "frozen",
            original.clone(),
            &[word],
            FiniteProofConfig::default(),
        )
        .unwrap();
        assert_eq!(w.fixed()[0][0], (word ^ 3) & 3);
        let row = study::interpret(w.specialized(), w.profile(), w.scope(), &w.fixed()[0]).unwrap();
        assert_eq!(row.gas, (original.main.end - original.main.start) as u64);
        let exact = FiniteProofConfig {
            max_instructions: row.gas,
            ..FiniteProofConfig::default()
        };
        assert!(Workload::from_program("exact-gas", original.clone(), &[word], exact).is_ok());
        let short = FiniteProofConfig {
            max_instructions: row.gas - 1,
            ..FiniteProofConfig::default()
        };
        assert!(Workload::from_program("short-gas", original.clone(), &[word], short).is_err());
    }
    for source in [
        "TEMPORAL 0 1 BITS 2 { 1 IF { INPUT } ELSE { INPUT } 0 PROPHECY }",
        "PROCEDURE helper { 0 } TEMPORAL 0 1 BITS 2 { helper POP INPUT 0 PROPHECY }",
        "TEMPORAL 0 1 BITS 2 { WHILE { 0 } { NOP } INPUT 0 PROPHECY }",
    ] {
        assert!(study::specialize_input(&compile(source), &[1]).is_err());
    }
    assert!(
        study::specialize_input(&compile("TEMPORAL 0 1 BITS 2 { 0 0 PROPHECY }"), &[1]).is_err()
    );
}

#[test]
fn independent_word_arithmetic_wraps_before_finite_store_and_queries_stay_scoped() {
    let w = workload("TEMPORAL 0 1 BITS 4 { 0 ORACLE 18446744073709551615 ADD 2 MUL 0 PROPHECY }");
    assert_eq!(w.fixed().len(), 1);
    assert_eq!(w.fixed()[0][0], 2);
    for query in &w.queries {
        w.validate_answer(query, &study::solve_fresh(&w, query).unwrap())
            .unwrap();
    }
    let wrong = Query::CellEquals { cell: 0, value: 3 };
    assert!(w
        .validate_answer(&wrong, &Answer::Sat(w.fixed()[0].clone()))
        .is_err());
    // The SMT query has the identical explicit finite domain, even when its
    // predicate asks for a full-width word outside that domain.
    let outside = Query::CellEquals {
        cell: 0,
        value: u64::MAX,
    };
    assert_eq!(study::solve_fresh(&w, &outside).unwrap(), Answer::Unsat);
    w.validate_answer(&outside, &Answer::Unsat).unwrap();
    assert!(study::solve_reused(&w, &vec![Query::Exists; 33]).is_err());
}

#[test]
fn rejects_unsupported_neighbors_and_any_finite_row_gas_or_bounds_failure() {
    for source in [
        "TEMPORAL 0 1 BITS 2 { 0 ORACLE OUTPUT }",
        "TEMPORAL 0 1 BITS 2 { WHILE { 0 } { NOP } }",
        "TEMPORAL 0 1 BITS 2 { CLOCK POP }",
        "TEMPORAL 0 1 BITS 2 { VEC_NEW POP }",
        "TEMPORAL 0 1 BITS 2 { TEMPORAL 0 1 BITS 1 { NOP } }",
        "TEMPORAL 1 1 BITS 2 { 0 0 PROPHECY }",
        "TEMPORAL 0 1 BITS 9 { 0 0 PROPHECY }",
        "TEMPORAL 0 1 BITS 2 { 0 0 PROPHECY } 0 1 PROPHECY",
    ] {
        assert!(
            Workload::compile("unsupported", source, &[], FiniteProofConfig::default()).is_err(),
            "{source}"
        );
    }
    let source = "TEMPORAL 0 1 BITS 2 { 0 ORACLE IF { 3 } ELSE { 1 } 0 PROPHECY }";
    let gas = FiniteProofConfig {
        max_instructions: 3,
        ..FiniteProofConfig::default()
    };
    assert!(Workload::compile("gas", source, &[], gas).is_err());
    let stack = FiniteProofConfig {
        max_stack_depth: 1,
        ..FiniteProofConfig::default()
    };
    assert!(Workload::compile("stack", source, &[], stack).is_err());
    // A failure on only SOME rows still prevents an exhaustive/cache verdict.
    let dynamic = "TEMPORAL 0 2 BITS 2 { 0 ORACLE ORACLE 0 PROPHECY }";
    assert!(Workload::compile("partial", dynamic, &[], FiniteProofConfig::default()).is_err());
    for policy in [BoundsPolicy::Wrap, BoundsPolicy::Clamp] {
        let cfg = FiniteProofConfig {
            memory_bounds: policy,
            ..FiniteProofConfig::default()
        };
        let w = Workload::compile("bounded", dynamic, &[], cfg).unwrap();
        for q in &w.queries {
            w.validate_answer(q, &study::solve_fresh(&w, q).unwrap())
                .unwrap();
        }
    }
}
