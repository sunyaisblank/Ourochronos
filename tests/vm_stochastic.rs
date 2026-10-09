//! Finite joint-tape bytecode extraction against the independent numeric core.
//! Production source is serialized with an explicit whole-main scope; the
//! oracle separately evaluates the declared body with bounded masked storage.

#[allow(dead_code)] // The shared oracle admits more operations than this target.
mod oracle;

use num_bigint::BigInt;
use num_traits::One;
use oracle::{Bounds, Environment, Fault, Observation, Stop};
use ourochronos::hir::HirProgram;
use ourochronos::temporal::sparse_markov::{
    ExactRational as R, SparseMarkovError, SparseMarkovResource, SparseStationaryDecision,
};
use ourochronos::temporal::vm_stochastic::{
    extract_vm_markov, RandomTapeScenario as Scenario, VmStochasticConfig as Config,
    VmStochasticError as Error, VmStochasticModel, VmStochasticResource as Resource,
};
use ourochronos::{
    BoundsPolicy, BytecodeProgram, BytecodeVm, BytecodeVmConfig, BytecodeVmError, BytecodeVmStatus,
    OutputItem, PagedMemory, Value,
};

fn r(n: i64, d: i64) -> R {
    R::new(n.into(), d.into())
}
fn scenario(words: &[u64], n: i64, d: i64) -> Scenario {
    Scenario {
        words: words.to_vec(),
        probability: r(n, d),
    }
}

fn compile(body: &str, cells: usize, bits: u8, procedures: &str) -> BytecodeProgram {
    // This explicit bytecode profile has separate admission. Ordinary source
    // region/type gates remain strict about INPUT/RANDOM inside TEMPORAL.
    let source = format!("{procedures} TEMPORAL 0 {cells} BITS {bits} {{ {body} }}");
    BytecodeProgram::compile(
        &HirProgram::resolve(&ourochronos::parser::parse(&source).unwrap()).unwrap(),
    )
    .unwrap()
}

fn observations(items: &[OutputItem]) -> Vec<Observation> {
    items
        .iter()
        .map(|item| match item {
            OutputItem::Val(value) => Observation::Number(value.val),
            OutputItem::Char(byte) => Observation::Byte(*byte),
        })
        .collect()
}

fn verify_against_oracle(model: &VmStochasticModel, body: &str, procedures: &str) {
    let core = oracle::parse(&format!("{procedures} {body}")).unwrap();
    let production = compile(body, model.scope.cells, model.scope.cell_bits, procedures);
    let mask = (1_u64 << model.scope.cell_bits) - 1;
    for (state, row) in model.transitions.iter().enumerate() {
        assert_eq!(row.len(), model.scenarios.len());
        for (scenario_index, transition) in row.iter().enumerate() {
            let input = model.state_words(state).unwrap();
            let environment = Environment {
                anamnesis: input.clone(),
                input: model.config.input.clone(),
                random: model.scenarios[scenario_index].words.clone(),
                write_mask: Some(mask),
                bounds: match model.config.memory_bounds {
                    BoundsPolicy::Error => Bounds::Error,
                    BoundsPolicy::Wrap => Bounds::Wrap,
                    BoundsPolicy::Clamp => Bounds::Clamp,
                },
                stack: model.config.max_stack_depth,
                calls: model.config.max_call_depth,
                output: model.config.max_output_items,
                ..Environment::default()
            };
            let expected = oracle::evaluate(&core, &environment).unwrap();
            assert_eq!(expected.snapshot.stop, Stop::Finished);
            assert_eq!(transition.scenario_index, scenario_index);
            assert_eq!(
                transition
                    .stack
                    .iter()
                    .map(|word| word.val)
                    .collect::<Vec<_>>(),
                expected.snapshot.stack
            );
            assert_eq!(observations(&transition.output), expected.snapshot.output);
            assert_eq!(transition.inputs_consumed, expected.snapshot.consumed);
            assert_eq!(transition.random_consumed, expected.random_consumed);
            let successor = expected
                .snapshot
                .present
                .iter()
                .enumerate()
                .map(|(address, &word)| {
                    (word as usize) << (address * model.scope.cell_bits as usize)
                })
                .sum::<usize>();
            assert_eq!(transition.successor, successor);
            assert_eq!(
                model.state_words(successor).unwrap(),
                expected.snapshot.present
            );
            // A second direct VM invocation compares full production Value
            // provenance and output kinds, without projecting evidence away.
            let mut memory = PagedMemory::with_size(model.config.memory_cells).unwrap();
            for (address, &word) in input.iter().enumerate() {
                memory.write(address as u64, Value::new(word)).unwrap();
            }
            let execution = BytecodeVm::with_config(BytecodeVmConfig {
                max_instructions: model.config.max_instructions,
                max_call_depth: model.config.max_call_depth,
                max_stack_depth: model.config.max_stack_depth,
                max_temporal_depth: 1,
                max_output_items: model.config.max_output_items,
                max_output_bytes: model.config.max_output_bytes,
                memory_bounds: model.config.memory_bounds,
                input: model.config.input.clone(),
                random_input: model.scenarios[scenario_index].words.clone(),
                max_collections: 0,
                max_collection_items: 0,
                max_dynamic_bytes: 0,
                max_effects: 0,
                max_effect_bytes: 0,
                ..BytecodeVmConfig::default()
            })
            .run(&production, &memory)
            .unwrap();
            assert_eq!(execution.status, BytecodeVmStatus::Finished);
            assert_eq!(execution.stack, transition.stack);
            assert_eq!(execution.output, transition.output);
            assert_eq!(
                execution.instructions_executed,
                transition.instructions_executed
            );
            assert!(execution.effects.is_empty());
            for address in 0..model.config.memory_cells {
                assert_eq!(
                    execution.present.get(address as u64).unwrap().val,
                    expected.snapshot.present.get(address).copied().unwrap_or(0)
                );
            }
        }
    }
    assert!(model.state_words(model.chain.states()).is_none());
    assert!(model.stats.evidence_bytes <= model.stats.preflight_evidence_bytes);
}

#[test]
fn masking_coalesces_numeric_successors_without_losing_typed_observations() {
    let body = "RANDOM DUP OUTPUT 65 EMIT 0 PROPHECY";
    let scenarios = [scenario(&[0], 1, 3), scenario(&[2], 2, 3)];
    let model =
        extract_vm_markov(&compile(body, 1, 1, ""), &Config::default(), &scenarios).unwrap();
    assert_eq!(
        model.chain.rows(),
        &[vec![(0, R::one())], vec![(0, R::one())]]
    );
    for row in &model.transitions {
        assert_eq!(
            observations(&row[0].output),
            vec![Observation::Number(0), Observation::Byte(65)]
        );
        assert_eq!(
            observations(&row[1].output),
            vec![Observation::Number(2), Observation::Byte(65)]
        );
    }
    assert_eq!(model.scenarios, scenarios);
    verify_against_oracle(&model, body, "");
    let mut malformed = model;
    malformed.scope.cell_bits = 64;
    assert!(malformed.state_words(0).is_none());
}

#[test]
fn unused_suffixes_retain_joint_mass_and_frozen_input_resets_for_every_run() {
    let body = "RANDOM IF { RANDOM DUP OUTPUT 0 PROPHECY } ELSE { INPUT DUP OUTPUT 0 PROPHECY }";
    let config = Config {
        input: vec![9, 7],
        ..Config::default()
    };
    let scenarios = [scenario(&[0, 999], 1, 4), scenario(&[1, 2, 456], 3, 4)];
    let model = extract_vm_markov(&compile(body, 1, 1, ""), &config, &scenarios).unwrap();
    assert_eq!(
        model.chain.rows(),
        &[
            vec![(0, r(3, 4)), (1, r(1, 4))],
            vec![(0, r(3, 4)), (1, r(1, 4))]
        ]
    );
    for row in &model.transitions {
        assert_eq!(row[0].inputs_consumed, vec![9]);
        assert_eq!(row[0].random_consumed, vec![0]);
        assert!(row[1].inputs_consumed.is_empty());
        assert_eq!(row[1].random_consumed, vec![1, 2]);
    }
    let family = model.chain.stationary_family().unwrap();
    assert_eq!(family.extremal[0].weights, vec![(0, r(3, 4)), (1, r(1, 4))]);
    assert_eq!(
        model
            .chain
            .analyze(&[false, true], r(2, 3), r(1, 3))
            .unwrap()
            .decision,
        SparseStationaryDecision::Reject
    );
    verify_against_oracle(&model, body, "");
}

#[test]
fn correlated_joint_tapes_are_not_replaced_by_iid_marginals() {
    let body = "RANDOM RANDOM XOR 0 PROPHECY";
    let scenarios = [scenario(&[0, 0], 1, 2), scenario(&[1, 1], 1, 2)];
    let model =
        extract_vm_markov(&compile(body, 1, 1, ""), &Config::default(), &scenarios).unwrap();
    // Both joint scenarios have XOR zero, although each individual position
    // has a uniform marginal. An iid replacement would invent XOR-one edges.
    assert_eq!(
        model.chain.rows(),
        &[vec![(0, R::one())], vec![(0, R::one())]]
    );
    assert_eq!(
        model.chain.stationary_family().unwrap().extremal[0].weights,
        vec![(0, R::one())]
    );
    verify_against_oracle(&model, body, "");
}

#[test]
fn numeric_scope_bounds_and_all_recurrent_classes_match_independent_rows() {
    for (bounds, body) in [
        (BoundsPolicy::Error, "0 ORACLE 2 XOR 0 PROPHECY"),
        (BoundsPolicy::Wrap, "4 ORACLE 2 XOR 4 PROPHECY"),
        (
            BoundsPolicy::Clamp,
            "18446744073709551615 ORACLE 2 XOR 18446744073709551615 PROPHECY",
        ),
    ] {
        let config = Config {
            memory_bounds: bounds,
            ..Config::default()
        };
        let model =
            extract_vm_markov(&compile(body, 1, 2, ""), &config, &[scenario(&[], 1, 1)]).unwrap();
        assert_eq!(
            model.chain.rows(),
            &(0..4)
                .map(|state| vec![(state ^ 2, R::one())])
                .collect::<Vec<_>>()
        );
        let family = model.chain.stationary_family().unwrap();
        assert_eq!(family.extremal[0].weights, vec![(0, r(1, 2)), (2, r(1, 2))]);
        assert_eq!(family.extremal[1].weights, vec![(1, r(1, 2)), (3, r(1, 2))]);
        assert_eq!(
            model
                .chain
                .analyze(&[false, true, false, true], r(2, 3), r(1, 3))
                .unwrap()
                .decision,
            SparseStationaryDecision::Ambiguous
        );
        verify_against_oracle(&model, body, "");
    }
    let body = "0 ORACLE 1 ORACLE ADD DUP OUTPUT RANDOM XOR 1 PROPHECY 7 0 PROPHECY";
    let model = extract_vm_markov(
        &compile(body, 2, 2, ""),
        &Config::default(),
        &[scenario(&[0], 1, 2), scenario(&[7], 1, 2)],
    )
    .unwrap();
    verify_against_oracle(&model, body, "");
    let OutputItem::Val(value) = &model.transitions[1][0].output[0] else {
        panic!("numeric output kind lost")
    };
    // Retention check only: the independent oracle does not prove provenance.
    assert!(value.prov.deps.as_ref().unwrap().contains(&0));
    assert!(value.prov.deps.as_ref().unwrap().contains(&1));
}

#[test]
fn acyclic_calls_and_finite_dynamic_loops_replay_with_frozen_random() {
    let procedures = "PROCEDURE countdown PURE { WHILE { DUP 0 GT } { 1 SUB } }";
    let body = "0 ORACLE 17 countdown OUTPUT RANDOM 0 PROPHECY";
    let model = extract_vm_markov(
        &compile(body, 1, 1, procedures),
        &Config::default(),
        &[scenario(&[0], 1, 2), scenario(&[1], 1, 2)],
    )
    .unwrap();
    verify_against_oracle(&model, body, procedures);
    let dynamic = "0 ORACLE WHILE { DUP } { 1 SUB } POP RANDOM 0 PROPHECY";
    let model = extract_vm_markov(
        &compile(dynamic, 1, 2, ""),
        &Config::default(),
        &[scenario(&[9], 1, 1)],
    )
    .unwrap();
    verify_against_oracle(&model, dynamic, "");
}

#[test]
fn arbitrary_precision_scenario_weights_remain_exact_after_extraction() {
    let denominator = (BigInt::one() << 180_usize) + BigInt::from(13);
    let scenarios = [
        Scenario {
            words: vec![0],
            probability: R::new(1.into(), denominator.clone()),
        },
        Scenario {
            words: vec![1],
            probability: R::new(&denominator - 1, denominator.clone()),
        },
    ];
    let body = "RANDOM 0 PROPHECY";
    let model =
        extract_vm_markov(&compile(body, 1, 1, ""), &Config::default(), &scenarios).unwrap();
    let wanted = vec![
        (0, scenarios[0].probability.clone()),
        (1, scenarios[1].probability.clone()),
    ];
    assert_eq!(
        model.chain.stationary_family().unwrap().extremal[0].weights,
        wanted
    );
    assert_eq!(model.scenarios, scenarios);
    verify_against_oracle(&model, body, "");
}

#[test]
fn probability_and_tape_boundaries_never_drop_generated_cases() {
    let program = compile("RANDOM 0 PROPHECY", 1, 1, "");
    assert!(matches!(
        extract_vm_markov(&program, &Config::default(), &[]),
        Err(Error::InvalidScenarios(_))
    ));
    for scenarios in [
        vec![scenario(&[0], 0, 1), scenario(&[1], 1, 1)],
        vec![scenario(&[0], -1, 1), scenario(&[1], 2, 1)],
        vec![scenario(&[0], 1, 2)],
    ] {
        assert!(extract_vm_markov(&program, &Config::default(), &scenarios).is_err());
    }
    assert!(matches!(
        extract_vm_markov(
            &program,
            &Config::default(),
            &[Scenario {
                words: vec![0],
                probability: R::new_raw(1.into(), 0.into())
            }]
        ),
        Err(Error::Chain(SparseMarkovError::ZeroDenominator { .. }))
    ));
    let oversized = Scenario {
        words: vec![0],
        probability: R::new_raw(BigInt::one() << 5000_usize, BigInt::one() << 5000_usize),
    };
    assert!(matches!(
        extract_vm_markov(&program, &Config::default(), &[oversized]),
        Err(Error::Chain(SparseMarkovError::ResourceLimit {
            resource: SparseMarkovResource::IntegerBits,
            ..
        }))
    ));
    let duplicates = [scenario(&[0], 1, 3), scenario(&[0], 2, 3)];
    let model = extract_vm_markov(&program, &Config::default(), &duplicates).unwrap();
    assert_eq!(model.transitions[0].len(), 2);
    assert_eq!(model.scenarios, duplicates);
    assert_eq!(model.chain.edges(), 2);
}

#[test]
fn exact_gas_and_dynamic_failures_refuse_partial_transition_models() {
    let source = "RANDOM 0 PROPHECY";
    let program = compile(source, 1, 1, "");
    let config = Config {
        max_instructions: 6,
        ..Config::default()
    };
    let model = extract_vm_markov(&program, &config, &[scenario(&[0], 1, 1)]).unwrap();
    assert!(model
        .transitions
        .iter()
        .flatten()
        .all(|transition| transition.instructions_executed == 6));
    assert!(matches!(
        extract_vm_markov(
            &program,
            &Config {
                max_instructions: 5,
                ..config.clone()
            },
            &[scenario(&[0], 1, 1)]
        ),
        Err(Error::Execution {
            state: 0,
            scenario: 0,
            error
        }) if matches!(*error, BytecodeVmError::GasExhausted { limit: 5 })
    ));
    assert!(matches!(
        extract_vm_markov(&program, &config, &[scenario(&[], 1, 1)]),
        Err(Error::Execution {
            error,
            ..
        }) if matches!(*error, BytecodeVmError::RandomInputExhausted { consumed: 0 })
    ));
    // Successful earlier work cannot hide a later failing scenario or state.
    assert!(matches!(
        extract_vm_markov(
            &program,
            &config,
            &[scenario(&[0], 1, 2), scenario(&[], 1, 2)]
        ),
        Err(Error::Execution {
            state: 0,
            scenario: 1,
            error
        }) if matches!(*error, BytecodeVmError::RandomInputExhausted { consumed: 0 })
    ));
    let later_state = compile(
        "0 ORACLE IF { RANDOM POP RANDOM 0 PROPHECY } ELSE { RANDOM 0 PROPHECY }",
        1,
        1,
        "",
    );
    assert!(matches!(
        extract_vm_markov(&later_state, &Config::default(), &[scenario(&[0], 1, 1)]),
        Err(Error::Execution {
            state: 1,
            scenario: 0,
            error
        }) if matches!(*error, BytecodeVmError::RandomInputExhausted { consumed: 1 })
    ));
    assert!(matches!(
        extract_vm_markov(
            &compile("INPUT 0 PROPHECY", 1, 1, ""),
            &Config::default(),
            &[scenario(&[], 1, 1)]
        ),
        Err(Error::Execution {
            error,
            ..
        }) if matches!(*error, BytecodeVmError::InputExhausted { consumed: 0 })
    ));
    for (body, status) in [
        ("HALT", BytecodeVmStatus::Halted),
        ("PARADOX", BytecodeVmStatus::Paradox),
    ] {
        assert!(
            matches!(extract_vm_markov(&compile(body, 1, 1, ""), &Config::default(), &[scenario(&[], 1, 1)]), Err(Error::NonFinished { status: actual, .. }) if actual == status)
        );
    }
    let nonterminating = "WHILE { 1 } { NOP }";
    assert!(matches!(
        extract_vm_markov(
            &compile(nonterminating, 1, 1, ""),
            &Config {
                max_instructions: 32,
                ..Config::default()
            },
            &[scenario(&[], 1, 1)]
        ),
        Err(Error::Execution {
            error,
            ..
        }) if matches!(*error, BytecodeVmError::GasExhausted { limit: 32 })
    ));
    assert_eq!(
        oracle::evaluate(
            &oracle::parse("RANDOM RANDOM").unwrap(),
            &Environment {
                random: vec![7],
                ..Environment::default()
            }
        )
        .unwrap_err(),
        Fault::RandomExhausted { consumed: 1 }
    );
    for (config, body, procedures) in [
        (
            Config {
                max_stack_depth: 0,
                ..Config::default()
            },
            source,
            "",
        ),
        (
            Config {
                max_call_depth: 0,
                ..Config::default()
            },
            "value 0 PROPHECY",
            "PROCEDURE value PURE { 0 }",
        ),
        (
            Config {
                max_output_items: 0,
                ..Config::default()
            },
            "7 OUTPUT",
            "",
        ),
        (
            Config {
                max_output_bytes: 63,
                ..Config::default()
            },
            "7 OUTPUT",
            "",
        ),
    ] {
        assert!(matches!(
            extract_vm_markov(
                &compile(body, 1, 1, procedures),
                &config,
                &[scenario(&[0], 1, 1)]
            ),
            Err(Error::Execution { .. })
        ));
    }
}

#[test]
fn reachable_capability_denial_ignores_unused_procedures_without_hiding_later_calls() {
    let procedures = "PROCEDURE unused { CLOCK POP }";
    let body = "RANDOM 0 PROPHECY";
    let model = extract_vm_markov(
        &compile(body, 1, 1, procedures),
        &Config::default(),
        &[scenario(&[0], 1, 1)],
    )
    .unwrap();
    verify_against_oracle(&model, body, ""); // Unused definition has no transition effect.
    for body in [
        "CLOCK POP",
        "VEC_NEW POP",
        "0 SLEEP",
        "[ 1 ] POP",
        "TEMPORAL 0 1 BITS 1 { NOP }",
    ] {
        assert!(
            extract_vm_markov(
                &compile(body, 1, 1, ""),
                &Config::default(),
                &[scenario(&[], 1, 1)]
            )
            .is_err(),
            "{body}"
        );
    }
    let procedures = "PROCEDURE first PURE { 65 EMIT } PROCEDURE host { CLOCK POP }";
    assert!(matches!(
        extract_vm_markov(
            &compile("host", 1, 1, procedures),
            &Config::default(),
            &[scenario(&[], 1, 1)]
        ),
        Err(Error::Unsupported { .. })
    ));
    assert!(extract_vm_markov(
        &compile("recur", 1, 1, "PROCEDURE recur { recur }"),
        &Config::default(),
        &[scenario(&[], 1, 1)]
    )
    .is_err());
    let unscoped = BytecodeProgram::compile(
        &HirProgram::resolve(&ourochronos::parser::parse("RANDOM 0 PROPHECY").unwrap()).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        extract_vm_markov(&unscoped, &Config::default(), &[scenario(&[0], 1, 1)]),
        Err(Error::Unsupported { .. })
    ));
    let prelude = BytecodeProgram::compile(
        &HirProgram::resolve(
            &ourochronos::parser::parse("NOP TEMPORAL 0 1 BITS 1 { RANDOM 0 PROPHECY }").unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(matches!(
        extract_vm_markov(&prelude, &Config::default(), &[scenario(&[0], 1, 1)]),
        Err(Error::Unsupported { .. })
    ));
    assert!(matches!(
        extract_vm_markov(
            &compile(body, 1, 13, ""),
            &Config::default(),
            &[scenario(&[0], 1, 1)]
        ),
        Err(Error::Unsupported { .. })
    ));
}

#[test]
fn product_work_and_observation_caps_have_adjacent_preflight_boundaries() {
    let scenarios = [scenario(&[0], 1, 2), scenario(&[1], 1, 2)];
    let config = Config {
        memory_cells: 2,
        max_instructions: 6,
        max_stack_depth: 2,
        max_output_items: 0,
        max_output_bytes: 0,
        ..Config::default()
    };
    let program = compile("RANDOM 0 PROPHECY", 1, 1, "");
    let model = extract_vm_markov(&program, &config, &scenarios).unwrap();
    assert_eq!(model.stats.evaluations, 4);
    assert_eq!(model.stats.preflight_work, 32);
    assert_eq!(model.stats.preflight_evidence_bytes, 1056);
    for resource in [
        Resource::States,
        Resource::Scenarios,
        Resource::TapeWords,
        Resource::Evaluations,
        Resource::Work,
        Resource::EvidenceBytes,
    ] {
        let required = match resource {
            Resource::States | Resource::Scenarios | Resource::TapeWords => 2_u64,
            Resource::Evaluations => 4,
            Resource::Work => 32,
            Resource::EvidenceBytes => 1056,
        };
        let configure = |limit: u64| {
            let mut candidate = config.clone();
            match resource {
                Resource::States => candidate.extraction_limits.max_states = limit as usize,
                Resource::Scenarios => candidate.extraction_limits.max_scenarios = limit as usize,
                Resource::TapeWords => candidate.extraction_limits.max_tape_words = limit as usize,
                Resource::Evaluations => {
                    candidate.extraction_limits.max_evaluations = limit as usize
                }
                Resource::Work => candidate.extraction_limits.max_work = limit,
                Resource::EvidenceBytes => {
                    candidate.extraction_limits.max_evidence_bytes = limit as usize
                }
            }
            candidate
        };
        assert!(extract_vm_markov(&program, &configure(required), &scenarios).is_ok());
        assert!(
            matches!(extract_vm_markov(&program, &configure(required - 1), &scenarios), Err(Error::ResourceLimit { resource: actual, required: actual_required, limit }) if actual == resource && actual_required == required && limit == required - 1)
        );
    }
    let mut chain_limit = config;
    chain_limit.chain_limits.max_raw_edges = 3;
    assert!(matches!(
        extract_vm_markov(&program, &chain_limit, &scenarios),
        Err(Error::Chain(SparseMarkovError::ResourceLimit {
            resource: SparseMarkovResource::RawEdges,
            limit: 3,
            required: 4
        }))
    ));
}
