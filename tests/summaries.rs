//! Exact literal countdown summaries against the independent core machine.
//! Oracle fuel counts source continuations; gas assertions below separately
//! derive fetched-bytecode costs for declared instruction shapes.

// The shared oracle exposes a larger core than this summary-only target uses.
#[allow(dead_code)]
mod oracle;

use oracle::{Bounds, Environment, Fault, Observation, Snapshot, Stop};
use ourochronos::{
    admit_program, analyze_bytecode_temporal, lower_bytecode_temporal_ir, AdmissionConfig,
    BoundsPolicy, BytecodeProgram, BytecodeTemporalDisposition, BytecodeVm, BytecodeVmConfig,
    BytecodeVmError, BytecodeVmStatus, CompareOp, FixedPointWitness, GlobalFixedPointSolver,
    GlobalSolveConfig, GlobalSolveResult, GlobalUniquenessResult, IrCompleteness, IrExprKind,
    ObservationKind, OutputItem, PagedMemory, PropertyVerificationResult, TemporalIr,
    TemporalIrConfig, Value, WordBinaryOp,
};

fn bytecode(source: &str, width: usize) -> BytecodeProgram {
    let program = ourochronos::parser::parse(source).expect("declared source parses");
    admit_program(
        &program,
        AdmissionConfig {
            memory_cells: width,
        },
    )
    .expect("declared source admits")
    .into_program()
}

fn config(width: usize, gas: u64) -> GlobalSolveConfig {
    GlobalSolveConfig {
        memory_cells: width,
        loop_unroll_limit: 1,
        solver_timeout_ms: 5000,
        max_instructions: gas,
        bounds_policy: BoundsPolicy::Error,
    }
}

fn ir(program: &BytecodeProgram, width: usize, unroll: usize) -> TemporalIr {
    lower_bytecode_temporal_ir(
        program,
        TemporalIrConfig {
            memory_cells: width,
            loop_unroll_limit: unroll,
            bounds_policy: BoundsPolicy::Error,
        },
    )
    .unwrap()
}

fn observations(items: &[OutputItem]) -> Vec<Observation> {
    items
        .iter()
        .map(|item| match item {
            OutputItem::Val(word) => Observation::Number(word.val),
            OutputItem::Char(byte) => Observation::Byte(*byte),
        })
        .collect()
}

fn run(
    program: &BytecodeProgram,
    environment: &Environment,
    gas: u64,
) -> Result<(Snapshot, u64), BytecodeVmError> {
    let mut memory = PagedMemory::with_size(environment.anamnesis.len()).unwrap();
    for (address, &word) in environment.anamnesis.iter().enumerate() {
        memory.write(address as u64, Value::new(word)).unwrap();
    }
    let result = BytecodeVm::with_config(BytecodeVmConfig {
        max_instructions: gas,
        memory_bounds: BoundsPolicy::Error,
        ..BytecodeVmConfig::default()
    })
    .run(program, &memory)?;
    assert!(result.effects.is_empty());
    assert!(result.inputs_consumed.is_empty());
    let snapshot = Snapshot {
        stack: result.stack.iter().map(|word| word.val).collect(),
        present: (0..environment.anamnesis.len())
            .map(|address| {
                result
                    .present
                    .read_checked(address as u64, BoundsPolicy::Error, Default::default())
                    .unwrap()
                    .val
            })
            .collect(),
        output: observations(&result.output),
        consumed: result.inputs_consumed,
        stop: match result.status {
            BytecodeVmStatus::Finished => Stop::Finished,
            BytecodeVmStatus::Halted => Stop::Halted,
            BytecodeVmStatus::Paradox => Stop::Paradox,
        },
    };
    Ok((snapshot, result.instructions_executed))
}

fn expected(source: &str, environment: &Environment) -> Snapshot {
    oracle::evaluate(&oracle::parse(source).unwrap(), environment)
        .unwrap()
        .snapshot
}

// Adapter for inspecting the actual lowered IR, not an expected-value oracle.
// Its deliberately small expression fragment matches this target's programs.
// All expected memory/output comes from oracle's independent text machine.
#[derive(Clone)]
enum Datum {
    Word(u64),
    Bool(bool),
    Memory(Vec<u64>),
}
fn word(value: &Datum) -> u64 {
    match value {
        Datum::Word(word) => *word,
        _ => panic!("word required"),
    }
}
fn boolean(value: &Datum) -> bool {
    match value {
        Datum::Bool(value) => *value,
        _ => panic!("Boolean required"),
    }
}
fn memory(value: &Datum) -> &Vec<u64> {
    match value {
        Datum::Memory(value) => value,
        _ => panic!("memory required"),
    }
}

fn check_ir(ir: &TemporalIr, environment: &Environment, wanted: &Snapshot) {
    assert_eq!(ir.completeness, IrCompleteness::Complete);
    assert_eq!(ir.bounds_policy, BoundsPolicy::Error);
    let mut values = Vec::<Datum>::new();
    for expression in &ir.expressions {
        let value = match expression.kind {
            IrExprKind::WordConst(value) => Datum::Word(value),
            IrExprKind::BoolConst(value) => Datum::Bool(value),
            IrExprKind::Anamnesis => Datum::Memory(environment.anamnesis.clone()),
            IrExprKind::ZeroMemory => Datum::Memory(vec![0; environment.anamnesis.len()]),
            IrExprKind::Select {
                memory: source,
                address,
            } => Datum::Word(memory(&values[source])[word(&values[address]) as usize]),
            IrExprKind::Store {
                memory: source,
                address,
                value,
            } => {
                let mut cells = memory(&values[source]).clone();
                cells[word(&values[address]) as usize] = word(&values[value]);
                Datum::Memory(cells)
            }
            IrExprKind::WordBinary {
                op: WordBinaryOp::Add,
                lhs,
                rhs,
            } => Datum::Word(word(&values[lhs]).wrapping_add(word(&values[rhs]))),
            IrExprKind::Compare { op, lhs, rhs } => {
                let (a, b) = (word(&values[lhs]), word(&values[rhs]));
                Datum::Bool(match op {
                    CompareOp::Eq => a == b,
                    CompareOp::Ne => a != b,
                    CompareOp::Ult => a < b,
                    CompareOp::Ugt => a > b,
                    CompareOp::Ule => a <= b,
                    CompareOp::Uge => a >= b,
                    _ => panic!("signed compare outside inspection fragment"),
                })
            }
            IrExprKind::BoolNot(value) => Datum::Bool(!boolean(&values[value])),
            IrExprKind::BoolAnd(a, b) => Datum::Bool(boolean(&values[a]) && boolean(&values[b])),
            IrExprKind::BoolOr(a, b) => Datum::Bool(boolean(&values[a]) || boolean(&values[b])),
            IrExprKind::Ite {
                condition,
                when_true,
                when_false,
            } => values[if boolean(&values[condition]) {
                when_true
            } else {
                when_false
            }]
            .clone(),
            _ => panic!(
                "expression outside inspection fragment: {:?}",
                expression.kind
            ),
        };
        values.push(value);
    }
    assert!(boolean(&values[ir.valid]));
    assert_eq!(memory(&values[ir.final_memory]), &wanted.present);
    let observed: Vec<_> = ir
        .observations
        .iter()
        .filter(|item| boolean(&values[item.guard]))
        .map(|item| match item.kind {
            ObservationKind::Value => Observation::Number(word(&values[item.value])),
            ObservationKind::Character => Observation::Byte(word(&values[item.value]) as u8),
        })
        .collect();
    assert_eq!(observed, wanted.output);
}

fn check_witness(source: &str, witness: &FixedPointWitness, gas: u64) {
    assert_eq!(witness.completeness, IrCompleteness::Complete);
    assert!(witness.is_replay_verified());
    let state: Vec<_> = (0..witness.memory.len())
        .map(|address| witness.memory.read(address as u64).val)
        .collect();
    let environment = Environment {
        anamnesis: state.clone(),
        ..Environment::default()
    };
    let wanted = expected(source, &environment);
    assert_eq!(wanted.present, state);
    assert_eq!(observations(&witness.output), wanted.output);
    assert_eq!(
        run(&bytecode(source, state.len()), &environment, gas)
            .unwrap()
            .0,
        wanted
    );
}

#[test]
fn literal_affine_campaign_preserves_complete_observations_and_unroll_independence() {
    let environment = Environment {
        anamnesis: vec![13, 29, 41, 53],
        bounds: Bounds::Error,
        ..Environment::default()
    };
    for condition in ["DUP", "DUP 0 GT"] {
        for counter in [0_u64, 1, 2, 17, 64, 257] {
            for initial in [0_u64, 1, u64::MAX - 3, u64::MAX] {
                for delta in [0_u64, 1, 5, u64::MAX] {
                    let source = format!("99 2 PROPHECY 2 PRESENT OUTPUT 777 {initial} {counter} WHILE {{ {condition} }} {{ SWAP {delta} ADD SWAP 1 SUB }} DUP OUTPUT POP DUP OUTPUT 65 EMIT DUP 0 PROPHECY");
                    let program = bytecode(&source, 4);
                    let wanted = expected(&source, &environment);
                    // Conventional arithmetic justifies the independent machine's
                    // final accumulator as well as its iterative execution.
                    let accumulator = initial.wrapping_add(counter.wrapping_mul(delta));
                    assert_eq!(wanted.stack, vec![777, accumulator]);
                    assert_eq!(wanted.present, vec![accumulator, 0, 99, 0]);
                    let tests = if condition == "DUP" { 2 } else { 4 };
                    // 19 non-loop primitive/literal records plus main RETURN;
                    // each body has six records, followed by LOOP_BACK.
                    let gas = 20 + (counter + 1) * tests + counter * 7;
                    let (actual, fetched) = run(&program, &environment, gas).unwrap();
                    assert_eq!(actual, wanted, "{source}");
                    assert_eq!(fetched, gas);
                    let lowered = ir(&program, 4, 0);
                    check_ir(&lowered, &environment, &wanted);
                    assert_eq!(ir(&program, 4, 1).to_smt2(false), lowered.to_smt2(false));
                    assert_eq!(ir(&program, 4, 7).to_smt2(false), lowered.to_smt2(false));
                }
            }
        }
    }
}

#[test]
fn acyclic_procedure_countdown_has_exact_fetched_gas_boundary() {
    let source =
        "PROCEDURE dec PURE { WHILE { DUP } { 1 SUB } } PROCEDURE via PURE { dec } 23 via OUTPUT";
    let program = bytecode(source, 1);
    let environment = Environment {
        anamnesis: vec![0],
        ..Environment::default()
    };
    let wanted = expected(source, &environment);
    // 23 * (DUP,WHILE_FALSE,1,SUB,LOOP_BACK), final two-record
    // condition, literal, two calls, two returns, OUTPUT, main RETURN.
    let gas = 23 * 5 + 9;
    assert_eq!(
        run(&program, &environment, gas).unwrap(),
        (wanted.clone(), gas)
    );
    assert!(
        matches!(run(&program, &environment, gas - 1), Err(BytecodeVmError::GasExhausted { limit }) if limit == gas - 1)
    );
    check_ir(&ir(&program, 1, 1), &environment, &wanted);
    match GlobalFixedPointSolver::solve_bytecode(&program, config(1, gas)) {
        GlobalSolveResult::Found(witness) => {
            assert_eq!(witness.instructions_executed, gas);
            check_witness(source, &witness, gas);
        }
        other => panic!("finite procedure summary was not decided: {other:?}"),
    }
    match GlobalFixedPointSolver::solve_bytecode(&program, config(1, gas - 1)) {
        GlobalSolveResult::Unknown {
            reason,
            completeness: Some(IrCompleteness::Complete),
            ..
        } => assert!(reason.contains(&format!("path bound {gas}"))),
        other => panic!("gas mismatch was not unknown: {other:?}"),
    }
    // Value-insensitive readiness remains conservative; native lowering above
    // establishes Complete with the literal argument propagated through calls.
    assert_eq!(
        analyze_bytecode_temporal(
            &program,
            TemporalIrConfig {
                memory_cells: 1,
                loop_unroll_limit: 1,
                bounds_policy: BoundsPolicy::Error
            }
        )
        .unwrap()
        .disposition(),
        BytecodeTemporalDisposition::Unknown
    );
}

#[test]
fn branch_cost_sums_both_summary_paths_conservatively() {
    let source = "PROCEDURE dec PURE { WHILE { DUP } { 1 SUB } } 0 ORACLE IF { 23 dec } ELSE { 2 dec } OUTPUT";
    let program = bytecode(source, 1);
    // Two branch costs: literal + call + (5k+2) + return = 5k+5;
    // six shared fetches include IF_FALSE and the then-to-end JUMP.
    let upper_bound = (5 * 23 + 5) + (5 * 2 + 5) + 6;
    for (initial, fetched) in [(0, 20), (1, 126)] {
        let environment = Environment {
            anamnesis: vec![initial],
            ..Environment::default()
        };
        let wanted = expected(source, &environment);
        check_ir(&ir(&program, 1, 1), &environment, &wanted);
        assert_eq!(
            run(&program, &environment, fetched).unwrap(),
            (wanted, fetched)
        );
        assert!(fetched <= upper_bound);
    }
    match GlobalFixedPointSolver::solve_bytecode(&program, config(1, upper_bound)) {
        GlobalSolveResult::Found(witness) => check_witness(source, &witness, upper_bound),
        other => panic!("both finite branch summaries were not usable: {other:?}"),
    }
    // Conservative acceptance can require more gas than any one path. The
    // diagnostic must report the actual bound instead of claiming exhaustion.
    match GlobalFixedPointSolver::solve_bytecode(&program, config(1, upper_bound - 1)) {
        GlobalSolveResult::Unknown { reason, .. } => {
            assert!(reason.contains(&format!("path bound {upper_bound}")))
        }
        other => panic!("conservative branch bound was bypassed: {other:?}"),
    }
}

#[test]
fn summarized_affine_solver_property_and_uniqueness_agree_with_oracle() {
    let source = "PROCEDURE add_many PURE { WHILE { DUP 0 GT } { SWAP 5 ADD SWAP 1 SUB } } PROCEDURE via PURE { add_many } 18446744073709551612 17 via POP DUP OUTPUT 65 EMIT 0 PROPHECY";
    let program = bytecode(source, 1);
    let solve_config = config(1, 1000);
    let wanted = expected(
        source,
        &Environment {
            anamnesis: vec![0],
            ..Environment::default()
        },
    );
    assert_eq!(wanted.present, vec![81]);
    match GlobalFixedPointSolver::analyze_uniqueness_bytecode(&program, solve_config) {
        GlobalUniquenessResult::Unique {
            witness,
            certificate,
        } => {
            assert_eq!(certificate.completeness, IrCompleteness::Complete);
            check_witness(source, &witness, 1000);
        }
        other => panic!("constant summarized store was not unique: {other:?}"),
    }
    let declaration =
        ourochronos::parser::parse("PROPERTY answer { ALL_FIXED CELL 0 EQ 81; }").unwrap();
    match GlobalFixedPointSolver::verify_property_bytecode(
        &program,
        &declaration.temporal_properties[0],
        solve_config,
    ) {
        PropertyVerificationResult::Proven {
            exemplar,
            certificate,
            ..
        } => {
            assert_eq!(certificate.completeness, IrCompleteness::Complete);
            check_witness(source, &exemplar, 1000);
        }
        other => panic!("summary property was not proved: {other:?}"),
    }
    let wrong = ourochronos::parser::parse("PROPERTY wrong { ALL_FIXED CELL 0 EQ 80; }").unwrap();
    assert!(matches!(
        GlobalFixedPointSolver::verify_property_bytecode(
            &program,
            &wrong.temporal_properties[0],
            solve_config
        ),
        PropertyVerificationResult::Refuted { .. }
    ));
}

#[test]
fn nonzero_modular_increment_supports_complete_unsat_beyond_unroll_one() {
    let source = "0 ORACLE 17 WHILE { DUP } { SWAP 5 ADD SWAP 1 SUB } POP 0 PROPHECY";
    let program = bytecode(source, 1);
    // k*delta=85 mod2^64 is nonzero, so x+85=x has no word solution.
    for initial in [0_u64, 1, 255, u64::MAX - 2, u64::MAX] {
        let environment = Environment {
            anamnesis: vec![initial],
            ..Environment::default()
        };
        let wanted = expected(source, &environment);
        assert_eq!(wanted.present, vec![initial.wrapping_add(85)]);
        assert_ne!(wanted.present, environment.anamnesis);
        check_ir(&ir(&program, 1, 1), &environment, &wanted);
        assert_eq!(run(&program, &environment, 162).unwrap().0, wanted);
    }
    match GlobalFixedPointSolver::solve_bytecode(&program, config(1, 162)) {
        GlobalSolveResult::ProvenNoFixedPoint(certificate) => {
            assert_eq!(certificate.completeness, IrCompleteness::Complete)
        }
        other => panic!("exact affine no-fixed-state query was not complete: {other:?}"),
    }
    assert!(matches!(
        GlobalFixedPointSolver::solve_bytecode(&program, config(1, 161)),
        GlobalSolveResult::Unknown {
            completeness: Some(IrCompleteness::Complete),
            ..
        }
    ));
    // Modular zero increment preserves the arbitrary incoming word, yielding
    // multiple states instead of an unsound non-wrapping arithmetic result.
    let wrapping =
        "0 ORACLE 2 WHILE { DUP } { SWAP 9223372036854775808 ADD SWAP 1 SUB } POP 0 PROPHECY";
    match GlobalFixedPointSolver::analyze_uniqueness_bytecode(
        &bytecode(wrapping, 1),
        config(1, 100),
    ) {
        GlobalUniquenessResult::Multiple { first, second, .. } => {
            check_witness(wrapping, &first, 100);
            check_witness(wrapping, &second, 100);
        }
        other => panic!("wrapping identity was not multiple: {other:?}"),
    }
}

#[test]
fn near_neighbors_and_dynamic_counters_retain_bounded_completeness() {
    for source in [
        "0 ORACLE WHILE { DUP } { 1 SUB } POP",
        "17 WHILE { DUP } { 1 SUB NOP } POP",
        "17 WHILE { DUP 1 GT } { 1 SUB } POP",
        "16 WHILE { DUP } { 2 SUB } POP",
        "17 WHILE { DUP } { 1 ADD } POP",
        "17 WHILE { DUP DUP AND } { 1 SUB } POP",
    ] {
        let program = bytecode(source, 1);
        assert!(
            matches!(
                ir(&program, 1, 1).completeness,
                IrCompleteness::BoundedLoops {
                    loop_count: 1,
                    unroll_limit: 1
                }
            ),
            "{source}"
        );
        assert!(
            !matches!(
                GlobalFixedPointSolver::solve_bytecode(&program, config(1, 100)),
                GlobalSolveResult::ProvenNoFixedPoint(_)
            ),
            "{source}"
        );
    }
    // A single admitted loop is insufficient when another loop has no summary.
    let mixed = "17 WHILE { DUP } { 1 SUB } POP WHILE { 1 } { NOP }";
    assert!(matches!(
        ir(&bytecode(mixed, 1), 1, 1).completeness,
        IrCompleteness::BoundedLoops { loop_count: 1, .. }
    ));
    let nonterminating = "1 WHILE { DUP } { 2 SUB } POP";
    let environment = Environment {
        anamnesis: vec![0],
        steps: 400,
        ..Environment::default()
    };
    assert_eq!(
        oracle::evaluate(&oracle::parse(nonterminating).unwrap(), &environment).unwrap_err(),
        Fault::StepLimit
    );
    assert!(matches!(
        run(&bytecode(nonterminating, 1), &environment, 100),
        Err(BytecodeVmError::GasExhausted { limit: 100 })
    ));
}

#[test]
fn checked_gas_overflow_cannot_turn_semantic_completeness_into_a_proof() {
    let source = "18446744073709551615 WHILE { DUP } { 1 SUB } POP";
    let program = bytecode(source, 1);
    assert_eq!(ir(&program, 1, 0).completeness, IrCompleteness::Complete);
    match GlobalFixedPointSolver::solve_bytecode(&program, config(1, u64::MAX)) {
        GlobalSolveResult::Unknown {
            reason,
            completeness: Some(IrCompleteness::Complete),
            ..
        } => assert!(reason.contains("no finite executable path bound")),
        other => panic!("overflowing finite fetched count was not unknown: {other:?}"),
    }
    let environment = Environment {
        anamnesis: vec![0],
        steps: 400,
        ..Environment::default()
    };
    assert_eq!(
        oracle::evaluate(&oracle::parse(source).unwrap(), &environment).unwrap_err(),
        Fault::StepLimit
    );
    assert!(matches!(
        run(&program, &environment, 100),
        Err(BytecodeVmError::GasExhausted { limit: 100 })
    ));
}
