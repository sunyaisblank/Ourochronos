//! Reproducible bytecode-authoritative benchmark harness for the three
//! temporal case studies.

use ourochronos::{
    link, BytecodeProgram, BytecodeTransitionAnalyzer, GlobalFixedPointSolver, GlobalSolveConfig,
    GlobalSolveResult, GlobalUniquenessResult, ModuleGraph, ProgramGraphConfig,
    PropertyVerificationResult, StdLib, TemporalPropertyDeclaration,
};
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments.first().map(String::as_str) == Some("--phase") {
        assert_eq!(
            arguments.len(),
            3,
            "usage: bench_case_studies --phase <name> <1..1000 samples>"
        );
        let count = arguments[2].parse::<usize>().expect("integer sample count");
        assert!((1..=1000).contains(&count), "sample count outside 1..1000");
        phase(&arguments[1], count);
        return;
    }
    let iterations = std::env::args()
        .nth(1)
        .map(|value| {
            value
                .parse::<usize>()
                .expect("iterations must be a positive integer")
        })
        .unwrap_or(20);
    assert!(iterations > 0, "iterations must be positive");

    let cases = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/case_studies");
    let (mutual, mutual_properties) = compile_case(cases.join("mutual_exclusion.ouro"));
    let (circular, _) = compile_case(cases.join("circular_dataflow.ouro"));
    let (game, _) = compile_case(cases.join("retrocausal_game.ouro"));

    let mutual_config = GlobalSolveConfig {
        memory_cells: 1,
        ..GlobalSolveConfig::default()
    };
    let circular_config = GlobalSolveConfig {
        memory_cells: 2,
        ..GlobalSolveConfig::default()
    };
    let game_config = ProgramGraphConfig {
        memory_cells: 1,
        cell_bits: 2,
        ..ProgramGraphConfig::default()
    };

    benchmark("self-consistency + 2 properties", iterations, || {
        assert!(matches!(
            GlobalFixedPointSolver::analyze_uniqueness_bytecode(&mutual, mutual_config),
            GlobalUniquenessResult::Multiple { .. }
        ));
        for property in &mutual_properties {
            assert!(matches!(
                GlobalFixedPointSolver::verify_property_bytecode(&mutual, property, mutual_config),
                PropertyVerificationResult::Proven { .. }
            ));
        }
    });

    benchmark("circular dataflow unique solve", iterations, || {
        let result = GlobalFixedPointSolver::solve_bytecode(&circular, circular_config);
        assert!(matches!(result, GlobalSolveResult::Found(_)));
        black_box(result);
        assert!(matches!(
            GlobalFixedPointSolver::analyze_uniqueness_bytecode(&circular, circular_config),
            GlobalUniquenessResult::Unique { .. }
        ));
    });

    benchmark("complete four-state recurrence", iterations, || {
        let analysis = BytecodeTransitionAnalyzer::analyze(&game, game_config)
            .expect("closed complete game domain");
        assert_eq!(analysis.recurrent.recurrent_classes.len(), 3);
        black_box(analysis);
    });
}

fn compile_case(path: PathBuf) -> (BytecodeProgram, Vec<TemporalPropertyDeclaration>) {
    let graph = ModuleGraph::load(&path, StdLib::procedures())
        .unwrap_or_else(|error| panic!("cannot load {}: {error}", path.display()));
    let properties = graph.program().temporal_properties.clone();
    let objects = graph
        .compile_objects()
        .unwrap_or_else(|error| panic!("cannot compile {}: {error}", path.display()));
    let bytecode =
        link(&objects).unwrap_or_else(|error| panic!("cannot link {}: {error}", path.display()));
    (bytecode, properties)
}

fn benchmark(name: &str, iterations: usize, mut operation: impl FnMut()) {
    operation();
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let started = Instant::now();
        operation();
        samples.push(started.elapsed().as_micros());
    }
    samples.sort_unstable();
    let median = samples[samples.len() / 2];
    let p95_index = ((samples.len() * 95).div_ceil(100)).saturating_sub(1);
    let p95 = samples[p95_index];
    println!(
        "{:<38} median {:>8} us   p95 {:>8} us   range {}..{} us   n={}",
        name,
        median,
        p95,
        samples[0],
        samples[samples.len() - 1],
        iterations
    );
}

/// Each phase may run in a fresh process so GNU time can report its peak RSS.
/// Preparation and one warmup precede the timed samples; RSS includes both.
fn phase(name: &str, samples: usize) {
    use num_traits::{One, Zero};
    use ourochronos::bytecode_verifier::verify_default;
    use ourochronos::core::provenance::Provenance;
    use ourochronos::package::{PackageManifest, PortablePackage};
    use ourochronos::temporal::sparse_markov::{
        ExactRational, SparseMarkovChain, SparseMarkovLimits,
    };
    use ourochronos::{
        BytecodeTimeLoop, BytecodeTimeLoopConfig, BytecodeVm, BytecodeVmConfig, BytecodeVmStatus,
        ConvergenceStatus, HirProgram, PagedMemory, PreparedBytecode,
    };

    let cases = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/case_studies");
    let path = cases.join("circular_dataflow.ouro");
    let source = std::fs::read_to_string(&path).expect("canonical fixture");
    let graph = ModuleGraph::load(&path, StdLib::procedures()).unwrap();
    let objects = graph.compile_objects().unwrap();
    let circular = link(&objects).unwrap();
    let solver_config = GlobalSolveConfig {
        memory_cells: 2,
        ..GlobalSolveConfig::default()
    };
    match name {
        "parse" => benchmark(name, samples, || {
            let parsed = ourochronos::parser::parse(&source).unwrap();
            assert_eq!(parsed.temporal_properties.len(), 2);
            black_box(parsed);
        }),
        "modules-hir" => benchmark(name, samples, || {
            let loaded = ModuleGraph::load(&path, StdLib::procedures()).unwrap();
            assert_eq!(loaded.program().temporal_properties.len(), 2);
            black_box(loaded);
        }),
        "objects" => benchmark(name, samples, || {
            let compiled = graph.compile_objects().unwrap();
            assert_eq!(compiled, objects);
            black_box(compiled);
        }),
        "link" => benchmark(name, samples, || {
            let code = link(&objects).unwrap();
            assert_eq!(code, circular);
            black_box(code);
        }),
        "verify" => benchmark(name, samples, || {
            black_box(verify_default(&circular).unwrap());
        }),
        "ir" => benchmark(name, samples, || {
            let ir = GlobalFixedPointSolver::compile_bytecode(&circular, solver_config).unwrap();
            assert!(matches!(ir.completeness, ourochronos::temporal::ir::IrCompleteness::Complete));
            black_box(ir);
        }),
        "smt-text" => {
            let ir = GlobalFixedPointSolver::compile_bytecode(&circular, solver_config).unwrap();
            let expected = ir.to_smt2(true);
            benchmark(name, samples, || {
                let encoded = ir.to_smt2(true);
                assert_eq!(encoded, expected);
                black_box(encoded);
            });
        }
        "package" => {
            let mut manifest = PackageManifest::current("circular");
            manifest.memory_cells = 2;
            let package = PortablePackage::new(manifest, circular).unwrap();
            let expected = package.to_bytes().unwrap();
            benchmark(name, samples, || {
                let encoded = package.to_bytes().unwrap();
                assert_eq!(encoded, expected);
                let decoded = PortablePackage::from_bytes(&encoded).unwrap();
                assert_eq!(decoded, package);
                black_box(decoded);
            });
        }
        "vm-dense" | "vm-wide" | "vm-repeated" => {
            let dense = name == "vm-dense";
            let (text, cells) = if dense {
                ("0 WHILE { DUP 4096 LT } { DUP DUP PROPHECY 1 ADD } POP", 4096)
            } else {
                ("7 999999 PROPHECY 999999 PRESENT OUTPUT", 1_000_000)
            };
            let parsed = ourochronos::parser::parse(text).unwrap();
            let code = BytecodeProgram::compile(&HirProgram::resolve(&parsed).unwrap()).unwrap();
            let prepared = PreparedBytecode::new(code.clone()).unwrap();
            let vm = BytecodeVm::with_config(BytecodeVmConfig { max_instructions: 100_000, ..BytecodeVmConfig::default() });
            let memory = PagedMemory::with_size(cells).unwrap();
            benchmark(name, samples, || {
                let result = if name == "vm-repeated" {
                    vm.run_prepared(&prepared, &memory)
                } else { vm.run(&code, &memory) }.unwrap();
                assert_eq!(result.status, BytecodeVmStatus::Finished);
                assert!(result.stack.is_empty());
                assert!(result.effects.is_empty());
                if dense {
                    assert!(result.output.is_empty());
                    for i in 0..cells { assert_eq!(result.present.get(i as u64).unwrap().val, i as u64); }
                } else {
                    assert_eq!(result.present.numeric_sparse_state(), vec![(999999, 7)]);
                    assert_eq!(result.output, vec![ourochronos::core::OutputItem::Val(ourochronos::Value::new(7))]);
                }
                assert!(memory.numeric_sparse_state().is_empty());
                black_box(result);
            });
        }
        "provenance" => benchmark(name, samples, || {
            let mut deps = Provenance::none();
            for address in 0..512 { deps = deps.merge(&Provenance::single(address)); }
            assert!(deps.is_saturated());
            black_box(deps);
        }),
        "orbit-journal" => {
            let parsed = ourochronos::parser::parse("0 ORACLE DUP 0 GT IF { 1 SUB } ELSE { POP 0 } 0 PROPHECY").unwrap();
            let code = BytecodeProgram::compile(&HirProgram::resolve(&parsed).unwrap()).unwrap();
            let driver = BytecodeTimeLoop::new(BytecodeTimeLoopConfig {
                memory_cells: 1, max_epochs: 66, initial_state: vec![(0, 64)],
                ..BytecodeTimeLoopConfig::default()
            }).unwrap();
            benchmark(name, samples, || {
                match driver.run(&code) {
                    ConvergenceStatus::Consistent { memory, epochs, .. } => {
                        assert_eq!(epochs, 65);
                        assert_eq!(memory.read(0).val, 0);
                    }
                    other => panic!("expected 65 exact epochs: {other:?}"),
                }
            });
        }
        "exact" => {
            let rows = (0..128).map(|i| vec![(i, ExactRational::new(1.into(), 2.into())),
                ((i + 1) % 128, ExactRational::new(1.into(), 2.into()))]).collect();
            let chain = SparseMarkovChain::new(rows, SparseMarkovLimits::default()).unwrap();
            benchmark(name, samples, || {
                let analysis = chain.analyze(&[false; 128], ExactRational::one(), ExactRational::zero()).unwrap();
                assert_eq!(analysis.family.extremal.len(), 1);
                assert!(analysis.family.extremal[0].certificate.is_stationary());
                for (_, weight) in &analysis.family.extremal[0].weights {
                    assert_eq!(*weight, ExactRational::new(1.into(), 128.into()));
                }
                black_box(analysis);
            });
        }
        _ => panic!("phase must be parse, modules-hir, objects, link, verify, ir, smt-text, package, vm-dense, vm-wide, vm-repeated, provenance, orbit-journal, or exact"),
    }
}
