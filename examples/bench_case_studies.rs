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
        "{:<38} median {:>8} us   p95 {:>8} us   n={}",
        name, median, p95, iterations
    );
}
