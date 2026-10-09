//! Configurable exact models and a deliberately restricted frozen-VM adapter.
//! Run each measured workload in its own process to measure peak RSS externally.
use num_bigint::BigInt;
use num_traits::{One, Zero};
use ourochronos::temporal::sparse_markov::{
    ExactRational as R, SparseMarkovChain, SparseMarkovLimits,
};
use ourochronos::temporal::vm_stochastic::{
    extract_vm_markov, RandomTapeScenario, VmStochasticConfig,
};
use ourochronos::{BytecodeProgram, HirProgram};
use std::time::Instant;

fn integer(text: &str, maximum: usize) -> Result<usize, String> {
    text.parse::<usize>()
        .ok()
        .filter(|value| *value > 0 && *value <= maximum)
        .ok_or_else(|| format!("expected an integer in 1..={maximum}"))
}
fn run() -> Result<(), String> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let start = Instant::now();
    let (chain, accepting, expected, vm_work) = match arguments.first().map(String::as_str) {
        Some("absorbing") if arguments.len() == 2 => {
            let states = integer(&arguments[1], 16_384)?;
            let rows = (0..states).map(|state| vec![(state, R::one())]).collect();
            (SparseMarkovChain::new(rows, SparseMarkovLimits::default()),
                (0..states).map(|state| state >= states / 2).collect::<Vec<_>>(), None, 0)
        }
        Some("cycle") if arguments.len() == 3 => {
            let states = integer(&arguments[1], 16_384)?;
            let bits = integer(&arguments[2], 512)?;
            let base = BigInt::one() << bits;
            let mut rows = Vec::with_capacity(states);
            for state in 0..states {
                let rate = R::new(BigInt::one(), &base + BigInt::from(state));
                rows.push(vec![(state, R::one() - &rate), ((state + 1) % states, rate)]);
            }
            // Analytic reference: pi_i is proportional to the mean waiting
            // time 1/rate_i, including the singleton-cycle special case.
            let normalization = &base * BigInt::from(states) + BigInt::from(states * (states - 1) / 2);
            let weights = (0..states).map(|state| R::new(&base + BigInt::from(state), normalization.clone())).collect::<Vec<_>>();
            (SparseMarkovChain::new(rows, SparseMarkovLimits::default()),
                (0..states).map(|state| state >= states / 2).collect::<Vec<_>>(), Some(weights), 0)
        }
        Some("vm") if arguments.len() == 4 => {
            let numerator = integer(&arguments[1], 1_000_000)?;
            let denominator = integer(&arguments[2], 1_000_000)?;
            if numerator >= denominator { return Err("VM probability must satisfy 0 < numerator < denominator".into()); }
            // Ordinary source admission rejects live INPUT/RANDOM in a scope.
            // This explicitly constructed bytecode instead enters the separate
            // finite adapter, whose own admission/totality checks are mandatory.
            let parsed = ourochronos::parser::parse("TEMPORAL 0 2 BITS 1 { INPUT POP 0 ORACLE 0 PROPHECY RANDOM 1 PROPHECY 1 PRESENT OUTPUT }")
                .map_err(|error| error.to_string())?;
            let code = BytecodeProgram::compile(&HirProgram::resolve(&parsed).map_err(|error| format!("{error:?}"))?)
                .map_err(|error| error.to_string())?;
            let probability = R::new(numerator.into(), denominator.into());
            let scenarios = vec![
                RandomTapeScenario { words: vec![0], probability: R::one() - &probability },
                RandomTapeScenario { words: vec![1], probability },
            ];
            let extracted = extract_vm_markov(&code, &VmStochasticConfig {
                memory_cells: 2, max_instructions: 64, input: vec![42], ..VmStochasticConfig::default()
            }, &scenarios).map_err(|error| error.to_string())?;
            let readout_bit = match arguments[3].as_str() { "random" => 1, "tag" => 0, _ => return Err("VM readout must be random or tag".into()) };
            let accepting = (0..extracted.chain.states()).map(|state| state & (1 << readout_bit) != 0).collect();
            (Ok(extracted.chain), accepting, None, extracted.stats.instructions_executed)
        }
        _ => return Err("usage: stochastic_models absorbing <states> | cycle <states> <denominator-bits> | vm <numerator> <denominator> <random|tag>".into()),
    };
    let chain = chain.map_err(|error| error.to_string())?;
    let built = start.elapsed();
    let analysis = chain
        .analyze(
            &accepting,
            R::new(2.into(), 3.into()),
            R::new(1.into(), 3.into()),
        )
        .map_err(|error| error.to_string())?;
    for distribution in &analysis.family.extremal {
        if !distribution.certificate.is_stationary() {
            return Err("exact stationary residual failed".into());
        }
    }
    if let Some(expected) = expected {
        if analysis.family.extremal.len() != 1 {
            return Err("analytic cycle must have one recurrent class".into());
        }
        let actual = &analysis.family.extremal[0].weights;
        if actual.len() != expected.len()
            || actual
                .iter()
                .any(|(state, weight)| *weight != expected[*state])
        {
            return Err(
                "exact cycle weights disagree with independent waiting-time formula".into(),
            );
        }
    }
    let minimum = &analysis.minimum.probability;
    let maximum = &analysis.maximum.probability;
    if minimum < &R::zero() || maximum > &R::one() {
        return Err("readout escaped probability domain".into());
    }
    println!("model,states,edges,classes,integer_bits,matrix_entries,operations,build_us,total_us,vm_instructions,min_probability,max_probability,decision");
    println!(
        "{},{},{},{},{},{},{},{},{},{},{},{},{:?}",
        arguments[0],
        chain.states(),
        chain.edges(),
        analysis.family.extremal.len(),
        analysis
            .family
            .stats
            .peak_integer_bits
            .max(chain.admission_stats().peak_integer_bits),
        analysis.family.stats.peak_matrix_entries,
        analysis.family.stats.charged_operations,
        built.as_micros(),
        start.elapsed().as_micros(),
        vm_work,
        minimum,
        maximum,
        analysis.decision
    );
    Ok(())
}
fn main() {
    if std::env::args().nth(1).as_deref() == Some("supervise") {
        match supervise() {
            Ok(code) => std::process::exit(code),
            Err(error) => {
                eprintln!("Exact stochastic model refused: {error}");
                std::process::exit(1);
            }
        }
    }
    if let Err(error) = run() {
        eprintln!("Exact stochastic model refused: {error}");
        std::process::exit(1);
    }
}

fn supervise() -> Result<i32, String> {
    use ourochronos::runtime::isolation::{run_bounded, ProcessLimits};
    use std::io::Write;
    use std::process::Command;
    use std::time::Duration;

    let arguments: Vec<String> = std::env::args().skip(2).collect();
    if arguments.len() < 3 || arguments[2] == "supervise" {
        return Err(
            "usage: stochastic_models supervise <wall-ms> <memory-MiB> <model arguments>".into(),
        );
    }
    let milliseconds = integer(&arguments[0], 3_600_000)?;
    let memory = integer(&arguments[1], 16_384)?;
    let mut command = Command::new(std::env::current_exe().map_err(|error| error.to_string())?);
    command.args(&arguments[2..]);
    let output = run_bounded(
        command,
        ProcessLimits {
            wall_time: Duration::from_millis(milliseconds as u64),
            address_space_bytes: memory as u64 * 1024 * 1024,
            output_bytes: 64 * 1024,
        },
        None,
    )
    .map_err(|error| error.to_string())?;
    // A killed dependency or allocator abort supplies no usable partial result.
    if output.stopped.is_some() || !matches!(output.status.code(), Some(0 | 1)) {
        eprintln!(
            "UNKNOWN: supervised exact analysis stopped ({:?}, {})",
            output.stopped, output.status
        );
        return Ok(3);
    }
    std::io::stdout()
        .write_all(&output.stdout)
        .map_err(|error| error.to_string())?;
    std::io::stderr()
        .write_all(&output.stderr)
        .map_err(|error| error.to_string())?;
    Ok(output.status.code().expect("normal exit checked above"))
}
