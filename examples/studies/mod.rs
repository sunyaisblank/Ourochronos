//! Bounded complete workflows shared by the runnable example and its tests.
//! Parameters become literal source constants; no mutable INPUT or host effect
//! is admitted. Conventional answers use ordinary arithmetic and orbit walks.

use ourochronos::{
    admit_program, AdmissionConfig, BoundsPolicy, BytecodeProgram, BytecodeTimeLoop,
    BytecodeTimeLoopConfig, BytecodeTransitionAnalyzer, BytecodeVm, BytecodeVmConfig,
    BytecodeVmStatus, ConvergenceStatus, GlobalFixedPointSolver, GlobalSolveConfig,
    GlobalUniquenessResult, IrCompleteness, OutputItem, PackageManifest, PackageWitness,
    PagedMemory, PortablePackage, ProgramGraphConfig, PropertyVerificationResult, Value,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::time::Instant;

pub const MAX_GAS: u64 = 1024;
const MAX_OUTPUT_ITEMS: usize = 8;
const MAX_SAVED_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Study {
    Exclusion {
        contenders: usize,
        eligible: u64,
    },
    Dataflow {
        nodes: usize,
        bits: u8,
        gains: Vec<u64>,
        biases: Vec<u64>,
    },
    Game {
        actions: usize,
        successors: Vec<usize>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecurrentClass {
    /// Directed cycle, canonically rotated to its smallest state.
    pub cycle: Vec<usize>,
    pub basin_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conventional {
    pub fixed_states: Vec<Vec<u64>>,
    pub candidate_trials: u64,
    /// Mask predicate evaluations, ring node evaluations, or orbit edge
    /// traversals. These are NOT the same cost unit as fetched VM records.
    pub operations: u64,
    pub operation_unit: &'static str,
    pub transitions: Vec<usize>,
    pub recurrent: Vec<RecurrentClass>,
}

#[derive(Debug)]
pub struct Replay {
    pub state: Vec<u64>,
    pub output: Vec<u64>,
    pub fetched_records: u64,
    pub package: Vec<u8>,
    pub package_output: Vec<u64>,
}

#[derive(Debug)]
pub struct Report {
    pub study: Study,
    pub gas: u64,
    pub source: String,
    pub code: BytecodeProgram,
    pub code_sha256: String,
    pub conventional: Conventional,
    pub solver: GlobalUniquenessResult,
    pub property: PropertyVerificationResult,
    pub replays: Vec<Replay>,
    pub graph_fetched_records: u64,
    pub orbit_package: Option<Vec<u8>>,
    pub orbit_outcome: Option<String>,
    pub conventional_ns: u128,
    pub admission_ns: u128,
    pub solver_ns: u128,
    pub replay_package_ns: u128,
}

impl Study {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Exclusion {
                contenders,
                eligible,
            } => {
                if !(1..=8).contains(contenders) || *eligible >= (1 << contenders) {
                    return Err(
                        "exclusion requires 1..8 contenders and an in-domain eligible mask".into(),
                    );
                }
            }
            Self::Dataflow {
                nodes,
                bits,
                gains,
                biases,
            } => {
                if !(1..=8).contains(nodes)
                    || !(1..=4).contains(bits)
                    || biases.len() != *nodes
                    || gains.len() != *nodes
                    || gains.iter().any(|&gain| gain >= (1 << bits))
                    || biases.iter().any(|&bias| bias >= (1 << bits))
                {
                    return Err("dataflow requires 1..8 nodes, 1..4 bits, exact bias count and in-domain gain/biases".into());
                }
            }
            Self::Game {
                actions,
                successors,
            } => {
                if !(1..=16).contains(actions)
                    || successors.len() != *actions
                    || successors.iter().any(|&target| target >= *actions)
                {
                    return Err("game requires 1..16 actions and exactly one in-domain successor per action".into());
                }
            }
        }
        Ok(())
    }

    pub fn memory_cells(&self) -> usize {
        match self {
            Self::Dataflow { nodes, .. } => *nodes,
            _ => 1,
        }
    }

    fn bits(&self) -> u8 {
        match self {
            Self::Exclusion { contenders, .. } => *contenders as u8,
            Self::Dataflow { bits, .. } => *bits,
            Self::Game { actions, .. } => {
                (usize::BITS - (actions.saturating_sub(1)).leading_zeros()).max(1) as u8
            }
        }
    }

    /// Complete conventional formulations, independent of generated source,
    /// AST, lowering, VM dispatch, and production graph classification.
    pub fn conventional(&self) -> Result<Conventional, String> {
        self.validate()?;
        let mut answer = Conventional {
            fixed_states: Vec::new(),
            candidate_trials: 0,
            operations: 0,
            operation_unit: "",
            transitions: Vec::new(),
            recurrent: Vec::new(),
        };
        match self {
            Self::Exclusion {
                contenders,
                eligible,
            } => {
                answer.operation_unit = "ownership-mask predicates";
                for mask in 0..1u64 << contenders {
                    answer.candidate_trials += 1;
                    answer.operations += 1;
                    if mask != 0 && mask.count_ones() == 1 && mask & eligible == mask {
                        answer.fixed_states.push(vec![mask]);
                    }
                }
            }
            Self::Dataflow {
                nodes,
                bits,
                gains,
                biases,
            } => {
                answer.operation_unit = "ring-node modular multiply/add evaluations";
                let modulus = 1u64 << bits;
                // Any solution has one of these x0 values. From that value,
                // reverse traversal determines xn-1, xn-2, ..., x1 uniquely;
                // the final x0 equation tests closure, with no division or
                // invertibility assumption on the gain.
                for initial in 0..modulus {
                    answer.candidate_trials += 1;
                    let mut words = vec![0; *nodes];
                    let mut next = initial;
                    for node in (0..*nodes).rev() {
                        words[node] = (gains[node] * next + biases[node]) % modulus;
                        next = words[node];
                        answer.operations += 1;
                    }
                    if words[0] == initial {
                        answer.fixed_states.push(words);
                    }
                }
            }
            Self::Game {
                actions,
                successors,
            } => {
                answer.operation_unit = "independent orbit edge traversals";
                answer.transitions = (0..1usize << self.bits())
                    .map(|state| {
                        if state < *actions {
                            successors[state]
                        } else {
                            0
                        }
                    })
                    .collect();
                answer.fixed_states = answer
                    .transitions
                    .iter()
                    .enumerate()
                    .filter(|&(state, &next)| state == next)
                    .map(|(state, _)| vec![state as u64])
                    .collect();
                let (classes, traversals) = recurrent_classes(&answer.transitions);
                answer.recurrent = classes;
                answer.operations = traversals;
                answer.candidate_trials = answer.transitions.len() as u64;
            }
        }
        Ok(answer)
    }

    fn source(&self, conventional: &Conventional) -> String {
        let property = if conventional.fixed_states.is_empty() {
            // A syntactically bounded predicate whose truth is explicitly
            // vacuous when complete enumeration and the solver find no point.
            "CELL 0 EQ 0".to_owned()
        } else if let Self::Dataflow { bits, .. } = self {
            // In a fixed ring, x0 uniquely reconstructs xn-1 ... x1 using
            // the retained equations. This complete first-cell membership
            // predicate therefore covers the full conventional solution set.
            // It avoids a large equivalent full-vector DNF backend proof.
            if conventional.fixed_states.len() == 1usize << bits {
                format!("CELL 0 LTE {}", (1u64 << bits) - 1)
            } else {
                conventional
                    .fixed_states
                    .iter()
                    .map(|state| format!("CELL 0 EQ {}", state[0]))
                    .collect::<Vec<_>>()
                    .join(" OR ")
            }
        } else {
            conventional
                .fixed_states
                .iter()
                .map(|state| {
                    format!(
                        "({})",
                        state
                            .iter()
                            .enumerate()
                            .map(|(cell, value)| format!("CELL {cell} EQ {value}"))
                            .collect::<Vec<_>>()
                            .join(" AND ")
                    )
                })
                .collect::<Vec<_>>()
                .join(" OR ")
        };
        let body = match self {
            Self::Exclusion {
                contenders,
                eligible,
            } => {
                let owners: Vec<_> = (0..*contenders)
                    .map(|bit| 1u64 << bit)
                    .filter(|owner| owner & eligible != 0)
                    .collect();
                let condition = match owners.split_first() {
                    None => "0".to_owned(),
                    Some((first, rest)) => format!(
                        "DUP {first} EQ {}",
                        rest.iter()
                            .map(|owner| format!("OVER {owner} EQ OR"))
                            .collect::<Vec<_>>()
                            .join(" ")
                    ),
                };
                format!(
                    "0 ORACLE {condition} IF {{ DUP 0 PROPHECY OUTPUT }} ELSE {{ POP PARADOX }}"
                )
            }
            Self::Dataflow {
                nodes,
                bits,
                gains,
                biases,
            } => {
                let mask = (1u64 << bits) - 1;
                let mut body = String::new();
                for (node, bias) in biases.iter().enumerate() {
                    write!(
                        body,
                        "{} ORACLE {} MUL {bias} ADD {mask} AND {node} PROPHECY ",
                        (node + 1) % nodes,
                        gains[node]
                    )
                    .unwrap();
                }
                for node in 0..*nodes {
                    write!(body, "{node} PRESENT OUTPUT ").unwrap();
                }
                body
            }
            Self::Game { successors, .. } => {
                // <=16 entries at <=4 bits each fit one word. Unused encoded
                // entries stay zero. A fixed state's whole-word masked frame
                // makes its index in-domain; the ordinary SHR modulo64 rule
                // therefore selects exactly the retained literal table entry.
                let bits = self.bits();
                let packed = successors
                    .iter()
                    .enumerate()
                    .fold(0u64, |table, (state, &next)| {
                        table | ((next as u64) << (state * bits as usize))
                    });
                let mask = (1u64 << bits) - 1;
                format!("{packed} 0 ORACLE {bits} MUL SHR {mask} AND DUP 0 PROPHECY OUTPUT")
            }
        };
        format!(
            "# Frozen workflow parameters: {self:?}\n\
            PROPERTY conventional_solutions {{ ALL_FIXED {property}; }}\n\
            TEMPORAL 0 {} BITS {} {{\n{body}\n}}\n",
            self.memory_cells(),
            self.bits()
        )
    }
}

fn recurrent_classes(edges: &[usize]) -> (Vec<RecurrentClass>, u64) {
    let mut basins = BTreeMap::<Vec<usize>, usize>::new();
    let mut traversals = 0;
    for start in 0..edges.len() {
        let mut path = Vec::new();
        let mut current = start;
        loop {
            if let Some(begin) = path.iter().position(|&state| state == current) {
                let mut cycle = path[begin..].to_vec();
                let least = cycle
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, state)| *state)
                    .unwrap()
                    .0;
                cycle.rotate_left(least);
                *basins.entry(cycle).or_default() += 1;
                break;
            }
            path.push(current);
            current = edges[current];
            traversals += 1;
        }
    }
    (
        basins
            .into_iter()
            .map(|(cycle, basin_size)| RecurrentClass { cycle, basin_size })
            .collect(),
        traversals,
    )
}

fn vm_config(gas: u64) -> BytecodeVmConfig {
    BytecodeVmConfig {
        max_instructions: gas,
        memory_bounds: BoundsPolicy::Error,
        max_output_items: MAX_OUTPUT_ITEMS,
        max_output_bytes: 4096,
        max_stack_depth: 128,
        max_call_depth: 16,
        max_temporal_depth: 1,
        ..BytecodeVmConfig::default()
    }
}

fn output_words(output: &[OutputItem]) -> Result<Vec<u64>, String> {
    output
        .iter()
        .map(|item| match item {
            OutputItem::Val(value) => Ok(value.val),
            OutputItem::Char(_) => Err("workflow unexpectedly emitted a character".into()),
        })
        .collect()
}

fn paged(words: &[u64]) -> Result<PagedMemory, String> {
    let mut memory = PagedMemory::with_size(words.len()).map_err(|error| error.to_string())?;
    for (cell, &word) in words.iter().enumerate() {
        memory
            .write(cell as u64, Value::new(word))
            .map_err(|error| error.to_string())?;
    }
    Ok(memory)
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn package_name(study: &Study, policy: &str) -> String {
    // Exact retained parameter text distinguishes operationally equivalent
    // tables with different active-action declarations as well as different
    // literal transitions. The existing package witness binds its manifest.
    format!("{policy}-{}", hash(format!("{study:?}").as_bytes()))
}

fn expected_output(study: &Study, state: &[u64]) -> Vec<u64> {
    match study {
        Study::Dataflow { .. } => state.to_vec(),
        _ => vec![state[0]],
    }
}

fn ordinary_transition(study: &Study, state: &[u64]) -> Option<Vec<u64>> {
    match study {
        Study::Exclusion { eligible, .. } => {
            let mask = state[0];
            (mask != 0 && mask.count_ones() == 1 && mask & eligible == mask).then(|| vec![mask])
        }
        Study::Dataflow {
            nodes,
            bits,
            gains,
            biases,
        } => Some(
            (0..*nodes)
                .map(|node| (gains[node] * state[(node + 1) % nodes] + biases[node]) % (1 << bits))
                .collect(),
        ),
        Study::Game {
            actions,
            successors,
        } => Some(vec![if state[0] < *actions as u64 {
            successors[state[0] as usize] as u64
        } else {
            0
        }]),
    }
}

fn no_point_orbit(report: &mut Report) -> Result<(), String> {
    let manifest = PackageManifest::with_runtime(
        package_name(&report.study, "no-point-orbit"),
        report.study.memory_cells(),
        report.gas,
        BoundsPolicy::Error,
    );
    let package =
        PortablePackage::new(manifest, report.code.clone()).map_err(|error| error.to_string())?;
    let bytes = package.to_bytes().map_err(|error| error.to_string())?;
    let decoded = PortablePackage::from_bytes(&bytes).map_err(|error| error.to_string())?;
    if decoded.to_bytes().map_err(|error| error.to_string())? != bytes {
        return Err("no-point package encoding differs".into());
    }
    let mut state = vec![0; report.study.memory_cells()];
    let mut history = Vec::new();
    let mut prediction = "timeout after 32 epochs".to_owned();
    let mut predicted_period = None;
    let mut predicted_paradox = None;
    for epoch in 0..32 {
        if let Some(begin) = history.iter().position(|prior| prior == &state) {
            let period = history.len() - begin;
            prediction =
                format!("oscillation period {period}; no selected point on orbit from zero");
            predicted_period = Some(period);
            break;
        }
        history.push(state.clone());
        match ordinary_transition(&report.study, &state) {
            Some(next) if next == state => {
                return Err(
                    "conventional no-point instance unexpectedly has a point on its orbit".into(),
                )
            }
            Some(next) => state = next,
            None => {
                predicted_paradox = Some(epoch + 1);
                prediction = format!("paradox at epoch {}; no eligible point", epoch + 1);
                break;
            }
        }
    }
    let driver = BytecodeTimeLoop::new(BytecodeTimeLoopConfig {
        memory_cells: report.study.memory_cells(),
        max_epochs: 32,
        vm: vm_config(report.gas),
        ..BytecodeTimeLoopConfig::default()
    })
    .map_err(|error| error.to_string())?;
    let agrees = match driver.run(&decoded.program) {
        ConvergenceStatus::Paradox { epoch, .. } => predicted_paradox == Some(epoch),
        ConvergenceStatus::Oscillation { period, .. } => predicted_period == Some(period),
        ConvergenceStatus::Timeout { max_epochs: 32 } => {
            predicted_period.is_none() && predicted_paradox.is_none()
        }
        _ => false,
    };
    if !agrees {
        return Err("no-point package orbit differs from independent bounded history".into());
    }
    report.orbit_package = Some(bytes);
    report.orbit_outcome = Some(prediction);
    Ok(())
}

fn replay_point(
    code: &BytecodeProgram,
    study: &Study,
    state: &[u64],
    gas: u64,
) -> Result<Replay, String> {
    let execution = BytecodeVm::with_config(vm_config(gas))
        .run(code, &paged(state)?)
        .map_err(|error| error.to_string())?;
    let actual: Vec<_> = (0..state.len())
        .map(|cell| {
            execution
                .present
                .read_checked(cell as u64, BoundsPolicy::Error, Default::default())
                .unwrap()
                .val
        })
        .collect();
    let output = output_words(&execution.output)?;
    if execution.status != BytecodeVmStatus::Finished
        || actual != state
        || !execution.stack.is_empty()
        || output != expected_output(study, state)
        || !execution.effects.is_empty()
        || execution.observation_transcript()
            != ourochronos::temporal::transaction::ObservationTranscript::default()
    {
        return Err("independent conventional point disagrees with full numeric VM replay".into());
    }
    let manifest = PackageManifest::with_runtime(
        package_name(study, "application-study"),
        state.len(),
        gas,
        BoundsPolicy::Error,
    )
    .embedded_point_witness();
    let sparse = state
        .iter()
        .enumerate()
        .filter(|&(_, &value)| value != 0)
        .map(|(cell, &value)| (cell as u64, value))
        .collect();
    let witness =
        PackageWitness::replay_bound(&manifest, code, sparse, execution.instructions_executed)
            .map_err(|error| error.to_string())?;
    let package = PortablePackage::with_replay_witness(manifest, code.clone(), witness)
        .map_err(|error| error.to_string())?;
    let bytes = package.to_bytes().map_err(|error| error.to_string())?;
    let decoded = PortablePackage::from_bytes(&bytes).map_err(|error| error.to_string())?;
    if decoded.to_bytes().map_err(|error| error.to_string())? != bytes {
        return Err("package encoding is not deterministic".into());
    }
    let driver = BytecodeTimeLoop::new(BytecodeTimeLoopConfig {
        memory_cells: state.len(),
        max_epochs: 2,
        initial_state: decoded.witness.as_ref().unwrap().state.clone(),
        vm: vm_config(decoded.manifest.max_instructions),
        ..BytecodeTimeLoopConfig::default()
    })
    .map_err(|error| error.to_string())?;
    let package_output = match driver.run(&decoded.program) {
        ConvergenceStatus::Consistent {
            memory,
            output,
            epochs: 1,
        } => {
            if (0..state.len()).any(|cell| memory.read(cell as u64).val != state[cell]) {
                return Err("package state differs".into());
            }
            output_words(&output)?
        }
        other => {
            return Err(format!(
                "embedded point package did not complete one replay: {other:?}"
            ))
        }
    };
    if output != package_output {
        return Err("package output differs from direct replay".into());
    }
    Ok(Replay {
        state: state.to_vec(),
        output,
        fetched_records: execution.instructions_executed,
        package: bytes,
        package_output,
    })
}

fn check_solver(
    solver: &GlobalUniquenessResult,
    conventional: &Conventional,
    study: &Study,
) -> Result<(), String> {
    let valid = |witness: &ourochronos::FixedPointWitness| {
        let state: Vec<_> = (0..study.memory_cells())
            .map(|cell| witness.memory.read(cell as u64).val)
            .collect();
        witness.completeness == IrCompleteness::Complete
            && witness.is_replay_verified()
            && conventional.fixed_states.contains(&state)
            && output_words(&witness.output)
                .is_ok_and(|output| output == expected_output(study, &state))
    };
    let agrees = match solver {
        GlobalUniquenessResult::NoFixedPoint(certificate) => {
            conventional.fixed_states.is_empty()
                && certificate.completeness == IrCompleteness::Complete
        }
        GlobalUniquenessResult::Unique {
            witness,
            certificate,
        } => {
            conventional.fixed_states.len() == 1
                && valid(witness)
                && certificate.completeness == IrCompleteness::Complete
        }
        GlobalUniquenessResult::Multiple {
            first,
            second,
            differing_cells,
        } => {
            conventional.fixed_states.len() > 1
                && valid(first)
                && valid(second)
                && !differing_cells.is_empty()
        }
        // Unknown retains its status even though the separate conventional
        // method finishes. No package is labeled a solver-selected result.
        GlobalUniquenessResult::Unknown { .. } => true,
        _ => false,
    };
    if agrees {
        Ok(())
    } else {
        Err(format!(
            "solver classification disagrees with conventional complete result: {solver:?}"
        ))
    }
}

pub fn run(study: Study, gas: u64) -> Result<Report, String> {
    study.validate()?;
    if !(1..=MAX_GAS).contains(&gas) {
        return Err("gas must be in 1..1024".into());
    }
    let clock = Instant::now();
    let conventional = study.conventional()?;
    let conventional_ns = clock.elapsed().as_nanos();
    let source = study.source(&conventional);
    if source.len() > 32768 {
        return Err("generated source exceeds workflow ceiling".into());
    }
    let clock = Instant::now();
    let parsed = ourochronos::parser::parse(&source)
        .map_err(|error| format!("generated source parse: {error}"))?;
    let code = admit_program(
        &parsed,
        AdmissionConfig {
            memory_cells: study.memory_cells(),
        },
    )
    .map_err(|error| format!("canonical admission: {error}"))?
    .into_program();
    let bytes = code.to_bytes().map_err(|error| error.to_string())?;
    let code_sha256 = hash(&bytes);
    let admission_ns = clock.elapsed().as_nanos();
    let config = GlobalSolveConfig {
        memory_cells: study.memory_cells(),
        loop_unroll_limit: 0,
        solver_timeout_ms: 5000,
        max_instructions: gas,
        bounds_policy: BoundsPolicy::Error,
    };
    let clock = Instant::now();
    let solver = GlobalFixedPointSolver::analyze_uniqueness_bytecode(&code, config);
    check_solver(&solver, &conventional, &study)?;
    let property = GlobalFixedPointSolver::verify_property_bytecode(
        &code,
        &parsed.temporal_properties[0],
        config,
    );
    match (&solver, &property) {
        (_, PropertyVerificationResult::Unknown { .. }) => {}
        (_, PropertyVerificationResult::Vacuous { no_fixed_point, .. })
            if conventional.fixed_states.is_empty()
                && no_fixed_point.completeness == IrCompleteness::Complete => {}
        (_, PropertyVerificationResult::Proven { certificate, .. })
            if !conventional.fixed_states.is_empty()
                && certificate.completeness == IrCompleteness::Complete => {}
        _ => {
            return Err(format!(
                "all-point membership property has unexpected status: {property:?}"
            ))
        }
    }
    let solver_ns = clock.elapsed().as_nanos();
    let mut report = Report {
        study,
        gas,
        source,
        code,
        code_sha256,
        conventional,
        solver,
        property,
        replays: Vec::new(),
        graph_fetched_records: 0,
        orbit_package: None,
        orbit_outcome: None,
        conventional_ns,
        admission_ns,
        solver_ns,
        replay_package_ns: 0,
    };
    if report.is_unknown() {
        return Ok(report);
    }
    let clock = Instant::now();
    for state in &report.conventional.fixed_states {
        report
            .replays
            .push(replay_point(&report.code, &report.study, state, gas)?);
    }
    let selected: Vec<_> = match &report.solver {
        GlobalUniquenessResult::Unique { witness, .. } => vec![witness],
        GlobalUniquenessResult::Multiple { first, second, .. } => vec![first, second],
        _ => Vec::new(),
    };
    for witness in selected {
        let state: Vec<_> = (0..report.study.memory_cells())
            .map(|cell| witness.memory.read(cell as u64).val)
            .collect();
        let replay = report
            .replays
            .iter()
            .find(|replay| replay.state == state)
            .ok_or("selected solver state has no conventional replay")?;
        if replay.fetched_records != witness.instructions_executed
            || replay.output != output_words(&witness.output)?
        {
            return Err("selected solver instruction/output evidence differs from fresh conventional replay".into());
        }
    }
    if matches!(report.study, Study::Game { .. }) {
        let graph = BytecodeTransitionAnalyzer::analyze(
            &report.code,
            ProgramGraphConfig {
                memory_cells: 1,
                cell_bits: report.study.bits(),
                max_states: 1 << report.study.bits(),
                max_instructions: gas,
                bounds_policy: BoundsPolicy::Error,
            },
        )
        .map_err(|error| error.to_string())?;
        if graph.graph.successors() != report.conventional.transitions {
            return Err("game VM transition table differs from conventional rules".into());
        }
        let mut classes: Vec<_> = graph
            .recurrent
            .recurrent_classes
            .iter()
            .cloned()
            .zip(graph.recurrent.basin_sizes.iter().copied())
            .map(|(cycle, basin_size)| RecurrentClass { cycle, basin_size })
            .collect();
        classes.sort_by(|a, b| a.cycle.cmp(&b.cycle));
        if classes != report.conventional.recurrent {
            return Err("game recurrent classes or basins disagree".into());
        }
        for (state, output) in graph.outputs.iter().enumerate() {
            if output_words(output)? != vec![report.conventional.transitions[state] as u64] {
                return Err("game typed numeric readout differs".into());
            }
        }
        report.graph_fetched_records = graph.instructions_executed.iter().sum();
        let package = PortablePackage::new(
            PackageManifest::with_runtime(
                package_name(&report.study, "game-orbit"),
                1,
                gas,
                BoundsPolicy::Error,
            ),
            report.code.clone(),
        )
        .map_err(|error| error.to_string())?;
        let bytes = package.to_bytes().map_err(|error| error.to_string())?;
        let decoded = PortablePackage::from_bytes(&bytes).map_err(|error| error.to_string())?;
        if decoded.to_bytes().map_err(|error| error.to_string())? != bytes {
            return Err("orbit package encoding differs".into());
        }
        let driver = BytecodeTimeLoop::new(BytecodeTimeLoopConfig {
            memory_cells: 1,
            max_epochs: 32,
            vm: vm_config(gas),
            ..BytecodeTimeLoopConfig::default()
        })
        .map_err(|error| error.to_string())?;
        let mut state = 0;
        let mut path = Vec::new();
        while !path.contains(&state) {
            path.push(state);
            state = report.conventional.transitions[state];
        }
        let period = path.len() - path.iter().position(|&prior| prior == state).unwrap();
        let outcome = match driver.run(&decoded.program) {
            ConvergenceStatus::Consistent { memory, output, .. }
                if period == 1
                    && memory.read(0).val == state as u64
                    && output_words(&output)? == vec![state as u64] =>
            {
                format!("point {state}")
            }
            ConvergenceStatus::Oscillation { period: actual, .. }
                if actual == period && period > 1 =>
            {
                format!("oscillation period {period}; no selected point on orbit from zero")
            }
            other => {
                return Err(format!(
                    "game package orbit disagrees with conventional history: {other:?}"
                ))
            }
        };
        report.orbit_package = Some(bytes);
        report.orbit_outcome = Some(outcome);
    } else if report.conventional.fixed_states.is_empty() {
        no_point_orbit(&mut report)?;
    }
    report.replay_package_ns = clock.elapsed().as_nanos();
    Ok(report)
}

impl Report {
    pub fn solver_status(&self) -> &'static str {
        match self.solver {
            GlobalUniquenessResult::NoFixedPoint(_) => "no-fixed-point",
            GlobalUniquenessResult::Unique { .. } => "unique",
            GlobalUniquenessResult::Multiple { .. } => "multiple",
            GlobalUniquenessResult::Unknown { .. } => "unknown",
            GlobalUniquenessResult::Unsupported { .. } => "unsupported",
            GlobalUniquenessResult::InternalError { .. } => "internal-error",
        }
    }

    fn property_summary(&self) -> String {
        match &self.property {
            PropertyVerificationResult::Proven { .. } => {
                "proven: all point states belong to conventional complete solution set".into()
            }
            PropertyVerificationResult::Vacuous { .. } => "vacuous: no point state exists".into(),
            PropertyVerificationResult::Unknown { reason, .. } => format!("unknown: {reason:?}"),
            other => format!("{other:?}"),
        }
    }
    pub fn is_unknown(&self) -> bool {
        matches!(self.solver, GlobalUniquenessResult::Unknown { .. })
            || matches!(self.property, PropertyVerificationResult::Unknown { .. })
    }

    /// Bounded human-readable result. Source and parameters are retained in
    /// full; SHA binds exact executable bytes. Runtime measurements vary by
    /// machine and include explicit scopes, so this is not a speedup claim.
    pub fn text(&self) -> Result<String, String> {
        let mut text = format!(
            "Ourochronos application study v1\nparameters: {:?}\n\
             frozen-input-contract: parameters embedded as source literals; no ordinary INPUT, RANDOM, clock, FFI or host effects\n\
             vm-config: memory_cells={} gas={} bounds=Error stack=128 calls=16 temporal_depth=1 output_items=8 output_bytes=4096 orbit_epochs=32\n\
             solver-config: loop_unroll=0 timeout_ms=5000; complete IR required for no-point/uniqueness claims; UNSAT evidence trusts Z3 backend, independently cross-checked by the conventional formulation\n\
             code-sha256: {}\ncode-records: {}\nconventional-fixed-states: {:?}\n\
             conventional-candidate-trials: {}\nconventional-cost: {} {}\n\
             complete-game-transition-table: {:?}\ncomplete-game-classes: {:?}\n\
             solver-outcome: {}\nall-point-membership-property: {}\n\
             timings-nanoseconds: conventional={} source_admission_encode={} solver_uniqueness_property={} conventional_point_replay_package_and_game_graph={}\n\
             comparison-contract: conventional operation units and fetched VM records are different work models; wall times are one local measurement, not ideal-CTC complexity or a performance promise\n\
             graph-replay-fetched-records: {}\n",
            self.study, self.study.memory_cells(), self.gas, self.code_sha256, self.code.instructions.len(),
            self.conventional.fixed_states, self.conventional.candidate_trials, self.conventional.operations, self.conventional.operation_unit,
            self.conventional.transitions, self.conventional.recurrent, self.solver_status(), self.property_summary(),
            self.conventional_ns, self.admission_ns, self.solver_ns, self.replay_package_ns, self.graph_fetched_records,
        );
        writeln!(
            text,
            "frozen-parameter-text-sha256: {}",
            hash(format!("{:?}", self.study).as_bytes())
        )
        .unwrap();
        if matches!(self.study, Study::Dataflow { .. }) {
            text.push_str("dataflow-completeness-rule: every possible x0 is tried; reverse ring equations uniquely determine all other cells without inverting gains; final x0 closure is checked; first-cell membership plus fixed equations implies full-vector membership\n");
        }
        match &self.solver {
            GlobalUniquenessResult::Unique { witness, .. } => {
                writeln!(text, "selected-solver-witness: {}", witness.to_json()).unwrap()
            }
            GlobalUniquenessResult::Multiple { first, second, .. } => {
                writeln!(text, "selected-solver-witness: {}", first.to_json()).unwrap();
                writeln!(text, "second-solver-witness: {}", second.to_json()).unwrap();
            }
            _ => {}
        }
        if let GlobalUniquenessResult::Unknown { reason, .. } = &self.solver {
            writeln!(text, "solver-unknown-reason: {reason:?}").unwrap();
        }
        let mut certificates = Vec::new();
        match &self.solver {
            GlobalUniquenessResult::NoFixedPoint(certificate)
            | GlobalUniquenessResult::Unique { certificate, .. } => {
                certificates.push(("point-classification", certificate))
            }
            _ => {}
        }
        match &self.property {
            PropertyVerificationResult::Proven { certificate, .. } => {
                certificates.push(("membership-property", certificate))
            }
            PropertyVerificationResult::Vacuous { no_fixed_point, .. } => {
                certificates.push(("vacuous-property", no_fixed_point))
            }
            _ => {}
        }
        for (label, certificate) in certificates {
            writeln!(text, "solver-evidence: claim={label} backend={} completeness={:?} query_bytes={} query_sha256={} proof_bytes={} proof_sha256={}; full backend term is held by the API result, rerun exact frozen source/config to regenerate",
                certificate.backend, certificate.completeness, certificate.solver_query.len(), hash(certificate.solver_query.as_bytes()), certificate.backend_proof.len(), hash(certificate.backend_proof.as_bytes())).unwrap();
        }
        for replay in &self.replays {
            writeln!(text, "point-replay: state={:?} stack=[] typed-numeric-output={:?} fetched_records={} effects=[] package_bytes={} package_sha256={} package_output={:?}",
                replay.state, replay.output, replay.fetched_records, replay.package.len(), hash(&replay.package), replay.package_output).unwrap();
        }
        if let Some(bytes) = &self.orbit_package {
            writeln!(
                text,
                "orbit-package: bytes={} sha256={} outcome={:?}",
                bytes.len(),
                hash(bytes),
                self.orbit_outcome
            )
            .unwrap();
        }
        text.push_str("generated-source-begin\n");
        text.push_str(&self.source);
        text.push_str("generated-source-end\n");
        if text.len() > MAX_SAVED_BYTES {
            return Err("result exceeds one-MiB retained record ceiling".into());
        }
        Ok(text)
    }
}

pub fn parse_cli(args: &[String]) -> Result<(Study, u64, Option<std::path::PathBuf>), String> {
    let usage = "usage: application_studies exclusion CONTENDERS ELIGIBLE_MASK | dataflow NODES BITS GAIN_OR_GAINS_CSV BIASES_CSV | game ACTIONS SUCCESSORS_CSV [--gas 1..1024] [--save NEW_PATH]";
    let number = |value: &str| {
        value
            .parse::<u64>()
            .map_err(|_| format!("invalid decimal parameter {value:?}"))
    };
    let count = |value: &str| {
        number(value)
            .and_then(|value| usize::try_from(value).map_err(|_| "count exceeds host size".into()))
    };
    let csv = |value: &str, cap: usize| -> Result<Vec<u64>, String> {
        if value.len() > 512 || value.split(',').count() > cap {
            return Err("parameter list exceeds workflow ceiling".into());
        }
        value.split(',').map(number).collect()
    };
    let (study, mut cursor) = match args.first().map(String::as_str) {
        Some("exclusion") if args.len() >= 3 => (
            Study::Exclusion {
                contenders: count(&args[1])?,
                eligible: number(&args[2])?,
            },
            3,
        ),
        Some("dataflow") if args.len() >= 5 => {
            let nodes = count(&args[1])?;
            if !(1..=8).contains(&nodes) {
                return Err("dataflow nodes must be in 1..8".into());
            }
            let mut gains = csv(&args[3], 8)?;
            if gains.len() == 1 {
                gains = vec![gains[0]; nodes];
            }
            (
                Study::Dataflow {
                    nodes,
                    bits: u8::try_from(number(&args[2])?).map_err(|_| "bits exceed u8")?,
                    gains,
                    biases: csv(&args[4], 8)?,
                },
                5,
            )
        }
        Some("game") if args.len() >= 3 => (
            Study::Game {
                actions: count(&args[1])?,
                successors: csv(&args[2], 16)?
                    .into_iter()
                    .map(|word| {
                        usize::try_from(word).map_err(|_| "action exceeds host size".into())
                    })
                    .collect::<Result<Vec<_>, String>>()?,
            },
            3,
        ),
        _ => return Err(usage.into()),
    };
    let mut gas = MAX_GAS;
    let mut save = None;
    let mut seen_gas = false;
    while cursor < args.len() {
        let value = args.get(cursor + 1).ok_or(usage)?;
        match args[cursor].as_str() {
            "--gas" if !seen_gas => {
                gas = number(value)?;
                seen_gas = true;
            }
            "--save" if save.is_none() => {
                save = Some(std::path::PathBuf::from(value));
            }
            _ => return Err(usage.into()),
        }
        cursor += 2;
    }
    study.validate()?;
    if !(1..=MAX_GAS).contains(&gas) {
        return Err("gas must be in 1..1024".into());
    }
    Ok((study, gas, save))
}

pub fn save_new(path: &std::path::Path, text: &str) -> Result<(), String> {
    use std::io::Write;
    if text.len() > MAX_SAVED_BYTES {
        return Err("result exceeds save ceiling".into());
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create result {}: {error}", path.display()))?;
    file.write_all(text.as_bytes())
        .map_err(|error| format!("cannot save result: {error}"))
}
