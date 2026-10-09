//! Restricted bytecode-to-sparse-Markov adapter with explicit joint tapes.
//!
//! A scenario is chosen with its full exact probability before execution.
//! RANDOM consumes its frozen word prefix; an unused suffix neither conditions
//! nor renormalizes that probability. This API makes no iid or live-RANDOM
//! claim. The same ordinary INPUT tape starts at cursor zero in every run.
//! The returned one-step kernel applies the same state-independent scenario
//! law on each application, preserving correlations within each joint tape.
//!
//! Whole-main base-zero masked scope admission gives a complete finite memory
//! domain with a fresh zero outside frame. Every state/scenario must finish and
//! exit the scope under the declared resources. Numeric successor edges may
//! coalesce; typed/provenanced terminal observations remain separate evidence.
//! This is an explicit bytecode profile, not ordinary source admission or a
//! package environment policy. No host/effect/heap capability is admitted.
//! Ceilings bound declared work/storage estimates and observed charges; they
//! are not a hard wall-clock or host-allocation theorem.

use super::sparse_markov::{
    ExactRational, SparseMarkovChain, SparseMarkovError, SparseMarkovLimits,
};
use crate::ast::OpCode;
use crate::bytecode::{BytecodeProgram, Instruction};
use crate::bytecode_vm::{
    BytecodeVm, BytecodeVmConfig, BytecodeVmError, BytecodeVmStatus, PreparedBytecode,
};
use crate::core::{BoundsPolicy, OutputItem, PagedMemory, Value};
use num_traits::{One, Zero};
use std::fmt;

pub const MAX_VM_STOCHASTIC_STATE_BITS: usize = 12;
pub const MAX_VM_STOCHASTIC_STATES: usize = 1 << MAX_VM_STOCHASTIC_STATE_BITS;
pub const MAX_VM_STOCHASTIC_SCENARIOS: usize = 64;
pub const MAX_VM_STOCHASTIC_TAPE_WORDS: usize = 65_536;
pub const MAX_VM_STOCHASTIC_EVALUATIONS: usize = 65_536;
pub const MAX_VM_STOCHASTIC_WORK: u64 = 16 * 1024 * 1024;
pub const MAX_VM_STOCHASTIC_EVIDENCE_BYTES: usize = 64 * 1024 * 1024;
const MAX_MEMORY: usize = 4096;
const MAX_CODE: usize = 4096;
const MAX_PROCEDURES: usize = 64;
const MAX_STACK: usize = 256;
const MAX_OUTPUT: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VmStochasticLimits {
    pub max_states: usize,
    pub max_scenarios: usize,
    /// Frozen ordinary input plus all full scenario tapes, including unused
    /// suffixes and duplicates. Scenarios are never silently discarded.
    pub max_tape_words: usize,
    pub max_evaluations: usize,
    /// Preflight states*scenarios*(per-run fetched gas + full frame scan).
    pub max_work: u64,
    /// Conservative retained observation evidence bound, preflighted using
    /// the configured stack/output ceilings and every possible tape prefix.
    pub max_evidence_bytes: usize,
}
impl Default for VmStochasticLimits {
    fn default() -> Self {
        Self {
            max_states: MAX_VM_STOCHASTIC_STATES,
            max_scenarios: MAX_VM_STOCHASTIC_SCENARIOS,
            max_tape_words: MAX_VM_STOCHASTIC_TAPE_WORDS,
            max_evaluations: MAX_VM_STOCHASTIC_EVALUATIONS,
            max_work: MAX_VM_STOCHASTIC_WORK,
            max_evidence_bytes: MAX_VM_STOCHASTIC_EVIDENCE_BYTES,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VmStochasticConfig {
    pub memory_cells: usize,
    pub memory_bounds: BoundsPolicy,
    pub max_instructions: u64,
    pub max_call_depth: usize,
    pub max_stack_depth: usize,
    pub max_output_items: usize,
    pub max_output_bytes: usize,
    pub input: Vec<u64>,
    pub extraction_limits: VmStochasticLimits,
    pub chain_limits: SparseMarkovLimits,
}
impl Default for VmStochasticConfig {
    fn default() -> Self {
        Self {
            memory_cells: 16,
            memory_bounds: BoundsPolicy::Error,
            max_instructions: 1024,
            max_call_depth: 64,
            max_stack_depth: 32,
            max_output_items: 8,
            max_output_bytes: 4096,
            input: vec![],
            extraction_limits: VmStochasticLimits::default(),
            chain_limits: SparseMarkovLimits::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RandomTapeScenario {
    pub words: Vec<u64>,
    /// Strictly positive exact mass. All scenarios must sum exactly to one.
    pub probability: ExactRational,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VmStochasticScope {
    pub cells: usize,
    pub cell_bits: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VmStochasticTransition {
    pub scenario_index: usize,
    pub successor: usize,
    pub stack: Vec<Value>,
    pub output: Vec<OutputItem>,
    pub inputs_consumed: Vec<u64>,
    pub random_consumed: Vec<u64>,
    pub instructions_executed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VmStochasticStats {
    pub evaluations: usize,
    pub preflight_work: u64,
    pub preflight_evidence_bytes: usize,
    pub instructions_executed: u64,
    pub evidence_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VmStochasticModel {
    pub chain: SparseMarkovChain,
    pub scope: VmStochasticScope,
    pub config: VmStochasticConfig,
    pub scenarios: Vec<RandomTapeScenario>,
    /// Row order is little-endian cell-zero-first finite state identity;
    /// within each row, every submitted scenario is retained in input order.
    pub transitions: Vec<Vec<VmStochasticTransition>>,
    pub stats: VmStochasticStats,
}

impl VmStochasticModel {
    pub fn state_words(&self, state: usize) -> Option<Vec<u64>> {
        let bits = self
            .scope
            .cells
            .checked_mul(self.scope.cell_bits as usize)?;
        if self.scope.cells == 0
            || self.scope.cell_bits == 0
            || bits > MAX_VM_STOCHASTIC_STATE_BITS
            || self.chain.states() != 1usize << bits
        {
            return None;
        }
        (state < self.chain.states()).then(|| state_words(state, self.scope))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmStochasticResource {
    States,
    Scenarios,
    TapeWords,
    Evaluations,
    Work,
    EvidenceBytes,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VmStochasticError {
    InvalidConfig(&'static str),
    InvalidScenarios(&'static str),
    ResourceLimit {
        resource: VmStochasticResource,
        limit: u64,
        required: u64,
    },
    Unsupported {
        pc: Option<u32>,
        reason: &'static str,
    },
    InvalidBytecode(Box<BytecodeVmError>),
    Chain(SparseMarkovError),
    Execution {
        state: usize,
        scenario: usize,
        error: Box<BytecodeVmError>,
    },
    NonFinished {
        state: usize,
        scenario: usize,
        status: BytecodeVmStatus,
    },
    InvalidFrame {
        state: usize,
        scenario: usize,
        address: usize,
        word: u64,
    },
    Invariant(&'static str),
}
impl fmt::Display for VmStochasticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(reason) | Self::InvalidScenarios(reason) | Self::Invariant(reason) => f.write_str(reason),
            Self::ResourceLimit { resource, limit, required } => write!(f, "VM stochastic {resource:?} requires {required}, exceeding {limit}"),
            Self::Unsupported { pc, reason } => write!(f, "unsupported VM stochastic profile at {pc:?}: {reason}"),
            Self::InvalidBytecode(error) => write!(f, "invalid VM stochastic bytecode: {error}"),
            Self::Chain(error) => error.fmt(f),
            Self::Execution { state, scenario, error } => write!(f, "VM stochastic state {state} scenario {scenario} failed: {error}"),
            Self::NonFinished { state, scenario, status } => write!(f, "VM stochastic state {state} scenario {scenario} stopped as {status:?}"),
            Self::InvalidFrame { state, scenario, address, word } => write!(f, "VM stochastic state {state} scenario {scenario} produced invalid cell {address}={word}"),
        }
    }
}
impl std::error::Error for VmStochasticError {}
impl From<SparseMarkovError> for VmStochasticError {
    fn from(error: SparseMarkovError) -> Self {
        Self::Chain(error)
    }
}

/// Extract an exact finite joint-scenario transition matrix. Limits count raw
/// state/scenario products before running any row or coalescing any edge.
pub fn extract_vm_markov(
    program: &BytecodeProgram,
    config: &VmStochasticConfig,
    scenarios: &[RandomTapeScenario],
) -> Result<VmStochasticModel, VmStochasticError> {
    validate_config(config)?;
    let scope = admit_profile(program, config)?;
    let states = 1usize << (scope.cells * scope.cell_bits as usize);
    let mut stats = preflight(config, scope, states, scenarios)?;
    // Validate chain limits without cloning an untrusted large integer first.
    SparseMarkovChain::new(vec![vec![(0, ExactRational::one())]], config.chain_limits)?;
    for scenario in scenarios {
        let required = scenario
            .probability
            .numer()
            .bits()
            .max(scenario.probability.denom().bits());
        if required > config.chain_limits.max_integer_bits {
            return Err(SparseMarkovError::ResourceLimit {
                resource: super::sparse_markov::SparseMarkovResource::IntegerBits,
                limit: config.chain_limits.max_integer_bits,
                required,
            }
            .into());
        }
    }
    // Reuse exact probability validation and conservative integer/work caps.
    // A one-state row validates the joint mass before any VM run/allocation.
    SparseMarkovChain::new(
        vec![scenarios
            .iter()
            .map(|scenario| (0, scenario.probability.clone()))
            .collect()],
        config.chain_limits,
    )?;
    let mut scenarios = scenarios.to_vec();
    for scenario in &mut scenarios {
        scenario.probability = ExactRational::new(
            scenario.probability.numer().clone(),
            scenario.probability.denom().clone(),
        );
        if scenario.probability.is_zero() {
            return Err(VmStochasticError::InvalidScenarios(
                "every random-tape scenario must have positive mass",
            ));
        }
    }
    let prepared = PreparedBytecode::new(program.clone())
        .map_err(|error| VmStochasticError::InvalidBytecode(Box::new(error)))?;
    let mut vms = Vec::with_capacity(scenarios.len());
    for scenario in &scenarios {
        vms.push(BytecodeVm::with_config(vm_config(config, &scenario.words)));
    }
    let mut transitions = Vec::with_capacity(states);
    let mut rows = Vec::with_capacity(states);
    for state in 0..states {
        let mut memory = PagedMemory::with_size(config.memory_cells)
            .map_err(|_| VmStochasticError::Invariant("finite anamnesis allocation failed"))?;
        for (address, word) in state_words(state, scope).into_iter().enumerate() {
            memory
                .write(address as u64, Value::new(word))
                .map_err(|_| VmStochasticError::Invariant("finite state address is invalid"))?;
        }
        let mut evidence = Vec::with_capacity(scenarios.len());
        let mut edges = Vec::with_capacity(scenarios.len());
        for (scenario_index, vm) in vms.iter().enumerate() {
            let execution = vm.run_prepared(&prepared, &memory).map_err(|error| {
                VmStochasticError::Execution {
                    state,
                    scenario: scenario_index,
                    error: Box::new(error),
                }
            })?;
            if execution.status != BytecodeVmStatus::Finished {
                return Err(VmStochasticError::NonFinished {
                    state,
                    scenario: scenario_index,
                    status: execution.status,
                });
            }
            if !execution.effects.is_empty()
                || !execution.clock_inputs_consumed.is_empty()
                || !execution.file_snapshots_consumed.is_empty()
                || !execution.endpoint_tapes_consumed.is_empty()
                || !execution.process_results_consumed.is_empty()
                || execution.temporal_entries != 1
            {
                return Err(VmStochasticError::Invariant(
                    "admitted VM run escaped frozen numeric profile",
                ));
            }
            if !config.input.starts_with(&execution.inputs_consumed)
                || !scenarios[scenario_index]
                    .words
                    .starts_with(&execution.random_inputs_consumed)
            {
                return Err(VmStochasticError::Invariant(
                    "VM did not consume a supplied tape prefix",
                ));
            }
            let mut successor = 0_usize;
            let mask = (1u64 << scope.cell_bits) - 1;
            for address in 0..config.memory_cells {
                let word = execution
                    .present
                    .get(address as u64)
                    .ok_or(VmStochasticError::Invariant("VM omitted configured memory"))?
                    .val;
                if (address < scope.cells && word > mask) || (address >= scope.cells && word != 0) {
                    return Err(VmStochasticError::InvalidFrame {
                        state,
                        scenario: scenario_index,
                        address,
                        word,
                    });
                }
                if address < scope.cells {
                    successor |= (word as usize) << (address * scope.cell_bits as usize);
                }
            }
            let transition = VmStochasticTransition {
                scenario_index,
                successor,
                stack: execution.stack,
                output: execution.output,
                inputs_consumed: execution.inputs_consumed,
                random_consumed: execution.random_inputs_consumed,
                instructions_executed: execution.instructions_executed,
            };
            let bytes = evidence_charge(&transition)?;
            stats.evidence_bytes =
                stats
                    .evidence_bytes
                    .checked_add(bytes)
                    .ok_or(VmStochasticError::Invariant(
                        "retained evidence byte count overflowed",
                    ))?;
            check_limit(
                VmStochasticResource::EvidenceBytes,
                stats.evidence_bytes as u64,
                config.extraction_limits.max_evidence_bytes as u64,
            )?;
            stats.instructions_executed = stats
                .instructions_executed
                .checked_add(transition.instructions_executed)
                .ok_or(VmStochasticError::Invariant(
                    "fetched instruction count overflowed",
                ))?;
            edges.push((successor, scenarios[scenario_index].probability.clone()));
            evidence.push(transition);
        }
        rows.push(edges);
        transitions.push(evidence);
    }
    let chain = SparseMarkovChain::new(rows, config.chain_limits)?;
    if stats.evidence_bytes > stats.preflight_evidence_bytes {
        return Err(VmStochasticError::Invariant(
            "observations exceeded admitted provenance/tape byte bound",
        ));
    }
    Ok(VmStochasticModel {
        chain,
        scope,
        config: config.clone(),
        scenarios,
        transitions,
        stats,
    })
}

fn check_limit(
    resource: VmStochasticResource,
    required: u64,
    limit: u64,
) -> Result<(), VmStochasticError> {
    if required > limit {
        Err(VmStochasticError::ResourceLimit {
            resource,
            limit,
            required,
        })
    } else {
        Ok(())
    }
}

fn validate_config(config: &VmStochasticConfig) -> Result<(), VmStochasticError> {
    if config.memory_cells == 0
        || config.memory_cells > MAX_MEMORY
        || config.max_call_depth > MAX_PROCEDURES
        || config.max_stack_depth > MAX_STACK
        || config.max_output_items > MAX_OUTPUT
        || config.max_output_bytes > MAX_VM_STOCHASTIC_EVIDENCE_BYTES
        || config.max_instructions > MAX_VM_STOCHASTIC_WORK
    {
        return Err(VmStochasticError::InvalidConfig(
            "VM stochastic resource profile exceeds fixed ceilings",
        ));
    }
    let limits = config.extraction_limits;
    for (resource, requested, maximum) in [
        (
            VmStochasticResource::States,
            limits.max_states as u64,
            MAX_VM_STOCHASTIC_STATES as u64,
        ),
        (
            VmStochasticResource::Scenarios,
            limits.max_scenarios as u64,
            MAX_VM_STOCHASTIC_SCENARIOS as u64,
        ),
        (
            VmStochasticResource::TapeWords,
            limits.max_tape_words as u64,
            MAX_VM_STOCHASTIC_TAPE_WORDS as u64,
        ),
        (
            VmStochasticResource::Evaluations,
            limits.max_evaluations as u64,
            MAX_VM_STOCHASTIC_EVALUATIONS as u64,
        ),
        (
            VmStochasticResource::Work,
            limits.max_work,
            MAX_VM_STOCHASTIC_WORK,
        ),
        (
            VmStochasticResource::EvidenceBytes,
            limits.max_evidence_bytes as u64,
            MAX_VM_STOCHASTIC_EVIDENCE_BYTES as u64,
        ),
    ] {
        if requested == 0 {
            return Err(VmStochasticError::InvalidConfig(
                "VM stochastic extraction limit must be positive and at most its ceiling",
            ));
        }
        check_limit(resource, requested, maximum)?;
    }
    Ok(())
}

fn preflight(
    config: &VmStochasticConfig,
    scope: VmStochasticScope,
    states: usize,
    scenarios: &[RandomTapeScenario],
) -> Result<VmStochasticStats, VmStochasticError> {
    if scenarios.is_empty() {
        return Err(VmStochasticError::InvalidScenarios(
            "at least one random-tape scenario is required",
        ));
    }
    let limits = config.extraction_limits;
    check_limit(
        VmStochasticResource::States,
        states as u64,
        limits.max_states as u64,
    )?;
    check_limit(
        VmStochasticResource::Scenarios,
        scenarios.len() as u64,
        limits.max_scenarios as u64,
    )?;
    let words = scenarios
        .iter()
        .try_fold(config.input.len(), |sum, scenario| {
            sum.checked_add(scenario.words.len())
        })
        .ok_or(VmStochasticError::Invariant(
            "scenario tape count overflowed",
        ))?;
    check_limit(
        VmStochasticResource::TapeWords,
        words as u64,
        limits.max_tape_words as u64,
    )?;
    let evaluations = states
        .checked_mul(scenarios.len())
        .ok_or(VmStochasticError::Invariant("evaluation count overflowed"))?;
    check_limit(
        VmStochasticResource::Evaluations,
        evaluations as u64,
        limits.max_evaluations as u64,
    )?;
    // Raw edge caps count every state/scenario even if successors coincide.
    for (resource, required, limit) in [
        (
            super::sparse_markov::SparseMarkovResource::States,
            states,
            config.chain_limits.max_states,
        ),
        (
            super::sparse_markov::SparseMarkovResource::RawEdges,
            evaluations,
            config.chain_limits.max_raw_edges,
        ),
    ] {
        if required > limit {
            return Err(SparseMarkovError::ResourceLimit {
                resource,
                required: required as u64,
                limit: limit as u64,
            }
            .into());
        }
    }
    let work = config
        .max_instructions
        .checked_add(config.memory_cells as u64)
        .and_then(|per_run| per_run.checked_mul(evaluations as u64))
        .ok_or(VmStochasticError::Invariant(
            "aggregate work estimate overflowed",
        ))?;
    check_limit(VmStochasticResource::Work, work, limits.max_work)?;
    let max_random = scenarios
        .iter()
        .map(|scenario| scenario.words.len())
        .max()
        .unwrap_or(0);
    let evidence_bytes = 128_usize
        .checked_add(config.max_stack_depth * (32 + 32 * scope.cells))
        .and_then(|sum| sum.checked_add(config.max_output_bytes))
        .and_then(|sum| {
            config
                .input
                .len()
                .checked_add(max_random)
                .and_then(|words| words.checked_mul(8))
                .and_then(|tapes| sum.checked_add(tapes))
        })
        .and_then(|per_run| per_run.checked_mul(evaluations))
        .ok_or(VmStochasticError::Invariant(
            "aggregate evidence estimate overflowed",
        ))?;
    check_limit(
        VmStochasticResource::EvidenceBytes,
        evidence_bytes as u64,
        limits.max_evidence_bytes as u64,
    )?;
    Ok(VmStochasticStats {
        evaluations,
        preflight_work: work,
        preflight_evidence_bytes: evidence_bytes,
        instructions_executed: 0,
        evidence_bytes: 0,
    })
}

fn evidence_charge(transition: &VmStochasticTransition) -> Result<usize, VmStochasticError> {
    let mut bytes = 128_usize;
    for word in &transition.stack {
        bytes =
            bytes
                .checked_add(word.retained_size_charge())
                .ok_or(VmStochasticError::Invariant(
                    "stack evidence byte count overflowed",
                ))?;
    }
    for item in &transition.output {
        bytes =
            bytes
                .checked_add(item.retained_size_charge())
                .ok_or(VmStochasticError::Invariant(
                    "output evidence byte count overflowed",
                ))?;
    }
    bytes
        .checked_add(8 * (transition.inputs_consumed.len() + transition.random_consumed.len()))
        .ok_or(VmStochasticError::Invariant(
            "tape evidence byte count overflowed",
        ))
}

fn vm_config(config: &VmStochasticConfig, random: &[u64]) -> BytecodeVmConfig {
    BytecodeVmConfig {
        max_instructions: config.max_instructions,
        max_call_depth: config.max_call_depth,
        max_stack_depth: config.max_stack_depth,
        max_temporal_depth: 1,
        max_output_items: config.max_output_items,
        max_output_bytes: config.max_output_bytes,
        memory_bounds: config.memory_bounds,
        input: config.input.clone(),
        random_input: random.to_vec(),
        max_collections: 0,
        max_collection_items: 0,
        max_dynamic_bytes: 0,
        max_file_snapshots: 0,
        max_file_snapshot_bytes: 0,
        max_endpoint_tape_bytes: 0,
        max_process_result_bytes: 0,
        max_effects: 0,
        max_effect_bytes: 0,
        ..BytecodeVmConfig::default()
    }
}

fn state_words(state: usize, scope: VmStochasticScope) -> Vec<u64> {
    let mask = (1u64 << scope.cell_bits) - 1;
    (0..scope.cells)
        .map(|cell| ((state as u64) >> (cell * scope.cell_bits as usize)) & mask)
        .collect()
}

fn allowed(op: OpCode) -> bool {
    matches!(
        op,
        OpCode::Nop
            | OpCode::Halt
            | OpCode::Paradox
            | OpCode::Pop
            | OpCode::Dup
            | OpCode::Swap
            | OpCode::Over
            | OpCode::Rot
            | OpCode::Depth
            | OpCode::Pick
            | OpCode::Roll
            | OpCode::Reverse
            | OpCode::Add
            | OpCode::Sub
            | OpCode::Mul
            | OpCode::Div
            | OpCode::Mod
            | OpCode::Neg
            | OpCode::Abs
            | OpCode::Min
            | OpCode::Max
            | OpCode::Sign
            | OpCode::Not
            | OpCode::And
            | OpCode::Or
            | OpCode::Xor
            | OpCode::Shl
            | OpCode::Shr
            | OpCode::Eq
            | OpCode::Neq
            | OpCode::Lt
            | OpCode::Gt
            | OpCode::Lte
            | OpCode::Gte
            | OpCode::Slt
            | OpCode::Sgt
            | OpCode::Slte
            | OpCode::Sgte
            | OpCode::Oracle
            | OpCode::Prophecy
            | OpCode::PresentRead
            | OpCode::Index
            | OpCode::Store
            | OpCode::Input
            | OpCode::Random
            | OpCode::Output
            | OpCode::Emit
    )
}

fn admit_profile(
    program: &BytecodeProgram,
    config: &VmStochasticConfig,
) -> Result<VmStochasticScope, VmStochasticError> {
    let unsupported = |pc, reason| VmStochasticError::Unsupported { pc, reason };
    if program.instructions.len() > MAX_CODE
        || program.procedures.len() > MAX_PROCEDURES
        || program.source_map.len() > MAX_CODE
    {
        return Err(VmStochasticError::InvalidConfig(
            "bytecode exceeds stochastic profile instruction/procedure ceilings",
        ));
    }
    if !program.quotations.is_empty() || !program.foreigns.is_empty() {
        return Err(unsupported(
            None,
            "quotation and foreign declarations are excluded",
        ));
    }
    program.validate().map_err(|error| {
        VmStochasticError::InvalidBytecode(Box::new(BytecodeVmError::InvalidArtifact(error)))
    })?;
    let main = program.main;
    if main.end - main.start < 3 {
        return Err(unsupported(
            None,
            "main must be one whole masked temporal scope",
        ));
    }
    let exit = main.end - 2;
    let scope = match program.instructions[main.start as usize] {
        Instruction::TemporalEnter {
            base: 0,
            size,
            cell_bits,
            exit_target,
        } if exit_target == exit => VmStochasticScope {
            cells: usize::try_from(size)
                .map_err(|_| unsupported(None, "scope size does not fit"))?,
            cell_bits,
        },
        _ => {
            return Err(unsupported(
                Some(main.start),
                "main must begin with a base-zero masked temporal scope",
            ))
        }
    };
    if scope.cells == 0
        || scope.cell_bits == 0
        || scope
            .cells
            .checked_mul(scope.cell_bits as usize)
            .is_none_or(|bits| bits > MAX_VM_STOCHASTIC_STATE_BITS)
        || scope.cells > config.memory_cells
    {
        return Err(unsupported(
            Some(main.start),
            "scope must fit memory and contain 1..12 total masked state bits",
        ));
    }
    if program.instructions[exit as usize]
        != (Instruction::TemporalExit {
            enter_target: main.start,
        })
        || program.instructions[(main.end - 1) as usize] != Instruction::Return
    {
        return Err(unsupported(
            Some(exit),
            "main must end with paired scope EXIT and RETURN",
        ));
    }
    let mut ranges = vec![main];
    ranges.extend(program.procedures.iter().map(|entry| entry.range));
    let mut calls = vec![Vec::new(); ranges.len()];
    // Direct calls are the only admitted executable identities. Build the
    // complete graph before deciding which units' capabilities are reachable.
    for (unit, range) in ranges.iter().enumerate() {
        for pc in range.start..range.end {
            if let Instruction::CallProcedure(target) = program.instructions[pc as usize] {
                calls[unit].push(target.index() + 1);
            }
        }
    }
    let mut seen = vec![false; ranges.len()];
    let mut pending = vec![0];
    while let Some(unit) = pending.pop() {
        if seen[unit] {
            continue;
        }
        seen[unit] = true;
        pending.extend(calls[unit].iter().copied());
        for pc in ranges[unit].start..ranges[unit].end {
            match program.instructions[pc as usize] {
                Instruction::Primitive(op) if allowed(op) => {}
                Instruction::PushWord(_)
                | Instruction::CallProcedure(_)
                | Instruction::IfFalse { .. }
                | Instruction::Jump { .. }
                | Instruction::WhileFalse { .. }
                | Instruction::LoopBack { .. }
                | Instruction::Return => {}
                Instruction::TemporalEnter { .. } if pc == main.start => {}
                Instruction::TemporalExit { .. } if pc == exit => {}
                _ => {
                    return Err(unsupported(
                        Some(pc),
                        "reachable instruction is outside frozen numeric profile",
                    ))
                }
            }
        }
    }
    fn visit(unit: usize, calls: &[Vec<usize>], colors: &mut [u8]) -> bool {
        if colors[unit] == 1 {
            return false;
        }
        if colors[unit] == 2 {
            return true;
        }
        colors[unit] = 1;
        for &target in &calls[unit] {
            if !visit(target, calls, colors) {
                return false;
            }
        }
        colors[unit] = 2;
        true
    }
    if !visit(0, &calls, &mut vec![0; ranges.len()]) {
        return Err(unsupported(
            None,
            "recursive procedures are outside stochastic profile",
        ));
    }
    Ok(scope)
}
