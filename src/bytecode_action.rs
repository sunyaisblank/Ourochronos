//! Action-guided fixed-state selection over validated bytecode.
//!
//! This is the bytecode-native counterpart of the legacy AST action policy.
//! Every seed executes the same prepared artifact through [`BytecodeVm`],
//! exact epoch results are shared when seed orbits meet, and candidate output
//! remains buffered until the least-action fixed state has been selected.

use crate::bytecode::BytecodeProgram;
use crate::bytecode_timeloop::{
    is_trivially_consistent, retain_orbit_state, transaction_limits, BytecodeTimeLoop,
    BytecodeTimeLoopConfig,
};
use crate::bytecode_vm::{
    BytecodeExecution, BytecodeVm, BytecodeVmConfig, BytecodeVmStatus, FrozenEndpointTape,
    FrozenFileSnapshot, FrozenProcessResult, PreparedBytecode,
};
use crate::core::{Address, Memory, OutputItem, PagedMemory, Value};
use crate::temporal::action::{ActionConfig, ActionPrinciple};
use crate::temporal::timeloop::ConvergenceStatus;
use crate::temporal::transaction::{
    CommitLog, CommitToken, CommittedBatch, EffectIntent, ObservationTranscript,
    TemporalTransaction,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

const ACTION_EPOCH_CACHE_CAPACITY: usize = 1_024;
const ACTION_EPOCH_CACHE_MAX_RETAINED_BYTES: usize = 256 * 1024 * 1024;
/// Maximum deterministic seed orbits accepted by one action-guided run.
pub const MAX_ACTION_SEEDS: usize = 1_024;

/// Complete configuration for bytecode-native action-guided selection.
#[derive(Debug, Clone, PartialEq)]
pub struct BytecodeActionConfig {
    /// Orbit, memory, gas, and deterministic-input configuration.
    pub runtime: BytecodeTimeLoopConfig,
    /// Weights used to rank consistent timelines.
    pub action: ActionConfig,
    /// Number of deterministic seed states to explore.
    pub num_seeds: usize,
}

impl Default for BytecodeActionConfig {
    fn default() -> Self {
        Self {
            runtime: BytecodeTimeLoopConfig::default(),
            action: ActionConfig::default(),
            num_seeds: 4,
        }
    }
}

/// Auditable counters for one action-guided selection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BytecodeActionStats {
    /// Number of deterministic seeds actually attempted.
    pub attempted_seeds: usize,
    /// Number of seeds that reached a point fixed state.
    pub consistent_candidates: usize,
    /// Exact-state epoch-cache hits shared within and across seed orbits.
    pub cache_hits: usize,
    /// Epochs that required actual bytecode dispatch.
    pub cache_misses: usize,
    /// Entries retained when the run-local cache was dropped.
    pub cache_entries: usize,
    /// Deterministic bounded-cache evictions during the run.
    pub cache_evictions: usize,
}

/// Result and selection evidence from a bytecode action run.
#[derive(Debug)]
pub struct BytecodeActionReport {
    /// Language-level convergence result.
    pub status: ConvergenceStatus,
    /// Action of the selected fixed state, if action ranking was performed.
    pub selected_action: Option<f64>,
    /// In-memory commit evidence for the selected timeline only. Its output
    /// and irreversible intents have not been applied to a host; callers may
    /// pass the receipt and effects to an explicit capability-scoped adapter.
    pub selected_batch: Option<CommittedBatch>,
    /// Deterministic exploration and cache counters.
    pub stats: BytecodeActionStats,
}

/// Least-action fixed-state runner backed exclusively by validated bytecode.
#[derive(Debug, Clone)]
pub struct BytecodeActionRunner {
    /// Selection and runtime configuration.
    pub config: BytecodeActionConfig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FrozenTapes {
    input: Vec<u64>,
    clock: Vec<u64>,
    random: Vec<u64>,
    files: Vec<FrozenFileSnapshot>,
    endpoints: Vec<FrozenEndpointTape>,
    processes: Vec<FrozenProcessResult>,
}

impl FrozenTapes {
    fn from_config(config: &BytecodeVmConfig) -> Self {
        Self {
            input: config.input.clone(),
            clock: config.clock_input.clone(),
            random: config.random_input.clone(),
            files: config.file_snapshots.clone(),
            endpoints: config.endpoint_tapes.clone(),
            processes: config.process_results.clone(),
        }
    }
}

#[derive(Debug, Clone)]
struct Candidate {
    memory: Memory,
    seed: Memory,
    output: Vec<OutputItem>,
    observations: ObservationTranscript,
    effects: Vec<EffectIntent>,
    epochs: usize,
    action: f64,
}

#[derive(Debug)]
struct EpochCache {
    entries: HashMap<PagedMemory, (Arc<BytecodeExecution>, usize)>,
    insertion_order: VecDeque<PagedMemory>,
    retained_bytes: usize,
    max_retained_bytes: usize,
    hits: usize,
    misses: usize,
    evictions: usize,
}

impl Default for EpochCache {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            insertion_order: VecDeque::new(),
            retained_bytes: 0,
            max_retained_bytes: ACTION_EPOCH_CACHE_MAX_RETAINED_BYTES,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }
}

impl EpochCache {
    fn get(&mut self, state: &PagedMemory) -> Option<Arc<BytecodeExecution>> {
        if let Some((result, _)) = self.entries.get(state) {
            self.hits += 1;
            return Some(Arc::clone(result));
        }
        self.misses += 1;
        None
    }

    fn insert(&mut self, state: PagedMemory, result: Arc<BytecodeExecution>) {
        let charge = state
            .retained_size_charge()
            .saturating_add(result.retained_size_charge());
        if charge > self.max_retained_bytes {
            return;
        }
        if self.entries.contains_key(&state) {
            return;
        }
        while self.entries.len() >= ACTION_EPOCH_CACHE_CAPACITY
            || self.retained_bytes.saturating_add(charge) > self.max_retained_bytes
        {
            // FIFO is deterministic and avoids lexicographically comparing
            // every cell of every cached state during resource-pressure
            // eviction. Eviction affects performance only.
            let Some(victim) = self.insertion_order.pop_front() else {
                break;
            };
            if let Some((_, victim_charge)) = self.entries.remove(&victim) {
                self.retained_bytes = self.retained_bytes.saturating_sub(victim_charge);
                self.evictions += 1;
            }
        }
        self.retained_bytes = self.retained_bytes.saturating_add(charge);
        self.insertion_order.push_back(state.clone());
        self.entries.insert(state, (result, charge));
    }
}

impl BytecodeActionRunner {
    /// Construct a runner after validating all resource limits.
    pub fn new(config: BytecodeActionConfig) -> Result<Self, String> {
        if config.num_seeds == 0 {
            return Err("num_seeds must be greater than zero".to_string());
        }
        if config.num_seeds > MAX_ACTION_SEEDS {
            return Err(format!(
                "num_seeds {} exceeds limit {MAX_ACTION_SEEDS}",
                config.num_seeds
            ));
        }
        BytecodeTimeLoop::new(config.runtime.clone())?;
        Ok(Self { config })
    }

    /// Select and return the preferred fixed state.
    pub fn run(&self, program: &BytecodeProgram) -> ConvergenceStatus {
        self.run_report(program).status
    }

    /// Select a fixed state while retaining action and cache evidence.
    pub fn run_report(&self, program: &BytecodeProgram) -> BytecodeActionReport {
        let prepared = match PreparedBytecode::new(program.clone()) {
            Ok(prepared) => prepared,
            Err(error) => {
                return BytecodeActionReport {
                    status: ConvergenceStatus::Error {
                        message: format!("invalid bytecode: {error}"),
                        epoch: 0,
                    },
                    selected_action: None,
                    selected_batch: None,
                    stats: BytecodeActionStats::default(),
                };
            }
        };

        let mut vm_config = self.config.runtime.vm.clone();
        let mut tapes = FrozenTapes::from_config(&vm_config);
        let mut cache = EpochCache::default();

        // Preserve the established mode-independent shortcut: without an
        // oracle read there is one selected, zero-anamnesis epoch.
        if is_trivially_consistent(program) {
            return self.run_trivial(&prepared, &mut vm_config, &mut tapes, &mut cache);
        }

        let principle = ActionPrinciple::new(self.config.action.clone());
        let seeds = generate_diverse_seeds(
            self.config.num_seeds,
            self.config.runtime.memory_cells,
            self.config.runtime.seed,
            &self.config.runtime.initial_state,
        );
        let mut best_candidate: Option<Candidate> = None;
        let mut stats = BytecodeActionStats::default();

        for seed in seeds {
            stats.attempted_seeds += 1;
            match self.run_seed(&prepared, seed, &mut vm_config, &mut tapes, &mut cache) {
                Ok(Some(mut candidate)) => {
                    stats.consistent_candidates += 1;
                    candidate.action = principle.compute(
                        &candidate.memory,
                        &candidate.seed,
                        candidate.output.len(),
                    );
                    if best_candidate
                        .as_ref()
                        .is_none_or(|best| candidate_order(&candidate, best).is_lt())
                    {
                        best_candidate = Some(candidate);
                    }
                }
                Ok(None) => {}
                Err((message, epoch)) => {
                    stats.cache_hits = cache.hits;
                    stats.cache_misses = cache.misses;
                    stats.cache_entries = cache.entries.len();
                    stats.cache_evictions = cache.evictions;
                    return BytecodeActionReport {
                        status: ConvergenceStatus::Error { message, epoch },
                        selected_action: None,
                        selected_batch: None,
                        stats,
                    };
                }
            }
        }

        stats.cache_hits = cache.hits;
        stats.cache_misses = cache.misses;
        stats.cache_entries = cache.entries.len();
        stats.cache_evictions = cache.evictions;

        // Replacing only on strict improvement preserves deterministic seed
        // chronology for exact ties, matching the previous stable sort.
        if let Some(best) = best_candidate {
            let batch = match commit_selected_candidate(
                &best.memory,
                &best.output,
                &best.observations,
                &best.effects,
                &tapes.input,
                &self.config.runtime.vm,
            ) {
                Ok(batch) => batch,
                Err(message) => {
                    return BytecodeActionReport {
                        status: ConvergenceStatus::Error {
                            message,
                            epoch: best.epochs,
                        },
                        selected_action: Some(best.action),
                        selected_batch: None,
                        stats,
                    };
                }
            };
            let output = batch.output.clone();
            return BytecodeActionReport {
                status: ConvergenceStatus::Consistent {
                    memory: best.memory,
                    output,
                    epochs: best.epochs,
                },
                selected_action: Some(best.action),
                selected_batch: Some(batch),
                stats,
            };
        }

        // The legacy policy reports the zero/configured-seed standard orbit's
        // terminal reason when no explored seed is consistent. It also keeps
        // any external input captured by the first attempted epoch frozen.
        let mut fallback = self.config.runtime.clone();
        fallback.vm = vm_config;
        fallback.vm.input = tapes.input;
        fallback.vm.clock_input = tapes.clock;
        fallback.vm.random_input = tapes.random;
        fallback.vm.file_snapshots = tapes.files;
        fallback.vm.endpoint_tapes = tapes.endpoints;
        fallback.vm.process_results = tapes.processes;
        let status = match BytecodeTimeLoop::new(fallback) {
            Ok(driver) => driver.run(program),
            Err(message) => ConvergenceStatus::Error { message, epoch: 0 },
        };
        BytecodeActionReport {
            status,
            selected_action: None,
            selected_batch: None,
            stats,
        }
    }

    fn run_trivial(
        &self,
        program: &PreparedBytecode,
        vm_config: &mut BytecodeVmConfig,
        tapes: &mut FrozenTapes,
        cache: &mut EpochCache,
    ) -> BytecodeActionReport {
        let memory = match PagedMemory::with_size(self.config.runtime.memory_cells) {
            Ok(memory) => memory,
            Err(error) => {
                return BytecodeActionReport {
                    status: ConvergenceStatus::Error {
                        message: error.to_string(),
                        epoch: 0,
                    },
                    selected_action: None,
                    selected_batch: None,
                    stats: BytecodeActionStats::default(),
                };
            }
        };
        let result = match execute_epoch(program, &memory, vm_config, tapes, cache) {
            Ok(result) => result,
            Err(message) => {
                return BytecodeActionReport {
                    status: ConvergenceStatus::Error { message, epoch: 1 },
                    selected_action: None,
                    selected_batch: None,
                    stats: BytecodeActionStats {
                        attempted_seeds: 1,
                        cache_hits: cache.hits,
                        cache_misses: cache.misses,
                        cache_entries: cache.entries.len(),
                        cache_evictions: cache.evictions,
                        ..BytecodeActionStats::default()
                    },
                };
            }
        };
        let stats = BytecodeActionStats {
            attempted_seeds: 1,
            consistent_candidates: usize::from(result.status != BytecodeVmStatus::Paradox),
            cache_hits: cache.hits,
            cache_misses: cache.misses,
            cache_entries: cache.entries.len(),
            cache_evictions: cache.evictions,
        };
        match result.status {
            BytecodeVmStatus::Paradox => BytecodeActionReport {
                status: ConvergenceStatus::Paradox {
                    message: "Explicit PARADOX in trivial program".to_string(),
                    epoch: 1,
                },
                selected_action: None,
                selected_batch: None,
                stats,
            },
            BytecodeVmStatus::Finished | BytecodeVmStatus::Halted => {
                let memory = dense_memory(&result.present);
                let batch = match commit_selected_candidate(
                    &memory,
                    &result.output,
                    &result.observation_transcript(),
                    &result.effects,
                    &tapes.input,
                    &self.config.runtime.vm,
                ) {
                    Ok(batch) => batch,
                    Err(message) => {
                        return BytecodeActionReport {
                            status: ConvergenceStatus::Error { message, epoch: 1 },
                            selected_action: None,
                            selected_batch: None,
                            stats,
                        };
                    }
                };
                let output = batch.output.clone();
                BytecodeActionReport {
                    status: ConvergenceStatus::Consistent {
                        memory,
                        output,
                        epochs: 1,
                    },
                    selected_action: None,
                    selected_batch: Some(batch),
                    stats,
                }
            }
        }
    }

    fn run_seed(
        &self,
        program: &PreparedBytecode,
        seed: PagedMemory,
        vm_config: &mut BytecodeVmConfig,
        tapes: &mut FrozenTapes,
        cache: &mut EpochCache,
    ) -> Result<Option<Candidate>, (String, usize)> {
        let seed_dense = dense_memory(&seed);
        let mut anamnesis = seed;
        let mut seen = HashSet::<(usize, Vec<(Address, u64)>)>::new();
        let mut retained_orbit_bytes = 0usize;

        for epoch in 0..self.config.runtime.max_epochs {
            let sparse = anamnesis.numeric_sparse_state();
            let state_key = (anamnesis.len(), sparse);
            if seen.contains(&state_key) {
                return Ok(None);
            }
            retained_orbit_bytes =
                retain_orbit_state(retained_orbit_bytes, &anamnesis, &state_key.1)
                    .map_err(|message| (message, epoch))?;
            seen.insert(state_key);
            let result = execute_epoch(program, &anamnesis, vm_config, tapes, cache)
                .map_err(|message| (message, epoch + 1))?;
            match result.status {
                BytecodeVmStatus::Finished | BytecodeVmStatus::Halted => {
                    if result.present.numeric_values_equal(&anamnesis) {
                        let observations = result.observation_transcript();
                        return Ok(Some(Candidate {
                            memory: dense_memory(&result.present),
                            seed: seed_dense,
                            output: result.output.clone(),
                            observations,
                            effects: result.effects.clone(),
                            epochs: epoch + 1,
                            action: 0.0,
                        }));
                    }
                    anamnesis = result.present.clone();
                }
                BytecodeVmStatus::Paradox => return Ok(None),
            }
        }
        Ok(None)
    }
}

fn candidate_order(left: &Candidate, right: &Candidate) -> std::cmp::Ordering {
    left.action
        .partial_cmp(&right.action)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| left.memory.cmp(&right.memory))
}

fn execute_epoch(
    program: &PreparedBytecode,
    anamnesis: &PagedMemory,
    vm_config: &mut BytecodeVmConfig,
    tapes: &mut FrozenTapes,
    cache: &mut EpochCache,
) -> Result<Arc<BytecodeExecution>, String> {
    if let Some(result) = cache.get(anamnesis) {
        return Ok(result);
    }
    vm_config.endpoint_tapes = tapes.endpoints.clone();
    vm_config.process_results = tapes.processes.clone();
    let result = Arc::new(
        BytecodeVm::with_config(vm_config.clone())
            .run_prepared(program, anamnesis)
            .map_err(|error| error.to_string())?,
    );
    if vm_config.allow_interactive_input {
        if result.inputs_consumed.len() > tapes.input.len() {
            tapes.input = result.inputs_consumed.clone();
        }
        vm_config.input = tapes.input.clone();
        vm_config.allow_interactive_input = false;
    }
    if vm_config.allow_live_system_inputs {
        if result.clock_inputs_consumed.len() > tapes.clock.len() {
            tapes.clock = result.clock_inputs_consumed.clone();
        }
        if result.random_inputs_consumed.len() > tapes.random.len() {
            tapes.random = result.random_inputs_consumed.clone();
        }
        vm_config.clock_input = tapes.clock.clone();
        vm_config.random_input = tapes.random.clone();
        vm_config.allow_live_system_inputs = false;
    }
    if vm_config.allow_live_file_reads {
        for snapshot in &result.file_snapshots_consumed {
            if let Some(existing) = tapes
                .files
                .iter()
                .find(|configured| configured.path == snapshot.path)
            {
                if existing != snapshot {
                    return Err(format!(
                        "captured file snapshot for '{}' disagrees with frozen input",
                        snapshot.path
                    ));
                }
            } else {
                tapes.files.push(snapshot.clone());
            }
        }
        vm_config.file_snapshots = tapes.files.clone();
        vm_config.allow_live_file_reads = false;
    }
    cache.insert(anamnesis.clone(), Arc::clone(&result));
    Ok(result)
}

fn generate_diverse_seeds(
    count: usize,
    memory_cells: usize,
    configured_seed: u64,
    initial_state: &[(Address, u64)],
) -> Vec<PagedMemory> {
    let mut seeds = Vec::with_capacity(count);
    let zero =
        || PagedMemory::with_size(memory_cells).expect("configuration validated memory size");
    seeds.push(zero());
    if count > 1 {
        let mut primes = zero();
        for (index, value) in [
            2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
        ]
        .into_iter()
        .take(memory_cells)
        .enumerate()
        {
            primes
                .write(index as Address, Value::new(value))
                .expect("seed address in range");
        }
        seeds.push(primes);
    }
    if count > 2 {
        let mut sequential = zero();
        for index in 0..16.min(memory_cells) {
            sequential
                .write(index as Address, Value::new(index as u64 + 1))
                .expect("seed address in range");
        }
        seeds.push(sequential);
    }
    if count > 3 {
        let mut powers = zero();
        for index in 0..16.min(memory_cells) {
            powers
                .write(index as Address, Value::new(1u64 << index))
                .expect("seed address in range");
        }
        seeds.push(powers);
    }

    for seed_index in 4..count {
        let mut memory = zero();
        let base = configured_seed.wrapping_add(seed_index as u64 * 12_345);
        for index in 0..16.min(memory_cells) {
            let value = base
                .wrapping_mul(index as u64 + 1)
                .wrapping_add(seed_index as u64)
                % 1_000;
            memory
                .write(index as Address, Value::new(value))
                .expect("seed address in range");
        }
        seeds.push(memory);
    }

    // The configured standard-orbit initial state is part of the action
    // search, not a ledgerless success fallback. Preserve the requested seed
    // budget by replacing the last heuristic only when it is distinct.
    let mut configured = zero();
    if !initial_state.is_empty() {
        for (address, value) in initial_state {
            configured
                .write(*address, Value::new(*value))
                .expect("validated explicit initial-state address");
        }
    } else if configured_seed != 0 {
        for index in 0..16.min(memory_cells) {
            configured
                .write(
                    index as Address,
                    Value::new(configured_seed.wrapping_mul(index as u64 + 1)),
                )
                .expect("configured seed address in range");
        }
    }
    if !seeds.contains(&configured) {
        let last = seeds
            .last_mut()
            .expect("positive action seed count was validated");
        *last = configured;
    }
    seeds
}

fn commit_selected_candidate(
    memory: &Memory,
    output: &[OutputItem],
    observations: &ObservationTranscript,
    effects: &[EffectIntent],
    frozen_input: &[u64],
    vm_config: &BytecodeVmConfig,
) -> Result<CommittedBatch, String> {
    let mut transaction =
        TemporalTransaction::new(frozen_input.to_vec(), transaction_limits(vm_config))
            .map_err(|error| error.to_string())?;
    let mut context = transaction
        .begin_candidate()
        .map_err(|error| error.to_string())?;
    for expected in &observations.input {
        let frozen = context.read_input().map_err(|error| error.to_string())?;
        if frozen != *expected {
            return Err("bytecode input consumption disagrees with frozen tape".to_string());
        }
    }
    for item in output {
        context
            .stage_output(item.clone())
            .map_err(|error| error.to_string())?;
    }
    for effect in effects {
        context
            .stage_effect(effect.clone())
            .map_err(|error| error.to_string())?;
    }
    context
        .set_observation_transcript(observations.clone())
        .map_err(|error| error.to_string())?;
    let state = memory.exact_sparse_state().cells().to_vec();
    let candidate = context.finish(state).map_err(|error| error.to_string())?;
    let timeline = transaction
        .stage_candidate(candidate)
        .map_err(|error| error.to_string())?;
    transaction
        .select(timeline)
        .map_err(|error| error.to_string())?;
    let mut log = CommitLog::default();
    transaction
        .commit_selected(CommitToken(u128::from(timeline.0)), &mut log)
        .map_err(|error| error.to_string())?;
    log.batches()
        .first()
        .cloned()
        .ok_or_else(|| "selected timeline produced no commit record".to_string())
}

fn dense_memory(memory: &PagedMemory) -> Memory {
    let mut dense = Memory::with_size(memory.len());
    for (address, value) in memory.iter_sparse() {
        dense.write(address, value.clone());
    }
    dense
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Effect, OpCode, Procedure, Program, QuoteId, Stmt};
    use crate::hir::HirProgram;
    use crate::temporal::timeloop::{ExecutionMode, TimeLoop, TimeLoopConfig};

    fn compile(program: &Program) -> BytecodeProgram {
        BytecodeProgram::compile(&HirProgram::resolve(program).unwrap()).unwrap()
    }

    fn numeric_output(output: &[OutputItem]) -> Vec<(u8, u64)> {
        output
            .iter()
            .map(|item| match item {
                OutputItem::Val(value) => (0, value.val),
                OutputItem::Char(value) => (1, u64::from(*value)),
            })
            .collect()
    }

    fn source_action(program: &Program, num_seeds: usize) -> ConvergenceStatus {
        TimeLoop::new(TimeLoopConfig {
            max_epochs: 100,
            mode: ExecutionMode::ActionGuided {
                config: ActionConfig::anti_trivial(),
                num_seeds,
            },
            memory_cells: 4,
            ..TimeLoopConfig::default()
        })
        .unwrap()
        .run(program)
    }

    fn runner(num_seeds: usize) -> BytecodeActionRunner {
        BytecodeActionRunner::new(BytecodeActionConfig {
            runtime: BytecodeTimeLoopConfig {
                memory_cells: 4,
                max_epochs: 100,
                ..BytecodeTimeLoopConfig::default()
            },
            action: ActionConfig::anti_trivial(),
            num_seeds,
        })
        .unwrap()
    }

    #[test]
    fn seed_count_is_bounded_before_seed_vector_allocation() {
        let error = BytecodeActionRunner::new(BytecodeActionConfig {
            num_seeds: usize::MAX,
            ..BytecodeActionConfig::default()
        })
        .unwrap_err();
        assert!(error.contains("exceeds limit"), "{error}");
    }

    #[test]
    fn epoch_cache_evicts_by_aggregate_retained_bytes() {
        let bytecode = compile(&Program {
            body: vec![Stmt::Push(Value::new(1)), Stmt::Op(OpCode::Pop)],
            ..Program::default()
        });
        let first_state = PagedMemory::with_size(4).unwrap();
        let result = Arc::new(BytecodeVm::new().run(&bytecode, &first_state).unwrap());
        let one_entry = first_state
            .retained_size_charge()
            .saturating_add(result.retained_size_charge());
        let mut cache = EpochCache {
            max_retained_bytes: one_entry,
            ..EpochCache::default()
        };
        cache.insert(first_state, Arc::clone(&result));
        assert_eq!(cache.entries.len(), 1);

        let mut second_state = PagedMemory::with_size(4).unwrap();
        second_state.write(0, Value::new(1)).unwrap();
        cache.insert(second_state.clone(), result);
        assert_eq!(cache.entries.len(), 1);
        assert!(cache.entries.contains_key(&second_state));
        assert_eq!(cache.evictions, 1);
        assert!(cache.retained_bytes <= cache.max_retained_bytes);
    }

    #[test]
    fn output_guided_selection_matches_legacy_policy() {
        let mut source = Program::new();
        source.body = vec![
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Oracle),
            Stmt::Op(OpCode::Dup),
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Eq),
            Stmt::Op(OpCode::Not),
            Stmt::If {
                then_branch: vec![
                    Stmt::Op(OpCode::Dup),
                    Stmt::Op(OpCode::Output),
                    Stmt::Op(OpCode::Dup),
                    Stmt::Push(Value::new(0)),
                    Stmt::Op(OpCode::Prophecy),
                ],
                else_branch: Some(vec![
                    Stmt::Push(Value::new(1)),
                    Stmt::Push(Value::new(0)),
                    Stmt::Op(OpCode::Prophecy),
                ]),
            },
        ];
        let legacy = source_action(&source, 4);
        let report = runner(4).run_report(&compile(&source));

        match (legacy, report.status) {
            (
                ConvergenceStatus::Consistent {
                    memory: legacy_memory,
                    output: legacy_output,
                    ..
                },
                ConvergenceStatus::Consistent { memory, output, .. },
            ) => {
                assert_eq!(memory.sparse_state(), legacy_memory.sparse_state());
                assert_eq!(numeric_output(&output), numeric_output(&legacy_output));
                assert!(!output.is_empty());
            }
            other => panic!("action policies diverged: {other:?}"),
        }
        assert!(report.selected_action.is_some());
        assert!(report.stats.consistent_candidates >= 1);
    }

    #[test]
    fn procedures_and_quotations_execute_on_the_bytecode_policy_path() {
        let mut source = Program::new();
        source
            .quotes
            .push(vec![Stmt::Op(OpCode::Dup), Stmt::Op(OpCode::Output)]);
        source.procedures.push(Procedure {
            name: "choose".to_string(),
            params: Vec::new(),
            returns: 0,
            effects: vec![Effect::Temporal, Effect::IO],
            body: vec![
                Stmt::Push(Value::new(0)),
                Stmt::Op(OpCode::Oracle),
                Stmt::PushQuote(QuoteId::new(0)),
                Stmt::Op(OpCode::Exec),
                Stmt::Push(Value::new(0)),
                Stmt::Op(OpCode::Prophecy),
            ],
        });
        source.body = vec![Stmt::Call {
            name: "choose".to_string(),
        }];

        // The compatibility executor historically requires this preprocessing
        // step; the bytecode path deliberately executes the real procedure
        // table without source inlining.
        let legacy = source_action(&source.inline_procedures(), 4);
        let bytecode = compile(&source);
        assert!(!bytecode.procedures.is_empty());
        assert!(!bytecode.quotations.is_empty());
        let report = runner(4).run_report(&bytecode);

        match (legacy, report.status) {
            (
                ConvergenceStatus::Consistent {
                    memory: legacy_memory,
                    output: legacy_output,
                    ..
                },
                ConvergenceStatus::Consistent { memory, output, .. },
            ) => {
                assert_eq!(memory.sparse_state(), legacy_memory.sparse_state());
                assert_eq!(numeric_output(&output), numeric_output(&legacy_output));
            }
            other => panic!("procedure/quotation policies diverged: {other:?}"),
        }
        assert_eq!(report.stats.attempted_seeds, 4);
    }

    #[test]
    fn unreachable_oracle_procedure_does_not_disable_trivial_shortcut() {
        let mut source = Program::new();
        source.procedures.push(Procedure {
            name: "unused_oracle".to_string(),
            params: Vec::new(),
            returns: 0,
            effects: vec![Effect::Temporal],
            body: vec![
                Stmt::Push(Value::ZERO),
                Stmt::Op(OpCode::Oracle),
                Stmt::Op(OpCode::Pop),
            ],
        });
        source.body = vec![Stmt::Push(Value::new(9)), Stmt::Op(OpCode::Output)];
        let bytecode = compile(&source);
        assert!(bytecode.instructions.iter().any(|instruction| matches!(
            instruction,
            crate::bytecode::Instruction::Primitive(OpCode::Oracle)
        )));

        let mut runner = runner(4);
        runner.config.runtime.seed = 99;
        let report = runner.run_report(&bytecode);
        match report.status {
            ConvergenceStatus::Consistent {
                memory,
                output,
                epochs,
            } => {
                assert_eq!(epochs, 1);
                assert!(memory.sparse_state().is_empty());
                assert_eq!(numeric_output(&output), vec![(0, 9)]);
            }
            other => panic!("unused procedure changed trivial semantics: {other:?}"),
        }
        assert_eq!(report.stats.attempted_seeds, 1);
    }

    #[test]
    fn runtime_error_terminates_action_search_without_retrying_other_seeds() {
        let mut source = Program::new();
        source.body = vec![
            Stmt::Op(OpCode::Input),
            Stmt::Op(OpCode::Pop),
            Stmt::Push(Value::new(77)),
            Stmt::Op(OpCode::BufferLen),
            Stmt::Op(OpCode::Pop),
            Stmt::Op(OpCode::Pop),
            Stmt::Push(Value::ZERO),
            Stmt::Op(OpCode::Oracle),
            Stmt::Op(OpCode::Pop),
        ];
        let mut runner = runner(4);
        runner.config.runtime.vm.input = vec![1];
        let report = runner.run_report(&compile(&source));
        assert!(matches!(
            report.status,
            ConvergenceStatus::Error { epoch: 1, ref message }
                if message.contains("invalid buffer handle 77")
        ));
        assert_eq!(report.stats.attempted_seeds, 1);
    }

    #[test]
    fn exact_epoch_cache_is_shared_when_seed_orbits_meet() {
        let mut source = Program::new();
        source.body = vec![
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Oracle),
            Stmt::Op(OpCode::Pop),
            Stmt::Push(Value::new(7)),
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Prophecy),
        ];
        let report = runner(4).run_report(&compile(&source));
        assert!(matches!(
            report.status,
            ConvergenceStatus::Consistent { .. }
        ));
        assert!(report.stats.cache_hits >= 1, "stats: {:?}", report.stats);
    }

    #[test]
    fn gas_errors_fall_back_to_the_standard_seed_reason() {
        let mut source = Program::new();
        source.body = vec![
            Stmt::Push(Value::ZERO),
            Stmt::Op(OpCode::Oracle),
            Stmt::Op(OpCode::Pop),
            Stmt::While {
                cond: vec![Stmt::Push(Value::ONE)],
                body: vec![Stmt::Op(OpCode::Nop)],
            },
        ];
        let action = BytecodeActionRunner::new(BytecodeActionConfig {
            runtime: BytecodeTimeLoopConfig {
                memory_cells: 4,
                max_epochs: 2,
                vm: BytecodeVmConfig {
                    max_instructions: 8,
                    ..BytecodeVmConfig::default()
                },
                ..BytecodeTimeLoopConfig::default()
            },
            action: ActionConfig::default(),
            num_seeds: 2,
        })
        .unwrap();
        match action.run(&compile(&source)) {
            ConvergenceStatus::Error { message, epoch } => {
                assert_eq!(epoch, 1);
                assert!(
                    message.contains("instruction limit") && message.contains("exhausted"),
                    "{message}"
                );
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn trivial_shortcut_is_one_selected_epoch() {
        let mut source = Program::new();
        source.body = vec![
            Stmt::Push(Value::new(42)),
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Prophecy),
            Stmt::Push(Value::new(9)),
            Stmt::Op(OpCode::Output),
        ];
        assert!(matches!(
            runner(4).run(&compile(&source)),
            ConvergenceStatus::Consistent { epochs: 1, .. }
        ));
    }

    #[test]
    fn selected_action_batch_retains_non_input_observation_transcript() {
        let mut source = Program::new();
        source.body = vec![Stmt::Op(OpCode::Clock), Stmt::Op(OpCode::Output)];
        let mut runner = runner(4);
        runner.config.runtime.vm.clock_input = vec![123];
        let report = runner.run_report(&compile(&source));
        assert!(matches!(
            report.status,
            ConvergenceStatus::Consistent { epochs: 1, .. }
        ));
        assert_eq!(report.selected_batch.unwrap().observations.clock, vec![123]);
    }

    #[test]
    fn configured_seed_success_retains_the_selected_effect_batch() {
        let mut source = Program::new();
        source.body = vec![
            Stmt::Push(Value::ZERO),
            Stmt::Op(OpCode::Oracle),
            Stmt::Push(Value::new(99)),
            Stmt::Op(OpCode::Eq),
            Stmt::If {
                then_branch: vec![
                    Stmt::Push(Value::new(99)),
                    Stmt::Push(Value::ZERO),
                    Stmt::Op(OpCode::Prophecy),
                    Stmt::Push(Value::ONE),
                    Stmt::Op(OpCode::Sleep),
                ],
                else_branch: Some(vec![Stmt::Op(OpCode::Paradox)]),
            },
        ];
        let action = BytecodeActionRunner::new(BytecodeActionConfig {
            runtime: BytecodeTimeLoopConfig {
                memory_cells: 4,
                max_epochs: 4,
                seed: 99,
                vm: BytecodeVmConfig {
                    max_sleep_milliseconds: Some(1),
                    ..BytecodeVmConfig::default()
                },
                ..BytecodeTimeLoopConfig::default()
            },
            action: ActionConfig::default(),
            num_seeds: 4,
        })
        .unwrap();

        let report = action.run_report(&compile(&source));
        assert!(matches!(
            report.status,
            ConvergenceStatus::Consistent { epochs: 2, .. }
        ));
        assert_eq!(report.stats.attempted_seeds, 4);
        assert_eq!(report.stats.consistent_candidates, 1);
        assert!(matches!(
            report.selected_batch.unwrap().effects.as_slice(),
            [EffectIntent::Sleep { milliseconds: 1 }]
        ));
    }
}
