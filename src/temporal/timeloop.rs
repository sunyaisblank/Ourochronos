//! Fixed-point computation and temporal loop execution.
//!
//! This module implements the core OUROCHRONOS execution model:
//! repeatedly run epochs until Present = Anamnesis (fixed point achieved).
//!
//! The module also provides point-cycle diagnosis and Deutsch stationary-cycle
//! semantics for deterministic transitions.
//!
//! ## Action-Guided Execution
//!
//! The `ActionGuided` mode implements the **Action Principle** to solve the "Genie Effect".
//! Instead of accepting the first fixed point found, it explores multiple seed strategies
//! and selects the solution with minimum action (cost), preferring non-trivial,
//! output-producing solutions.

use crate::ast::{EffectClass, OpCode, Program};
use crate::bytecode::{BytecodeProgram, Instruction};
use crate::bytecode_action::{BytecodeActionConfig, BytecodeActionRunner, MAX_ACTION_SEEDS};
use crate::bytecode_timeloop::{is_trivially_consistent, BytecodeTimeLoop, BytecodeTimeLoopConfig};
use crate::bytecode_verifier::verify_default as verify_bytecode;
use crate::bytecode_vm::BytecodeVmConfig;
use crate::core::error::ErrorConfig;
use crate::core::{Address, Memory, OutputItem};
use crate::hir::HirProgram;
use crate::temporal::action::ActionConfig;
use crate::temporal::cache::CacheStats;
use crate::vm::EffectsPolicy;
use std::collections::{HashSet, VecDeque};

/// Result of fixed-point search.
#[derive(Debug)]
pub enum ConvergenceStatus {
    /// Fixed point found: P = A.
    Consistent {
        /// The consistent memory state.
        memory: Memory,
        /// Output produced.
        output: Vec<OutputItem>,
        /// Number of epochs to converge.
        epochs: usize,
    },

    /// Deutsch-consistent stationary ensemble for a deterministic cycle.
    ///
    /// A deterministic map need not have a point fixed state, but the
    /// uniform distribution over every finite cycle is a fixed point of the
    /// induced stochastic map. This is the classical Deutsch semantics used
    /// by the Aaronson--Watrous complexity model.
    DeutschConsistent {
        /// Recurrent states in transition order. Each has probability 1/period.
        cycle: Vec<Memory>,
        /// Chronology-respecting output produced from each corresponding state.
        outputs: Vec<Vec<OutputItem>>,
        /// Cycle length (and denominator of each uniform probability).
        period: usize,
        /// Number of transient transitions before the recurrent class.
        transient_epochs: usize,
        /// The common observable output, only when every cycle state agrees.
        unanimous_output: Option<Vec<OutputItem>>,
    },

    /// Explicit PARADOX instruction reached.
    Paradox {
        /// Diagnostic message.
        message: String,
        /// Epoch where paradox occurred.
        epoch: usize,
    },

    /// Oscillation detected (cycle of length > 1).
    Oscillation {
        /// Cycle period.
        period: usize,
        /// Addresses that oscillate.
        oscillating_cells: Vec<Address>,
        /// Diagnosis of the paradox.
        diagnosis: ParadoxDiagnosis,
    },

    /// Heuristic monotone-growth diagnosis from a finite trajectory prefix.
    Divergence {
        /// Cells that diverge.
        diverging_cells: Vec<Address>,
        /// Direction of divergence.
        direction: Direction,
    },

    /// Epoch limit reached without convergence.
    Timeout {
        /// Maximum epochs attempted.
        max_epochs: usize,
    },

    /// Runtime error during execution.
    Error { message: String, epoch: usize },
}

/// Direction of divergence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Increasing,
    Decreasing,
}

/// Diagnosis of why a paradox occurred.
#[derive(Debug, Clone)]
pub enum ParadoxDiagnosis {
    /// Negative causal loop (grandfather paradox).
    NegativeLoop {
        /// Cells involved in the loop.
        cells: Vec<Address>,
        /// Human-readable explanation.
        explanation: String,
    },

    /// General oscillation.
    Oscillation {
        /// Cycle of memory states.
        cycle: Vec<Vec<(Address, u64)>>,
    },

    /// Unknown cause.
    Unknown,
}

/// Execution mode.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionMode {
    /// Basic iteration with cycle detection.
    Standard,
    /// Full trajectory recording and analysis.
    Diagnostic,
    /// Deutsch consistency: accept a deterministic cycle as the uniform
    /// stationary distribution of the induced stochastic transition.
    Deutsch,
    /// Uncached iteration with a high implementation safety ceiling.
    Pure,
    /// Action-guided search: explore multiple seeds and select the best fixed point.
    /// This mode implements the Action Principle to solve the "Genie Effect".
    ActionGuided {
        /// Configuration for the action principle.
        config: ActionConfig,
        /// Number of different seeds to try.
        num_seeds: usize,
    },
}

/// Default epoch budget shared by the CLI, the REPL, and the library, so a
/// programme behaves identically at every entry point unless overridden.
pub const DEFAULT_MAX_EPOCHS: usize = 1000;

/// Configuration for the time loop.
#[derive(Debug, Clone)]
pub struct TimeLoopConfig {
    /// Maximum epochs before timeout.
    pub max_epochs: usize,
    /// Execution mode.
    pub mode: ExecutionMode,
    /// Initial seed for anamnesis (0 = all zeros).
    pub seed: u64,
    /// Whether to print progress.
    pub verbose: bool,
    /// Pre-specified inputs (frozen). If non-empty, these are used instead of interactive input.
    /// This ensures deterministic execution across epochs.
    pub frozen_inputs: Vec<u64>,
    /// Maximum instructions per epoch (gas limit).
    pub max_instructions: u64,
    /// Number of temporal memory cells for this run.
    pub memory_cells: usize,
    /// Error handling configuration for bounds checking, division, etc.
    pub error_config: ErrorConfig,
    /// Provenance saturation limit. Defaults to DEFAULT_PROVENANCE_SATURATION_LIMIT.
    pub provenance_limit: usize,
    /// Policy for external effects and non-determinism during the search.
    pub effects: crate::vm::EffectsPolicy,
}

impl Default for TimeLoopConfig {
    fn default() -> Self {
        Self {
            max_epochs: DEFAULT_MAX_EPOCHS,
            mode: ExecutionMode::Standard,
            seed: 0,
            verbose: false,
            frozen_inputs: Vec::new(),
            max_instructions: 10_000_000,
            memory_cells: crate::core::MEMORY_SIZE,
            error_config: ErrorConfig::default(),
            provenance_limit: crate::core::provenance::DEFAULT_PROVENANCE_SATURATION_LIMIT,
            effects: crate::vm::EffectsPolicy::default(),
        }
    }
}

impl TimeLoopConfig {
    /// Validate configuration values, returning any constraint violations.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if self.max_epochs == 0 {
            errors.push("max_epochs must be > 0".to_string());
        }
        if self.max_instructions == 0 {
            errors.push("max_instructions must be > 0".to_string());
        }
        if self.memory_cells == 0 {
            errors.push("memory_cells must be > 0".to_string());
        } else if self.memory_cells > crate::core::MAX_DENSE_MEMORY_CELLS {
            errors.push(format!(
                "memory_cells {} exceeds dense result limit {}",
                self.memory_cells,
                crate::core::MAX_DENSE_MEMORY_CELLS
            ));
        }
        if self.provenance_limit == 0 {
            errors.push("provenance_limit must be > 0".to_string());
        }
        if let ExecutionMode::ActionGuided { num_seeds, .. } = &self.mode {
            if *num_seeds == 0 {
                errors.push("num_seeds must be > 0 in ActionGuided mode".to_string());
            } else if *num_seeds > MAX_ACTION_SEEDS {
                errors.push(format!(
                    "num_seeds {num_seeds} exceeds ActionGuided limit {MAX_ACTION_SEEDS}"
                ));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Compatibility source facade over the canonical bytecode temporal engines.
///
/// `Program` remains the stable library input type, but it is never executed
/// directly: every run resolves typed HIR, compiles and independently verifies
/// one bytecode artifact, then selects a bytecode-only orbit policy. Source
/// tooling performs stricter mandatory static admission separately.
pub struct TimeLoop {
    config: TimeLoopConfig,
    last_cache_stats: CacheStats,
}

impl TimeLoop {
    /// Create a new time loop, rejecting invalid configurations.
    ///
    /// Applies `config.provenance_limit` to the thread-local saturation limit
    /// consulted by every provenance merge. Construct one driver per run on a
    /// thread when different saturation policies are required.
    pub fn new(config: TimeLoopConfig) -> crate::core::error::OuroResult<Self> {
        config
            .validate()
            .map_err(|errors| crate::core::error::OuroError::InvalidConfiguration { errors })?;
        crate::core::provenance::set_saturation_limit(config.provenance_limit);
        crate::core::provenance::set_address_space_size(config.memory_cells);
        Ok(Self {
            config,
            last_cache_stats: CacheStats::default(),
        })
    }

    /// Statistics from the most recent run.
    ///
    /// Standard, diagnostic, Deutsch, and pure bytecode orbits deliberately do
    /// not memoize epoch results and therefore report zeroes. Action-guided
    /// selection exposes its exact cross-seed cache hit and miss counters. The
    /// `size` records how many entries the run-local action cache retained at
    /// completion, while the driver retains only these counters, not results.
    pub fn cache_stats(&self) -> CacheStats {
        self.last_cache_stats.clone()
    }

    /// Compile the source program once and dispatch the configured bytecode
    /// fixed-point policy. No source-AST executor is reachable from this API.
    pub fn run(&mut self, program: &Program) -> ConvergenceStatus {
        self.last_cache_stats = CacheStats::default();
        let bytecode = match compile_program(program) {
            Ok(bytecode) => bytecode,
            Err(message) => return ConvergenceStatus::Error { message, epoch: 0 },
        };
        let trivial = is_trivially_consistent(&bytecode);
        if !trivial && self.config.effects == EffectsPolicy::Decline {
            if let Some((name, class)) = first_reachable_effect(&bytecode) {
                let kind = match class {
                    EffectClass::Pure => unreachable!("only rejected effects are returned"),
                    EffectClass::External => "external effect",
                    EffectClass::NonDeterministic => "non-deterministic operation",
                };
                return ConvergenceStatus::Error {
                    message: format!(
                        "{kind} {name} declined during fixed-point search; use EffectsPolicy::Unrestricted to permit deterministic capture or explicit capabilities"
                    ),
                    epoch: 1,
                };
            }
        }

        let runtime = match self.bytecode_config(trivial) {
            Ok(runtime) => runtime,
            Err(message) => return ConvergenceStatus::Error { message, epoch: 0 },
        };
        match &self.config.mode {
            ExecutionMode::ActionGuided { config, num_seeds } => {
                let runner = match BytecodeActionRunner::new(BytecodeActionConfig {
                    runtime,
                    action: config.clone(),
                    num_seeds: *num_seeds,
                }) {
                    Ok(runner) => runner,
                    Err(message) => return ConvergenceStatus::Error { message, epoch: 0 },
                };
                let report = runner.run_report(&bytecode);
                let attempts = report.stats.cache_hits + report.stats.cache_misses;
                self.last_cache_stats = CacheStats {
                    size: report.stats.cache_entries,
                    hits: report.stats.cache_hits,
                    misses: report.stats.cache_misses,
                    evictions: report.stats.cache_evictions,
                    hit_rate: if attempts == 0 {
                        0.0
                    } else {
                        report.stats.cache_hits as f64 / attempts as f64
                    },
                };
                report.status
            }
            mode => {
                let driver = match BytecodeTimeLoop::new(runtime) {
                    Ok(driver) => driver,
                    Err(message) => return ConvergenceStatus::Error { message, epoch: 0 },
                };
                match mode {
                    ExecutionMode::Standard => driver.run(&bytecode),
                    ExecutionMode::Diagnostic => driver.run_diagnostic(&bytecode),
                    ExecutionMode::Deutsch => driver.run_deutsch(&bytecode),
                    ExecutionMode::Pure => driver.run_pure(&bytecode),
                    ExecutionMode::ActionGuided { .. } => unreachable!("handled above"),
                }
            }
        }
    }

    fn bytecode_config(
        &self,
        permit_live_observations: bool,
    ) -> Result<BytecodeTimeLoopConfig, String> {
        let mut vm = BytecodeVmConfig {
            max_instructions: self.config.max_instructions,
            memory_bounds: self.config.error_config.memory_bounds,
            input: self.config.frozen_inputs.clone(),
            allow_interactive_input: self.config.frozen_inputs.is_empty(),
            // A one-candidate program is the selected timeline itself. For a
            // real search, Unrestricted authorizes first-candidate capture;
            // the bytecode loop freezes and replays the observation thereafter.
            allow_live_system_inputs: permit_live_observations
                || self.config.effects == EffectsPolicy::Unrestricted,
            ..BytecodeVmConfig::default()
        };
        if self.config.error_config.max_stack_depth != 0 {
            vm.max_stack_depth = self.config.error_config.max_stack_depth;
        }
        BytecodeTimeLoop::new(BytecodeTimeLoopConfig {
            max_epochs: self.config.max_epochs,
            memory_cells: self.config.memory_cells,
            seed: self.config.seed,
            initial_state: Vec::new(),
            vm,
        })
        .map(|driver| driver.config)
    }
}

fn compile_program(program: &Program) -> Result<BytecodeProgram, String> {
    let hir = HirProgram::resolve(program).map_err(|errors| {
        format!(
            "typed name resolution failed: {}",
            errors
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; ")
        )
    })?;
    let bytecode = BytecodeProgram::compile(&hir)
        .map_err(|error| format!("bytecode compilation failed: {error}"))?;
    verify_bytecode(&bytecode)
        .map_err(|error| format!("independent bytecode verification failed: {error}"))?;
    Ok(bytecode)
}

/// Return the first effect reachable from main. Direct procedure calls are
/// followed exactly; a reachable dynamic quotation combinator conservatively
/// makes every quotation reachable.
fn first_reachable_effect(program: &BytecodeProgram) -> Option<(String, EffectClass)> {
    let mut pending = VecDeque::from([program.main]);
    let mut visited = HashSet::new();
    let mut all_quotations_added = false;
    while let Some(range) = pending.pop_front() {
        if !visited.insert((range.start, range.end)) {
            continue;
        }
        let instructions = program
            .instructions
            .get(range.start as usize..range.end as usize)?;
        for instruction in instructions {
            match instruction {
                Instruction::Primitive(opcode) if opcode.effect_class() != EffectClass::Pure => {
                    return Some((opcode.name().to_string(), opcode.effect_class()));
                }
                Instruction::CallForeign(id) => {
                    let foreign = program.foreigns.get(id.index())?;
                    if !foreign.effects.is_pure() {
                        return Some((
                            format!("FOREIGN {}::{}", foreign.library, foreign.symbol),
                            EffectClass::External,
                        ));
                    }
                }
                Instruction::CallProcedure(id) => {
                    pending.push_back(program.procedures.get(id.index())?.range);
                }
                Instruction::Primitive(
                    OpCode::Exec | OpCode::Dip | OpCode::Keep | OpCode::Bi | OpCode::Rec,
                ) if !all_quotations_added => {
                    pending.extend(program.quotations.iter().map(|quote| quote.range));
                    all_quotations_added = true;
                }
                _ => {}
            }
        }
    }
    None
}

// Retained only in test builds as an independent differential oracle. The
// production `TimeLoop` above has no field, call, or branch referring to it.
#[cfg(test)]
mod legacy_ast_oracle {
    use super::*;
    use crate::core::memory::ExactMemoryState;
    use crate::core::Value;
    use crate::temporal::action::{ActionPrinciple, FixedPointSelector};
    use crate::temporal::cache::{EpochCache, MemoizedResult};
    use crate::vm::{EpochStatus, Executor, ExecutorConfig};
    use std::collections::HashMap;

    type JournalEntry = (usize, usize, Vec<(Address, u64)>);

    /// Journal of anamnesis states visited during the differential AST oracle.
    struct StateJournal {
        visits: HashMap<u64, Vec<JournalEntry>>,
    }

    impl StateJournal {
        fn new() -> Self {
            Self {
                visits: HashMap::new(),
            }
        }

        fn check_and_insert(
            &mut self,
            hash: u64,
            width: usize,
            state: &[(Address, u64)],
            epoch: usize,
        ) -> Option<usize> {
            let entry = self.visits.entry(hash).or_default();
            if let Some((previous, _, _)) = entry.iter().find(|(_, stored_width, stored)| {
                *stored_width == width && stored.as_slice() == state
            }) {
                return Some(*previous);
            }
            entry.push((epoch, width, state.to_vec()));
            None
        }
    }

    enum Step {
        Done(ConvergenceStatus),
        Continue(Vec<OutputItem>),
    }

    /// The temporal loop driver.
    pub struct TimeLoop {
        config: TimeLoopConfig,
        executor: Executor,
        /// Epoch result cache for memoization during fixed-point search.
        cache: EpochCache,
    }

    impl TimeLoop {
        /// Create a new time loop, rejecting invalid configurations.
        ///
        /// Applies `config.provenance_limit` to the thread-local saturation
        /// limit consulted by every Provenance merge. The limit is necessarily
        /// thread-wide (Value's operator overloads cannot carry configuration),
        /// so the most recently constructed TimeLoop on a thread wins; construct
        /// one TimeLoop per run.
        pub fn new(config: TimeLoopConfig) -> crate::core::error::OuroResult<Self> {
            config
                .validate()
                .map_err(|errors| crate::core::error::OuroError::InvalidConfiguration { errors })?;
            crate::core::provenance::set_saturation_limit(config.provenance_limit);
            crate::core::provenance::set_address_space_size(config.memory_cells);

            let mut exec_config = ExecutorConfig {
                immediate_output: config.verbose,
                max_instructions: config.max_instructions,
                error_config: config.error_config.clone(),
                context: crate::vm::EpochContext::SingleEpoch,
                ..ExecutorConfig::default()
            };

            // Use frozen inputs if provided
            if !config.frozen_inputs.is_empty() {
                exec_config.input = config.frozen_inputs.clone();
            }

            Ok(Self {
                config,
                executor: Executor::with_config(exec_config),
                cache: EpochCache::with_capacity(1024),
            })
        }

        /// Get cache statistics.
        pub fn cache_stats(&self) -> crate::temporal::cache::CacheStats {
            self.cache.stats()
        }

        /// Run the fixed-point search.
        pub fn run(&mut self, program: &Program) -> ConvergenceStatus {
            // Cache entries are keyed on the anamnesis alone, not the programme;
            // a retained entry from a previous run() of a different programme
            // would be served as that programme's result. One run, one cache.
            self.cache.clear();

            // A trivially consistent programme's single epoch IS the consistent
            // timeline, so the effect gate does not apply to it.
            if program.is_trivially_consistent() {
                self.executor.config.context = crate::vm::EpochContext::SingleEpoch;
                return self.run_trivial(program);
            }
            self.executor.config.context =
                crate::vm::EpochContext::FixedPointSearch(self.config.effects);

            match &self.config.mode {
                ExecutionMode::Standard => self.run_standard(program),
                ExecutionMode::Diagnostic => self.run_diagnostic(program),
                ExecutionMode::Deutsch => self.run_deutsch(program),
                ExecutionMode::Pure => self.run_pure(program),
                ExecutionMode::ActionGuided { config, num_seeds } => {
                    self.run_action_guided(program, config.clone(), *num_seeds)
                }
            }
        }

        /// Capture the inputs consumed by the first epoch that read any, and
        /// replay them in every subsequent epoch.
        ///
        /// This is the Temporal Input Invariant: information flows from the
        /// external world into the loop, never backwards. Without freezing, an
        /// interactive INPUT would be re-read on every epoch of the search, so
        /// the epoch function F would vary between iterations and the fixed
        /// point would depend on how often the user was prompted.
        fn freeze_inputs(&mut self, result: &crate::vm::EpochResult) {
            // A non-empty executor input IS the frozen stream, whether supplied
            // up front via frozen_inputs or captured here; one home for the data.
            if self.executor.config.input.is_empty() && !result.inputs_consumed.is_empty() {
                self.executor.config.input = result.inputs_consumed.clone();
                if self.config.verbose {
                    println!("Input frozen: {:?}", self.executor.config.input);
                }
            }
        }

        /// The source executor is retained as a differential compatibility oracle,
        /// but its legacy `IOContext` historically survived between epochs. That
        /// made buffer handles accidental temporal state. Recreate the context for
        /// every candidate so epoch-local buffers match the canonical bytecode VM;
        /// external resources remain governed by the effect gate.
        fn execute_fresh_epoch(
            &mut self,
            program: &Program,
            anamnesis: &Memory,
        ) -> crate::vm::EpochResult {
            self.executor.io_context = None;
            self.executor.run_epoch(program, anamnesis)
        }

        /// Execute one epoch against the cache: serve a verified cached result
        /// when one exists (this is how action-guided seeds share work once
        /// their orbits meet), otherwise run the epoch, freeze first inputs,
        /// and memoize.
        ///
        /// This sequence is the drift-prone core every search loop previously
        /// carried its own copy of; the missing-freeze bug fixed earlier in
        /// this programme existed precisely because the copies diverged.
        fn step_epoch(
            &mut self,
            program: &Program,
            anamnesis: &mut Memory,
            state_hash: u64,
            state: &ExactMemoryState,
            epoch: usize,
        ) -> Step {
            let cached = self
                .cache
                .get(state_hash, state)
                .filter(|c| c.status != EpochStatus::Running)
                .map(|c| (c.present.clone(), c.output.clone(), c.status.clone()));
            if let Some((present, output, status)) = cached {
                return Self::resolve(present, output, status, anamnesis, epoch);
            }

            let result = self.execute_fresh_epoch(program, anamnesis);
            self.freeze_inputs(&result);
            self.cache.insert(
                state_hash,
                state,
                MemoizedResult::simple(
                    result.present.clone(),
                    result.output.clone(),
                    result.status.clone(),
                ),
            );
            Self::resolve(
                result.present,
                result.output,
                result.status,
                anamnesis,
                epoch,
            )
        }

        /// Execute one epoch without touching the cache (pure mode, which has
        /// no revisit detection and therefore nothing to memoize against).
        fn step_epoch_uncached(
            &mut self,
            program: &Program,
            anamnesis: &mut Memory,
            epoch: usize,
        ) -> Step {
            let result = self.execute_fresh_epoch(program, anamnesis);
            self.freeze_inputs(&result);
            Self::resolve(
                result.present,
                result.output,
                result.status,
                anamnesis,
                epoch,
            )
        }

        /// Classify an epoch's result for the enclosing search loop: a fixed
        /// point or terminal status finishes the search; otherwise the present
        /// becomes the next anamnesis.
        fn resolve(
            present: Memory,
            output: Vec<OutputItem>,
            status: EpochStatus,
            anamnesis: &mut Memory,
            epoch: usize,
        ) -> Step {
            match status {
                EpochStatus::Finished => {
                    if present.numeric_values_equal(anamnesis) {
                        Step::Done(ConvergenceStatus::Consistent {
                            memory: present,
                            output,
                            epochs: epoch + 1,
                        })
                    } else {
                        *anamnesis = present;
                        Step::Continue(output)
                    }
                }
                EpochStatus::Paradox => Step::Done(ConvergenceStatus::Paradox {
                    message: "Explicit PARADOX instruction".to_string(),
                    epoch: epoch + 1,
                }),
                EpochStatus::Error(e) => Step::Done(ConvergenceStatus::Error {
                    message: e,
                    epoch: epoch + 1,
                }),
                EpochStatus::Running => Step::Continue(output),
            }
        }

        /// Run a trivially consistent program (no oracle operations).
        fn run_trivial(&mut self, program: &Program) -> ConvergenceStatus {
            let result =
                self.execute_fresh_epoch(program, &Memory::with_size(self.config.memory_cells));

            match result.status {
            EpochStatus::Finished => ConvergenceStatus::Consistent {
                memory: result.present,
                output: result.output,
                epochs: 1,
            },
            EpochStatus::Paradox => ConvergenceStatus::Paradox {
                message: "Explicit PARADOX in trivial program".to_string(),
                epoch: 1,
            },
            EpochStatus::Error(e) => ConvergenceStatus::Error {
                message: e,
                epoch: 1,
            },
            EpochStatus::Running => ConvergenceStatus::Error {
                message: "Epoch terminated in Running state: executor failed to produce a terminal status".to_string(),
                epoch: 1,
            },
        }
        }

        /// Standard execution: verified cycle detection, epoch caching, first
        /// fixed point wins.
        fn run_standard(&mut self, program: &Program) -> ConvergenceStatus {
            let mut anamnesis = self.create_initial_anamnesis();
            let mut seen_states = StateJournal::new();

            for epoch in 0..self.config.max_epochs {
                let state_hash = anamnesis.state_hash();
                let journal_state = anamnesis.sparse_state();
                let state = anamnesis.exact_sparse_state();

                if let Some(previous_epoch) =
                    seen_states.check_and_insert(state_hash, anamnesis.len(), &journal_state, epoch)
                {
                    let period = epoch - previous_epoch;
                    return self.diagnose_oscillation(program, &anamnesis, period);
                }

                match self.step_epoch(program, &mut anamnesis, state_hash, &state, epoch) {
                    Step::Done(status) => {
                        if self.config.verbose {
                            if let ConvergenceStatus::Consistent { .. } = status {
                                println!("{}", self.cache.stats());
                            }
                        }
                        return status;
                    }
                    Step::Continue(_) => {}
                }
            }

            ConvergenceStatus::Timeout {
                max_epochs: self.config.max_epochs,
            }
        }

        /// Deutsch execution for this deterministic implementation.
        ///
        /// The orbit from the configured seed eventually reaches a cycle because
        /// the concrete temporal memory is finite. If the period is one, this is
        /// the usual point fixed point. For a longer cycle, the uniform ensemble
        /// over its states is stationary. This finds one reachable recurrent
        /// class; it does not prove that every recurrent class has the same
        /// observable output.
        fn run_deutsch(&mut self, program: &Program) -> ConvergenceStatus {
            let mut anamnesis = self.create_initial_anamnesis();
            let mut seen_states = StateJournal::new();
            let mut trajectory: Vec<Memory> = Vec::new();
            let mut epoch_outputs: Vec<Vec<OutputItem>> = Vec::new();

            for epoch in 0..self.config.max_epochs {
                let state_hash = anamnesis.state_hash();
                let journal_state = anamnesis.sparse_state();
                let state = anamnesis.exact_sparse_state();

                if let Some(first_epoch) =
                    seen_states.check_and_insert(state_hash, anamnesis.len(), &journal_state, epoch)
                {
                    let cycle = trajectory[first_epoch..epoch].to_vec();
                    let outputs = epoch_outputs[first_epoch..epoch].to_vec();
                    let period = cycle.len();
                    let unanimous_output = outputs.first().and_then(|first| {
                        outputs
                            .iter()
                            .all(|candidate| Self::observable_outputs_equal(first, candidate))
                            .then(|| first.clone())
                    });
                    return ConvergenceStatus::DeutschConsistent {
                        cycle,
                        outputs,
                        period,
                        transient_epochs: first_epoch,
                        unanimous_output,
                    };
                }

                trajectory.push(anamnesis.clone());
                match self.step_epoch(program, &mut anamnesis, state_hash, &state, epoch) {
                    Step::Done(ConvergenceStatus::Consistent { memory, output, .. }) => {
                        return ConvergenceStatus::DeutschConsistent {
                            cycle: vec![memory],
                            outputs: vec![output.clone()],
                            period: 1,
                            transient_epochs: epoch,
                            unanimous_output: Some(output),
                        };
                    }
                    Step::Done(status) => return status,
                    Step::Continue(output) => epoch_outputs.push(output),
                }
            }

            ConvergenceStatus::Timeout {
                max_epochs: self.config.max_epochs,
            }
        }

        /// Compare public output while ignoring provenance analysis metadata.
        fn observable_outputs_equal(left: &[OutputItem], right: &[OutputItem]) -> bool {
            left.len() == right.len()
                && left.iter().zip(right).all(|(a, b)| match (a, b) {
                    (OutputItem::Val(a), OutputItem::Val(b)) => a.val == b.val,
                    (OutputItem::Char(a), OutputItem::Char(b)) => a == b,
                    _ => false,
                })
        }

        /// Diagnostic execution: the standard search plus trajectory recording,
        /// divergence detection, and oscillation diagnosis.
        fn run_diagnostic(&mut self, program: &Program) -> ConvergenceStatus {
            let mut anamnesis = self.create_initial_anamnesis();
            let mut trajectory: Vec<Memory> = Vec::new();
            let mut seen_states = StateJournal::new();

            for epoch in 0..self.config.max_epochs {
                let state_hash = anamnesis.state_hash();
                let journal_state = anamnesis.sparse_state();
                let state = anamnesis.exact_sparse_state();

                if let Some(previous_epoch) =
                    seen_states.check_and_insert(state_hash, anamnesis.len(), &journal_state, epoch)
                {
                    let period = epoch - previous_epoch;
                    let oscillating = self.find_oscillating_cells(&trajectory[previous_epoch..]);
                    let diagnosis =
                        self.create_oscillation_diagnosis(&trajectory[previous_epoch..]);

                    return ConvergenceStatus::Oscillation {
                        period,
                        oscillating_cells: oscillating,
                        diagnosis,
                    };
                }

                trajectory.push(anamnesis.clone());

                // Check for divergence (monotonic growth)
                if epoch > 10 {
                    if let Some((cells, direction)) = self.detect_divergence(&trajectory) {
                        return ConvergenceStatus::Divergence {
                            diverging_cells: cells,
                            direction,
                        };
                    }
                }

                match self.step_epoch(program, &mut anamnesis, state_hash, &state, epoch) {
                    Step::Done(status) => return status,
                    Step::Continue(_) => {}
                }
            }

            ConvergenceStatus::Timeout {
                max_epochs: self.config.max_epochs,
            }
        }

        /// Pure execution without revisit detection, so no journal and no cache;
        /// the concrete implementation retains a high interactive safety ceiling.
        fn run_pure(&mut self, program: &Program) -> ConvergenceStatus {
            let mut anamnesis = self.create_initial_anamnesis();

            for epoch in 0.. {
                match self.step_epoch_uncached(program, &mut anamnesis, epoch) {
                    Step::Done(status) => return status,
                    Step::Continue(_) => {}
                }

                // Safety check for interactive use
                if epoch >= 1_000_000 {
                    return ConvergenceStatus::Timeout {
                        max_epochs: epoch + 1,
                    };
                }
            }
            unreachable!("the epoch loop above only exits by returning")
        }

        /// Action-guided execution: explore multiple seeds and select the best fixed point.
        ///
        /// This implements the Action Principle to solve the "Genie Effect". Instead of
        /// accepting the first fixed point found, we explore multiple starting conditions
        /// and select the solution with minimum action (cost).
        fn run_action_guided(
            &mut self,
            program: &Program,
            action_config: ActionConfig,
            num_seeds: usize,
        ) -> ConvergenceStatus {
            let principle = ActionPrinciple::new(action_config);
            let mut selector = FixedPointSelector::new(principle);

            // Generate diverse seeds
            let seeds = self.generate_diverse_seeds(num_seeds);

            // Explore each seed
            for seed in seeds {
                let result = self.run_with_seed(program, &seed);

                if let ConvergenceStatus::Consistent {
                    memory,
                    output,
                    epochs,
                } = result
                {
                    selector.add_candidate(memory, epochs, output, seed);
                }
            }

            // Get count before consuming selector
            let candidate_count = selector.candidate_count();

            // Select the best candidate
            if let Some(best) = selector.select_best() {
                if self.config.verbose {
                    println!("Selected fixed point with action: {:.4}", best.action);
                    println!("Explored {} candidate(s)", candidate_count);
                }

                ConvergenceStatus::Consistent {
                    memory: best.memory,
                    output: best.output,
                    epochs: best.epochs,
                }
            } else {
                // No consistent fixed point found with any seed
                // Fall back to standard execution to get proper error
                self.run_standard(program)
            }
        }

        /// Run with a specific seed memory state. Uses the shared cache, so
        /// seeds whose orbits meet reuse each other's epochs instead of
        /// re-executing them.
        fn run_with_seed(&mut self, program: &Program, seed: &Memory) -> ConvergenceStatus {
            let mut anamnesis = seed.clone();
            let mut seen_states = StateJournal::new();

            for epoch in 0..self.config.max_epochs {
                let state_hash = anamnesis.state_hash();
                let journal_state = anamnesis.sparse_state();
                let state = anamnesis.exact_sparse_state();

                // A verified cycle means this seed does not lead to a fixed point
                if seen_states
                    .check_and_insert(state_hash, anamnesis.len(), &journal_state, epoch)
                    .is_some()
                {
                    return ConvergenceStatus::Timeout { max_epochs: epoch };
                }

                match self.step_epoch(program, &mut anamnesis, state_hash, &state, epoch) {
                    Step::Done(status) => return status,
                    Step::Continue(_) => {}
                }
            }

            ConvergenceStatus::Timeout {
                max_epochs: self.config.max_epochs,
            }
        }

        /// Generate diverse seed memory states for action-guided search.
        fn generate_diverse_seeds(&self, num_seeds: usize) -> Vec<Memory> {
            let mut seeds = Vec::with_capacity(num_seeds);

            // Seed 0: All zeros (standard)
            seeds.push(Memory::with_size(self.config.memory_cells));

            if num_seeds <= 1 {
                return seeds;
            }

            // Seed 1: Small primes (good for factorization)
            let mut mem = Memory::with_size(self.config.memory_cells);
            let primes = [
                2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
            ];
            for (i, &p) in primes.iter().take(self.config.memory_cells).enumerate() {
                mem.write(i as Address, Value::new(p));
            }
            seeds.push(mem);

            if num_seeds <= 2 {
                return seeds;
            }

            // Seed 2: Sequential values 1..16 (good for permutation problems)
            let mut mem = Memory::with_size(self.config.memory_cells);
            for i in 0..16.min(self.config.memory_cells) {
                mem.write(i as Address, Value::new(i as u64 + 1));
            }
            seeds.push(mem);

            if num_seeds <= 3 {
                return seeds;
            }

            // Seed 3: Powers of 2 (good for bit manipulation)
            let mut mem = Memory::with_size(self.config.memory_cells);
            for i in 0..16.min(self.config.memory_cells) {
                mem.write(i as Address, Value::new(1u64 << i));
            }
            seeds.push(mem);

            // Additional seeds: Random-ish values based on config seed
            for seed_idx in 4..num_seeds {
                let mut mem = Memory::with_size(self.config.memory_cells);
                let base = self.config.seed.wrapping_add(seed_idx as u64 * 12345);
                for i in 0..16.min(self.config.memory_cells) {
                    let val = base
                        .wrapping_mul(i as u64 + 1)
                        .wrapping_add(seed_idx as u64);
                    mem.write(i as Address, Value::new(val % 1000)); // Keep values reasonable
                }
                seeds.push(mem);
            }

            seeds
        }

        /// Create initial anamnesis based on seed.
        fn create_initial_anamnesis(&self) -> Memory {
            let mut mem = Memory::with_size(self.config.memory_cells);

            if self.config.seed != 0 {
                // Seed first few cells for variety
                for i in 0..16.min(self.config.memory_cells) {
                    let val = self.config.seed.wrapping_mul(i as u64 + 1);
                    mem.write(i as Address, Value::new(val));
                }
            }

            mem
        }

        /// Diagnose oscillation and identify oscillating cells.
        fn diagnose_oscillation(
            &mut self,
            program: &Program,
            anamnesis: &Memory,
            period: usize,
        ) -> ConvergenceStatus {
            // Re-run to capture the cycle
            let mut states = Vec::new();
            let mut current = anamnesis.clone();

            for _ in 0..period + 1 {
                states.push(current.clone());
                let result = self.executor.run_epoch(program, &current);
                current = result.present;
            }

            let oscillating = self.find_oscillating_cells(&states);
            let diagnosis = self.create_oscillation_diagnosis(&states);

            ConvergenceStatus::Oscillation {
                period,
                oscillating_cells: oscillating,
                diagnosis,
            }
        }

        /// Find cells that change within a cycle.
        /// Uses Memory::diff() to scan all 65,536 cells rather than just the first 256.
        fn find_oscillating_cells(&self, states: &[Memory]) -> Vec<Address> {
            if states.len() < 2 {
                return Vec::new();
            }

            let mut oscillating = std::collections::HashSet::new();
            for window in states.windows(2) {
                oscillating.extend(window[0].diff(&window[1]));
            }
            let mut result: Vec<_> = oscillating.into_iter().collect();
            result.sort();
            result
        }

        /// Create a diagnosis of the oscillation.
        fn create_oscillation_diagnosis(&self, states: &[Memory]) -> ParadoxDiagnosis {
            if states.len() == 2 {
                // Period-2 oscillation: likely grandfather paradox
                let diffs = states[0].diff(&states[1]);

                if diffs.len() == 1 {
                    let addr = diffs[0];
                    let v1 = states[0].read(addr).val;
                    let v2 = states[1].read(addr).val;

                    // Check for negation pattern
                    if v1 == !v2 || (v1 == 0 && v2 != 0) || (v1 != 0 && v2 == 0) {
                        return ParadoxDiagnosis::NegativeLoop {
                            cells: vec![addr],
                            explanation: format!(
                                "Cell {} oscillates between {} and {}. \
                             This is a grandfather paradox: the cell's value \
                             determines its own opposite.",
                                addr, v1, v2
                            ),
                        };
                    }
                }
            }

            // General oscillation
            let cycle: Vec<Vec<(Address, u64)>> = states
                .iter()
                .map(|s| {
                    s.non_zero_cells()
                        .iter()
                        .map(|(a, v)| (*a, v.val))
                        .collect()
                })
                .collect();

            ParadoxDiagnosis::Oscillation { cycle }
        }

        /// Detect a monotonic trend in the recorded finite prefix. With wrapping
        /// u64 arithmetic this is diagnostic evidence, not a proof of unboundedness.
        /// Scans all addresses that changed across the trajectory, not just the first 256.
        fn detect_divergence(&self, trajectory: &[Memory]) -> Option<(Vec<Address>, Direction)> {
            if trajectory.len() < 5 {
                return None;
            }

            // Find addresses that changed across the trajectory
            let mut changed = std::collections::HashSet::new();
            for window in trajectory.windows(2) {
                changed.extend(window[0].diff(&window[1]));
            }

            let mut diverging = Vec::new();
            let mut direction = Direction::Increasing;

            // Check monotonicity only for addresses that actually changed
            for &addr in &changed {
                let values: Vec<u64> = trajectory.iter().map(|m| m.read(addr).val).collect();

                let increasing = values.windows(2).all(|w| w[1] > w[0]);
                let decreasing = values.windows(2).all(|w| w[1] < w[0]);

                if increasing {
                    diverging.push(addr);
                    direction = Direction::Increasing;
                } else if decreasing {
                    diverging.push(addr);
                    direction = Direction::Decreasing;
                }
            }

            if diverging.is_empty() {
                None
            } else {
                Some((diverging, direction))
            }
        }
    }
}

/// Format convergence status for display.
pub fn format_status(status: &ConvergenceStatus) -> String {
    match status {
        ConvergenceStatus::Consistent { epochs, output, .. } => {
            let mut s = format!("CONSISTENT after {} epoch(s)\n", epochs);
            if !output.is_empty() {
                let formatted: Vec<String> = output
                    .iter()
                    .map(|item| match item {
                        OutputItem::Val(v) => v.val.to_string(),
                        OutputItem::Char(c) => (*c as char).to_string(),
                    })
                    .collect();
                s.push_str(&format!("Output: {:?}\n", formatted));
            }
            s
        }

        ConvergenceStatus::DeutschConsistent {
            period,
            transient_epochs,
            unanimous_output,
            ..
        } => {
            let readout = if unanimous_output.is_some() {
                "unanimous"
            } else {
                "ambiguous"
            };
            format!(
                "DEUTSCH-CONSISTENT stationary cycle (period {}, transient {}, readout {})\n",
                period, transient_epochs, readout
            )
        }

        ConvergenceStatus::Paradox { message, epoch } => {
            format!("PARADOX at epoch {}: {}\n", epoch, message)
        }

        ConvergenceStatus::Oscillation {
            period,
            oscillating_cells,
            diagnosis,
        } => {
            let mut s = format!("OSCILLATION detected (period {})\n", period);
            s.push_str(&format!("Oscillating cells: {:?}\n", oscillating_cells));

            match diagnosis {
                ParadoxDiagnosis::NegativeLoop { explanation, .. } => {
                    s.push_str(&format!(
                        "\nDIAGNOSIS (Grandfather Paradox):\n{}\n",
                        explanation
                    ));
                }
                ParadoxDiagnosis::Oscillation { cycle } => {
                    s.push_str("\nCycle states:\n");
                    for (i, state) in cycle.iter().enumerate() {
                        if !state.is_empty() {
                            s.push_str(&format!("  State {}: {:?}\n", i, state));
                        }
                    }
                }
                ParadoxDiagnosis::Unknown => {
                    s.push_str("\nDIAGNOSIS: Unknown cause\n");
                }
            }
            s
        }

        ConvergenceStatus::Divergence {
            diverging_cells,
            direction,
        } => {
            format!(
                "DIVERGENCE detected\n\
                    Diverging cells: {:?}\n\
                    Direction: {:?}\n\
                    \n\
                    DIAGNOSIS: Cell value(s) are monotone on the recorded prefix.\n\
                    This heuristic does not prove mathematical divergence.\n",
                diverging_cells, direction
            )
        }

        ConvergenceStatus::Timeout { max_epochs } => {
            format!("TIMEOUT after {} epochs\n", max_epochs)
        }

        ConvergenceStatus::Error { message, epoch } => {
            format!("ERROR at epoch {}: {}\n", epoch, message)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn run_program(source: &str) -> ConvergenceStatus {
        let program = parse(source).expect("Parse failed");
        let config = TimeLoopConfig::default();
        let mut driver = TimeLoop::new(config).expect("valid configuration");
        driver.run(&program)
    }

    #[test]
    fn test_trivial_consistency() {
        let status = run_program("10 20 ADD OUTPUT");
        assert!(matches!(
            status,
            ConvergenceStatus::Consistent { epochs: 1, .. }
        ));
    }

    #[test]
    fn test_self_fulfilling() {
        // Oracle reads 0, writes 0 → consistent immediately
        let status = run_program("0 ORACLE 0 PROPHECY");
        assert!(matches!(status, ConvergenceStatus::Consistent { .. }));
    }

    #[test]
    fn test_grandfather_paradox() {
        // Reads A[0], writes NOT(A[0]) → oscillates
        let status = run_program("0 ORACLE NOT 0 PROPHECY");
        assert!(matches!(
            status,
            ConvergenceStatus::Oscillation { period: 2, .. }
        ));
    }

    #[test]
    fn test_divergence() {
        // Reads A[0], writes A[0]+1 → diverges
        let status = run_program("0 ORACLE 1 ADD 0 PROPHECY");

        // Either detected as divergence or timeout
        assert!(matches!(
            status,
            ConvergenceStatus::Divergence { .. } | ConvergenceStatus::Timeout { .. }
        ));
    }

    #[test]
    fn test_oscillation_detected_above_256() {
        // Programme that oscillates at address 500 (above the previous 256-cell limit).
        // Reads A[500], writes NOT(A[500]) at address 500.
        let status = run_program("500 ORACLE NOT 500 PROPHECY");
        match &status {
            ConvergenceStatus::Oscillation {
                oscillating_cells, ..
            } => {
                assert!(
                    oscillating_cells.contains(&500),
                    "Address 500 should be in oscillating cells, got {:?}",
                    oscillating_cells
                );
            }
            _ => panic!("Expected Oscillation, got {:?}", status),
        }
    }

    #[test]
    fn public_facade_executes_procedures_and_quotations_as_bytecode() {
        let status =
            run_program("PROCEDURE choose { 0 ORACLE DUP [ DUP OUTPUT ] EXEC 0 PROPHECY } choose");
        match status {
            ConvergenceStatus::Consistent { output, epochs, .. } => {
                assert_eq!(epochs, 1);
                assert!(matches!(&output[..], [OutputItem::Val(value)] if value.val == 0));
            }
            other => panic!("expected quoted procedure to converge, got {other:?}"),
        }
    }

    #[test]
    fn public_facade_bounds_recursive_procedures_in_bytecode_frames() {
        let status = run_program("PROCEDURE recurse { recurse } recurse");
        assert!(matches!(
            status,
            ConvergenceStatus::Error { message, epoch: 1 }
                if message.contains("bytecode call depth limit")
        ));
    }

    #[test]
    fn public_facade_never_grants_sleep_capability_implicitly() {
        let status = run_program("0 SLEEP");
        assert!(matches!(
            status,
            ConvergenceStatus::Error { message, epoch: 1 }
                if message.contains("sleep capability denied")
        ));
    }

    #[test]
    fn test_only_ast_oracle_agrees_on_a_temporal_cycle() {
        let program = parse("0 ORACLE NOT 0 PROPHECY").expect("Parse failed");
        let canonical = TimeLoop::new(TimeLoopConfig::default())
            .expect("valid configuration")
            .run(&program);
        let mut legacy_driver = super::legacy_ast_oracle::TimeLoop::new(TimeLoopConfig::default())
            .expect("valid configuration");
        let legacy = legacy_driver.run(&program);
        let _differential_cache_evidence = legacy_driver.cache_stats();
        assert!(matches!(
            (canonical, legacy),
            (
                ConvergenceStatus::Oscillation { period: 2, .. },
                ConvergenceStatus::Oscillation { period: 2, .. }
            )
        ));
    }
}
