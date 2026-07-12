//! Fixed-point computation and temporal loop execution.
//!
//! This module implements the core OUROCHRONOS execution model:
//! repeatedly run epochs until Present = Anamnesis (fixed point achieved).
//!
//! The module also provides paradox diagnosis when no fixed point exists.
//!
//! ## Action-Guided Execution
//!
//! The `ActionGuided` mode implements the **Action Principle** to solve the "Genie Effect".
//! Instead of accepting the first fixed point found, it explores multiple seed strategies
//! and selects the solution with minimum action (cost), preferring non-trivial,
//! output-producing solutions.

use crate::core::{Memory, Address, Value, OutputItem};
use crate::core::error::ErrorConfig;
use crate::ast::Program;
use crate::vm::{Executor, ExecutorConfig, EpochStatus};
use crate::temporal::action::{ActionConfig, ActionPrinciple, FixedPointSelector};
use crate::temporal::cache::{EpochCache, MemoizedResult};
use std::collections::HashMap;

/// Journal of anamnesis states visited during a fixed-point search.
///
/// States are keyed by `Memory::state_hash`, a 64-bit XOR fold, so distinct
/// states can collide; declaring an oscillation on a bare hash match would
/// fabricate a paradox. A hash hit therefore only counts as a revisit after
/// the stored sparse state (the non-zero cells, which determine the state
/// completely because absent cells are zero) compares equal.
struct StateJournal {
    visits: HashMap<u64, Vec<JournalEntry>>,
}

/// One verified visit: the epoch it occurred at and the sparse (address,
/// value) form of the memory state.
type JournalEntry = (usize, Vec<(u16, u64)>);

impl StateJournal {
    fn new() -> Self {
        Self { visits: HashMap::new() }
    }

    /// Record the state (given as its sparse form, computed once per epoch
    /// by the caller) as visited at `epoch`, or return the epoch of the
    /// earlier verified-identical visit.
    fn check_and_insert(&mut self, hash: u64, sparse: &[(u16, u64)], epoch: usize) -> Option<usize> {
        let entry = self.visits.entry(hash).or_default();
        if let Some((previous, _)) = entry.iter().find(|(_, state)| state.as_slice() == sparse) {
            return Some(*previous);
        }
        entry.push((epoch, sparse.to_vec()));
        None
    }
}

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
    
    /// Divergence detected (monotonic unbounded growth).
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
    Error {
        message: String,
        epoch: usize,
    },
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
    /// Unbounded iteration (may not terminate).
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
        if self.provenance_limit == 0 {
            errors.push("provenance_limit must be > 0".to_string());
        }
        if let ExecutionMode::ActionGuided { num_seeds, .. } = &self.mode {
            if *num_seeds == 0 {
                errors.push("num_seeds must be > 0 in ActionGuided mode".to_string());
            }
        }

        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }
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
        config.validate()
            .map_err(|errors| crate::core::error::OuroError::InvalidConfiguration { errors })?;
        crate::core::provenance::set_saturation_limit(config.provenance_limit);

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

    /// Run a trivially consistent program (no oracle operations).
    fn run_trivial(&mut self, program: &Program) -> ConvergenceStatus {
        let result = self.executor.run_epoch(program, &Memory::new());
        
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
    
    /// Standard execution with cycle detection.
    /// 
    /// Implements the Temporal Input Invariant: inputs are captured during epoch 0
    /// and frozen for replay in all subsequent epochs. This ensures that external
    /// inputs cannot be influenced by the temporal loop (External → Loop only).
    /// 
    /// Uses epoch caching to memoize results and skip redundant executions.
    fn run_standard(&mut self, program: &Program) -> ConvergenceStatus {
        let mut anamnesis = self.create_initial_anamnesis();
        let mut seen_states = StateJournal::new();

        for epoch in 0..self.config.max_epochs {
            let state_hash = anamnesis.state_hash();
            let sparse = anamnesis.sparse_state();

            if let Some(previous_epoch) = seen_states.check_and_insert(state_hash, &sparse, epoch) {
                let period = epoch - previous_epoch;
                return self.diagnose_oscillation(program, &anamnesis, period);
            }

            // Check cache first
            if let Some(cached) = self.cache.get(state_hash, &sparse) {
                // Use cached result
                match &cached.status {
                    EpochStatus::Finished => {
                        if cached.present.values_equal(&anamnesis) {
                            return ConvergenceStatus::Consistent {
                                memory: cached.present.clone(),
                                output: cached.output.clone(),
                                epochs: epoch + 1,
                            };
                        }
                        anamnesis = cached.present.clone();
                        continue;
                    }
                    EpochStatus::Paradox => {
                        return ConvergenceStatus::Paradox {
                            message: "Explicit PARADOX instruction (cached)".to_string(),
                            epoch: epoch + 1,
                        };
                    }
                    _ => {} // Run epoch for other statuses
                }
            }
            
            // Run epoch (cache miss)
            let result = self.executor.run_epoch(program, &anamnesis);
            self.freeze_inputs(&result);

            // Store in cache
            self.cache.insert(state_hash, &sparse, MemoizedResult::simple(
                result.present.clone(),
                result.output.clone(),
                result.status.clone(),
            ));

            match result.status {
                EpochStatus::Finished => {
                    // Check for fixed point
                    if result.present.values_equal(&anamnesis) {
                        if self.config.verbose {
                            println!("{}", self.cache.stats());
                        }
                        return ConvergenceStatus::Consistent {
                            memory: result.present,
                            output: result.output,
                            epochs: epoch + 1,
                        };
                    }

                    // Continue iteration
                    anamnesis = result.present;
                }

                EpochStatus::Paradox => {
                    return ConvergenceStatus::Paradox {
                        message: "Explicit PARADOX instruction".to_string(),
                        epoch: epoch + 1,
                    };
                }

                EpochStatus::Error(e) => {
                    return ConvergenceStatus::Error {
                        message: e,
                        epoch: epoch + 1,
                    };
                }

                EpochStatus::Running => {}
            }
        }

        ConvergenceStatus::Timeout {
            max_epochs: self.config.max_epochs,
        }
    }

    /// Diagnostic execution with full trajectory recording.
    fn run_diagnostic(&mut self, program: &Program) -> ConvergenceStatus {
        let mut anamnesis = self.create_initial_anamnesis();
        let mut trajectory: Vec<Memory> = Vec::new();
        let mut seen_states = StateJournal::new();

        for epoch in 0..self.config.max_epochs {
            let state_hash = anamnesis.state_hash();
            let sparse = anamnesis.sparse_state();

            if let Some(previous_epoch) = seen_states.check_and_insert(state_hash, &sparse, epoch) {
                let period = epoch - previous_epoch;
                let oscillating = self.find_oscillating_cells(&trajectory[previous_epoch..]);
                let diagnosis = self.create_oscillation_diagnosis(&trajectory[previous_epoch..]);

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
            
            // Run epoch
            let result = self.executor.run_epoch(program, &anamnesis);
            self.freeze_inputs(&result);

            match result.status {
                EpochStatus::Finished => {
                    if result.present.values_equal(&anamnesis) {
                        return ConvergenceStatus::Consistent {
                            memory: result.present,
                            output: result.output,
                            epochs: epoch + 1,
                        };
                    }
                    anamnesis = result.present;
                }

                EpochStatus::Paradox => {
                    return ConvergenceStatus::Paradox {
                        message: "Explicit PARADOX instruction".to_string(),
                        epoch: epoch + 1,
                    };
                }

                EpochStatus::Error(e) => {
                    return ConvergenceStatus::Error {
                        message: e,
                        epoch: epoch + 1,
                    };
                }

                EpochStatus::Running => {}
            }
        }

        ConvergenceStatus::Timeout {
            max_epochs: self.config.max_epochs,
        }
    }

    /// Pure execution (unbounded, for theoretical exploration).
    fn run_pure(&mut self, program: &Program) -> ConvergenceStatus {
        let mut anamnesis = self.create_initial_anamnesis();
        let mut epoch = 0;

        loop {
            let result = self.executor.run_epoch(program, &anamnesis);
            self.freeze_inputs(&result);
            epoch += 1;
            
            match result.status {
                EpochStatus::Finished => {
                    if result.present.values_equal(&anamnesis) {
                        return ConvergenceStatus::Consistent {
                            memory: result.present,
                            output: result.output,
                            epochs: epoch,
                        };
                    }
                    anamnesis = result.present;
                }
                
                EpochStatus::Paradox => {
                    return ConvergenceStatus::Paradox {
                        message: "Explicit PARADOX instruction".to_string(),
                        epoch,
                    };
                }
                
                EpochStatus::Error(e) => {
                    return ConvergenceStatus::Error {
                        message: e,
                        epoch,
                    };
                }

                EpochStatus::Running => {}
            }

            // Safety check for interactive use
            if epoch > 1_000_000 {
                return ConvergenceStatus::Timeout { max_epochs: epoch };
            }
        }
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
            
            if let ConvergenceStatus::Consistent { memory, output, epochs } = result {
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
    
    /// Run with a specific seed memory state.
    fn run_with_seed(&mut self, program: &Program, seed: &Memory) -> ConvergenceStatus {
        let mut anamnesis = seed.clone();
        let mut seen_states = StateJournal::new();

        for epoch in 0..self.config.max_epochs {
            let state_hash = anamnesis.state_hash();
            let sparse = anamnesis.sparse_state();

            // A verified cycle means this seed does not lead to a fixed point
            if seen_states.check_and_insert(state_hash, &sparse, epoch).is_some() {
                return ConvergenceStatus::Timeout { max_epochs: epoch };
            }

            // Run epoch
            let result = self.executor.run_epoch(program, &anamnesis);
            self.freeze_inputs(&result);

            match result.status {
                EpochStatus::Finished => {
                    // Check for fixed point
                    if result.present.values_equal(&anamnesis) {
                        return ConvergenceStatus::Consistent {
                            memory: result.present,
                            output: result.output,
                            epochs: epoch + 1,
                        };
                    }
                    anamnesis = result.present;
                }
                EpochStatus::Paradox => {
                    return ConvergenceStatus::Paradox {
                        message: "Explicit PARADOX instruction".to_string(),
                        epoch: epoch + 1,
                    };
                }
                EpochStatus::Error(e) => {
                    return ConvergenceStatus::Error {
                        message: e,
                        epoch: epoch + 1,
                    };
                }
                EpochStatus::Running => {}
            }
        }

        ConvergenceStatus::Timeout { max_epochs: self.config.max_epochs }
    }
    
    /// Generate diverse seed memory states for action-guided search.
    fn generate_diverse_seeds(&self, num_seeds: usize) -> Vec<Memory> {
        let mut seeds = Vec::with_capacity(num_seeds);
        
        // Seed 0: All zeros (standard)
        seeds.push(Memory::new());
        
        if num_seeds <= 1 {
            return seeds;
        }
        
        // Seed 1: Small primes (good for factorization)
        let mut mem = Memory::new();
        let primes = [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];
        for (i, &p) in primes.iter().enumerate() {
            mem.write(i as Address, Value::new(p));
        }
        seeds.push(mem);
        
        if num_seeds <= 2 {
            return seeds;
        }
        
        // Seed 2: Sequential values 1..16 (good for permutation problems)
        let mut mem = Memory::new();
        for i in 0..16 {
            mem.write(i as Address, Value::new(i as u64 + 1));
        }
        seeds.push(mem);
        
        if num_seeds <= 3 {
            return seeds;
        }
        
        // Seed 3: Powers of 2 (good for bit manipulation)
        let mut mem = Memory::new();
        for i in 0..16 {
            mem.write(i as Address, Value::new(1u64 << i));
        }
        seeds.push(mem);
        
        // Additional seeds: Random-ish values based on config seed
        for seed_idx in 4..num_seeds {
            let mut mem = Memory::new();
            let base = self.config.seed.wrapping_add(seed_idx as u64 * 12345);
            for i in 0..16 {
                let val = base.wrapping_mul(i as u64 + 1).wrapping_add(seed_idx as u64);
                mem.write(i as Address, Value::new(val % 1000)); // Keep values reasonable
            }
            seeds.push(mem);
        }
        
        seeds
    }
    
    /// Create initial anamnesis based on seed.
    fn create_initial_anamnesis(&self) -> Memory {
        let mut mem = Memory::new();
        
        if self.config.seed != 0 {
            // Seed first few cells for variety
            for i in 0..16 {
                let val = self.config.seed.wrapping_mul(i as u64 + 1);
                mem.write(i as Address, Value::new(val));
            }
        }
        
        mem
    }
    
    /// Diagnose oscillation and identify oscillating cells.
    fn diagnose_oscillation(&mut self, program: &Program, 
                           anamnesis: &Memory, period: usize) -> ConvergenceStatus {
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
        let cycle: Vec<Vec<(Address, u64)>> = states.iter()
            .map(|s| s.non_zero_cells().iter()
                .map(|(a, v)| (*a, v.val))
                .collect())
            .collect();
        
        ParadoxDiagnosis::Oscillation { cycle }
    }
    
    /// Detect divergence (monotonic unbounded growth).
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
            let values: Vec<u64> = trajectory.iter()
                .map(|m| m.read(addr).val)
                .collect();

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

/// Format convergence status for display.
pub fn format_status(status: &ConvergenceStatus) -> String {
    match status {
        ConvergenceStatus::Consistent { epochs, output, .. } => {
            let mut s = format!("CONSISTENT after {} epoch(s)\n", epochs);
            if !output.is_empty() {
                let formatted: Vec<String> = output.iter().map(|item| match item {
                    OutputItem::Val(v) => v.val.to_string(),
                    OutputItem::Char(c) => (*c as char).to_string(),
                }).collect();
                s.push_str(&format!("Output: {:?}\n", formatted));
            }
            s
        }
        
        ConvergenceStatus::Paradox { message, epoch } => {
            format!("PARADOX at epoch {}: {}\n", epoch, message)
        }
        
        ConvergenceStatus::Oscillation { period, oscillating_cells, diagnosis } => {
            let mut s = format!("OSCILLATION detected (period {})\n", period);
            s.push_str(&format!("Oscillating cells: {:?}\n", oscillating_cells));
            
            match diagnosis {
                ParadoxDiagnosis::NegativeLoop { explanation, .. } => {
                    s.push_str(&format!("\nDIAGNOSIS (Grandfather Paradox):\n{}\n", explanation));
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
        
        ConvergenceStatus::Divergence { diverging_cells, direction } => {
            format!("DIVERGENCE detected\n\
                    Diverging cells: {:?}\n\
                    Direction: {:?}\n\
                    \n\
                    DIAGNOSIS: Cell value(s) grow without bound.\n\
                    No fixed point is reachable.\n",
                    diverging_cells, direction)
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
        assert!(matches!(status, ConvergenceStatus::Consistent { epochs: 1, .. }));
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
        assert!(matches!(status, ConvergenceStatus::Oscillation { period: 2, .. }));
    }
    
    #[test]
    fn test_divergence() {
        // Reads A[0], writes A[0]+1 → diverges
        let status = run_program("0 ORACLE 1 ADD 0 PROPHECY");

        // Either detected as divergence or timeout
        assert!(matches!(status,
            ConvergenceStatus::Divergence { .. } |
            ConvergenceStatus::Timeout { .. }
        ));
    }

    #[test]
    fn test_oscillation_detected_above_256() {
        // Programme that oscillates at address 500 (above the previous 256-cell limit).
        // Reads A[500], writes NOT(A[500]) at address 500.
        let status = run_program("500 ORACLE NOT 500 PROPHECY");
        match &status {
            ConvergenceStatus::Oscillation { oscillating_cells, .. } => {
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
    fn test_find_oscillating_cells_uses_diff() {
        // Verify that find_oscillating_cells detects changes at high addresses.
        let mut m1 = Memory::new();
        let mut m2 = Memory::new();
        m1.write(1000, crate::core::Value::new(42));
        m2.write(1000, crate::core::Value::new(99));

        let config = TimeLoopConfig::default();
        let tl = TimeLoop::new(config).expect("valid configuration");
        let cells = tl.find_oscillating_cells(&[m1, m2]);
        assert!(cells.contains(&1000), "Should detect oscillation at address 1000");
    }

    #[test]
    fn journal_reports_revisit_only_for_identical_state() {
        let mut journal = StateJournal::new();
        let mut a = Memory::new();
        a.write(3, Value::new(5));
        let mut b = Memory::new();
        b.write(3, Value::new(6));
        let (a, b) = (a.sparse_state(), b.sparse_state());

        // Same hash key, different contents: not a revisit.
        assert_eq!(journal.check_and_insert(42, &a, 0), None);
        assert_eq!(journal.check_and_insert(42, &b, 1), None);

        // Identical contents under the same hash: verified revisit at its
        // original epoch.
        assert_eq!(journal.check_and_insert(42, &a, 2), Some(0));
        assert_eq!(journal.check_and_insert(42, &b, 3), Some(1));
    }
}
