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

use crate::core_types::{Memory, Address, Value, OutputItem};
use crate::ast::Program;
use crate::vm::{Executor, ExecutorConfig, EpochStatus};
use crate::action::{ActionConfig, ActionPrinciple, FixedPointSelector};
use crate::memo::{EpochCache, MemoizedResult};
use std::collections::HashMap;

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
    /// Parallel exploration of multiple seeds simultaneously.
    /// Requires the `parallel` feature.
    Parallel {
        /// Number of parallel workers (0 = auto-detect).
        num_workers: usize,
        /// Number of seeds to explore.
        num_seeds: usize,
    },
}

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
}

impl Default for TimeLoopConfig {
    fn default() -> Self {
        Self {
            max_epochs: 10_000,
            mode: ExecutionMode::Standard,
            seed: 0,
            verbose: false,
            frozen_inputs: Vec::new(),
            max_instructions: 10_000_000,
        }
    }
}

/// The temporal loop driver.
pub struct TimeLoop {
    config: TimeLoopConfig,
    executor: Executor,
    /// Inputs captured during epoch 0, frozen for replay in subsequent epochs.
    /// This ensures the Temporal Input Invariant: External → Loop causality only.
    captured_inputs: Option<Vec<u64>>,
    /// Epoch result cache for memoization during fixed-point search.
    cache: EpochCache,
}

impl TimeLoop {
    /// Create a new time loop with given configuration.
    pub fn new(config: TimeLoopConfig) -> Self {
        let mut exec_config = ExecutorConfig::default();
        exec_config.immediate_output = config.verbose;
        exec_config.max_instructions = config.max_instructions;

        // Use frozen inputs if provided
        if !config.frozen_inputs.is_empty() {
            exec_config.input = config.frozen_inputs.clone();
        }

        Self {
            config,
            executor: Executor::with_config(exec_config),
            captured_inputs: None,
            cache: EpochCache::with_capacity(1024),
        }
    }
    
    /// Get cache statistics.
    pub fn cache_stats(&self) -> crate::memo::CacheStats {
        self.cache.stats()
    }
    
    /// Run the fixed-point search.
    pub fn run(&mut self, program: &Program) -> ConvergenceStatus {
        // Check for trivial consistency
        if program.is_trivially_consistent() {
            return self.run_trivial(program);
        }
        
        match &self.config.mode {
            ExecutionMode::Standard => self.run_standard(program),
            ExecutionMode::Diagnostic => self.run_diagnostic(program),
            ExecutionMode::Pure => self.run_pure(program),
            ExecutionMode::ActionGuided { config, num_seeds } => {
                self.run_action_guided(program, config.clone(), *num_seeds)
            }
            ExecutionMode::Parallel { num_workers, num_seeds } => {
                self.run_parallel(program, *num_workers, *num_seeds)
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
            _ => unreachable!(),
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
        let mut seen_states: HashMap<u64, usize> = HashMap::new();
        
        for epoch in 0..self.config.max_epochs {
            let state_hash = anamnesis.state_hash();
            
            // Check for cycle
            if let Some(&previous_epoch) = seen_states.get(&state_hash) {
                let period = epoch - previous_epoch;
                return self.diagnose_oscillation(program, &anamnesis, period);
            }
            
            seen_states.insert(state_hash, epoch);
            
            // Check cache first
            if let Some(cached) = self.cache.get(state_hash) {
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
            
            // On epoch 0, capture inputs and freeze them for subsequent epochs
            if epoch == 0 && !result.inputs_consumed.is_empty() && self.captured_inputs.is_none() {
                self.captured_inputs = Some(result.inputs_consumed.clone());
                self.executor.config.input = result.inputs_consumed;
                if self.config.verbose {
                    println!("Input frozen: {:?}", self.captured_inputs.as_ref().unwrap());
                }
            }
            
            // Store in cache
            self.cache.insert(state_hash, MemoizedResult::simple(
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
                
                _ => {}
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
        let mut seen_states: HashMap<u64, usize> = HashMap::new();
        
        for epoch in 0..self.config.max_epochs {
            let state_hash = anamnesis.state_hash();
            
            // Check for cycle
            if let Some(&previous_epoch) = seen_states.get(&state_hash) {
                let period = epoch - previous_epoch;
                let oscillating = self.find_oscillating_cells(&trajectory[previous_epoch..]);
                let diagnosis = self.create_oscillation_diagnosis(&trajectory[previous_epoch..]);
                
                return ConvergenceStatus::Oscillation {
                    period,
                    oscillating_cells: oscillating,
                    diagnosis,
                };
            }
            
            seen_states.insert(state_hash, epoch);
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
                
                _ => {}
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
                
                _ => {}
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
        let mut seen_states: HashMap<u64, usize> = HashMap::new();
        
        for epoch in 0..self.config.max_epochs {
            let state_hash = anamnesis.state_hash();
            
            // Check for cycle
            if seen_states.contains_key(&state_hash) {
                // Cycle detected - this seed doesn't lead to a fixed point
                return ConvergenceStatus::Timeout { max_epochs: epoch };
            }
            
            seen_states.insert(state_hash, epoch);
            
            // Run epoch
            let result = self.executor.run_epoch(program, &anamnesis);
            
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
                _ => {}
            }
        }
        
        ConvergenceStatus::Timeout { max_epochs: self.config.max_epochs }
    }
    
    /// Parallel exploration of multiple seeds simultaneously.
    /// 
    /// When the `parallel` feature is enabled, this uses rayon for parallel execution.
    /// Otherwise, it falls back to sequential execution (same as action-guided).
    fn run_parallel(
        &mut self,
        program: &Program,
        num_workers: usize,
        num_seeds: usize,
    ) -> ConvergenceStatus {
        // NOTE: True parallel execution is blocked because Provenance uses Rc<BTreeSet>
        // which is not Send. A future refactor could use Arc for thread-safety.
        // For now, we use sequential multi-seed exploration (same as action-guided).
        
        #[cfg(feature = "parallel")]
        {
            if self.config.verbose {
                println!(
                    "Parallel mode requested ({} workers, {} seeds) - using optimized sequential exploration",
                    if num_workers == 0 { num_cpus::get() } else { num_workers },
                    num_seeds
                );
                println!("Note: True parallelism requires Arc-based Provenance (future work)");
            }
        }
        
        #[cfg(not(feature = "parallel"))]
        {
            let _ = num_workers;
        }
        
        // Use action-guided search which explores multiple seeds sequentially
        self.run_action_guided(program, ActionConfig::anti_trivial(), num_seeds)
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
    fn find_oscillating_cells(&self, states: &[Memory]) -> Vec<Address> {
        if states.is_empty() {
            return Vec::new();
        }
        
        let mut oscillating = Vec::new();
        let first = &states[0];
        
        for addr in 0..256u16 { // Check first 256 cells for efficiency
            let base_val = first.read(addr).val;
            for state in states.iter().skip(1) {
                if state.read(addr).val != base_val {
                    oscillating.push(addr);
                    break;
                }
            }
        }
        
        oscillating
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
    fn detect_divergence(&self, trajectory: &[Memory]) -> Option<(Vec<Address>, Direction)> {
        if trajectory.len() < 5 {
            return None;
        }
        
        let mut diverging = Vec::new();
        let mut direction = Direction::Increasing;
        
        // Check each cell for monotonic growth
        for addr in 0..256u16 {
            let values: Vec<u64> = trajectory.iter()
                .map(|m| m.read(addr).val)
                .collect();
            
            // Check if strictly increasing
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
        let mut driver = TimeLoop::new(config);
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
}
