//! Action Principle for OUROCHRONOS fixed-point selection.
//!
//! This module implements a variational approach to selecting preferred fixed points
//! when multiple solutions exist. Analogous to the Principle of Least Action in physics,
//! we define a cost function (the "action") that penalizes trivial solutions and
//! rewards meaningful causal computation.
//!
//! # The Genie Effect
//!
//! In temporal programming, the system seeks any S such that S = F(S). However,
//! trivial solutions (e.g., all zeros, identity loops) often satisfy this constraint
//! more easily than the intended solution. This is the "Genie Effect" - the system
//! finds the path of least resistance rather than the most meaningful path.
//!
//! # The Solution: Action-Weighted Selection
//!
//! We define an action functional:
//!
//! ```text
//! Action(S) = Σᵢ [ w_zero · δ(Sᵢ = 0) + w_pure · δ(Sᵢ is pure) - w_causal · depth(Sᵢ) ]
//! ```
//!
//! Where:
//! - `w_zero`: Penalty for trivial zero values
//! - `w_pure`: Penalty for values with no oracle dependency  
//! - `w_causal`: Bonus for values with high causal depth
//!
//! The runtime explores multiple fixed points and selects the one with minimum action.

use crate::core_types::{Memory, Address, Value, OutputItem};
use crate::provenance::Provenance;
use std::collections::HashMap;

/// Configuration for the action principle.
/// 
/// Tune these weights to control what kinds of fixed points are preferred.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionConfig {
    /// Penalty applied for each cell that is zero (trivial value).
    /// Higher values discourage trivial solutions.
    pub zero_penalty: f64,
    
    /// Penalty applied for cells with no temporal dependency (pure values).
    /// Encourages solutions that actually use information from the oracle.
    pub pure_penalty: f64,
    
    /// Bonus (negative penalty) for each unit of causal depth.
    /// Rewards computations that depend on oracle reads.
    pub causal_bonus: f64,
    
    /// Penalty for cells that have the same value as their seed.
    /// Discourages "lazy" fixed points where nothing changes.
    pub unchanged_penalty: f64,
    
    /// Bonus for output-producing fixed points.
    /// Encourages solutions that actually produce observable results.
    pub output_bonus: f64,
    
    /// Weight for entropy (diversity of values).
    /// Higher values prefer solutions with more varied cell values.
    pub entropy_weight: f64,
}

impl Default for ActionConfig {
    fn default() -> Self {
        Self {
            zero_penalty: 0.1,
            pure_penalty: 0.5,
            causal_bonus: 1.0,
            unchanged_penalty: 0.2,
            output_bonus: 2.0,
            entropy_weight: 0.3,
        }
    }
}

impl ActionConfig {
    /// Create a configuration that strongly penalizes trivial solutions.
    pub fn anti_trivial() -> Self {
        Self {
            zero_penalty: 1.0,
            pure_penalty: 2.0,
            causal_bonus: 0.5,
            unchanged_penalty: 1.0,
            output_bonus: 3.0,
            entropy_weight: 0.5,
        }
    }
    
    /// Create a configuration that prefers minimal computation.
    /// Useful when any valid fixed point is acceptable.
    pub fn minimal() -> Self {
        Self {
            zero_penalty: 0.0,
            pure_penalty: 0.0,
            causal_bonus: 0.0,
            unchanged_penalty: 0.0,
            output_bonus: 0.0,
            entropy_weight: 0.0,
        }
    }
    
    /// Create a configuration optimized for factorization/witness problems.
    pub fn witness_finding() -> Self {
        Self {
            zero_penalty: 0.5,
            pure_penalty: 1.0,
            causal_bonus: 2.0,
            unchanged_penalty: 0.3,
            output_bonus: 5.0,
            entropy_weight: 0.1,
        }
    }
}

/// Seed generation strategy for constraint-based seeding.
/// Instead of always starting with zeros, use domain-appropriate seeds.
#[derive(Debug, Clone)]
pub enum SeedStrategy {
    /// All zeros (classic default).
    Zero,
    
    /// Random values in a specified range.
    Random { min: u64, max: u64 },
    
    /// Small primes (optimized for factor-finding).
    SmallPrimes,
    
    /// Incremental sequence (2, 3, 4, ...).
    Incremental,
    
    /// Constraint-based: satisfy PREFER/REQUIRE annotations.
    Constraint { constraints: Vec<SeedConstraint> },
    
    /// User-provided custom seed.
    Custom(Memory),
}

/// A constraint for seed generation.
#[derive(Debug, Clone)]
pub struct SeedConstraint {
    /// Address to constrain.
    pub address: Address,
    /// Constraint type.
    pub constraint: ConstraintKind,
}

/// Types of constraints for seeding.
#[derive(Debug, Clone)]
pub enum ConstraintKind {
    /// Value must be non-zero.
    NonZero,
    /// Value must be in range [min, max].
    InRange { min: u64, max: u64 },
    /// Value must equal a specific value.
    Equals(u64),
    /// Value must be distinct from other constrained values.
    Distinct,
}

/// Configuration for heuristic-based seeding.
#[derive(Debug, Clone, Default)]
pub struct HeuristicConfig {
    /// Bias toward small primes for factorization problems.
    pub factor_bias: bool,
    /// Bias toward permutations for sorting problems.
    pub permutation_bias: bool,
    /// Expected value range based on constants in program.
    pub value_range: Option<(u64, u64)>,
}

impl SeedStrategy {
    /// Infer the best seed strategy from program characteristics.
    /// 
    /// Analyzes the program to detect common patterns:
    /// - MOD operations suggest factor-finding → SmallPrimes
    /// - Comparison chains suggest sorting → permutation values
    /// - Constants in program suggest bounded ranges
    pub fn infer_from_program(program: &crate::ast::Program) -> Self {
        let mut has_mod = false;
        let mut has_comparisons = false;
        let mut max_constant = 0u64;
        
        // Simple AST scan for heuristics
        for stmt in &program.body {
            Self::scan_stmt(stmt, &mut has_mod, &mut has_comparisons, &mut max_constant);
        }
        
        if has_mod && max_constant > 1 {
            // Likely factorization/divisibility problem
            SeedStrategy::SmallPrimes
        } else if has_comparisons && !has_mod {
            // Likely sorting/ordering problem
            SeedStrategy::Incremental
        } else if max_constant > 100 {
            // Large constants suggest bounded range
            SeedStrategy::Random { min: 1, max: max_constant }
        } else {
            // Default to zero
            SeedStrategy::Zero
        }
    }
    
    fn scan_stmt(stmt: &crate::ast::Stmt, has_mod: &mut bool, has_comp: &mut bool, max_const: &mut u64) {
        use crate::ast::{Stmt, OpCode};
        
        match stmt {
            Stmt::Op(OpCode::Mod) => *has_mod = true,
            Stmt::Op(OpCode::Lt) | Stmt::Op(OpCode::Gt) | 
            Stmt::Op(OpCode::Lte) | Stmt::Op(OpCode::Gte) => *has_comp = true,
            Stmt::Push(val) => *max_const = (*max_const).max(val.val),
            Stmt::If { then_branch, else_branch } => {
                for s in then_branch { Self::scan_stmt(s, has_mod, has_comp, max_const); }
                if let Some(eb) = else_branch {
                    for s in eb { Self::scan_stmt(s, has_mod, has_comp, max_const); }
                }
            }
            Stmt::While { cond, body } => {
                for s in cond { Self::scan_stmt(s, has_mod, has_comp, max_const); }
                for s in body { Self::scan_stmt(s, has_mod, has_comp, max_const); }
            }
            Stmt::Block(stmts) => {
                for s in stmts { Self::scan_stmt(s, has_mod, has_comp, max_const); }
            }
            _ => {}
        }
    }
}

impl Default for SeedStrategy {
    fn default() -> Self {
        SeedStrategy::Zero
    }
}

/// Generator for producing seeds based on a strategy.
pub struct SeedGenerator {
    strategy: SeedStrategy,
    iteration: usize,
}

impl SeedGenerator {
    /// Create a new generator with the given strategy.
    pub fn new(strategy: SeedStrategy) -> Self {
        Self { strategy, iteration: 0 }
    }
    
    /// Generate the next seed memory state.
    pub fn next_seed(&mut self) -> Memory {
        let seed = self.generate_seed();
        self.iteration += 1;
        seed
    }
    
    /// Generate a batch of diverse seeds.
    pub fn generate_batch(&mut self, count: usize) -> Vec<Memory> {
        (0..count).map(|_| self.next_seed()).collect()
    }
    
    fn generate_seed(&self) -> Memory {
        match &self.strategy {
            SeedStrategy::Zero => Memory::new(),
            
            SeedStrategy::Random { min, max } => {
                let mut mem = Memory::new();
                // Simple pseudo-random: mix iteration with address
                for addr in 0..16 {
                    let val = *min + ((self.iteration as u64 * 31 + addr as u64 * 17) % (*max - *min + 1));
                    if val != 0 {
                        mem.write(addr as Address, Value::new(val));
                    }
                }
                mem
            }
            
            SeedStrategy::SmallPrimes => {
                let primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];
                let mut mem = Memory::new();
                // Use iteration to select different primes
                let idx = self.iteration % primes.len();
                mem.write(0, Value::new(primes[idx]));
                mem
            }
            
            SeedStrategy::Incremental => {
                let mut mem = Memory::new();
                let val = (self.iteration + 2) as u64; // Start at 2
                mem.write(0, Value::new(val));
                mem
            }
            
            SeedStrategy::Constraint { constraints } => {
                let mut mem = Memory::new();
                for constraint in constraints {
                    let val = match &constraint.constraint {
                        ConstraintKind::NonZero => (self.iteration + 1) as u64,
                        ConstraintKind::InRange { min, max } => {
                            *min + (self.iteration as u64 % (*max - *min + 1))
                        }
                        ConstraintKind::Equals(v) => *v,
                        ConstraintKind::Distinct => (self.iteration + 1) as u64,
                    };
                    mem.write(constraint.address, Value::new(val));
                }
                mem
            }
            
            SeedStrategy::Custom(mem) => mem.clone(),
        }
    }
    
    /// Reset the iteration counter.
    pub fn reset(&mut self) {
        self.iteration = 0;
    }
}

/// Selection rule for choosing among equal-action fixed points.
/// 
/// When multiple fixed points have the same action value, this rule
/// determines which one is selected. This is crucial for determinism.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelectionRule {
    /// Select by minimum action only (non-deterministic tiebreaking).
    /// Use only for backward compatibility or when non-determinism is acceptable.
    MinAction,
    
    /// Canonical chronology: action first, then lexicographic memory comparison.
    /// This guarantees identical selection across runs and platforms.
    #[default]
    Canonical,
    
    /// Prefer faster convergence: action first, then minimum epochs.
    /// Useful when convergence speed is more important than canonical ordering.
    MinEpochs,
}

/// Tracks causal depth and provenance for action computation.
#[derive(Debug, Clone, Default)]
pub struct ProvenanceMap {
    /// Map from address to causal depth (longest chain from oracle read).
    depths: HashMap<Address, usize>,
    
    /// Map from address to set of oracle dependencies.
    dependencies: HashMap<Address, Vec<Address>>,
}

impl ProvenanceMap {
    /// Create a new empty provenance map.
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Record that a value at `addr` was derived from oracle read at `oracle_addr`.
    pub fn record_oracle_dependency(&mut self, addr: Address, oracle_addr: Address) {
        self.dependencies.entry(addr).or_default().push(oracle_addr);
        // Oracle reads have depth 1
        self.depths.insert(addr, 1);
    }
    
    /// Record that `addr` was computed from `sources` via some operation.
    pub fn record_computation(&mut self, addr: Address, sources: &[Address]) {
        let max_source_depth = sources.iter()
            .filter_map(|s| self.depths.get(s))
            .max()
            .copied()
            .unwrap_or(0);
        
        self.depths.insert(addr, max_source_depth + 1);
        
        // Merge dependencies from all sources
        let mut deps: Vec<Address> = Vec::new();
        for source in sources {
            if let Some(source_deps) = self.dependencies.get(source) {
                deps.extend(source_deps.iter().copied());
            }
        }
        deps.sort();
        deps.dedup();
        if !deps.is_empty() {
            self.dependencies.insert(addr, deps);
        }
    }
    
    /// Get the causal depth of a cell (0 = pure constant, higher = more causal chain).
    pub fn causal_depth(&self, addr: Address) -> usize {
        self.depths.get(&addr).copied().unwrap_or(0)
    }
    
    /// Check if a cell has any oracle dependency.
    pub fn is_temporal(&self, addr: Address) -> bool {
        self.dependencies.get(&addr).map(|d| !d.is_empty()).unwrap_or(false)
    }
    
    /// Get oracle dependencies for a cell.
    pub fn oracle_dependencies(&self, addr: Address) -> Option<&[Address]> {
        self.dependencies.get(&addr).map(|v| v.as_slice())
    }
    
    /// Build provenance map from a value's embedded provenance.
    pub fn from_value_provenance(addr: Address, prov: &Provenance) -> Self {
        let mut map = Self::new();
        if prov.is_temporal() {
            let deps: Vec<Address> = prov.dependencies().collect();
            map.dependencies.insert(addr, deps);
            map.depths.insert(addr, 1);
        }
        map
    }
    
    /// Merge another provenance map into this one.
    pub fn merge(&mut self, other: &ProvenanceMap) {
        for (addr, depth) in &other.depths {
            let current = self.depths.entry(*addr).or_insert(0);
            *current = (*current).max(*depth);
        }
        for (addr, deps) in &other.dependencies {
            let current = self.dependencies.entry(*addr).or_default();
            current.extend(deps.iter().copied());
            current.sort();
            current.dedup();
        }
    }
}

/// The Action calculator for evaluating fixed-point quality.
#[derive(Debug, Clone)]
pub struct ActionPrinciple {
    config: ActionConfig,
}

impl ActionPrinciple {
    /// Create an action principle with the given configuration.
    pub fn new(config: ActionConfig) -> Self {
        Self { config }
    }
    
    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(ActionConfig::default())
    }
    
    /// Compute the action (cost) of a memory state.
    /// 
    /// Lower action = more preferred timeline.
    /// 
    /// # Arguments
    /// * `memory` - The fixed-point memory state
    /// * `seed` - The initial (seed) memory state
    /// * `output_count` - Number of values output during execution
    pub fn compute(
        &self,
        memory: &Memory,
        seed: &Memory,
        output_count: usize,
    ) -> f64 {
        let mut action = 0.0;
        
        // Count non-zero cells and compute penalties
        let mut non_zero_count = 0usize;
        let mut unique_values: HashMap<u64, usize> = HashMap::new();
        
        for (addr, value) in memory.iter_nonzero() {
            non_zero_count += 1;
            *unique_values.entry(value.val).or_insert(0) += 1;
            
            // Penalty for pure (non-temporal) values
            if value.prov.is_pure() {
                action += self.config.pure_penalty;
            } else {
                // Bonus for causal depth
                let depth = value.prov.dependency_count();
                action -= self.config.causal_bonus * (depth as f64).ln_1p();
            }
            
            // Penalty if value unchanged from seed
            if memory.read(addr) == seed.read(addr) {
                action += self.config.unchanged_penalty;
            }
        }
        
        // CRITICAL: Penalize trivial (all-zero) solutions heavily
        // This is the main mechanism to avoid the "Genie Effect"
        if non_zero_count == 0 {
            // All-zero solution: maximum penalty for triviality
            action += self.config.zero_penalty * 100.0;
        } else {
            // Bonus for having meaningful content (more non-zero cells = better)
            action -= self.config.zero_penalty * (non_zero_count as f64).ln_1p();
        }
        
        // Entropy bonus (prefer diverse values)
        if non_zero_count > 1 {
            let entropy = self.shannon_entropy(&unique_values, non_zero_count);
            action -= self.config.entropy_weight * entropy;
        }
        
        // Output bonus - THIS IS THE KEY REWARD
        // Solutions that produce output are strongly preferred
        if output_count > 0 {
            action -= self.config.output_bonus * (output_count as f64 + 1.0).ln();
        }
        
        action
    }
    
    /// Compute Shannon entropy of value distribution.
    fn shannon_entropy(&self, value_counts: &HashMap<u64, usize>, total: usize) -> f64 {
        if total == 0 {
            return 0.0;
        }
        
        let total_f = total as f64;
        value_counts.values()
            .map(|&count| {
                let p = count as f64 / total_f;
                if p > 0.0 { -p * p.log2() } else { 0.0 }
            })
            .sum()
    }
    
    /// Compare two fixed points and return which is preferred.
    /// Returns Ordering::Less if `a` is preferred over `b`.
    pub fn compare(
        &self,
        a: &Memory,
        b: &Memory,
        seed: &Memory,
        output_a: usize,
        output_b: usize,
    ) -> std::cmp::Ordering {
        let action_a = self.compute(a, seed, output_a);
        let action_b = self.compute(b, seed, output_b);
        
        action_a.partial_cmp(&action_b).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl Default for ActionPrinciple {
    fn default() -> Self {
        Self::default_config()
    }
}

/// A candidate fixed point with its computed action.
#[derive(Debug, Clone)]
pub struct FixedPointCandidate {
    /// The memory state.
    pub memory: Memory,
    /// Computed action (lower = better).
    pub action: f64,
    /// Number of epochs to reach this fixed point.
    pub epochs: usize,
    /// Output produced during execution.
    pub output: Vec<OutputItem>,
    /// The seed that led to this fixed point.
    pub seed: Memory,
}

impl FixedPointCandidate {
    /// Create a new candidate.
    pub fn new(memory: Memory, action: f64, epochs: usize, output: Vec<OutputItem>, seed: Memory) -> Self {
        Self { memory, action, epochs, output, seed }
    }
    
    /// Compare candidates using canonical ordering (action, then memory).
    pub fn canonical_cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.action.partial_cmp(&other.action) {
            Some(std::cmp::Ordering::Equal) | None => self.memory.cmp(&other.memory),
            Some(ord) => ord,
        }
    }
    
    /// Compare candidates using min-epochs ordering (action, then epochs, then memory).
    pub fn min_epochs_cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.action.partial_cmp(&other.action) {
            Some(std::cmp::Ordering::Equal) | None => {
                match self.epochs.cmp(&other.epochs) {
                    std::cmp::Ordering::Equal => self.memory.cmp(&other.memory),
                    ord => ord,
                }
            }
            Some(ord) => ord,
        }
    }
}

/// Selector for choosing among multiple fixed-point candidates.
pub struct FixedPointSelector {
    principle: ActionPrinciple,
    candidates: Vec<FixedPointCandidate>,
    selection_rule: SelectionRule,
}

impl FixedPointSelector {
    /// Create a new selector with the given action principle (uses Canonical selection).
    pub fn new(principle: ActionPrinciple) -> Self {
        Self {
            principle,
            candidates: Vec::new(),
            selection_rule: SelectionRule::Canonical,
        }
    }
    
    /// Create a selector with a specific selection rule.
    pub fn with_rule(principle: ActionPrinciple, rule: SelectionRule) -> Self {
        Self {
            principle,
            candidates: Vec::new(),
            selection_rule: rule,
        }
    }
    
    /// Add a candidate fixed point.
    pub fn add_candidate(&mut self, memory: Memory, epochs: usize, output: Vec<OutputItem>, seed: Memory) {
        let action = self.principle.compute(&memory, &seed, output.len());
        self.candidates.push(FixedPointCandidate::new(memory, action, epochs, output, seed));
    }
    
    /// Get the number of candidates.
    pub fn candidate_count(&self) -> usize {
        self.candidates.len()
    }
    
    /// Select the best candidate according to the selection rule.
    /// 
    /// Returns the candidate with the lowest action. When actions are equal,
    /// the selection rule determines the tiebreaker:
    /// - `Canonical`: lexicographic memory comparison (deterministic)
    /// - `MinEpochs`: fewer epochs wins, then memory comparison
    /// - `MinAction`: action only (non-deterministic under parallelism)
    pub fn select_best(mut self) -> Option<FixedPointCandidate> {
        if self.candidates.is_empty() {
            return None;
        }
        
        self.candidates.sort_by(|a, b| {
            match self.selection_rule {
                SelectionRule::Canonical => a.canonical_cmp(b),
                SelectionRule::MinEpochs => a.min_epochs_cmp(b),
                SelectionRule::MinAction => {
                    a.action.partial_cmp(&b.action).unwrap_or(std::cmp::Ordering::Equal)
                }
            }
        });
        
        Some(self.candidates.remove(0))
    }
    
    /// Get all candidates sorted according to the selection rule.
    pub fn all_sorted(mut self) -> Vec<FixedPointCandidate> {
        self.candidates.sort_by(|a, b| {
            match self.selection_rule {
                SelectionRule::Canonical => a.canonical_cmp(b),
                SelectionRule::MinEpochs => a.min_epochs_cmp(b),
                SelectionRule::MinAction => {
                    a.action.partial_cmp(&b.action).unwrap_or(std::cmp::Ordering::Equal)
                }
            }
        });
        self.candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_types::Memory;
    
    #[test]
    fn test_action_prefers_non_trivial() {
        let principle = ActionPrinciple::new(ActionConfig::anti_trivial());
        
        // Trivial solution: all zeros
        let trivial = Memory::new();
        let seed = Memory::new();
        
        // Non-trivial solution: has some values
        let mut non_trivial = Memory::new();
        non_trivial.write(0, Value::new(42));
        non_trivial.write(1, Value::new(17));
        
        let action_trivial = principle.compute(&trivial, &seed, 0);
        let action_non_trivial = principle.compute(&non_trivial, &seed, 1);
        
        // Non-trivial should have lower action (better)
        assert!(action_non_trivial < action_trivial, 
            "Non-trivial ({}) should have lower action than trivial ({})", 
            action_non_trivial, action_trivial);
    }
    
    #[test]
    fn test_action_prefers_output() {
        let principle = ActionPrinciple::default_config();
        
        let mut memory = Memory::new();
        memory.write(0, Value::new(42));
        let seed = Memory::new();
        
        let action_no_output = principle.compute(&memory, &seed, 0);
        let action_with_output = principle.compute(&memory, &seed, 5);
        
        assert!(action_with_output < action_no_output,
            "With output ({}) should have lower action than without ({})",
            action_with_output, action_no_output);
    }
    
    #[test]
    fn test_selector_picks_best() {
        let principle = ActionPrinciple::new(ActionConfig::anti_trivial());
        let mut selector = FixedPointSelector::new(principle);
        
        let seed = Memory::new();
        
        // Trivial candidate
        let trivial = Memory::new();
        selector.add_candidate(trivial, 1, vec![], seed.clone());
        
        // Better candidate
        let mut better = Memory::new();
        better.write(0, Value::new(42));
        selector.add_candidate(better.clone(), 2, vec![OutputItem::Val(Value::new(42))], seed.clone());
        
        let best = selector.select_best().unwrap();
        assert_eq!(best.memory.read(0).val, 42);
    }
    
    #[test]
    fn test_provenance_map() {
        let mut prov_map = ProvenanceMap::new();
        
        // Record oracle read at address 0
        prov_map.record_oracle_dependency(0, 0);
        assert_eq!(prov_map.causal_depth(0), 1);
        assert!(prov_map.is_temporal(0));
        
        // Record computation from address 0 to address 1
        prov_map.record_computation(1, &[0]);
        assert_eq!(prov_map.causal_depth(1), 2);
        assert!(prov_map.is_temporal(1));
        
        // Pure address has depth 0
        assert_eq!(prov_map.causal_depth(100), 0);
        assert!(!prov_map.is_temporal(100));
    }
    
    #[test]
    fn test_entropy_calculation() {
        let principle = ActionPrinciple::default_config();
        
        // All same values = low entropy
        let mut same_values: HashMap<u64, usize> = HashMap::new();
        same_values.insert(1, 10);
        let low_entropy = principle.shannon_entropy(&same_values, 10);
        
        // All different values = high entropy
        let mut diff_values: HashMap<u64, usize> = HashMap::new();
        for i in 0..10 {
            diff_values.insert(i, 1);
        }
        let high_entropy = principle.shannon_entropy(&diff_values, 10);
        
        assert!(high_entropy > low_entropy,
            "Different values ({}) should have higher entropy than same ({})",
            high_entropy, low_entropy);
    }
}
