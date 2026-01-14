//! Temporal Optimizations for OUROCHRONOS
//!
//! Implements Phase 6 temporal optimizations:
//! - Fixed-point memoization with LRU eviction
//! - Epoch caching with smart invalidation
//! - Speculative execution for parallel fixed-point search
//! - Parallel epoch evaluation
//! - Delta tracking for incremental computation
//! - Fixed-point acceleration (Aitken, Anderson)

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use crate::core_types::{Memory, OutputItem, Value, MEMORY_SIZE};
use crate::vm::EpochStatus;

/// Maximum cache size before eviction.
const DEFAULT_MAX_CACHE_SIZE: usize = 1024;
/// Default speculative lookahead depth.
const DEFAULT_LOOKAHEAD: usize = 3;

/// Computes a hash of a memory state for cache lookup.
fn hash_memory(memory: &Memory) -> u64 {
    let mut hasher = DefaultHasher::new();
    for addr in 0..MEMORY_SIZE {
        let val = memory.read(addr as u16);
        val.hash(&mut hasher);
    }
    hasher.finish()
}

/// Computes a partial hash of only specific cells (faster for sparse changes).
fn hash_memory_sparse(memory: &Memory, cells: &HashSet<u16>) -> u64 {
    let mut hasher = DefaultHasher::new();
    let mut sorted_cells: Vec<_> = cells.iter().copied().collect();
    sorted_cells.sort();
    for addr in sorted_cells {
        addr.hash(&mut hasher);
        memory.read(addr).val.hash(&mut hasher);
    }
    hasher.finish()
}

/// The result of a memoized epoch execution.
#[derive(Debug, Clone)]
pub struct MemoizedResult {
    /// The resulting present state.
    pub present: Memory,
    /// Output produced.
    pub output: Vec<OutputItem>,
    /// Status of execution.
    pub status: EpochStatus,
    /// Cells that were modified.
    pub modified_cells: HashSet<u16>,
    /// Oracle addresses read.
    pub oracle_reads: HashSet<u16>,
    /// Prophecy addresses written.
    pub prophecy_writes: HashSet<u16>,
    /// Time taken to compute.
    pub compute_time: Duration,
    /// Number of instructions executed.
    pub instruction_count: usize,
}

impl MemoizedResult {
    /// Create a simple memoized result.
    pub fn simple(present: Memory, output: Vec<OutputItem>, status: EpochStatus) -> Self {
        Self {
            present,
            output,
            status,
            modified_cells: HashSet::new(),
            oracle_reads: HashSet::new(),
            prophecy_writes: HashSet::new(),
            compute_time: Duration::ZERO,
            instruction_count: 0,
        }
    }

    /// Builder: set compute time.
    pub fn with_compute_time(mut self, time: Duration) -> Self {
        self.compute_time = time;
        self
    }

    /// Builder: set instruction count.
    pub fn with_instruction_count(mut self, count: usize) -> Self {
        self.instruction_count = count;
        self
    }

    /// Builder: set modified cells.
    pub fn with_modified_cells(mut self, cells: HashSet<u16>) -> Self {
        self.modified_cells = cells;
        self
    }
}

/// Cache entry with metadata for LRU eviction.
#[derive(Debug)]
struct CacheEntry {
    result: MemoizedResult,
    access_count: usize,
    last_access: Instant,
    validated: bool,
}

impl CacheEntry {
    fn new(result: MemoizedResult) -> Self {
        Self {
            result,
            access_count: 1,
            last_access: Instant::now(),
            validated: false,
        }
    }

    fn touch(&mut self) {
        self.access_count += 1;
        self.last_access = Instant::now();
    }
}

/// Cache for memoizing epoch results.
///
/// Keys are state hashes, values are the computed results.
/// Uses LRU eviction when cache is full.
#[derive(Debug)]
pub struct EpochCache {
    /// Cached results indexed by memory hash.
    entries: HashMap<u64, CacheEntry>,
    /// Maximum cache size.
    max_size: usize,
    /// Legacy simple cache for backward compatibility.
    legacy_cache: HashMap<u64, MemoizedResult>,
    /// Statistics.
    hits: usize,
    misses: usize,
    evictions: usize,
    invalidations: usize,
    /// Invalidation tracking.
    invalidation_set: HashSet<u64>,
    /// Dependency graph: entry -> entries that depend on it.
    dependencies: HashMap<u64, HashSet<u64>>,
}

impl Default for EpochCache {
    fn default() -> Self {
        Self::new()
    }
}

impl EpochCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_MAX_CACHE_SIZE)
    }

    /// Create a cache with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            max_size: capacity,
            legacy_cache: HashMap::with_capacity(capacity),
            hits: 0,
            misses: 0,
            evictions: 0,
            invalidations: 0,
            invalidation_set: HashSet::new(),
            dependencies: HashMap::new(),
        }
    }

    /// Look up a cached result for the given memory state.
    pub fn lookup(&mut self, memory: &Memory) -> Option<&MemoizedResult> {
        let hash = hash_memory(memory);
        self.lookup_by_hash(hash)
    }

    /// Look up by precomputed hash.
    pub fn lookup_by_hash(&mut self, hash: u64) -> Option<&MemoizedResult> {
        // Check if this entry is invalidated
        if self.invalidation_set.contains(&hash) {
            self.misses += 1;
            return None;
        }

        if let Some(entry) = self.entries.get_mut(&hash) {
            entry.touch();
            self.hits += 1;
            Some(&entry.result)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Look up a cached result by state hash (legacy API).
    pub fn get(&mut self, state_hash: u64) -> Option<&MemoizedResult> {
        if let Some(result) = self.legacy_cache.get(&state_hash) {
            self.hits += 1;
            Some(result)
        } else if let Some(entry) = self.entries.get_mut(&state_hash) {
            entry.touch();
            self.hits += 1;
            Some(&entry.result)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Store a result in the cache.
    pub fn store(&mut self, memory: &Memory, result: MemoizedResult) {
        let hash = hash_memory(memory);
        self.store_by_hash(hash, result);
    }

    /// Store by precomputed hash.
    pub fn store_by_hash(&mut self, hash: u64, result: MemoizedResult) {
        // Remove from invalidation set if present
        self.invalidation_set.remove(&hash);

        // Evict if at capacity
        if self.entries.len() >= self.max_size {
            self.evict_lru();
        }

        self.entries.insert(hash, CacheEntry::new(result));
    }

    /// Store a result in the cache (legacy API).
    pub fn insert(&mut self, state_hash: u64, result: MemoizedResult) {
        if self.legacy_cache.len() >= self.max_size {
            // Evict oldest entry
            if let Some(&key) = self.legacy_cache.keys().next() {
                self.legacy_cache.remove(&key);
                self.evictions += 1;
            }
        }
        self.legacy_cache.insert(state_hash, result);
    }

    /// Store with dependency tracking.
    pub fn store_with_deps(&mut self, hash: u64, result: MemoizedResult, depends_on: &[u64]) {
        for &dep_hash in depends_on {
            self.dependencies
                .entry(dep_hash)
                .or_insert_with(HashSet::new)
                .insert(hash);
        }
        self.store_by_hash(hash, result);
    }

    /// Invalidate a cache entry and its dependents.
    pub fn invalidate(&mut self, memory: &Memory) {
        let hash = hash_memory(memory);
        self.invalidate_by_hash(hash);
    }

    /// Invalidate by hash.
    pub fn invalidate_by_hash(&mut self, hash: u64) {
        self.invalidation_set.insert(hash);
        self.invalidations += 1;

        // Cascade invalidation to dependents
        if let Some(deps) = self.dependencies.remove(&hash) {
            for dep_hash in deps {
                self.invalidate_by_hash(dep_hash);
            }
        }
    }

    /// Invalidate entries that depend on specific memory addresses.
    pub fn invalidate_by_addresses(&mut self, addresses: &HashSet<u16>) {
        let to_invalidate: Vec<u64> = self.entries.iter()
            .filter(|(_, entry)| {
                !entry.result.oracle_reads.is_disjoint(addresses) ||
                !entry.result.prophecy_writes.is_disjoint(addresses)
            })
            .map(|(&hash, _)| hash)
            .collect();

        for hash in to_invalidate {
            self.invalidate_by_hash(hash);
        }
    }

    /// Evict the least recently used entry.
    fn evict_lru(&mut self) {
        if let Some((&hash, _)) = self.entries.iter()
            .min_by(|(_, a), (_, b)| {
                a.access_count.cmp(&b.access_count)
                    .then_with(|| a.last_access.cmp(&b.last_access))
            })
        {
            self.entries.remove(&hash);
            self.dependencies.remove(&hash);
            self.evictions += 1;
        }
    }

    /// Check if a state is cached (legacy API).
    pub fn contains(&self, state_hash: u64) -> bool {
        self.legacy_cache.contains_key(&state_hash) || self.entries.contains_key(&state_hash)
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            size: self.entries.len() + self.legacy_cache.len(),
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            invalidations: self.invalidations,
            hit_rate: if self.hits + self.misses > 0 {
                self.hits as f64 / (self.hits + self.misses) as f64
            } else {
                0.0
            },
        }
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.legacy_cache.clear();
        self.invalidation_set.clear();
        self.dependencies.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Get current cache size.
    pub fn len(&self) -> usize {
        self.entries.len() + self.legacy_cache.len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.legacy_cache.is_empty()
    }

    /// Get hit rate.
    pub fn hit_rate(&self) -> f64 {
        self.stats().hit_rate
    }
}

/// Statistics about cache performance.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of entries in cache.
    pub size: usize,
    /// Number of cache hits.
    pub hits: usize,
    /// Number of cache misses.
    pub misses: usize,
    /// Number of evictions.
    pub evictions: usize,
    /// Number of invalidations.
    pub invalidations: usize,
    /// Hit rate (0.0 to 1.0).
    pub hit_rate: f64,
}

impl CacheStats {
    /// Total lookups.
    pub fn total_lookups(&self) -> usize {
        self.hits + self.misses
    }

    /// Reset statistics.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

impl std::fmt::Display for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cache: {} entries, {}/{} hits ({:.1}%), {} evictions",
            self.size, self.hits, self.hits + self.misses, self.hit_rate * 100.0,
            self.evictions)
    }
}

/// Incremental computation helper.
///
/// Detects which memory cells have changed between epochs
/// and enables delta-based execution for faster convergence detection.
#[derive(Debug, Default, Clone)]
pub struct DeltaTracker {
    /// Cells modified in current epoch.
    modified_cells: HashSet<u16>,
    /// Previous memory state hash.
    prev_hash: Option<u64>,
    /// Previous non-zero cells for tracking.
    previous_nonzero: Vec<u16>,
    /// Number of unchanged epochs.
    stable_count: usize,
    /// Threshold for declaring convergence.
    stability_threshold: usize,
}

impl DeltaTracker {
    /// Create a new delta tracker.
    pub fn new() -> Self {
        Self::with_threshold(3)
    }

    /// Create with custom stability threshold.
    pub fn with_threshold(threshold: usize) -> Self {
        Self {
            modified_cells: HashSet::new(),
            prev_hash: None,
            previous_nonzero: Vec::new(),
            stable_count: 0,
            stability_threshold: threshold,
        }
    }

    /// Record a memory modification.
    pub fn record_modification(&mut self, addr: u16) {
        self.modified_cells.insert(addr);
    }

    /// Record multiple modifications.
    pub fn record_modifications(&mut self, addrs: impl Iterator<Item = u16>) {
        self.modified_cells.extend(addrs);
    }

    /// Check if memory state matches previous epoch.
    pub fn check_convergence(&mut self, memory: &Memory) -> bool {
        let hash = hash_memory(memory);

        if Some(hash) == self.prev_hash {
            self.stable_count += 1;
        } else {
            self.stable_count = 0;
        }

        self.prev_hash = Some(hash);
        self.modified_cells.clear();

        self.stable_count >= self.stability_threshold
    }

    /// Check convergence using only modified cells (faster).
    pub fn check_convergence_incremental(&mut self, memory: &Memory) -> bool {
        if self.modified_cells.is_empty() {
            self.stable_count += 1;
            return self.stable_count >= self.stability_threshold;
        }

        let hash = hash_memory_sparse(memory, &self.modified_cells);

        if Some(hash) == self.prev_hash {
            self.stable_count += 1;
        } else {
            self.stable_count = 0;
        }

        self.prev_hash = Some(hash);
        self.modified_cells.clear();

        self.stable_count >= self.stability_threshold
    }

    /// Compute the delta between two memory states.
    /// Returns the addresses that changed.
    pub fn compute_delta(&mut self, old: &Memory, new: &Memory) -> Vec<u16> {
        let mut changed = Vec::new();

        // Check all addresses that were non-zero in old
        for &addr in &self.previous_nonzero {
            if old.read(addr) != new.read(addr) {
                changed.push(addr);
            }
        }

        // Update tracking for next iteration
        self.previous_nonzero = new.non_zero_cells().into_iter().map(|(addr, _)| addr).collect();

        // Also check new non-zero addresses
        for &addr in &self.previous_nonzero {
            if !changed.contains(&addr) && old.read(addr) != new.read(addr) {
                changed.push(addr);
            }
        }

        changed
    }

    /// Get modified cells.
    pub fn modified_cells(&self) -> &HashSet<u16> {
        &self.modified_cells
    }

    /// Get stable count.
    pub fn stable_count(&self) -> usize {
        self.stable_count
    }

    /// Check if states are identical (no delta).
    pub fn is_unchanged(old: &Memory, new: &Memory) -> bool {
        old.values_equal(new)
    }

    /// Reset tracker state.
    pub fn reset(&mut self) {
        self.modified_cells.clear();
        self.prev_hash = None;
        self.previous_nonzero.clear();
        self.stable_count = 0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Speculative Execution
// ═══════════════════════════════════════════════════════════════════════════

/// Speculative execution engine.
///
/// Executes epochs speculatively with multiple potential seeds,
/// keeping only the results that lead to valid fixed points.
#[derive(Debug)]
pub struct SpeculativeExecutor {
    /// Maximum speculative branches.
    max_branches: usize,
    /// Lookahead depth for speculation.
    lookahead: usize,
    /// Pending speculative branches.
    pending_branches: VecDeque<SpeculativeBranch>,
    /// Completed branches with results.
    completed_branches: Vec<SpeculativeResult>,
    /// Statistics.
    stats: SpeculativeStats,
}

/// A speculative execution branch.
#[derive(Debug, Clone)]
pub struct SpeculativeBranch {
    /// Branch ID.
    pub id: usize,
    /// Initial memory state.
    pub memory: Memory,
    /// Speculation depth (0 = root).
    pub depth: usize,
    /// Parent branch ID.
    pub parent: Option<usize>,
    /// Seed values used for this branch.
    pub seeds: Vec<(u16, u64)>,
}

/// Result of a speculative branch.
#[derive(Debug, Clone)]
pub struct SpeculativeResult {
    /// Branch that produced this result.
    pub branch_id: usize,
    /// Final memory state.
    pub memory: Memory,
    /// Whether this branch found a valid fixed point.
    pub found_fixed_point: bool,
    /// Number of epochs to convergence.
    pub epochs_to_converge: usize,
    /// Score for this result (lower is better).
    pub score: f64,
}

/// Statistics for speculative execution.
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    /// Total branches created.
    pub branches_created: usize,
    /// Branches that converged.
    pub branches_converged: usize,
    /// Branches pruned.
    pub branches_pruned: usize,
    /// Best result score.
    pub best_score: Option<f64>,
}

impl Default for SpeculativeExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeculativeExecutor {
    /// Create a new speculative executor.
    pub fn new() -> Self {
        Self::with_config(8, DEFAULT_LOOKAHEAD)
    }

    /// Create with custom configuration.
    pub fn with_config(max_branches: usize, lookahead: usize) -> Self {
        Self {
            max_branches,
            lookahead,
            pending_branches: VecDeque::new(),
            completed_branches: Vec::new(),
            stats: SpeculativeStats::default(),
        }
    }

    /// Start speculation from an initial memory state.
    pub fn start(&mut self, memory: Memory) {
        self.pending_branches.clear();
        self.completed_branches.clear();

        self.pending_branches.push_back(SpeculativeBranch {
            id: 0,
            memory,
            depth: 0,
            parent: None,
            seeds: Vec::new(),
        });

        self.stats.branches_created = 1;
    }

    /// Create a speculative branch with alternate seed values.
    pub fn branch(&mut self, parent: &SpeculativeBranch, seeds: Vec<(u16, u64)>) -> Option<usize> {
        if self.pending_branches.len() >= self.max_branches {
            self.stats.branches_pruned += 1;
            return None;
        }

        if parent.depth >= self.lookahead {
            return None;
        }

        let id = self.stats.branches_created;
        let mut new_memory = parent.memory.clone();

        // Apply seeds to memory
        for &(addr, val) in &seeds {
            new_memory.write(addr, Value::new(val));
        }

        self.pending_branches.push_back(SpeculativeBranch {
            id,
            memory: new_memory,
            depth: parent.depth + 1,
            parent: Some(parent.id),
            seeds,
        });

        self.stats.branches_created += 1;
        Some(id)
    }

    /// Take the next pending branch for execution.
    pub fn next_branch(&mut self) -> Option<SpeculativeBranch> {
        self.pending_branches.pop_front()
    }

    /// Record a completed branch.
    pub fn complete(&mut self, result: SpeculativeResult) {
        if result.found_fixed_point {
            self.stats.branches_converged += 1;
        }

        // Update best score
        if let Some(best) = self.stats.best_score {
            if result.score < best {
                self.stats.best_score = Some(result.score);
            }
        } else if result.found_fixed_point {
            self.stats.best_score = Some(result.score);
        }

        self.completed_branches.push(result);
    }

    /// Get the best converged result.
    pub fn best_result(&self) -> Option<&SpeculativeResult> {
        self.completed_branches.iter()
            .filter(|r| r.found_fixed_point)
            .min_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get all converged results sorted by score.
    pub fn converged_results(&self) -> Vec<&SpeculativeResult> {
        let mut results: Vec<_> = self.completed_branches.iter()
            .filter(|r| r.found_fixed_point)
            .collect();
        results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Check if there are pending branches.
    pub fn has_pending(&self) -> bool {
        !self.pending_branches.is_empty()
    }

    /// Get statistics.
    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    /// Reset executor state.
    pub fn reset(&mut self) {
        self.pending_branches.clear();
        self.completed_branches.clear();
        self.stats = SpeculativeStats::default();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Parallel Epoch Evaluation
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for parallel epoch evaluation.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of worker threads.
    pub num_workers: usize,
    /// Chunk size for work distribution.
    pub chunk_size: usize,
    /// Whether to enable work stealing.
    pub work_stealing: bool,
    /// Maximum parallel speculative branches.
    pub max_parallel_speculation: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_workers: num_cpus(),
            chunk_size: 64,
            work_stealing: true,
            max_parallel_speculation: 4,
        }
    }
}

/// Get number of CPUs (fallback if num_cpus crate not available).
fn num_cpus() -> usize {
    #[cfg(feature = "parallel")]
    {
        num_cpus::get()
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

/// Thread-safe epoch cache for parallel execution.
#[derive(Debug)]
pub struct SharedEpochCache {
    cache: Arc<RwLock<EpochCache>>,
}

impl Default for SharedEpochCache {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SharedEpochCache {
    fn clone(&self) -> Self {
        self.clone_shared()
    }
}

impl SharedEpochCache {
    /// Create a new shared cache.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_MAX_CACHE_SIZE)
    }

    /// Create with specified capacity.
    pub fn with_capacity(max_size: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(EpochCache::with_capacity(max_size))),
        }
    }

    /// Look up a cached result.
    pub fn lookup(&self, memory: &Memory) -> Option<MemoizedResult> {
        let hash = hash_memory(memory);
        let mut cache = self.cache.write().ok()?;
        cache.lookup_by_hash(hash).cloned()
    }

    /// Store a result.
    pub fn store(&self, memory: &Memory, result: MemoizedResult) {
        let hash = hash_memory(memory);
        if let Ok(mut cache) = self.cache.write() {
            cache.store_by_hash(hash, result);
        }
    }

    /// Invalidate by hash.
    pub fn invalidate(&self, hash: u64) {
        if let Ok(mut cache) = self.cache.write() {
            cache.invalidate_by_hash(hash);
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> Option<CacheStats> {
        self.cache.read().ok().map(|c| c.stats())
    }

    /// Clone the Arc for sharing.
    pub fn clone_shared(&self) -> Self {
        Self {
            cache: Arc::clone(&self.cache),
        }
    }
}

/// Parallel epoch evaluator.
///
/// Coordinates multiple threads to evaluate epochs in parallel,
/// useful for speculative execution and seed exploration.
#[derive(Debug)]
pub struct ParallelEvaluator {
    /// Configuration.
    config: ParallelConfig,
    /// Shared cache.
    cache: SharedEpochCache,
    /// Statistics.
    stats: ParallelStats,
}

/// Statistics for parallel evaluation.
#[derive(Debug, Clone, Default)]
pub struct ParallelStats {
    /// Total epochs evaluated.
    pub epochs_evaluated: usize,
    /// Epochs evaluated in parallel.
    pub parallel_epochs: usize,
    /// Cache hits during parallel execution.
    pub cache_hits: usize,
    /// Synchronization waits.
    pub sync_waits: usize,
    /// Total wall clock time.
    pub wall_time: Duration,
    /// Total CPU time across all threads.
    pub cpu_time: Duration,
}

impl ParallelStats {
    /// Calculate parallel efficiency.
    pub fn efficiency(&self) -> f64 {
        if self.wall_time.is_zero() {
            return 0.0;
        }
        self.cpu_time.as_secs_f64() / (self.wall_time.as_secs_f64() * num_cpus() as f64)
    }

    /// Calculate speedup vs single-threaded.
    pub fn speedup(&self) -> f64 {
        if self.wall_time.is_zero() {
            return 0.0;
        }
        self.cpu_time.as_secs_f64() / self.wall_time.as_secs_f64()
    }
}

impl std::fmt::Display for ParallelStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Parallel: {} epochs, {:.1}x speedup, {:.1}% efficiency",
            self.epochs_evaluated, self.speedup(), self.efficiency() * 100.0)
    }
}

impl Default for ParallelEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl ParallelEvaluator {
    /// Create a new parallel evaluator.
    pub fn new() -> Self {
        Self::with_config(ParallelConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: ParallelConfig) -> Self {
        Self {
            config,
            cache: SharedEpochCache::new(),
            stats: ParallelStats::default(),
        }
    }

    /// Get the shared cache.
    pub fn cache(&self) -> &SharedEpochCache {
        &self.cache
    }

    /// Get configuration.
    pub fn config(&self) -> &ParallelConfig {
        &self.config
    }

    /// Get statistics.
    pub fn stats(&self) -> &ParallelStats {
        &self.stats
    }

    /// Evaluate multiple memory states in parallel (when parallel feature enabled).
    #[cfg(feature = "parallel")]
    pub fn evaluate_parallel<F>(&mut self, states: Vec<Memory>, eval_fn: F) -> Vec<MemoizedResult>
    where
        F: Fn(&Memory) -> MemoizedResult + Send + Sync,
    {
        use rayon::prelude::*;

        let start = Instant::now();

        let cache = self.cache.clone_shared();
        let results: Vec<MemoizedResult> = states
            .par_iter()
            .map(|mem| {
                // Check cache first
                if let Some(cached) = cache.lookup(mem) {
                    return cached;
                }

                // Evaluate
                let result = eval_fn(mem);

                // Store in cache
                cache.store(mem, result.clone());

                result
            })
            .collect();

        self.stats.wall_time += start.elapsed();
        self.stats.epochs_evaluated += results.len();
        self.stats.parallel_epochs += results.len();

        results
    }

    /// Evaluate sequentially (fallback when parallel feature disabled).
    #[cfg(not(feature = "parallel"))]
    pub fn evaluate_parallel<F>(&mut self, states: Vec<Memory>, eval_fn: F) -> Vec<MemoizedResult>
    where
        F: Fn(&Memory) -> MemoizedResult,
    {
        let start = Instant::now();

        let results: Vec<MemoizedResult> = states
            .iter()
            .map(|mem| {
                // Check cache first
                if let Some(cached) = self.cache.lookup(mem) {
                    return cached;
                }

                // Evaluate
                let result = eval_fn(mem);

                // Store in cache
                self.cache.store(mem, result.clone());

                result
            })
            .collect();

        self.stats.wall_time += start.elapsed();
        self.stats.epochs_evaluated += results.len();

        results
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = ParallelStats::default();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Fixed Point Acceleration
// ═══════════════════════════════════════════════════════════════════════════

/// Accelerator for faster fixed-point convergence.
///
/// Uses techniques like:
/// - Aitken's delta-squared process
/// - Anderson mixing
/// - Memorized partial results
#[derive(Debug)]
pub struct FixedPointAccelerator {
    /// History of memory states.
    history: VecDeque<Memory>,
    /// Maximum history depth.
    max_history: usize,
    /// Acceleration method.
    method: AccelerationMethod,
    /// Statistics.
    stats: AcceleratorStats,
}

/// Acceleration methods for fixed-point computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelerationMethod {
    /// No acceleration (standard iteration).
    None,
    /// Aitken's delta-squared process.
    Aitken,
    /// Anderson mixing (m-step).
    Anderson(usize),
    /// Adaptive switching between methods.
    Adaptive,
}

impl Default for AccelerationMethod {
    fn default() -> Self {
        AccelerationMethod::Adaptive
    }
}

/// Statistics for acceleration.
#[derive(Debug, Clone, Default)]
pub struct AcceleratorStats {
    /// Iterations without acceleration.
    pub base_iterations: usize,
    /// Iterations saved by acceleration.
    pub iterations_saved: usize,
    /// Number of acceleration steps applied.
    pub acceleration_steps: usize,
    /// Current method being used.
    pub current_method: String,
}

impl Default for FixedPointAccelerator {
    fn default() -> Self {
        Self::new()
    }
}

impl FixedPointAccelerator {
    /// Create a new accelerator.
    pub fn new() -> Self {
        Self::with_method(AccelerationMethod::Adaptive)
    }

    /// Create with specific method.
    pub fn with_method(method: AccelerationMethod) -> Self {
        let max_history = match method {
            AccelerationMethod::None => 1,
            AccelerationMethod::Aitken => 3,
            AccelerationMethod::Anderson(m) => m + 2,
            AccelerationMethod::Adaptive => 5,
        };

        Self {
            history: VecDeque::with_capacity(max_history),
            max_history,
            method,
            stats: AcceleratorStats::default(),
        }
    }

    /// Record a new iteration.
    pub fn record(&mut self, memory: &Memory) {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(memory.clone());
        self.stats.base_iterations += 1;
    }

    /// Try to compute an accelerated extrapolation.
    pub fn accelerate(&mut self) -> Option<Memory> {
        match self.method {
            AccelerationMethod::None => None,
            AccelerationMethod::Aitken => self.aitken_accelerate(),
            AccelerationMethod::Anderson(m) => self.anderson_accelerate(m),
            AccelerationMethod::Adaptive => self.adaptive_accelerate(),
        }
    }

    /// Aitken's delta-squared acceleration.
    fn aitken_accelerate(&mut self) -> Option<Memory> {
        if self.history.len() < 3 {
            return None;
        }

        let len = self.history.len();
        let x0 = &self.history[len - 3];
        let x1 = &self.history[len - 2];
        let x2 = &self.history[len - 1];

        // Apply Aitken's formula to each cell
        let mut result = x2.clone();

        for addr in 0..MEMORY_SIZE as u16 {
            let v0 = x0.read(addr).val as f64;
            let v1 = x1.read(addr).val as f64;
            let v2 = x2.read(addr).val as f64;

            let delta1 = v1 - v0;
            let delta2 = v2 - v1;
            let delta_delta = delta2 - delta1;

            if delta_delta.abs() > 1e-10 {
                let accelerated = v2 - (delta2 * delta2) / delta_delta;
                if accelerated.is_finite() && accelerated >= 0.0 {
                    result.write(addr, Value::new(accelerated as u64));
                }
            }
        }

        self.stats.acceleration_steps += 1;
        self.stats.current_method = "Aitken".to_string();
        Some(result)
    }

    /// Anderson mixing acceleration.
    fn anderson_accelerate(&mut self, m: usize) -> Option<Memory> {
        if self.history.len() < m + 1 {
            return None;
        }

        // Simple Anderson mixing: use weighted average of recent iterations
        let weights: Vec<f64> = (0..=m).map(|i| {
            1.0 / (m + 1) as f64 * (i + 1) as f64
        }).collect();

        let mut result = Memory::new();

        for addr in 0..MEMORY_SIZE as u16 {
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for (i, weight) in weights.iter().enumerate() {
                if i < self.history.len() {
                    let val = self.history[self.history.len() - 1 - i].read(addr).val as f64;
                    weighted_sum += weight * val;
                    weight_sum += weight;
                }
            }

            if weight_sum > 0.0 {
                result.write(addr, Value::new((weighted_sum / weight_sum) as u64));
            }
        }

        self.stats.acceleration_steps += 1;
        self.stats.current_method = format!("Anderson({})", m);
        Some(result)
    }

    /// Adaptive acceleration - switches methods based on convergence pattern.
    fn adaptive_accelerate(&mut self) -> Option<Memory> {
        if self.history.len() < 3 {
            return None;
        }

        // Try Aitken first
        let aitken_result = self.aitken_accelerate();
        if aitken_result.is_some() {
            return aitken_result;
        }

        // Fall back to Anderson
        self.anderson_accelerate(2)
    }

    /// Get statistics.
    pub fn stats(&self) -> &AcceleratorStats {
        &self.stats
    }

    /// Reset accelerator state.
    pub fn reset(&mut self) {
        self.history.clear();
        self.stats = AcceleratorStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epoch_cache() {
        let mut cache = EpochCache::new();

        let result = MemoizedResult::simple(
            Memory::new(),
            vec![OutputItem::Val(Value::new(42))],
            EpochStatus::Finished,
        );

        cache.insert(12345, result.clone());

        assert!(cache.contains(12345));
        assert!(!cache.contains(99999));

        let retrieved = cache.get(12345);
        assert!(retrieved.is_some());

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
    }

    #[test]
    fn test_epoch_cache_lru() {
        let mut cache = EpochCache::with_capacity(2);

        for i in 0..5u64 {
            let mut memory = Memory::new();
            memory.write(0, Value::new(i));
            let result = MemoizedResult::simple(memory.clone(), vec![], EpochStatus::Finished);
            cache.store(&memory, result);
        }

        assert_eq!(cache.entries.len(), 2);
        assert!(cache.stats().evictions > 0);
    }

    #[test]
    fn test_delta_tracker() {
        let mut tracker = DeltaTracker::new();

        let mut old = Memory::new();
        old.write(0, Value::new(10));
        old.write(1, Value::new(20));

        let mut new = Memory::new();
        new.write(0, Value::new(10));  // Same
        new.write(1, Value::new(30));  // Changed
        new.write(2, Value::new(40));  // New

        let delta = tracker.compute_delta(&old, &new);

        // Should detect changes
        assert!(!delta.is_empty());
    }

    #[test]
    fn test_delta_tracker_convergence() {
        let mut tracker = DeltaTracker::with_threshold(2);
        let memory = Memory::new();

        assert!(!tracker.check_convergence(&memory));
        assert!(!tracker.check_convergence(&memory));
        assert!(tracker.check_convergence(&memory)); // Third time converges
    }

    #[test]
    fn test_speculative_executor() {
        let mut spec = SpeculativeExecutor::new();
        let memory = Memory::new();

        spec.start(memory.clone());
        assert!(spec.has_pending());

        let branch = spec.next_branch().unwrap();
        assert_eq!(branch.id, 0);
        assert_eq!(branch.depth, 0);

        // Create child branch
        let child_id = spec.branch(&branch, vec![(0, 42)]);
        assert!(child_id.is_some());
        assert_eq!(spec.stats().branches_created, 2);
    }

    #[test]
    fn test_shared_cache() {
        let cache = SharedEpochCache::new();
        let memory = Memory::new();
        let result = MemoizedResult::simple(memory.clone(), vec![], EpochStatus::Finished);

        cache.store(&memory, result);
        assert!(cache.lookup(&memory).is_some());
    }

    #[test]
    fn test_accelerator_aitken() {
        let mut accel = FixedPointAccelerator::with_method(AccelerationMethod::Aitken);

        // Record three iterations with converging sequence
        let mut m1 = Memory::new();
        m1.write(0, Value::new(100));
        accel.record(&m1);

        let mut m2 = Memory::new();
        m2.write(0, Value::new(110));
        accel.record(&m2);

        let mut m3 = Memory::new();
        m3.write(0, Value::new(115));
        accel.record(&m3);

        // Should be able to accelerate
        let accelerated = accel.accelerate();
        assert!(accelerated.is_some());
    }

    #[test]
    fn test_cache_stats_display() {
        let mut cache = EpochCache::new();
        let memory = Memory::new();
        let result = MemoizedResult::simple(memory.clone(), vec![], EpochStatus::Finished);

        cache.store(&memory, result);
        cache.lookup(&memory);
        cache.lookup(&memory);

        let s = format!("{}", cache.stats());
        assert!(s.contains("hits"));
    }

    #[test]
    fn test_parallel_config_default() {
        let config = ParallelConfig::default();
        assert!(config.num_workers >= 1);
        assert!(config.chunk_size > 0);
    }

    #[test]
    fn test_memoized_result_builder() {
        let memory = Memory::new();
        let result = MemoizedResult::simple(memory, vec![], EpochStatus::Finished)
            .with_compute_time(Duration::from_millis(100))
            .with_instruction_count(500);

        assert_eq!(result.compute_time, Duration::from_millis(100));
        assert_eq!(result.instruction_count, 500);
    }

    #[test]
    fn test_cache_invalidation() {
        let mut cache = EpochCache::new();
        let memory = Memory::new();
        let result = MemoizedResult::simple(memory.clone(), vec![], EpochStatus::Finished);

        cache.store(&memory, result);
        assert!(cache.lookup(&memory).is_some());

        cache.invalidate(&memory);
        assert!(cache.lookup(&memory).is_none());
        assert!(cache.stats().invalidations > 0);
    }
}
