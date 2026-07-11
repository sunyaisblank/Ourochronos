//! Epoch result cache for the fixed-point search.
//!
//! During the search for a fixed point S = F(S), the same anamnesis state can
//! recur (multi-seed exploration revisits states; oscillations revisit them by
//! definition). Caching the epoch result keyed on the anamnesis hash avoids
//! re-executing the programme for a state already evaluated.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use crate::core::{Memory, OutputItem};
use crate::vm::EpochStatus;

/// Maximum cache size before eviction.
const DEFAULT_MAX_CACHE_SIZE: usize = 1024;

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

/// Cache of epoch results keyed by anamnesis state hash.
///
/// `Memory::state_hash` is a 64-bit XOR fold, so distinct states can collide.
/// Each entry therefore stores the sparse form of the anamnesis that produced
/// it, and a lookup only hits when that stored state compares equal; serving
/// a colliding entry would hand the search a wrong next state. Bounded: when
/// full, an arbitrary entry is evicted on insert, which only forces
/// re-execution of an epoch.
#[derive(Debug)]
pub struct EpochCache {
    entries: HashMap<u64, (Vec<(u16, u64)>, MemoizedResult)>,
    max_size: usize,
    hits: usize,
    misses: usize,
    evictions: usize,
}

/// Sparse form of a memory state: its non-zero cells. Absent cells are zero,
/// so this determines the full state.
fn sparse_state(memory: &Memory) -> Vec<(u16, u64)> {
    memory
        .non_zero_cells()
        .into_iter()
        .map(|(addr, value)| (addr, value.val))
        .collect()
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
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Look up the cached result for this anamnesis. A hash match alone is
    /// not a hit; the stored state must compare equal.
    pub fn get(&mut self, state_hash: u64, anamnesis: &Memory) -> Option<&MemoizedResult> {
        match self.entries.get(&state_hash) {
            Some((stored, _)) if *stored == sparse_state(anamnesis) => {
                self.hits += 1;
                self.entries.get(&state_hash).map(|(_, result)| result)
            }
            _ => {
                self.misses += 1;
                None
            }
        }
    }

    /// Store a result for this anamnesis, evicting an arbitrary entry when full.
    pub fn insert(&mut self, state_hash: u64, anamnesis: &Memory, result: MemoizedResult) {
        if self.entries.len() >= self.max_size {
            if let Some(&key) = self.entries.keys().next() {
                self.entries.remove(&key);
                self.evictions += 1;
            }
        }
        self.entries.insert(state_hash, (sparse_state(anamnesis), result));
    }

    /// Check if a state hash has an entry (unverified; testing hook).
    pub fn contains(&self, state_hash: u64) -> bool {
        self.entries.contains_key(&state_hash)
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            size: self.entries.len(),
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            hit_rate: if self.hits + self.misses > 0 {
                self.hits as f64 / (self.hits + self.misses) as f64
            } else {
                0.0
            },
        }
    }

    /// Clear the cache and statistics.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.hits = 0;
        self.misses = 0;
        self.evictions = 0;
    }

    /// Get current cache size.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
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
    /// Hit rate (0.0 to 1.0).
    pub hit_rate: f64,
}

impl std::fmt::Display for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cache: {} entries, {}/{} hits ({:.1}%), {} evictions",
            self.size, self.hits, self.hits + self.misses, self.hit_rate * 100.0,
            self.evictions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Memory;

    fn dummy_result() -> MemoizedResult {
        MemoizedResult::simple(Memory::new(), Vec::new(), EpochStatus::Finished)
    }

    #[test]
    fn cached_result_is_returned_on_hit_and_counted() {
        let mut cache = EpochCache::new();
        let anamnesis = Memory::new();
        assert!(cache.get(42, &anamnesis).is_none());
        cache.insert(42, &anamnesis, dummy_result());
        assert!(cache.get(42, &anamnesis).is_some());
        assert!(cache.contains(42));
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.size, 1);
    }

    #[test]
    fn hash_collision_is_not_served_as_a_hit() {
        // Same hash key, different anamnesis contents: the entry must not be
        // returned, because its result belongs to a different state.
        let mut cache = EpochCache::new();
        let stored = Memory::new();
        cache.insert(42, &stored, dummy_result());

        let mut colliding = Memory::new();
        colliding.write(7, crate::core::Value::new(99));
        assert!(cache.get(42, &colliding).is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn insert_beyond_capacity_evicts_and_stays_bounded() {
        let mut cache = EpochCache::with_capacity(4);
        let anamnesis = Memory::new();
        for hash in 0..10u64 {
            cache.insert(hash, &anamnesis, dummy_result());
        }
        assert!(cache.len() <= 4);
        assert_eq!(cache.stats().evictions, 6);
    }

    #[test]
    fn clear_resets_entries_and_statistics() {
        let mut cache = EpochCache::new();
        let anamnesis = Memory::new();
        cache.insert(1, &anamnesis, dummy_result());
        let _ = cache.get(1, &anamnesis);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.stats().hits, 0);
    }

    #[test]
    fn cache_stats_display_reports_hit_rate() {
        let stats = CacheStats {
            size: 3,
            hits: 9,
            misses: 1,
            evictions: 2,
            hit_rate: 0.9,
        };
        let text = format!("{}", stats);
        assert!(text.contains("3 entries"));
        assert!(text.contains("90.0%"));
        assert!(text.contains("2 evictions"));
    }

    #[test]
    fn memoized_result_builders_set_metadata() {
        let mut cells = HashSet::new();
        cells.insert(7u16);
        let result = dummy_result()
            .with_compute_time(Duration::from_millis(5))
            .with_instruction_count(123)
            .with_modified_cells(cells.clone());
        assert_eq!(result.compute_time, Duration::from_millis(5));
        assert_eq!(result.instruction_count, 123);
        assert_eq!(result.modified_cells, cells);
    }
}
