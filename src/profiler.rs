//! Profiler for OUROCHRONOS execution analysis.
//!
//! This module provides comprehensive profiling capabilities:
//! - Execution timing (per-epoch, per-instruction)
//! - Memory usage tracking
//! - Hotspot detection
//! - Temporal operation statistics
//! - Profile-guided optimization support
//!
//! # Usage
//!
//! ```ignore
//! let mut profiler = Profiler::new();
//! profiler.start_epoch(0);
//! // ... execute epoch ...
//! profiler.end_epoch(0, &result);
//! println!("{}", profiler.summary());
//!
//! // Extract profile data for optimization
//! let profile_data = profiler.to_profile_data();
//! let optimizer = Optimizer::with_profile(OptLevel::Aggressive, profile_data);
//! ```

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use std::fmt;

use crate::ast::OpCode;
use crate::core_types::Address;
use crate::optimizer::ProfileData;

/// Configuration for the profiler.
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Track per-instruction timing (more overhead).
    pub track_instructions: bool,
    /// Track memory access patterns.
    pub track_memory: bool,
    /// Track temporal operation statistics.
    pub track_temporal: bool,
    /// Maximum epochs to track in detail (0 = unlimited).
    pub max_epoch_history: usize,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            track_instructions: true,
            track_memory: true,
            track_temporal: true,
            max_epoch_history: 100,
        }
    }
}

impl ProfilerConfig {
    /// Minimal profiling (low overhead).
    pub fn minimal() -> Self {
        Self {
            track_instructions: false,
            track_memory: false,
            track_temporal: true,
            max_epoch_history: 10,
        }
    }

    /// Full profiling (higher overhead).
    pub fn full() -> Self {
        Self {
            track_instructions: true,
            track_memory: true,
            track_temporal: true,
            max_epoch_history: 1000,
        }
    }
}

/// Statistics for a single epoch.
#[derive(Debug, Clone)]
pub struct EpochProfile {
    /// Epoch number.
    pub epoch: usize,
    /// Wall-clock execution time.
    pub duration: Duration,
    /// Number of instructions executed.
    pub instruction_count: u64,
    /// Number of ORACLE operations.
    pub oracle_count: u64,
    /// Number of PROPHECY operations.
    pub prophecy_count: u64,
    /// Memory cells read.
    pub memory_reads: u64,
    /// Memory cells written.
    pub memory_writes: u64,
    /// Peak stack depth during this epoch.
    pub peak_stack_depth: usize,
    /// Whether this epoch terminated in paradox.
    pub paradox: bool,
}

impl EpochProfile {
    fn new(epoch: usize) -> Self {
        Self {
            epoch,
            duration: Duration::ZERO,
            instruction_count: 0,
            oracle_count: 0,
            prophecy_count: 0,
            memory_reads: 0,
            memory_writes: 0,
            peak_stack_depth: 0,
            paradox: false,
        }
    }
}

/// Per-instruction timing statistics.
#[derive(Debug, Clone, Default)]
pub struct InstructionStats {
    /// Total count of this instruction.
    pub count: u64,
    /// Total time spent in this instruction.
    pub total_time: Duration,
}

impl InstructionStats {
    /// Average time per instruction.
    pub fn avg_time(&self) -> Duration {
        if self.count == 0 {
            Duration::ZERO
        } else {
            self.total_time / self.count as u32
        }
    }
}

/// Memory access pattern statistics.
#[derive(Debug, Clone, Default)]
pub struct MemoryProfile {
    /// Read count per address.
    pub reads: HashMap<Address, u64>,
    /// Write count per address.
    pub writes: HashMap<Address, u64>,
    /// Oracle reads per address.
    pub oracle_reads: HashMap<Address, u64>,
    /// Prophecy writes per address.
    pub prophecy_writes: HashMap<Address, u64>,
}

impl MemoryProfile {
    /// Get the most frequently accessed addresses.
    pub fn hotspots(&self, top_n: usize) -> Vec<(Address, u64)> {
        let mut access_counts: HashMap<Address, u64> = HashMap::new();

        for (&addr, &count) in &self.reads {
            *access_counts.entry(addr).or_insert(0) += count;
        }
        for (&addr, &count) in &self.writes {
            *access_counts.entry(addr).or_insert(0) += count;
        }

        let mut sorted: Vec<_> = access_counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.truncate(top_n);
        sorted
    }

    /// Get addresses involved in temporal operations.
    pub fn temporal_hotspots(&self, top_n: usize) -> Vec<(Address, u64)> {
        let mut temporal_counts: HashMap<Address, u64> = HashMap::new();

        for (&addr, &count) in &self.oracle_reads {
            *temporal_counts.entry(addr).or_insert(0) += count;
        }
        for (&addr, &count) in &self.prophecy_writes {
            *temporal_counts.entry(addr).or_insert(0) += count;
        }

        let mut sorted: Vec<_> = temporal_counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.truncate(top_n);
        sorted
    }
}

/// Temporal operation statistics.
#[derive(Debug, Clone, Default)]
pub struct TemporalStats {
    /// Total ORACLE operations.
    pub total_oracle: u64,
    /// Total PROPHECY operations.
    pub total_prophecy: u64,
    /// Total epochs executed.
    pub total_epochs: usize,
    /// Epochs that terminated in paradox.
    pub paradox_epochs: usize,
    /// Epochs until convergence (if achieved).
    pub convergence_epoch: Option<usize>,
    /// Time spent in temporal operations.
    pub temporal_time: Duration,
}

impl TemporalStats {
    /// Convergence rate (epochs to convergence / total epochs).
    pub fn convergence_rate(&self) -> Option<f64> {
        self.convergence_epoch.map(|e| e as f64 / self.total_epochs.max(1) as f64)
    }

    /// Paradox rate (paradox epochs / total epochs).
    pub fn paradox_rate(&self) -> f64 {
        self.paradox_epochs as f64 / self.total_epochs.max(1) as f64
    }
}

/// Main profiler for OUROCHRONOS execution.
#[derive(Debug, Clone)]
pub struct Profiler {
    /// Configuration.
    pub config: ProfilerConfig,
    /// Start time of profiling session.
    start_time: Option<Instant>,
    /// Current epoch being profiled.
    current_epoch: Option<EpochProfile>,
    /// Current epoch start time.
    current_epoch_start: Option<Instant>,
    /// Epoch profiles (history).
    epoch_profiles: Vec<EpochProfile>,
    /// Per-instruction statistics.
    instruction_stats: HashMap<OpCode, InstructionStats>,
    /// Memory access profile.
    memory_profile: MemoryProfile,
    /// Temporal statistics.
    temporal_stats: TemporalStats,
    /// Total execution time.
    total_time: Duration,
    // Profile-Guided Optimization fields
    /// Procedure call counts for hot procedure detection.
    proc_call_counts: HashMap<String, usize>,
    /// Loop iteration counts for loop unrolling decisions.
    loop_iteration_counts: HashMap<usize, Vec<usize>>,
    /// Branch taken/not-taken counts for branch prediction.
    branch_counts: HashMap<usize, (usize, usize)>,
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Profiler {
    /// Create a new profiler with default configuration.
    pub fn new() -> Self {
        Self::with_config(ProfilerConfig::default())
    }

    /// Create a profiler with custom configuration.
    pub fn with_config(config: ProfilerConfig) -> Self {
        Self {
            config,
            start_time: None,
            current_epoch: None,
            current_epoch_start: None,
            epoch_profiles: Vec::new(),
            instruction_stats: HashMap::new(),
            memory_profile: MemoryProfile::default(),
            temporal_stats: TemporalStats::default(),
            total_time: Duration::ZERO,
            proc_call_counts: HashMap::new(),
            loop_iteration_counts: HashMap::new(),
            branch_counts: HashMap::new(),
        }
    }

    /// Start profiling session.
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Stop profiling session.
    pub fn stop(&mut self) {
        if let Some(start) = self.start_time.take() {
            self.total_time = start.elapsed();
        }
    }

    /// Begin profiling an epoch.
    pub fn start_epoch(&mut self, epoch: usize) {
        self.current_epoch = Some(EpochProfile::new(epoch));
        self.current_epoch_start = Some(Instant::now());
    }

    /// End profiling an epoch.
    pub fn end_epoch(&mut self, instruction_count: u64, paradox: bool) {
        if let Some(start) = self.current_epoch_start.take() {
            if let Some(mut profile) = self.current_epoch.take() {
                profile.duration = start.elapsed();
                profile.instruction_count = instruction_count;
                profile.paradox = paradox;

                // Update temporal stats
                self.temporal_stats.total_epochs += 1;
                if paradox {
                    self.temporal_stats.paradox_epochs += 1;
                }
                self.temporal_stats.total_oracle += profile.oracle_count;
                self.temporal_stats.total_prophecy += profile.prophecy_count;

                // Keep history
                if self.config.max_epoch_history == 0
                    || self.epoch_profiles.len() < self.config.max_epoch_history {
                    self.epoch_profiles.push(profile);
                }
            }
        }
    }

    /// Record an instruction execution.
    pub fn record_instruction(&mut self, op: OpCode, duration: Duration) {
        if self.config.track_instructions {
            let stats = self.instruction_stats.entry(op).or_default();
            stats.count += 1;
            stats.total_time += duration;
        }

        // Track temporal operations
        if self.config.track_temporal {
            if let Some(ref mut epoch) = self.current_epoch {
                match op {
                    OpCode::Oracle => epoch.oracle_count += 1,
                    OpCode::Prophecy => epoch.prophecy_count += 1,
                    _ => {}
                }
            }
        }
    }

    /// Record a memory read.
    pub fn record_memory_read(&mut self, addr: Address) {
        if self.config.track_memory {
            *self.memory_profile.reads.entry(addr).or_insert(0) += 1;
            if let Some(ref mut epoch) = self.current_epoch {
                epoch.memory_reads += 1;
            }
        }
    }

    /// Record a memory write.
    pub fn record_memory_write(&mut self, addr: Address) {
        if self.config.track_memory {
            *self.memory_profile.writes.entry(addr).or_insert(0) += 1;
            if let Some(ref mut epoch) = self.current_epoch {
                epoch.memory_writes += 1;
            }
        }
    }

    /// Record an oracle read.
    pub fn record_oracle(&mut self, addr: Address) {
        if self.config.track_temporal {
            *self.memory_profile.oracle_reads.entry(addr).or_insert(0) += 1;
        }
    }

    /// Record a prophecy write.
    pub fn record_prophecy(&mut self, addr: Address) {
        if self.config.track_temporal {
            *self.memory_profile.prophecy_writes.entry(addr).or_insert(0) += 1;
        }
    }

    /// Record peak stack depth.
    pub fn record_stack_depth(&mut self, depth: usize) {
        if let Some(ref mut epoch) = self.current_epoch {
            epoch.peak_stack_depth = epoch.peak_stack_depth.max(depth);
        }
    }

    /// Mark convergence achieved.
    pub fn mark_convergence(&mut self, epoch: usize) {
        self.temporal_stats.convergence_epoch = Some(epoch);
    }

    /// Get epoch profiles.
    pub fn epochs(&self) -> &[EpochProfile] {
        &self.epoch_profiles
    }

    /// Get instruction statistics.
    pub fn instruction_stats(&self) -> &HashMap<OpCode, InstructionStats> {
        &self.instruction_stats
    }

    /// Get memory profile.
    pub fn memory_profile(&self) -> &MemoryProfile {
        &self.memory_profile
    }

    /// Get temporal statistics.
    pub fn temporal_stats(&self) -> &TemporalStats {
        &self.temporal_stats
    }

    /// Get total execution time.
    pub fn total_time(&self) -> Duration {
        self.total_time
    }

    /// Get instruction hotspots (most frequently executed).
    pub fn instruction_hotspots(&self, top_n: usize) -> Vec<(OpCode, &InstructionStats)> {
        let mut sorted: Vec<_> = self.instruction_stats.iter().collect();
        sorted.sort_by(|a, b| b.1.count.cmp(&a.1.count));
        sorted.truncate(top_n);
        sorted.into_iter().map(|(op, stats)| (*op, stats)).collect()
    }

    /// Get slowest instructions (by total time).
    pub fn slowest_instructions(&self, top_n: usize) -> Vec<(OpCode, &InstructionStats)> {
        let mut sorted: Vec<_> = self.instruction_stats.iter().collect();
        sorted.sort_by(|a, b| b.1.total_time.cmp(&a.1.total_time));
        sorted.truncate(top_n);
        sorted.into_iter().map(|(op, stats)| (*op, stats)).collect()
    }

    /// Average instructions per epoch.
    pub fn avg_instructions_per_epoch(&self) -> f64 {
        if self.epoch_profiles.is_empty() {
            0.0
        } else {
            self.epoch_profiles.iter().map(|e| e.instruction_count).sum::<u64>() as f64
                / self.epoch_profiles.len() as f64
        }
    }

    /// Average epoch duration.
    pub fn avg_epoch_duration(&self) -> Duration {
        if self.epoch_profiles.is_empty() {
            Duration::ZERO
        } else {
            let total: Duration = self.epoch_profiles.iter().map(|e| e.duration).sum();
            total / self.epoch_profiles.len() as u32
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Profile-Guided Optimization Methods
    // ═══════════════════════════════════════════════════════════════════════

    /// Record a procedure call for hot procedure detection.
    pub fn record_proc_call(&mut self, name: &str) {
        *self.proc_call_counts.entry(name.to_string()).or_insert(0) += 1;
    }

    /// Record loop iteration count.
    pub fn record_loop_iteration(&mut self, loop_id: usize, iteration_count: usize) {
        self.loop_iteration_counts
            .entry(loop_id)
            .or_insert_with(Vec::new)
            .push(iteration_count);
    }

    /// Record branch outcome (taken vs not-taken).
    pub fn record_branch(&mut self, branch_id: usize, taken: bool) {
        let entry = self.branch_counts.entry(branch_id).or_insert((0, 0));
        if taken {
            entry.0 += 1;
        } else {
            entry.1 += 1;
        }
    }

    /// Get hot procedures (above threshold).
    pub fn hot_procedures(&self, threshold: usize) -> Vec<&str> {
        self.proc_call_counts
            .iter()
            .filter(|(_, &count)| count >= threshold)
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Get branch probability (taken / total).
    pub fn branch_probability(&self, branch_id: usize) -> Option<f64> {
        self.branch_counts.get(&branch_id).map(|&(taken, not_taken)| {
            let total = taken + not_taken;
            if total == 0 { 0.5 } else { taken as f64 / total as f64 }
        })
    }

    /// Get average loop iteration count.
    pub fn avg_loop_iterations(&self, loop_id: usize) -> Option<f64> {
        self.loop_iteration_counts.get(&loop_id).map(|counts| {
            if counts.is_empty() {
                0.0
            } else {
                counts.iter().sum::<usize>() as f64 / counts.len() as f64
            }
        })
    }

    /// Convert profiler data to ProfileData for optimizer.
    pub fn to_profile_data(&self) -> ProfileData {
        let mut data = ProfileData::default();

        // Extract hot procedures (called > 100 times)
        for (name, &count) in &self.proc_call_counts {
            if count > 100 {
                data.hot_procs.insert(name.clone());
            }
        }

        // Extract hot loops (avg iterations > 10)
        for (&loop_id, counts) in &self.loop_iteration_counts {
            if !counts.is_empty() {
                let avg = counts.iter().sum::<usize>() / counts.len();
                if avg > 10 {
                    data.hot_loops.insert(loop_id, avg);
                }
            }
        }

        // Extract branch probabilities
        for (&branch_id, &(taken, not_taken)) in &self.branch_counts {
            let total = taken + not_taken;
            if total > 0 {
                data.branch_probs.insert(branch_id, taken as f64 / total as f64);
            }
        }

        // Extract memory hotspots
        for (addr, _) in self.memory_profile.hotspots(20) {
            data.memory_hotspots.insert(addr);
        }

        data
    }

    /// Check if a procedure is hot (frequently called).
    pub fn is_hot_proc(&self, name: &str, threshold: usize) -> bool {
        self.proc_call_counts.get(name).map(|&c| c >= threshold).unwrap_or(false)
    }

    /// Get procedure call count.
    pub fn proc_call_count(&self, name: &str) -> usize {
        *self.proc_call_counts.get(name).unwrap_or(&0)
    }

    /// Generate optimization hints based on profile data.
    pub fn optimization_hints(&self) -> OptimizationHints {
        OptimizationHints {
            hot_procedures: self.hot_procedures(100).iter().map(|s| s.to_string()).collect(),
            branch_predictions: self.branch_counts.iter()
                .filter_map(|(&id, &(taken, not_taken))| {
                    let total = taken + not_taken;
                    if total >= 10 {
                        let prob = taken as f64 / total as f64;
                        if prob > 0.8 || prob < 0.2 {
                            Some((id, prob > 0.5))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect(),
            unroll_candidates: self.loop_iteration_counts.iter()
                .filter_map(|(&id, counts)| {
                    if counts.len() >= 5 {
                        let avg = counts.iter().sum::<usize>() as f64 / counts.len() as f64;
                        let variance: f64 = counts.iter()
                            .map(|&c| (c as f64 - avg).powi(2))
                            .sum::<f64>() / counts.len() as f64;
                        // Low variance and small iteration count = good unroll candidate
                        if variance.sqrt() < 2.0 && avg < 16.0 && avg > 0.0 {
                            Some((id, avg as usize))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect(),
            memory_hotspots: self.memory_profile.hotspots(10)
                .into_iter()
                .map(|(addr, _)| addr)
                .collect(),
        }
    }

    /// Generate a summary report.
    pub fn summary(&self) -> ProfileSummary<'_> {
        ProfileSummary {
            total_time: self.total_time,
            total_epochs: self.temporal_stats.total_epochs,
            paradox_epochs: self.temporal_stats.paradox_epochs,
            convergence_epoch: self.temporal_stats.convergence_epoch,
            total_instructions: self.instruction_stats.values().map(|s| s.count).sum(),
            total_oracle: self.temporal_stats.total_oracle,
            total_prophecy: self.temporal_stats.total_prophecy,
            avg_epoch_duration: self.avg_epoch_duration(),
            avg_instructions_per_epoch: self.avg_instructions_per_epoch(),
            instruction_hotspots: self.instruction_hotspots(5),
            memory_hotspots: self.memory_profile.hotspots(5),
            temporal_hotspots: self.memory_profile.temporal_hotspots(5),
        }
    }

    /// Reset the profiler.
    pub fn reset(&mut self) {
        self.start_time = None;
        self.current_epoch = None;
        self.current_epoch_start = None;
        self.epoch_profiles.clear();
        self.instruction_stats.clear();
        self.memory_profile = MemoryProfile::default();
        self.temporal_stats = TemporalStats::default();
        self.total_time = Duration::ZERO;
        self.proc_call_counts.clear();
        self.loop_iteration_counts.clear();
        self.branch_counts.clear();
    }
}

/// Optimization hints derived from profile data.
#[derive(Debug, Clone, Default)]
pub struct OptimizationHints {
    /// Hot procedures that should be aggressively optimized.
    pub hot_procedures: Vec<String>,
    /// Branch predictions: (branch_id, predicted_taken).
    pub branch_predictions: Vec<(usize, bool)>,
    /// Loop unroll candidates: (loop_id, recommended_iterations).
    pub unroll_candidates: Vec<(usize, usize)>,
    /// Memory hotspots to optimize access for.
    pub memory_hotspots: Vec<Address>,
}

impl OptimizationHints {
    /// Check if procedure is hot.
    pub fn is_hot(&self, name: &str) -> bool {
        self.hot_procedures.iter().any(|p| p == name)
    }

    /// Get branch prediction for a branch.
    pub fn predict_branch(&self, branch_id: usize) -> Option<bool> {
        self.branch_predictions.iter()
            .find(|(id, _)| *id == branch_id)
            .map(|(_, pred)| *pred)
    }

    /// Get recommended unroll count for a loop.
    pub fn unroll_count(&self, loop_id: usize) -> Option<usize> {
        self.unroll_candidates.iter()
            .find(|(id, _)| *id == loop_id)
            .map(|(_, count)| *count)
    }

    /// Check if memory address is a hotspot.
    pub fn is_memory_hotspot(&self, addr: Address) -> bool {
        self.memory_hotspots.contains(&addr)
    }
}

impl fmt::Display for OptimizationHints {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "═══════════════════════════════════════════════════════")?;
        writeln!(f, "              OPTIMIZATION HINTS                        ")?;
        writeln!(f, "═══════════════════════════════════════════════════════")?;

        if !self.hot_procedures.is_empty() {
            writeln!(f, "Hot Procedures (JIT candidates):")?;
            for proc in &self.hot_procedures {
                writeln!(f, "  - {}", proc)?;
            }
            writeln!(f)?;
        }

        if !self.branch_predictions.is_empty() {
            writeln!(f, "Branch Predictions:")?;
            for (id, taken) in &self.branch_predictions {
                writeln!(f, "  Branch {}: predict {}", id, if *taken { "TAKEN" } else { "NOT_TAKEN" })?;
            }
            writeln!(f)?;
        }

        if !self.unroll_candidates.is_empty() {
            writeln!(f, "Loop Unroll Candidates:")?;
            for (id, count) in &self.unroll_candidates {
                writeln!(f, "  Loop {}: unroll {} times", id, count)?;
            }
            writeln!(f)?;
        }

        if !self.memory_hotspots.is_empty() {
            writeln!(f, "Memory Hotspots:")?;
            for addr in &self.memory_hotspots {
                writeln!(f, "  Address {}", addr)?;
            }
        }

        writeln!(f, "═══════════════════════════════════════════════════════")
    }
}

/// Summary of profiling results.
pub struct ProfileSummary<'a> {
    pub total_time: Duration,
    pub total_epochs: usize,
    pub paradox_epochs: usize,
    pub convergence_epoch: Option<usize>,
    pub total_instructions: u64,
    pub total_oracle: u64,
    pub total_prophecy: u64,
    pub avg_epoch_duration: Duration,
    pub avg_instructions_per_epoch: f64,
    pub instruction_hotspots: Vec<(OpCode, &'a InstructionStats)>,
    pub memory_hotspots: Vec<(Address, u64)>,
    pub temporal_hotspots: Vec<(Address, u64)>,
}

impl<'a> fmt::Display for ProfileSummary<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "═══════════════════════════════════════════════════════")?;
        writeln!(f, "                    PROFILE SUMMARY                     ")?;
        writeln!(f, "═══════════════════════════════════════════════════════")?;
        writeln!(f)?;

        writeln!(f, "EXECUTION TIME")?;
        writeln!(f, "  Total:          {:?}", self.total_time)?;
        writeln!(f, "  Avg per epoch:  {:?}", self.avg_epoch_duration)?;
        writeln!(f)?;

        writeln!(f, "EPOCHS")?;
        writeln!(f, "  Total:          {}", self.total_epochs)?;
        writeln!(f, "  Paradox:        {} ({:.1}%)",
                 self.paradox_epochs,
                 self.paradox_epochs as f64 / self.total_epochs.max(1) as f64 * 100.0)?;
        if let Some(conv) = self.convergence_epoch {
            writeln!(f, "  Convergence:    epoch {}", conv)?;
        }
        writeln!(f)?;

        writeln!(f, "INSTRUCTIONS")?;
        writeln!(f, "  Total:          {}", self.total_instructions)?;
        writeln!(f, "  Avg per epoch:  {:.1}", self.avg_instructions_per_epoch)?;
        writeln!(f)?;

        writeln!(f, "TEMPORAL OPERATIONS")?;
        writeln!(f, "  ORACLE:         {}", self.total_oracle)?;
        writeln!(f, "  PROPHECY:       {}", self.total_prophecy)?;
        writeln!(f)?;

        if !self.instruction_hotspots.is_empty() {
            writeln!(f, "INSTRUCTION HOTSPOTS (by count)")?;
            for (op, stats) in &self.instruction_hotspots {
                writeln!(f, "  {:15} {:>10} calls  ({:?} total)",
                         op.name(), stats.count, stats.total_time)?;
            }
            writeln!(f)?;
        }

        if !self.memory_hotspots.is_empty() {
            writeln!(f, "MEMORY HOTSPOTS (by access count)")?;
            for (addr, count) in &self.memory_hotspots {
                writeln!(f, "  Address {:5}: {:>10} accesses", addr, count)?;
            }
            writeln!(f)?;
        }

        if !self.temporal_hotspots.is_empty() {
            writeln!(f, "TEMPORAL HOTSPOTS (oracle/prophecy)")?;
            for (addr, count) in &self.temporal_hotspots {
                writeln!(f, "  Address {:5}: {:>10} temporal ops", addr, count)?;
            }
        }

        writeln!(f, "═══════════════════════════════════════════════════════")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn test_profiler_creation() {
        let profiler = Profiler::new();
        assert_eq!(profiler.temporal_stats().total_epochs, 0);
    }

    #[test]
    fn test_profiler_config() {
        let minimal = ProfilerConfig::minimal();
        assert!(!minimal.track_instructions);

        let full = ProfilerConfig::full();
        assert!(full.track_instructions);
    }

    #[test]
    fn test_epoch_profiling() {
        let mut profiler = Profiler::new();
        profiler.start();

        profiler.start_epoch(0);
        sleep(Duration::from_millis(1));
        profiler.end_epoch(100, false);

        profiler.start_epoch(1);
        profiler.end_epoch(50, true);

        profiler.stop();

        assert_eq!(profiler.epochs().len(), 2);
        assert_eq!(profiler.temporal_stats().total_epochs, 2);
        assert_eq!(profiler.temporal_stats().paradox_epochs, 1);
    }

    #[test]
    fn test_instruction_recording() {
        let mut profiler = Profiler::new();

        profiler.record_instruction(OpCode::Add, Duration::from_nanos(100));
        profiler.record_instruction(OpCode::Add, Duration::from_nanos(150));
        profiler.record_instruction(OpCode::Mul, Duration::from_nanos(200));

        let stats = profiler.instruction_stats();
        assert_eq!(stats.get(&OpCode::Add).unwrap().count, 2);
        assert_eq!(stats.get(&OpCode::Mul).unwrap().count, 1);
    }

    #[test]
    fn test_memory_tracking() {
        let mut profiler = Profiler::new();

        profiler.record_memory_read(0);
        profiler.record_memory_read(0);
        profiler.record_memory_write(0);
        profiler.record_memory_read(1);

        let profile = profiler.memory_profile();
        assert_eq!(*profile.reads.get(&0).unwrap(), 2);
        assert_eq!(*profile.writes.get(&0).unwrap(), 1);
        assert_eq!(*profile.reads.get(&1).unwrap(), 1);
    }

    #[test]
    fn test_temporal_tracking() {
        let mut profiler = Profiler::new();
        profiler.start_epoch(0);

        // record_instruction updates epoch oracle/prophecy counts
        profiler.record_instruction(OpCode::Oracle, Duration::from_nanos(100));
        profiler.record_instruction(OpCode::Oracle, Duration::from_nanos(100));
        profiler.record_instruction(OpCode::Prophecy, Duration::from_nanos(100));

        // record_oracle/prophecy updates memory profile hotspots
        profiler.record_oracle(0);
        profiler.record_prophecy(0);

        profiler.end_epoch(10, false);

        let epochs = profiler.epochs();
        assert_eq!(epochs[0].oracle_count, 2);
        assert_eq!(epochs[0].prophecy_count, 1);

        // Verify memory profile too
        let profile = profiler.memory_profile();
        assert_eq!(*profile.oracle_reads.get(&0).unwrap(), 1);
        assert_eq!(*profile.prophecy_writes.get(&0).unwrap(), 1);
    }

    #[test]
    fn test_hotspots() {
        let mut profiler = Profiler::new();

        // Record some instructions
        for _ in 0..100 {
            profiler.record_instruction(OpCode::Add, Duration::from_nanos(10));
        }
        for _ in 0..50 {
            profiler.record_instruction(OpCode::Mul, Duration::from_nanos(20));
        }
        for _ in 0..25 {
            profiler.record_instruction(OpCode::Div, Duration::from_nanos(30));
        }

        let hotspots = profiler.instruction_hotspots(2);
        assert_eq!(hotspots.len(), 2);
        assert_eq!(hotspots[0].0, OpCode::Add);
        assert_eq!(hotspots[1].0, OpCode::Mul);
    }

    #[test]
    fn test_convergence() {
        let mut profiler = Profiler::new();

        profiler.start_epoch(0);
        profiler.end_epoch(10, false);
        profiler.start_epoch(1);
        profiler.end_epoch(10, false);
        profiler.mark_convergence(1);

        assert_eq!(profiler.temporal_stats().convergence_epoch, Some(1));
    }

    #[test]
    fn test_summary_display() {
        let mut profiler = Profiler::new();
        profiler.start();

        profiler.start_epoch(0);
        profiler.record_instruction(OpCode::Add, Duration::from_nanos(100));
        profiler.record_memory_read(0);
        profiler.end_epoch(10, false);

        profiler.stop();

        let summary = profiler.summary();
        let display = format!("{}", summary);

        assert!(display.contains("PROFILE SUMMARY"));
        assert!(display.contains("EPOCHS"));
    }

    #[test]
    fn test_reset() {
        let mut profiler = Profiler::new();

        profiler.start_epoch(0);
        profiler.record_instruction(OpCode::Add, Duration::from_nanos(100));
        profiler.end_epoch(10, false);

        profiler.reset();

        assert!(profiler.epochs().is_empty());
        assert!(profiler.instruction_stats().is_empty());
        assert_eq!(profiler.temporal_stats().total_epochs, 0);
    }
}
