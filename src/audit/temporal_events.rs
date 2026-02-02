//! Domain-specific temporal event types for Ourochronos audit logging.
//!
//! This module provides first-principle based primitives for auditing
//! temporal computation operations. Each event type captures the essential
//! information needed for causal analysis and convergence tracking.
//!
//! # Design Principles
//!
//! - **Primitive-Based**: Events derive from fundamental temporal operations
//! - **Causally Complete**: All information needed for dependency analysis
//! - **Numerically Precise**: No approximations in core measurements
//! - **Epoch-Scoped**: Events are tied to specific temporal iterations

use std::collections::BTreeSet;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::core::Address;

// =============================================================================
// Temporal Operation Types
// =============================================================================

/// Fundamental temporal operations in Ourochronos.
///
/// These are the primitive operations that interact with the temporal
/// memory model (Anamnesis/Present).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemporalOperation {
    /// ORACLE: Read from Anamnesis (future state from previous epoch).
    Oracle,
    /// PROPHECY: Write to Present (establishing future state).
    Prophecy,
    /// PRESENT: Read from Present (current epoch state).
    Present,
    /// PARADOX: Explicit paradox trigger.
    Paradox,
}

impl TemporalOperation {
    pub fn as_str(&self) -> &'static str {
        match self {
            TemporalOperation::Oracle => "ORACLE",
            TemporalOperation::Prophecy => "PROPHECY",
            TemporalOperation::Present => "PRESENT",
            TemporalOperation::Paradox => "PARADOX",
        }
    }

    /// Returns true if this operation reads from temporal memory.
    pub fn is_read(&self) -> bool {
        matches!(self, TemporalOperation::Oracle | TemporalOperation::Present)
    }

    /// Returns true if this operation writes to temporal memory.
    pub fn is_write(&self) -> bool {
        matches!(self, TemporalOperation::Prophecy)
    }

    /// Returns true if this operation terminates execution.
    pub fn is_terminal(&self) -> bool {
        matches!(self, TemporalOperation::Paradox)
    }
}

impl std::fmt::Display for TemporalOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// =============================================================================
// Temporal Event
// =============================================================================

/// A single temporal operation event.
///
/// Captures all information about a temporal memory access for
/// provenance tracking and causal analysis.
#[derive(Debug, Clone)]
pub struct TemporalEvent {
    /// The operation performed.
    pub operation: TemporalOperation,
    /// Memory address accessed.
    pub address: Address,
    /// Value read or written.
    pub value: u64,
    /// Previous value at address (for writes).
    pub previous_value: Option<u64>,
    /// Current epoch number.
    pub epoch: usize,
    /// Instruction index within program.
    pub instruction_index: usize,
    /// Timestamp in microseconds since epoch start.
    pub timestamp_us: u64,
    /// Whether this value is temporally tainted (depends on Oracle).
    pub is_temporal: bool,
    /// Dependency addresses (for computed values).
    pub dependencies: Option<BTreeSet<Address>>,
}

impl TemporalEvent {
    /// Create a new temporal event.
    pub fn new(
        operation: TemporalOperation,
        address: Address,
        value: u64,
        epoch: usize,
        instruction_index: usize,
    ) -> Self {
        let timestamp_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);

        Self {
            operation,
            address,
            value,
            previous_value: None,
            epoch,
            instruction_index,
            timestamp_us,
            is_temporal: false,
            dependencies: None,
        }
    }

    /// Create an ORACLE read event.
    pub fn oracle(address: Address, value: u64, epoch: usize, instruction_index: usize) -> Self {
        let mut event = Self::new(TemporalOperation::Oracle, address, value, epoch, instruction_index);
        event.is_temporal = true; // Oracle reads are always temporal
        let mut deps = BTreeSet::new();
        deps.insert(address);
        event.dependencies = Some(deps);
        event
    }

    /// Create a PROPHECY write event.
    pub fn prophecy(
        address: Address,
        value: u64,
        previous: u64,
        epoch: usize,
        instruction_index: usize,
        is_temporal: bool,
        dependencies: Option<BTreeSet<Address>>,
    ) -> Self {
        let mut event = Self::new(TemporalOperation::Prophecy, address, value, epoch, instruction_index);
        event.previous_value = Some(previous);
        event.is_temporal = is_temporal;
        event.dependencies = dependencies;
        event
    }

    /// Create a PRESENT read event.
    pub fn present(address: Address, value: u64, epoch: usize, instruction_index: usize) -> Self {
        Self::new(TemporalOperation::Present, address, value, epoch, instruction_index)
    }

    /// Create a PARADOX event.
    pub fn paradox(epoch: usize, instruction_index: usize) -> Self {
        Self::new(TemporalOperation::Paradox, 0, 0, epoch, instruction_index)
    }

    /// Format as a concise log line.
    pub fn format_line(&self) -> String {
        let deps_str = self.dependencies
            .as_ref()
            .map(|d| {
                let addrs: Vec<String> = d.iter().map(|a| a.to_string()).collect();
                format!(" deps=[{}]", addrs.join(","))
            })
            .unwrap_or_default();

        let temporal_str = if self.is_temporal { " [T]" } else { "" };

        match self.operation {
            TemporalOperation::Oracle => {
                format!(
                    "E{:04} I{:06} | {} @{} -> {}{}{}",
                    self.epoch, self.instruction_index, self.operation,
                    self.address, self.value, temporal_str, deps_str
                )
            }
            TemporalOperation::Prophecy => {
                let prev = self.previous_value.unwrap_or(0);
                format!(
                    "E{:04} I{:06} | {} @{} <- {} (was {}){}{}",
                    self.epoch, self.instruction_index, self.operation,
                    self.address, self.value, prev, temporal_str, deps_str
                )
            }
            TemporalOperation::Present => {
                format!(
                    "E{:04} I{:06} | {} @{} -> {}",
                    self.epoch, self.instruction_index, self.operation,
                    self.address, self.value
                )
            }
            TemporalOperation::Paradox => {
                format!(
                    "E{:04} I{:06} | {} triggered",
                    self.epoch, self.instruction_index, self.operation
                )
            }
        }
    }
}

// =============================================================================
// Epoch Event
// =============================================================================

/// Convergence status after an epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EpochOutcome {
    /// Epoch completed, continuing iteration.
    Continuing,
    /// Fixed point achieved (P = A).
    Converged,
    /// Oscillation detected.
    Oscillating,
    /// Divergence detected.
    Diverging,
    /// Explicit paradox triggered.
    Paradox,
    /// Runtime error occurred.
    Error,
    /// Epoch limit reached.
    Timeout,
}

impl EpochOutcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            EpochOutcome::Continuing => "CONTINUING",
            EpochOutcome::Converged => "CONVERGED",
            EpochOutcome::Oscillating => "OSCILLATING",
            EpochOutcome::Diverging => "DIVERGING",
            EpochOutcome::Paradox => "PARADOX",
            EpochOutcome::Error => "ERROR",
            EpochOutcome::Timeout => "TIMEOUT",
        }
    }

    pub fn is_terminal(&self) -> bool {
        !matches!(self, EpochOutcome::Continuing)
    }

    pub fn is_success(&self) -> bool {
        matches!(self, EpochOutcome::Converged)
    }
}

impl std::fmt::Display for EpochOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Summary of a single epoch execution.
#[derive(Debug, Clone)]
pub struct EpochEvent {
    /// Epoch number (0-indexed).
    pub epoch: usize,
    /// Outcome of this epoch.
    pub outcome: EpochOutcome,
    /// Duration of epoch execution in microseconds.
    pub duration_us: u64,
    /// Number of instructions executed.
    pub instructions_executed: u64,
    /// Number of ORACLE operations.
    pub oracle_count: u64,
    /// Number of PROPHECY operations.
    pub prophecy_count: u64,
    /// Number of addresses that changed from Anamnesis.
    pub changed_cells: usize,
    /// Sum of absolute value changes (L1 distance from Anamnesis).
    pub total_delta: u64,
    /// Maximum single cell delta.
    pub max_delta: u64,
    /// Number of temporally-tainted values in Present.
    pub temporal_values: usize,
    /// Number of pure (non-temporal) values in Present.
    pub pure_values: usize,
    /// Hash of the resulting Present state.
    pub state_hash: u64,
    /// Error message if outcome is Error.
    pub error_message: Option<String>,
}

impl EpochEvent {
    /// Create a new epoch event.
    pub fn new(epoch: usize) -> Self {
        Self {
            epoch,
            outcome: EpochOutcome::Continuing,
            duration_us: 0,
            instructions_executed: 0,
            oracle_count: 0,
            prophecy_count: 0,
            changed_cells: 0,
            total_delta: 0,
            max_delta: 0,
            temporal_values: 0,
            pure_values: 0,
            state_hash: 0,
            error_message: None,
        }
    }

    /// Builder: set outcome.
    pub fn with_outcome(mut self, outcome: EpochOutcome) -> Self {
        self.outcome = outcome;
        self
    }

    /// Builder: set duration.
    pub fn with_duration_us(mut self, us: u64) -> Self {
        self.duration_us = us;
        self
    }

    /// Builder: set instruction count.
    pub fn with_instructions(mut self, count: u64) -> Self {
        self.instructions_executed = count;
        self
    }

    /// Builder: set temporal operation counts.
    pub fn with_temporal_ops(mut self, oracle: u64, prophecy: u64) -> Self {
        self.oracle_count = oracle;
        self.prophecy_count = prophecy;
        self
    }

    /// Builder: set change metrics.
    pub fn with_changes(mut self, changed: usize, total_delta: u64, max_delta: u64) -> Self {
        self.changed_cells = changed;
        self.total_delta = total_delta;
        self.max_delta = max_delta;
        self
    }

    /// Builder: set provenance metrics.
    pub fn with_provenance(mut self, temporal: usize, pure: usize) -> Self {
        self.temporal_values = temporal;
        self.pure_values = pure;
        self
    }

    /// Builder: set state hash.
    pub fn with_state_hash(mut self, hash: u64) -> Self {
        self.state_hash = hash;
        self
    }

    /// Builder: set error message.
    pub fn with_error(mut self, msg: impl Into<String>) -> Self {
        self.error_message = Some(msg.into());
        self.outcome = EpochOutcome::Error;
        self
    }

    /// Calculate the convergence rate (0.0 = no progress, 1.0 = converged).
    /// Based on the ratio of unchanged cells to total modified cells.
    pub fn convergence_rate(&self) -> f64 {
        let total_active = self.oracle_count + self.prophecy_count;
        if total_active == 0 {
            return 1.0; // No temporal operations = trivially converged
        }
        let unchanged = total_active.saturating_sub(self.changed_cells as u64);
        unchanged as f64 / total_active as f64
    }

    /// Format as a concise log line.
    pub fn format_line(&self) -> String {
        let duration_str = if self.duration_us >= 1_000_000 {
            format!("{:.2}s", self.duration_us as f64 / 1_000_000.0)
        } else if self.duration_us >= 1_000 {
            format!("{:.2}ms", self.duration_us as f64 / 1_000.0)
        } else {
            format!("{}us", self.duration_us)
        };

        format!(
            "EPOCH {:04} | {} | {} | inst={} oracle={} prophecy={} changed={} delta={} | T={} P={} | hash={:016x}",
            self.epoch,
            self.outcome,
            duration_str,
            self.instructions_executed,
            self.oracle_count,
            self.prophecy_count,
            self.changed_cells,
            self.total_delta,
            self.temporal_values,
            self.pure_values,
            self.state_hash
        )
    }
}

// =============================================================================
// Convergence Event
// =============================================================================

/// Summary of the entire convergence process.
#[derive(Debug, Clone)]
pub struct ConvergenceEvent {
    /// Program identifier (e.g., filename).
    pub program_id: String,
    /// Final outcome.
    pub outcome: EpochOutcome,
    /// Total epochs executed.
    pub total_epochs: usize,
    /// Total execution time in microseconds.
    pub total_duration_us: u64,
    /// Total instructions executed across all epochs.
    pub total_instructions: u64,
    /// Total ORACLE operations.
    pub total_oracle_ops: u64,
    /// Total PROPHECY operations.
    pub total_prophecy_ops: u64,
    /// Oscillation period (if oscillating).
    pub oscillation_period: Option<usize>,
    /// Addresses involved in oscillation/divergence.
    pub problem_cells: Vec<Address>,
    /// Output values produced.
    pub output_count: usize,
    /// Correlation ID for this execution.
    pub correlation_id: String,
    /// Timestamp when convergence search started.
    pub start_timestamp_ms: u64,
    /// Timestamp when convergence search ended.
    pub end_timestamp_ms: u64,
    /// Seed used for initial anamnesis.
    pub seed: u64,
    /// Action principle score (if action-guided mode).
    pub action_score: Option<f64>,
    /// Number of seeds explored (if action-guided mode).
    pub seeds_explored: Option<usize>,
}

impl ConvergenceEvent {
    /// Create a new convergence event.
    pub fn new(program_id: impl Into<String>, correlation_id: impl Into<String>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            program_id: program_id.into(),
            outcome: EpochOutcome::Continuing,
            total_epochs: 0,
            total_duration_us: 0,
            total_instructions: 0,
            total_oracle_ops: 0,
            total_prophecy_ops: 0,
            oscillation_period: None,
            problem_cells: Vec::new(),
            output_count: 0,
            correlation_id: correlation_id.into(),
            start_timestamp_ms: now,
            end_timestamp_ms: now,
            seed: 0,
            action_score: None,
            seeds_explored: None,
        }
    }

    /// Finalize the event with outcome and end timestamp.
    pub fn finalize(mut self, outcome: EpochOutcome) -> Self {
        self.outcome = outcome;
        self.end_timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self
    }

    /// Calculate average epoch duration in microseconds.
    pub fn avg_epoch_duration_us(&self) -> f64 {
        if self.total_epochs == 0 {
            0.0
        } else {
            self.total_duration_us as f64 / self.total_epochs as f64
        }
    }

    /// Calculate instructions per second.
    pub fn instructions_per_second(&self) -> f64 {
        if self.total_duration_us == 0 {
            0.0
        } else {
            (self.total_instructions as f64 * 1_000_000.0) / self.total_duration_us as f64
        }
    }

    /// Format as a detailed log block.
    pub fn format_block(&self) -> String {
        let duration_str = if self.total_duration_us >= 1_000_000 {
            format!("{:.3}s", self.total_duration_us as f64 / 1_000_000.0)
        } else if self.total_duration_us >= 1_000 {
            format!("{:.2}ms", self.total_duration_us as f64 / 1_000.0)
        } else {
            format!("{}us", self.total_duration_us)
        };

        let mut lines = vec![
            "═".repeat(60),
            format!("CONVERGENCE RESULT: {}", self.outcome),
            "═".repeat(60),
            format!("Program:      {}", self.program_id),
            format!("Correlation:  {}", self.correlation_id),
            format!("Seed:         {}", self.seed),
            format!("Epochs:       {}", self.total_epochs),
            format!("Duration:     {}", duration_str),
            format!("Instructions: {} ({:.0}/s)", self.total_instructions, self.instructions_per_second()),
            format!("Oracle ops:   {}", self.total_oracle_ops),
            format!("Prophecy ops: {}", self.total_prophecy_ops),
            format!("Output items: {}", self.output_count),
        ];

        if let Some(period) = self.oscillation_period {
            lines.push(format!("Oscillation:  period {}", period));
        }

        if !self.problem_cells.is_empty() {
            let cells: Vec<String> = self.problem_cells.iter().map(|a| a.to_string()).collect();
            lines.push(format!("Problem cells: [{}]", cells.join(", ")));
        }

        if let Some(action) = self.action_score {
            lines.push(format!("Action score: {:.6}", action));
        }

        if let Some(seeds) = self.seeds_explored {
            lines.push(format!("Seeds explored: {}", seeds));
        }

        lines.push("═".repeat(60));
        lines.join("\n")
    }
}

// =============================================================================
// Action Principle Event
// =============================================================================

/// Event for action principle fixed-point selection.
#[derive(Debug, Clone)]
pub struct ActionPrincipleEvent {
    /// Correlation ID for the convergence search.
    pub correlation_id: String,
    /// Seed index that produced this candidate.
    pub seed_index: usize,
    /// Seed values used.
    pub seed_hash: u64,
    /// Number of epochs to converge.
    pub epochs: usize,
    /// Computed action (cost) for this fixed point.
    pub action: f64,
    /// Breakdown of action components.
    pub action_breakdown: ActionBreakdown,
    /// Whether this candidate was selected as best.
    pub selected: bool,
}

/// Breakdown of action principle components.
#[derive(Debug, Clone, Default)]
pub struct ActionBreakdown {
    /// Penalty for zero values.
    pub zero_penalty: f64,
    /// Penalty for pure (non-temporal) values.
    pub purity_penalty: f64,
    /// Reward for causal depth.
    pub depth_reward: f64,
    /// Reward for output production.
    pub output_reward: f64,
    /// Total action value.
    pub total: f64,
}

impl ActionPrincipleEvent {
    pub fn new(correlation_id: impl Into<String>, seed_index: usize, seed_hash: u64) -> Self {
        Self {
            correlation_id: correlation_id.into(),
            seed_index,
            seed_hash,
            epochs: 0,
            action: f64::MAX,
            action_breakdown: ActionBreakdown::default(),
            selected: false,
        }
    }

    pub fn with_result(mut self, epochs: usize, action: f64, breakdown: ActionBreakdown) -> Self {
        self.epochs = epochs;
        self.action = action;
        self.action_breakdown = breakdown;
        self
    }

    pub fn mark_selected(mut self) -> Self {
        self.selected = true;
        self
    }

    pub fn format_line(&self) -> String {
        let selected_str = if self.selected { " [SELECTED]" } else { "" };
        format!(
            "ACTION seed[{}] hash={:016x} epochs={} action={:.6} (zero={:.3} pure={:.3} depth={:.3} output={:.3}){}",
            self.seed_index,
            self.seed_hash,
            self.epochs,
            self.action,
            self.action_breakdown.zero_penalty,
            self.action_breakdown.purity_penalty,
            self.action_breakdown.depth_reward,
            self.action_breakdown.output_reward,
            selected_str
        )
    }
}

// =============================================================================
// Provenance Event
// =============================================================================

/// Event capturing provenance/causal information.
#[derive(Debug, Clone)]
pub struct ProvenanceEvent {
    /// Epoch in which this provenance was recorded.
    pub epoch: usize,
    /// Address being analyzed.
    pub address: Address,
    /// Value at this address.
    pub value: u64,
    /// Whether the value is temporally tainted.
    pub is_temporal: bool,
    /// Number of oracle dependencies.
    pub dependency_count: usize,
    /// List of dependency addresses (truncated if large).
    pub dependencies: Vec<Address>,
    /// Causal depth (max chain length from any oracle).
    pub causal_depth: usize,
}

impl ProvenanceEvent {
    pub fn new(epoch: usize, address: Address, value: u64) -> Self {
        Self {
            epoch,
            address,
            value,
            is_temporal: false,
            dependency_count: 0,
            dependencies: Vec::new(),
            causal_depth: 0,
        }
    }

    pub fn with_temporal(mut self, deps: impl IntoIterator<Item = Address>) -> Self {
        self.is_temporal = true;
        let deps_vec: Vec<Address> = deps.into_iter().collect();
        self.dependency_count = deps_vec.len();
        // Keep only first 16 for logging, store count for accuracy
        self.dependencies = deps_vec.into_iter().take(16).collect();
        self
    }

    pub fn with_causal_depth(mut self, depth: usize) -> Self {
        self.causal_depth = depth;
        self
    }

    pub fn format_line(&self) -> String {
        let temporal_str = if self.is_temporal { "T" } else { "P" };
        let deps_str = if self.dependencies.is_empty() {
            String::new()
        } else {
            let shown: Vec<String> = self.dependencies.iter().map(|a| a.to_string()).collect();
            let suffix = if self.dependency_count > 16 {
                format!("...+{}", self.dependency_count - 16)
            } else {
                String::new()
            };
            format!(" <- [{}{}]", shown.join(","), suffix)
        };
        format!(
            "E{:04} @{:05} = {} [{}] depth={}{}",
            self.epoch, self.address, self.value, temporal_str, self.causal_depth, deps_str
        )
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_operation() {
        assert!(TemporalOperation::Oracle.is_read());
        assert!(!TemporalOperation::Oracle.is_write());
        assert!(TemporalOperation::Prophecy.is_write());
        assert!(TemporalOperation::Paradox.is_terminal());
    }

    #[test]
    fn test_temporal_event_creation() {
        let event = TemporalEvent::oracle(42, 100, 5, 1000);
        assert_eq!(event.operation, TemporalOperation::Oracle);
        assert_eq!(event.address, 42);
        assert_eq!(event.value, 100);
        assert_eq!(event.epoch, 5);
        assert!(event.is_temporal);
        assert!(event.dependencies.as_ref().unwrap().contains(&42));
    }

    #[test]
    fn test_epoch_event() {
        let event = EpochEvent::new(10)
            .with_outcome(EpochOutcome::Converged)
            .with_duration_us(5000)
            .with_instructions(1000)
            .with_temporal_ops(50, 25)
            .with_changes(3, 15, 7);

        assert_eq!(event.epoch, 10);
        assert_eq!(event.outcome, EpochOutcome::Converged);
        assert!(event.outcome.is_success());
        assert!(event.outcome.is_terminal());

        let line = event.format_line();
        assert!(line.contains("EPOCH 0010"));
        assert!(line.contains("CONVERGED"));
    }

    #[test]
    fn test_convergence_event() {
        let event = ConvergenceEvent::new("test.ouro", "cid-123")
            .finalize(EpochOutcome::Converged);

        assert_eq!(event.outcome, EpochOutcome::Converged);
        assert!(!event.correlation_id.is_empty());

        let block = event.format_block();
        assert!(block.contains("CONVERGED"));
        assert!(block.contains("test.ouro"));
    }

    #[test]
    fn test_epoch_outcome_properties() {
        assert!(!EpochOutcome::Continuing.is_terminal());
        assert!(EpochOutcome::Converged.is_terminal());
        assert!(EpochOutcome::Oscillating.is_terminal());
        assert!(EpochOutcome::Converged.is_success());
        assert!(!EpochOutcome::Oscillating.is_success());
    }

    #[test]
    fn test_action_principle_event() {
        let breakdown = ActionBreakdown {
            zero_penalty: 0.1,
            purity_penalty: 0.2,
            depth_reward: -0.15,
            output_reward: -0.05,
            total: 0.1,
        };
        let event = ActionPrincipleEvent::new("cid-123", 0, 0x12345678)
            .with_result(5, 0.1, breakdown)
            .mark_selected();

        assert!(event.selected);
        assert_eq!(event.epochs, 5);

        let line = event.format_line();
        assert!(line.contains("[SELECTED]"));
    }
}
