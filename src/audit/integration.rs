//! Integration between audit system and core Ourochronos components.
//!
//! This module provides bridges between the audit system and:
//! - Provenance tracking (causal dependency analysis)
//! - Error handling (OuroError → AuditEntry conversion)
//! - TimeLoop (automatic convergence logging)
//!
//! # Design Principles
//!
//! - Zero-cost when audit is disabled
//! - No circular dependencies (audit → core only)
//! - Correlation IDs for distributed tracing across epochs

use crate::core::{Provenance, OuroError, ErrorCategory, Address};
use super::entry::{
    AuditEntry, ActionCategory, Severity, Outcome,
    ConvergenceOutcome, ProvenanceSummary,
};
use super::temporal_events::{EpochOutcome, ConvergenceEvent, EpochEvent};

// =============================================================================
// Provenance Integration
// =============================================================================

/// Convert a Provenance to a ProvenanceSummary for audit logging.
///
/// This extracts the essential causal information without copying
/// the full dependency set.
pub fn provenance_to_summary(prov: &Provenance) -> ProvenanceSummary {
    if prov.is_pure() {
        ProvenanceSummary {
            pure_count: 1,
            temporal_count: 0,
            max_depth: 0,
            oracle_sources: 0,
        }
    } else {
        let dep_count = prov.dependency_count();
        ProvenanceSummary {
            pure_count: 0,
            temporal_count: 1,
            max_depth: 1, // Would need full graph for true depth
            oracle_sources: dep_count as u16,
        }
    }
}

/// Aggregate provenance summaries for multiple values.
pub fn aggregate_provenance<'a>(
    provenances: impl Iterator<Item = &'a Provenance>,
) -> ProvenanceSummary {
    let mut summary = ProvenanceSummary::default();
    let mut total_deps = 0usize;

    for prov in provenances {
        if prov.is_pure() {
            summary.pure_count += 1;
        } else {
            summary.temporal_count += 1;
            let deps = prov.dependency_count();
            total_deps += deps;
            summary.max_depth = summary.max_depth.max(1);
        }
    }

    // Average oracle sources per temporal value
    if summary.temporal_count > 0 {
        summary.oracle_sources = (total_deps as f64 / summary.temporal_count as f64).ceil() as u16;
    }

    summary
}

/// Create audit metadata from a Provenance.
pub fn provenance_to_metadata(prov: &Provenance) -> Vec<(String, String)> {
    let mut meta = Vec::new();

    meta.push(("temporal".to_string(), prov.is_temporal().to_string()));
    meta.push(("dep_count".to_string(), prov.dependency_count().to_string()));

    // Include first few dependencies if temporal
    if prov.is_temporal() {
        let deps: Vec<String> = prov
            .dependencies()
            .take(8)
            .map(|a| a.to_string())
            .collect();
        if !deps.is_empty() {
            meta.push(("deps".to_string(), deps.join(",")));
        }
    }

    meta
}

// =============================================================================
// Error Integration
// =============================================================================

/// Convert an OuroError to an AuditEntry.
pub fn error_to_entry(error: &OuroError) -> AuditEntry {
    let category = error_category_to_action(error.category());
    let severity = error_to_severity(error);
    let (entity_type, entity_id) = error_entity(error);

    let mut entry = AuditEntry::new(
        error_action(error),
        entity_type,
        entity_id,
        error.to_string(),
    )
    .with_category(category)
    .with_severity(severity)
    .with_outcome(Outcome::Failure)
    .with_meta("error_code", error.code().to_string());

    // Add location if available
    if let Some(loc) = error.location() {
        if loc.line > 0 {
            entry = entry.with_meta("line", loc.line.to_string());
            entry = entry.with_meta("column", loc.column.to_string());
        }
        if loc.stmt_index > 0 {
            entry = entry.with_meta("stmt_index", loc.stmt_index.to_string());
        }
    }

    // Add error-specific metadata
    match error {
        OuroError::StackUnderflow { required, available, .. } => {
            entry = entry
                .with_meta("required", required.to_string())
                .with_meta("available", available.to_string());
        }
        OuroError::Oscillation { period, oscillating_cells, epoch } => {
            entry = entry
                .with_epoch(*epoch)
                .with_convergence_outcome(ConvergenceOutcome::Oscillating)
                .with_meta("period", period.to_string())
                .with_meta("oscillating_cells", format!("{:?}", oscillating_cells));
        }
        OuroError::Divergence { epoch, diverging_cells } => {
            entry = entry
                .with_epoch(*epoch)
                .with_convergence_outcome(ConvergenceOutcome::Diverging)
                .with_meta("diverging_cells", format!("{:?}", diverging_cells));
        }
        OuroError::MaxEpochsExceeded { max_epochs } => {
            entry = entry
                .with_convergence_outcome(ConvergenceOutcome::Timeout)
                .with_meta("max_epochs", max_epochs.to_string());
        }
        OuroError::ExplicitParadox { .. } => {
            entry = entry.with_convergence_outcome(ConvergenceOutcome::Paradox);
        }
        _ => {}
    }

    entry
}

/// Map ErrorCategory to ActionCategory.
fn error_category_to_action(cat: ErrorCategory) -> ActionCategory {
    match cat {
        ErrorCategory::Runtime => ActionCategory::Execute,
        ErrorCategory::Temporal => ActionCategory::Convergence,
        ErrorCategory::Parse => ActionCategory::Parse,
        ErrorCategory::Type => ActionCategory::TypeCheck,
        ErrorCategory::FFI => ActionCategory::Ffi,
        ErrorCategory::IO => ActionCategory::Io,
        ErrorCategory::Internal => ActionCategory::Error,
    }
}

/// Map OuroError to Severity.
fn error_to_severity(error: &OuroError) -> Severity {
    if error.is_recoverable() {
        Severity::Warning
    } else {
        match error.category() {
            ErrorCategory::Internal => Severity::Critical,
            ErrorCategory::Temporal => Severity::Error,
            _ => Severity::Error,
        }
    }
}

/// Extract entity type and ID from error.
fn error_entity(error: &OuroError) -> (String, String) {
    match error {
        OuroError::StackUnderflow { operation, .. } => {
            ("Stack".to_string(), operation.clone())
        }
        OuroError::MemoryBoundsViolation { address, operation, .. } => {
            ("Memory".to_string(), format!("{}@{}", operation, address))
        }
        OuroError::Oscillation { epoch, .. } |
        OuroError::Divergence { epoch, .. } => {
            ("TimeLoop".to_string(), format!("epoch_{}", epoch))
        }
        OuroError::UndefinedProcedure { name, .. } => {
            ("Procedure".to_string(), name.clone())
        }
        _ => ("System".to_string(), "error".to_string()),
    }
}

/// Get action name for error.
fn error_action(error: &OuroError) -> &'static str {
    match error {
        OuroError::StackUnderflow { .. } |
        OuroError::StackOverflow { .. } => "STACK_ERROR",
        OuroError::DivisionByZero { .. } |
        OuroError::ModuloByZero { .. } => "ARITH_ERROR",
        OuroError::MemoryBoundsViolation { .. } => "MEMORY_ERROR",
        OuroError::ExplicitParadox { .. } => "PARADOX",
        OuroError::Oscillation { .. } => "OSCILLATION",
        OuroError::Divergence { .. } => "DIVERGENCE",
        OuroError::MaxEpochsExceeded { .. } => "TIMEOUT",
        OuroError::UnexpectedToken { .. } |
        OuroError::UnexpectedEof { .. } |
        OuroError::UnknownKeyword { .. } => "PARSE_ERROR",
        _ => "ERROR",
    }
}

// =============================================================================
// Convergence Status Integration
// =============================================================================

/// Convert timeloop ConvergenceStatus to EpochOutcome.
pub fn convergence_status_to_outcome(status_str: &str) -> EpochOutcome {
    match status_str.to_uppercase().as_str() {
        "CONSISTENT" => EpochOutcome::Converged,
        "CONTINUING" | "ITERATING" => EpochOutcome::Continuing,
        "OSCILLATION" | "OSCILLATING" => EpochOutcome::Oscillating,
        "DIVERGENCE" | "DIVERGING" => EpochOutcome::Diverging,
        "PARADOX" => EpochOutcome::Paradox,
        "ERROR" => EpochOutcome::Error,
        "TIMEOUT" => EpochOutcome::Timeout,
        _ => EpochOutcome::Error,
    }
}

/// Convert EpochOutcome to ConvergenceOutcome (entry type).
pub fn epoch_outcome_to_convergence(outcome: EpochOutcome) -> ConvergenceOutcome {
    match outcome {
        EpochOutcome::Converged => ConvergenceOutcome::Consistent,
        EpochOutcome::Continuing => ConvergenceOutcome::Iterating,
        EpochOutcome::Oscillating => ConvergenceOutcome::Oscillating,
        EpochOutcome::Diverging => ConvergenceOutcome::Diverging,
        EpochOutcome::Paradox => ConvergenceOutcome::Paradox,
        EpochOutcome::Error => ConvergenceOutcome::Error,
        EpochOutcome::Timeout => ConvergenceOutcome::Timeout,
    }
}

// =============================================================================
// Temporal Event to AuditEntry Conversion
// =============================================================================

/// Convert an EpochEvent to an AuditEntry.
pub fn epoch_event_to_entry(event: &EpochEvent) -> AuditEntry {
    let outcome = epoch_outcome_to_convergence(event.outcome);

    let mut entry = AuditEntry::epoch_iteration(
        event.epoch,
        event.duration_us,
        event.instructions_executed,
        event.changed_cells,
    )
    .with_convergence_outcome(outcome)
    .with_state_hash(event.state_hash)
    .with_meta("oracle_ops", event.oracle_count.to_string())
    .with_meta("prophecy_ops", event.prophecy_count.to_string())
    .with_meta("total_delta", event.total_delta.to_string())
    .with_provenance_counts(event.pure_values as u32, event.temporal_values as u32);

    if let Some(ref msg) = event.error_message {
        entry = entry.with_meta("error", msg.clone());
    }

    entry
}

/// Convert a ConvergenceEvent to an AuditEntry.
pub fn convergence_event_to_entry(event: &ConvergenceEvent) -> AuditEntry {
    let outcome = epoch_outcome_to_convergence(event.outcome);

    let mut entry = AuditEntry::convergence_result(
        outcome,
        event.total_epochs,
        event.total_duration_us,
    )
    .with_correlation(&event.correlation_id)
    .with_meta("program", event.program_id.clone())
    .with_meta("total_instructions", event.total_instructions.to_string())
    .with_meta("oracle_ops", event.total_oracle_ops.to_string())
    .with_meta("prophecy_ops", event.total_prophecy_ops.to_string())
    .with_meta("output_count", event.output_count.to_string())
    .with_meta("seed", event.seed.to_string());

    if let Some(period) = event.oscillation_period {
        entry = entry.with_meta("oscillation_period", period.to_string());
    }

    if !event.problem_cells.is_empty() {
        let cells: Vec<String> = event.problem_cells.iter().map(|a| a.to_string()).collect();
        entry = entry.with_meta("problem_cells", cells.join(","));
    }

    if let Some(action) = event.action_score {
        entry = entry.with_meta("action_score", format!("{:.6}", action));
    }

    if let Some(seeds) = event.seeds_explored {
        entry = entry.with_meta("seeds_explored", seeds.to_string());
    }

    entry
}

// =============================================================================
// Audit Session
// =============================================================================

/// An audit session for tracking a complete execution.
///
/// Maintains correlation ID and provides convenient logging methods.
#[derive(Debug, Clone)]
pub struct AuditSession {
    /// Correlation ID for this session.
    pub correlation_id: String,
    /// Program identifier.
    pub program_id: String,
    /// Session start timestamp (ms since epoch).
    pub start_timestamp_ms: u64,
    /// Current epoch (for temporal context).
    pub current_epoch: usize,
}

impl AuditSession {
    /// Create a new audit session.
    pub fn new(program_id: impl Into<String>) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};

        let correlation_id = super::logger::generate_correlation_id("ouro");
        let start_timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            correlation_id,
            program_id: program_id.into(),
            start_timestamp_ms,
            current_epoch: 0,
        }
    }

    /// Create an audit entry with session context.
    pub fn entry(
        &self,
        action: impl Into<String>,
        entity_type: impl Into<String>,
        entity_id: impl Into<String>,
        description: impl Into<String>,
    ) -> AuditEntry {
        AuditEntry::new(action, entity_type, entity_id, description)
            .with_correlation(&self.correlation_id)
            .with_epoch(self.current_epoch)
    }

    /// Create an ORACLE read entry.
    pub fn oracle_entry(&self, address: Address, value: u64, is_temporal: bool) -> AuditEntry {
        AuditEntry::oracle_read(address, value, self.current_epoch, is_temporal)
            .with_correlation(&self.correlation_id)
    }

    /// Create a PROPHECY write entry.
    pub fn prophecy_entry(&self, address: Address, value: u64, previous: u64) -> AuditEntry {
        AuditEntry::prophecy_write(address, value, previous, self.current_epoch)
            .with_correlation(&self.correlation_id)
    }

    /// Set the current epoch.
    pub fn set_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;
    }

    /// Advance to next epoch.
    pub fn next_epoch(&mut self) {
        self.current_epoch += 1;
    }

    /// Get elapsed time since session start in microseconds.
    pub fn elapsed_us(&self) -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        (now.saturating_sub(self.start_timestamp_ms)) * 1000
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provenance_to_summary_pure() {
        let prov = Provenance::none();
        let summary = provenance_to_summary(&prov);

        assert_eq!(summary.pure_count, 1);
        assert_eq!(summary.temporal_count, 0);
    }

    #[test]
    fn test_provenance_to_summary_temporal() {
        let prov = Provenance::single(42);
        let summary = provenance_to_summary(&prov);

        assert_eq!(summary.pure_count, 0);
        assert_eq!(summary.temporal_count, 1);
        assert_eq!(summary.oracle_sources, 1);
    }

    #[test]
    fn test_aggregate_provenance() {
        let provenances = vec![
            Provenance::none(),
            Provenance::none(),
            Provenance::single(1),
            Provenance::single(2),
        ];

        let summary = aggregate_provenance(provenances.iter());

        assert_eq!(summary.pure_count, 2);
        assert_eq!(summary.temporal_count, 2);
    }

    #[test]
    fn test_error_to_entry() {
        use crate::core::error::SourceLocation;

        let error = OuroError::StackUnderflow {
            operation: "ADD".to_string(),
            required: 2,
            available: 1,
            location: SourceLocation::new(10, 5),
        };

        let entry = error_to_entry(&error);

        assert_eq!(entry.category, ActionCategory::Execute);
        assert_eq!(entry.severity, Severity::Warning); // Recoverable
        assert_eq!(entry.outcome, Outcome::Failure);
        assert!(entry.description.contains("Stack underflow"));
    }

    #[test]
    fn test_error_to_entry_temporal() {
        let error = OuroError::Oscillation {
            period: 2,
            oscillating_cells: vec![0, 1],
            epoch: 5,
        };

        let entry = error_to_entry(&error);

        assert_eq!(entry.category, ActionCategory::Convergence);
        assert_eq!(entry.convergence_outcome, Some(ConvergenceOutcome::Oscillating));
        assert_eq!(entry.epoch, Some(5));
    }

    #[test]
    fn test_convergence_status_to_outcome() {
        assert_eq!(convergence_status_to_outcome("CONSISTENT"), EpochOutcome::Converged);
        assert_eq!(convergence_status_to_outcome("oscillation"), EpochOutcome::Oscillating);
        assert_eq!(convergence_status_to_outcome("TIMEOUT"), EpochOutcome::Timeout);
    }

    #[test]
    fn test_audit_session() {
        let mut session = AuditSession::new("test.ouro");

        assert!(!session.correlation_id.is_empty());
        assert_eq!(session.current_epoch, 0);

        session.next_epoch();
        assert_eq!(session.current_epoch, 1);

        let entry = session.entry("TEST", "Unit", "1", "Test entry");
        assert_eq!(entry.correlation_id, Some(session.correlation_id.clone()));
        assert_eq!(entry.epoch, Some(1));
    }

    #[test]
    fn test_epoch_event_to_entry() {
        use super::super::temporal_events::EpochEvent;

        let event = EpochEvent::new(5)
            .with_outcome(EpochOutcome::Converged)
            .with_duration_us(1000)
            .with_instructions(500)
            .with_temporal_ops(10, 5)
            .with_changes(3, 15, 7)
            .with_provenance(20, 80)
            .with_state_hash(0x12345678);

        let entry = epoch_event_to_entry(&event);

        assert_eq!(entry.epoch, Some(5));
        assert_eq!(entry.convergence_outcome, Some(ConvergenceOutcome::Consistent));
        assert_eq!(entry.state_hash, Some(0x12345678));
    }
}
