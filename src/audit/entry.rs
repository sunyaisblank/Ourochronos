//! Audit entry types and structures.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use super::escape_json;

// =============================================================================
// Core Types
// =============================================================================

/// Severity level for audit entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Severity {
    /// Informational message (normal operation).
    Info = 0,
    /// Warning (potential issue, operation continued).
    Warning = 1,
    /// Error (operation failed).
    Error = 2,
    /// Critical (system-level failure).
    Critical = 3,
}

impl Severity {
    /// Convert to string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Severity::Info => "INFO",
            Severity::Warning => "WARN",
            Severity::Error => "ERROR",
            Severity::Critical => "CRITICAL",
        }
    }

    /// Parse from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "INFO" | "INFORMATION" => Some(Severity::Info),
            "WARN" | "WARNING" => Some(Severity::Warning),
            "ERROR" | "ERR" => Some(Severity::Error),
            "CRITICAL" | "CRIT" => Some(Severity::Critical),
            _ => None,
        }
    }
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Outcome of an audited operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Outcome {
    /// Operation completed successfully.
    Success = 0,
    /// Operation failed.
    Failure = 1,
    /// Operation partially completed.
    Partial = 2,
}

impl Outcome {
    /// Convert to string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Outcome::Success => "SUCCESS",
            Outcome::Failure => "FAILURE",
            Outcome::Partial => "PARTIAL",
        }
    }

    /// Parse from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "SUCCESS" | "OK" => Some(Outcome::Success),
            "FAILURE" | "FAIL" | "FAILED" => Some(Outcome::Failure),
            "PARTIAL" => Some(Outcome::Partial),
            _ => None,
        }
    }
}

impl std::fmt::Display for Outcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Category of audited action.
///
/// Provides fine-grained categorization for temporal computation auditing.
/// Categories are organized hierarchically from general to specific.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ActionCategory {
    // ═══════════════════════════════════════════════════════════════════
    // Compilation Phase
    // ═══════════════════════════════════════════════════════════════════

    /// Parsing operations.
    Parse,
    /// Compilation/analysis.
    Compile,
    /// Type checking and effect analysis.
    TypeCheck,
    /// SMT constraint generation.
    SmtEncode,

    // ═══════════════════════════════════════════════════════════════════
    // Execution Phase
    // ═══════════════════════════════════════════════════════════════════

    /// General program execution.
    Execute,
    /// Stack operations.
    Stack,
    /// Memory read/write (non-temporal).
    Memory,

    // ═══════════════════════════════════════════════════════════════════
    // Temporal Operations (First-Principle Primitives)
    // ═══════════════════════════════════════════════════════════════════

    /// ORACLE: Read from Anamnesis (future state from previous epoch).
    OracleRead,
    /// PROPHECY: Write to Present (establishing future state).
    ProphecyWrite,
    /// PRESENT: Read from Present (current epoch state).
    PresentRead,
    /// PARADOX: Explicit paradox trigger.
    Paradox,
    /// Generic temporal operation (deprecated, use specific variants).
    #[deprecated(note = "Use OracleRead, ProphecyWrite, PresentRead, or Paradox instead")]
    Temporal,

    // ═══════════════════════════════════════════════════════════════════
    // Convergence/Fixed-Point
    // ═══════════════════════════════════════════════════════════════════

    /// Epoch iteration event.
    EpochIteration,
    /// Fixed-point check (comparing Present to Anamnesis).
    ConvergenceCheck,
    /// Cycle/oscillation detection.
    CycleDetection,
    /// Divergence detection.
    DivergenceDetection,
    /// Action principle evaluation.
    ActionPrinciple,
    /// Generic convergence (for backward compatibility).
    Convergence,

    // ═══════════════════════════════════════════════════════════════════
    // Optimization
    // ═══════════════════════════════════════════════════════════════════

    /// JIT compilation.
    Jit,
    /// Epoch caching/memoization.
    Memoization,
    /// Tracing optimization.
    Tracing,

    // ═══════════════════════════════════════════════════════════════════
    // System
    // ═══════════════════════════════════════════════════════════════════

    /// Configuration changes.
    Config,
    /// System-level events.
    System,
    /// I/O operations.
    Io,
    /// FFI calls.
    Ffi,
    /// Error handling.
    Error,

    /// Custom category.
    Custom(String),
}

impl ActionCategory {
    pub fn as_str(&self) -> &str {
        match self {
            // Compilation
            ActionCategory::Parse => "PARSE",
            ActionCategory::Compile => "COMPILE",
            ActionCategory::TypeCheck => "TYPECHECK",
            ActionCategory::SmtEncode => "SMT_ENCODE",

            // Execution
            ActionCategory::Execute => "EXECUTE",
            ActionCategory::Stack => "STACK",
            ActionCategory::Memory => "MEMORY",

            // Temporal
            ActionCategory::OracleRead => "ORACLE_READ",
            ActionCategory::ProphecyWrite => "PROPHECY_WRITE",
            ActionCategory::PresentRead => "PRESENT_READ",
            ActionCategory::Paradox => "PARADOX",
            #[allow(deprecated)]
            ActionCategory::Temporal => "TEMPORAL",

            // Convergence
            ActionCategory::EpochIteration => "EPOCH_ITER",
            ActionCategory::ConvergenceCheck => "CONV_CHECK",
            ActionCategory::CycleDetection => "CYCLE_DETECT",
            ActionCategory::DivergenceDetection => "DIV_DETECT",
            ActionCategory::ActionPrinciple => "ACTION_PRINC",
            ActionCategory::Convergence => "CONVERGENCE",

            // Optimization
            ActionCategory::Jit => "JIT",
            ActionCategory::Memoization => "MEMO",
            ActionCategory::Tracing => "TRACING",

            // System
            ActionCategory::Config => "CONFIG",
            ActionCategory::System => "SYSTEM",
            ActionCategory::Io => "IO",
            ActionCategory::Ffi => "FFI",
            ActionCategory::Error => "ERROR",

            ActionCategory::Custom(s) => s,
        }
    }

    /// Check if this category is a temporal operation.
    pub fn is_temporal(&self) -> bool {
        matches!(
            self,
            ActionCategory::OracleRead
                | ActionCategory::ProphecyWrite
                | ActionCategory::PresentRead
                | ActionCategory::Paradox
        )
    }

    /// Check if this category is convergence-related.
    pub fn is_convergence(&self) -> bool {
        matches!(
            self,
            ActionCategory::EpochIteration
                | ActionCategory::ConvergenceCheck
                | ActionCategory::CycleDetection
                | ActionCategory::DivergenceDetection
                | ActionCategory::ActionPrinciple
                | ActionCategory::Convergence
        )
    }

    /// Get the parent category for hierarchical filtering.
    pub fn parent(&self) -> Option<ActionCategory> {
        match self {
            // Temporal operations are children of Execute
            ActionCategory::OracleRead
            | ActionCategory::ProphecyWrite
            | ActionCategory::PresentRead
            | ActionCategory::Paradox => Some(ActionCategory::Execute),

            // Convergence sub-categories
            ActionCategory::EpochIteration
            | ActionCategory::ConvergenceCheck
            | ActionCategory::CycleDetection
            | ActionCategory::DivergenceDetection
            | ActionCategory::ActionPrinciple => Some(ActionCategory::Convergence),

            // Optimization is under System
            ActionCategory::Jit | ActionCategory::Memoization | ActionCategory::Tracing => {
                Some(ActionCategory::System)
            }

            _ => None,
        }
    }
}

impl std::fmt::Display for ActionCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// =============================================================================
// Convergence Outcome
// =============================================================================

/// Detailed outcome for convergence operations.
///
/// Provides more granular status than the generic Outcome enum
/// for temporal computation tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ConvergenceOutcome {
    /// Fixed point achieved (Present = Anamnesis).
    Consistent = 0,
    /// Still iterating, not yet converged.
    Iterating = 1,
    /// Oscillation detected (periodic non-convergence).
    Oscillating = 2,
    /// Divergence detected (unbounded growth).
    Diverging = 3,
    /// Explicit PARADOX instruction triggered.
    Paradox = 4,
    /// Runtime error during execution.
    Error = 5,
    /// Epoch limit exceeded.
    Timeout = 6,
}

impl ConvergenceOutcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            ConvergenceOutcome::Consistent => "CONSISTENT",
            ConvergenceOutcome::Iterating => "ITERATING",
            ConvergenceOutcome::Oscillating => "OSCILLATING",
            ConvergenceOutcome::Diverging => "DIVERGING",
            ConvergenceOutcome::Paradox => "PARADOX",
            ConvergenceOutcome::Error => "ERROR",
            ConvergenceOutcome::Timeout => "TIMEOUT",
        }
    }

    /// Check if this is a terminal (final) state.
    pub fn is_terminal(&self) -> bool {
        !matches!(self, ConvergenceOutcome::Iterating)
    }

    /// Check if this is a successful convergence.
    pub fn is_success(&self) -> bool {
        matches!(self, ConvergenceOutcome::Consistent)
    }

    /// Convert to generic Outcome.
    pub fn to_outcome(&self) -> Outcome {
        match self {
            ConvergenceOutcome::Consistent => Outcome::Success,
            ConvergenceOutcome::Iterating => Outcome::Partial,
            _ => Outcome::Failure,
        }
    }
}

impl std::fmt::Display for ConvergenceOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// =============================================================================
// Provenance Summary
// =============================================================================

/// Summary of provenance information for an audit entry.
///
/// Captures the causal dependency characteristics without
/// storing the full provenance graph.
#[derive(Debug, Clone, Default)]
pub struct ProvenanceSummary {
    /// Number of pure (non-temporal) values involved.
    pub pure_count: u32,
    /// Number of temporal (oracle-dependent) values involved.
    pub temporal_count: u32,
    /// Maximum causal depth (longest dependency chain).
    pub max_depth: u16,
    /// Number of distinct oracle addresses depended upon.
    pub oracle_sources: u16,
}

impl ProvenanceSummary {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_counts(pure: u32, temporal: u32) -> Self {
        Self {
            pure_count: pure,
            temporal_count: temporal,
            ..Default::default()
        }
    }

    /// Calculate the temporal ratio (0.0 = all pure, 1.0 = all temporal).
    pub fn temporal_ratio(&self) -> f64 {
        let total = self.pure_count + self.temporal_count;
        if total == 0 {
            0.0
        } else {
            self.temporal_count as f64 / total as f64
        }
    }

    /// Format as a compact string.
    pub fn format_compact(&self) -> String {
        format!(
            "P{}T{}D{}O{}",
            self.pure_count,
            self.temporal_count,
            self.max_depth,
            self.oracle_sources
        )
    }
}

// =============================================================================
// Audit Entry
// =============================================================================

/// An immutable audit log entry.
///
/// Captures WHO did WHAT to WHICH entity, WHEN, and with what OUTCOME.
///
/// # Temporal Extensions
///
/// For temporal computation auditing, entries can include:
/// - Epoch number for temporal context
/// - Convergence outcome for fixed-point tracking
/// - Provenance summary for causal analysis
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// Unique identifier for this entry.
    pub id: u64,
    /// UTC timestamp (milliseconds since epoch).
    pub timestamp_ms: u64,
    /// Action performed (e.g., "PARSE", "EXECUTE", "CONVERGE").
    pub action: String,
    /// Category of the action.
    pub category: ActionCategory,
    /// Type of entity involved (e.g., "Program", "Epoch", "Memory").
    pub entity_type: String,
    /// Identifier of the entity (e.g., filename, address).
    pub entity_id: String,
    /// Human-readable description.
    pub description: String,
    /// Severity level.
    pub severity: Severity,
    /// Outcome of the operation.
    pub outcome: Outcome,
    /// Correlation ID for tracing related operations.
    pub correlation_id: Option<String>,
    /// Duration in microseconds (if applicable).
    pub duration_us: Option<u64>,
    /// Additional structured data.
    pub metadata: HashMap<String, String>,

    // ═══════════════════════════════════════════════════════════════════
    // Temporal Extensions
    // ═══════════════════════════════════════════════════════════════════

    /// Current epoch number (0-indexed, None for non-temporal events).
    pub epoch: Option<usize>,
    /// Convergence-specific outcome (more detailed than generic outcome).
    pub convergence_outcome: Option<ConvergenceOutcome>,
    /// Provenance summary for causal analysis.
    pub provenance: Option<ProvenanceSummary>,
    /// Memory address involved (for temporal memory operations).
    pub address: Option<u16>,
    /// Value involved (for temporal memory operations).
    pub value: Option<u64>,
    /// State hash (for convergence tracking).
    pub state_hash: Option<u64>,
}

impl AuditEntry {
    /// Create a new audit entry with required fields.
    pub fn new(
        action: impl Into<String>,
        entity_type: impl Into<String>,
        entity_id: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            id: COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            timestamp_ms,
            action: action.into(),
            category: ActionCategory::System,
            entity_type: entity_type.into(),
            entity_id: entity_id.into(),
            description: description.into(),
            severity: Severity::Info,
            outcome: Outcome::Success,
            correlation_id: None,
            duration_us: None,
            metadata: HashMap::new(),
            // Temporal extensions
            epoch: None,
            convergence_outcome: None,
            provenance: None,
            address: None,
            value: None,
            state_hash: None,
        }
    }

    /// Create an entry for an ORACLE read operation.
    pub fn oracle_read(
        address: u16,
        value: u64,
        epoch: usize,
        is_temporal: bool,
    ) -> Self {
        Self::new(
            "ORACLE",
            "Memory",
            address.to_string(),
            format!("Read {} from anamnesis[{}]", value, address),
        )
        .with_category(ActionCategory::OracleRead)
        .with_epoch(epoch)
        .with_address(address)
        .with_value(value)
        .with_meta("temporal", is_temporal.to_string())
    }

    /// Create an entry for a PROPHECY write operation.
    pub fn prophecy_write(
        address: u16,
        value: u64,
        previous: u64,
        epoch: usize,
    ) -> Self {
        Self::new(
            "PROPHECY",
            "Memory",
            address.to_string(),
            format!("Write {} to present[{}] (was {})", value, address, previous),
        )
        .with_category(ActionCategory::ProphecyWrite)
        .with_epoch(epoch)
        .with_address(address)
        .with_value(value)
        .with_meta("previous", previous.to_string())
    }

    /// Create an entry for an epoch iteration.
    pub fn epoch_iteration(
        epoch: usize,
        duration_us: u64,
        instructions: u64,
        changed_cells: usize,
    ) -> Self {
        Self::new(
            "EPOCH",
            "TimeLoop",
            epoch.to_string(),
            format!("Epoch {} completed", epoch),
        )
        .with_category(ActionCategory::EpochIteration)
        .with_epoch(epoch)
        .with_duration_us(duration_us)
        .with_meta("instructions", instructions.to_string())
        .with_meta("changed_cells", changed_cells.to_string())
    }

    /// Create an entry for a convergence check.
    pub fn convergence_result(
        outcome: ConvergenceOutcome,
        total_epochs: usize,
        total_duration_us: u64,
    ) -> Self {
        let severity = if outcome.is_success() {
            Severity::Info
        } else if matches!(outcome, ConvergenceOutcome::Iterating) {
            Severity::Info
        } else {
            Severity::Warning
        };

        Self::new(
            "CONVERGE",
            "TimeLoop",
            "result",
            format!("{} after {} epochs", outcome, total_epochs),
        )
        .with_category(ActionCategory::ConvergenceCheck)
        .with_severity(severity)
        .with_outcome(outcome.to_outcome())
        .with_convergence_outcome(outcome)
        .with_epoch(total_epochs.saturating_sub(1))
        .with_duration_us(total_duration_us)
    }

    /// Set the action category.
    pub fn with_category(mut self, category: ActionCategory) -> Self {
        self.category = category;
        self
    }

    /// Set the severity level.
    pub fn with_severity(mut self, severity: Severity) -> Self {
        self.severity = severity;
        self
    }

    /// Set the outcome.
    pub fn with_outcome(mut self, outcome: Outcome) -> Self {
        self.outcome = outcome;
        self
    }

    /// Set the correlation ID for distributed tracing.
    pub fn with_correlation(mut self, id: impl Into<String>) -> Self {
        self.correlation_id = Some(id.into());
        self
    }

    /// Set the duration in microseconds.
    pub fn with_duration_us(mut self, us: u64) -> Self {
        self.duration_us = Some(us);
        self
    }

    /// Add metadata key-value pair.
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    // ═══════════════════════════════════════════════════════════════════
    // Temporal Extension Builders
    // ═══════════════════════════════════════════════════════════════════

    /// Set the epoch number.
    pub fn with_epoch(mut self, epoch: usize) -> Self {
        self.epoch = Some(epoch);
        self
    }

    /// Set the convergence outcome.
    pub fn with_convergence_outcome(mut self, outcome: ConvergenceOutcome) -> Self {
        self.convergence_outcome = Some(outcome);
        self
    }

    /// Set the provenance summary.
    pub fn with_provenance(mut self, prov: ProvenanceSummary) -> Self {
        self.provenance = Some(prov);
        self
    }

    /// Set the memory address.
    pub fn with_address(mut self, addr: u16) -> Self {
        self.address = Some(addr);
        self
    }

    /// Set the value.
    pub fn with_value(mut self, val: u64) -> Self {
        self.value = Some(val);
        self
    }

    /// Set the state hash.
    pub fn with_state_hash(mut self, hash: u64) -> Self {
        self.state_hash = Some(hash);
        self
    }

    /// Convenience: set both pure and temporal counts.
    pub fn with_provenance_counts(mut self, pure: u32, temporal: u32) -> Self {
        self.provenance = Some(ProvenanceSummary::with_counts(pure, temporal));
        self
    }

    /// Format timestamp as ISO 8601.
    pub fn timestamp_iso(&self) -> String {
        let secs = self.timestamp_ms / 1000;
        let ms = self.timestamp_ms % 1000;

        // Simple UTC formatting (no chrono dependency)
        let days_since_epoch = secs / 86400;
        let time_of_day = secs % 86400;

        let hours = time_of_day / 3600;
        let minutes = (time_of_day % 3600) / 60;
        let seconds = time_of_day % 60;

        // Approximate date calculation (good enough for logging)
        let mut year = 1970u32;
        let mut remaining_days = days_since_epoch;

        loop {
            let days_in_year = if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
                366
            } else {
                365
            };
            if remaining_days < days_in_year {
                break;
            }
            remaining_days -= days_in_year;
            year += 1;
        }

        let days_in_months: [u64; 12] = if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
            [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        } else {
            [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        };

        let mut month = 1u32;
        for days in days_in_months.iter() {
            if remaining_days < *days {
                break;
            }
            remaining_days -= *days;
            month += 1;
        }

        let day = remaining_days + 1;

        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
            year, month, day, hours, minutes, seconds, ms
        )
    }

    /// Format as a single structured log line.
    ///
    /// Format: `TIMESTAMP | SEVERITY | CATEGORY | ACTION | ENTITY | OUTCOME | DESCRIPTION [metadata]`
    ///
    /// For temporal entries, includes epoch number in brackets.
    pub fn format_line(&self) -> String {
        // Include epoch prefix for temporal entries
        let epoch_prefix = self
            .epoch
            .map(|e| format!("E{:04} ", e))
            .unwrap_or_default();

        let mut line = format!(
            "{}{} | {:8} | {:12} | {:16} | {}:{} | {:8} | {}",
            epoch_prefix,
            self.timestamp_iso(),
            self.severity.as_str(),
            self.category.as_str(),
            self.action,
            self.entity_type,
            self.entity_id,
            self.outcome.as_str(),
            self.description,
        );

        // Add duration if present
        if let Some(us) = self.duration_us {
            if us >= 1_000_000 {
                line.push_str(&format!(" [{:.2}s]", us as f64 / 1_000_000.0));
            } else if us >= 1_000 {
                line.push_str(&format!(" [{:.2}ms]", us as f64 / 1_000.0));
            } else {
                line.push_str(&format!(" [{}us]", us));
            }
        }

        // Add correlation if present
        if let Some(ref cid) = self.correlation_id {
            line.push_str(&format!(" [cid:{}]", cid));
        }

        // Add convergence outcome if present and different from generic outcome
        if let Some(conv) = &self.convergence_outcome {
            line.push_str(&format!(" [conv:{}]", conv));
        }

        // Add provenance summary if present
        if let Some(ref prov) = self.provenance {
            line.push_str(&format!(" [{}]", prov.format_compact()));
        }

        // Add address/value for temporal memory operations
        if let (Some(addr), Some(val)) = (self.address, self.value) {
            line.push_str(&format!(" [@{}={}]", addr, val));
        }

        // Add state hash if present
        if let Some(hash) = self.state_hash {
            line.push_str(&format!(" [h:{:016x}]", hash));
        }

        // Add metadata if present
        if !self.metadata.is_empty() {
            let meta: Vec<String> = self
                .metadata
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            line.push_str(&format!(" {{{}}}", meta.join(", ")));
        }

        line
    }

    /// Serialize to JSON format.
    pub fn to_json(&self) -> String {
        let mut json = format!(
            r#"{{"id":{},"timestamp":"{}","severity":"{}","category":"{}","action":"{}","entity_type":"{}","entity_id":"{}","outcome":"{}","description":"{}"#,
            self.id,
            self.timestamp_iso(),
            self.severity.as_str(),
            self.category.as_str(),
            escape_json(&self.action),
            escape_json(&self.entity_type),
            escape_json(&self.entity_id),
            self.outcome.as_str(),
            escape_json(&self.description),
        );

        if let Some(us) = self.duration_us {
            json.push_str(&format!(r#","duration_us":{}"#, us));
        }

        if let Some(ref cid) = self.correlation_id {
            json.push_str(&format!(r#","correlation_id":"{}""#, escape_json(cid)));
        }

        // Temporal extensions
        if let Some(epoch) = self.epoch {
            json.push_str(&format!(r#","epoch":{}"#, epoch));
        }

        if let Some(ref conv) = self.convergence_outcome {
            json.push_str(&format!(r#","convergence_outcome":"{}""#, conv.as_str()));
        }

        if let Some(ref prov) = self.provenance {
            json.push_str(&format!(
                r#","provenance":{{"pure":{},"temporal":{},"max_depth":{},"oracle_sources":{}}}"#,
                prov.pure_count, prov.temporal_count, prov.max_depth, prov.oracle_sources
            ));
        }

        if let Some(addr) = self.address {
            json.push_str(&format!(r#","address":{}"#, addr));
        }

        if let Some(val) = self.value {
            json.push_str(&format!(r#","value":{}"#, val));
        }

        if let Some(hash) = self.state_hash {
            json.push_str(&format!(r#","state_hash":{}"#, hash));
        }

        if !self.metadata.is_empty() {
            json.push_str(r#","metadata":{"#);
            let pairs: Vec<String> = self
                .metadata
                .iter()
                .map(|(k, v)| format!(r#""{}":"{}""#, escape_json(k), escape_json(v)))
                .collect();
            json.push_str(&pairs.join(","));
            json.push('}');
        }

        json.push('}');
        json
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_entry_creation() {
        let entry = AuditEntry::new("PARSE", "Program", "test.ouro", "Parsed successfully");

        assert_eq!(entry.action, "PARSE");
        assert_eq!(entry.entity_type, "Program");
        assert_eq!(entry.entity_id, "test.ouro");
        assert_eq!(entry.severity, Severity::Info);
        assert_eq!(entry.outcome, Outcome::Success);
        assert!(entry.epoch.is_none());
    }

    #[test]
    fn test_audit_entry_builder() {
        let entry = AuditEntry::new("EXECUTE", "Epoch", "0", "Executed epoch")
            .with_category(ActionCategory::Execute)
            .with_severity(Severity::Warning)
            .with_outcome(Outcome::Partial)
            .with_duration_us(1500)
            .with_correlation("tx-12345")
            .with_meta("instructions", "42");

        assert_eq!(entry.category, ActionCategory::Execute);
        assert_eq!(entry.severity, Severity::Warning);
        assert_eq!(entry.outcome, Outcome::Partial);
        assert_eq!(entry.duration_us, Some(1500));
        assert_eq!(entry.correlation_id, Some("tx-12345".to_string()));
        assert_eq!(entry.metadata.get("instructions"), Some(&"42".to_string()));
    }

    #[test]
    fn test_temporal_entry_builder() {
        let entry = AuditEntry::new("ORACLE", "Memory", "42", "Read from anamnesis")
            .with_category(ActionCategory::OracleRead)
            .with_epoch(5)
            .with_address(42)
            .with_value(100)
            .with_provenance_counts(10, 5);

        assert_eq!(entry.category, ActionCategory::OracleRead);
        assert_eq!(entry.epoch, Some(5));
        assert_eq!(entry.address, Some(42));
        assert_eq!(entry.value, Some(100));

        let prov = entry.provenance.as_ref().unwrap();
        assert_eq!(prov.pure_count, 10);
        assert_eq!(prov.temporal_count, 5);
    }

    #[test]
    fn test_oracle_read_factory() {
        let entry = AuditEntry::oracle_read(42, 100, 5, true);

        assert_eq!(entry.category, ActionCategory::OracleRead);
        assert_eq!(entry.epoch, Some(5));
        assert_eq!(entry.address, Some(42));
        assert_eq!(entry.value, Some(100));
        assert_eq!(entry.metadata.get("temporal"), Some(&"true".to_string()));
    }

    #[test]
    fn test_prophecy_write_factory() {
        let entry = AuditEntry::prophecy_write(10, 200, 50, 3);

        assert_eq!(entry.category, ActionCategory::ProphecyWrite);
        assert_eq!(entry.epoch, Some(3));
        assert_eq!(entry.address, Some(10));
        assert_eq!(entry.value, Some(200));
        assert_eq!(entry.metadata.get("previous"), Some(&"50".to_string()));
    }

    #[test]
    fn test_convergence_result_factory() {
        let entry = AuditEntry::convergence_result(
            ConvergenceOutcome::Consistent,
            10,
            5_000_000,
        );

        assert_eq!(entry.category, ActionCategory::ConvergenceCheck);
        assert_eq!(entry.convergence_outcome, Some(ConvergenceOutcome::Consistent));
        assert_eq!(entry.outcome, Outcome::Success);
        assert_eq!(entry.epoch, Some(9)); // total_epochs - 1
    }

    #[test]
    fn test_format_line() {
        let entry = AuditEntry::new("TEST", "Unit", "test_1", "Test passed")
            .with_duration_us(500)
            .with_meta("assertions", "3");

        let line = entry.format_line();

        assert!(line.contains("TEST"));
        assert!(line.contains("Unit:test_1"));
        assert!(line.contains("SUCCESS"));
        assert!(line.contains("500us"));
        assert!(line.contains("assertions=3"));
    }

    #[test]
    fn test_format_line_with_epoch() {
        let entry = AuditEntry::new("ORACLE", "Memory", "42", "Read value")
            .with_epoch(5)
            .with_address(42)
            .with_value(100);

        let line = entry.format_line();

        assert!(line.contains("E0005"));
        assert!(line.contains("@42=100"));
    }

    #[test]
    fn test_json_output() {
        let entry = AuditEntry::new("TEST", "Unit", "test_2", "Description with \"quotes\"");

        let json = entry.to_json();

        assert!(json.contains("\"action\":\"TEST\""));
        assert!(json.contains("\\\"quotes\\\""));
    }

    #[test]
    fn test_json_output_with_temporal() {
        let entry = AuditEntry::new("ORACLE", "Memory", "42", "Read")
            .with_epoch(5)
            .with_address(42)
            .with_value(100)
            .with_convergence_outcome(ConvergenceOutcome::Iterating)
            .with_provenance_counts(10, 5);

        let json = entry.to_json();

        assert!(json.contains("\"epoch\":5"));
        assert!(json.contains("\"address\":42"));
        assert!(json.contains("\"value\":100"));
        assert!(json.contains("\"convergence_outcome\":\"ITERATING\""));
        assert!(json.contains("\"provenance\":{"));
        assert!(json.contains("\"pure\":10"));
        assert!(json.contains("\"temporal\":5"));
    }

    #[test]
    fn test_severity_ordering() {
        assert!((Severity::Info as u8) < (Severity::Warning as u8));
        assert!((Severity::Warning as u8) < (Severity::Error as u8));
        assert!((Severity::Error as u8) < (Severity::Critical as u8));
    }

    #[test]
    fn test_action_category_temporal() {
        assert!(ActionCategory::OracleRead.is_temporal());
        assert!(ActionCategory::ProphecyWrite.is_temporal());
        assert!(ActionCategory::PresentRead.is_temporal());
        assert!(ActionCategory::Paradox.is_temporal());
        assert!(!ActionCategory::Parse.is_temporal());
        assert!(!ActionCategory::Execute.is_temporal());
    }

    #[test]
    fn test_action_category_convergence() {
        assert!(ActionCategory::EpochIteration.is_convergence());
        assert!(ActionCategory::ConvergenceCheck.is_convergence());
        assert!(ActionCategory::CycleDetection.is_convergence());
        assert!(!ActionCategory::Execute.is_convergence());
    }

    #[test]
    fn test_convergence_outcome() {
        assert!(ConvergenceOutcome::Consistent.is_success());
        assert!(ConvergenceOutcome::Consistent.is_terminal());
        assert!(!ConvergenceOutcome::Iterating.is_terminal());
        assert!(!ConvergenceOutcome::Oscillating.is_success());
        assert!(ConvergenceOutcome::Oscillating.is_terminal());

        assert_eq!(ConvergenceOutcome::Consistent.to_outcome(), Outcome::Success);
        assert_eq!(ConvergenceOutcome::Oscillating.to_outcome(), Outcome::Failure);
    }

    #[test]
    fn test_provenance_summary() {
        let prov = ProvenanceSummary::with_counts(80, 20);
        assert!((prov.temporal_ratio() - 0.2).abs() < 1e-10);
        assert_eq!(prov.format_compact(), "P80T20D0O0");
    }
}
