//! Structured Audit Logging for Ourochronos.
//!
//! Component: AUDT001A
//! Domain: AU (Audit) | Category: DT (Data Types)
//!
//! Provides immutable, queryable audit logging following event sourcing principles.
//! All entries are append-only and include structured fields for traceability.
//!
//! # Design Principles
//!
//! - **Immutable**: Entries cannot be modified after creation
//! - **Structured**: WHO/WHAT/WHEN/OUTCOME pattern for all events
//! - **Queryable**: Support for filtering by time, entity, correlation
//! - **Clean Output**: Human-readable with machine-parseable structure
//! - **Temporal-Aware**: First-class support for temporal computation auditing
//! - **Numerically Precise**: Statistical computations use stable algorithms
//!
//! # Modules
//!
//! - `entry`: Core audit entry types (AuditEntry, Severity, Outcome, ActionCategory)
//! - `logger`: Thread-safe logging with batch support
//! - `global`: Global singleton logger for application-wide use
//! - `temporal_events`: Domain-specific events for temporal operations
//! - `metrics`: Statistical metrics with numerical stability guarantees
//! - `integration`: Bridges to provenance, error handling, and timeloop
//!
//! # Example
//!
//! ```ignore
//! use ourochronos::audit::{AuditLogger, AuditEntry, Severity, Outcome, ActionCategory};
//!
//! // Basic logging
//! let logger = AuditLogger::new("/path/to/audit.log")?;
//! logger.log(AuditEntry::new(
//!     "COMPILE",
//!     "Program",
//!     "fibonacci.ouro",
//!     "Compiled program with 42 instructions",
//! ).with_outcome(Outcome::Success))?;
//!
//! // Temporal operation logging
//! let entry = AuditEntry::oracle_read(42, 100, 5, true);
//! logger.log(entry)?;
//!
//! // Batch logging for high-frequency operations
//! use ourochronos::audit::{BatchLogger, EpochLogger};
//! let epoch_logger = EpochLogger::new(Arc::new(logger), "correlation-id");
//! ```

mod entry;
mod logger;
mod global;
mod temporal_events;
mod metrics;
mod integration;

// Core entry types
pub use entry::{
    AuditEntry, Severity, Outcome, ActionCategory,
    ConvergenceOutcome, ProvenanceSummary,
};

// Logger types
pub use logger::{
    AuditLogger, AuditConfig, AuditFormat,
    BatchLogger, BatchLoggerStats, EpochLogger,
    generate_correlation_id,
};

// Global logger functions
pub use global::{
    init_global_logger, init_stdout_logger, global_logger,
    audit, audit_info, audit_warn, audit_error,
};

// Temporal event types
pub use temporal_events::{
    TemporalOperation, TemporalEvent,
    EpochOutcome, EpochEvent,
    ConvergenceEvent,
    ActionPrincipleEvent, ActionBreakdown,
    ProvenanceEvent,
};

// Metrics types
pub use metrics::{
    OnlineStatistics, DistributionalMeasures,
    ConvergenceMetrics, ProvenanceMetrics,
    ErrorAccumulator,
};

// Integration functions
pub use integration::{
    provenance_to_summary, aggregate_provenance, provenance_to_metadata,
    error_to_entry,
    convergence_status_to_outcome, epoch_outcome_to_convergence,
    epoch_event_to_entry, convergence_event_to_entry,
    AuditSession,
};

/// Escape a string for JSON.
pub(crate) fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

// =============================================================================
// Macros for Convenient Logging
// =============================================================================

/// Log an audit entry with structured fields.
///
/// # Example
///
/// ```ignore
/// audit_log!(
///     action = "PARSE",
///     entity = ("Program", "test.ouro"),
///     description = "Parsed 42 statements",
///     category = Parse,
///     outcome = Success,
/// );
/// ```
#[macro_export]
macro_rules! audit_log {
    (
        action = $action:expr,
        entity = ($etype:expr, $eid:expr),
        description = $desc:expr
        $(, category = $cat:ident)?
        $(, severity = $sev:ident)?
        $(, outcome = $out:ident)?
        $(, duration_us = $dur:expr)?
        $(, correlation = $cid:expr)?
        $(, epoch = $epoch:expr)?
        $(, $key:ident = $val:expr)*
        $(,)?
    ) => {{
        #[allow(unused_mut)]
        let mut entry = $crate::audit::AuditEntry::new($action, $etype, $eid, $desc);
        $(entry = entry.with_category($crate::audit::ActionCategory::$cat);)?
        $(entry = entry.with_severity($crate::audit::Severity::$sev);)?
        $(entry = entry.with_outcome($crate::audit::Outcome::$out);)?
        $(entry = entry.with_duration_us($dur);)?
        $(entry = entry.with_correlation($cid);)?
        $(entry = entry.with_epoch($epoch);)?
        $(entry = entry.with_meta(stringify!($key), $val.to_string());)*
        $crate::audit::audit(entry);
    }};
}

/// Log an ORACLE read operation.
///
/// # Example
///
/// ```ignore
/// audit_oracle!(address: 42, value: 100, epoch: 5, temporal: true);
/// ```
#[macro_export]
macro_rules! audit_oracle {
    (address: $addr:expr, value: $val:expr, epoch: $epoch:expr, temporal: $temp:expr) => {{
        let entry = $crate::audit::AuditEntry::oracle_read($addr, $val, $epoch, $temp);
        $crate::audit::audit(entry);
    }};
    (address: $addr:expr, value: $val:expr, epoch: $epoch:expr, temporal: $temp:expr, correlation: $cid:expr) => {{
        let entry = $crate::audit::AuditEntry::oracle_read($addr, $val, $epoch, $temp)
            .with_correlation($cid);
        $crate::audit::audit(entry);
    }};
}

/// Log a PROPHECY write operation.
///
/// # Example
///
/// ```ignore
/// audit_prophecy!(address: 10, value: 200, previous: 50, epoch: 3);
/// ```
#[macro_export]
macro_rules! audit_prophecy {
    (address: $addr:expr, value: $val:expr, previous: $prev:expr, epoch: $epoch:expr) => {{
        let entry = $crate::audit::AuditEntry::prophecy_write($addr, $val, $prev, $epoch);
        $crate::audit::audit(entry);
    }};
    (address: $addr:expr, value: $val:expr, previous: $prev:expr, epoch: $epoch:expr, correlation: $cid:expr) => {{
        let entry = $crate::audit::AuditEntry::prophecy_write($addr, $val, $prev, $epoch)
            .with_correlation($cid);
        $crate::audit::audit(entry);
    }};
}

/// Log an epoch completion.
///
/// # Example
///
/// ```ignore
/// audit_epoch!(epoch: 5, duration_us: 1500, instructions: 1000, changed: 3);
/// ```
#[macro_export]
macro_rules! audit_epoch {
    (epoch: $epoch:expr, duration_us: $dur:expr, instructions: $inst:expr, changed: $changed:expr) => {{
        let entry = $crate::audit::AuditEntry::epoch_iteration($epoch, $dur, $inst, $changed);
        $crate::audit::audit(entry);
    }};
    (epoch: $epoch:expr, duration_us: $dur:expr, instructions: $inst:expr, changed: $changed:expr, correlation: $cid:expr) => {{
        let entry = $crate::audit::AuditEntry::epoch_iteration($epoch, $dur, $inst, $changed)
            .with_correlation($cid);
        $crate::audit::audit(entry);
    }};
}

/// Log a convergence result.
///
/// # Example
///
/// ```ignore
/// audit_convergence!(outcome: Consistent, epochs: 10, duration_us: 50000);
/// ```
#[macro_export]
macro_rules! audit_convergence {
    (outcome: $outcome:ident, epochs: $epochs:expr, duration_us: $dur:expr) => {{
        let entry = $crate::audit::AuditEntry::convergence_result(
            $crate::audit::ConvergenceOutcome::$outcome,
            $epochs,
            $dur,
        );
        $crate::audit::audit(entry);
    }};
    (outcome: $outcome:ident, epochs: $epochs:expr, duration_us: $dur:expr, correlation: $cid:expr) => {{
        let entry = $crate::audit::AuditEntry::convergence_result(
            $crate::audit::ConvergenceOutcome::$outcome,
            $epochs,
            $dur,
        ).with_correlation($cid);
        $crate::audit::audit(entry);
    }};
}
