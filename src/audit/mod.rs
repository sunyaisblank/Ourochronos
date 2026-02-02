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
//!
//! # Example
//!
//! ```ignore
//! use ourochronos::audit::{AuditLogger, AuditEntry, Severity, Outcome};
//!
//! let logger = AuditLogger::new("/path/to/audit.log")?;
//! logger.log(AuditEntry::new(
//!     "COMPILE",
//!     "Program",
//!     "fibonacci.ouro",
//!     "Compiled program with 42 instructions",
//! ).with_outcome(Outcome::Success))?;
//! ```

mod entry;
mod logger;
mod global;

pub use entry::{AuditEntry, Severity, Outcome, ActionCategory};
pub use logger::{AuditLogger, AuditConfig, AuditFormat};
pub use global::{init_global_logger, init_stdout_logger, global_logger, audit, audit_info, audit_warn, audit_error};

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
        $(entry = entry.with_meta(stringify!($key), $val.to_string());)*
        $crate::audit::audit(entry);
    }};
}
