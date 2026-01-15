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

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ActionCategory {
    /// Parsing operations.
    Parse,
    /// Compilation/analysis.
    Compile,
    /// Program execution.
    Execute,
    /// Temporal operations (ORACLE/PROPHECY).
    Temporal,
    /// Convergence/fixed-point search.
    Convergence,
    /// Configuration changes.
    Config,
    /// System-level events.
    System,
    /// Custom category.
    Custom(String),
}

impl ActionCategory {
    pub fn as_str(&self) -> &str {
        match self {
            ActionCategory::Parse => "PARSE",
            ActionCategory::Compile => "COMPILE",
            ActionCategory::Execute => "EXECUTE",
            ActionCategory::Temporal => "TEMPORAL",
            ActionCategory::Convergence => "CONVERGENCE",
            ActionCategory::Config => "CONFIG",
            ActionCategory::System => "SYSTEM",
            ActionCategory::Custom(s) => s,
        }
    }
}

impl std::fmt::Display for ActionCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// =============================================================================
// Audit Entry
// =============================================================================

/// An immutable audit log entry.
///
/// Captures WHO did WHAT to WHICH entity, WHEN, and with what OUTCOME.
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
        }
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
    pub fn format_line(&self) -> String {
        let mut line = format!(
            "{} | {:8} | {:12} | {:16} | {}:{} | {:8} | {}",
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

/// Escape a string for JSON.
fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

// =============================================================================
// Audit Logger
// =============================================================================

/// Configuration for the audit logger.
#[derive(Debug, Clone)]
pub struct AuditConfig {
    /// Path to the audit log file.
    pub log_path: PathBuf,
    /// Minimum severity to log.
    pub min_severity: Severity,
    /// Whether to also print to stdout.
    pub echo_stdout: bool,
    /// Output format.
    pub format: AuditFormat,
}

/// Output format for audit logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditFormat {
    /// Structured text lines (default).
    Text,
    /// JSON Lines format.
    JsonLines,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            log_path: PathBuf::from("ourochronos-audit.log"),
            min_severity: Severity::Info,
            echo_stdout: false,
            format: AuditFormat::Text,
        }
    }
}

/// Thread-safe audit logger with file-based persistence.
pub struct AuditLogger {
    config: AuditConfig,
    writer: Arc<Mutex<Option<BufWriter<File>>>>,
    sequence: std::sync::atomic::AtomicU64,
}

impl AuditLogger {
    /// Create a new audit logger with the given configuration.
    pub fn new(config: AuditConfig) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&config.log_path)?;

        let writer = BufWriter::new(file);

        Ok(Self {
            config,
            writer: Arc::new(Mutex::new(Some(writer))),
            sequence: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Create a logger that writes to the specified path.
    pub fn with_path(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let config = AuditConfig {
            log_path: path.as_ref().to_path_buf(),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create a logger that only echoes to stdout (no file).
    pub fn stdout_only() -> Self {
        Self {
            config: AuditConfig {
                echo_stdout: true,
                ..Default::default()
            },
            writer: Arc::new(Mutex::new(None)),
            sequence: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Log an audit entry.
    pub fn log(&self, entry: AuditEntry) -> std::io::Result<()> {
        // Check severity filter
        if (entry.severity as u8) < (self.config.min_severity as u8) {
            return Ok(());
        }

        let seq = self.sequence.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        let line = match self.config.format {
            AuditFormat::Text => format!("{:08} | {}\n", seq, entry.format_line()),
            AuditFormat::JsonLines => format!("{}\n", entry.to_json()),
        };

        // Write to file if configured
        if let Ok(mut guard) = self.writer.lock() {
            if let Some(ref mut w) = *guard {
                w.write_all(line.as_bytes())?;
                w.flush()?;
            }
        }

        // Echo to stdout if configured
        if self.config.echo_stdout {
            print!("{}", line);
        }

        Ok(())
    }

    /// Create a quick info entry and log it.
    pub fn info(
        &self,
        action: &str,
        entity_type: &str,
        entity_id: &str,
        description: &str,
    ) -> std::io::Result<()> {
        self.log(AuditEntry::new(action, entity_type, entity_id, description))
    }

    /// Create a quick warning entry and log it.
    pub fn warn(
        &self,
        action: &str,
        entity_type: &str,
        entity_id: &str,
        description: &str,
    ) -> std::io::Result<()> {
        self.log(
            AuditEntry::new(action, entity_type, entity_id, description)
                .with_severity(Severity::Warning),
        )
    }

    /// Create a quick error entry and log it.
    pub fn error(
        &self,
        action: &str,
        entity_type: &str,
        entity_id: &str,
        description: &str,
    ) -> std::io::Result<()> {
        self.log(
            AuditEntry::new(action, entity_type, entity_id, description)
                .with_severity(Severity::Error)
                .with_outcome(Outcome::Failure),
        )
    }

    /// Flush the log buffer.
    pub fn flush(&self) -> std::io::Result<()> {
        if let Ok(mut guard) = self.writer.lock() {
            if let Some(ref mut w) = *guard {
                w.flush()?;
            }
        }
        Ok(())
    }
}

// =============================================================================
// Global Logger
// =============================================================================

use std::sync::OnceLock;

static GLOBAL_LOGGER: OnceLock<AuditLogger> = OnceLock::new();

/// Initialize the global audit logger.
pub fn init_global_logger(config: AuditConfig) -> std::io::Result<()> {
    let logger = AuditLogger::new(config)?;
    GLOBAL_LOGGER
        .set(logger)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::AlreadyExists, "Logger already initialized"))
}

/// Initialize a stdout-only global logger (for CLI use).
pub fn init_stdout_logger() {
    let _ = GLOBAL_LOGGER.set(AuditLogger::stdout_only());
}

/// Get a reference to the global logger (if initialized).
pub fn global_logger() -> Option<&'static AuditLogger> {
    GLOBAL_LOGGER.get()
}

/// Log to the global logger (no-op if not initialized).
pub fn audit(entry: AuditEntry) {
    if let Some(logger) = global_logger() {
        let _ = logger.log(entry);
    }
}

/// Quick info log to global logger.
pub fn audit_info(action: &str, entity_type: &str, entity_id: &str, description: &str) {
    audit(AuditEntry::new(action, entity_type, entity_id, description));
}

/// Quick warning log to global logger.
pub fn audit_warn(action: &str, entity_type: &str, entity_id: &str, description: &str) {
    audit(
        AuditEntry::new(action, entity_type, entity_id, description)
            .with_severity(Severity::Warning),
    );
}

/// Quick error log to global logger.
pub fn audit_error(action: &str, entity_type: &str, entity_id: &str, description: &str) {
    audit(
        AuditEntry::new(action, entity_type, entity_id, description)
            .with_severity(Severity::Error)
            .with_outcome(Outcome::Failure),
    );
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
    fn test_json_output() {
        let entry = AuditEntry::new("TEST", "Unit", "test_2", "Description with \"quotes\"");

        let json = entry.to_json();

        assert!(json.contains("\"action\":\"TEST\""));
        assert!(json.contains("\\\"quotes\\\""));
    }

    #[test]
    fn test_severity_ordering() {
        assert!((Severity::Info as u8) < (Severity::Warning as u8));
        assert!((Severity::Warning as u8) < (Severity::Error as u8));
        assert!((Severity::Error as u8) < (Severity::Critical as u8));
    }

    #[test]
    fn test_stdout_logger() {
        let logger = AuditLogger::stdout_only();
        // Should not panic
        let _ = logger.info("TEST", "Logger", "stdout", "Testing stdout output");
    }
}
