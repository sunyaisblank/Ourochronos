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
}
