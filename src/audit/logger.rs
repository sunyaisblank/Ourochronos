//! Audit logger implementation.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use super::entry::{AuditEntry, Severity, Outcome};

// =============================================================================
// Configuration
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

// =============================================================================
// Audit Logger
// =============================================================================

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stdout_logger() {
        let logger = AuditLogger::stdout_only();
        // Should not panic
        let _ = logger.info("TEST", "Logger", "stdout", "Testing stdout output");
    }
}
