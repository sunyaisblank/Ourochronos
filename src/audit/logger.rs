//! Audit logger implementation.
//!
//! Provides thread-safe, high-performance audit logging with:
//! - Buffered file I/O for efficiency
//! - Batch logging for high-frequency operations
//! - Severity-based filtering and routing
//! - Multiple output formats (text, JSON Lines)
//!
//! # Performance Considerations
//!
//! For high-frequency temporal operations (ORACLE/PROPHECY), use the
//! `BatchLogger` to accumulate entries and flush them in batches,
//! reducing lock contention and I/O overhead.

use std::collections::VecDeque;
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

    /// Log multiple entries atomically.
    ///
    /// This is more efficient than logging entries individually
    /// as it only acquires the lock once.
    pub fn log_batch(&self, entries: &[AuditEntry]) -> std::io::Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        // Pre-format all entries
        let mut lines = String::with_capacity(entries.len() * 256);
        let mut seq = self.sequence.load(std::sync::atomic::Ordering::SeqCst);

        for entry in entries {
            // Check severity filter
            if (entry.severity as u8) < (self.config.min_severity as u8) {
                continue;
            }

            let line = match self.config.format {
                AuditFormat::Text => format!("{:08} | {}\n", seq, entry.format_line()),
                AuditFormat::JsonLines => format!("{}\n", entry.to_json()),
            };
            lines.push_str(&line);
            seq += 1;
        }

        // Update sequence counter
        self.sequence.store(seq, std::sync::atomic::Ordering::SeqCst);

        // Write all lines at once
        if let Ok(mut guard) = self.writer.lock() {
            if let Some(ref mut w) = *guard {
                w.write_all(lines.as_bytes())?;
                w.flush()?;
            }
        }

        // Echo to stdout if configured
        if self.config.echo_stdout {
            print!("{}", lines);
        }

        Ok(())
    }

    /// Get the current sequence number.
    pub fn sequence(&self) -> u64 {
        self.sequence.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Get configuration reference.
    pub fn config(&self) -> &AuditConfig {
        &self.config
    }
}

// =============================================================================
// Batch Logger
// =============================================================================

/// Batch logger for high-frequency operations.
///
/// Accumulates entries in a buffer and flushes them in batches to reduce
/// lock contention and I/O overhead. Ideal for logging temporal operations
/// (ORACLE/PROPHECY) during epoch execution.
///
/// # Usage
///
/// ```ignore
/// let mut batch = BatchLogger::new(logger, 100);
/// batch.push(entry1);
/// batch.push(entry2);
/// batch.flush()?; // Or rely on auto-flush when buffer is full
/// ```
pub struct BatchLogger {
    /// Underlying logger.
    logger: Arc<AuditLogger>,
    /// Buffer of pending entries.
    buffer: VecDeque<AuditEntry>,
    /// Maximum buffer size before auto-flush.
    max_size: usize,
    /// Total entries logged.
    total_logged: u64,
    /// Total flushes performed.
    flush_count: u64,
}

impl BatchLogger {
    /// Create a new batch logger.
    ///
    /// # Arguments
    /// - `logger`: The underlying audit logger
    /// - `max_size`: Maximum entries to buffer before auto-flush
    pub fn new(logger: Arc<AuditLogger>, max_size: usize) -> Self {
        Self {
            logger,
            buffer: VecDeque::with_capacity(max_size),
            max_size,
            total_logged: 0,
            flush_count: 0,
        }
    }

    /// Create with default buffer size (100 entries).
    pub fn with_default_size(logger: Arc<AuditLogger>) -> Self {
        Self::new(logger, 100)
    }

    /// Add an entry to the buffer.
    ///
    /// Automatically flushes if buffer exceeds max_size.
    pub fn push(&mut self, entry: AuditEntry) -> std::io::Result<()> {
        self.buffer.push_back(entry);
        self.total_logged += 1;

        if self.buffer.len() >= self.max_size {
            self.flush()?;
        }

        Ok(())
    }

    /// Flush all buffered entries to the logger.
    pub fn flush(&mut self) -> std::io::Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        let entries: Vec<AuditEntry> = self.buffer.drain(..).collect();
        self.logger.log_batch(&entries)?;
        self.flush_count += 1;

        Ok(())
    }

    /// Number of entries currently in buffer.
    pub fn buffered(&self) -> usize {
        self.buffer.len()
    }

    /// Total entries logged through this batch logger.
    pub fn total_logged(&self) -> u64 {
        self.total_logged
    }

    /// Number of flushes performed.
    pub fn flush_count(&self) -> u64 {
        self.flush_count
    }

    /// Get statistics summary.
    pub fn stats(&self) -> BatchLoggerStats {
        BatchLoggerStats {
            total_logged: self.total_logged,
            flush_count: self.flush_count,
            buffered: self.buffer.len(),
            avg_batch_size: if self.flush_count > 0 {
                (self.total_logged - self.buffer.len() as u64) as f64 / self.flush_count as f64
            } else {
                0.0
            },
        }
    }
}

impl Drop for BatchLogger {
    fn drop(&mut self) {
        // Flush remaining entries on drop
        let _ = self.flush();
    }
}

/// Statistics for batch logger.
#[derive(Debug, Clone)]
pub struct BatchLoggerStats {
    pub total_logged: u64,
    pub flush_count: u64,
    pub buffered: usize,
    pub avg_batch_size: f64,
}

impl std::fmt::Display for BatchLoggerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BatchLogger: logged={} flushes={} buffered={} avg_batch={:.1}",
            self.total_logged, self.flush_count, self.buffered, self.avg_batch_size
        )
    }
}

// =============================================================================
// Epoch Logger
// =============================================================================

/// Specialized logger for epoch execution tracing.
///
/// Provides convenience methods for logging temporal operations
/// during epoch execution with automatic correlation tracking.
pub struct EpochLogger {
    /// Underlying batch logger.
    batch: BatchLogger,
    /// Current correlation ID.
    correlation_id: String,
    /// Current epoch number.
    epoch: usize,
    /// Oracle operation count.
    oracle_count: u64,
    /// Prophecy operation count.
    prophecy_count: u64,
}

impl EpochLogger {
    /// Create a new epoch logger.
    pub fn new(logger: Arc<AuditLogger>, correlation_id: impl Into<String>) -> Self {
        Self {
            batch: BatchLogger::with_default_size(logger),
            correlation_id: correlation_id.into(),
            epoch: 0,
            oracle_count: 0,
            prophecy_count: 0,
        }
    }

    /// Set the current epoch.
    pub fn set_epoch(&mut self, epoch: usize) {
        self.epoch = epoch;
    }

    /// Log an ORACLE read operation.
    pub fn log_oracle(&mut self, address: u16, value: u64, is_temporal: bool) -> std::io::Result<()> {
        self.oracle_count += 1;
        let entry = AuditEntry::oracle_read(address, value, self.epoch, is_temporal)
            .with_correlation(&self.correlation_id);
        self.batch.push(entry)
    }

    /// Log a PROPHECY write operation.
    pub fn log_prophecy(&mut self, address: u16, value: u64, previous: u64) -> std::io::Result<()> {
        self.prophecy_count += 1;
        let entry = AuditEntry::prophecy_write(address, value, previous, self.epoch)
            .with_correlation(&self.correlation_id);
        self.batch.push(entry)
    }

    /// Log epoch completion.
    pub fn log_epoch_complete(
        &mut self,
        duration_us: u64,
        instructions: u64,
        changed_cells: usize,
    ) -> std::io::Result<()> {
        let entry = AuditEntry::epoch_iteration(self.epoch, duration_us, instructions, changed_cells)
            .with_correlation(&self.correlation_id)
            .with_meta("oracle_ops", self.oracle_count.to_string())
            .with_meta("prophecy_ops", self.prophecy_count.to_string());
        self.batch.push(entry)?;

        // Reset counters for next epoch
        self.oracle_count = 0;
        self.prophecy_count = 0;

        Ok(())
    }

    /// Flush all buffered entries.
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.batch.flush()
    }

    /// Get the correlation ID.
    pub fn correlation_id(&self) -> &str {
        &self.correlation_id
    }

    /// Get batch logger statistics.
    pub fn stats(&self) -> BatchLoggerStats {
        self.batch.stats()
    }
}

// =============================================================================
// Correlation ID Generation
// =============================================================================

/// Generate a unique correlation ID.
///
/// Format: `{prefix}-{timestamp_ms}-{counter}`
pub fn generate_correlation_id(prefix: &str) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    let count = COUNTER.fetch_add(1, Ordering::SeqCst);

    format!("{}-{:013x}-{:04x}", prefix, timestamp, count & 0xFFFF)
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

    #[test]
    fn test_batch_logger() {
        let logger = Arc::new(AuditLogger::stdout_only());
        let mut batch = BatchLogger::new(logger, 10);

        for i in 0..5 {
            let entry = AuditEntry::new("TEST", "Batch", i.to_string(), "Test entry");
            batch.push(entry).unwrap();
        }

        assert_eq!(batch.buffered(), 5);
        assert_eq!(batch.total_logged(), 5);
        assert_eq!(batch.flush_count(), 0);

        batch.flush().unwrap();

        assert_eq!(batch.buffered(), 0);
        assert_eq!(batch.flush_count(), 1);
    }

    #[test]
    fn test_batch_logger_auto_flush() {
        let logger = Arc::new(AuditLogger::stdout_only());
        let mut batch = BatchLogger::new(logger, 5);

        // Add 7 entries - should auto-flush at 5
        for i in 0..7 {
            let entry = AuditEntry::new("TEST", "Batch", i.to_string(), "Test entry");
            batch.push(entry).unwrap();
        }

        assert_eq!(batch.flush_count(), 1); // Auto-flushed once
        assert_eq!(batch.buffered(), 2); // 2 entries remaining
    }

    #[test]
    fn test_epoch_logger() {
        let logger = Arc::new(AuditLogger::stdout_only());
        let mut epoch_logger = EpochLogger::new(logger, "test-cid");

        epoch_logger.set_epoch(0);
        epoch_logger.log_oracle(10, 100, true).unwrap();
        epoch_logger.log_prophecy(20, 200, 0).unwrap();
        epoch_logger.log_epoch_complete(1000, 500, 5).unwrap();

        epoch_logger.set_epoch(1);
        epoch_logger.log_oracle(10, 100, true).unwrap();

        let stats = epoch_logger.stats();
        assert_eq!(stats.total_logged, 4);
    }

    #[test]
    fn test_correlation_id_generation() {
        let id1 = generate_correlation_id("ouro");
        let id2 = generate_correlation_id("ouro");

        assert!(id1.starts_with("ouro-"));
        assert!(id2.starts_with("ouro-"));
        assert_ne!(id1, id2); // Should be unique
    }

    #[test]
    fn test_log_batch() {
        let logger = AuditLogger::stdout_only();

        let entries: Vec<AuditEntry> = (0..3)
            .map(|i| AuditEntry::new("TEST", "Batch", i.to_string(), "Batch entry"))
            .collect();

        let result = logger.log_batch(&entries);
        assert!(result.is_ok());
        assert_eq!(logger.sequence(), 3);
    }
}
