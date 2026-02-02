//! Global audit logger singleton.

use std::sync::OnceLock;
use super::entry::{AuditEntry, Severity, Outcome};
use super::logger::{AuditLogger, AuditConfig};

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
