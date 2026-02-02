//! Keyword lookup map for fast domain parser dispatch.
//!
//! Maps uppercase keywords to their corresponding domain for fast lookup.

use super::domain::{DomainParser, ParseContext};
use super::stack_ops::StackOpsParser;
use super::arithmetic::ArithmeticParser;
use super::temporal::TemporalParser;
use super::data_structures::DataStructuresParser;
use super::io_ops::IoOpsParser;
use super::string_ops::StringOpsParser;
use crate::ast::Stmt;

/// Identifies which domain parser handles a keyword.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Domain {
    Stack,
    Arithmetic,
    Temporal,
    DataStructures,
    Io,
    String,
}

/// Registry of all domain parsers with efficient keyword lookup.
pub struct DomainRegistry {
    stack: StackOpsParser,
    arithmetic: ArithmeticParser,
    temporal: TemporalParser,
    data_structures: DataStructuresParser,
    io: IoOpsParser,
    string: StringOpsParser,
}

impl Default for DomainRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl DomainRegistry {
    pub fn new() -> Self {
        Self {
            stack: StackOpsParser,
            arithmetic: ArithmeticParser,
            temporal: TemporalParser,
            data_structures: DataStructuresParser,
            io: IoOpsParser,
            string: StringOpsParser,
        }
    }

    /// Look up the domain for a keyword using binary search on static arrays.
    pub fn lookup_domain(&self, keyword: &str) -> Option<Domain> {
        // Check each domain's keywords
        // Using contains() on static slices which is O(n) but n is small
        if StackOpsParser::KEYWORDS.contains(&keyword) {
            return Some(Domain::Stack);
        }
        if ArithmeticParser::KEYWORDS.contains(&keyword) {
            return Some(Domain::Arithmetic);
        }
        if TemporalParser::KEYWORDS.contains(&keyword) {
            return Some(Domain::Temporal);
        }
        if DataStructuresParser::KEYWORDS.contains(&keyword) {
            return Some(Domain::DataStructures);
        }
        if IoOpsParser::KEYWORDS.contains(&keyword) {
            return Some(Domain::Io);
        }
        if StringOpsParser::KEYWORDS.contains(&keyword) {
            return Some(Domain::String);
        }
        None
    }

    /// Parse a keyword using the appropriate domain parser.
    pub fn parse(&self, keyword: &str, ctx: &mut ParseContext) -> Option<Result<Stmt, String>> {
        let domain = self.lookup_domain(keyword)?;

        let result = match domain {
            Domain::Stack => self.stack.parse(keyword, ctx),
            Domain::Arithmetic => self.arithmetic.parse(keyword, ctx),
            Domain::Temporal => self.temporal.parse(keyword, ctx),
            Domain::DataStructures => self.data_structures.parse(keyword, ctx),
            Domain::Io => self.io.parse(keyword, ctx),
            Domain::String => self.string.parse(keyword, ctx),
        };

        Some(result)
    }

    /// Check if a keyword is handled by any domain parser.
    pub fn handles(&self, keyword: &str) -> bool {
        self.lookup_domain(keyword).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_lookup() {
        let registry = DomainRegistry::new();
        assert_eq!(registry.lookup_domain("DUP"), Some(Domain::Stack));
        assert_eq!(registry.lookup_domain("ADD"), Some(Domain::Arithmetic));
        assert_eq!(registry.lookup_domain("ORACLE"), Some(Domain::Temporal));
        assert_eq!(registry.lookup_domain("VEC_NEW"), Some(Domain::DataStructures));
        assert_eq!(registry.lookup_domain("FILE_OPEN"), Some(Domain::Io));
        assert_eq!(registry.lookup_domain("STR_REV"), Some(Domain::String));
    }

    #[test]
    fn test_registry_handles() {
        let registry = DomainRegistry::new();
        assert!(registry.handles("DUP"));
        assert!(registry.handles("ADD"));
        assert!(registry.handles("ORACLE"));
        assert!(!registry.handles("UNKNOWN"));
        assert!(!registry.handles("IF")); // Control flow not in domain parsers
    }

    #[test]
    fn test_all_domains_covered() {
        let registry = DomainRegistry::new();
        // Sample from each domain to verify coverage
        assert!(registry.handles("NOP"));       // Stack
        assert!(registry.handles("MUL"));       // Arithmetic
        assert!(registry.handles("PROPHECY"));  // Temporal
        assert!(registry.handles("HASH_PUT"));  // DataStructures
        assert!(registry.handles("CLOCK"));     // IO
        assert!(registry.handles("CONCAT"));    // String
    }
}
