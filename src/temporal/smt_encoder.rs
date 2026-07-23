//! Backwards-compatible source-AST SMT exporter used as a parity oracle.
//!
//! Production global solving and the CLI lower verified linked bytecode into
//! temporal IR. This historical library facade retains the older source
//! compiler for differential tests; it is not an executable or proof
//! authority, and supported-case parity is regression-tested explicitly.

use crate::ast::Program;
use crate::core::BoundsPolicy;
use crate::temporal::ir::{TemporalIrCompiler, TemporalIrConfig};

#[derive(Debug, Clone, Copy)]
pub struct SmtEncoder {
    max_unroll: usize,
    memory_cells: usize,
    bounds_policy: BoundsPolicy,
}

impl SmtEncoder {
    pub fn new() -> Self {
        Self {
            max_unroll: 10,
            memory_cells: crate::core::MEMORY_SIZE,
            bounds_policy: BoundsPolicy::Wrap,
        }
    }

    pub fn with_unroll_limit(max_unroll: usize) -> Self {
        Self {
            max_unroll,
            ..Self::new()
        }
    }

    pub fn with_memory_cells(memory_cells: usize) -> Self {
        assert!(memory_cells > 0, "memory must contain at least one cell");
        Self {
            memory_cells,
            ..Self::new()
        }
    }

    pub fn with_bounds_policy(mut self, bounds_policy: BoundsPolicy) -> Self {
        self.bounds_policy = bounds_policy;
        self
    }

    pub fn encode(&mut self, program: &Program) -> Result<String, String> {
        TemporalIrCompiler::compile(
            program,
            TemporalIrConfig {
                memory_cells: self.memory_cells,
                loop_unroll_limit: self.max_unroll,
                bounds_policy: self.bounds_policy,
            },
        )
        .map(|ir| ir.to_smt2(true))
        .map_err(|error| error.to_string())
    }
}

impl Default for SmtEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    #[test]
    fn simple_export_uses_typed_ir() {
        let program = parse("0 ORACLE 0 PROPHECY").unwrap();
        let smt = SmtEncoder::new().encode(&program).unwrap();
        assert!(smt.contains("Ourochronos typed temporal IR"));
        assert!(smt.contains("(declare-const anamnesis"));
        assert!(smt.contains("(check-sat)"));
    }

    #[test]
    fn unsupported_constructs_are_errors_not_noops() {
        for (source, needle) in [
            ("0 ORACLE 0 PROPHECY VEC_NEW POP", "VEC_NEW"),
            ("0 ORACLE 0 PROPHECY CLOCK POP", "CLOCK"),
            ("[ 1 ] EXEC 0 ORACLE 0 PROPHECY", "quotation reference 0"),
        ] {
            let program = parse(source).unwrap();
            let error = SmtEncoder::new().encode(&program).expect_err(source);
            assert!(error.contains(needle), "{}: {}", source, error);
        }
    }

    #[test]
    fn paradox_and_typed_scopes_are_first_class_ir_constructs() {
        let program =
            parse("TEMPORAL 2 1 BITS 2 { 0 ORACLE IF { 1 0 PROPHECY } ELSE { PARADOX } }").unwrap();
        let smt = SmtEncoder::with_memory_cells(4).encode(&program).unwrap();
        assert!(smt.contains("(assert e"));
        assert!(smt.contains("bvand"));
    }

    #[test]
    fn logical_not_is_not_bitwise_not() {
        let program = parse("0 ORACLE NOT 0 PROPHECY").unwrap();
        let smt = SmtEncoder::new().encode(&program).unwrap();
        assert!(!smt.contains("bvnot"));
        assert!(smt.contains("ite"));
    }

    #[test]
    fn non_power_of_two_memory_uses_exact_modulo_addressing() {
        let program = parse("42 65536 PROPHECY").unwrap();
        let smt = SmtEncoder::with_memory_cells(70_000)
            .encode(&program)
            .unwrap();
        assert!(smt.contains("(Array (_ BitVec 64) (_ BitVec 64))"));
        assert!(smt.contains("(_ bv70000 64)"));
        assert!(smt.contains("bvurem"));
    }

    #[test]
    fn loop_header_records_the_exact_bound() {
        let program = parse("WHILE { 0 } { NOP } 0 0 PROPHECY").unwrap();
        let smt = SmtEncoder::with_unroll_limit(7).encode(&program).unwrap();
        assert!(smt.contains("BoundedLoops"));
        assert!(smt.contains("unroll_limit: 7"));
    }
}
