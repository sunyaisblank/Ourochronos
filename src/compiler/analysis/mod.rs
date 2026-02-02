//! Analysis passes for OUROCHRONOS programs.
//!
//! This module provides unified access to various analysis passes:
//!
//! - **Type Checking**: Stack effect verification, type inference
//! - **Linearity Analysis**: Affine type enforcement for ORACLE results
//! - **Temporal Analysis**: Dependency graph construction, causality checking
//! - **Provenance Analysis**: Tracking data flow from temporal operations
//! - **Effect Analysis**: Side effect verification
//!
//! # Analysis Pipeline
//!
//! ```text
//! Program → Type Check → Linearity Check → Temporal Analysis → Results
//!              ↓               ↓                   ↓
//!           Types          Warnings             TDG
//! ```

// Re-export from analysis module
pub use crate::analysis::{
    TemporalDependencyGraph, NegativeLoop,
};

// Re-export type checking
pub use crate::types::{
    TypeChecker, TypeCheckResult, TemporalType, ComputedEffect, EffectSet,
};

/// Combined analysis result.
#[derive(Debug, Clone, Default)]
pub struct AnalysisResult {
    /// Type checking result.
    pub type_result: Option<TypeCheckResult>,
    /// Temporal dependency graph.
    pub temporal_graph: Option<TemporalDependencyGraph>,
    /// Negative causal loops detected.
    pub negative_loops: Vec<NegativeLoop>,
    /// Linearity violations.
    pub linearity_violations: Vec<LinearityViolation>,
    /// Effect analysis.
    pub effects: Option<EffectSet>,
}

impl AnalysisResult {
    /// Check if analysis found no errors.
    pub fn is_ok(&self) -> bool {
        self.negative_loops.is_empty() &&
        self.linearity_violations.is_empty() &&
        self.type_result.as_ref().map_or(true, |r| r.errors.is_empty())
    }

    /// Get all errors as strings.
    pub fn errors(&self) -> Vec<String> {
        let mut errors = Vec::new();

        if let Some(ref type_result) = self.type_result {
            for err in &type_result.errors {
                errors.push(format!("{}", err));
            }
        }

        for violation in &self.linearity_violations {
            errors.push(format!("{}", violation));
        }

        for neg_loop in &self.negative_loops {
            errors.push(neg_loop.explanation.clone());
        }

        errors
    }

    /// Get all warnings.
    pub fn warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if let Some(ref type_result) = self.type_result {
            warnings.extend(type_result.warnings.iter().cloned());
        }

        warnings
    }
}

/// Linearity violation from affine type analysis.
#[derive(Debug, Clone)]
pub struct LinearityViolation {
    /// Kind of violation.
    pub kind: LinearityKind,
    /// Description.
    pub message: String,
    /// Source location (if available).
    pub location: Option<crate::compiler::lexer::Span>,
}

impl std::fmt::Display for LinearityViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ref loc) = self.location {
            write!(f, "linearity error at {}: {}", loc, self.message)
        } else {
            write!(f, "linearity error: {}", self.message)
        }
    }
}

/// Kinds of linearity violations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearityKind {
    /// DUP on a linear value.
    DuplicateLinear,
    /// PICK on a linear value.
    PickLinear,
    /// OVER on a linear value.
    OverLinear,
    /// Linear value unused (dropped without consumption).
    UnusedLinear,
    /// Linear value used multiple times.
    MultipleUse,
}

/// Unified compiler analysis.
pub struct CompilerAnalysis {
    /// Enable type checking.
    pub type_check: bool,
    /// Enable temporal analysis.
    pub temporal_analysis: bool,
    /// Strict linearity mode (errors instead of warnings).
    pub strict_linearity: bool,
}

impl Default for CompilerAnalysis {
    fn default() -> Self {
        Self {
            type_check: true,
            temporal_analysis: true,
            strict_linearity: true,
        }
    }
}

impl CompilerAnalysis {
    /// Create a new analysis with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze a program.
    pub fn analyze(&self, program: &crate::ast::Program) -> AnalysisResult {
        let mut result = AnalysisResult::default();

        // Type checking
        if self.type_check {
            let mut checker = TypeChecker::new();
            let type_result = checker.check(program);

            // Extract linearity violations if strict mode
            if self.strict_linearity {
                for warning in &type_result.warnings {
                    if warning.contains("linear") || warning.contains("ORACLE") {
                        result.linearity_violations.push(LinearityViolation {
                            kind: LinearityKind::DuplicateLinear,
                            message: warning.clone(),
                            location: None,
                        });
                    }
                }
            }

            result.type_result = Some(type_result);
        }

        // Temporal analysis
        if self.temporal_analysis {
            let tdg = TemporalDependencyGraph::build(program);
            result.negative_loops = tdg.find_negative_loops();
            result.temporal_graph = Some(tdg);
        }

        result
    }
}

/// Summary statistics from analysis.
#[derive(Debug, Clone, Default)]
pub struct AnalysisStats {
    /// Number of statements analyzed.
    pub statements: usize,
    /// Number of ORACLE operations.
    pub oracle_count: usize,
    /// Number of PROPHECY operations.
    pub prophecy_count: usize,
    /// Size of temporal core.
    pub temporal_core_size: usize,
    /// Number of warnings.
    pub warning_count: usize,
    /// Number of errors.
    pub error_count: usize,
}

impl AnalysisStats {
    /// Build stats from analysis result.
    pub fn from_result(result: &AnalysisResult, program: &crate::ast::Program) -> Self {
        let mut stats = Self::default();

        stats.statements = count_statements(program);

        if let Some(ref type_result) = result.type_result {
            stats.warning_count = type_result.warnings.len();
            stats.error_count = type_result.errors.len();
        }

        // Count temporal operations from the program
        stats.oracle_count = count_oracle_ops(program);
        stats.prophecy_count = count_prophecy_ops(program);

        if let Some(ref tdg) = result.temporal_graph {
            stats.temporal_core_size = tdg.temporal_core().len();
        }

        stats.error_count += result.linearity_violations.len();
        stats.error_count += result.negative_loops.len();

        stats
    }
}

/// Count ORACLE operations in a program.
fn count_oracle_ops(program: &crate::ast::Program) -> usize {
    use crate::ast::{OpCode, Stmt};
    fn count_in_stmts(stmts: &[Stmt]) -> usize {
        stmts.iter().map(|s| match s {
            Stmt::Op(OpCode::Oracle) => 1,
            Stmt::If { then_branch, else_branch } => {
                count_in_stmts(then_branch) + else_branch.as_ref().map_or(0, |e| count_in_stmts(e))
            }
            Stmt::While { cond, body } => count_in_stmts(cond) + count_in_stmts(body),
            Stmt::Block(inner) => count_in_stmts(inner),
            Stmt::Match { cases, default } => {
                cases.iter().map(|(_, body)| count_in_stmts(body)).sum::<usize>()
                    + default.as_ref().map_or(0, |d| count_in_stmts(d))
            }
            Stmt::TemporalScope { body, .. } => count_in_stmts(body),
            _ => 0,
        }).sum()
    }
    count_in_stmts(&program.body)
}

/// Count PROPHECY operations in a program.
fn count_prophecy_ops(program: &crate::ast::Program) -> usize {
    use crate::ast::{OpCode, Stmt};
    fn count_in_stmts(stmts: &[Stmt]) -> usize {
        stmts.iter().map(|s| match s {
            Stmt::Op(OpCode::Prophecy) => 1,
            Stmt::If { then_branch, else_branch } => {
                count_in_stmts(then_branch) + else_branch.as_ref().map_or(0, |e| count_in_stmts(e))
            }
            Stmt::While { cond, body } => count_in_stmts(cond) + count_in_stmts(body),
            Stmt::Block(inner) => count_in_stmts(inner),
            Stmt::Match { cases, default } => {
                cases.iter().map(|(_, body)| count_in_stmts(body)).sum::<usize>()
                    + default.as_ref().map_or(0, |d| count_in_stmts(d))
            }
            Stmt::TemporalScope { body, .. } => count_in_stmts(body),
            _ => 0,
        }).sum()
    }
    count_in_stmts(&program.body)
}

/// Count statements in a program.
fn count_statements(program: &crate::ast::Program) -> usize {
    fn count_stmts(stmts: &[crate::ast::Stmt]) -> usize {
        stmts.iter().map(|s| match s {
            crate::ast::Stmt::If { then_branch, else_branch } => {
                let else_count = else_branch.as_ref().map_or(0, |e| count_stmts(e));
                1 + count_stmts(then_branch) + else_count
            }
            crate::ast::Stmt::While { cond, body } => {
                1 + count_stmts(cond) + count_stmts(body)
            }
            crate::ast::Stmt::Block(inner) => count_stmts(inner),
            _ => 1,
        }).sum()
    }
    count_stmts(&program.body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analysis_result() {
        let result = AnalysisResult::default();
        assert!(result.is_ok());
        assert!(result.errors().is_empty());
    }

    #[test]
    fn test_linearity_violation_display() {
        let violation = LinearityViolation {
            kind: LinearityKind::DuplicateLinear,
            message: "cannot duplicate ORACLE result".to_string(),
            location: Some(crate::compiler::lexer::Span::new(5, 10, 45, 3)),
        };
        let display = format!("{}", violation);
        assert!(display.contains("5:10"));
        assert!(display.contains("duplicate"));
    }
}
