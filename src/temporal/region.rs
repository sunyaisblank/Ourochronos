//! Static contracts for finite temporal regions embedded in general-purpose
//! Ourochronos programs.
//!
//! Ordinary host code may use the language's I/O, collections, FFI, and
//! system facilities. A `TEMPORAL ... BITS ...` block is held to a stronger
//! contract: finite storage, deterministic total IR operations, no nesting,
//! and no recursive calls. This report makes that boundary inspectable before
//! a solver is invoked.

use crate::ast::{Procedure, Program, Stmt};
use crate::temporal::ir::TemporalIrCompiler;
use std::collections::{BTreeSet, HashMap};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemporalRegionInfo {
    pub index: usize,
    pub base: u64,
    pub cells: u64,
    pub cell_bits: u8,
    pub state_bits: u128,
    pub contains_loop: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemporalRegionIssue {
    pub region: usize,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemporalRegionReport {
    pub regions: Vec<TemporalRegionInfo>,
    pub issues: Vec<TemporalRegionIssue>,
    /// Effects in ordinary code are legal; listing them makes the host/solver
    /// staging boundary visible to tooling.
    pub host_effects: Vec<String>,
}

impl TemporalRegionReport {
    pub fn analyze(program: &Program, memory_cells: usize) -> Self {
        let procedures: HashMap<&str, &Procedure> = program
            .procedures
            .iter()
            .map(|procedure| (procedure.name.as_str(), procedure))
            .collect();
        let mut report = Self {
            regions: Vec::new(),
            issues: Vec::new(),
            host_effects: Vec::new(),
        };
        let mut host_effects = BTreeSet::new();
        scan_host(
            &program.body,
            &procedures,
            memory_cells,
            &mut report,
            &mut host_effects,
        );
        // Procedures and quotations are independently executable stores. Scan
        // each definition once as a host root so invalid regions cannot be
        // hidden in unused code or behind an indirect quotation reference.
        for procedure in &program.procedures {
            scan_host(
                &procedure.body,
                &procedures,
                memory_cells,
                &mut report,
                &mut host_effects,
            );
        }
        for quote in &program.quotes {
            scan_host(
                quote,
                &procedures,
                memory_cells,
                &mut report,
                &mut host_effects,
            );
        }
        report.host_effects = host_effects.into_iter().collect();
        report
    }

    pub fn is_valid(&self) -> bool {
        self.issues.is_empty()
    }
}

impl fmt::Display for TemporalRegionReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.regions.is_empty() {
            writeln!(f, "Finite temporal regions: none")?;
        } else {
            writeln!(f, "Finite temporal regions: {}", self.regions.len())?;
            for region in &self.regions {
                writeln!(
                    f,
                    "  #{} [{}..{}): {} cell(s) x {} bits = {} state bits{}",
                    region.index,
                    region.base,
                    region.base.saturating_add(region.cells),
                    region.cells,
                    region.cell_bits,
                    region.state_bits,
                    if region.contains_loop {
                        "; loop lowering is bounded"
                    } else {
                        ""
                    }
                )?;
            }
        }
        if !self.host_effects.is_empty() {
            writeln!(
                f,
                "Ordinary host effects outside regions: {}",
                self.host_effects.join(", ")
            )?;
        }
        for issue in &self.issues {
            writeln!(f, "  region #{} error: {}", issue.region, issue.message)?;
        }
        Ok(())
    }
}

fn scan_host<'a>(
    statements: &'a [Stmt],
    procedures: &HashMap<&'a str, &'a Procedure>,
    memory_cells: usize,
    report: &mut TemporalRegionReport,
    host_effects: &mut BTreeSet<String>,
) {
    for statement in statements {
        match statement {
            Stmt::TemporalScope {
                base,
                size,
                cell_bits,
                body,
            } => {
                let index = report.regions.len();
                let end = base.checked_add(*size);
                let state_bits = (*size as u128) * (*cell_bits as u128);
                let mut info = TemporalRegionInfo {
                    index,
                    base: *base,
                    cells: *size,
                    cell_bits: *cell_bits,
                    state_bits,
                    contains_loop: false,
                };
                if *size == 0 {
                    issue(report, index, "region has zero cells");
                }
                if !(1..=64).contains(cell_bits) {
                    issue(report, index, "cell bit width must be in 1..=64");
                }
                match end {
                    Some(end) if end <= memory_cells as u64 => {}
                    Some(end) => issue(
                        report,
                        index,
                        format!(
                            "range [{}..{}) exceeds configured memory width {}",
                            base, end, memory_cells
                        ),
                    ),
                    None => issue(report, index, "address range overflows u64"),
                }
                let mut active_calls = Vec::new();
                scan_region(
                    body,
                    procedures,
                    report,
                    index,
                    &mut info.contains_loop,
                    &mut active_calls,
                );
                report.regions.push(info);
            }
            Stmt::Op(op) => {
                if !matches!(op.effect_class(), crate::ast::EffectClass::Pure) {
                    host_effects.insert(op.name().to_string());
                }
            }
            _ => {
                for child in statement.child_blocks() {
                    scan_host(child, procedures, memory_cells, report, host_effects);
                }
            }
        }
    }
}

fn scan_region<'a>(
    statements: &'a [Stmt],
    procedures: &HashMap<&'a str, &'a Procedure>,
    report: &mut TemporalRegionReport,
    region: usize,
    contains_loop: &mut bool,
    active_calls: &mut Vec<String>,
) {
    for statement in statements {
        match statement {
            Stmt::Push(_) | Stmt::PushConstant { .. } | Stmt::ReadTemporal { .. } => {}
            Stmt::PushQuote(id) => issue(
                report,
                region,
                format!(
                    "quotation reference {} has no typed lowering in the finite temporal IR",
                    id.as_u64()
                ),
            ),
            Stmt::Op(op) if TemporalIrCompiler::supports_opcode(*op) => {}
            Stmt::Op(op) => issue(
                report,
                region,
                format!(
                    "{} has no total deterministic lowering in the finite temporal IR",
                    op.name()
                ),
            ),
            Stmt::If {
                then_branch,
                else_branch,
            } => {
                scan_region(
                    then_branch,
                    procedures,
                    report,
                    region,
                    contains_loop,
                    active_calls,
                );
                if let Some(branch) = else_branch {
                    scan_region(
                        branch,
                        procedures,
                        report,
                        region,
                        contains_loop,
                        active_calls,
                    );
                }
            }
            Stmt::While { cond, body } => {
                *contains_loop = true;
                scan_region(
                    cond,
                    procedures,
                    report,
                    region,
                    contains_loop,
                    active_calls,
                );
                scan_region(
                    body,
                    procedures,
                    report,
                    region,
                    contains_loop,
                    active_calls,
                );
            }
            Stmt::Block(body) => scan_region(
                body,
                procedures,
                report,
                region,
                contains_loop,
                active_calls,
            ),
            Stmt::TemporalScope { .. } => {
                issue(report, region, "nested TEMPORAL scopes are not supported")
            }
            Stmt::Call { name } => {
                if active_calls.iter().any(|active| active == name) {
                    issue(
                        report,
                        region,
                        format!("recursive call to '{}' is not a finite total region", name),
                    );
                    continue;
                }
                let Some(procedure) = procedures.get(name.as_str()) else {
                    issue(report, region, format!("unknown procedure '{}'", name));
                    continue;
                };
                active_calls.push(name.clone());
                scan_region(
                    &procedure.body,
                    procedures,
                    report,
                    region,
                    contains_loop,
                    active_calls,
                );
                active_calls.pop();
            }
        }
    }
}

fn issue(report: &mut TemporalRegionReport, region: usize, message: impl Into<String>) {
    report.issues.push(TemporalRegionIssue {
        region,
        message: message.into(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    #[test]
    fn deterministic_region_and_effectful_host_are_separated() {
        let program = parse("TEMPORAL 0 1 BITS 2 { 0 ORACLE 0 PROPHECY } CLOCK POP").unwrap();
        let report = TemporalRegionReport::analyze(&program, 4);
        assert!(report.is_valid(), "{:?}", report.issues);
        assert_eq!(report.regions[0].state_bits, 2);
        assert_eq!(report.host_effects, vec!["CLOCK"]);
    }

    #[test]
    fn effect_inside_region_is_rejected() {
        let program = parse("TEMPORAL 0 1 BITS 2 { CLOCK POP }").unwrap();
        let report = TemporalRegionReport::analyze(&program, 4);
        assert!(!report.is_valid());
        assert!(report.issues[0].message.contains("CLOCK"));
    }

    #[test]
    fn loops_are_explicitly_reported_as_bounded_lowering() {
        let program = parse("TEMPORAL 0 1 BITS 2 { WHILE { 0 } { NOP } 0 0 PROPHECY }").unwrap();
        let report = TemporalRegionReport::analyze(&program, 4);
        assert!(report.is_valid());
        assert!(report.regions[0].contains_loop);
    }

    #[test]
    fn regions_in_quotations_are_always_validated() {
        let program = parse("[ TEMPORAL 0 1 BITS 1 { CLOCK POP } ] EXEC").unwrap();
        let report = TemporalRegionReport::analyze(&program, 4);
        assert_eq!(report.regions.len(), 1);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.message.contains("CLOCK")));
    }

    #[test]
    fn regions_in_unused_procedures_are_always_validated() {
        let program = parse("PROCEDURE bad { TEMPORAL 0 1 BITS 1 { CLOCK POP } } 0 POP").unwrap();
        let report = TemporalRegionReport::analyze(&program, 4);
        assert_eq!(report.regions.len(), 1);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.message.contains("CLOCK")));
    }
}
