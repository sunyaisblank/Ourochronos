//! Canonical source admission for every executable and proof-facing API.
//!
//! Source syntax is intentionally not an executable authority.  A program is
//! admitted only after the temporal type/effect system, finite-region rules,
//! typed HIR resolution, structural semantics, bytecode lowering, runtime
//! capability check, structural validation, and independent CFG verification
//! all agree.  The resulting [`AdmittedProgram`] owns a sealed
//! [`PreparedBytecode`] value, so downstream code cannot accidentally execute
//! the source AST or a mutable, unverified artifact.

use crate::ast::Program;
use crate::bytecode::{BytecodeProgram, Instruction};
use crate::bytecode_vm::{bytecode_vm_supports, PreparedBytecode};
use crate::core::MEMORY_SIZE;
use crate::hir::HirProgram;
use crate::semantics::{check as check_semantics, SemanticsReport};
use crate::temporal::region::TemporalRegionReport;
use crate::types::{type_check, TypeCheckResult};
use std::error::Error;
use std::fmt;

/// Compiler stage that rejected a source program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionPhase {
    Configuration,
    Types,
    Regions,
    Resolution,
    Semantics,
    Bytecode,
    RuntimeCapabilities,
    Verification,
}

impl fmt::Display for AdmissionPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Configuration => "configuration",
            Self::Types => "type/effect analysis",
            Self::Regions => "temporal-region analysis",
            Self::Resolution => "typed name resolution",
            Self::Semantics => "structural semantics",
            Self::Bytecode => "bytecode lowering",
            Self::RuntimeCapabilities => "runtime capability analysis",
            Self::Verification => "independent bytecode verification",
        };
        f.write_str(name)
    }
}

/// A fail-closed source-admission error with stable phase attribution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionError {
    pub phase: AdmissionPhase,
    pub diagnostics: Vec<String>,
}

impl AdmissionError {
    fn one(phase: AdmissionPhase, diagnostic: impl Into<String>) -> Self {
        Self {
            phase,
            diagnostics: vec![diagnostic.into()],
        }
    }

    fn many(phase: AdmissionPhase, diagnostics: Vec<String>) -> Self {
        debug_assert!(!diagnostics.is_empty());
        Self { phase, diagnostics }
    }
}

impl fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} failed", self.phase)?;
        if !self.diagnostics.is_empty() {
            write!(f, ": {}", self.diagnostics.join("; "))?;
        }
        Ok(())
    }
}

impl Error for AdmissionError {}

/// Source-level parameters that affect mandatory admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionConfig {
    pub memory_cells: usize,
}

impl Default for AdmissionConfig {
    fn default() -> Self {
        Self {
            memory_cells: MEMORY_SIZE,
        }
    }
}

/// Reports and typed HIR produced before executable lowering.
#[derive(Debug, Clone)]
pub struct SourceAdmission {
    types: TypeCheckResult,
    regions: TemporalRegionReport,
    hir: HirProgram,
    semantics: SemanticsReport,
}

impl SourceAdmission {
    pub fn types(&self) -> &TypeCheckResult {
        &self.types
    }

    pub fn regions(&self) -> &TemporalRegionReport {
        &self.regions
    }

    pub fn hir(&self) -> &HirProgram {
        &self.hir
    }

    pub fn semantics(&self) -> &SemanticsReport {
        &self.semantics
    }
}

/// A source program sealed behind all mandatory compiler and verifier gates.
#[derive(Debug, Clone)]
pub struct AdmittedProgram {
    source: SourceAdmission,
    executable: PreparedBytecode,
}

impl AdmittedProgram {
    pub fn source(&self) -> &SourceAdmission {
        &self.source
    }

    pub fn executable(&self) -> &PreparedBytecode {
        &self.executable
    }

    pub fn program(&self) -> &BytecodeProgram {
        self.executable.program()
    }

    pub fn into_executable(self) -> PreparedBytecode {
        self.executable
    }

    pub fn into_program(self) -> BytecodeProgram {
        self.executable.into_program()
    }
}

/// Apply every mandatory source-level analysis and retain the typed result.
pub fn analyze_program(
    program: &Program,
    config: AdmissionConfig,
) -> Result<SourceAdmission, AdmissionError> {
    if config.memory_cells == 0 {
        return Err(AdmissionError::one(
            AdmissionPhase::Configuration,
            "memory width must be greater than zero",
        ));
    }

    let types = type_check(program);
    if !types.is_valid {
        let mut diagnostics: Vec<String> = types.errors.iter().map(ToString::to_string).collect();
        diagnostics.extend(types.effect_violations.iter().map(|violation| {
            format!(
                "procedure '{}' declares {:?} but has effects {}",
                violation.procedure_name,
                violation.declared,
                violation.actual.summary()
            )
        }));
        diagnostics.extend(types.linear_violations.iter().map(|violation| {
            format!(
                "statement {} {}: {}",
                violation.stmt_index, violation.operation, violation.message
            )
        }));
        return Err(AdmissionError::many(AdmissionPhase::Types, diagnostics));
    }

    let regions = TemporalRegionReport::analyze(program, config.memory_cells);
    if !regions.is_valid() {
        return Err(AdmissionError::many(
            AdmissionPhase::Regions,
            regions
                .issues
                .iter()
                .map(|issue| format!("region #{}: {}", issue.region, issue.message))
                .collect(),
        ));
    }

    let hir = HirProgram::resolve(program).map_err(|errors| {
        AdmissionError::many(
            AdmissionPhase::Resolution,
            errors.into_iter().map(|error| error.to_string()).collect(),
        )
    })?;
    analyze_resolved(program, types, regions, hir)
}

/// Compile source through the canonical admission boundary and seal the exact
/// executable representation after independent verification.
pub fn admit_program(
    program: &Program,
    config: AdmissionConfig,
) -> Result<AdmittedProgram, AdmissionError> {
    let source = analyze_program(program, config)?;
    admit_analyzed(source)
}

pub(crate) fn analyze_resolved_program(
    program: &Program,
    hir: HirProgram,
    config: AdmissionConfig,
) -> Result<SourceAdmission, AdmissionError> {
    if config.memory_cells == 0 {
        return Err(AdmissionError::one(
            AdmissionPhase::Configuration,
            "memory width must be greater than zero",
        ));
    }
    let types = type_check(program);
    if !types.is_valid {
        return analyze_program(program, config);
    }
    let regions = TemporalRegionReport::analyze(program, config.memory_cells);
    if !regions.is_valid() {
        return analyze_program(program, config);
    }
    analyze_resolved(program, types, regions, hir)
}

fn analyze_resolved(
    _program: &Program,
    types: TypeCheckResult,
    regions: TemporalRegionReport,
    hir: HirProgram,
) -> Result<SourceAdmission, AdmissionError> {
    let semantics = check_semantics(&hir);
    if !semantics.is_accepted_for_interpreter() {
        return Err(AdmissionError::many(
            AdmissionPhase::Semantics,
            semantics
                .errors
                .iter()
                .map(|error| {
                    format!(
                        "{:?} at {:?} {:?}",
                        error.kind, error.site.owner, error.site.context
                    )
                })
                .collect(),
        ));
    }
    Ok(SourceAdmission {
        types,
        regions,
        hir,
        semantics,
    })
}

pub(crate) fn admit_analyzed(source: SourceAdmission) -> Result<AdmittedProgram, AdmissionError> {
    let bytecode = BytecodeProgram::compile(source.hir())
        .map_err(|error| AdmissionError::one(AdmissionPhase::Bytecode, error.to_string()))?;
    seal_analyzed(source, bytecode)
}

pub(crate) fn seal_analyzed(
    source: SourceAdmission,
    bytecode: BytecodeProgram,
) -> Result<AdmittedProgram, AdmissionError> {
    if let Some(opcode) = bytecode.instructions.iter().find_map(|instruction| {
        let Instruction::Primitive(opcode) = instruction else {
            return None;
        };
        (!bytecode_vm_supports(*opcode)).then_some(*opcode)
    }) {
        return Err(AdmissionError::one(
            AdmissionPhase::RuntimeCapabilities,
            format!(
                "{} has no authoritative bytecode runtime semantics",
                opcode.name()
            ),
        ));
    }
    let executable = PreparedBytecode::new(bytecode)
        .map_err(|error| AdmissionError::one(AdmissionPhase::Verification, error.to_string()))?;
    Ok(AdmittedProgram { source, executable })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Memory;
    use crate::family_verifier::{
        PspaceFamilyVerifier, PspaceInstanceConfig, PspaceInstanceVerificationResult,
    };
    use crate::halting::{BoundedHaltingAnalyzer, BoundedHaltingResult};
    use crate::parser::parse;
    use crate::temporal::global_solver::{
        GlobalFixedPointSolver, GlobalSolveConfig, GlobalSolveResult,
    };
    use crate::temporal::timeloop::{ConvergenceStatus, TimeLoop, TimeLoopConfig};
    use crate::temporal::transition_graph::{
        ProgramGraphConfig, ProgramTransitionAnalyzer, TransitionGraphError,
    };
    use crate::vm::{execute_with_fast_path, EpochStatus, Executor, ExecutorConfig};

    #[test]
    fn admission_seals_a_fully_checked_executable() {
        let program = parse("0 ORACLE DUP 0 PROPHECY").unwrap();
        let admitted = admit_program(&program, AdmissionConfig::default()).unwrap();
        assert!(admitted.source().types().is_valid);
        assert!(admitted.source().regions().is_valid());
        assert!(admitted.source().semantics().is_accepted_for_interpreter());
        assert!(!admitted.program().instructions.is_empty());
    }

    #[test]
    fn effect_contract_failure_never_reaches_bytecode() {
        let program = parse("PROCEDURE hidden PURE { 1 OUTPUT } 0").unwrap();
        let error = admit_program(&program, AdmissionConfig::default()).unwrap_err();
        assert_eq!(error.phase, AdmissionPhase::Types);
        assert!(error.to_string().contains("hidden"));
    }

    #[test]
    fn invalid_unused_region_is_mandatory() {
        let program = parse("PROCEDURE hidden { TEMPORAL 4 2 BITS 1 { 0 ORACLE } } 0").unwrap();
        let error = admit_program(&program, AdmissionConfig { memory_cells: 4 }).unwrap_err();
        assert_eq!(error.phase, AdmissionPhase::Regions);
    }

    #[test]
    fn obsolete_dynamic_ffi_is_rejected_during_admission() {
        let program = parse("0 FFI_CALL").unwrap();
        let error = admit_program(&program, AdmissionConfig::default()).unwrap_err();
        assert_eq!(error.phase, AdmissionPhase::RuntimeCapabilities);
    }

    #[test]
    fn every_source_execution_and_proof_facade_enforces_the_same_admission_gate() {
        let program = parse("PROCEDURE hidden PURE { 1 OUTPUT } 0").unwrap();
        let marker = "type/effect analysis failed";

        let mut executor = Executor::new();
        let epoch = executor.run_epoch(&program, &Memory::new());
        assert!(matches!(epoch.status, EpochStatus::Error(ref error) if error.contains(marker)));

        let fast = execute_with_fast_path(&program, &ExecutorConfig::default());
        assert!(matches!(fast.status, EpochStatus::Error(ref error) if error.contains(marker)));

        let mut timeloop = TimeLoop::new(TimeLoopConfig::default()).unwrap();
        assert!(matches!(
            timeloop.run(&program),
            ConvergenceStatus::Error { message, .. } if message.contains(marker)
        ));

        assert!(matches!(
            BoundedHaltingAnalyzer::analyze(&program, 10, 1),
            BoundedHaltingResult::Unsupported { reason } if reason.contains(marker)
        ));

        assert!(matches!(
            GlobalFixedPointSolver::solve(
                &program,
                GlobalSolveConfig {
                    memory_cells: 1,
                    ..GlobalSolveConfig::default()
                }
            ),
            GlobalSolveResult::Unsupported { reason } if reason.contains(marker)
        ));

        assert!(matches!(
            ProgramTransitionAnalyzer::analyze(&program, ProgramGraphConfig::default()),
            Err(TransitionGraphError::Unsupported(reason)) if reason.contains(marker)
        ));

        let family_program = parse(
            "FAMILY gate { CTC_CELLS POLY 1 1 0; CHRONOLOGY_BITS POLY 1000 0 0; TRANSITION_STEPS POLY 20 0 0; UNIFORM; TOTAL; READOUT_INVARIANT; IDEAL_DEUTSCH; EFFECTS_FROZEN; }\n\
             PROCEDURE hidden PURE { 1 OUTPUT } 0",
        )
        .unwrap();
        assert!(matches!(
            PspaceFamilyVerifier::verify(&family_program, PspaceInstanceConfig::default()),
            PspaceInstanceVerificationResult::Unsupported { reason }
                if reason.contains(marker)
        ));
    }
}
