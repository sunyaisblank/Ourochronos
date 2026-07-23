//! Integrated global point-fixed-state solving.
//!
//! SAT is never trusted on its own: every decoded model is replayed through
//! the reference VM and accepted only when the epoch finishes normally with
//! `present == anamnesis`.  UNSAT is called a global proof only for a complete
//! IR; an unrolled-loop UNSAT result is deliberately `Unknown`.

#[cfg(test)]
use crate::ast::Stmt;
use crate::ast::{
    OpCode, Program, PropertyComparison, PropertyPredicate, TemporalPropertyDeclaration,
};
use crate::bytecode::{BytecodeProgram, Instruction};
use crate::bytecode_temporal::{lower_bytecode_temporal_ir, BytecodeTemporalLoweringError};
use crate::bytecode_vm::{BytecodeVm, BytecodeVmConfig, BytecodeVmError, BytecodeVmStatus};
use crate::core::{BoundsPolicy, Memory, OutputItem, PagedMemory, Value};
use crate::hir::HirProgram;
use crate::temporal::ir::{
    IrCompleteness, TemporalIr, TemporalIrCompiler, TemporalIrConfig, TemporalIrError,
};
#[cfg(test)]
use std::collections::HashMap;
use std::collections::HashSet;
use z3::ast::{Array, Ast, Dynamic};
use z3::{Config as Z3Config, Context, DeclKind, Model, SatResult, Solver, Sort};

#[derive(Debug, Clone, Copy)]
pub struct GlobalSolveConfig {
    pub memory_cells: usize,
    pub loop_unroll_limit: usize,
    pub solver_timeout_ms: u64,
    pub max_instructions: u64,
    pub bounds_policy: BoundsPolicy,
}

impl Default for GlobalSolveConfig {
    fn default() -> Self {
        Self {
            memory_cells: crate::core::MEMORY_SIZE,
            loop_unroll_limit: 10,
            solver_timeout_ms: 30_000,
            max_instructions: 10_000_000,
            bounds_policy: BoundsPolicy::Wrap,
        }
    }
}

/// A checkable positive certificate.  `memory` is the solver model and
/// `replayed_present` is the independent reference-VM result.
#[derive(Debug, Clone)]
pub struct FixedPointWitness {
    pub memory: Memory,
    pub replayed_present: Memory,
    pub output: Vec<OutputItem>,
    pub instructions_executed: u64,
    pub constraint_digest: u64,
    pub completeness: IrCompleteness,
}

impl FixedPointWitness {
    pub fn is_replay_verified(&self) -> bool {
        self.memory.numeric_values_equal(&self.replayed_present)
    }

    /// Stable, dependency-free interchange form for certificates and
    /// counterexamples.
    pub fn to_json(&self) -> String {
        let cells = self
            .memory
            .iter_nonzero()
            .map(|(address, value)| format!("[{},{}]", address, value.val))
            .collect::<Vec<_>>()
            .join(",");
        let output = self
            .output
            .iter()
            .map(|item| match item {
                OutputItem::Val(value) => format!("{{\"kind\":\"value\",\"value\":{}}}", value.val),
                OutputItem::Char(value) => {
                    format!("{{\"kind\":\"character\",\"value\":{}}}", value)
                }
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"memory_cells\":{},\"nonzero_memory\":[{}],\"output\":[{}],\"instructions\":{},\"replay_verified\":{},\"completeness\":{},\"constraint_digest\":\"{:016x}\"}}",
            self.memory.len(),
            cells,
            output,
            self.instructions_executed,
            self.is_replay_verified(),
            completeness_json(self.completeness),
            self.constraint_digest
        )
    }
}

#[derive(Debug, Clone)]
pub struct UnsatCertificate {
    /// Deterministic FNV-1a digest of the exact SMT artifact sent to Z3.
    pub constraint_digest: u64,
    pub backend: &'static str,
    pub completeness: IrCompleteness,
}

/// Exhaustive outcome algebra for integrated solving.
#[derive(Debug, Clone)]
pub enum GlobalSolveResult {
    Found(FixedPointWitness),
    ProvenNoFixedPoint(UnsatCertificate),
    Unknown {
        reason: String,
        completeness: Option<IrCompleteness>,
        constraint_digest: Option<u64>,
    },
    Unsupported {
        reason: String,
    },
    InternalError {
        reason: String,
    },
}

/// Result of quantifying over all point fixed states in the finite temporal
/// region.  `Multiple` carries two independently replayed counterexamples.
#[derive(Debug, Clone)]
pub enum GlobalUniquenessResult {
    NoFixedPoint(UnsatCertificate),
    Unique {
        witness: FixedPointWitness,
        certificate: UnsatCertificate,
    },
    Multiple {
        first: FixedPointWitness,
        second: FixedPointWitness,
        differing_cells: Vec<u64>,
    },
    Unknown {
        reason: String,
        witness: Option<FixedPointWitness>,
        completeness: Option<IrCompleteness>,
        constraint_digest: Option<u64>,
    },
    Unsupported {
        reason: String,
    },
    InternalError {
        reason: String,
    },
}

/// Universal source-property outcome over all point fixed states.
#[derive(Debug, Clone)]
pub enum PropertyVerificationResult {
    Proven {
        property: TemporalPropertyDeclaration,
        exemplar: FixedPointWitness,
        certificate: UnsatCertificate,
    },
    Refuted {
        property: TemporalPropertyDeclaration,
        counterexample: FixedPointWitness,
    },
    Vacuous {
        property: TemporalPropertyDeclaration,
        no_fixed_point: UnsatCertificate,
    },
    Unknown {
        property: TemporalPropertyDeclaration,
        reason: String,
        exemplar: Option<FixedPointWitness>,
        constraint_digest: Option<u64>,
    },
    Unsupported {
        property: TemporalPropertyDeclaration,
        reason: String,
    },
    InternalError {
        property: TemporalPropertyDeclaration,
        reason: String,
    },
}

impl GlobalSolveResult {
    pub fn is_decided(&self) -> bool {
        matches!(self, Self::Found(_) | Self::ProvenNoFixedPoint(_))
    }

    pub fn to_json(&self) -> String {
        match self {
            Self::Found(witness) => artifact_json(
                "fixed-point",
                "found",
                &format!("\"witness\":{}", witness.to_json()),
            ),
            Self::ProvenNoFixedPoint(certificate) => artifact_json(
                "fixed-point",
                "proven-no-fixed-point",
                &format!("\"certificate\":{}", certificate_json(certificate)),
            ),
            Self::Unknown {
                reason,
                constraint_digest,
                ..
            } => artifact_json(
                "fixed-point",
                "unknown",
                &format!(
                    "\"reason\":\"{}\",\"constraint_digest\":{}",
                    json_escape(reason),
                    optional_digest(*constraint_digest)
                ),
            ),
            Self::Unsupported { reason } => artifact_json(
                "fixed-point",
                "unsupported",
                &format!("\"reason\":\"{}\"", json_escape(reason)),
            ),
            Self::InternalError { reason } => artifact_json(
                "fixed-point",
                "internal-error",
                &format!("\"reason\":\"{}\"", json_escape(reason)),
            ),
        }
    }
}

impl GlobalUniquenessResult {
    pub fn to_json(&self) -> String {
        match self {
            Self::NoFixedPoint(certificate) => artifact_json(
                "all-fixed",
                "no-fixed-point",
                &format!("\"certificate\":{}", certificate_json(certificate)),
            ),
            Self::Unique {
                witness,
                certificate,
            } => artifact_json(
                "all-fixed",
                "unique",
                &format!(
                    "\"witness\":{},\"certificate\":{}",
                    witness.to_json(),
                    certificate_json(certificate)
                ),
            ),
            Self::Multiple {
                first,
                second,
                differing_cells,
            } => artifact_json(
                "all-fixed",
                "multiple",
                &format!(
                    "\"first\":{},\"second\":{},\"differing_cells\":[{}]",
                    first.to_json(),
                    second.to_json(),
                    differing_cells
                        .iter()
                        .map(u64::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                ),
            ),
            Self::Unknown {
                reason,
                witness,
                constraint_digest,
                ..
            } => artifact_json(
                "all-fixed",
                "unknown",
                &format!(
                    "\"reason\":\"{}\",\"witness\":{},\"constraint_digest\":{}",
                    json_escape(reason),
                    witness
                        .as_ref()
                        .map(FixedPointWitness::to_json)
                        .unwrap_or_else(|| "null".to_string()),
                    optional_digest(*constraint_digest)
                ),
            ),
            Self::Unsupported { reason } => artifact_json(
                "all-fixed",
                "unsupported",
                &format!("\"reason\":\"{}\"", json_escape(reason)),
            ),
            Self::InternalError { reason } => artifact_json(
                "all-fixed",
                "internal-error",
                &format!("\"reason\":\"{}\"", json_escape(reason)),
            ),
        }
    }
}

impl PropertyVerificationResult {
    pub fn to_json(&self) -> String {
        let (property, status, detail) = match self {
            Self::Proven {
                property,
                exemplar,
                certificate,
            } => (
                property,
                "proven",
                format!(
                    "\"exemplar\":{},\"certificate\":{}",
                    exemplar.to_json(),
                    certificate_json(certificate)
                ),
            ),
            Self::Refuted {
                property,
                counterexample,
            } => (
                property,
                "refuted",
                format!("\"counterexample\":{}", counterexample.to_json()),
            ),
            Self::Vacuous {
                property,
                no_fixed_point,
            } => (
                property,
                "vacuous",
                format!("\"certificate\":{}", certificate_json(no_fixed_point)),
            ),
            Self::Unknown {
                property,
                reason,
                exemplar,
                constraint_digest,
            } => (
                property,
                "unknown",
                format!(
                    "\"reason\":\"{}\",\"exemplar\":{},\"constraint_digest\":{}",
                    json_escape(reason),
                    exemplar
                        .as_ref()
                        .map(FixedPointWitness::to_json)
                        .unwrap_or_else(|| "null".to_string()),
                    optional_digest(*constraint_digest)
                ),
            ),
            Self::Unsupported { property, reason } => (
                property,
                "unsupported",
                format!("\"reason\":\"{}\"", json_escape(reason)),
            ),
            Self::InternalError { property, reason } => (
                property,
                "internal-error",
                format!("\"reason\":\"{}\"", json_escape(reason)),
            ),
        };
        artifact_json(
            "property",
            status,
            &format!("\"property\":{},{}", property_json(property), detail),
        )
    }
}

fn property_json(property: &TemporalPropertyDeclaration) -> String {
    let predicate = property.predicate.as_ref().map_or_else(
        || {
            format!(
                "{{\"kind\":\"compare\",\"cell\":{{\"address\":{}}},\"comparison\":\"{:?}\",\"value\":{}}}",
                property.address, property.comparison, property.value
            )
        },
        property_predicate_json,
    );
    let touched = property
        .touched_addresses()
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"name\":\"{}\",\"predicate\":{},\"slice\":{{\"kind\":\"touched-addresses\",\"incremental_reuse\":false,\"addresses\":[{}],\"cells\":[{}]}}}}",
        json_escape(&property.name),
        predicate,
        touched,
        property_slice_cells_json(property)
    )
}

fn property_slice_cells_json(property: &TemporalPropertyDeclaration) -> String {
    let mut cells = Vec::<(u64, Option<String>)>::new();
    if let Some(predicate) = &property.predicate {
        let mut work = vec![predicate];
        while let Some(predicate) = work.pop() {
            match predicate {
                PropertyPredicate::Compare { cell, .. } => {
                    cells.push((cell.address, cell.name.clone()));
                }
                PropertyPredicate::Not(inner) => work.push(inner),
                PropertyPredicate::And(left, right) | PropertyPredicate::Or(left, right) => {
                    work.push(right);
                    work.push(left);
                }
            }
        }
    } else {
        cells.push((property.address, None));
    }
    cells.sort();
    cells.dedup();
    cells
        .into_iter()
        .map(|(address, name)| {
            let name = name.map_or_else(
                || "null".to_string(),
                |name| format!("\"{}\"", json_escape(&name)),
            );
            format!("{{\"address\":{address},\"name\":{name}}}")
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn property_predicate_json(predicate: &PropertyPredicate) -> String {
    match predicate {
        PropertyPredicate::Compare {
            cell,
            comparison,
            value,
        } => {
            let name = cell.name.as_ref().map_or_else(
                || "null".to_string(),
                |name| format!("\"{}\"", json_escape(name)),
            );
            format!(
                "{{\"kind\":\"compare\",\"cell\":{{\"address\":{},\"name\":{}}},\"comparison\":\"{:?}\",\"value\":{}}}",
                cell.address, name, comparison, value
            )
        }
        PropertyPredicate::Not(inner) => format!(
            "{{\"kind\":\"not\",\"operand\":{}}}",
            property_predicate_json(inner)
        ),
        PropertyPredicate::And(left, right) => format!(
            "{{\"kind\":\"and\",\"left\":{},\"right\":{}}}",
            property_predicate_json(left),
            property_predicate_json(right)
        ),
        PropertyPredicate::Or(left, right) => format!(
            "{{\"kind\":\"or\",\"left\":{},\"right\":{}}}",
            property_predicate_json(left),
            property_predicate_json(right)
        ),
    }
}

pub struct GlobalFixedPointSolver;

impl GlobalFixedPointSolver {
    pub fn compile(
        program: &Program,
        config: GlobalSolveConfig,
    ) -> Result<TemporalIr, TemporalIrError> {
        TemporalIrCompiler::compile(
            program,
            TemporalIrConfig {
                memory_cells: config.memory_cells,
                loop_unroll_limit: config.loop_unroll_limit,
                bounds_policy: config.bounds_policy,
            },
        )
    }

    /// Lower the validated linked executable directly into temporal IR.
    pub fn compile_bytecode(
        program: &BytecodeProgram,
        config: GlobalSolveConfig,
    ) -> Result<TemporalIr, BytecodeTemporalLoweringError> {
        lower_bytecode_temporal_ir(
            program,
            TemporalIrConfig {
                memory_cells: config.memory_cells,
                loop_unroll_limit: config.loop_unroll_limit,
                bounds_policy: config.bounds_policy,
            },
        )
    }

    pub fn solve(program: &Program, config: GlobalSolveConfig) -> GlobalSolveResult {
        let bytecode = match source_program_bytecode(program) {
            Ok(bytecode) => bytecode,
            Err(reason) => return GlobalSolveResult::Unsupported { reason },
        };
        Self::solve_bytecode(&bytecode, config)
    }

    /// Solve a temporal IR compiled from the linked bytecode and replay every
    /// SAT witness through that same bytecode machine.
    pub fn solve_bytecode(
        program: &BytecodeProgram,
        config: GlobalSolveConfig,
    ) -> GlobalSolveResult {
        if let Some(reason) = dense_memory_config_error(config) {
            return GlobalSolveResult::Unsupported { reason };
        }
        let ir = match Self::compile_bytecode(program, config) {
            Ok(ir) => ir,
            Err(error) => {
                return GlobalSolveResult::Unsupported {
                    reason: error.to_string(),
                }
            }
        };
        Self::solve_ir(program, config, &ir, None)
    }

    /// Decide whether the finite temporal region has zero, one, or multiple
    /// point fixed states.  This is an all-model query, not orbit exploration.
    pub fn analyze_uniqueness(
        program: &Program,
        config: GlobalSolveConfig,
    ) -> GlobalUniquenessResult {
        let bytecode = match source_program_bytecode(program) {
            Ok(bytecode) => bytecode,
            Err(reason) => return GlobalUniquenessResult::Unsupported { reason },
        };
        Self::analyze_uniqueness_bytecode(&bytecode, config)
    }

    /// Bytecode-authoritative zero/one/multiple point-fixed-state analysis.
    pub fn analyze_uniqueness_bytecode(
        program: &BytecodeProgram,
        config: GlobalSolveConfig,
    ) -> GlobalUniquenessResult {
        if let Some(reason) = dense_memory_config_error(config) {
            return GlobalUniquenessResult::Unsupported { reason };
        }
        let ir = match Self::compile_bytecode(program, config) {
            Ok(ir) => ir,
            Err(error) => {
                return GlobalUniquenessResult::Unsupported {
                    reason: error.to_string(),
                }
            }
        };
        Self::analyze_uniqueness_ir(program, config, &ir)
    }

    fn analyze_uniqueness_ir(
        program: &BytecodeProgram,
        config: GlobalSolveConfig,
        ir: &TemporalIr,
    ) -> GlobalUniquenessResult {
        let first = match Self::solve_ir(program, config, ir, None) {
            GlobalSolveResult::Found(witness) => witness,
            GlobalSolveResult::ProvenNoFixedPoint(certificate) => {
                return GlobalUniquenessResult::NoFixedPoint(certificate)
            }
            GlobalSolveResult::Unknown {
                reason,
                completeness,
                constraint_digest,
            } => {
                return GlobalUniquenessResult::Unknown {
                    reason,
                    witness: None,
                    completeness,
                    constraint_digest,
                }
            }
            GlobalSolveResult::Unsupported { reason } => {
                return GlobalUniquenessResult::Unsupported { reason }
            }
            GlobalSolveResult::InternalError { reason } => {
                return GlobalUniquenessResult::InternalError { reason }
            }
        };

        let blocking_assertion = block_memory_assertion(&first.memory);
        match Self::solve_ir(program, config, ir, Some(&blocking_assertion)) {
            GlobalSolveResult::Found(second) => {
                let differing_cells = first.memory.diff(&second.memory);
                GlobalUniquenessResult::Multiple {
                    first,
                    second,
                    differing_cells,
                }
            }
            GlobalSolveResult::ProvenNoFixedPoint(certificate) => GlobalUniquenessResult::Unique {
                witness: first,
                certificate,
            },
            GlobalSolveResult::Unknown {
                reason,
                completeness,
                constraint_digest,
            } => GlobalUniquenessResult::Unknown {
                reason,
                witness: Some(first),
                completeness,
                constraint_digest,
            },
            GlobalSolveResult::Unsupported { reason } => {
                GlobalUniquenessResult::Unsupported { reason }
            }
            GlobalSolveResult::InternalError { reason } => {
                GlobalUniquenessResult::InternalError { reason }
            }
        }
    }

    /// Prove or refute one `ALL_FIXED CELL ...` property. A refutation is a
    /// replayed fixed-state counterexample; a proof is complete-IR UNSAT for
    /// the negated property. No-fixed-state truth is reported as vacuous.
    pub fn verify_property(
        program: &Program,
        property: &TemporalPropertyDeclaration,
        config: GlobalSolveConfig,
    ) -> PropertyVerificationResult {
        let bytecode = match source_program_bytecode(program) {
            Ok(bytecode) => bytecode,
            Err(reason) => {
                return PropertyVerificationResult::Unsupported {
                    property: property.clone(),
                    reason,
                };
            }
        };
        Self::verify_property_bytecode(&bytecode, property, config)
    }

    /// Prove or refute one source-declared property over the linked executable.
    pub fn verify_property_bytecode(
        program: &BytecodeProgram,
        property: &TemporalPropertyDeclaration,
        config: GlobalSolveConfig,
    ) -> PropertyVerificationResult {
        if let Some(reason) = dense_memory_config_error(config) {
            return PropertyVerificationResult::Unsupported {
                property: property.clone(),
                reason,
            };
        }
        if let Some(address) = property
            .touched_addresses()
            .into_iter()
            .find(|address| *address >= config.memory_cells as u64)
        {
            return PropertyVerificationResult::Unsupported {
                property: property.clone(),
                reason: format!(
                    "property cell {} is outside configured memory width {}",
                    address, config.memory_cells
                ),
            };
        }
        let ir = match Self::compile_bytecode(program, config) {
            Ok(ir) => ir,
            Err(error) => {
                return PropertyVerificationResult::Unsupported {
                    property: property.clone(),
                    reason: error.to_string(),
                }
            }
        };
        Self::verify_property_ir(program, property, config, &ir)
    }

    fn verify_property_ir(
        program: &BytecodeProgram,
        property: &TemporalPropertyDeclaration,
        config: GlobalSolveConfig,
        ir: &TemporalIr,
    ) -> PropertyVerificationResult {
        let exemplar = match Self::solve_ir(program, config, ir, None) {
            GlobalSolveResult::Found(witness) => witness,
            GlobalSolveResult::ProvenNoFixedPoint(no_fixed_point) => {
                return PropertyVerificationResult::Vacuous {
                    property: property.clone(),
                    no_fixed_point,
                }
            }
            GlobalSolveResult::Unknown {
                reason,
                constraint_digest,
                ..
            } => {
                return PropertyVerificationResult::Unknown {
                    property: property.clone(),
                    reason,
                    exemplar: None,
                    constraint_digest,
                }
            }
            GlobalSolveResult::Unsupported { reason } => {
                return PropertyVerificationResult::Unsupported {
                    property: property.clone(),
                    reason,
                }
            }
            GlobalSolveResult::InternalError { reason } => {
                return PropertyVerificationResult::InternalError {
                    property: property.clone(),
                    reason,
                }
            }
        };

        let violation = property_violation_assertion(property);
        match Self::solve_ir(program, config, ir, Some(&violation)) {
            GlobalSolveResult::Found(counterexample) => {
                if property_holds(property, &counterexample.memory) {
                    return PropertyVerificationResult::InternalError {
                        property: property.clone(),
                        reason: "solver's alleged property counterexample satisfies the property"
                            .to_string(),
                    };
                }
                PropertyVerificationResult::Refuted {
                    property: property.clone(),
                    counterexample,
                }
            }
            GlobalSolveResult::ProvenNoFixedPoint(certificate) => {
                PropertyVerificationResult::Proven {
                    property: property.clone(),
                    exemplar,
                    certificate,
                }
            }
            GlobalSolveResult::Unknown {
                reason,
                constraint_digest,
                ..
            } => PropertyVerificationResult::Unknown {
                property: property.clone(),
                reason,
                exemplar: Some(exemplar),
                constraint_digest,
            },
            GlobalSolveResult::Unsupported { reason } => PropertyVerificationResult::Unsupported {
                property: property.clone(),
                reason,
            },
            GlobalSolveResult::InternalError { reason } => {
                PropertyVerificationResult::InternalError {
                    property: property.clone(),
                    reason,
                }
            }
        }
    }

    fn solve_ir(
        program: &BytecodeProgram,
        config: GlobalSolveConfig,
        ir: &TemporalIr,
        extra_assertion: Option<&str>,
    ) -> GlobalSolveResult {
        let mut smt = ir.to_smt2(false);
        if let Some(assertion) = extra_assertion {
            smt.push_str(assertion);
            smt.push('\n');
        }
        let digest = fnv1a64(smt.as_bytes());

        // The symbolic IR does not yet model gas exhaustion. A complete proof
        // is therefore sound only when every represented acyclic path fits in
        // the configured executable instruction budget. SAT replay alone
        // cannot protect UNSAT from this mismatch because there is no witness
        // to replay.
        match replay_instruction_upper_bound_for(program) {
            Ok(Some(required)) if config.max_instructions < required => {
                return GlobalSolveResult::Unknown {
                    reason: format!(
                        "configured instruction limit {} is below the executable path bound {}; gas exhaustion is not encoded in the temporal formula",
                        config.max_instructions, required
                    ),
                    completeness: Some(ir.completeness),
                    constraint_digest: Some(digest),
                };
            }
            Ok(None) if ir.completeness.proves_global_unsat() => {
                return GlobalSolveResult::Unknown {
                    reason: "complete temporal IR has no finite executable path bound; gas exhaustion is not encoded in the temporal formula"
                        .to_string(),
                    completeness: Some(ir.completeness),
                    constraint_digest: Some(digest),
                };
            }
            Ok(_) => {}
            Err(reason) => return GlobalSolveResult::Unsupported { reason },
        }

        let mut z3_config = Z3Config::new();
        z3_config.set_model_generation(true);
        z3_config.set_timeout_msec(config.solver_timeout_ms);
        let context = Context::new(&z3_config);
        let solver = Solver::new(&context);
        // `Z3_solver_from_string` accepts declarations, definitions, and
        // assertions, but older libz3 releases treat session commands such as
        // set-logic/set-option as a parser error for this API.  The Context
        // above already carries those options.
        let solver_smt: String = smt.lines().filter(|line| !line.starts_with("(set-")).fold(
            String::new(),
            |mut output, line| {
                output.push_str(line);
                output.push('\n');
                output
            },
        );
        solver.from_string(solver_smt.as_bytes());
        #[cfg(test)]
        if std::env::var_os("OURO_DEBUG_SOLVER").is_some() {
            eprintln!("{}", smt);
            eprintln!("assertions: {:?}", solver.get_assertions());
        }

        match solver.check() {
            SatResult::Unsat if ir.completeness.proves_global_unsat() => {
                GlobalSolveResult::ProvenNoFixedPoint(UnsatCertificate {
                    constraint_digest: digest,
                    backend: "Z3",
                    completeness: ir.completeness,
                })
            }
            SatResult::Unsat => GlobalSolveResult::Unknown {
                reason: format!(
                    "no fixed point exists within the represented loop bound; {:?} is not a global proof",
                    ir.completeness
                ),
                completeness: Some(ir.completeness),
                constraint_digest: Some(digest),
            },
            SatResult::Unknown => GlobalSolveResult::Unknown {
                reason: solver
                    .get_reason_unknown()
                    .unwrap_or_else(|| "solver returned unknown".to_string()),
                completeness: Some(ir.completeness),
                constraint_digest: Some(digest),
            },
            SatResult::Sat => {
                let model = match solver.get_model() {
                    Some(model) => model,
                    None => {
                        return GlobalSolveResult::InternalError {
                            reason: "Z3 returned SAT without a model".to_string(),
                        }
                    }
                };
                let word_sort = Sort::bitvector(&context, 64);
                let anamnesis = Array::new_const(
                    &context,
                    "anamnesis",
                    &word_sort,
                    &word_sort,
                );
                let memory = match decode_array_model(&model, &anamnesis, config.memory_cells) {
                    Ok(memory) => memory,
                    Err(reason) => return GlobalSolveResult::InternalError { reason },
                };

                Self::replay(program, config, ir.completeness, digest, memory)
            }
        }
    }

    fn replay(
        program: &BytecodeProgram,
        config: GlobalSolveConfig,
        completeness: IrCompleteness,
        constraint_digest: u64,
        memory: Memory,
    ) -> GlobalSolveResult {
        replay_bytecode(program, config, completeness, constraint_digest, memory)
    }
}

fn dense_memory_config_error(config: GlobalSolveConfig) -> Option<String> {
    (config.memory_cells > crate::core::MAX_DENSE_MEMORY_CELLS).then(|| {
        format!(
            "global solver memory width {} exceeds dense witness limit {}",
            config.memory_cells,
            crate::core::MAX_DENSE_MEMORY_CELLS
        )
    })
}

fn source_program_bytecode(program: &Program) -> Result<BytecodeProgram, String> {
    let hir = HirProgram::resolve(program).map_err(|errors| {
        errors
            .into_iter()
            .map(|error| error.to_string())
            .collect::<Vec<_>>()
            .join("; ")
    })?;
    BytecodeProgram::compile(&hir).map_err(|error| error.to_string())
}

fn replay_bytecode(
    program: &BytecodeProgram,
    config: GlobalSolveConfig,
    completeness: IrCompleteness,
    constraint_digest: u64,
    memory: Memory,
) -> GlobalSolveResult {
    let mut anamnesis = match PagedMemory::with_size(config.memory_cells) {
        Ok(memory) => memory,
        Err(error) => {
            return GlobalSolveResult::InternalError {
                reason: format!("cannot construct bytecode replay memory: {error}"),
            }
        }
    };
    for (address, value) in memory.iter() {
        if *value != Value::ZERO {
            if let Err(error) = anamnesis.write(address, value.clone()) {
                return GlobalSolveResult::InternalError {
                    reason: format!("cannot initialize bytecode replay memory: {error}"),
                };
            }
        }
    }
    let vm = BytecodeVm::with_config(BytecodeVmConfig {
        max_instructions: config.max_instructions,
        memory_bounds: config.bounds_policy,
        ..BytecodeVmConfig::default()
    });
    match vm.run(program, &anamnesis) {
        Ok(epoch)
            if matches!(
                epoch.status,
                BytecodeVmStatus::Finished | BytecodeVmStatus::Halted
            ) =>
        {
            let replayed_present = dense_from_paged(&epoch.present);
            if replayed_present.numeric_values_equal(&memory) {
                GlobalSolveResult::Found(FixedPointWitness {
                    memory,
                    replayed_present,
                    output: epoch.output,
                    instructions_executed: epoch.instructions_executed,
                    constraint_digest,
                    completeness,
                })
            } else {
                GlobalSolveResult::InternalError {
                    reason: format!(
                        "solver model failed bytecode replay; differing cells: {:?}",
                        memory.diff(&replayed_present)
                    ),
                }
            }
        }
        Ok(epoch) if epoch.status == BytecodeVmStatus::Paradox => {
            GlobalSolveResult::InternalError {
                reason: "solver model replayed into bytecode PARADOX".to_string(),
            }
        }
        Ok(epoch) => GlobalSolveResult::InternalError {
            reason: format!("unexpected bytecode replay status {:?}", epoch.status),
        },
        Err(BytecodeVmError::GasExhausted { .. }) => GlobalSolveResult::Unknown {
            reason: format!(
                "solver model requires execution beyond the configured gas bound {}",
                config.max_instructions
            ),
            completeness: Some(completeness),
            constraint_digest: Some(constraint_digest),
        },
        Err(error) => GlobalSolveResult::InternalError {
            reason: format!("solver model failed bytecode replay: {error}"),
        },
    }
}

fn dense_from_paged(memory: &PagedMemory) -> Memory {
    let mut dense = Memory::with_size(memory.len());
    for (address, value) in memory.sparse_state() {
        dense.write(address, value);
    }
    dense
}

/// Decode a Z3 array value by its sparse representation rather than issuing a
/// separate `select` query for every memory cell.  The latter makes an almost
/// empty 65,536-cell witness retain tens of thousands of Z3 ASTs.  Z3 models
/// arrays either as a constant array plus `store` updates or as a finite
/// function interpretation; both representations are exact here.
fn decode_array_model<'ctx>(
    model: &Model<'ctx>,
    array: &Array<'ctx>,
    memory_cells: usize,
) -> Result<Memory, String> {
    if let Some(interp) = model.get_func_interp(&array.decl()) {
        let default = decode_dynamic_word(model, interp.get_else())
            .ok_or_else(|| "could not decode array model's default value".to_string())?;
        let mut memory = memory_with_default(memory_cells, default);
        for entry in interp.get_entries() {
            let arguments = entry.get_args();
            if arguments.len() != 1 {
                return Err(format!(
                    "array model entry had {} indices instead of one",
                    arguments.len()
                ));
            }
            let address = decode_dynamic_word(model, arguments[0].clone())
                .ok_or_else(|| "could not decode array model entry index".to_string())?;
            let value = decode_dynamic_word(model, entry.get_value())
                .ok_or_else(|| "could not decode array model entry value".to_string())?;
            if address < memory_cells as u64 {
                memory.write(address, Value::new(value));
            }
        }
        return Ok(memory);
    }

    let evaluated = model
        .eval(array, true)
        .ok_or_else(|| "could not evaluate anamnesis array in the solver model".to_string())?;
    let mut node = Dynamic::from_ast(&evaluated);
    // Stores are encountered outside-in; replay them inside-out so the outer
    // (latest) store wins when an index occurs more than once.
    let mut stores = Vec::new();

    let default = loop {
        match node.decl().kind() {
            DeclKind::STORE => {
                let mut children = node.children();
                if children.len() != 3 {
                    return Err(format!(
                        "array store had {} children instead of three",
                        children.len()
                    ));
                }
                let value = decode_dynamic_word(model, children.pop().unwrap())
                    .ok_or_else(|| "could not decode array store value".to_string())?;
                let address = decode_dynamic_word(model, children.pop().unwrap())
                    .ok_or_else(|| "could not decode array store index".to_string())?;
                stores.push((address, value));
                node = children.pop().unwrap();
            }
            DeclKind::CONST_ARRAY => {
                let children = node.children();
                if children.len() != 1 {
                    return Err(format!(
                        "constant array had {} children instead of one",
                        children.len()
                    ));
                }
                break decode_dynamic_word(model, children.into_iter().next().unwrap())
                    .ok_or_else(|| "could not decode constant array value".to_string())?;
            }
            kind => {
                return Err(format!(
                    "unsupported Z3 array model representation {:?}: {}",
                    kind, node
                ));
            }
        }
    };

    let mut memory = memory_with_default(memory_cells, default);
    for (address, value) in stores.into_iter().rev() {
        if address < memory_cells as u64 {
            memory.write(address, Value::new(value));
        }
    }
    Ok(memory)
}

fn decode_dynamic_word<'ctx>(model: &Model<'ctx>, value: Dynamic<'ctx>) -> Option<u64> {
    let word = value.as_bv()?;
    word.as_u64().or_else(|| {
        model
            .eval(&word, true)
            .and_then(|evaluated| evaluated.as_u64())
    })
}

fn memory_with_default(memory_cells: usize, default: u64) -> Memory {
    let mut memory = Memory::with_size(memory_cells);
    if default != 0 {
        for address in 0..memory_cells {
            memory.write(address as u64, Value::new(default));
        }
    }
    memory
}

/// Block one complete finite memory value.  The typed IR fixes anamnesis to a
/// zero-based finite present array, so cells not listed here are zero as well.
fn block_memory_assertion(memory: &Memory) -> String {
    let mut concrete = "((as const (Array (_ BitVec 64) (_ BitVec 64))) (_ bv0 64))".to_string();
    for (address, value) in memory.iter_nonzero() {
        concrete = format!(
            "(store {} (_ bv{} 64) (_ bv{} 64))",
            concrete, address, value.val
        );
    }
    format!("(assert (not (= anamnesis {})))", concrete)
}

fn property_atom(property: &TemporalPropertyDeclaration) -> String {
    property
        .predicate
        .as_ref()
        .map(property_predicate_smt)
        .unwrap_or_else(|| {
            property_comparison_smt(property.address, property.comparison, property.value)
        })
}

fn property_predicate_smt(predicate: &PropertyPredicate) -> String {
    match predicate {
        PropertyPredicate::Compare {
            cell,
            comparison,
            value,
        } => property_comparison_smt(cell.address, *comparison, *value),
        PropertyPredicate::Not(inner) => format!("(not {})", property_predicate_smt(inner)),
        PropertyPredicate::And(left, right) => format!(
            "(and {} {})",
            property_predicate_smt(left),
            property_predicate_smt(right)
        ),
        PropertyPredicate::Or(left, right) => format!(
            "(or {} {})",
            property_predicate_smt(left),
            property_predicate_smt(right)
        ),
    }
}

fn property_comparison_smt(address: u64, comparison: PropertyComparison, value: u64) -> String {
    let cell = format!("(select anamnesis (_ bv{} 64))", address);
    let value = format!("(_ bv{} 64)", value);
    match comparison {
        PropertyComparison::Eq => format!("(= {} {})", cell, value),
        PropertyComparison::Ne => format!("(not (= {} {}))", cell, value),
        PropertyComparison::Ult => format!("(bvult {} {})", cell, value),
        PropertyComparison::Ule => format!("(bvule {} {})", cell, value),
        PropertyComparison::Ugt => format!("(bvugt {} {})", cell, value),
        PropertyComparison::Uge => format!("(bvuge {} {})", cell, value),
    }
}

fn property_violation_assertion(property: &TemporalPropertyDeclaration) -> String {
    format!("(assert (not {}))", property_atom(property))
}

fn property_holds(property: &TemporalPropertyDeclaration, memory: &Memory) -> bool {
    property.predicate.as_ref().map_or_else(
        || {
            property_comparison_holds(
                memory.read(property.address).val,
                property.comparison,
                property.value,
            )
        },
        |predicate| property_predicate_holds(predicate, memory),
    )
}

fn property_predicate_holds(predicate: &PropertyPredicate, memory: &Memory) -> bool {
    match predicate {
        PropertyPredicate::Compare {
            cell,
            comparison,
            value,
        } => property_comparison_holds(memory.read(cell.address).val, *comparison, *value),
        PropertyPredicate::Not(inner) => !property_predicate_holds(inner, memory),
        PropertyPredicate::And(left, right) => {
            property_predicate_holds(left, memory) && property_predicate_holds(right, memory)
        }
        PropertyPredicate::Or(left, right) => {
            property_predicate_holds(left, memory) || property_predicate_holds(right, memory)
        }
    }
}

fn property_comparison_holds(actual: u64, comparison: PropertyComparison, value: u64) -> bool {
    match comparison {
        PropertyComparison::Eq => actual == value,
        PropertyComparison::Ne => actual != value,
        PropertyComparison::Ult => actual < value,
        PropertyComparison::Ule => actual <= value,
        PropertyComparison::Ugt => actual > value,
        PropertyComparison::Uge => actual >= value,
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn artifact_json(kind: &str, status: &str, detail: &str) -> String {
    format!(
        "{{\"schema\":\"ourochronos.verification/v1\",\"kind\":\"{}\",\"status\":\"{}\",{}}}",
        kind, status, detail
    )
}

fn completeness_json(completeness: IrCompleteness) -> String {
    match completeness {
        IrCompleteness::Complete => "{\"kind\":\"complete\"}".to_string(),
        IrCompleteness::BoundedLoops {
            loop_count,
            unroll_limit,
        } => format!(
            "{{\"kind\":\"bounded-loops\",\"loop_count\":{},\"unroll_limit\":{}}}",
            loop_count, unroll_limit
        ),
    }
}

fn certificate_json(certificate: &UnsatCertificate) -> String {
    format!(
        "{{\"backend\":\"{}\",\"completeness\":{},\"constraint_digest\":\"{:016x}\"}}",
        certificate.backend,
        completeness_json(certificate.completeness),
        certificate.constraint_digest
    )
}

fn optional_digest(digest: Option<u64>) -> String {
    digest
        .map(|value| format!("\"{:016x}\"", value))
        .unwrap_or_else(|| "null".to_string())
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '\"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character.is_control() => {
                escaped.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => escaped.push(character),
        }
    }
    escaped
}

/// Safely inline only reachable procedure calls for reference replay.  Unlike
/// the legacy convenience inliner, recursion is reported rather than recursed
/// through indefinitely.
#[cfg(test)]
pub(crate) fn inline_for_replay(program: &Program) -> Result<Program, String> {
    let procedures: HashMap<&str, &[Stmt]> = program
        .procedures
        .iter()
        .map(|procedure| (procedure.name.as_str(), procedure.body.as_slice()))
        .collect();

    fn expand(
        stmts: &[Stmt],
        procedures: &HashMap<&str, &[Stmt]>,
        active: &mut Vec<String>,
    ) -> Result<Vec<Stmt>, String> {
        let mut result = Vec::new();
        for stmt in stmts {
            match stmt {
                Stmt::Call { name } => {
                    if active.iter().any(|item| item == name) {
                        return Err(format!(
                            "recursive procedure '{}' cannot be replay-inlined",
                            name
                        ));
                    }
                    let body = procedures
                        .get(name.as_str())
                        .ok_or_else(|| format!("unknown procedure '{}' during replay", name))?;
                    active.push(name.clone());
                    result.extend(expand(body, procedures, active)?);
                    active.pop();
                }
                Stmt::If {
                    then_branch,
                    else_branch,
                } => result.push(Stmt::If {
                    then_branch: expand(then_branch, procedures, active)?,
                    else_branch: else_branch
                        .as_ref()
                        .map(|branch| expand(branch, procedures, active))
                        .transpose()?,
                }),
                Stmt::While { cond, body } => result.push(Stmt::While {
                    cond: expand(cond, procedures, active)?,
                    body: expand(body, procedures, active)?,
                }),
                Stmt::Block(body) => {
                    result.push(Stmt::Block(expand(body, procedures, active)?));
                }
                Stmt::TemporalScope {
                    base,
                    size,
                    cell_bits,
                    body,
                } => result.push(Stmt::TemporalScope {
                    base: *base,
                    size: *size,
                    cell_bits: *cell_bits,
                    body: expand(body, procedures, active)?,
                }),
                Stmt::Op(_)
                | Stmt::Push(_)
                | Stmt::PushConstant { .. }
                | Stmt::ReadTemporal { .. }
                | Stmt::PushQuote(_) => result.push(stmt.clone()),
            }
        }
        Ok(result)
    }

    let mut replay = program.clone();
    replay.body = expand(&program.body, &procedures, &mut Vec::new())?;
    replay.procedures.clear();
    Ok(replay)
}

/// Conservative maximum number of statement dispatches on any acyclic replay
/// path after safe procedure expansion. `None` means a reachable WHILE loop is
/// present; loop-bearing IR is already marked bounded and cannot support
/// complete UNSAT.
fn replay_instruction_upper_bound_for(program: &BytecodeProgram) -> Result<Option<u64>, String> {
    bytecode_instruction_upper_bound(program)
}

#[derive(Clone, Copy)]
struct BytecodeFlowBound {
    returned: Option<u64>,
    terminated: Option<u64>,
}

impl BytecodeFlowBound {
    fn prefix(self, cost: u64) -> Option<Self> {
        Some(Self {
            returned: match self.returned {
                Some(value) => Some(value.checked_add(cost)?),
                None => None,
            },
            terminated: match self.terminated {
                Some(value) => Some(value.checked_add(cost)?),
                None => None,
            },
        })
    }

    fn merge(self, other: Self) -> Self {
        Self {
            returned: max_optional(self.returned, other.returned),
            terminated: max_optional(self.terminated, other.terminated),
        }
    }

    fn maximum(self) -> Option<u64> {
        max_optional(self.returned, self.terminated)
    }
}

fn max_optional(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

enum BytecodeBoundAttempt {
    Ready(BytecodeFlowBound),
    Need(usize),
    Unbounded,
}

/// Conservative exact-gas upper bound over the reachable acyclic bytecode
/// CFG. Both IF arms contribute their maximum, repeated call sites are counted
/// independently, and an always-terminal callee suppresses dead continuation
/// code. A reachable loop, recursive call component, or arithmetic overflow
/// returns `None`, which prevents complete UNSAT claims.
fn bytecode_instruction_upper_bound(program: &BytecodeProgram) -> Result<Option<u64>, String> {
    program
        .validate()
        .map_err(|error| format!("cannot bound malformed bytecode: {error}"))?;
    let unit_count = program.procedures.len().saturating_add(1);
    let mut summaries = vec![None::<BytecodeFlowBound>; unit_count];
    let mut active = HashSet::<usize>::new();
    let mut tasks = vec![0usize];
    active.insert(0);

    while let Some(&unit) = tasks.last() {
        match summarize_bytecode_unit(program, unit, &summaries)? {
            BytecodeBoundAttempt::Ready(summary) => {
                summaries[unit] = Some(summary);
                active.remove(&unit);
                tasks.pop();
            }
            BytecodeBoundAttempt::Need(callee) => {
                if callee >= unit_count {
                    return Err(format!("bytecode call references missing unit {callee}"));
                }
                if summaries[callee].is_some() {
                    continue;
                }
                if !active.insert(callee) {
                    return Ok(None);
                }
                tasks.push(callee);
            }
            BytecodeBoundAttempt::Unbounded => return Ok(None),
        }
    }

    Ok(summaries[0].and_then(BytecodeFlowBound::maximum))
}

fn summarize_bytecode_unit(
    program: &BytecodeProgram,
    unit: usize,
    summaries: &[Option<BytecodeFlowBound>],
) -> Result<BytecodeBoundAttempt, String> {
    let range = if unit == 0 {
        program.main
    } else {
        program
            .procedures
            .get(unit - 1)
            .ok_or_else(|| format!("missing bytecode procedure unit {unit}"))?
            .range
    };
    let length = usize::try_from(range.end - range.start)
        .map_err(|_| "bytecode unit length does not fit host".to_string())?;
    let mut flows = Vec::<BytecodeBoundAttempt>::with_capacity(length);
    flows.resize_with(length, || BytecodeBoundAttempt::Unbounded);

    let successor = |target: u32| -> Result<usize, String> {
        if target < range.start || target >= range.end {
            return Err(format!(
                "bytecode bound target {target} leaves unit {}..{}",
                range.start, range.end
            ));
        }
        Ok((target - range.start) as usize)
    };

    for pc in (range.start..range.end).rev() {
        let index = (pc - range.start) as usize;
        let instruction = program
            .instructions
            .get(pc as usize)
            .ok_or_else(|| format!("missing bytecode instruction {pc}"))?;
        let next = || successor(pc + 1);
        let prefixed = |attempt: &BytecodeBoundAttempt, cost: u64| -> BytecodeBoundAttempt {
            match attempt {
                BytecodeBoundAttempt::Ready(flow) => flow
                    .prefix(cost)
                    .map(BytecodeBoundAttempt::Ready)
                    .unwrap_or(BytecodeBoundAttempt::Unbounded),
                BytecodeBoundAttempt::Need(unit) => BytecodeBoundAttempt::Need(*unit),
                BytecodeBoundAttempt::Unbounded => BytecodeBoundAttempt::Unbounded,
            }
        };
        flows[index] = match *instruction {
            Instruction::Return => BytecodeBoundAttempt::Ready(BytecodeFlowBound {
                returned: Some(1),
                terminated: None,
            }),
            Instruction::Primitive(OpCode::Halt | OpCode::Paradox) => {
                BytecodeBoundAttempt::Ready(BytecodeFlowBound {
                    returned: None,
                    terminated: Some(1),
                })
            }
            Instruction::WhileFalse { .. } | Instruction::LoopBack { .. } => {
                BytecodeBoundAttempt::Unbounded
            }
            Instruction::IfFalse { else_target, .. } => {
                let then_flow = &flows[next()?];
                let else_flow = &flows[successor(else_target)?];
                match (then_flow, else_flow) {
                    (BytecodeBoundAttempt::Ready(left), BytecodeBoundAttempt::Ready(right)) => left
                        .merge(*right)
                        .prefix(1)
                        .map(BytecodeBoundAttempt::Ready)
                        .unwrap_or(BytecodeBoundAttempt::Unbounded),
                    (BytecodeBoundAttempt::Need(unit), _) => BytecodeBoundAttempt::Need(*unit),
                    (_, BytecodeBoundAttempt::Need(unit)) => BytecodeBoundAttempt::Need(*unit),
                    _ => BytecodeBoundAttempt::Unbounded,
                }
            }
            Instruction::Jump { target } => prefixed(&flows[successor(target)?], 1),
            Instruction::CallProcedure(target) => {
                let callee = target.index().saturating_add(1);
                match summaries.get(callee).copied().flatten() {
                    None => BytecodeBoundAttempt::Need(callee),
                    Some(callee_flow) => {
                        let direct_terminal = match callee_flow.terminated {
                            Some(cost) => cost.checked_add(1),
                            None => Some(0),
                        };
                        let Some(direct_terminal) = direct_terminal else {
                            flows[index] = BytecodeBoundAttempt::Unbounded;
                            continue;
                        };
                        let mut combined = BytecodeFlowBound {
                            returned: None,
                            terminated: callee_flow.terminated.map(|_| direct_terminal),
                        };
                        match callee_flow.returned {
                            None => BytecodeBoundAttempt::Ready(combined),
                            Some(return_cost) => match &flows[next()?] {
                                BytecodeBoundAttempt::Ready(continuation) => {
                                    let continued = return_cost
                                        .checked_add(1)
                                        .and_then(|cost| continuation.prefix(cost));
                                    match continued {
                                        Some(continued) => {
                                            combined = combined.merge(continued);
                                            BytecodeBoundAttempt::Ready(combined)
                                        }
                                        None => BytecodeBoundAttempt::Unbounded,
                                    }
                                }
                                BytecodeBoundAttempt::Need(unit) => {
                                    BytecodeBoundAttempt::Need(*unit)
                                }
                                BytecodeBoundAttempt::Unbounded => BytecodeBoundAttempt::Unbounded,
                            },
                        }
                    }
                }
            }
            _ => prefixed(&flows[next()?], 1),
        };
    }

    Ok(match flows.into_iter().next() {
        Some(flow) => flow,
        None => BytecodeBoundAttempt::Unbounded,
    })
}

#[cfg(test)]
fn replay_instruction_upper_bound(program: &Program) -> Result<Option<u64>, String> {
    #[derive(Clone, Copy)]
    struct FlowBound {
        cost: u64,
        falls_through: bool,
    }

    fn block(statements: &[Stmt]) -> Option<FlowBound> {
        let mut total = 0u64;
        for statement in statements {
            let flow = match statement {
                Stmt::Op(op) => FlowBound {
                    cost: 1,
                    falls_through: !matches!(op, OpCode::Halt | OpCode::Paradox),
                },
                Stmt::Push(_) | Stmt::PushConstant { .. } | Stmt::PushQuote(_) => FlowBound {
                    cost: 1,
                    falls_through: true,
                },
                Stmt::ReadTemporal { .. } => FlowBound {
                    cost: 2,
                    falls_through: true,
                },
                Stmt::Call { .. } => return None,
                Stmt::Block(body) => {
                    let body = block(body)?;
                    FlowBound {
                        cost: 1u64.saturating_add(body.cost),
                        falls_through: body.falls_through,
                    }
                }
                Stmt::If {
                    then_branch,
                    else_branch,
                } => {
                    let then_flow = block(then_branch)?;
                    let else_flow = match else_branch {
                        Some(branch) => block(branch)?,
                        None => FlowBound {
                            cost: 0,
                            falls_through: true,
                        },
                    };
                    FlowBound {
                        cost: 1u64.saturating_add(then_flow.cost.max(else_flow.cost)),
                        falls_through: then_flow.falls_through || else_flow.falls_through,
                    }
                }
                Stmt::While { .. } => return None,
                Stmt::TemporalScope { body, .. } => {
                    let body = block(body)?;
                    FlowBound {
                        cost: 1u64.saturating_add(body.cost),
                        falls_through: body.falls_through,
                    }
                }
            };
            total = total.saturating_add(flow.cost);
            if !flow.falls_through {
                return Some(FlowBound {
                    cost: total,
                    falls_through: false,
                });
            }
        }
        Some(FlowBound {
            cost: total,
            falls_through: true,
        })
    }

    let replay = inline_for_replay(program)?;
    Ok(block(&replay.body).map(|bound| bound.cost))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::BytecodeProgram;
    use crate::hir::HirProgram;
    use crate::parser::parse;
    use z3::ast::BV;

    fn small_config() -> GlobalSolveConfig {
        GlobalSolveConfig {
            memory_cells: 4,
            solver_timeout_ms: 5_000,
            ..GlobalSolveConfig::default()
        }
    }

    fn bytecode(program: &Program) -> BytecodeProgram {
        BytecodeProgram::compile(&HirProgram::resolve(program).unwrap()).unwrap()
    }

    #[test]
    fn global_solver_rejects_dense_witness_width_before_lowering() {
        let fixed = bytecode(&parse("0 ORACLE 0 PROPHECY").unwrap());
        let result = GlobalFixedPointSolver::solve_bytecode(
            &fixed,
            GlobalSolveConfig {
                memory_cells: crate::core::MAX_DENSE_MEMORY_CELLS + 1,
                ..GlobalSolveConfig::default()
            },
        );
        assert!(matches!(
            result,
            GlobalSolveResult::Unsupported { ref reason }
                if reason.contains("dense witness limit")
        ));
    }

    #[test]
    fn linked_bytecode_is_the_solver_and_replay_authority() {
        let fixed = parse("42 0 PROPHECY").unwrap();
        match GlobalFixedPointSolver::solve_bytecode(&bytecode(&fixed), small_config()) {
            GlobalSolveResult::Found(witness) => {
                assert_eq!(witness.memory.read(0).val, 42);
                assert!(witness.is_replay_verified());
            }
            result => panic!("unexpected bytecode result: {result:?}"),
        }

        let paradox = parse("0 ORACLE NOT 0 PROPHECY").unwrap();
        match GlobalFixedPointSolver::solve_bytecode(&bytecode(&paradox), small_config()) {
            GlobalSolveResult::ProvenNoFixedPoint(certificate) => {
                assert_eq!(certificate.completeness, IrCompleteness::Complete);
            }
            result => panic!("unexpected bytecode result: {result:?}"),
        }
    }

    #[test]
    fn bytecode_gas_bound_counts_calls_and_ignores_terminal_dead_code() {
        let repeated = parse("PROCEDURE step { NOP } step step").unwrap();
        assert_eq!(
            bytecode_instruction_upper_bound(&bytecode(&repeated)).unwrap(),
            Some(7)
        );

        let terminal =
            parse("PROCEDURE stop { HALT WHILE { 1 } { NOP } } stop WHILE { 1 } { NOP }").unwrap();
        assert_eq!(
            bytecode_instruction_upper_bound(&bytecode(&terminal)).unwrap(),
            Some(2)
        );
    }

    #[test]
    fn low_bytecode_gas_cannot_produce_complete_unsat() {
        let paradox = bytecode(&parse("0 ORACLE NOT 0 PROPHECY").unwrap());
        let mut config = small_config();
        config.max_instructions = 1;
        match GlobalFixedPointSolver::solve_bytecode(&paradox, config) {
            GlobalSolveResult::Unknown { reason, .. } => {
                assert!(reason.contains("instruction limit"), "{reason}");
                assert!(reason.contains("not encoded"), "{reason}");
            }
            result => panic!("low gas was overclaimed: {result:?}"),
        }
    }

    #[test]
    fn sparse_array_decoder_preserves_defaults_overrides_and_store_order() {
        let context = Context::new(&Z3Config::new());
        let word_sort = Sort::bitvector(&context, 64);
        let array = Array::new_const(&context, "array_under_test", &word_sort, &word_sort);
        let default = BV::from_u64(&context, 7, 64);
        let one = BV::from_u64(&context, 1, 64);
        let eight = BV::from_u64(&context, 8, 64);
        let nine = BV::from_u64(&context, 9, 64);
        let eleven = BV::from_u64(&context, 11, 64);
        let ignored = BV::from_u64(&context, 13, 64);
        let explicit = Array::const_array(&context, &word_sort, &default)
            .store(&one, &nine)
            .store(&one, &eleven)
            .store(&eight, &ignored);
        let solver = Solver::new(&context);
        solver.assert(&array._eq(&explicit));
        assert_eq!(solver.check(), SatResult::Sat);
        let model = solver.get_model().unwrap();

        let memory = decode_array_model(&model, &array, 4).unwrap();
        assert_eq!(memory.read(0).val, 7);
        assert_eq!(memory.read(1).val, 11);
        assert_eq!(memory.read(2).val, 7);
        assert_eq!(memory.read(3).val, 7);
    }

    #[test]
    fn globally_finds_a_fixed_point_unreachable_from_zero_orbit() {
        let program =
            parse("0 ORACLE DUP 42 EQ IF { 0 PROPHECY } ELSE { POP 0 ORACLE NOT 0 PROPHECY }")
                .unwrap();
        match GlobalFixedPointSolver::solve(&program, small_config()) {
            GlobalSolveResult::Found(witness) => {
                assert_eq!(witness.memory.read(0).val, 42);
                assert!(witness.is_replay_verified());
            }
            result => panic!("unexpected result: {:?}", result),
        }
    }

    #[test]
    fn globally_proves_logical_negation_has_no_point_fixed_state() {
        let program = parse("0 ORACLE NOT 0 PROPHECY").unwrap();
        match GlobalFixedPointSolver::solve(&program, small_config()) {
            GlobalSolveResult::ProvenNoFixedPoint(certificate) => {
                assert_eq!(certificate.completeness, IrCompleteness::Complete);
            }
            result => panic!("unexpected result: {:?}", result),
        }
    }

    #[test]
    fn low_gas_cannot_be_overclaimed_as_complete_unsat() {
        let program = parse("0 ORACLE NOT 0 PROPHECY").unwrap();
        let mut config = small_config();
        config.max_instructions = 1;
        match GlobalFixedPointSolver::solve(&program, config) {
            GlobalSolveResult::Unknown { reason, .. } => {
                assert!(reason.contains("instruction limit"), "{}", reason);
                assert!(reason.contains("not encoded"), "{}", reason);
            }
            result => panic!("low-gas result was overclaimed: {:?}", result),
        }
    }

    #[test]
    fn replay_gas_bound_follows_executable_termination() {
        let halted = parse("HALT 0 ORACLE NOT 0 PROPHECY").unwrap();
        assert_eq!(replay_instruction_upper_bound(&halted).unwrap(), Some(1));

        let dead_loop = parse("0 ORACLE NOT 0 PROPHECY HALT WHILE { 0 } { NOP }").unwrap();
        assert_eq!(replay_instruction_upper_bound(&dead_loop).unwrap(), Some(6));

        let branch_halts = parse("1 IF { HALT } ELSE { HALT } WHILE { 0 } { NOP }").unwrap();
        assert_eq!(
            replay_instruction_upper_bound(&branch_halts).unwrap(),
            Some(3)
        );

        let procedure_halts = parse("PROCEDURE stop { HALT } stop WHILE { 0 } { NOP }").unwrap();
        assert_eq!(
            replay_instruction_upper_bound(&procedure_halts).unwrap(),
            Some(1)
        );

        let scoped_halt = parse("TEMPORAL 0 1 BITS 1 { HALT WHILE { 0 } { NOP } }").unwrap();
        assert_eq!(
            replay_instruction_upper_bound(&scoped_halt).unwrap(),
            Some(2)
        );
    }

    #[test]
    fn dead_loop_cannot_bypass_complete_ir_gas_precondition() {
        let program = parse("0 ORACLE NOT 0 PROPHECY HALT WHILE { 0 } { NOP }").unwrap();
        let mut config = small_config();
        config.max_instructions = 1;
        match GlobalFixedPointSolver::solve(&program, config) {
            GlobalSolveResult::Unknown { reason, .. } => {
                assert!(reason.contains("path bound 6"), "{reason}");
            }
            result => panic!("low-gas result was overclaimed: {result:?}"),
        }

        config.max_instructions = 6;
        assert!(matches!(
            GlobalFixedPointSolver::solve(&program, config),
            GlobalSolveResult::ProvenNoFixedPoint(_)
        ));
    }

    #[test]
    fn paradox_eliminates_invalid_models_and_selects_42() {
        let program = parse("0 ORACLE DUP 42 EQ IF { 0 PROPHECY } ELSE { POP PARADOX }").unwrap();
        match GlobalFixedPointSolver::solve(&program, small_config()) {
            GlobalSolveResult::Found(witness) => assert_eq!(witness.memory.read(0).val, 42),
            result => panic!("unexpected result: {:?}", result),
        }
    }

    #[test]
    fn bounded_loop_unsat_is_unknown_not_a_global_proof() {
        let program = parse("0 ORACLE WHILE { DUP 100 GT } { 1 SUB } 0 PROPHECY").unwrap();
        let mut config = small_config();
        config.loop_unroll_limit = 0;
        match GlobalFixedPointSolver::solve(&program, config) {
            GlobalSolveResult::Unknown {
                completeness: Some(IrCompleteness::BoundedLoops { .. }),
                ..
            }
            | GlobalSolveResult::Found(_) => {}
            result => panic!("bounded result was overclaimed: {:?}", result),
        }
    }

    #[test]
    fn all_fixed_analysis_proves_uniqueness() {
        let program = parse("0 ORACLE DUP 42 EQ IF { 0 PROPHECY } ELSE { POP PARADOX }").unwrap();
        match GlobalFixedPointSolver::analyze_uniqueness(&program, small_config()) {
            GlobalUniquenessResult::Unique {
                witness,
                certificate,
            } => {
                assert_eq!(witness.memory.read(0).val, 42);
                assert_eq!(certificate.completeness, IrCompleteness::Complete);
            }
            result => panic!("unexpected result: {:?}", result),
        }
    }

    #[test]
    fn all_fixed_analysis_returns_two_replayed_ambiguity_witnesses() {
        let program = parse("0 ORACLE 0 PROPHECY").unwrap();
        match GlobalFixedPointSolver::analyze_uniqueness(&program, small_config()) {
            GlobalUniquenessResult::Multiple {
                first,
                second,
                differing_cells,
            } => {
                assert!(first.is_replay_verified());
                assert!(second.is_replay_verified());
                assert!(!differing_cells.is_empty());
            }
            result => panic!("unexpected result: {:?}", result),
        }
    }

    #[test]
    fn typed_temporal_scope_is_relative_and_truncates_writes() {
        let program = parse("TEMPORAL 2 1 BITS 2 { 7 0 PROPHECY }").unwrap();
        match GlobalFixedPointSolver::analyze_uniqueness(&program, small_config()) {
            GlobalUniquenessResult::Unique { witness, .. } => {
                assert_eq!(witness.memory.read(0).val, 0);
                assert_eq!(witness.memory.read(2).val, 3);
                assert!(witness.is_replay_verified());
            }
            result => panic!("unexpected result: {:?}", result),
        }
    }

    #[test]
    fn typed_two_bit_scope_can_prove_a_modular_cycle_has_no_point_state() {
        let program = parse("TEMPORAL 2 1 BITS 2 { 0 ORACLE 1 ADD 0 PROPHECY }").unwrap();
        match GlobalFixedPointSolver::solve(&program, small_config()) {
            GlobalSolveResult::ProvenNoFixedPoint(_) => {}
            result => panic!("unexpected result: {:?}", result),
        }
    }

    #[test]
    fn all_fixed_property_is_proved_for_every_model() {
        let program = parse(
            "PROPERTY answer { ALL_FIXED CELL 0 EQ 42; }\n\
             0 ORACLE DUP 42 EQ IF { 0 PROPHECY } ELSE { POP PARADOX }",
        )
        .unwrap();
        let property = &program.temporal_properties[0];
        match GlobalFixedPointSolver::verify_property(&program, property, small_config()) {
            PropertyVerificationResult::Proven {
                exemplar,
                certificate,
                ..
            } => {
                assert_eq!(exemplar.memory.read(0).val, 42);
                assert_eq!(certificate.completeness, IrCompleteness::Complete);
            }
            result => panic!("unexpected result: {:?}", result),
        }
    }

    #[test]
    fn boolean_named_property_is_proved_from_bytecode_and_reports_sound_slice() {
        let program = parse(
            "TEMPORAL answer @ 0 DEFAULT 9;\n\
             TEMPORAL spare @ 1 DEFAULT 8;\n\
             PROPERTY safe { ALL_FIXED CELL answer EQ 42 AND NOT (CELL spare NE 0); }\n\
             0 ORACLE DUP 42 EQ IF { 0 PROPHECY } ELSE { POP PARADOX }",
        )
        .unwrap();
        let hir = crate::hir::HirProgram::resolve(&program).unwrap();
        let bytecode = BytecodeProgram::compile(&hir).unwrap();
        let property = &program.temporal_properties[0];
        let result =
            GlobalFixedPointSolver::verify_property_bytecode(&bytecode, property, small_config());
        assert!(matches!(result, PropertyVerificationResult::Proven { .. }));
        let json = result.to_json();
        assert!(json.contains("\"name\":\"answer\""), "{json}");
        assert!(json.contains("\"addresses\":[0,1]"), "{json}");
        assert!(json.contains("\"incremental_reuse\":false"), "{json}");
    }

    #[test]
    fn all_fixed_property_returns_a_replayed_counterexample() {
        let program =
            parse("PROPERTY zero { ALL_FIXED CELL 0 EQ 0; }\n0 ORACLE 0 PROPHECY").unwrap();
        let property = &program.temporal_properties[0];
        match GlobalFixedPointSolver::verify_property(&program, property, small_config()) {
            PropertyVerificationResult::Refuted { counterexample, .. } => {
                assert_ne!(counterexample.memory.read(0).val, 0);
                assert!(counterexample.is_replay_verified());
            }
            result => panic!("unexpected result: {:?}", result),
        }
    }

    #[test]
    fn all_fixed_property_reports_vacuous_truth() {
        let program =
            parse("PROPERTY anything { ALL_FIXED CELL 0 EQ 7; }\n0 ORACLE NOT 0 PROPHECY").unwrap();
        let property = &program.temporal_properties[0];
        assert!(matches!(
            GlobalFixedPointSolver::verify_property(&program, property, small_config()),
            PropertyVerificationResult::Vacuous { .. }
        ));
    }

    #[test]
    fn typed_ir_fixed_state_cardinality_matches_exhaustive_reference_vm() {
        use crate::temporal::transition_graph::{ProgramGraphConfig, ProgramTransitionAnalyzer};

        let sources = [
            "TEMPORAL 0 1 BITS 2 { 0 ORACLE 0 PROPHECY }",
            "TEMPORAL 0 1 BITS 2 { 0 ORACLE NOT 0 PROPHECY }",
            "TEMPORAL 0 1 BITS 2 { 0 ORACLE 1 ADD 0 PROPHECY }",
            "TEMPORAL 0 1 BITS 2 { 0 ORACLE 3 XOR 0 PROPHECY }",
            "TEMPORAL 0 1 BITS 2 { 0 ORACLE 2 MUL 0 PROPHECY }",
            "TEMPORAL 0 1 BITS 2 { 0 ORACLE DUP 2 LT IF { NOT } ELSE { } 0 PROPHECY }",
        ];
        let solve_config = GlobalSolveConfig {
            memory_cells: 1,
            ..small_config()
        };
        let graph_config = ProgramGraphConfig {
            memory_cells: 1,
            cell_bits: 2,
            ..ProgramGraphConfig::default()
        };

        for source in sources {
            let program = parse(source).unwrap();
            let exhaustive = ProgramTransitionAnalyzer::analyze(&program, graph_config).unwrap();
            let fixed_count = exhaustive.recurrent.fixed_states.len();
            let symbolic = GlobalFixedPointSolver::analyze_uniqueness(&program, solve_config);
            match (fixed_count, symbolic) {
                (0, GlobalUniquenessResult::NoFixedPoint(_)) => {}
                (1, GlobalUniquenessResult::Unique { .. }) => {}
                (count, GlobalUniquenessResult::Multiple { .. }) if count > 1 => {}
                (count, result) => panic!(
                    "IR/VM cardinality mismatch for '{}': exhaustive {}, symbolic {:?}",
                    source, count, result
                ),
            }
        }
    }
}
