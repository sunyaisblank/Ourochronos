//! Machine-checked finite-instance evidence for declared PSPACE/Deutsch families.
//!
//! A source `FAMILY` declaration contains asymptotic claims. One finite run
//! cannot prove polynomial-time uniform generation or supply Nature's ideal
//! Deutsch selector. This module nevertheless discharges every obligation that
//! is decidable for one explicitly bounded deterministic instance: exact
//! polynomial resource bounds, transition totality and closure, isolation from
//! external state, and one Boolean readout across every recurrent class.

use crate::admission::{admit_program, AdmissionConfig};
use crate::ast::Program;
use crate::bytecode::BytecodeProgram;
use crate::complexity::PspaceFamilyContract;
use crate::core::{BoundsPolicy, OutputItem};
use crate::temporal::transition_graph::{
    BytecodeTransitionAnalyzer, DeterministicTransitionGraph, ProgramGraphAnalysis,
    ProgramGraphConfig, TransitionGraphError, MAX_RECURRENT_FROZEN_INPUTS, MAX_RECURRENT_STATES,
};
use std::fmt::{self, Write as _};

/// Version of the finite FAMILY-instance certificate envelope.
pub const PSPACE_INSTANCE_CERTIFICATE_VERSION: u16 = 1;
/// A specialized instance has one chronology-respecting Boolean decision bit.
pub const PSPACE_INSTANCE_DECISION_BITS: u128 = 1;

/// Minimal observable classification needed to check recurrent decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PspaceReadoutEvidence {
    Boolean(u8),
    Other,
}

/// Complete finite transition table carried by a verified certificate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PspaceTransitionEvidence {
    pub successors: Vec<u32>,
    pub instructions_executed: Vec<u64>,
    pub maximum_stack_depths: Vec<usize>,
    pub maximum_dynamic_bytes: Vec<usize>,
    pub maximum_call_depths: Vec<usize>,
    pub maximum_temporal_depths: Vec<usize>,
    pub readouts: Vec<PspaceReadoutEvidence>,
}

/// Failure of the VM-independent certificate checker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PspaceCertificateError {
    UnsupportedVersion {
        found: u16,
    },
    FamilyNameMismatch,
    MissingDeclaration {
        obligation: &'static str,
    },
    PolynomialOverflow {
        field: &'static str,
    },
    EvaluatedBoundMismatch {
        field: &'static str,
    },
    ContractDigestMismatch,
    InvalidDecision {
        found: u8,
    },
    InputLengthMismatch {
        declared: u64,
        found: usize,
    },
    NonBooleanInput {
        index: usize,
        value: u8,
    },
    InputTooLarge {
        found: usize,
    },
    InvalidDomain,
    InvalidProgramCounterBits {
        found: u32,
    },
    DomainTooLarge {
        exponent: usize,
    },
    StateCountMismatch {
        expected: usize,
        found: usize,
    },
    EvidenceLengthMismatch,
    InvalidSuccessor {
        state: usize,
        successor: u32,
    },
    MaximumStepsMismatch,
    WorkspaceMaximumMismatch {
        field: &'static str,
        expected: usize,
        found: usize,
    },
    RecurrentClassCountMismatch,
    InvalidRecurrentReadout {
        class: usize,
        state: usize,
    },
    RecurrentReadoutDisagreement {
        class: usize,
        state: usize,
    },
    CellBoundExceeded,
    ChronologyBoundExceeded,
    TransitionBoundExceeded,
    ChronologyMeasurementOverflow,
    ChronologyMeasurementMismatch,
    TransitionDigestMismatch,
}

impl fmt::Display for PspaceCertificateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedVersion { found } => write!(
                formatter,
                "unsupported FAMILY certificate version {found}; expected {PSPACE_INSTANCE_CERTIFICATE_VERSION}"
            ),
            Self::FamilyNameMismatch => {
                formatter.write_str("certificate family name differs from its retained contract")
            }
            Self::MissingDeclaration { obligation } => {
                write!(formatter, "retained contract does not declare {obligation}")
            }
            Self::PolynomialOverflow { field } => {
                write!(formatter, "retained {field} polynomial overflows u128")
            }
            Self::EvaluatedBoundMismatch { field } => {
                write!(formatter, "retained evaluated {field} bound is incorrect")
            }
            Self::ContractDigestMismatch => {
                formatter.write_str("retained contract identifier is incorrect")
            }
            Self::InvalidDecision { found } => {
                write!(formatter, "certificate decision {found} is not Boolean")
            }
            Self::InputLengthMismatch { declared, found } => write!(
                formatter,
                "certificate input length {found} differs from declared input_bits {declared}"
            ),
            Self::NonBooleanInput { index, value } => write!(
                formatter,
                "certificate input bit {index} has non-Boolean value {value}"
            ),
            Self::InputTooLarge { found } => write!(
                formatter,
                "certificate input length {found} exceeds hard ceiling {MAX_RECURRENT_FROZEN_INPUTS}"
            ),
            Self::InvalidDomain => formatter
                .write_str("certificate temporal domain requires cells > 0 and bits in 1..=64"),
            Self::InvalidProgramCounterBits { found } => write!(
                formatter,
                "certificate program-counter width {found} is outside the bytecode range 1..=32"
            ),
            Self::DomainTooLarge { exponent } => write!(
                formatter,
                "certificate domain exponent {exponent} is not explicitly representable"
            ),
            Self::StateCountMismatch { expected, found } => write!(
                formatter,
                "certificate state count {found} differs from exact domain size {expected}"
            ),
            Self::EvidenceLengthMismatch => {
                formatter.write_str("certificate transition-evidence vectors have different lengths")
            }
            Self::InvalidSuccessor { state, successor } => write!(
                formatter,
                "certificate state {state} targets out-of-domain state {successor}"
            ),
            Self::MaximumStepsMismatch => {
                formatter.write_str("certificate maximum transition step count is incorrect")
            }
            Self::WorkspaceMaximumMismatch {
                field,
                expected,
                found,
            } => write!(
                formatter,
                "certificate {field} maximum {found} differs from transition evidence maximum {expected}"
            ),
            Self::RecurrentClassCountMismatch => {
                formatter.write_str("certificate recurrent-class count is incorrect")
            }
            Self::InvalidRecurrentReadout { class, state } => write!(
                formatter,
                "certificate recurrent class {class}, state {state} lacks one Boolean readout"
            ),
            Self::RecurrentReadoutDisagreement { class, state } => write!(
                formatter,
                "certificate recurrent class {class}, state {state} disagrees with the retained decision"
            ),
            Self::CellBoundExceeded => {
                formatter.write_str("certificate temporal-cell count exceeds its evaluated bound")
            }
            Self::ChronologyBoundExceeded => formatter
                .write_str("certificate chronology workspace exceeds its evaluated bound"),
            Self::TransitionBoundExceeded => formatter
                .write_str("certificate transition work exceeds its evaluated bound"),
            Self::ChronologyMeasurementOverflow => {
                formatter.write_str("certificate chronology measurement overflows u128")
            }
            Self::ChronologyMeasurementMismatch => {
                formatter.write_str("certificate chronology workspace measurement is incorrect")
            }
            Self::TransitionDigestMismatch => {
                formatter.write_str("certificate transition-evidence identifier is incorrect")
            }
        }
    }
}

impl std::error::Error for PspaceCertificateError {}

/// Exact finite domain and input index used to check one family instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PspaceInstanceConfig {
    /// Length `n` used when evaluating the declared polynomial bounds.
    pub input_bits: u64,
    /// Exact immutable Boolean input, one `0`/`1` VM `INPUT` word per bit.
    pub input: Vec<u8>,
    /// Complete deterministic temporal-state domain to enumerate.
    pub graph: ProgramGraphConfig,
}

impl Default for PspaceInstanceConfig {
    fn default() -> Self {
        Self {
            input_bits: 1,
            input: vec![0],
            graph: ProgramGraphConfig::default(),
        }
    }
}

/// Replayable summary of a completely enumerated, readout-invariant instance.
///
/// The two `*_declared` fields remain assumptions about the whole input-indexed
/// family. Every other recorded obligation was checked over the full concrete
/// domain represented by this certificate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PspaceInstanceCertificate {
    pub format_version: u16,
    pub family_name: String,
    /// Exact declaration; digest fields are never equality authority.
    pub contract: PspaceFamilyContract,
    pub input_bits: u64,
    /// Exact Boolean input used to specialize the transition map.
    pub input: Vec<u8>,
    pub temporal_cells: usize,
    pub cell_bits: u8,
    pub state_count: usize,
    pub chronology_respecting_bits: u128,
    pub maximum_transition_steps: u64,
    pub maximum_stack_depth: usize,
    pub maximum_dynamic_bytes: usize,
    pub maximum_call_depth: usize,
    pub maximum_temporal_depth: usize,
    pub program_counter_bits: u32,
    pub recurrent_class_count: usize,
    pub decision: u8,
    pub bounds_policy: BoundsPolicy,
    /// Evaluated declaration bounds for this exact `input_bits`.
    pub ctc_cell_bound: u128,
    pub chronology_bit_bound: u128,
    pub transition_step_bound: u128,
    /// Deterministic identifier of the exact source FAMILY contract.
    pub contract_digest: u64,
    /// Deterministic identifier of the exact validated `OUROBC` bytes.
    pub bytecode_digest: u64,
    /// Deterministic identifier of the retained successors, instruction
    /// counts, and decision-readout classifications.
    pub transition_digest: u64,
    /// Complete bounded proof object checked without invoking the VM.
    pub transition_evidence: PspaceTransitionEvidence,
}

impl PspaceInstanceCertificate {
    /// Check the complete retained proof object without invoking Z3 or the VM.
    pub fn check_structure(&self) -> Result<(), PspaceCertificateError> {
        if self.format_version != PSPACE_INSTANCE_CERTIFICATE_VERSION {
            return Err(PspaceCertificateError::UnsupportedVersion {
                found: self.format_version,
            });
        }
        if self.family_name != self.contract.name {
            return Err(PspaceCertificateError::FamilyNameMismatch);
        }
        if let Some(obligation) = self.contract.missing_obligations().into_iter().next() {
            return Err(PspaceCertificateError::MissingDeclaration { obligation });
        }
        let ctc_bound = self
            .contract
            .ctc_cells
            .evaluate(self.input_bits)
            .ok_or(PspaceCertificateError::PolynomialOverflow { field: "CTC_CELLS" })?;
        let chronology_bound = self
            .contract
            .chronology_respecting_bits
            .evaluate(self.input_bits)
            .ok_or(PspaceCertificateError::PolynomialOverflow {
                field: "CHRONOLOGY_BITS",
            })?;
        let transition_bound = self
            .contract
            .transition_steps
            .evaluate(self.input_bits)
            .ok_or(PspaceCertificateError::PolynomialOverflow {
                field: "TRANSITION_STEPS",
            })?;
        for (field, retained, actual) in [
            ("CTC_CELLS", self.ctc_cell_bound, ctc_bound),
            (
                "CHRONOLOGY_BITS",
                self.chronology_bit_bound,
                chronology_bound,
            ),
            (
                "TRANSITION_STEPS",
                self.transition_step_bound,
                transition_bound,
            ),
        ] {
            if retained != actual {
                return Err(PspaceCertificateError::EvaluatedBoundMismatch { field });
            }
        }
        if self.contract_digest != contract_digest(&self.contract) {
            return Err(PspaceCertificateError::ContractDigestMismatch);
        }
        if self.decision > 1 {
            return Err(PspaceCertificateError::InvalidDecision {
                found: self.decision,
            });
        }
        if self.input.len() as u128 != u128::from(self.input_bits) {
            return Err(PspaceCertificateError::InputLengthMismatch {
                declared: self.input_bits,
                found: self.input.len(),
            });
        }
        if self.input.len() > MAX_RECURRENT_FROZEN_INPUTS {
            return Err(PspaceCertificateError::InputTooLarge {
                found: self.input.len(),
            });
        }
        if let Some((index, &value)) = self.input.iter().enumerate().find(|(_, value)| **value > 1)
        {
            return Err(PspaceCertificateError::NonBooleanInput { index, value });
        }
        if self.temporal_cells == 0 || !(1..=64).contains(&self.cell_bits) {
            return Err(PspaceCertificateError::InvalidDomain);
        }
        if !(1..=32).contains(&self.program_counter_bits) {
            return Err(PspaceCertificateError::InvalidProgramCounterBits {
                found: self.program_counter_bits,
            });
        }
        let exponent = self
            .temporal_cells
            .checked_mul(usize::from(self.cell_bits))
            .ok_or(PspaceCertificateError::DomainTooLarge {
                exponent: usize::MAX,
            })?;
        if exponent >= usize::BITS as usize {
            return Err(PspaceCertificateError::DomainTooLarge { exponent });
        }
        let expected_states = 1usize << exponent;
        if expected_states > MAX_RECURRENT_STATES {
            return Err(PspaceCertificateError::DomainTooLarge { exponent });
        }
        if self.state_count != expected_states {
            return Err(PspaceCertificateError::StateCountMismatch {
                expected: expected_states,
                found: self.state_count,
            });
        }
        if self.transition_evidence.successors.len() != expected_states
            || self.transition_evidence.instructions_executed.len() != expected_states
            || self.transition_evidence.maximum_stack_depths.len() != expected_states
            || self.transition_evidence.maximum_dynamic_bytes.len() != expected_states
            || self.transition_evidence.maximum_call_depths.len() != expected_states
            || self.transition_evidence.maximum_temporal_depths.len() != expected_states
            || self.transition_evidence.readouts.len() != expected_states
        {
            return Err(PspaceCertificateError::EvidenceLengthMismatch);
        }
        let mut successors = Vec::with_capacity(expected_states);
        for (state, &successor) in self.transition_evidence.successors.iter().enumerate() {
            let successor_usize = successor as usize;
            if successor_usize >= expected_states {
                return Err(PspaceCertificateError::InvalidSuccessor { state, successor });
            }
            successors.push(successor_usize);
        }
        let maximum_steps = self
            .transition_evidence
            .instructions_executed
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        if maximum_steps != self.maximum_transition_steps {
            return Err(PspaceCertificateError::MaximumStepsMismatch);
        }
        for (field, expected, found) in [
            (
                "operand-stack depth",
                self.transition_evidence
                    .maximum_stack_depths
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0),
                self.maximum_stack_depth,
            ),
            (
                "dynamic-byte",
                self.transition_evidence
                    .maximum_dynamic_bytes
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0),
                self.maximum_dynamic_bytes,
            ),
            (
                "call-depth",
                self.transition_evidence
                    .maximum_call_depths
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0),
                self.maximum_call_depth,
            ),
            (
                "temporal-depth",
                self.transition_evidence
                    .maximum_temporal_depths
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0),
                self.maximum_temporal_depth,
            ),
        ] {
            if expected != found {
                return Err(PspaceCertificateError::WorkspaceMaximumMismatch {
                    field,
                    expected,
                    found,
                });
            }
        }
        let graph = DeterministicTransitionGraph::new(successors)
            .expect("successors were checked against the nonempty exact domain");
        let recurrent = graph.analyze();
        if recurrent.recurrent_classes.len() != self.recurrent_class_count {
            return Err(PspaceCertificateError::RecurrentClassCountMismatch);
        }
        for (class, states) in recurrent.recurrent_classes.iter().enumerate() {
            for &state in states {
                match self.transition_evidence.readouts[state] {
                    PspaceReadoutEvidence::Boolean(decision) if decision == self.decision => {}
                    PspaceReadoutEvidence::Boolean(_) => {
                        return Err(PspaceCertificateError::RecurrentReadoutDisagreement {
                            class,
                            state,
                        })
                    }
                    PspaceReadoutEvidence::Other => {
                        return Err(PspaceCertificateError::InvalidRecurrentReadout {
                            class,
                            state,
                        })
                    }
                }
            }
        }
        if self.temporal_cells as u128 > self.ctc_cell_bound {
            return Err(PspaceCertificateError::CellBoundExceeded);
        }
        if self.maximum_transition_steps as u128 > self.transition_step_bound {
            return Err(PspaceCertificateError::TransitionBoundExceeded);
        }
        let measured_chronology = conservative_chronology_bits_from_metrics(
            self.program_counter_bits,
            self.maximum_stack_depth,
            self.maximum_dynamic_bytes,
            self.maximum_call_depth,
            self.maximum_temporal_depth,
            self.temporal_cells,
            self.cell_bits,
            self.input_bits,
        )
        .ok_or(PspaceCertificateError::ChronologyMeasurementOverflow)?;
        if measured_chronology != self.chronology_respecting_bits {
            return Err(PspaceCertificateError::ChronologyMeasurementMismatch);
        }
        if measured_chronology > self.chronology_bit_bound {
            return Err(PspaceCertificateError::ChronologyBoundExceeded);
        }
        if self.transition_digest
            != transition_evidence_digest(
                &self.transition_evidence,
                &self.input,
                self.temporal_cells,
                self.cell_bits,
                self.bounds_policy,
            )
        {
            return Err(PspaceCertificateError::TransitionDigestMismatch);
        }
        Ok(())
    }

    /// Recompute the complete instance and require every certificate field to
    /// match. This is an independent invocation of the bounded VM enumeration,
    /// not a proof of the two explicitly external family assumptions.
    pub fn recheck_bytecode(
        &self,
        contract: &PspaceFamilyContract,
        program: &BytecodeProgram,
        config: PspaceInstanceConfig,
    ) -> Result<(), String> {
        self.check_structure()
            .map_err(|error| format!("invalid retained FAMILY evidence: {error}"))?;
        match PspaceFamilyVerifier::verify_bytecode(contract, program, config) {
            PspaceInstanceVerificationResult::Verified(replayed) if *replayed == *self => Ok(()),
            PspaceInstanceVerificationResult::Verified(_) => {
                Err("fresh FAMILY-instance enumeration produced different evidence".to_string())
            }
            other => Err(format!(
                "fresh FAMILY-instance enumeration did not verify: {}",
                other.summary()
            )),
        }
    }

    pub fn to_json(&self) -> String {
        format!(
            "{{\"schema\":\"ourochronos.pspace-instance/v1\",\"status\":\"verified\",\"format_version\":{},\"family\":\"{}\",\"contract\":{},\"input_bits\":{},\"input\":\"{}\",\"temporal_cells\":{},\"cell_bits\":{},\"state_count\":{},\"ctc_cell_bound\":{},\"chronology_bit_bound\":{},\"transition_step_bound\":{},\"chronology_respecting_bits\":{},\"maximum_transition_steps\":{},\"maximum_stack_depth\":{},\"maximum_dynamic_bytes\":{},\"maximum_call_depth\":{},\"maximum_temporal_depth\":{},\"program_counter_bits\":{},\"recurrent_class_count\":{},\"decision\":{},\"bounds_policy\":\"{}\",\"contract_digest\":\"{:016x}\",\"bytecode_digest\":\"{:016x}\",\"transition_digest\":\"{:016x}\",\"transition_evidence\":{},\"verified_obligations\":[\"concrete-polynomial-bounds\",\"total-closed-transition\",\"effects-isolated\",\"all-recurrent-class-readout\"],\"external_assumptions\":{{\"polynomial_time_uniform_declared\":{},\"ideal_deutsch_selector_declared\":{}}}}}",
            self.format_version,
            json_escape(&self.family_name),
            contract_json(&self.contract),
            self.input_bits,
            boolean_input_string(&self.input),
            self.temporal_cells,
            self.cell_bits,
            self.state_count,
            self.ctc_cell_bound,
            self.chronology_bit_bound,
            self.transition_step_bound,
            self.chronology_respecting_bits,
            self.maximum_transition_steps,
            self.maximum_stack_depth,
            self.maximum_dynamic_bytes,
            self.maximum_call_depth,
            self.maximum_temporal_depth,
            self.program_counter_bits,
            self.recurrent_class_count,
            self.decision,
            bounds_policy_name(self.bounds_policy),
            self.contract_digest,
            self.bytecode_digest,
            self.transition_digest,
            transition_evidence_json(&self.transition_evidence),
            self.contract.polynomial_time_uniform,
            self.contract.ideal_deutsch_selector,
        )
    }
}

/// Exhaustive outcome algebra for one declared family instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PspaceInstanceVerificationResult {
    Verified(Box<PspaceInstanceCertificate>),
    /// The complete bounded evidence contradicts the declaration.
    Refuted {
        reason: String,
    },
    /// Resource exhaustion or an unrepresentable bound prevented a decision.
    Unknown {
        reason: String,
    },
    /// The program, declaration, artifact, or requested domain is unsupported.
    Unsupported {
        reason: String,
    },
}

impl PspaceInstanceVerificationResult {
    pub fn is_verified(&self) -> bool {
        matches!(self, Self::Verified(_))
    }

    pub fn summary(&self) -> String {
        match self {
            Self::Verified(certificate) => format!(
                "verified decision {} across {} recurrent class(es)",
                certificate.decision, certificate.recurrent_class_count
            ),
            Self::Refuted { reason } => format!("refuted: {reason}"),
            Self::Unknown { reason } => format!("unknown: {reason}"),
            Self::Unsupported { reason } => format!("unsupported: {reason}"),
        }
    }

    pub fn to_json(&self) -> String {
        match self {
            Self::Verified(certificate) => certificate.to_json(),
            Self::Refuted { reason } => result_json("refuted", reason),
            Self::Unknown { reason } => result_json("unknown", reason),
            Self::Unsupported { reason } => result_json("unsupported", reason),
        }
    }
}

/// Verifies the decidable obligations of one finite specialization of a
/// declared Aaronson--Watrous family.
pub struct PspaceFamilyVerifier;

impl PspaceFamilyVerifier {
    /// Canonically admit source before inspecting its FAMILY declaration.
    pub fn verify(
        program: &Program,
        config: PspaceInstanceConfig,
    ) -> PspaceInstanceVerificationResult {
        let admitted = match admit_program(
            program,
            AdmissionConfig {
                memory_cells: config.graph.memory_cells,
            },
        ) {
            Ok(admitted) => admitted,
            Err(error) => {
                return PspaceInstanceVerificationResult::Unsupported {
                    reason: format!("source admission failed: {error}"),
                }
            }
        };
        let Some(declaration) = &program.family_declaration else {
            return PspaceInstanceVerificationResult::Unsupported {
                reason: "family verification requires a source FAMILY declaration".to_string(),
            };
        };
        let contract = PspaceFamilyContract::from(declaration);
        Self::verify_bytecode(&contract, admitted.program(), config)
    }

    /// Verify an already admitted and linked executable against its retained
    /// source contract.
    pub fn verify_bytecode(
        contract: &PspaceFamilyContract,
        program: &BytecodeProgram,
        config: PspaceInstanceConfig,
    ) -> PspaceInstanceVerificationResult {
        let frozen_input = match frozen_input_words(&config) {
            Ok(input) => input,
            Err(result) => return result,
        };
        let missing = contract.missing_obligations();
        if !missing.is_empty() {
            return PspaceInstanceVerificationResult::Unknown {
                reason: format!(
                    "FAMILY declaration is missing required assumption(s): {}",
                    missing.join(", ")
                ),
            };
        }

        let ctc_bound = match contract.ctc_cells.evaluate(config.input_bits) {
            Some(bound) => bound,
            None => return polynomial_overflow("CTC_CELLS", config.input_bits),
        };
        let chronology_bound = match contract
            .chronology_respecting_bits
            .evaluate(config.input_bits)
        {
            Some(bound) => bound,
            None => return polynomial_overflow("CHRONOLOGY_BITS", config.input_bits),
        };
        let transition_bound = match contract.transition_steps.evaluate(config.input_bits) {
            Some(bound) => bound,
            None => return polynomial_overflow("TRANSITION_STEPS", config.input_bits),
        };

        if config.graph.memory_cells as u128 > ctc_bound {
            return PspaceInstanceVerificationResult::Refuted {
                reason: format!(
                    "concrete instance uses {} temporal cells but CTC_CELLS({}) = {ctc_bound}",
                    config.graph.memory_cells, config.input_bits
                ),
            };
        }
        let analysis = match BytecodeTransitionAnalyzer::analyze_with_frozen_input(
            program,
            config.graph,
            &frozen_input,
        ) {
            Ok(analysis) => analysis,
            Err(error) => return graph_error_result(error),
        };
        let maximum_transition_steps = analysis.maximum_transition_instructions();
        if maximum_transition_steps as u128 > transition_bound {
            return PspaceInstanceVerificationResult::Refuted {
                reason: format!(
                    "a concrete transition executes {maximum_transition_steps} steps but TRANSITION_STEPS({}) = {transition_bound}",
                    config.input_bits
                ),
            };
        }

        let chronology_respecting_bits = match conservative_chronology_bits(
            program,
            &analysis,
            config.input_bits,
            config.graph.cell_bits,
        ) {
            Some(bits) => bits,
            None => {
                return PspaceInstanceVerificationResult::Unknown {
                    reason: "concrete chronology-respecting workspace bit count overflowed u128"
                        .to_string(),
                }
            }
        };
        if chronology_respecting_bits > chronology_bound {
            return PspaceInstanceVerificationResult::Refuted {
                reason: format!(
                    "concrete input/decision/control/workspace requires {chronology_respecting_bits} chronology-respecting bits but CHRONOLOGY_BITS({}) = {chronology_bound}",
                    config.input_bits
                ),
            };
        }

        let decision = match unanimous_boolean_readout(&analysis) {
            Ok(decision) => decision,
            Err(reason) => return PspaceInstanceVerificationResult::Refuted { reason },
        };
        let bytecode = match program.to_bytes() {
            Ok(bytecode) => bytecode,
            Err(error) => {
                return PspaceInstanceVerificationResult::Unsupported {
                    reason: format!("cannot encode validated bytecode evidence: {error}"),
                }
            }
        };
        let transition_evidence = match transition_evidence(&analysis) {
            Ok(evidence) => evidence,
            Err(reason) => return PspaceInstanceVerificationResult::Unsupported { reason },
        };
        let transition_digest = transition_evidence_digest(
            &transition_evidence,
            &config.input,
            analysis.memory_cells,
            analysis.cell_bits,
            config.graph.bounds_policy,
        );
        let certificate = PspaceInstanceCertificate {
            format_version: PSPACE_INSTANCE_CERTIFICATE_VERSION,
            family_name: contract.name.clone(),
            contract: contract.clone(),
            input_bits: config.input_bits,
            input: config.input.clone(),
            temporal_cells: config.graph.memory_cells,
            cell_bits: config.graph.cell_bits,
            state_count: analysis.graph.state_count(),
            chronology_respecting_bits,
            maximum_transition_steps,
            maximum_stack_depth: analysis.maximum_stack_depth,
            maximum_dynamic_bytes: analysis.maximum_dynamic_bytes,
            maximum_call_depth: analysis.maximum_call_depth,
            maximum_temporal_depth: analysis.maximum_temporal_depth,
            program_counter_bits: program_counter_bits(program),
            recurrent_class_count: analysis.recurrent.recurrent_classes.len(),
            decision,
            bounds_policy: config.graph.bounds_policy,
            ctc_cell_bound: ctc_bound,
            chronology_bit_bound: chronology_bound,
            transition_step_bound: transition_bound,
            contract_digest: contract_digest(contract),
            bytecode_digest: fnv1a64(&bytecode),
            transition_digest,
            transition_evidence,
        };
        if let Err(error) = certificate.check_structure() {
            return PspaceInstanceVerificationResult::Unsupported {
                reason: format!("internal FAMILY certificate construction failed: {error}"),
            };
        }
        PspaceInstanceVerificationResult::Verified(Box::new(certificate))
    }
}

fn polynomial_overflow(field: &str, input_bits: u64) -> PspaceInstanceVerificationResult {
    PspaceInstanceVerificationResult::Unknown {
        reason: format!("{field}({input_bits}) exceeds the verifier's exact u128 polynomial range"),
    }
}

fn frozen_input_words(
    config: &PspaceInstanceConfig,
) -> Result<Vec<u64>, PspaceInstanceVerificationResult> {
    if config.input.len() as u128 != u128::from(config.input_bits) {
        return Err(PspaceInstanceVerificationResult::Unsupported {
            reason: format!(
                "concrete Boolean input length {} differs from --verify-family input length {}",
                config.input.len(),
                config.input_bits
            ),
        });
    }
    if config.input.len() > MAX_RECURRENT_FROZEN_INPUTS {
        return Err(PspaceInstanceVerificationResult::Unknown {
            reason: format!(
                "concrete Boolean input length {} exceeds hard ceiling {MAX_RECURRENT_FROZEN_INPUTS}",
                config.input.len()
            ),
        });
    }
    if let Some((index, value)) = config
        .input
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| *value > 1)
    {
        return Err(PspaceInstanceVerificationResult::Unsupported {
            reason: format!("concrete input bit {index} has non-Boolean value {value}"),
        });
    }
    let mut words = Vec::new();
    words.try_reserve_exact(config.input.len()).map_err(|_| {
        PspaceInstanceVerificationResult::Unknown {
            reason: format!(
                "cannot reserve {} immutable Boolean input words",
                config.input.len()
            ),
        }
    })?;
    words.extend(config.input.iter().map(|&bit| u64::from(bit)));
    Ok(words)
}

fn graph_error_result(error: TransitionGraphError) -> PspaceInstanceVerificationResult {
    match error {
        TransitionGraphError::ResourceLimit(_) | TransitionGraphError::DomainTooLarge { .. } => {
            PspaceInstanceVerificationResult::Unknown {
                reason: error.to_string(),
            }
        }
        TransitionGraphError::UndefinedTransition { .. }
        | TransitionGraphError::DomainNotClosed { .. } => {
            PspaceInstanceVerificationResult::Refuted {
                reason: format!("declared total closed transition is false: {error}"),
            }
        }
        TransitionGraphError::InvalidGraph(_)
        | TransitionGraphError::InvalidDomain(_)
        | TransitionGraphError::Unsupported(_) => PspaceInstanceVerificationResult::Unsupported {
            reason: error.to_string(),
        },
    }
}

fn unanimous_boolean_readout(analysis: &ProgramGraphAnalysis) -> Result<u8, String> {
    let mut expected = None;
    for (class_index, class) in analysis.recurrent.recurrent_classes.iter().enumerate() {
        for &state in class {
            let output = &analysis.outputs[state];
            let decision = match output.as_slice() {
                [OutputItem::Val(value)] if value.val <= 1 => value.val as u8,
                [OutputItem::Val(value)] => {
                    return Err(format!(
                        "recurrent class {class_index}, state {state} emits non-Boolean decision {}",
                        value.val
                    ))
                }
                [OutputItem::Char(value)] => {
                    return Err(format!(
                        "recurrent class {class_index}, state {state} emits character decision {value}; one numeric Boolean is required"
                    ))
                }
                _ => {
                    return Err(format!(
                        "recurrent class {class_index}, state {state} emits {} items; exactly one numeric Boolean is required",
                        output.len()
                    ))
                }
            };
            match expected {
                Some(prior) if prior != decision => {
                    return Err(format!(
                        "readout invariance is false: recurrent class {class_index}, state {state} decides {decision}, previously observed {prior}"
                    ))
                }
                None => expected = Some(decision),
                _ => {}
            }
        }
    }
    expected.ok_or_else(|| "complete finite transition graph has no recurrent class".to_string())
}

/// Conservative numeric workspace for the specialized transition evaluator.
/// Source bytes and Rust allocation metadata are not chronology registers;
/// temporal input/output words, operand/dynamic state, and explicit control
/// frames are. Provenance is diagnostic metadata and cannot affect dispatch.
fn conservative_chronology_bits(
    program: &BytecodeProgram,
    analysis: &ProgramGraphAnalysis,
    input_bits: u64,
    cell_bits: u8,
) -> Option<u128> {
    conservative_chronology_bits_from_metrics(
        program_counter_bits(program),
        analysis.maximum_stack_depth,
        analysis.maximum_dynamic_bytes,
        analysis.maximum_call_depth,
        analysis.maximum_temporal_depth,
        analysis.memory_cells,
        cell_bits,
        input_bits,
    )
}

fn program_counter_bits(program: &BytecodeProgram) -> u32 {
    if program.instructions.len() <= 1 {
        1
    } else {
        usize::BITS - (program.instructions.len() - 1).leading_zeros()
    }
}

#[allow(clippy::too_many_arguments)]
fn conservative_chronology_bits_from_metrics(
    program_counter_bits: u32,
    maximum_stack_depth: usize,
    maximum_dynamic_bytes: usize,
    maximum_call_depth: usize,
    maximum_temporal_depth: usize,
    temporal_cells: usize,
    cell_bits: u8,
    input_bits: u64,
) -> Option<u128> {
    let pc_bits = u128::from(program_counter_bits);
    let stack_bits = (maximum_stack_depth as u128).checked_mul(64)?;
    let dynamic_bits = (maximum_dynamic_bytes as u128).checked_mul(8)?;
    // Caller cursor (pc/end), unit start, and a word-sized completion payload.
    let call_frame_bits = pc_bits.checked_mul(3)?.checked_add(64)?;
    let call_bits = (maximum_call_depth as u128).checked_mul(call_frame_bits)?;
    // Region metadata plus a conservative full-domain numeric rollback image.
    let temporal_snapshot_bits = (temporal_cells as u128)
        .checked_mul(u128::from(cell_bits))?
        .checked_add(3 * 64)?
        .checked_add(pc_bits)?;
    let temporal_bits = (maximum_temporal_depth as u128).checked_mul(temporal_snapshot_bits)?;

    u128::from(input_bits)
        .checked_add(PSPACE_INSTANCE_DECISION_BITS)?
        .checked_add(pc_bits)?
        .checked_add(stack_bits)?
        .checked_add(dynamic_bits)?
        .checked_add(call_bits)?
        .checked_add(temporal_bits)
}

fn transition_evidence(
    analysis: &ProgramGraphAnalysis,
) -> Result<PspaceTransitionEvidence, String> {
    let successors = analysis
        .graph
        .successors()
        .iter()
        .map(|&successor| {
            u32::try_from(successor)
                .map_err(|_| format!("transition successor {successor} does not fit u32 evidence"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let readouts = analysis
        .outputs
        .iter()
        .map(|output| match output.as_slice() {
            [OutputItem::Val(value)] if value.val <= 1 => {
                PspaceReadoutEvidence::Boolean(value.val as u8)
            }
            _ => PspaceReadoutEvidence::Other,
        })
        .collect();
    Ok(PspaceTransitionEvidence {
        successors,
        instructions_executed: analysis.instructions_executed.clone(),
        maximum_stack_depths: analysis.stack_depths.clone(),
        maximum_dynamic_bytes: analysis.dynamic_bytes.clone(),
        maximum_call_depths: analysis.call_depths.clone(),
        maximum_temporal_depths: analysis.temporal_depths.clone(),
        readouts,
    })
}

fn transition_evidence_digest(
    evidence: &PspaceTransitionEvidence,
    input: &[u8],
    memory_cells: usize,
    cell_bits: u8,
    bounds_policy: BoundsPolicy,
) -> u64 {
    let mut hash = StableFnv64::new();
    hash.add_u64(input.len() as u64);
    hash.add(input);
    hash.add_u64(memory_cells as u64);
    hash.add_u8(cell_bits);
    hash.add_u8(match bounds_policy {
        BoundsPolicy::Wrap => 0,
        BoundsPolicy::Error => 1,
        BoundsPolicy::Clamp => 2,
    });
    hash.add_u64(evidence.successors.len() as u64);
    for (
        (
            ((((&successor, &instructions), &stack_depth), &dynamic_bytes), &call_depth),
            &temporal_depth,
        ),
        readout,
    ) in evidence
        .successors
        .iter()
        .zip(&evidence.instructions_executed)
        .zip(&evidence.maximum_stack_depths)
        .zip(&evidence.maximum_dynamic_bytes)
        .zip(&evidence.maximum_call_depths)
        .zip(&evidence.maximum_temporal_depths)
        .zip(&evidence.readouts)
    {
        hash.add_u64(u64::from(successor));
        hash.add_u64(instructions);
        hash.add_u64(stack_depth as u64);
        hash.add_u64(dynamic_bytes as u64);
        hash.add_u64(call_depth as u64);
        hash.add_u64(temporal_depth as u64);
        match readout {
            PspaceReadoutEvidence::Boolean(decision) => {
                hash.add_u8(0);
                hash.add_u8(*decision);
            }
            PspaceReadoutEvidence::Other => hash.add_u8(1),
        }
    }
    hash.finish()
}

fn contract_digest(contract: &PspaceFamilyContract) -> u64 {
    let mut hash = StableFnv64::new();
    hash.add_u64(contract.name.len() as u64);
    hash.add(contract.name.as_bytes());
    for polynomial in [
        contract.ctc_cells,
        contract.chronology_respecting_bits,
        contract.transition_steps,
    ] {
        hash.add_u64(polynomial.coefficient);
        hash.add_u64(u64::from(polynomial.degree));
        hash.add_u64(polynomial.additive);
    }
    for declared in [
        contract.polynomial_time_uniform,
        contract.total_transition,
        contract.all_fixed_points_agree,
        contract.ideal_deutsch_selector,
        contract.effects_frozen_or_modeled,
    ] {
        hash.add_u8(u8::from(declared));
    }
    hash.finish()
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = StableFnv64::new();
    hash.add(bytes);
    hash.finish()
}

struct StableFnv64(u64);

impl StableFnv64 {
    fn new() -> Self {
        Self(0xcbf29ce484222325)
    }

    fn add(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }

    fn add_u64(&mut self, value: u64) {
        self.add(&value.to_le_bytes());
    }

    fn add_u8(&mut self, value: u8) {
        self.add(&[value]);
    }

    fn finish(self) -> u64 {
        self.0
    }
}

fn bounds_policy_name(policy: BoundsPolicy) -> &'static str {
    match policy {
        BoundsPolicy::Wrap => "wrap",
        BoundsPolicy::Error => "error",
        BoundsPolicy::Clamp => "clamp",
    }
}

fn contract_json(contract: &PspaceFamilyContract) -> String {
    let polynomial = |bound: crate::complexity::PolynomialBound| {
        format!(
            "{{\"coefficient\":{},\"degree\":{},\"additive\":{}}}",
            bound.coefficient, bound.degree, bound.additive
        )
    };
    format!(
        "{{\"name\":\"{}\",\"ctc_cells\":{},\"chronology_respecting_bits\":{},\"transition_steps\":{},\"polynomial_time_uniform\":{},\"total_transition\":{},\"all_fixed_points_agree\":{},\"ideal_deutsch_selector\":{},\"effects_frozen_or_modeled\":{}}}",
        json_escape(&contract.name),
        polynomial(contract.ctc_cells),
        polynomial(contract.chronology_respecting_bits),
        polynomial(contract.transition_steps),
        contract.polynomial_time_uniform,
        contract.total_transition,
        contract.all_fixed_points_agree,
        contract.ideal_deutsch_selector,
        contract.effects_frozen_or_modeled,
    )
}

fn transition_evidence_json(evidence: &PspaceTransitionEvidence) -> String {
    let mut json = String::from("{\"successors\":[");
    for (index, successor) in evidence.successors.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(json, "{successor}").expect("writing to String cannot fail");
    }
    json.push_str("],\"instructions_executed\":[");
    for (index, instructions) in evidence.instructions_executed.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(json, "{instructions}").expect("writing to String cannot fail");
    }
    for (name, values) in [
        ("maximum_stack_depths", &evidence.maximum_stack_depths),
        ("maximum_dynamic_bytes", &evidence.maximum_dynamic_bytes),
        ("maximum_call_depths", &evidence.maximum_call_depths),
        ("maximum_temporal_depths", &evidence.maximum_temporal_depths),
    ] {
        write!(json, "],\"{name}\":[").expect("writing to String cannot fail");
        for (index, value) in values.iter().enumerate() {
            if index != 0 {
                json.push(',');
            }
            write!(json, "{value}").expect("writing to String cannot fail");
        }
    }
    json.push_str("],\"readouts\":[");
    for (index, readout) in evidence.readouts.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        match readout {
            PspaceReadoutEvidence::Boolean(decision) => {
                write!(json, "{decision}").expect("writing to String cannot fail");
            }
            PspaceReadoutEvidence::Other => json.push_str("null"),
        }
    }
    json.push_str("]}");
    json
}

fn result_json(status: &str, reason: &str) -> String {
    format!(
        "{{\"schema\":\"ourochronos.pspace-instance/v1\",\"status\":\"{}\",\"reason\":\"{}\"}}",
        status,
        json_escape(reason)
    )
}

fn boolean_input_string(input: &[u8]) -> String {
    input.iter().map(|bit| char::from(b'0' + *bit)).collect()
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn family(body: &str, transition_bound: u64, flags: &str) -> Program {
        parse(&format!(
            "FAMILY checked {{\n\
             CTC_CELLS POLY 1 1 0;\n\
             CHRONOLOGY_BITS POLY 1000 0 0;\n\
             TRANSITION_STEPS POLY {transition_bound} 0 0;\n\
             {flags}\n\
             }}\n{body}"
        ))
        .unwrap()
    }

    fn config() -> PspaceInstanceConfig {
        PspaceInstanceConfig {
            input_bits: 1,
            input: vec![0],
            graph: ProgramGraphConfig {
                memory_cells: 1,
                cell_bits: 1,
                max_states: 2,
                max_instructions: 100,
                bounds_policy: BoundsPolicy::Wrap,
            },
        }
    }

    const ALL_FLAGS: &str = "UNIFORM; TOTAL; READOUT_INVARIANT; IDEAL_DEUTSCH; EFFECTS_FROZEN;";

    #[test]
    fn verifies_and_rechecks_every_concrete_instance_obligation() {
        let program = family("0 ORACLE DUP 0 PROPHECY POP 1 OUTPUT", 20, ALL_FLAGS);
        let certificate = match PspaceFamilyVerifier::verify(&program, config()) {
            PspaceInstanceVerificationResult::Verified(certificate) => certificate,
            result => panic!("expected verified instance, got {result:?}"),
        };
        assert_eq!(certificate.state_count, 2);
        assert_eq!(certificate.recurrent_class_count, 2);
        assert_eq!(certificate.decision, 1);
        assert!(certificate.maximum_transition_steps <= 20);
        certificate.check_structure().unwrap();
        let admitted = admit_program(&program, AdmissionConfig { memory_cells: 1 }).unwrap();
        let contract = PspaceFamilyContract::from(program.family_declaration.as_ref().unwrap());
        certificate
            .recheck_bytecode(&contract, admitted.program(), config())
            .unwrap();
        let mut substituted_contract = contract.clone();
        substituted_contract.transition_steps.additive += 1;
        assert!(certificate
            .recheck_bytecode(&substituted_contract, admitted.program(), config())
            .unwrap_err()
            .contains("different evidence"));
        let json = certificate.to_json();
        assert!(json.contains("\"status\":\"verified\""));
        assert!(json.contains("\"external_assumptions\""));
        assert!(json.contains("\"transition_evidence\""));

        let mut invalid_edge = (*certificate).clone();
        invalid_edge.transition_evidence.successors[0] = 99;
        assert!(matches!(
            invalid_edge.check_structure(),
            Err(PspaceCertificateError::InvalidSuccessor { state: 0, .. })
        ));

        let mut changed_readout = (*certificate).clone();
        changed_readout.transition_evidence.readouts[1] = PspaceReadoutEvidence::Boolean(0);
        assert!(matches!(
            changed_readout.check_structure(),
            Err(PspaceCertificateError::RecurrentReadoutDisagreement { .. })
        ));

        let mut changed_workspace = (*certificate).clone();
        changed_workspace.transition_evidence.maximum_stack_depths[0] =
            certificate.maximum_stack_depth + 1;
        assert!(matches!(
            changed_workspace.check_structure(),
            Err(PspaceCertificateError::WorkspaceMaximumMismatch {
                field: "operand-stack depth",
                ..
            })
        ));

        let mut invalid_pc = (*certificate).clone();
        invalid_pc.program_counter_bits = 0;
        assert!(matches!(
            invalid_pc.check_structure(),
            Err(PspaceCertificateError::InvalidProgramCounterBits { found: 0 })
        ));
    }

    #[test]
    fn refutes_a_declared_readout_that_varies_between_recurrent_classes() {
        let program = family("0 ORACLE DUP 0 PROPHECY OUTPUT", 20, ALL_FLAGS);
        assert!(matches!(
            PspaceFamilyVerifier::verify(&program, config()),
            PspaceInstanceVerificationResult::Refuted { ref reason }
                if reason.contains("readout invariance is false")
        ));
    }

    #[test]
    fn refutes_false_polynomial_resource_and_totality_claims() {
        let too_slow = family("0 ORACLE DUP 0 PROPHECY POP 1 OUTPUT", 1, ALL_FLAGS);
        assert!(matches!(
            PspaceFamilyVerifier::verify(&too_slow, config()),
            PspaceInstanceVerificationResult::Refuted { ref reason }
                if reason.contains("TRANSITION_STEPS")
        ));

        let partial = family(
            "INPUT POP INPUT POP 0 ORACLE DUP 0 PROPHECY POP 1 OUTPUT",
            30,
            ALL_FLAGS,
        );
        assert!(matches!(
            PspaceFamilyVerifier::verify(&partial, config()),
            PspaceInstanceVerificationResult::Refuted { ref reason }
                if reason.contains("declared total closed transition is false")
        ));
    }

    #[test]
    fn exact_frozen_boolean_input_specializes_and_binds_the_instance() {
        let program = family("INPUT 0 ORACLE DUP 0 PROPHECY POP OUTPUT", 20, ALL_FLAGS);
        let zero = match PspaceFamilyVerifier::verify(&program, config()) {
            PspaceInstanceVerificationResult::Verified(certificate) => certificate,
            result => panic!("expected zero-input verification, got {result:?}"),
        };
        assert_eq!(zero.input, vec![0]);
        assert_eq!(zero.decision, 0);

        let mut one_config = config();
        one_config.input[0] = 1;
        let one = match PspaceFamilyVerifier::verify(&program, one_config) {
            PspaceInstanceVerificationResult::Verified(certificate) => certificate,
            result => panic!("expected one-input verification, got {result:?}"),
        };
        assert_eq!(one.input, vec![1]);
        assert_eq!(one.decision, 1);
        assert_ne!(zero.transition_digest, one.transition_digest);

        let mut substituted_input = (*zero).clone();
        substituted_input.input[0] = 1;
        assert!(matches!(
            substituted_input.check_structure(),
            Err(PspaceCertificateError::TransitionDigestMismatch)
        ));

        let mut bad_length = config();
        bad_length.input.clear();
        assert!(matches!(
            PspaceFamilyVerifier::verify(&program, bad_length),
            PspaceInstanceVerificationResult::Unsupported { ref reason }
                if reason.contains("input length")
        ));

        let mut non_boolean = config();
        non_boolean.input[0] = 2;
        assert!(matches!(
            PspaceFamilyVerifier::verify(&program, non_boolean),
            PspaceInstanceVerificationResult::Unsupported { ref reason }
                if reason.contains("non-Boolean")
        ));
    }

    #[test]
    fn missing_assumptions_are_unknown_and_false_cell_bounds_are_refuted() {
        let incomplete = family(
            "0 ORACLE DUP 0 PROPHECY POP 1 OUTPUT",
            20,
            "TOTAL; READOUT_INVARIANT; EFFECTS_FROZEN;",
        );
        assert!(matches!(
            PspaceFamilyVerifier::verify(&incomplete, config()),
            PspaceInstanceVerificationResult::Unknown { ref reason }
                if reason.contains("polynomial-time uniform")
        ));

        let complete = family("0 ORACLE DUP 0 PROPHECY POP 1 OUTPUT", 20, ALL_FLAGS);
        let mut false_bound = config();
        false_bound.graph.memory_cells = 2;
        assert!(matches!(
            PspaceFamilyVerifier::verify(&complete, false_bound),
            PspaceInstanceVerificationResult::Refuted { ref reason }
                if reason.contains("CTC_CELLS")
        ));

        let mut oversized = config();
        oversized.input_bits = 20;
        oversized.input = vec![0; 20];
        oversized.graph.memory_cells = 20;
        assert!(matches!(
            PspaceFamilyVerifier::verify(&complete, oversized),
            PspaceInstanceVerificationResult::Unknown { ref reason }
                if reason.contains("2^20")
        ));
    }

    #[test]
    fn source_facade_runs_canonical_admission_before_family_claims() {
        let program = parse(
            "FAMILY bad { CTC_CELLS POLY 1 1 0; CHRONOLOGY_BITS POLY 1 0 0; TRANSITION_STEPS POLY 20 0 0; UNIFORM; TOTAL; READOUT_INVARIANT; IDEAL_DEUTSCH; EFFECTS_FROZEN; }\n\
             PROCEDURE hidden PURE { 1 OUTPUT } 0 ORACLE DUP 0 PROPHECY POP 1 OUTPUT",
        )
        .unwrap();
        let result = PspaceFamilyVerifier::verify(&program, config());
        assert!(
            matches!(
                result,
                PspaceInstanceVerificationResult::Unsupported { ref reason }
                    if reason.contains("type/effect") && reason.contains("hidden")
            ),
            "{result:?}"
        );
    }
}
