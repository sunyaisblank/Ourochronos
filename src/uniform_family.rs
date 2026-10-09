//! Proof-carrying restricted uniform generation for PSPACE-family instances.
//!
//! The generator admitted here has a deliberately small proof surface: one
//! canonically admitted bytecode template is reused for every nonempty Boolean
//! input, temporal width follows a retained polynomial proved no larger than
//! `CTC_CELLS(n)`, and specialization only validates/copies the input, decodes
//! the constant artifact, and evaluates fixed-degree polynomials. Consequently generation
//! work and descriptor size are structurally linear in `n`. This proves
//! uniform generation for this restricted random-access program family; it
//! does not prove totality or readout invariance for every `n`, and it cannot
//! implement Nature's ideal Deutsch selector.

use crate::admission::{admit_program, AdmissionConfig};
use crate::ast::{OpCode, Program};
use crate::bytecode::{BytecodeProgram, Instruction};
use crate::bytecode_temporal::{analyze_bytecode_temporal, BytecodeTemporalIssueKind};
use crate::bytecode_vm::PreparedBytecode;
use crate::complexity::{PolynomialBound, PspaceFamilyContract};
use crate::core::BoundsPolicy;
use crate::family_verifier::{
    PspaceFamilyVerifier, PspaceInstanceCertificate, PspaceInstanceConfig,
    PspaceInstanceVerificationResult, PspaceReadoutEvidence,
};
use crate::temporal::transition_graph::{
    ProgramGraphConfig, MAX_RECURRENT_FROZEN_INPUTS, MAX_RECURRENT_STATES,
};
use crate::temporal::TemporalIrConfig;
use std::fmt::{self, Write as _};

pub const UNIFORM_FAMILY_CERTIFICATE_VERSION: u16 = 1;
pub const PROJECTION_FAMILY_CERTIFICATE_VERSION: u16 = 1;
pub const PROJECTION_CIRCUIT_VERSION: u16 = 1;
pub const MAX_PROJECTION_CIRCUIT_STATE_BITS: usize = 1_048_576;

/// Canonical fixed overhead charged by the restricted specialization
/// algorithm in addition to one unit per input bit and template byte.
const GENERATOR_WORK_OVERHEAD: u64 = 256;
/// Conservative modeled work per retained template byte (decode, validate,
/// verify, and copy passes).
const GENERATOR_TEMPLATE_BYTE_WORK: u64 = 8;
/// Canonical fixed envelope charge for the generated instance descriptor.
const GENERATOR_DESCRIPTOR_OVERHEAD: u64 = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UniformFamilyError {
    MissingFamilyDeclaration,
    MissingUniformDeclaration,
    UnsupportedVersion {
        found: u16,
    },
    UnsupportedProjectionVersion {
        found: u16,
    },
    FamilyNameMismatch,
    InvalidCellBits {
        found: u8,
    },
    PolynomialOverflow {
        field: &'static str,
        input_bits: u64,
    },
    ZeroTemporalWidth,
    WidthRuleExceedsContract,
    WidthUnrepresentable {
        width: u128,
    },
    TemporalRegionTooWide {
        required: usize,
        minimum: usize,
    },
    NonUniformTemplate {
        reason: String,
    },
    MissingSemanticDeclaration {
        obligation: &'static str,
    },
    UnsupportedFamilyTheorem {
        reason: String,
    },
    InvalidInputLength {
        found: usize,
    },
    NonBooleanInput {
        index: usize,
        value: u8,
    },
    InvalidStateLimit {
        found: usize,
    },
    InvalidInstructionLimit,
    Admission(String),
    Bytecode(String),
    CertificateMismatch {
        field: &'static str,
    },
}

impl fmt::Display for UniformFamilyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingFamilyDeclaration => {
                formatter.write_str("uniform generation requires a source FAMILY declaration")
            }
            Self::MissingUniformDeclaration => {
                formatter.write_str("the FAMILY declaration does not request UNIFORM generation")
            }
            Self::UnsupportedVersion { found } => write!(
                formatter,
                "unsupported uniform-family certificate version {found}; expected {UNIFORM_FAMILY_CERTIFICATE_VERSION}"
            ),
            Self::UnsupportedProjectionVersion { found } => write!(
                formatter,
                "unsupported projection-family certificate version {found}; expected {PROJECTION_FAMILY_CERTIFICATE_VERSION}"
            ),
            Self::FamilyNameMismatch => {
                formatter.write_str("uniform-family name differs from its retained contract")
            }
            Self::InvalidCellBits { found } => {
                write!(formatter, "uniform-family cell width {found} is outside 1..=64")
            }
            Self::PolynomialOverflow { field, input_bits } => write!(
                formatter,
                "uniform-family {field} polynomial overflows u128 at input length {input_bits}"
            ),
            Self::ZeroTemporalWidth => formatter
                .write_str("uniform-family CTC_CELLS(n) must stay positive for nonempty inputs"),
            Self::WidthRuleExceedsContract => formatter.write_str(
                "uniform-family temporal-width polynomial is not proved below CTC_CELLS(n) for every nonempty input",
            ),
            Self::WidthUnrepresentable { width } => write!(
                formatter,
                "uniform-family temporal width {width} is not representable on this host"
            ),
            Self::TemporalRegionTooWide { required, minimum } => write!(
                formatter,
                "uniform-family bytecode requires {required} temporal cells but CTC_CELLS(1) is {minimum}"
            ),
            Self::NonUniformTemplate { reason } => write!(
                formatter,
                "uniform-family template is not a complete acyclic finite transition: {reason}"
            ),
            Self::MissingSemanticDeclaration { obligation } => write!(
                formatter,
                "uniform-family theorem requires declared {obligation}"
            ),
            Self::UnsupportedFamilyTheorem { reason } => {
                write!(formatter, "uniform-family theorem is unsupported: {reason}")
            }
            Self::InvalidInputLength { found } => write!(
                formatter,
                "uniform-family input length {found} must be in 1..={MAX_RECURRENT_FROZEN_INPUTS}"
            ),
            Self::NonBooleanInput { index, value } => write!(
                formatter,
                "uniform-family input bit {index} has non-Boolean value {value}"
            ),
            Self::InvalidStateLimit { found } => write!(
                formatter,
                "uniform-family state limit {found} must be in 1..={MAX_RECURRENT_STATES}"
            ),
            Self::InvalidInstructionLimit => {
                formatter.write_str("uniform-family instruction limit must be positive")
            }
            Self::Admission(reason) => write!(formatter, "uniform-family admission failed: {reason}"),
            Self::Bytecode(reason) => {
                write!(formatter, "uniform-family bytecode is invalid: {reason}")
            }
            Self::CertificateMismatch { field } => {
                write!(formatter, "uniform-family certificate has incorrect {field}")
            }
        }
    }
}

impl std::error::Error for UniformFamilyError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectionCircuitError {
    Family(UniformFamilyError),
    InvalidInputBits {
        found: u64,
    },
    SizeOverflow,
    TooLarge {
        state_bits: usize,
    },
    StructureMismatch {
        field: &'static str,
    },
    InputLengthMismatch {
        expected: usize,
        found: usize,
    },
    StateLengthMismatch {
        expected: usize,
        found: usize,
    },
    NonBooleanInput {
        index: usize,
        value: u8,
    },
    NonBooleanState {
        index: usize,
        value: u8,
    },
    FiniteCertificate(String),
    FiniteMismatch {
        field: &'static str,
        state: Option<usize>,
    },
}

impl fmt::Display for ProjectionCircuitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Family(error) => write!(formatter, "projection circuit family proof failed: {error}"),
            Self::InvalidInputBits { found } => write!(
                formatter,
                "projection circuit input length {found} must be in 1..={MAX_RECURRENT_FROZEN_INPUTS}"
            ),
            Self::SizeOverflow => formatter.write_str("projection circuit size calculation overflowed"),
            Self::TooLarge { state_bits } => write!(
                formatter,
                "projection circuit state width {state_bits} exceeds hard ceiling {MAX_PROJECTION_CIRCUIT_STATE_BITS} bits"
            ),
            Self::StructureMismatch { field } => {
                write!(formatter, "projection circuit has incorrect {field}")
            }
            Self::InputLengthMismatch { expected, found } => write!(
                formatter,
                "projection circuit input has {found} bits; expected {expected}"
            ),
            Self::StateLengthMismatch { expected, found } => write!(
                formatter,
                "projection circuit temporal state has {found} bits; expected {expected}"
            ),
            Self::NonBooleanInput { index, value } => write!(
                formatter,
                "projection circuit input bit {index} has non-Boolean value {value}"
            ),
            Self::NonBooleanState { index, value } => write!(
                formatter,
                "projection circuit state bit {index} has non-Boolean value {value}"
            ),
            Self::FiniteCertificate(reason) => {
                write!(formatter, "finite certificate replay failed: {reason}")
            }
            Self::FiniteMismatch { field, state } => match state {
                Some(state) => write!(
                    formatter,
                    "projection circuit disagrees with finite {field} at state {state}"
                ),
                None => write!(
                    formatter,
                    "projection circuit disagrees with finite {field}"
                ),
            },
        }
    }
}

impl std::error::Error for ProjectionCircuitError {}

impl From<UniformFamilyError> for ProjectionCircuitError {
    fn from(error: UniformFamilyError) -> Self {
        Self::Family(error)
    }
}

/// Proof object for a constant-template, polynomial-width uniform generator.
///
/// Exact retained bytecode and contract are authoritative. Digests are stable
/// identifiers only. The two generated bounds are canonical, not caller-
/// supplied assertions, and `check_structure` recomputes them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UniformFamilyCertificate {
    pub format_version: u16,
    pub family_name: String,
    pub contract: PspaceFamilyContract,
    /// Exact generated width, mechanically proved below `CTC_CELLS(n)` for
    /// every nonempty input.
    pub temporal_width: PolynomialBound,
    pub cell_bits: u8,
    pub bounds_policy: BoundsPolicy,
    pub minimum_input_bits: u64,
    pub minimum_temporal_cells: usize,
    pub reachable_instructions: usize,
    pub cfg_analysis_steps: usize,
    pub call_graph_analysis_steps: usize,
    pub bytecode: Vec<u8>,
    pub bytecode_digest: u64,
    pub generation_work_bound: PolynomialBound,
    pub generated_descriptor_bound: PolynomialBound,
}

impl UniformFamilyCertificate {
    /// Recheck the restricted generator theorem without source or VM
    /// execution. Bytecode decoding and independent bytecode verification are
    /// still mandatory because the exact template is part of the proof.
    pub fn check_structure(&self) -> Result<(), UniformFamilyError> {
        if self.format_version != UNIFORM_FAMILY_CERTIFICATE_VERSION {
            return Err(UniformFamilyError::UnsupportedVersion {
                found: self.format_version,
            });
        }
        if self.family_name != self.contract.name {
            return Err(UniformFamilyError::FamilyNameMismatch);
        }
        if !self.contract.polynomial_time_uniform {
            return Err(UniformFamilyError::MissingUniformDeclaration);
        }
        if !(1..=64).contains(&self.cell_bits) {
            return Err(UniformFamilyError::InvalidCellBits {
                found: self.cell_bits,
            });
        }
        if self.minimum_input_bits != 1 {
            return Err(UniformFamilyError::CertificateMismatch {
                field: "minimum input length",
            });
        }
        if !polynomial_bounded_for_nonempty(self.temporal_width, self.contract.ctc_cells) {
            return Err(UniformFamilyError::WidthRuleExceedsContract);
        }
        let minimum_width = evaluate_polynomial(
            self.temporal_width,
            "temporal-width",
            self.minimum_input_bits,
        )?;
        if minimum_width == 0 {
            return Err(UniformFamilyError::ZeroTemporalWidth);
        }
        let minimum_width = usize::try_from(minimum_width).map_err(|_| {
            UniformFamilyError::WidthUnrepresentable {
                width: minimum_width,
            }
        })?;
        if minimum_width != self.minimum_temporal_cells {
            return Err(UniformFamilyError::CertificateMismatch {
                field: "minimum temporal width",
            });
        }
        let program = BytecodeProgram::from_bytes(&self.bytecode)
            .map_err(|error| UniformFamilyError::Bytecode(error.to_string()))?;
        let required_cells = required_temporal_cells(&program)?;
        if required_cells > self.minimum_temporal_cells {
            return Err(UniformFamilyError::TemporalRegionTooWide {
                required: required_cells,
                minimum: self.minimum_temporal_cells,
            });
        }
        let control =
            analyze_uniform_control(&program, self.minimum_temporal_cells, self.bounds_policy)?;
        if control
            != (
                self.reachable_instructions,
                self.cfg_analysis_steps,
                self.call_graph_analysis_steps,
            )
        {
            return Err(UniformFamilyError::CertificateMismatch {
                field: "acyclic control-flow proof",
            });
        }
        PreparedBytecode::new(program)
            .map_err(|error| UniformFamilyError::Bytecode(error.to_string()))?;
        if self.bytecode_digest != fnv1a64(&self.bytecode) {
            return Err(UniformFamilyError::CertificateMismatch {
                field: "bytecode identifier",
            });
        }
        let (work, descriptor) = canonical_generation_bounds(
            self.bytecode.len(),
            self.cfg_analysis_steps,
            self.call_graph_analysis_steps,
        )?;
        if self.generation_work_bound != work {
            return Err(UniformFamilyError::CertificateMismatch {
                field: "generation work polynomial",
            });
        }
        if self.generated_descriptor_bound != descriptor {
            return Err(UniformFamilyError::CertificateMismatch {
                field: "generated descriptor polynomial",
            });
        }
        Ok(())
    }

    /// Reconstruct the proof from an expected linked template and contract,
    /// then require exact certificate equality. This binds structural checking
    /// back to the compiler/linker boundary without trusting identifiers.
    pub fn recheck_bytecode(
        &self,
        contract: &PspaceFamilyContract,
        program: &BytecodeProgram,
    ) -> Result<(), UniformFamilyError> {
        self.check_structure()?;
        let replayed = PspaceUniformFamilyGenerator::certify_bytecode(
            contract,
            program,
            self.temporal_width,
            self.cell_bits,
            self.bounds_policy,
        )?;
        if replayed != *self {
            return Err(UniformFamilyError::CertificateMismatch {
                field: "fresh compiler-bound replay",
            });
        }
        Ok(())
    }

    /// Specialize the proved generator for one exact nonempty Boolean input.
    pub fn specialize(
        &self,
        input: &[u8],
        max_states: usize,
        max_instructions: u64,
    ) -> Result<UniformFamilyInstance, UniformFamilyError> {
        self.check_structure()?;
        validate_input(input)?;
        if !(1..=MAX_RECURRENT_STATES).contains(&max_states) {
            return Err(UniformFamilyError::InvalidStateLimit { found: max_states });
        }
        if max_instructions == 0 {
            return Err(UniformFamilyError::InvalidInstructionLimit);
        }
        let input_bits = input.len() as u64;
        let width = evaluate_polynomial(self.temporal_width, "temporal-width", input_bits)?;
        if width == 0 {
            return Err(UniformFamilyError::ZeroTemporalWidth);
        }
        let temporal_cells = usize::try_from(width)
            .map_err(|_| UniformFamilyError::WidthUnrepresentable { width })?;
        let program = BytecodeProgram::from_bytes(&self.bytecode)
            .map_err(|error| UniformFamilyError::Bytecode(error.to_string()))?;
        PreparedBytecode::new(program.clone())
            .map_err(|error| UniformFamilyError::Bytecode(error.to_string()))?;
        let generation_work_ceiling = self.generation_work_bound.evaluate(input_bits).ok_or(
            UniformFamilyError::PolynomialOverflow {
                field: "generation-work",
                input_bits,
            },
        )?;
        let generated_descriptor_ceiling = self
            .generated_descriptor_bound
            .evaluate(input_bits)
            .ok_or(UniformFamilyError::PolynomialOverflow {
                field: "generated-descriptor",
                input_bits,
            })?;
        Ok(UniformFamilyInstance {
            contract: self.contract.clone(),
            program,
            config: PspaceInstanceConfig {
                input_bits,
                input: input.to_vec(),
                graph: ProgramGraphConfig {
                    memory_cells: temporal_cells,
                    cell_bits: self.cell_bits,
                    max_states,
                    max_instructions,
                    bounds_policy: self.bounds_policy,
                },
            },
            generation: UniformSpecializationEvidence {
                input_bits,
                temporal_cells,
                cell_bits: self.cell_bits,
                bytecode_digest: self.bytecode_digest,
                generation_work_ceiling,
                generated_descriptor_ceiling,
            },
        })
    }

    /// Machine-readable proof envelope. Exact bytecode is hex-encoded so the
    /// artifact retains authority rather than relying on its FNV identifier.
    pub fn to_json(&self) -> String {
        let mut json = String::with_capacity(self.bytecode.len().saturating_mul(2) + 2_048);
        self.append_json(&mut json);
        json
    }

    fn append_json(&self, json: &mut String) {
        write!(
            json,
            "{{\"schema\":\"ourochronos.uniform-family/v1\",\"format_version\":{},\"family\":\"{}\",\"contract\":{},\"temporal_width\":{},\"cell_bits\":{},\"bounds_policy\":\"{}\",\"minimum_input_bits\":{},\"minimum_temporal_cells\":{},\"reachable_instructions\":{},\"cfg_analysis_steps\":{},\"call_graph_analysis_steps\":{},\"bytecode_hex\":\"",
            self.format_version,
            json_escape(&self.family_name),
            contract_json(&self.contract),
            polynomial_json(self.temporal_width),
            self.cell_bits,
            bounds_policy_name(self.bounds_policy),
            self.minimum_input_bits,
            self.minimum_temporal_cells,
            self.reachable_instructions,
            self.cfg_analysis_steps,
            self.call_graph_analysis_steps,
        )
        .expect("writing to String cannot fail");
        for byte in &self.bytecode {
            write!(json, "{byte:02x}").expect("writing to String cannot fail");
        }
        write!(
            json,
            "\",\"bytecode_digest\":\"{:016x}\",\"generation_work_bound\":{},\"generated_descriptor_bound\":{},\"proved_obligations\":[\"constant-template-generation\",\"complete-acyclic-control\",\"polynomial-width-below-contract\",\"linear-generation-work\",\"linear-descriptor-size\"],\"not_proved_family_wide\":[\"runtime-totality\",\"readout-invariance\"],\"external_model_assumption\":\"ideal-deutsch-selector\"}}",
            self.bytecode_digest,
            polynomial_json(self.generation_work_bound),
            polynomial_json(self.generated_descriptor_bound),
        )
        .expect("writing to String cannot fail");
    }

    /// Aggregate this uniform-generation proof with one finite-instance
    /// verification outcome without conflating their scopes.
    pub fn verification_json(&self, result: &PspaceInstanceVerificationResult) -> String {
        let finite = result.to_json();
        let mut json = String::with_capacity(
            self.bytecode
                .len()
                .saturating_mul(2)
                .saturating_add(finite.len())
                .saturating_add(2_128),
        );
        json.push_str(
            "{\"schema\":\"ourochronos.uniform-family-instance/v1\",\"uniform_generation\":",
        );
        self.append_json(&mut json);
        json.push_str(",\"finite_instance\":");
        json.push_str(&finite);
        json.push('}');
        json
    }
}

/// Exact evidence produced by one invocation of the proved generator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UniformSpecializationEvidence {
    pub input_bits: u64,
    pub temporal_cells: usize,
    pub cell_bits: u8,
    pub bytecode_digest: u64,
    pub generation_work_ceiling: u128,
    pub generated_descriptor_ceiling: u128,
}

/// Generated executable/configuration pair connected directly to finite
/// complete-instance verification.
#[derive(Debug, Clone)]
pub struct UniformFamilyInstance {
    pub contract: PspaceFamilyContract,
    pub program: BytecodeProgram,
    pub config: PspaceInstanceConfig,
    pub generation: UniformSpecializationEvidence,
}

impl UniformFamilyInstance {
    pub fn verify(&self) -> PspaceInstanceVerificationResult {
        PspaceFamilyVerifier::verify_bytecode(&self.contract, &self.program, self.config.clone())
    }
}

/// Family-wide decision rule proved for the canonical projection template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionDecisionRule {
    Constant(u8),
    FirstInputBit,
}

/// One exact assignment in the recognized sparse temporal routing template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionCellSource {
    Constant(u64),
    Temporal(u64),
    And(u64, u64),
    Or(u64, u64),
    Xor(u64, u64),
    AndConstant(u64, u64),
    OrConstant(u64, u64),
    XorConstant(u64, u64),
    ShiftRight(u64, u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjectionAssignment {
    pub target: u64,
    pub source: ProjectionCellSource,
}

/// One output node in the explicit projection/routing Boolean circuit.
///
/// A node is a constant, an immutable-input/prior-state wire, or one explicit
/// Boolean gate over prior-state wires. Retaining every output node makes the
/// topology exact and independently replayable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionWire {
    Constant(bool),
    Input(usize),
    Temporal(usize),
    Not(usize),
    And(usize, usize),
    Or(usize, usize),
    Xor(usize, usize),
}

/// Explicit Boolean circuit specialized for one input length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionCircuit {
    pub format_version: u16,
    pub input_bits: usize,
    pub temporal_cells: usize,
    pub cell_bits: u8,
    pub next_temporal: Vec<ProjectionWire>,
    pub decision: ProjectionWire,
}

impl ProjectionCircuit {
    pub fn state_bits(&self) -> usize {
        self.next_temporal.len()
    }

    pub fn output_wires(&self) -> usize {
        self.next_temporal.len().saturating_add(1)
    }

    /// Regenerate and compare the entire topology from its family theorem.
    pub fn check_structure(
        &self,
        theorem: &ProjectionFamilyCertificate,
    ) -> Result<(), ProjectionCircuitError> {
        theorem.check_structure()?;
        let input_bits =
            u64::try_from(self.input_bits).map_err(|_| ProjectionCircuitError::SizeOverflow)?;
        let shape = theorem.circuit_shape(input_bits)?;
        if self.format_version != PROJECTION_CIRCUIT_VERSION {
            return Err(ProjectionCircuitError::StructureMismatch {
                field: "format version",
            });
        }
        if self.input_bits != shape.input_bits {
            return Err(ProjectionCircuitError::StructureMismatch {
                field: "input width",
            });
        }
        if self.temporal_cells != shape.temporal_cells {
            return Err(ProjectionCircuitError::StructureMismatch {
                field: "temporal-cell count",
            });
        }
        if self.cell_bits != theorem.generator.cell_bits {
            return Err(ProjectionCircuitError::StructureMismatch {
                field: "cell width",
            });
        }
        let final_assignments = theorem.final_assignment_indices(shape.temporal_cells)?;
        if self.next_temporal.len() != shape.state_bits
            || self
                .next_temporal
                .iter()
                .enumerate()
                .any(|(bit, wire)| *wire != theorem.expected_circuit_wire(bit, &final_assignments))
        {
            return Err(ProjectionCircuitError::StructureMismatch {
                field: "next-state wiring",
            });
        }
        if self.decision != shape.decision {
            return Err(ProjectionCircuitError::StructureMismatch {
                field: "decision wiring",
            });
        }
        Ok(())
    }

    /// Evaluate only after exact topology replay and Boolean-domain checks.
    pub fn evaluate(
        &self,
        theorem: &ProjectionFamilyCertificate,
        input: &[u8],
        temporal_state: &[u8],
    ) -> Result<(Vec<u8>, u8), ProjectionCircuitError> {
        self.check_structure(theorem)?;
        self.evaluate_validated(input, temporal_state)
    }

    fn evaluate_validated(
        &self,
        input: &[u8],
        temporal_state: &[u8],
    ) -> Result<(Vec<u8>, u8), ProjectionCircuitError> {
        validate_circuit_bits(input, self.input_bits, true)?;
        validate_circuit_bits(temporal_state, self.state_bits(), false)?;
        let mut next = Vec::new();
        next.try_reserve_exact(self.state_bits()).map_err(|_| {
            ProjectionCircuitError::TooLarge {
                state_bits: self.state_bits(),
            }
        })?;
        for wire in &self.next_temporal {
            next.push(evaluate_wire(*wire, input, temporal_state)?);
        }
        let decision = evaluate_wire(self.decision, input, temporal_state)?;
        Ok((next, decision))
    }

    pub fn to_json(&self) -> String {
        let mut json = String::with_capacity(self.state_bits().saturating_mul(8) + 512);
        write!(
            json,
            "{{\"schema\":\"ourochronos.projection-circuit/v1\",\"format_version\":{},\"input_bits\":{},\"temporal_cells\":{},\"cell_bits\":{},\"state_bits\":{},\"next_temporal\":[",
            self.format_version,
            self.input_bits,
            self.temporal_cells,
            self.cell_bits,
            self.state_bits(),
        )
        .expect("writing to String cannot fail");
        for (index, wire) in self.next_temporal.iter().enumerate() {
            if index != 0 {
                json.push(',');
            }
            append_wire_json(&mut json, *wire);
        }
        json.push_str("],\"decision\":");
        append_wire_json(&mut json, self.decision);
        write!(json, ",\"output_wires\":{}}}", self.output_wires())
            .expect("writing to String cannot fail");
        json
    }
}

/// Symbolic all-input theorem for a small but useful sparse-routing family.
///
/// The recognized transition executes a fixed straight-line list of in-domain
/// cell copies and constants, with last assignment winning and every unwritten
/// present cell zero. Its readout is independent of temporal state: either a
/// Boolean constant or the first frozen input bit. Therefore every recurrent
/// class has the same decision for every nonempty input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionFamilyCertificate {
    pub format_version: u16,
    pub generator: UniformFamilyCertificate,
    /// Compatibility summary: target of the first retained assignment.
    pub temporal_address: u64,
    pub assignments: Vec<ProjectionAssignment>,
    pub decision_rule: ProjectionDecisionRule,
    pub transition_steps: u64,
    pub maximum_stack_depth: usize,
    pub program_counter_bits: u32,
    pub chronology_workspace_bound: PolynomialBound,
    pub transition_work_bound: PolynomialBound,
    /// Exact polynomial number of explicit next-state and decision output
    /// nodes generated for input length `n`.
    pub circuit_output_wire_bound: PolynomialBound,
}

impl ProjectionFamilyCertificate {
    pub fn prove(generator: UniformFamilyCertificate) -> Result<Self, UniformFamilyError> {
        generator.check_structure()?;
        let program = BytecodeProgram::from_bytes(&generator.bytecode)
            .map_err(|error| UniformFamilyError::Bytecode(error.to_string()))?;
        let (assignments, decision_rule, transition_steps, maximum_stack_depth) =
            recognize_projection_template(&program)?;
        let temporal_address = assignments[0].target;
        let program_counter_bits = program_counter_bits(&program);
        let chronology_workspace_bound =
            projection_chronology_bound(program_counter_bits, maximum_stack_depth)?;
        let transition_work_bound = PolynomialBound {
            coefficient: 0,
            degree: 1,
            additive: transition_steps,
        };
        let circuit_output_wire_bound =
            projection_circuit_wire_bound(generator.temporal_width, generator.cell_bits)?;
        let certificate = Self {
            format_version: PROJECTION_FAMILY_CERTIFICATE_VERSION,
            generator,
            temporal_address,
            assignments,
            decision_rule,
            transition_steps,
            maximum_stack_depth,
            program_counter_bits,
            chronology_workspace_bound,
            transition_work_bound,
            circuit_output_wire_bound,
        };
        certificate.check_structure()?;
        Ok(certificate)
    }

    /// Check the theorem from exact bytecode shape and symbolic polynomial
    /// inequalities; no finite-state enumeration is used.
    pub fn check_structure(&self) -> Result<(), UniformFamilyError> {
        if self.format_version != PROJECTION_FAMILY_CERTIFICATE_VERSION {
            return Err(UniformFamilyError::UnsupportedProjectionVersion {
                found: self.format_version,
            });
        }
        self.generator.check_structure()?;
        for (declared, obligation) in [
            (self.generator.contract.total_transition, "TOTAL transition"),
            (
                self.generator.contract.all_fixed_points_agree,
                "READOUT_INVARIANT",
            ),
            (
                self.generator.contract.effects_frozen_or_modeled,
                "EFFECTS_FROZEN",
            ),
        ] {
            if !declared {
                return Err(UniformFamilyError::MissingSemanticDeclaration { obligation });
            }
        }
        let program = BytecodeProgram::from_bytes(&self.generator.bytecode)
            .map_err(|error| UniformFamilyError::Bytecode(error.to_string()))?;
        let (assignments, decision_rule, steps, stack_depth) =
            recognize_projection_template(&program)?;
        if assignments != self.assignments
            || assignments[0].target != self.temporal_address
            || decision_rule != self.decision_rule
            || steps != self.transition_steps
            || stack_depth != self.maximum_stack_depth
        {
            return Err(UniformFamilyError::CertificateMismatch {
                field: "projection transition theorem",
            });
        }
        let program_counter_bits = program_counter_bits(&program);
        if self.program_counter_bits != program_counter_bits {
            return Err(UniformFamilyError::CertificateMismatch {
                field: "projection program-counter width",
            });
        }
        let minimum_width = self.generator.minimum_temporal_cells as u128;
        for assignment in &self.assignments {
            if u128::from(assignment.target) >= minimum_width {
                return Err(UniformFamilyError::UnsupportedFamilyTheorem {
                    reason: format!(
                        "routing target address {} is outside minimum generated width {minimum_width}",
                        assignment.target
                    ),
                });
            }
            let sources = match assignment.source {
                ProjectionCellSource::Constant(_) => [None, None],
                ProjectionCellSource::Temporal(source) => [Some(source), None],
                ProjectionCellSource::AndConstant(source, _)
                | ProjectionCellSource::OrConstant(source, _)
                | ProjectionCellSource::XorConstant(source, _)
                | ProjectionCellSource::ShiftRight(source, _) => [Some(source), None],
                ProjectionCellSource::And(left, right)
                | ProjectionCellSource::Or(left, right)
                | ProjectionCellSource::Xor(left, right) => [Some(left), Some(right)],
            };
            for address in sources.into_iter().flatten() {
                if u128::from(address) >= minimum_width {
                    return Err(UniformFamilyError::UnsupportedFamilyTheorem {
                        reason: format!(
                            "routing source address {address} is outside minimum generated width {minimum_width}"
                        ),
                    });
                }
            }
            let constant = match assignment.source {
                ProjectionCellSource::Constant(value)
                | ProjectionCellSource::AndConstant(_, value)
                | ProjectionCellSource::OrConstant(_, value)
                | ProjectionCellSource::XorConstant(_, value) => Some(value),
                _ => None,
            };
            if let Some(value) = constant {
                let fits =
                    self.generator.cell_bits == 64 || value < (1u64 << self.generator.cell_bits);
                if !fits {
                    return Err(UniformFamilyError::UnsupportedFamilyTheorem {
                        reason: format!(
                            "routing constant {value} does not fit {}-bit temporal cells",
                            self.generator.cell_bits
                        ),
                    });
                }
            }
        }
        if matches!(self.decision_rule, ProjectionDecisionRule::Constant(value) if value > 1) {
            return Err(UniformFamilyError::UnsupportedFamilyTheorem {
                reason: "projection readout constant is not Boolean".to_string(),
            });
        }
        let chronology = projection_chronology_bound(program_counter_bits, stack_depth)?;
        if chronology != self.chronology_workspace_bound {
            return Err(UniformFamilyError::CertificateMismatch {
                field: "projection chronology polynomial",
            });
        }
        let transition = PolynomialBound {
            coefficient: 0,
            degree: 1,
            additive: steps,
        };
        if transition != self.transition_work_bound {
            return Err(UniformFamilyError::CertificateMismatch {
                field: "projection transition-work polynomial",
            });
        }
        let circuit_outputs =
            projection_circuit_wire_bound(self.generator.temporal_width, self.generator.cell_bits)?;
        if circuit_outputs != self.circuit_output_wire_bound {
            return Err(UniformFamilyError::CertificateMismatch {
                field: "projection circuit-output polynomial",
            });
        }
        if !polynomial_bounded_for_nonempty(
            chronology,
            self.generator.contract.chronology_respecting_bits,
        ) {
            return Err(UniformFamilyError::UnsupportedFamilyTheorem {
                reason: "derived chronology polynomial exceeds CHRONOLOGY_BITS".to_string(),
            });
        }
        if !polynomial_bounded_for_nonempty(transition, self.generator.contract.transition_steps) {
            return Err(UniformFamilyError::UnsupportedFamilyTheorem {
                reason: "derived transition polynomial exceeds TRANSITION_STEPS".to_string(),
            });
        }
        Ok(())
    }

    /// Materialize the exact polynomial-size Boolean wiring circuit for one
    /// nonempty input length. This is a direct circuit artifact, not a
    /// random-access program descriptor.
    pub fn generate_circuit(
        &self,
        input_bits: u64,
    ) -> Result<ProjectionCircuit, ProjectionCircuitError> {
        self.check_structure()?;
        self.circuit_for_input_bits(input_bits)
    }

    fn circuit_for_input_bits(
        &self,
        input_bits: u64,
    ) -> Result<ProjectionCircuit, ProjectionCircuitError> {
        let shape = self.circuit_shape(input_bits)?;
        let mut next_temporal = Vec::new();
        next_temporal
            .try_reserve_exact(shape.state_bits)
            .map_err(|_| ProjectionCircuitError::TooLarge {
                state_bits: shape.state_bits,
            })?;
        let final_assignments = self.final_assignment_indices(shape.temporal_cells)?;
        for bit in 0..shape.state_bits {
            next_temporal.push(self.expected_circuit_wire(bit, &final_assignments));
        }
        Ok(ProjectionCircuit {
            format_version: PROJECTION_CIRCUIT_VERSION,
            input_bits: shape.input_bits,
            temporal_cells: shape.temporal_cells,
            cell_bits: self.generator.cell_bits,
            next_temporal,
            decision: shape.decision,
        })
    }

    fn circuit_shape(
        &self,
        input_bits: u64,
    ) -> Result<ProjectionCircuitShape, ProjectionCircuitError> {
        if input_bits == 0 || input_bits > MAX_RECURRENT_FROZEN_INPUTS as u64 {
            return Err(ProjectionCircuitError::InvalidInputBits { found: input_bits });
        }
        let input_bits =
            usize::try_from(input_bits).map_err(|_| ProjectionCircuitError::SizeOverflow)?;
        let temporal_cells_u128 = self
            .generator
            .temporal_width
            .evaluate(input_bits as u64)
            .ok_or(ProjectionCircuitError::SizeOverflow)?;
        let temporal_cells = usize::try_from(temporal_cells_u128)
            .map_err(|_| ProjectionCircuitError::SizeOverflow)?;
        let state_bits = temporal_cells
            .checked_mul(usize::from(self.generator.cell_bits))
            .ok_or(ProjectionCircuitError::SizeOverflow)?;
        if state_bits > MAX_PROJECTION_CIRCUIT_STATE_BITS {
            return Err(ProjectionCircuitError::TooLarge { state_bits });
        }
        let output_wires = state_bits
            .checked_add(1)
            .ok_or(ProjectionCircuitError::SizeOverflow)?;
        let proved_outputs = self
            .circuit_output_wire_bound
            .evaluate(input_bits as u64)
            .ok_or(ProjectionCircuitError::SizeOverflow)?;
        if proved_outputs != output_wires as u128 {
            return Err(ProjectionCircuitError::StructureMismatch {
                field: "evaluated circuit-output polynomial",
            });
        }
        let decision = match self.decision_rule {
            ProjectionDecisionRule::Constant(value) => ProjectionWire::Constant(value == 1),
            ProjectionDecisionRule::FirstInputBit => ProjectionWire::Input(0),
        };
        Ok(ProjectionCircuitShape {
            input_bits,
            temporal_cells,
            state_bits,
            decision,
        })
    }

    fn final_assignment_indices(
        &self,
        temporal_cells: usize,
    ) -> Result<Vec<Option<usize>>, ProjectionCircuitError> {
        let mut final_assignments = Vec::new();
        final_assignments
            .try_reserve_exact(temporal_cells)
            .map_err(|_| ProjectionCircuitError::TooLarge {
                state_bits: temporal_cells.saturating_mul(usize::from(self.generator.cell_bits)),
            })?;
        final_assignments.resize(temporal_cells, None);
        for (index, assignment) in self.assignments.iter().enumerate() {
            final_assignments[assignment.target as usize] = Some(index);
        }
        Ok(final_assignments)
    }

    fn expected_circuit_wire(
        &self,
        bit: usize,
        final_assignments: &[Option<usize>],
    ) -> ProjectionWire {
        let cell_bits = usize::from(self.generator.cell_bits);
        let target = bit / cell_bits;
        let offset = bit % cell_bits;
        let Some(assignment) = final_assignments[target].map(|index| self.assignments[index])
        else {
            return ProjectionWire::Constant(false);
        };
        match assignment.source {
            ProjectionCellSource::Constant(value) => {
                ProjectionWire::Constant(((value >> offset) & 1) == 1)
            }
            ProjectionCellSource::Temporal(source) => {
                let source = source as usize;
                ProjectionWire::Temporal(source * cell_bits + offset)
            }
            ProjectionCellSource::And(left, right) => ProjectionWire::And(
                left as usize * cell_bits + offset,
                right as usize * cell_bits + offset,
            ),
            ProjectionCellSource::Or(left, right) => ProjectionWire::Or(
                left as usize * cell_bits + offset,
                right as usize * cell_bits + offset,
            ),
            ProjectionCellSource::Xor(left, right) => ProjectionWire::Xor(
                left as usize * cell_bits + offset,
                right as usize * cell_bits + offset,
            ),
            ProjectionCellSource::AndConstant(source, value) => {
                let source = source as usize * cell_bits + offset;
                if ((value >> offset) & 1) == 1 {
                    ProjectionWire::Temporal(source)
                } else {
                    ProjectionWire::Constant(false)
                }
            }
            ProjectionCellSource::OrConstant(source, value) => {
                let source = source as usize * cell_bits + offset;
                if ((value >> offset) & 1) == 1 {
                    ProjectionWire::Constant(true)
                } else {
                    ProjectionWire::Temporal(source)
                }
            }
            ProjectionCellSource::XorConstant(source, value) => {
                let source = source as usize * cell_bits + offset;
                if ((value >> offset) & 1) == 1 {
                    ProjectionWire::Not(source)
                } else {
                    ProjectionWire::Temporal(source)
                }
            }
            ProjectionCellSource::ShiftRight(source, count) => {
                let source_offset = offset + (count % 64) as usize;
                if source_offset < cell_bits {
                    ProjectionWire::Temporal(source as usize * cell_bits + source_offset)
                } else {
                    ProjectionWire::Constant(false)
                }
            }
        }
    }

    /// Differentially compare the regenerated circuit against every edge and
    /// the decision in an independently replayed finite VM certificate.
    pub fn cross_check_finite(
        &self,
        circuit: &ProjectionCircuit,
        certificate: &PspaceInstanceCertificate,
    ) -> Result<(), ProjectionCircuitError> {
        self.check_structure()?;
        circuit.check_structure(self)?;
        certificate
            .check_structure()
            .map_err(|error| ProjectionCircuitError::FiniteCertificate(error.to_string()))?;
        if certificate.contract != self.generator.contract
            || certificate.bytecode_digest != self.generator.bytecode_digest
        {
            return Err(ProjectionCircuitError::FiniteMismatch {
                field: "family/program binding",
                state: None,
            });
        }
        if certificate.input_bits != circuit.input_bits as u64
            || certificate.input.len() != circuit.input_bits
        {
            return Err(ProjectionCircuitError::FiniteMismatch {
                field: "input binding",
                state: None,
            });
        }
        if certificate.temporal_cells != circuit.temporal_cells
            || certificate.cell_bits != circuit.cell_bits
            || certificate.state_count != certificate.transition_evidence.successors.len()
        {
            return Err(ProjectionCircuitError::FiniteMismatch {
                field: "temporal domain",
                state: None,
            });
        }
        let expected_chronology = self
            .chronology_workspace_bound
            .evaluate(certificate.input_bits)
            .ok_or(ProjectionCircuitError::SizeOverflow)?;
        if certificate.maximum_transition_steps != self.transition_steps
            || certificate.maximum_stack_depth != self.maximum_stack_depth
            || certificate.maximum_dynamic_bytes != 0
            || certificate.maximum_call_depth != 0
            || certificate.maximum_temporal_depth != 0
            || certificate.program_counter_bits != self.program_counter_bits
            || certificate.chronology_respecting_bits != expected_chronology
        {
            return Err(ProjectionCircuitError::FiniteMismatch {
                field: "resource measurements",
                state: None,
            });
        }
        let state_bits = circuit.state_bits();
        for state in 0..certificate.state_count {
            let prior = (0..state_bits)
                .map(|bit| ((state >> bit) & 1) as u8)
                .collect::<Vec<_>>();
            let (next, decision) = circuit.evaluate_validated(&certificate.input, &prior)?;
            let mut encoded_next = 0usize;
            for (bit, value) in next.into_iter().enumerate() {
                encoded_next |= usize::from(value) << bit;
            }
            if encoded_next != certificate.transition_evidence.successors[state] as usize {
                return Err(ProjectionCircuitError::FiniteMismatch {
                    field: "successor table",
                    state: Some(state),
                });
            }
            if decision != certificate.decision {
                return Err(ProjectionCircuitError::FiniteMismatch {
                    field: "decision readout",
                    state: Some(state),
                });
            }
            if certificate.transition_evidence.readouts[state]
                != PspaceReadoutEvidence::Boolean(decision)
            {
                return Err(ProjectionCircuitError::FiniteMismatch {
                    field: "per-state readout table",
                    state: Some(state),
                });
            }
        }
        Ok(())
    }

    /// Decide any nonempty Boolean input from the proved readout rule without
    /// invoking an ideal selector or enumerating temporal states.
    pub fn decision(&self, input: &[u8]) -> Result<u8, UniformFamilyError> {
        self.check_structure()?;
        validate_input(input)?;
        Ok(match self.decision_rule {
            ProjectionDecisionRule::Constant(value) => value,
            ProjectionDecisionRule::FirstInputBit => input[0],
        })
    }

    pub fn to_json(&self) -> String {
        let mut json = String::with_capacity(
            self.generator
                .bytecode
                .len()
                .saturating_mul(2)
                .saturating_add(3_072),
        );
        self.append_json(&mut json);
        json
    }

    fn append_json(&self, json: &mut String) {
        json.push_str("{\"schema\":\"ourochronos.projection-family/v1\",\"uniform_generation\":");
        self.generator.append_json(json);
        write!(
            json,
            ",\"format_version\":{},\"temporal_address\":{},\"assignments\":{},\"decision_rule\":{},\"transition_steps\":{},\"maximum_stack_depth\":{},\"program_counter_bits\":{},\"chronology_workspace_bound\":{},\"transition_work_bound\":{},\"circuit_output_wire_bound\":{},\"proved_for_all_nonempty_inputs\":[\"total-closed-sparse-routing-transition\",\"polynomial-resource-bounds\",\"polynomial-size-explicit-boolean-circuit\",\"all-recurrent-class-readout\",\"effects-isolated\"],\"external_model_assumption\":\"ideal-deutsch-selector\"}}",
            self.format_version,
            self.temporal_address,
            projection_assignments_json(&self.assignments),
            projection_decision_json(self.decision_rule),
            self.transition_steps,
            self.maximum_stack_depth,
            self.program_counter_bits,
            polynomial_json(self.chronology_workspace_bound),
            polynomial_json(self.transition_work_bound),
            polynomial_json(self.circuit_output_wire_bound),
        )
        .expect("writing to String cannot fail");
    }

    pub fn verification_json(&self, result: &PspaceInstanceVerificationResult) -> String {
        let finite = result.to_json();
        let mut json = String::with_capacity(
            self.generator
                .bytecode
                .len()
                .saturating_mul(2)
                .saturating_add(finite.len())
                .saturating_add(3_152),
        );
        json.push_str(
            "{\"schema\":\"ourochronos.projection-family-instance/v1\",\"family_theorem\":",
        );
        self.append_json(&mut json);
        json.push_str(",\"finite_instance\":");
        json.push_str(&finite);
        json.push('}');
        json
    }

    pub fn circuit_verification_json(
        &self,
        circuit: &ProjectionCircuit,
        result: &PspaceInstanceVerificationResult,
    ) -> String {
        let finite = result.to_json();
        let circuit = circuit.to_json();
        let mut json = String::with_capacity(
            self.generator
                .bytecode
                .len()
                .saturating_mul(2)
                .saturating_add(circuit.len())
                .saturating_add(finite.len())
                .saturating_add(3_256),
        );
        json.push_str(
            "{\"schema\":\"ourochronos.projection-circuit-instance/v1\",\"family_theorem\":",
        );
        self.append_json(&mut json);
        json.push_str(",\"boolean_circuit\":");
        json.push_str(&circuit);
        json.push_str(",\"finite_instance\":");
        json.push_str(&finite);
        json.push('}');
        json
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProjectionCircuitShape {
    input_bits: usize,
    temporal_cells: usize,
    state_bits: usize,
    decision: ProjectionWire,
}

/// Canonical source-facing constructor for the restricted uniform generator.
pub struct PspaceUniformFamilyGenerator;

impl PspaceUniformFamilyGenerator {
    pub fn admit(
        source: &Program,
        temporal_width: PolynomialBound,
        cell_bits: u8,
        bounds_policy: BoundsPolicy,
    ) -> Result<UniformFamilyCertificate, UniformFamilyError> {
        let declaration = source
            .family_declaration
            .as_ref()
            .ok_or(UniformFamilyError::MissingFamilyDeclaration)?;
        let contract = PspaceFamilyContract::from(declaration);
        if !contract.polynomial_time_uniform {
            return Err(UniformFamilyError::MissingUniformDeclaration);
        }
        if !(1..=64).contains(&cell_bits) {
            return Err(UniformFamilyError::InvalidCellBits { found: cell_bits });
        }
        if !polynomial_bounded_for_nonempty(temporal_width, contract.ctc_cells) {
            return Err(UniformFamilyError::WidthRuleExceedsContract);
        }
        let minimum_width = evaluate_polynomial(temporal_width, "temporal-width", 1)?;
        if minimum_width == 0 {
            return Err(UniformFamilyError::ZeroTemporalWidth);
        }
        let minimum_temporal_cells = usize::try_from(minimum_width).map_err(|_| {
            UniformFamilyError::WidthUnrepresentable {
                width: minimum_width,
            }
        })?;
        let admitted = admit_program(
            source,
            AdmissionConfig {
                memory_cells: minimum_temporal_cells,
            },
        )
        .map_err(|error| UniformFamilyError::Admission(error.to_string()))?;
        Self::certify_bytecode(
            &contract,
            admitted.program(),
            temporal_width,
            cell_bits,
            bounds_policy,
        )
    }

    /// Certify an already canonically admitted and linked bytecode template.
    /// This is the package/compiler-facing route; exact bytecode validation
    /// and minimum-width temporal-region checks are repeated here.
    pub fn certify_bytecode(
        contract: &PspaceFamilyContract,
        program: &BytecodeProgram,
        temporal_width: PolynomialBound,
        cell_bits: u8,
        bounds_policy: BoundsPolicy,
    ) -> Result<UniformFamilyCertificate, UniformFamilyError> {
        if !contract.polynomial_time_uniform {
            return Err(UniformFamilyError::MissingUniformDeclaration);
        }
        if !(1..=64).contains(&cell_bits) {
            return Err(UniformFamilyError::InvalidCellBits { found: cell_bits });
        }
        if !polynomial_bounded_for_nonempty(temporal_width, contract.ctc_cells) {
            return Err(UniformFamilyError::WidthRuleExceedsContract);
        }
        let minimum_width = evaluate_polynomial(temporal_width, "temporal-width", 1)?;
        if minimum_width == 0 {
            return Err(UniformFamilyError::ZeroTemporalWidth);
        }
        let minimum_temporal_cells = usize::try_from(minimum_width).map_err(|_| {
            UniformFamilyError::WidthUnrepresentable {
                width: minimum_width,
            }
        })?;
        PreparedBytecode::new(program.clone())
            .map_err(|error| UniformFamilyError::Bytecode(error.to_string()))?;
        let required_cells = required_temporal_cells(program)?;
        if required_cells > minimum_temporal_cells {
            return Err(UniformFamilyError::TemporalRegionTooWide {
                required: required_cells,
                minimum: minimum_temporal_cells,
            });
        }
        let (reachable_instructions, cfg_analysis_steps, call_graph_analysis_steps) =
            analyze_uniform_control(program, minimum_temporal_cells, bounds_policy)?;
        let bytecode = program
            .to_bytes()
            .map_err(|error| UniformFamilyError::Bytecode(error.to_string()))?;
        let (generation_work_bound, generated_descriptor_bound) = canonical_generation_bounds(
            bytecode.len(),
            cfg_analysis_steps,
            call_graph_analysis_steps,
        )?;
        let certificate = UniformFamilyCertificate {
            format_version: UNIFORM_FAMILY_CERTIFICATE_VERSION,
            family_name: contract.name.clone(),
            contract: contract.clone(),
            temporal_width,
            cell_bits,
            bounds_policy,
            minimum_input_bits: 1,
            minimum_temporal_cells,
            reachable_instructions,
            cfg_analysis_steps,
            call_graph_analysis_steps,
            bytecode_digest: fnv1a64(&bytecode),
            bytecode,
            generation_work_bound,
            generated_descriptor_bound,
        };
        certificate.check_structure()?;
        Ok(certificate)
    }
}

fn recognize_projection_template(
    program: &BytecodeProgram,
) -> Result<
    (
        Vec<ProjectionAssignment>,
        ProjectionDecisionRule,
        u64,
        usize,
    ),
    UniformFamilyError,
> {
    let start = program.main.start as usize;
    let end = program.main.end as usize;
    let instructions = program.instructions.get(start..end).ok_or_else(|| {
        UniformFamilyError::UnsupportedFamilyTheorem {
            reason: "main bytecode range is unavailable".to_string(),
        }
    })?;
    let unsupported = |detail: &str| UniformFamilyError::UnsupportedFamilyTheorem {
        reason: format!(
            "main is not the canonical one-cell projection or sparse routing template: {detail}"
        ),
    };
    let (decision_rule, mut cursor, body_end, base_depth) = match instructions {
        [Instruction::Primitive(OpCode::Input), .., Instruction::Primitive(OpCode::Output), Instruction::Return] => {
            (
                ProjectionDecisionRule::FirstInputBit,
                1,
                instructions.len() - 2,
                1usize,
            )
        }
        [.., Instruction::PushWord(decision), Instruction::Primitive(OpCode::Output), Instruction::Return]
            if *decision <= 1 =>
        {
            (
                ProjectionDecisionRule::Constant(*decision as u8),
                0,
                instructions.len() - 3,
                0usize,
            )
        }
        _ => {
            return Err(unsupported(
                "readout must be a Boolean constant or retained first input",
            ))
        }
    };
    if cursor >= body_end {
        return Err(unsupported("at least one temporal assignment is required"));
    }
    let mut assignments = Vec::new();
    assignments
        .try_reserve_exact((body_end - cursor) / 3)
        .map_err(|_| unsupported("assignment evidence exceeds allocation limits"))?;
    let mut maximum_stack_depth = base_depth.max(1);
    while cursor < body_end {
        let remaining = &instructions[cursor..body_end];
        match remaining {
            [Instruction::PushWord(left), Instruction::Primitive(OpCode::Oracle), Instruction::PushWord(right), Instruction::Primitive(OpCode::Oracle), Instruction::Primitive(operation @ (OpCode::And | OpCode::Or | OpCode::Xor)), Instruction::PushWord(target), Instruction::Primitive(OpCode::Prophecy), ..] =>
            {
                let source = match operation {
                    OpCode::And => ProjectionCellSource::And(*left, *right),
                    OpCode::Or => ProjectionCellSource::Or(*left, *right),
                    OpCode::Xor => ProjectionCellSource::Xor(*left, *right),
                    _ => unreachable!("slice pattern restricts the operation"),
                };
                assignments.push(ProjectionAssignment {
                    target: *target,
                    source,
                });
                maximum_stack_depth = maximum_stack_depth.max(base_depth + 2);
                cursor += 7;
            }
            [Instruction::PushWord(source), Instruction::Primitive(OpCode::Oracle), Instruction::PushWord(value), Instruction::Primitive(operation @ (OpCode::And | OpCode::Or | OpCode::Xor)), Instruction::PushWord(target), Instruction::Primitive(OpCode::Prophecy), ..] =>
            {
                let source = match operation {
                    OpCode::And => ProjectionCellSource::AndConstant(*source, *value),
                    OpCode::Or => ProjectionCellSource::OrConstant(*source, *value),
                    OpCode::Xor => ProjectionCellSource::XorConstant(*source, *value),
                    _ => unreachable!("slice pattern restricts the operation"),
                };
                assignments.push(ProjectionAssignment {
                    target: *target,
                    source,
                });
                maximum_stack_depth = maximum_stack_depth.max(base_depth + 2);
                cursor += 6;
            }
            [Instruction::PushWord(source), Instruction::Primitive(OpCode::Oracle), Instruction::PushWord(count), Instruction::Primitive(OpCode::Shr), Instruction::PushWord(target), Instruction::Primitive(OpCode::Prophecy), ..] =>
            {
                assignments.push(ProjectionAssignment {
                    target: *target,
                    source: ProjectionCellSource::ShiftRight(*source, *count),
                });
                maximum_stack_depth = maximum_stack_depth.max(base_depth + 2);
                cursor += 6;
            }
            [Instruction::PushWord(source), Instruction::Primitive(OpCode::Oracle), Instruction::Primitive(OpCode::Dup), Instruction::PushWord(target), Instruction::Primitive(OpCode::Prophecy), Instruction::Primitive(OpCode::Pop), ..] =>
            {
                assignments.push(ProjectionAssignment {
                    target: *target,
                    source: ProjectionCellSource::Temporal(*source),
                });
                maximum_stack_depth = maximum_stack_depth.max(base_depth + 3);
                cursor += 6;
            }
            [Instruction::PushWord(source), Instruction::Primitive(OpCode::Oracle), Instruction::PushWord(target), Instruction::Primitive(OpCode::Prophecy), ..] =>
            {
                assignments.push(ProjectionAssignment {
                    target: *target,
                    source: ProjectionCellSource::Temporal(*source),
                });
                maximum_stack_depth = maximum_stack_depth.max(base_depth + 2);
                cursor += 4;
            }
            [Instruction::PushWord(value), Instruction::PushWord(target), Instruction::Primitive(OpCode::Prophecy), ..] =>
            {
                assignments.push(ProjectionAssignment {
                    target: *target,
                    source: ProjectionCellSource::Constant(*value),
                });
                maximum_stack_depth = maximum_stack_depth.max(base_depth + 2);
                cursor += 3;
            }
            _ => {
                return Err(unsupported(
                    "transition body must contain only fixed cell-copy, constant, bitwise, or right-shift assignments",
                ));
            }
        }
    }
    Ok((
        assignments,
        decision_rule,
        instructions.len() as u64,
        maximum_stack_depth,
    ))
}

fn program_counter_bits(program: &BytecodeProgram) -> u32 {
    if program.instructions.len() <= 1 {
        1
    } else {
        usize::BITS - (program.instructions.len() - 1).leading_zeros()
    }
}

fn projection_chronology_bound(
    program_counter_bits: u32,
    maximum_stack_depth: usize,
) -> Result<PolynomialBound, UniformFamilyError> {
    let stack_depth = u64::try_from(maximum_stack_depth).map_err(|_| {
        UniformFamilyError::UnsupportedFamilyTheorem {
            reason: "projection stack depth is not representable".to_string(),
        }
    })?;
    let additive = stack_depth
        .checked_mul(64)
        .and_then(|bits| bits.checked_add(u64::from(program_counter_bits)))
        .and_then(|bits| bits.checked_add(1))
        .ok_or_else(|| UniformFamilyError::UnsupportedFamilyTheorem {
            reason: "projection chronology polynomial overflows u64".to_string(),
        })?;
    Ok(PolynomialBound {
        coefficient: 1,
        degree: 1,
        additive,
    })
}

fn projection_circuit_wire_bound(
    temporal_width: PolynomialBound,
    cell_bits: u8,
) -> Result<PolynomialBound, UniformFamilyError> {
    let cell_bits = u64::from(cell_bits);
    let coefficient = temporal_width
        .coefficient
        .checked_mul(cell_bits)
        .ok_or_else(|| UniformFamilyError::UnsupportedFamilyTheorem {
            reason: "projection circuit-output polynomial coefficient overflows u64".to_string(),
        })?;
    let additive = temporal_width
        .additive
        .checked_mul(cell_bits)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| UniformFamilyError::UnsupportedFamilyTheorem {
            reason: "projection circuit-output polynomial additive term overflows u64".to_string(),
        })?;
    Ok(PolynomialBound {
        coefficient,
        degree: temporal_width.degree,
        additive,
    })
}

fn validate_circuit_bits(
    bits: &[u8],
    expected: usize,
    input: bool,
) -> Result<(), ProjectionCircuitError> {
    if bits.len() != expected {
        return Err(if input {
            ProjectionCircuitError::InputLengthMismatch {
                expected,
                found: bits.len(),
            }
        } else {
            ProjectionCircuitError::StateLengthMismatch {
                expected,
                found: bits.len(),
            }
        });
    }
    if let Some((index, value)) = bits
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| *value > 1)
    {
        return Err(if input {
            ProjectionCircuitError::NonBooleanInput { index, value }
        } else {
            ProjectionCircuitError::NonBooleanState { index, value }
        });
    }
    Ok(())
}

fn evaluate_wire(
    wire: ProjectionWire,
    input: &[u8],
    temporal_state: &[u8],
) -> Result<u8, ProjectionCircuitError> {
    match wire {
        ProjectionWire::Constant(value) => Ok(u8::from(value)),
        ProjectionWire::Input(index) => {
            input
                .get(index)
                .copied()
                .ok_or(ProjectionCircuitError::StructureMismatch {
                    field: "input-wire index",
                })
        }
        ProjectionWire::Temporal(index) => {
            temporal_state
                .get(index)
                .copied()
                .ok_or(ProjectionCircuitError::StructureMismatch {
                    field: "temporal-wire index",
                })
        }
        ProjectionWire::Not(index) => {
            Ok(evaluate_wire(ProjectionWire::Temporal(index), input, temporal_state)? ^ 1)
        }
        ProjectionWire::And(left, right) => {
            Ok(
                evaluate_wire(ProjectionWire::Temporal(left), input, temporal_state)?
                    & evaluate_wire(ProjectionWire::Temporal(right), input, temporal_state)?,
            )
        }
        ProjectionWire::Or(left, right) => {
            Ok(
                evaluate_wire(ProjectionWire::Temporal(left), input, temporal_state)?
                    | evaluate_wire(ProjectionWire::Temporal(right), input, temporal_state)?,
            )
        }
        ProjectionWire::Xor(left, right) => {
            Ok(
                evaluate_wire(ProjectionWire::Temporal(left), input, temporal_state)?
                    ^ evaluate_wire(ProjectionWire::Temporal(right), input, temporal_state)?,
            )
        }
    }
}

fn append_wire_json(json: &mut String, wire: ProjectionWire) {
    match wire {
        ProjectionWire::Constant(value) => write!(
            json,
            "{{\"kind\":\"constant\",\"value\":{}}}",
            u8::from(value)
        ),
        ProjectionWire::Input(index) => {
            write!(json, "{{\"kind\":\"input\",\"index\":{index}}}")
        }
        ProjectionWire::Temporal(index) => {
            write!(json, "{{\"kind\":\"temporal\",\"index\":{index}}}")
        }
        ProjectionWire::Not(index) => {
            write!(json, "{{\"kind\":\"not\",\"input\":{index}}}")
        }
        ProjectionWire::And(left, right) => write!(
            json,
            "{{\"kind\":\"and\",\"left\":{left},\"right\":{right}}}"
        ),
        ProjectionWire::Or(left, right) => write!(
            json,
            "{{\"kind\":\"or\",\"left\":{left},\"right\":{right}}}"
        ),
        ProjectionWire::Xor(left, right) => write!(
            json,
            "{{\"kind\":\"xor\",\"left\":{left},\"right\":{right}}}"
        ),
    }
    .expect("writing to String cannot fail");
}

fn projection_decision_json(rule: ProjectionDecisionRule) -> String {
    match rule {
        ProjectionDecisionRule::Constant(value) => {
            format!("{{\"kind\":\"constant\",\"value\":{value}}}")
        }
        ProjectionDecisionRule::FirstInputBit => "{\"kind\":\"first-input-bit\"}".to_string(),
    }
}

fn projection_assignments_json(assignments: &[ProjectionAssignment]) -> String {
    let mut json = String::with_capacity(assignments.len().saturating_mul(64).saturating_add(2));
    json.push('[');
    for (index, assignment) in assignments.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        match assignment.source {
            ProjectionCellSource::Constant(value) => write!(
                json,
                "{{\"target\":{},\"source\":{{\"kind\":\"constant\",\"value\":{value}}}}}",
                assignment.target
            ),
            ProjectionCellSource::Temporal(source) => write!(
                json,
                "{{\"target\":{},\"source\":{{\"kind\":\"temporal\",\"address\":{source}}}}}",
                assignment.target
            ),
            ProjectionCellSource::And(left, right) => write!(
                json,
                "{{\"target\":{},\"source\":{{\"kind\":\"and\",\"left\":{left},\"right\":{right}}}}}",
                assignment.target
            ),
            ProjectionCellSource::Or(left, right) => write!(
                json,
                "{{\"target\":{},\"source\":{{\"kind\":\"or\",\"left\":{left},\"right\":{right}}}}}",
                assignment.target
            ),
            ProjectionCellSource::Xor(left, right) => write!(
                json,
                "{{\"target\":{},\"source\":{{\"kind\":\"xor\",\"left\":{left},\"right\":{right}}}}}",
                assignment.target
            ),
            ProjectionCellSource::AndConstant(source, value) => write!(
                json,
                "{{\"target\":{},\"source\":{{\"kind\":\"and-constant\",\"address\":{source},\"value\":{value}}}}}",
                assignment.target
            ),
            ProjectionCellSource::OrConstant(source, value) => write!(
                json,
                "{{\"target\":{},\"source\":{{\"kind\":\"or-constant\",\"address\":{source},\"value\":{value}}}}}",
                assignment.target
            ),
            ProjectionCellSource::XorConstant(source, value) => write!(
                json,
                "{{\"target\":{},\"source\":{{\"kind\":\"xor-constant\",\"address\":{source},\"value\":{value}}}}}",
                assignment.target
            ),
            ProjectionCellSource::ShiftRight(source, count) => write!(
                json,
                "{{\"target\":{},\"source\":{{\"kind\":\"shift-right\",\"address\":{source},\"count\":{count}}}}}",
                assignment.target
            ),
        }
        .expect("writing to String cannot fail");
    }
    json.push(']');
    json
}

fn required_temporal_cells(program: &BytecodeProgram) -> Result<usize, UniformFamilyError> {
    let mut required = 0u64;
    for instruction in &program.instructions {
        if let Instruction::TemporalEnter { base, size, .. } = instruction {
            let end = base.checked_add(*size).ok_or_else(|| {
                UniformFamilyError::Bytecode(
                    "temporal region endpoint overflows the u64 address domain".to_string(),
                )
            })?;
            required = required.max(end);
        }
    }
    usize::try_from(required).map_err(|_| UniformFamilyError::WidthUnrepresentable {
        width: u128::from(required),
    })
}

fn analyze_uniform_control(
    program: &BytecodeProgram,
    memory_cells: usize,
    bounds_policy: BoundsPolicy,
) -> Result<(usize, usize, usize), UniformFamilyError> {
    let analysis = analyze_bytecode_temporal(
        program,
        TemporalIrConfig {
            memory_cells,
            loop_unroll_limit: 1,
            bounds_policy,
        },
    )
    .map_err(|error| UniformFamilyError::NonUniformTemplate {
        reason: error.to_string(),
    })?;
    if let Some(issue) = analysis.issues.iter().find(|issue| {
        !matches!(
            &issue.kind,
            BytecodeTemporalIssueKind::UnsupportedPrimitive {
                opcode: OpCode::Input,
                ..
            }
        )
    }) {
        return Err(UniformFamilyError::NonUniformTemplate {
            reason: format!("{:?} at {:?}", issue.kind, issue.site),
        });
    }
    Ok((
        analysis.reachable_instructions,
        analysis.cfg_steps,
        analysis.call_graph_steps,
    ))
}

fn validate_input(input: &[u8]) -> Result<(), UniformFamilyError> {
    if input.is_empty() || input.len() > MAX_RECURRENT_FROZEN_INPUTS {
        return Err(UniformFamilyError::InvalidInputLength { found: input.len() });
    }
    if let Some((index, value)) = input
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| *value > 1)
    {
        return Err(UniformFamilyError::NonBooleanInput { index, value });
    }
    Ok(())
}

fn evaluate_polynomial(
    polynomial: PolynomialBound,
    field: &'static str,
    input_bits: u64,
) -> Result<u128, UniformFamilyError> {
    polynomial
        .evaluate(input_bits)
        .ok_or(UniformFamilyError::PolynomialOverflow { field, input_bits })
}

/// Prove a sufficient monotone inequality for the supported nonnegative
/// single-monomial polynomials over every integer n >= 1.
fn polynomial_bounded_for_nonempty(generated: PolynomialBound, declared: PolynomialBound) -> bool {
    let Some(generated_at_one) = generated.evaluate(1) else {
        return false;
    };
    let Some(declared_at_one) = declared.evaluate(1) else {
        return false;
    };
    if generated_at_one > declared_at_one {
        return false;
    }
    if generated.coefficient == 0 {
        return true;
    }
    declared.coefficient != 0
        && generated.coefficient <= declared.coefficient
        && generated.degree <= declared.degree
}

fn canonical_generation_bounds(
    bytecode_len: usize,
    cfg_analysis_steps: usize,
    call_graph_analysis_steps: usize,
) -> Result<(PolynomialBound, PolynomialBound), UniformFamilyError> {
    let bytes =
        u64::try_from(bytecode_len).map_err(|_| UniformFamilyError::CertificateMismatch {
            field: "representable bytecode length",
        })?;
    let cfg_steps =
        u64::try_from(cfg_analysis_steps).map_err(|_| UniformFamilyError::CertificateMismatch {
            field: "representable CFG analysis work",
        })?;
    let call_steps = u64::try_from(call_graph_analysis_steps).map_err(|_| {
        UniformFamilyError::CertificateMismatch {
            field: "representable call-graph analysis work",
        }
    })?;
    let work_additive = bytes
        .checked_mul(GENERATOR_TEMPLATE_BYTE_WORK)
        .and_then(|work| work.checked_add(cfg_steps))
        .and_then(|work| work.checked_add(call_steps))
        .and_then(|work| work.checked_add(GENERATOR_WORK_OVERHEAD))
        .ok_or(UniformFamilyError::CertificateMismatch {
            field: "generation work polynomial",
        })?;
    let descriptor_additive = bytes.checked_add(GENERATOR_DESCRIPTOR_OVERHEAD).ok_or(
        UniformFamilyError::CertificateMismatch {
            field: "generated descriptor polynomial",
        },
    )?;
    Ok((
        PolynomialBound {
            coefficient: 1,
            degree: 1,
            additive: work_additive,
        },
        PolynomialBound {
            coefficient: 1,
            degree: 1,
            additive: descriptor_additive,
        },
    ))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn bounds_policy_name(policy: BoundsPolicy) -> &'static str {
    match policy {
        BoundsPolicy::Wrap => "wrap",
        BoundsPolicy::Error => "error",
        BoundsPolicy::Clamp => "clamp",
    }
}

fn polynomial_json(polynomial: PolynomialBound) -> String {
    format!(
        "{{\"coefficient\":{},\"degree\":{},\"additive\":{}}}",
        polynomial.coefficient, polynomial.degree, polynomial.additive
    )
}

fn contract_json(contract: &PspaceFamilyContract) -> String {
    format!(
        "{{\"name\":\"{}\",\"ctc_cells\":{},\"chronology_respecting_bits\":{},\"transition_steps\":{},\"polynomial_time_uniform\":{},\"total_transition\":{},\"all_fixed_points_agree\":{},\"ideal_deutsch_selector\":{},\"effects_frozen_or_modeled\":{}}}",
        json_escape(&contract.name),
        polynomial_json(contract.ctc_cells),
        polynomial_json(contract.chronology_respecting_bits),
        polynomial_json(contract.transition_steps),
        contract.polynomial_time_uniform,
        contract.total_transition,
        contract.all_fixed_points_agree,
        contract.ideal_deutsch_selector,
        contract.effects_frozen_or_modeled,
    )
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
                write!(escaped, "\\u{:04x}", character as u32)
                    .expect("writing to String cannot fail");
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

    fn family(flags: &str) -> Program {
        parse(&format!(
            "FAMILY uniform_input {{\n\
             CTC_CELLS POLY 1 1 0;\n\
             CHRONOLOGY_BITS POLY 1000 1 0;\n\
             TRANSITION_STEPS POLY 20 1 0;\n\
             {flags}\n\
             }}\n\
             INPUT 0 ORACLE DUP 0 PROPHECY POP OUTPUT"
        ))
        .unwrap()
    }

    const ALL_FLAGS: &str = "UNIFORM; TOTAL; READOUT_INVARIANT; IDEAL_DEUTSCH; EFFECTS_FROZEN;";

    fn width() -> PolynomialBound {
        PolynomialBound {
            coefficient: 1,
            degree: 1,
            additive: 0,
        }
    }

    #[test]
    fn constant_template_proves_linear_generation_and_connects_instances() {
        let certificate =
            PspaceUniformFamilyGenerator::admit(&family(ALL_FLAGS), width(), 1, BoundsPolicy::Wrap)
                .unwrap();
        certificate.check_structure().unwrap();
        let linked = BytecodeProgram::from_bytes(&certificate.bytecode).unwrap();
        certificate
            .recheck_bytecode(&certificate.contract, &linked)
            .unwrap();
        assert_eq!(certificate.minimum_temporal_cells, 1);
        assert_eq!(certificate.generation_work_bound.degree, 1);
        let theorem = ProjectionFamilyCertificate::prove(certificate.clone()).unwrap();
        assert_eq!(theorem.decision(&[0]).unwrap(), 0);
        assert_eq!(theorem.decision(&[1, 0, 1]).unwrap(), 1);
        assert!(theorem
            .to_json()
            .contains("\"proved_for_all_nonempty_inputs\""));

        let zero = certificate.specialize(&[0], 2, 100).unwrap();
        assert_eq!(zero.config.graph.memory_cells, 1);
        let zero = match zero.verify() {
            PspaceInstanceVerificationResult::Verified(certificate) => certificate,
            result => panic!("expected verified zero instance, got {result:?}"),
        };
        assert_eq!(zero.decision, 0);

        let two_bits = certificate.specialize(&[1, 1], 4, 100).unwrap();
        assert_eq!(two_bits.config.graph.memory_cells, 2);
        assert!(
            two_bits.generation.generation_work_ceiling
                > certificate.generation_work_bound.evaluate(1).unwrap()
        );
        let two_bits = match two_bits.verify() {
            PspaceInstanceVerificationResult::Verified(certificate) => certificate,
            result => panic!("expected verified two-bit instance, got {result:?}"),
        };
        assert_eq!(two_bits.decision, 1);
    }

    #[test]
    fn certificate_and_inputs_fail_closed_under_substitution() {
        let certificate =
            PspaceUniformFamilyGenerator::admit(&family(ALL_FLAGS), width(), 1, BoundsPolicy::Wrap)
                .unwrap();
        let mut changed_bytecode = certificate.clone();
        changed_bytecode.bytecode.push(0);
        assert!(matches!(
            changed_bytecode.check_structure(),
            Err(UniformFamilyError::Bytecode(_))
                | Err(UniformFamilyError::CertificateMismatch { .. })
        ));

        let mut changed_bound = certificate.clone();
        changed_bound.generation_work_bound.additive += 1;
        assert!(matches!(
            changed_bound.check_structure(),
            Err(UniformFamilyError::CertificateMismatch {
                field: "generation work polynomial"
            })
        ));
        let mut excessive_width = certificate.clone();
        excessive_width.temporal_width.coefficient += 1;
        assert_eq!(
            excessive_width.check_structure(),
            Err(UniformFamilyError::WidthRuleExceedsContract)
        );
        let linked = BytecodeProgram::from_bytes(&certificate.bytecode).unwrap();
        let mut substituted_contract = certificate.contract.clone();
        substituted_contract.chronology_respecting_bits.additive += 1;
        assert_eq!(
            certificate.recheck_bytecode(&substituted_contract, &linked),
            Err(UniformFamilyError::CertificateMismatch {
                field: "fresh compiler-bound replay"
            })
        );
        let theorem = ProjectionFamilyCertificate::prove(certificate.clone()).unwrap();
        let mut changed_theorem = theorem.clone();
        changed_theorem.maximum_stack_depth += 1;
        assert_eq!(
            changed_theorem.check_structure(),
            Err(UniformFamilyError::CertificateMismatch {
                field: "projection transition theorem"
            })
        );
        assert!(matches!(
            certificate.specialize(&[], 2, 100),
            Err(UniformFamilyError::InvalidInputLength { found: 0 })
        ));
        assert!(matches!(
            certificate.specialize(&[2], 2, 100),
            Err(UniformFamilyError::NonBooleanInput { .. })
        ));
    }

    #[test]
    fn projection_circuit_is_explicit_bounded_and_differentially_checked() {
        let generator =
            PspaceUniformFamilyGenerator::admit(&family(ALL_FLAGS), width(), 1, BoundsPolicy::Wrap)
                .unwrap();
        let theorem = ProjectionFamilyCertificate::prove(generator.clone()).unwrap();
        assert_eq!(theorem.circuit_output_wire_bound.evaluate(2), Some(3));
        let circuit = theorem.generate_circuit(2).unwrap();
        circuit.check_structure(&theorem).unwrap();
        assert_eq!(circuit.state_bits(), 2);
        assert_eq!(circuit.output_wires(), 3);
        assert_eq!(circuit.decision, ProjectionWire::Input(0));
        for state in [[0, 0], [1, 0], [0, 1], [1, 1]] {
            let (next, decision) = circuit.evaluate(&theorem, &[1, 0], &state).unwrap();
            assert_eq!(next, vec![state[0], 0]);
            assert_eq!(decision, 1);
        }
        assert!(circuit
            .to_json()
            .contains("\"schema\":\"ourochronos.projection-circuit/v1\""));

        let finite = match generator.specialize(&[1, 0], 4, 100).unwrap().verify() {
            PspaceInstanceVerificationResult::Verified(certificate) => certificate,
            result => panic!("expected verified instance, got {result:?}"),
        };
        theorem.cross_check_finite(&circuit, &finite).unwrap();
        assert!(theorem
            .circuit_verification_json(
                &circuit,
                &PspaceInstanceVerificationResult::Verified(finite.clone()),
            )
            .contains("\"schema\":\"ourochronos.projection-circuit-instance/v1\""));

        let mut rewired = circuit.clone();
        rewired.next_temporal[1] = ProjectionWire::Temporal(1);
        assert_eq!(
            rewired.check_structure(&theorem),
            Err(ProjectionCircuitError::StructureMismatch {
                field: "next-state wiring"
            })
        );
        assert!(matches!(
            circuit.evaluate(&theorem, &[2, 0], &[0, 0]),
            Err(ProjectionCircuitError::NonBooleanInput { index: 0, value: 2 })
        ));
        assert!(matches!(
            circuit.evaluate(&theorem, &[1, 0], &[0]),
            Err(ProjectionCircuitError::StateLengthMismatch {
                expected: 2,
                found: 1
            })
        ));
        assert!(matches!(
            theorem.generate_circuit(0),
            Err(ProjectionCircuitError::InvalidInputBits { found: 0 })
        ));

        let excessive_width = crate::core::MAX_DENSE_MEMORY_CELLS as u64;
        let wide = parse(&format!(
            "FAMILY wide {{ CTC_CELLS POLY 0 1 {excessive_width}; CHRONOLOGY_BITS POLY 1000 1 0; TRANSITION_STEPS POLY 20 1 0; UNIFORM; TOTAL; READOUT_INVARIANT; IDEAL_DEUTSCH; EFFECTS_FROZEN; }}\n\
             INPUT 0 ORACLE DUP 0 PROPHECY POP OUTPUT"
        ))
        .unwrap();
        let wide_rule = PolynomialBound {
            coefficient: 0,
            degree: 1,
            additive: excessive_width,
        };
        let wide =
            PspaceUniformFamilyGenerator::admit(&wide, wide_rule, 64, BoundsPolicy::Wrap).unwrap();
        let wide = ProjectionFamilyCertificate::prove(wide).unwrap();
        assert!(matches!(
            wide.generate_circuit(1),
            Err(ProjectionCircuitError::TooLarge { state_bits })
                if state_bits == crate::core::MAX_DENSE_MEMORY_CELLS * 64
        ));

        let constant = parse(
            "FAMILY constant { CTC_CELLS POLY 1 1 0; CHRONOLOGY_BITS POLY 1000 1 0; TRANSITION_STEPS POLY 20 1 0; UNIFORM; TOTAL; READOUT_INVARIANT; IDEAL_DEUTSCH; EFFECTS_FROZEN; }\n\
             0 ORACLE DUP 0 PROPHECY POP 1 OUTPUT",
        )
        .unwrap();
        let constant =
            PspaceUniformFamilyGenerator::admit(&constant, width(), 1, BoundsPolicy::Wrap).unwrap();
        let constant = ProjectionFamilyCertificate::prove(constant).unwrap();
        assert_eq!(
            constant.generate_circuit(1).unwrap().decision,
            ProjectionWire::Constant(true)
        );
    }

    #[test]
    fn sparse_routing_family_proves_multiple_assignments_and_last_write_wins() {
        let source = parse(
            "FAMILY routing { CTC_CELLS POLY 0 1 3; CHRONOLOGY_BITS POLY 1000 1 0; TRANSITION_STEPS POLY 30 1 0; UNIFORM; TOTAL; READOUT_INVARIANT; IDEAL_DEUTSCH; EFFECTS_FROZEN; }\n\
             INPUT 0 ORACLE 1 PROPHECY 1 ORACLE 2 PROPHECY 1 0 PROPHECY 0 1 PROPHECY OUTPUT",
        )
        .unwrap();
        let constant_three = PolynomialBound {
            coefficient: 0,
            degree: 1,
            additive: 3,
        };
        let generator =
            PspaceUniformFamilyGenerator::admit(&source, constant_three, 1, BoundsPolicy::Wrap)
                .unwrap();
        let theorem = ProjectionFamilyCertificate::prove(generator.clone()).unwrap();
        assert_eq!(theorem.assignments.len(), 4);
        assert_eq!(
            theorem.assignments,
            vec![
                ProjectionAssignment {
                    target: 1,
                    source: ProjectionCellSource::Temporal(0),
                },
                ProjectionAssignment {
                    target: 2,
                    source: ProjectionCellSource::Temporal(1),
                },
                ProjectionAssignment {
                    target: 0,
                    source: ProjectionCellSource::Constant(1),
                },
                ProjectionAssignment {
                    target: 1,
                    source: ProjectionCellSource::Constant(0),
                },
            ]
        );
        assert_eq!(theorem.maximum_stack_depth, 3);
        let circuit = theorem.generate_circuit(2).unwrap();
        assert_eq!(
            circuit.next_temporal,
            vec![
                ProjectionWire::Constant(true),
                ProjectionWire::Constant(false),
                ProjectionWire::Temporal(1),
            ]
        );
        let (next, decision) = circuit.evaluate(&theorem, &[1, 0], &[0, 1, 1]).unwrap();
        assert_eq!(next, vec![1, 0, 1]);
        assert_eq!(decision, 1);

        let finite = match generator.specialize(&[1, 0], 8, 100).unwrap().verify() {
            PspaceInstanceVerificationResult::Verified(certificate) => certificate,
            result => panic!("expected verified routing instance, got {result:?}"),
        };
        theorem.cross_check_finite(&circuit, &finite).unwrap();

        let mut changed = theorem.clone();
        changed.assignments[0].target = 2;
        assert_eq!(
            changed.check_structure(),
            Err(UniformFamilyError::CertificateMismatch {
                field: "projection transition theorem"
            })
        );

        let oversized_constant = parse(
            "FAMILY bad_constant { CTC_CELLS POLY 0 1 1; CHRONOLOGY_BITS POLY 1000 1 0; TRANSITION_STEPS POLY 10 1 0; UNIFORM; TOTAL; READOUT_INVARIANT; IDEAL_DEUTSCH; EFFECTS_FROZEN; }\n\
             2 0 PROPHECY 1 OUTPUT",
        )
        .unwrap();
        let width_one = PolynomialBound {
            coefficient: 0,
            degree: 1,
            additive: 1,
        };
        let generator = PspaceUniformFamilyGenerator::admit(
            &oversized_constant,
            width_one,
            1,
            BoundsPolicy::Wrap,
        )
        .unwrap();
        assert!(matches!(
            ProjectionFamilyCertificate::prove(generator),
            Err(UniformFamilyError::UnsupportedFamilyTheorem { ref reason })
                if reason.contains("does not fit 1-bit temporal cells")
        ));
    }

    #[test]
    fn sparse_routing_circuit_lowers_multibit_boolean_gates_exactly() {
        let source = parse(
            "FAMILY gates { CTC_CELLS POLY 0 1 5; CHRONOLOGY_BITS POLY 1000 1 0; TRANSITION_STEPS POLY 40 1 0; UNIFORM; TOTAL; READOUT_INVARIANT; IDEAL_DEUTSCH; EFFECTS_FROZEN; }\n\
             INPUT 0 ORACLE 1 ORACLE XOR 2 PROPHECY 0 ORACLE 1 ORACLE AND 3 PROPHECY 0 ORACLE 1 ORACLE OR 4 PROPHECY OUTPUT",
        )
        .unwrap();
        let width = PolynomialBound {
            coefficient: 0,
            degree: 1,
            additive: 5,
        };
        let generator =
            PspaceUniformFamilyGenerator::admit(&source, width, 2, BoundsPolicy::Wrap).unwrap();
        let theorem = ProjectionFamilyCertificate::prove(generator.clone()).unwrap();
        assert_eq!(
            theorem.assignments,
            vec![
                ProjectionAssignment {
                    target: 2,
                    source: ProjectionCellSource::Xor(0, 1),
                },
                ProjectionAssignment {
                    target: 3,
                    source: ProjectionCellSource::And(0, 1),
                },
                ProjectionAssignment {
                    target: 4,
                    source: ProjectionCellSource::Or(0, 1),
                },
            ]
        );
        let circuit = theorem.generate_circuit(1).unwrap();
        assert_eq!(
            circuit.next_temporal,
            vec![
                ProjectionWire::Constant(false),
                ProjectionWire::Constant(false),
                ProjectionWire::Constant(false),
                ProjectionWire::Constant(false),
                ProjectionWire::Xor(0, 2),
                ProjectionWire::Xor(1, 3),
                ProjectionWire::And(0, 2),
                ProjectionWire::And(1, 3),
                ProjectionWire::Or(0, 2),
                ProjectionWire::Or(1, 3),
            ]
        );
        let state = [0, 1, 1, 1, 0, 0, 0, 0, 0, 0];
        let (next, decision) = circuit.evaluate(&theorem, &[1], &state).unwrap();
        assert_eq!(next, vec![0, 0, 0, 0, 1, 0, 0, 1, 1, 1]);
        assert_eq!(decision, 1);
        assert!(circuit.to_json().contains("\"kind\":\"xor\""));

        let finite = match generator.specialize(&[1], 1_024, 100).unwrap().verify() {
            PspaceInstanceVerificationResult::Verified(certificate) => certificate,
            result => panic!("expected verified gate instance, got {result:?}"),
        };
        theorem.cross_check_finite(&circuit, &finite).unwrap();

        let masked = parse(
            "FAMILY masked { CTC_CELLS POLY 0 1 2; CHRONOLOGY_BITS POLY 1000 1 0; TRANSITION_STEPS POLY 20 1 0; UNIFORM; TOTAL; READOUT_INVARIANT; IDEAL_DEUTSCH; EFFECTS_FROZEN; }\n\
             INPUT 0 ORACLE 3 XOR 1 PROPHECY OUTPUT",
        )
        .unwrap();
        let width = PolynomialBound {
            coefficient: 0,
            degree: 1,
            additive: 2,
        };
        let masked =
            PspaceUniformFamilyGenerator::admit(&masked, width, 2, BoundsPolicy::Wrap).unwrap();
        let masked_theorem = ProjectionFamilyCertificate::prove(masked.clone()).unwrap();
        assert_eq!(
            masked_theorem.assignments,
            vec![ProjectionAssignment {
                target: 1,
                source: ProjectionCellSource::XorConstant(0, 3),
            }]
        );
        let masked_circuit = masked_theorem.generate_circuit(1).unwrap();
        assert_eq!(
            masked_circuit.next_temporal,
            vec![
                ProjectionWire::Constant(false),
                ProjectionWire::Constant(false),
                ProjectionWire::Not(0),
                ProjectionWire::Not(1),
            ]
        );
        assert_eq!(
            masked_circuit
                .evaluate(&masked_theorem, &[1], &[0, 1, 0, 0])
                .unwrap(),
            (vec![0, 0, 1, 0], 1)
        );
        let finite = match masked.specialize(&[1], 16, 100).unwrap().verify() {
            PspaceInstanceVerificationResult::Verified(certificate) => certificate,
            result => panic!("expected verified masked instance, got {result:?}"),
        };
        masked_theorem
            .cross_check_finite(&masked_circuit, &finite)
            .unwrap();

        let shifted = parse(
            "FAMILY shifted { CTC_CELLS POLY 0 1 2; CHRONOLOGY_BITS POLY 1000 1 0; TRANSITION_STEPS POLY 20 1 0; UNIFORM; TOTAL; READOUT_INVARIANT; IDEAL_DEUTSCH; EFFECTS_FROZEN; }\n\
             INPUT 0 ORACLE 65 SHR 1 PROPHECY OUTPUT",
        )
        .unwrap();
        let shifted =
            PspaceUniformFamilyGenerator::admit(&shifted, width, 3, BoundsPolicy::Wrap).unwrap();
        let shifted_theorem = ProjectionFamilyCertificate::prove(shifted.clone()).unwrap();
        assert_eq!(
            shifted_theorem.assignments[0].source,
            ProjectionCellSource::ShiftRight(0, 65)
        );
        let shifted_circuit = shifted_theorem.generate_circuit(1).unwrap();
        assert_eq!(
            shifted_circuit.next_temporal,
            vec![
                ProjectionWire::Constant(false),
                ProjectionWire::Constant(false),
                ProjectionWire::Constant(false),
                ProjectionWire::Temporal(1),
                ProjectionWire::Temporal(2),
                ProjectionWire::Constant(false),
            ]
        );
        assert_eq!(
            shifted_circuit
                .evaluate(&shifted_theorem, &[1], &[0, 1, 1, 0, 0, 0])
                .unwrap(),
            (vec![0, 0, 0, 1, 1, 0], 1)
        );
        let finite = match shifted.specialize(&[1], 64, 100).unwrap().verify() {
            PspaceInstanceVerificationResult::Verified(certificate) => certificate,
            result => panic!("expected verified shifted instance, got {result:?}"),
        };
        shifted_theorem
            .cross_check_finite(&shifted_circuit, &finite)
            .unwrap();

        let addition = parse(
            "FAMILY addition { CTC_CELLS POLY 0 1 3; CHRONOLOGY_BITS POLY 1000 1 0; TRANSITION_STEPS POLY 20 1 0; UNIFORM; TOTAL; READOUT_INVARIANT; IDEAL_DEUTSCH; EFFECTS_FROZEN; }\n\
             INPUT 0 ORACLE 1 ORACLE ADD 2 PROPHECY OUTPUT",
        )
        .unwrap();
        let width = PolynomialBound {
            coefficient: 0,
            degree: 1,
            additive: 3,
        };
        let addition =
            PspaceUniformFamilyGenerator::admit(&addition, width, 2, BoundsPolicy::Wrap).unwrap();
        assert!(matches!(
            ProjectionFamilyCertificate::prove(addition),
            Err(UniformFamilyError::UnsupportedFamilyTheorem { ref reason })
                if reason.contains("transition body must contain only")
        ));
    }

    #[test]
    fn source_admission_and_uniform_declaration_are_mandatory() {
        let nonuniform = family("TOTAL; READOUT_INVARIANT; IDEAL_DEUTSCH; EFFECTS_FROZEN;");
        assert_eq!(
            PspaceUniformFamilyGenerator::admit(&nonuniform, width(), 1, BoundsPolicy::Wrap),
            Err(UniformFamilyError::MissingUniformDeclaration)
        );

        let invalid = parse(
            "FAMILY invalid { CTC_CELLS POLY 1 1 0; CHRONOLOGY_BITS POLY 1000 1 0; TRANSITION_STEPS POLY 20 1 0; UNIFORM; TOTAL; READOUT_INVARIANT; IDEAL_DEUTSCH; EFFECTS_FROZEN; }\n\
             PROCEDURE hidden PURE { 1 OUTPUT }\n\
             0 ORACLE DUP 0 PROPHECY POP 1 OUTPUT",
        )
        .unwrap();
        assert!(matches!(
            PspaceUniformFamilyGenerator::admit(&invalid, width(), 1, BoundsPolicy::Wrap),
            Err(UniformFamilyError::Admission(_))
        ));

        let dynamic = parse(
            "FAMILY dynamic { CTC_CELLS POLY 1 1 0; CHRONOLOGY_BITS POLY 1000 1 0; TRANSITION_STEPS POLY 20 1 0; UNIFORM; TOTAL; READOUT_INVARIANT; IDEAL_DEUTSCH; EFFECTS_FROZEN; }\n\
             VEC_NEW POP INPUT 0 ORACLE DUP 0 PROPHECY POP OUTPUT",
        )
        .unwrap();
        assert!(matches!(
            PspaceUniformFamilyGenerator::admit(&dynamic, width(), 1, BoundsPolicy::Wrap),
            Err(UniformFamilyError::NonUniformTemplate { ref reason })
                if reason.contains("RuntimeState")
        ));

        let looped = parse(
            "FAMILY looped { CTC_CELLS POLY 1 1 0; CHRONOLOGY_BITS POLY 1000 1 0; TRANSITION_STEPS POLY 20 1 0; UNIFORM; TOTAL; READOUT_INVARIANT; IDEAL_DEUTSCH; EFFECTS_FROZEN; }\n\
             WHILE { 0 } { NOP } INPUT 0 ORACLE DUP 0 PROPHECY POP OUTPUT",
        )
        .unwrap();
        assert!(matches!(
            PspaceUniformFamilyGenerator::admit(&looped, width(), 1, BoundsPolicy::Wrap),
            Err(UniformFamilyError::NonUniformTemplate { ref reason })
                if reason.contains("BoundedLoop")
        ));
    }

    #[test]
    fn width_inequality_is_proved_for_every_nonempty_length() {
        let n = width();
        let constant_one = PolynomialBound {
            coefficient: 0,
            degree: 7,
            additive: 1,
        };
        assert!(polynomial_bounded_for_nonempty(constant_one, n));
        assert!(polynomial_bounded_for_nonempty(n, n));
        assert!(!polynomial_bounded_for_nonempty(
            PolynomialBound {
                coefficient: 2,
                degree: 1,
                additive: 0,
            },
            n,
        ));
        assert!(!polynomial_bounded_for_nonempty(
            PolynomialBound {
                coefficient: 1,
                degree: 2,
                additive: 0,
            },
            n,
        ));
    }

    #[test]
    fn projection_theorem_rejects_noncanonical_or_underbounded_families() {
        let noncanonical = parse(
            "FAMILY extra { CTC_CELLS POLY 1 1 0; CHRONOLOGY_BITS POLY 1000 1 0; TRANSITION_STEPS POLY 20 1 0; UNIFORM; TOTAL; READOUT_INVARIANT; IDEAL_DEUTSCH; EFFECTS_FROZEN; }\n\
             NOP INPUT 0 ORACLE DUP 0 PROPHECY POP OUTPUT",
        )
        .unwrap();
        let generator =
            PspaceUniformFamilyGenerator::admit(&noncanonical, width(), 1, BoundsPolicy::Wrap)
                .unwrap();
        assert!(matches!(
            ProjectionFamilyCertificate::prove(generator),
            Err(UniformFamilyError::UnsupportedFamilyTheorem { ref reason })
                if reason.contains("canonical one-cell projection")
        ));

        let underbounded = parse(
            "FAMILY tight { CTC_CELLS POLY 1 1 0; CHRONOLOGY_BITS POLY 1 1 0; TRANSITION_STEPS POLY 1 0 0; UNIFORM; TOTAL; READOUT_INVARIANT; IDEAL_DEUTSCH; EFFECTS_FROZEN; }\n\
             INPUT 0 ORACLE DUP 0 PROPHECY POP OUTPUT",
        )
        .unwrap();
        let generator =
            PspaceUniformFamilyGenerator::admit(&underbounded, width(), 1, BoundsPolicy::Wrap)
                .unwrap();
        assert!(matches!(
            ProjectionFamilyCertificate::prove(generator),
            Err(UniformFamilyError::UnsupportedFamilyTheorem { ref reason })
                if reason.contains("chronology polynomial")
        ));
    }
}
