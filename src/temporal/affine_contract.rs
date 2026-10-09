//! Relational assume/guarantee contracts for finite affine Boolean transfers.
//!
//! Preconditions are parity equations on BEFORE; postconditions relate BEFORE
//! and AFTER, and protected bits must retain their value. Implication is proved
//! by bounded GF(2) elimination, without enumerating states or requiring any
//! recurrence/stabilization property. Unsatisfiable assumptions are reported as
//! vacuity, never as a successful contract.
//!
//! Composition means the model pipeline x -> F(x) -> G(F(x)), with the first
//! result supplied as the second component's input. It does not assert that
//! concatenating two TEMPORAL bodies has those semantics. Both component
//! contracts must hold, actual handoff must satisfy the right precondition,
//! and left declared guarantees/frames must entail that precondition. A weak
//! declaration yields InsufficientGuarantee, distinct from an actual handoff
//! counterexample. Result frames are the intersection. Left relations survive
//! when G frames their AFTER bits; right BEFORE terms are pulled back through
//! F, including its affine offset.
//!
//! Certificates bind exact models/contracts/version and optional source digest
//! and resource profiles. Checking uses separate dense elimination and direct
//! evaluation on an affine solution-space basis. Source extraction remains
//! trusted and reuses the conservative affine bytecode adapter. Source claims
//! concern its masked finite scope and fresh outside-zero domain, not arbitrary
//! full-word inputs, outside-memory preservation, compiler correctness, effects,
//! or termination of unsupported loops/calls/host operations.

use super::affine_recurrence::{
    extract_affine_bytecode, AffineBooleanSystem, AffineClassCount, AffineExtractionConfig,
    AffineRecurrenceError, AFFINE_RECURRENCE_SEMANTICS_VERSION,
};
use crate::bytecode::BytecodeProgram;
use std::fmt;

pub const AFFINE_CONTRACT_SEMANTICS_VERSION: u16 = 1;
pub const MAX_AFFINE_ASSUMPTIONS: usize = 256;
pub const MAX_AFFINE_GUARANTEES: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AffineParityEquation {
    pub mask: u64,
    pub value: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AffineTemporalRelation {
    pub before_mask: u64,
    pub after_mask: u64,
    pub value: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineTemporalContract {
    pub assumptions: Vec<AffineParityEquation>,
    pub guarantees: Vec<AffineTemporalRelation>,
    pub protected_bits: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineContractSourceBinding {
    pub extraction_semantics_version: u16,
    pub program_sha256: [u8; 32],
    pub resources: AffineExtractionConfig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineContractComponent {
    pub model: AffineBooleanSystem,
    pub contract: AffineTemporalContract,
    /// A declared source identity; validate it with source bytes through
    /// `check_affine_source_component` before claiming source correspondence.
    pub source: Option<AffineContractSourceBinding>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AffineContractFailure {
    Guarantee { index: usize },
    ProtectedBit { bit: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AffineContractCounterexample {
    pub initial: u64,
    pub successor: u64,
    pub failure: AffineContractFailure,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedAffineContract {
    pub assumption_rank: u8,
    pub satisfying_states: AffineClassCount,
    pub protected_bits: u64,
}

/// Bounded untrusted data; the checker recomputes both assumptions and proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineContractCertificate {
    pub semantics_version: u16,
    pub component: AffineContractComponent,
    pub canonical_assumptions: Vec<AffineParityEquation>,
    pub summary: VerifiedAffineContract,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AffineContractAnalysis {
    Proven(AffineContractCertificate),
    UnsatisfiableAssumptions,
    Counterexample(AffineContractCounterexample),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AffineComponentSide {
    Left,
    Right,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AffineCompositionAnalysis {
    Proven(Box<AffineCompositionCertificate>),
    UnsatisfiableAssumptions {
        side: AffineComponentSide,
    },
    ComponentCounterexample {
        side: AffineComponentSide,
        counterexample: AffineContractCounterexample,
    },
    /// A real F transition violates a right assumption.
    HandoffCounterexample {
        initial: u64,
        intermediate: u64,
        assumption_index: usize,
    },
    /// An abstract intermediate is permitted by left guarantees/frames but
    /// violates a right assumption; actual F handoff was proved valid first.
    InsufficientGuarantee {
        initial: u64,
        permitted_intermediate: u64,
        actual_intermediate: u64,
        assumption_index: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineCompositionCertificate {
    pub semantics_version: u16,
    pub left: AffineContractCertificate,
    pub right: AffineContractCertificate,
    pub composed: AffineContractCertificate,
    /// Canonical joint BEFORE/intermediate equations; at most 128 pivots.
    pub canonical_handoff: Vec<AffineTemporalRelation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedAffineComposition {
    pub model: AffineBooleanSystem,
    pub contract: AffineTemporalContract,
    pub summary: VerifiedAffineContract,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AffineContractError {
    Model(AffineRecurrenceError),
    InvalidContract(&'static str),
    ResourceLimit(&'static str),
    BindingMismatch(&'static str),
    InvalidCertificate(&'static str),
    VacuousAssumptions,
    InternalInvariant(&'static str),
}

impl From<AffineRecurrenceError> for AffineContractError {
    fn from(error: AffineRecurrenceError) -> Self {
        Self::Model(error)
    }
}
impl fmt::Display for AffineContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(error) => write!(f, "{error}"),
            Self::InvalidContract(reason) => write!(f, "invalid affine contract: {reason}"),
            Self::ResourceLimit(reason) => write!(f, "affine contract resource limit: {reason}"),
            Self::BindingMismatch(field) => write!(f, "affine contract binding mismatch: {field}"),
            Self::InvalidCertificate(reason) => {
                write!(f, "invalid affine contract certificate: {reason}")
            }
            Self::VacuousAssumptions => write!(f, "affine assumptions are unsatisfiable"),
            Self::InternalInvariant(reason) => {
                write!(f, "affine contract invariant failed: {reason}")
            }
        }
    }
}
impl std::error::Error for AffineContractError {}

fn mask(bits: usize) -> u64 {
    if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}
fn parity(word: u128) -> bool {
    word.count_ones() & 1 != 0
}

fn validate_component(component: &AffineContractComponent) -> Result<(), AffineContractError> {
    component.model.validate()?;
    let contract = &component.contract;
    if contract.assumptions.len() > MAX_AFFINE_ASSUMPTIONS
        || contract.guarantees.len() > MAX_AFFINE_GUARANTEES
    {
        return Err(AffineContractError::ResourceLimit(
            "equation count exceeds bounded contract profile",
        ));
    }
    let domain = mask(component.model.bits as usize);
    if contract.protected_bits & !domain != 0
        || contract
            .assumptions
            .iter()
            .any(|row| row.mask & !domain != 0)
        || contract
            .guarantees
            .iter()
            .any(|row| (row.before_mask | row.after_mask) & !domain != 0)
    {
        return Err(AffineContractError::InvalidContract(
            "mask outside declared model dimension",
        ));
    }
    if let Some(source) = &component.source {
        if source.extraction_semantics_version != AFFINE_RECURRENCE_SEMANTICS_VERSION {
            return Err(AffineContractError::BindingMismatch(
                "source extraction semantics version",
            ));
        }
        if source.resources.memory_cells < component.model.bits as usize {
            return Err(AffineContractError::InvalidContract(
                "source memory smaller than modeled scope",
            ));
        }
        if source.resources.memory_cells > 4096
            || source.resources.max_stack_depth > 4096
            || source.resources.max_instructions < 3
        {
            return Err(AffineContractError::ResourceLimit(
                "source resource profile cannot admit a scoped affine body",
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Row {
    coefficients: u128,
    value: bool,
}

struct SolutionSpace {
    rows: Vec<Row>,
    particular: u128,
    kernel: Vec<u128>,
}

fn generator_space(input: &[Row], variables: usize) -> Option<SolutionSpace> {
    let mut pivots = vec![None::<Row>; variables];
    for &incoming in input {
        let mut row = incoming;
        for pivot in pivots.iter().flatten() {
            if row.coefficients & (1u128 << pivot.coefficients.trailing_zeros()) != 0 {
                row.coefficients ^= pivot.coefficients;
                row.value ^= pivot.value;
            }
        }
        if row.coefficients == 0 {
            if row.value {
                return None;
            }
            continue;
        }
        let bit = row.coefficients.trailing_zeros() as usize;
        for pivot in pivots.iter_mut().flatten() {
            if pivot.coefficients & (1u128 << bit) != 0 {
                pivot.coefficients ^= row.coefficients;
                pivot.value ^= row.value;
            }
        }
        pivots[bit] = Some(row);
    }
    let rows: Vec<_> = pivots.iter().flatten().copied().collect();
    let mut particular = 0;
    for row in &rows {
        if row.value {
            particular |= 1u128 << row.coefficients.trailing_zeros();
        }
    }
    let mut kernel = Vec::new();
    for (free, pivot) in pivots.iter().enumerate() {
        if pivot.is_some() {
            continue;
        }
        let mut vector = 1u128 << free;
        for row in &rows {
            if row.coefficients & (1u128 << free) != 0 {
                vector |= 1u128 << row.coefficients.trailing_zeros();
            }
        }
        kernel.push(vector);
    }
    Some(SolutionSpace {
        rows,
        particular,
        kernel,
    })
}

fn implication_counterexample(
    space: &SolutionSpace,
    equation: Row,
) -> Result<Option<u128>, AffineContractError> {
    let mut reduced = equation;
    for pivot in &space.rows {
        if reduced.coefficients & (1u128 << pivot.coefficients.trailing_zeros()) != 0 {
            reduced.coefficients ^= pivot.coefficients;
            reduced.value ^= pivot.value;
        }
    }
    if reduced.coefficients == 0 && !reduced.value {
        return Ok(None);
    }
    if parity(equation.coefficients & space.particular) != equation.value {
        return Ok(Some(space.particular));
    }
    let vector = space
        .kernel
        .iter()
        .copied()
        .find(|&v| parity(equation.coefficients & v))
        .ok_or(AffineContractError::InternalInvariant(
            "non-entailed equation has no solution-space counterexample",
        ))?;
    Ok(Some(space.particular ^ vector))
}

fn pre_rows(contract: &AffineTemporalContract) -> Vec<Row> {
    contract
        .assumptions
        .iter()
        .map(|equation| Row {
            coefficients: equation.mask as u128,
            value: equation.value,
        })
        .collect()
}

fn pullback(model: &AffineBooleanSystem, relation: AffineTemporalRelation) -> Row {
    let mut coefficients = relation.before_mask;
    let mut after = relation.after_mask;
    while after != 0 {
        let bit = after.trailing_zeros() as usize;
        coefficients ^= model.rows[bit];
        after &= after - 1;
    }
    Row {
        coefficients: coefficients as u128,
        value: relation.value ^ parity((relation.after_mask & model.offset) as u128),
    }
}

fn frame_relations(
    contract: &AffineTemporalContract,
    bits: u8,
) -> impl Iterator<Item = AffineTemporalRelation> + '_ {
    (0..bits)
        .filter(|&bit| contract.protected_bits & (1u64 << bit) != 0)
        .map(|bit| AffineTemporalRelation {
            before_mask: 1u64 << bit,
            after_mask: 1u64 << bit,
            value: false,
        })
}

fn verify_witness(
    component: &AffineContractComponent,
    initial: u64,
    failure: AffineContractFailure,
) -> Result<AffineContractCounterexample, AffineContractError> {
    let successor = component.model.apply(initial)?;
    if component
        .contract
        .assumptions
        .iter()
        .any(|eq| parity((eq.mask & initial) as u128) != eq.value)
    {
        return Err(AffineContractError::InternalInvariant(
            "counterexample violates its assumptions",
        ));
    }
    let violated = match failure {
        AffineContractFailure::Guarantee { index } => {
            let row = component.contract.guarantees[index];
            parity(((row.before_mask & initial) ^ (row.after_mask & successor)) as u128)
                != row.value
        }
        AffineContractFailure::ProtectedBit { bit } => (initial ^ successor) & (1u64 << bit) != 0,
    };
    if !violated {
        return Err(AffineContractError::InternalInvariant(
            "counterexample does not violate its declared relation",
        ));
    }
    Ok(AffineContractCounterexample {
        initial,
        successor,
        failure,
    })
}

pub fn certify_affine_contract(
    component: &AffineContractComponent,
) -> Result<AffineContractAnalysis, AffineContractError> {
    validate_component(component)?;
    let Some(space) = generator_space(
        &pre_rows(&component.contract),
        component.model.bits as usize,
    ) else {
        return Ok(AffineContractAnalysis::UnsatisfiableAssumptions);
    };
    for (index, &relation) in component.contract.guarantees.iter().enumerate() {
        if let Some(state) =
            implication_counterexample(&space, pullback(&component.model, relation))?
        {
            return Ok(AffineContractAnalysis::Counterexample(verify_witness(
                component,
                state as u64,
                AffineContractFailure::Guarantee { index },
            )?));
        }
    }
    for relation in frame_relations(&component.contract, component.model.bits) {
        if let Some(state) =
            implication_counterexample(&space, pullback(&component.model, relation))?
        {
            return Ok(AffineContractAnalysis::Counterexample(verify_witness(
                component,
                state as u64,
                AffineContractFailure::ProtectedBit {
                    bit: relation.before_mask.trailing_zeros() as u8,
                },
            )?));
        }
    }
    let rank = space.rows.len() as u8;
    let certificate = AffineContractCertificate {
        semantics_version: AFFINE_CONTRACT_SEMANTICS_VERSION,
        component: component.clone(),
        canonical_assumptions: space
            .rows
            .iter()
            .map(|row| AffineParityEquation {
                mask: row.coefficients as u64,
                value: row.value,
            })
            .collect(),
        summary: VerifiedAffineContract {
            assumption_rank: rank,
            satisfying_states: AffineClassCount {
                exponent: component.model.bits - rank,
            },
            protected_bits: component.contract.protected_bits,
        },
    };
    check_affine_contract_certificate(&certificate, component)?;
    Ok(AffineContractAnalysis::Proven(certificate))
}

// Independent checker: dense row elimination, manual parity, and direct model
// evaluation on particular/kernel points. No generator pullback/composition or
// implication-reduction helpers are used to establish contract truth.
fn checker_parity(mut word: u128) -> bool {
    let mut odd = false;
    while word != 0 {
        odd = !odd;
        word &= word - 1;
    }
    odd
}
fn checker_evaluate(model: &AffineBooleanSystem, state: u64) -> u64 {
    let mut output = model.offset;
    for (bit, &row) in model.rows.iter().enumerate() {
        if checker_parity((row & state) as u128) {
            output ^= 1u64 << bit;
        }
    }
    output
}
fn checker_space(mut rows: Vec<Row>, variables: usize) -> Option<SolutionSpace> {
    let mut rank = 0;
    let mut pivot_bits = Vec::new();
    for bit in 0..variables {
        let Some(pivot) =
            (rank..rows.len()).find(|&index| rows[index].coefficients & (1u128 << bit) != 0)
        else {
            continue;
        };
        rows.swap(rank, pivot);
        let pivot = rows[rank];
        for (index, row) in rows.iter_mut().enumerate() {
            if index != rank && row.coefficients & (1u128 << bit) != 0 {
                row.coefficients ^= pivot.coefficients;
                row.value ^= pivot.value;
            }
        }
        pivot_bits.push(bit);
        rank += 1;
    }
    if rows.iter().any(|row| row.coefficients == 0 && row.value) {
        return None;
    }
    rows.truncate(rank);
    let mut particular = 0;
    for (&bit, row) in pivot_bits.iter().zip(&rows) {
        if row.value {
            particular |= 1u128 << bit;
        }
    }
    let mut kernel = Vec::new();
    for free in 0..variables {
        if pivot_bits.contains(&free) {
            continue;
        }
        let mut vector = 1u128 << free;
        for (&pivot, row) in pivot_bits.iter().zip(&rows) {
            if row.coefficients & (1u128 << free) != 0 {
                vector |= 1u128 << pivot;
            }
        }
        kernel.push(vector);
    }
    Some(SolutionSpace {
        rows,
        particular,
        kernel,
    })
}

pub fn check_affine_contract_certificate(
    certificate: &AffineContractCertificate,
    expected_component: &AffineContractComponent,
) -> Result<VerifiedAffineContract, AffineContractError> {
    if certificate.semantics_version != AFFINE_CONTRACT_SEMANTICS_VERSION {
        return Err(AffineContractError::BindingMismatch(
            "contract semantics version",
        ));
    }
    if certificate.component != *expected_component {
        return Err(AffineContractError::BindingMismatch(
            "exact component/model/contract/source resources",
        ));
    }
    validate_component(expected_component)?;
    if certificate.canonical_assumptions.len() > 64 {
        return Err(AffineContractError::InvalidCertificate(
            "too many canonical precondition pivots",
        ));
    }
    let pre: Vec<_> = expected_component
        .contract
        .assumptions
        .iter()
        .map(|eq| Row {
            coefficients: eq.mask as u128,
            value: eq.value,
        })
        .collect();
    let space = checker_space(pre, expected_component.model.bits as usize)
        .ok_or(AffineContractError::VacuousAssumptions)?;
    let canonical: Vec<_> = space
        .rows
        .iter()
        .map(|row| AffineParityEquation {
            mask: row.coefficients as u64,
            value: row.value,
        })
        .collect();
    let rank = canonical.len() as u8;
    let summary = VerifiedAffineContract {
        assumption_rank: rank,
        satisfying_states: AffineClassCount {
            exponent: expected_component.model.bits - rank,
        },
        protected_bits: expected_component.contract.protected_bits,
    };
    if certificate.canonical_assumptions != canonical || certificate.summary != summary {
        return Err(AffineContractError::InvalidCertificate(
            "incorrect assumption basis/rank/count/frame",
        ));
    }
    for point in std::iter::once(space.particular)
        .chain(space.kernel.iter().map(|&vector| space.particular ^ vector))
    {
        let initial = point as u64;
        let successor = checker_evaluate(&expected_component.model, initial);
        for relation in &expected_component.contract.guarantees {
            if checker_parity(
                ((relation.before_mask & initial) ^ (relation.after_mask & successor)) as u128,
            ) != relation.value
            {
                return Err(AffineContractError::InvalidCertificate(
                    "guarantee fails on affine solution-space basis",
                ));
            }
        }
        if (initial ^ successor) & expected_component.contract.protected_bits != 0 {
            return Err(AffineContractError::InvalidCertificate(
                "protected frame fails on affine solution-space basis",
            ));
        }
    }
    Ok(summary)
}

fn joint_rows(contract: &AffineTemporalContract, bits: u8) -> Vec<Row> {
    let mut rows = pre_rows(contract);
    rows.extend(
        contract
            .guarantees
            .iter()
            .copied()
            .chain(frame_relations(contract, bits))
            .map(|relation| Row {
                coefficients: relation.before_mask as u128
                    | ((relation.after_mask as u128) << bits),
                value: relation.value,
            }),
    );
    rows
}

fn composed_model(left: &AffineBooleanSystem, right: &AffineBooleanSystem) -> AffineBooleanSystem {
    let rows = right
        .rows
        .iter()
        .map(|&mask| {
            let mut coefficients = 0;
            let mut bits = mask;
            while bits != 0 {
                coefficients ^= left.rows[bits.trailing_zeros() as usize];
                bits &= bits - 1;
            }
            coefficients
        })
        .collect();
    let offset = right
        .rows
        .iter()
        .enumerate()
        .fold(right.offset, |offset, (bit, row)| {
            offset ^ (u64::from(parity((row & left.offset) as u128)) << bit)
        });
    AffineBooleanSystem {
        bits: left.bits,
        rows,
        offset,
    }
}

fn composed_contract(
    left: &AffineContractComponent,
    right: &AffineContractComponent,
) -> Result<AffineTemporalContract, AffineContractError> {
    let mut guarantees: Vec<_> = left
        .contract
        .guarantees
        .iter()
        .copied()
        .filter(|relation| relation.after_mask & !right.contract.protected_bits == 0)
        .collect();
    for &relation in &right.contract.guarantees {
        let pulled = pullback(
            &left.model,
            AffineTemporalRelation {
                before_mask: 0,
                after_mask: relation.before_mask,
                value: relation.value,
            },
        );
        guarantees.push(AffineTemporalRelation {
            before_mask: pulled.coefficients as u64,
            after_mask: relation.after_mask,
            value: pulled.value,
        });
    }
    if guarantees.len() > MAX_AFFINE_GUARANTEES {
        return Err(AffineContractError::ResourceLimit(
            "composed guarantee count exceeds ceiling",
        ));
    }
    Ok(AffineTemporalContract {
        assumptions: left.contract.assumptions.clone(),
        guarantees,
        protected_bits: left.contract.protected_bits & right.contract.protected_bits,
    })
}

pub fn compose_affine_contracts(
    left: &AffineContractComponent,
    right: &AffineContractComponent,
) -> Result<AffineCompositionAnalysis, AffineContractError> {
    validate_component(left)?;
    validate_component(right)?;
    if left.model.bits != right.model.bits {
        return Err(AffineContractError::InvalidContract(
            "composition dimensions differ",
        ));
    }
    let left_certificate = match certify_affine_contract(left)? {
        AffineContractAnalysis::Proven(certificate) => certificate,
        AffineContractAnalysis::UnsatisfiableAssumptions => {
            return Ok(AffineCompositionAnalysis::UnsatisfiableAssumptions {
                side: AffineComponentSide::Left,
            })
        }
        AffineContractAnalysis::Counterexample(counterexample) => {
            return Ok(AffineCompositionAnalysis::ComponentCounterexample {
                side: AffineComponentSide::Left,
                counterexample,
            })
        }
    };
    let right_certificate = match certify_affine_contract(right)? {
        AffineContractAnalysis::Proven(certificate) => certificate,
        AffineContractAnalysis::UnsatisfiableAssumptions => {
            return Ok(AffineCompositionAnalysis::UnsatisfiableAssumptions {
                side: AffineComponentSide::Right,
            })
        }
        AffineContractAnalysis::Counterexample(counterexample) => {
            return Ok(AffineCompositionAnalysis::ComponentCounterexample {
                side: AffineComponentSide::Right,
                counterexample,
            })
        }
    };
    let pre = generator_space(&pre_rows(&left.contract), left.model.bits as usize).ok_or(
        AffineContractError::InternalInvariant("proved left assumptions are vacuous"),
    )?;
    for (index, assumption) in right.contract.assumptions.iter().enumerate() {
        let target = pullback(
            &left.model,
            AffineTemporalRelation {
                before_mask: 0,
                after_mask: assumption.mask,
                value: assumption.value,
            },
        );
        if let Some(initial) = implication_counterexample(&pre, target)? {
            return Ok(AffineCompositionAnalysis::HandoffCounterexample {
                initial: initial as u64,
                intermediate: left.model.apply(initial as u64)?,
                assumption_index: index,
            });
        }
    }
    let bits = left.model.bits;
    let joint = generator_space(&joint_rows(&left.contract, bits), 2 * bits as usize).ok_or(
        AffineContractError::InternalInvariant("verified left relation has no joint solution"),
    )?;
    for (index, assumption) in right.contract.assumptions.iter().enumerate() {
        if let Some(point) = implication_counterexample(
            &joint,
            Row {
                coefficients: (assumption.mask as u128) << bits,
                value: assumption.value,
            },
        )? {
            let initial = (point as u64) & mask(bits as usize);
            return Ok(AffineCompositionAnalysis::InsufficientGuarantee {
                initial,
                permitted_intermediate: (point >> bits) as u64,
                actual_intermediate: left.model.apply(initial)?,
                assumption_index: index,
            });
        }
    }
    let component = AffineContractComponent {
        model: composed_model(&left.model, &right.model),
        contract: composed_contract(left, right)?,
        source: None,
    };
    let composed = match certify_affine_contract(&component)? {
        AffineContractAnalysis::Proven(certificate) => certificate,
        _ => {
            return Err(AffineContractError::InternalInvariant(
                "derived composed contract did not verify",
            ))
        }
    };
    let certificate = AffineCompositionCertificate {
        semantics_version: AFFINE_CONTRACT_SEMANTICS_VERSION,
        left: left_certificate,
        right: right_certificate,
        composed,
        canonical_handoff: joint
            .rows
            .iter()
            .map(|row| AffineTemporalRelation {
                before_mask: row.coefficients as u64 & mask(bits as usize),
                after_mask: (row.coefficients >> bits) as u64,
                value: row.value,
            })
            .collect(),
    };
    check_affine_composition_certificate(&certificate, left, right)?;
    Ok(AffineCompositionAnalysis::Proven(Box::new(certificate)))
}

pub fn check_affine_composition_certificate(
    certificate: &AffineCompositionCertificate,
    expected_left: &AffineContractComponent,
    expected_right: &AffineContractComponent,
) -> Result<VerifiedAffineComposition, AffineContractError> {
    if certificate.semantics_version != AFFINE_CONTRACT_SEMANTICS_VERSION {
        return Err(AffineContractError::BindingMismatch(
            "composition semantics version",
        ));
    }
    check_affine_contract_certificate(&certificate.left, expected_left)?;
    check_affine_contract_certificate(&certificate.right, expected_right)?;
    if expected_left.model.bits != expected_right.model.bits {
        return Err(AffineContractError::InvalidContract(
            "composition dimensions differ",
        ));
    }
    if certificate.canonical_handoff.len() > 128 {
        return Err(AffineContractError::InvalidCertificate(
            "too many handoff pivots",
        ));
    }
    let bits = expected_left.model.bits;
    let mut rows: Vec<_> = expected_left
        .contract
        .assumptions
        .iter()
        .map(|row| Row {
            coefficients: row.mask as u128,
            value: row.value,
        })
        .collect();
    rows.extend(expected_left.contract.guarantees.iter().map(|row| Row {
        coefficients: row.before_mask as u128 | ((row.after_mask as u128) << bits),
        value: row.value,
    }));
    for bit in 0..bits {
        if expected_left.contract.protected_bits & (1u64 << bit) != 0 {
            rows.push(Row {
                coefficients: (1u128 << bit) | (1u128 << (bit as usize + bits as usize)),
                value: false,
            });
        }
    }
    let joint =
        checker_space(rows, 2 * bits as usize).ok_or(AffineContractError::VacuousAssumptions)?;
    let canonical: Vec<_> = joint
        .rows
        .iter()
        .map(|row| AffineTemporalRelation {
            before_mask: row.coefficients as u64 & mask(bits as usize),
            after_mask: (row.coefficients >> bits) as u64,
            value: row.value,
        })
        .collect();
    if certificate.canonical_handoff != canonical {
        return Err(AffineContractError::InvalidCertificate(
            "incorrect relational handoff basis",
        ));
    }
    for point in std::iter::once(joint.particular)
        .chain(joint.kernel.iter().map(|&vector| joint.particular ^ vector))
    {
        let intermediate = (point >> bits) as u64;
        for assumption in &expected_right.contract.assumptions {
            if checker_parity((assumption.mask & intermediate) as u128) != assumption.value {
                return Err(AffineContractError::InvalidCertificate(
                    "left guarantees/frames do not imply right assumptions",
                ));
            }
        }
    }
    let offset = checker_evaluate(
        &expected_right.model,
        checker_evaluate(&expected_left.model, 0),
    );
    let mut composed_rows = vec![0; bits as usize];
    for input in 0..bits {
        let column = checker_evaluate(
            &expected_right.model,
            checker_evaluate(&expected_left.model, 1u64 << input),
        ) ^ offset;
        for (output, row) in composed_rows.iter_mut().enumerate() {
            if column & (1u64 << output) != 0 {
                *row |= 1u64 << input;
            }
        }
    }
    let model = AffineBooleanSystem {
        bits,
        rows: composed_rows,
        offset,
    };
    let mut guarantees: Vec<_> = expected_left
        .contract
        .guarantees
        .iter()
        .copied()
        .filter(|row| row.after_mask & !expected_right.contract.protected_bits == 0)
        .collect();
    for relation in &expected_right.contract.guarantees {
        let constant = checker_parity(
            (relation.before_mask & checker_evaluate(&expected_left.model, 0)) as u128,
        );
        let mut before_mask = 0;
        for input in 0..bits {
            if checker_parity(
                (relation.before_mask & checker_evaluate(&expected_left.model, 1u64 << input))
                    as u128,
            ) ^ constant
            {
                before_mask |= 1u64 << input;
            }
        }
        guarantees.push(AffineTemporalRelation {
            before_mask,
            after_mask: relation.after_mask,
            value: relation.value ^ constant,
        });
    }
    if guarantees.len() > MAX_AFFINE_GUARANTEES {
        return Err(AffineContractError::ResourceLimit(
            "composed guarantee count exceeds ceiling",
        ));
    }
    let contract = AffineTemporalContract {
        assumptions: expected_left.contract.assumptions.clone(),
        guarantees,
        protected_bits: expected_left.contract.protected_bits
            & expected_right.contract.protected_bits,
    };
    let component = AffineContractComponent {
        model: model.clone(),
        contract: contract.clone(),
        source: None,
    };
    let summary = check_affine_contract_certificate(&certificate.composed, &component)?;
    Ok(VerifiedAffineComposition {
        model,
        contract,
        summary,
    })
}

pub fn affine_component_from_bytecode(
    program: &BytecodeProgram,
    resources: &AffineExtractionConfig,
    contract: AffineTemporalContract,
) -> Result<AffineContractComponent, AffineContractError> {
    let extracted = extract_affine_bytecode(program, resources)?;
    let component = AffineContractComponent {
        model: extracted.model,
        contract,
        source: Some(AffineContractSourceBinding {
            extraction_semantics_version: extracted.semantics_version,
            program_sha256: extracted.program_sha256,
            resources: extracted.config,
        }),
    };
    validate_component(&component)?;
    Ok(component)
}

pub fn check_affine_source_component(
    component: &AffineContractComponent,
    expected_program: &BytecodeProgram,
    expected_resources: &AffineExtractionConfig,
) -> Result<(), AffineContractError> {
    validate_component(component)?;
    let expected = affine_component_from_bytecode(
        expected_program,
        expected_resources,
        component.contract.clone(),
    )?;
    if component.model != expected.model || component.source != expected.source {
        return Err(AffineContractError::BindingMismatch(
            "exact source model/digest/resource profile",
        ));
    }
    Ok(())
}
