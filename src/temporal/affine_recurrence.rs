//! Symbolic recurrence for coupled affine Boolean systems, without enumeration.
//!
//! F(x) = A x + b over GF(2), with 1..64 bits and arbitrary coupled rows.
//! Admission requires F^(n+1) = F^n. Thus P = F^n maps every state to a fixed
//! state, image(P) is exactly the recurrent set, and every recurrent class is
//! a singleton. Its cardinality is 2^rank(linear P). F^(2n) = F^n alone would
//! be insufficient: an involution in even dimension satisfies that equality.
//!
//! A linear Boolean readout is uniform precisely when it annihilates the image
//! basis. Otherwise two concrete fixed states witness different readouts and,
//! under the admitted singleton-class contract, different recurrent classes.
//! Checking reconstructs P by applying F to zero and unit vectors, separately
//! eliminates its image columns, and recomputes every claim. It does not call
//! the generator's matrix-composition/basis helpers, a VM, IR, or solver.
//!
//! Claims concern the supplied finite model and readout. The optional bytecode
//! adapter is an explicitly trusted extraction boundary, not a verified
//! compiler theorem. It extracts memory transition only, uses full 64-bit
//! symbolic words until a one-bit store masks them, and excludes all control,
//! calls, output, quotations, host operations, and heap operations.

use crate::ast::OpCode;
use crate::bytecode::{BytecodeProgram, Instruction};
use crate::core::BoundsPolicy;
use sha2::{Digest, Sha256};
use std::fmt;

pub const AFFINE_RECURRENCE_SEMANTICS_VERSION: u16 = 1;
pub const MAX_AFFINE_BITS: usize = 64;
const MAX_ADAPTER_INSTRUCTIONS: usize = 4096;
const MAX_ADAPTER_STACK: usize = 4096;
const MAX_ADAPTER_MEMORY: usize = 4096;

/// Row i is the coefficient mask for output bit i; offset is the XOR constant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineBooleanSystem {
    pub bits: u8,
    pub rows: Vec<u64>,
    pub offset: u64,
}

impl AffineBooleanSystem {
    pub fn validate(&self) -> Result<(), AffineRecurrenceError> {
        let bits = self.bits as usize;
        if bits == 0 || bits > MAX_AFFINE_BITS || self.rows.len() != bits {
            return Err(AffineRecurrenceError::InvalidModel(
                "dimension must be 1..64 with exactly one row per bit",
            ));
        }
        let mask = domain_mask(bits);
        if self.offset & !mask != 0 || self.rows.iter().any(|row| row & !mask != 0) {
            return Err(AffineRecurrenceError::InvalidModel(
                "coefficient or offset outside the declared domain",
            ));
        }
        Ok(())
    }

    pub fn apply(&self, state: u64) -> Result<u64, AffineRecurrenceError> {
        self.validate()?;
        if state & !domain_mask(self.bits as usize) != 0 {
            return Err(AffineRecurrenceError::InvalidModel(
                "input state outside the declared domain",
            ));
        }
        Ok(generator_evaluate(self, state))
    }
}

/// Readout on a recurrent memory state: parity(mask & x) XOR constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AffineReadout {
    pub mask: u64,
    pub constant: bool,
}

impl AffineReadout {
    pub fn evaluate(self, state: u64) -> bool {
        ((self.mask & state).count_ones() & 1 != 0) ^ self.constant
    }
}

/// Exact, including 2^64; no u64 overflow or enumeration is involved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AffineClassCount {
    pub exponent: u8,
}

impl AffineClassCount {
    pub fn as_u128(self) -> Option<u128> {
        if self.exponent <= 64 {
            Some(1u128 << self.exponent)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AffineReadoutOutcome {
    Uniform {
        value: bool,
    },
    /// Both are fixed states in image(P); zero/one name their readout values.
    Disagreement {
        zero_state: u64,
        one_state: u64,
    },
}

/// Untrusted bounded proof data until checked against an external model/query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineRecurrenceCertificate {
    pub semantics_version: u16,
    pub model: AffineBooleanSystem,
    pub query: AffineReadout,
    pub stabilized_power: AffineBooleanSystem,
    /// Unique reduced image basis ordered by increasing pivot bit.
    pub image_basis: Vec<u64>,
    pub rank: u8,
    pub class_count: AffineClassCount,
    pub outcome: AffineReadoutOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedAffineRecurrence {
    pub bits: u8,
    pub transient_bound: u8,
    pub rank: u8,
    pub class_count: AffineClassCount,
    pub outcome: AffineReadoutOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AffineRecurrenceError {
    InvalidModel(&'static str),
    InvalidQuery(&'static str),
    /// F^(n+1)(witness) differs from F^n(witness); no class claim is made.
    StabilizationFailed {
        witness: u64,
    },
    BindingMismatch(&'static str),
    InvalidCertificate(&'static str),
    UnsupportedBytecode(String),
    AdapterResourceLimit(&'static str),
}

impl fmt::Display for AffineRecurrenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidModel(reason) => write!(f, "invalid affine model: {reason}"),
            Self::InvalidQuery(reason) => write!(f, "invalid affine readout: {reason}"),
            Self::StabilizationFailed { witness } => write!(
                f,
                "affine transition does not stabilize by its dimension (witness {witness})"
            ),
            Self::BindingMismatch(field) => {
                write!(f, "affine certificate binding mismatch: {field}")
            }
            Self::InvalidCertificate(reason) => write!(f, "invalid affine certificate: {reason}"),
            Self::UnsupportedBytecode(reason) => write!(f, "unsupported affine bytecode: {reason}"),
            Self::AdapterResourceLimit(reason) => {
                write!(f, "affine adapter resource limit: {reason}")
            }
        }
    }
}

impl std::error::Error for AffineRecurrenceError {}

fn domain_mask(bits: usize) -> u64 {
    if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

fn validate_query(
    model: &AffineBooleanSystem,
    query: AffineReadout,
) -> Result<(), AffineRecurrenceError> {
    if query.mask & !domain_mask(model.bits as usize) != 0 {
        Err(AffineRecurrenceError::InvalidQuery(
            "readout mask outside the declared domain",
        ))
    } else {
        Ok(())
    }
}

fn generator_evaluate(model: &AffineBooleanSystem, state: u64) -> u64 {
    let mut output = model.offset;
    for (bit, &row) in model.rows.iter().enumerate() {
        output ^= u64::from((row & state).count_ones() & 1 != 0) << bit;
    }
    output
}

fn compose(after: &AffineBooleanSystem, before: &AffineBooleanSystem) -> AffineBooleanSystem {
    let mut rows = Vec::with_capacity(after.bits as usize);
    for &row in &after.rows {
        let mut coefficients = row;
        let mut composed = 0;
        while coefficients != 0 {
            let bit = coefficients.trailing_zeros() as usize;
            composed ^= before.rows[bit];
            coefficients &= coefficients - 1;
        }
        rows.push(composed);
    }
    AffineBooleanSystem {
        bits: after.bits,
        rows,
        offset: generator_evaluate(after, before.offset),
    }
}

fn generator_power(model: &AffineBooleanSystem) -> AffineBooleanSystem {
    let mut power = AffineBooleanSystem {
        bits: model.bits,
        rows: (0..model.bits).map(|bit| 1u64 << bit).collect(),
        offset: 0,
    };
    for _ in 0..model.bits {
        power = compose(model, &power);
    }
    power
}

fn columns(model: &AffineBooleanSystem) -> Vec<u64> {
    (0..model.bits)
        .map(|bit| {
            model
                .rows
                .iter()
                .enumerate()
                .fold(0, |column, (out, row)| column | (((row >> bit) & 1) << out))
        })
        .collect()
}

fn generator_basis(image_columns: &[u64], bits: usize) -> Vec<u64> {
    let mut pivots = [0u64; 64];
    for &column in image_columns {
        let mut value = column;
        for &pivot in &pivots[..bits] {
            if pivot != 0 && value & (1u64 << pivot.trailing_zeros()) != 0 {
                value ^= pivot;
            }
        }
        if value == 0 {
            continue;
        }
        let bit = value.trailing_zeros() as usize;
        for pivot in &mut pivots[..bits] {
            if *pivot & (1u64 << bit) != 0 {
                *pivot ^= value;
            }
        }
        pivots[bit] = value;
    }
    pivots[..bits]
        .iter()
        .copied()
        .filter(|&value| value != 0)
        .collect()
}

fn generator_outcome(
    power: &AffineBooleanSystem,
    basis: &[u64],
    query: AffineReadout,
) -> AffineReadoutOutcome {
    let value = query.evaluate(power.offset);
    match basis
        .iter()
        .copied()
        .find(|&vector| query.evaluate(vector) ^ query.constant)
    {
        None => AffineReadoutOutcome::Uniform { value },
        Some(vector) => {
            let other = power.offset ^ vector;
            if value {
                AffineReadoutOutcome::Disagreement {
                    zero_state: other,
                    one_state: power.offset,
                }
            } else {
                AffineReadoutOutcome::Disagreement {
                    zero_state: power.offset,
                    one_state: other,
                }
            }
        }
    }
}

pub fn certify_affine_recurrence(
    model: &AffineBooleanSystem,
    query: AffineReadout,
) -> Result<AffineRecurrenceCertificate, AffineRecurrenceError> {
    model.validate()?;
    validate_query(model, query)?;
    let stabilized_power = generator_power(model);
    if compose(model, &stabilized_power) != stabilized_power {
        let witness = std::iter::once(0)
            .chain((0..model.bits).map(|bit| 1u64 << bit))
            .find(|&state| {
                generator_evaluate(model, generator_evaluate(&stabilized_power, state))
                    != generator_evaluate(&stabilized_power, state)
            })
            .expect("different affine maps disagree on zero or a unit vector");
        return Err(AffineRecurrenceError::StabilizationFailed { witness });
    }
    let image_basis = generator_basis(&columns(&stabilized_power), model.bits as usize);
    let rank = image_basis.len() as u8;
    let outcome = generator_outcome(&stabilized_power, &image_basis, query);
    let certificate = AffineRecurrenceCertificate {
        semantics_version: AFFINE_RECURRENCE_SEMANTICS_VERSION,
        model: model.clone(),
        query,
        stabilized_power,
        image_basis,
        rank,
        class_count: AffineClassCount { exponent: rank },
        outcome,
    };
    check_affine_recurrence_certificate(&certificate, model, query)?;
    Ok(certificate)
}

// Checker arithmetic is independent of compose, generator_power, columns,
// generator_basis and generator_outcome above. It reconstructs affine maps
// using their action on zero and each unit vector.
fn checker_parity(mut word: u64) -> bool {
    let mut parity = false;
    while word != 0 {
        parity = !parity;
        word &= word - 1;
    }
    parity
}

fn checker_evaluate(model: &AffineBooleanSystem, state: u64) -> u64 {
    let mut result = 0;
    for bit in 0..model.bits as usize {
        let value = checker_parity(model.rows[bit] & state) ^ (((model.offset >> bit) & 1) != 0);
        if value {
            result |= 1u64 << bit;
        }
    }
    result
}

fn checker_power(model: &AffineBooleanSystem) -> (AffineBooleanSystem, Vec<u64>) {
    let mut images = Vec::with_capacity(model.bits as usize + 1);
    for seed in std::iter::once(0).chain((0..model.bits).map(|bit| 1u64 << bit)) {
        let mut state = seed;
        for _ in 0..model.bits {
            state = checker_evaluate(model, state);
        }
        images.push(state);
    }
    let offset = images[0];
    let image_columns: Vec<_> = images[1..].iter().map(|state| state ^ offset).collect();
    let mut rows = vec![0; model.bits as usize];
    for (input, &column) in image_columns.iter().enumerate() {
        for (output, row) in rows.iter_mut().enumerate() {
            if ((column >> output) & 1) != 0 {
                *row |= 1u64 << input;
            }
        }
    }
    (
        AffineBooleanSystem {
            bits: model.bits,
            rows,
            offset,
        },
        image_columns,
    )
}

fn checker_basis(mut vectors: Vec<u64>, bits: usize) -> Vec<u64> {
    let mut rank = 0;
    for bit in 0..bits {
        let Some(pivot) = (rank..vectors.len()).find(|&index| ((vectors[index] >> bit) & 1) != 0)
        else {
            continue;
        };
        vectors.swap(rank, pivot);
        let pivot = vectors[rank];
        for (index, vector) in vectors.iter_mut().enumerate() {
            if index != rank && ((*vector >> bit) & 1) != 0 {
                *vector ^= pivot;
            }
        }
        rank += 1;
    }
    vectors.truncate(rank);
    vectors
}

pub fn check_affine_recurrence_certificate(
    certificate: &AffineRecurrenceCertificate,
    expected_model: &AffineBooleanSystem,
    expected_query: AffineReadout,
) -> Result<VerifiedAffineRecurrence, AffineRecurrenceError> {
    if certificate.semantics_version != AFFINE_RECURRENCE_SEMANTICS_VERSION {
        return Err(AffineRecurrenceError::BindingMismatch("semantics version"));
    }
    if certificate.model != *expected_model {
        return Err(AffineRecurrenceError::BindingMismatch("exact model"));
    }
    if certificate.query != expected_query {
        return Err(AffineRecurrenceError::BindingMismatch(
            "exact readout query",
        ));
    }
    expected_model.validate()?;
    validate_query(expected_model, expected_query)?;
    if certificate.image_basis.len() > MAX_AFFINE_BITS {
        return Err(AffineRecurrenceError::InvalidCertificate(
            "image basis exceeds dimension ceiling",
        ));
    }
    let (power, image_columns) = checker_power(expected_model);
    if certificate.stabilized_power != power {
        return Err(AffineRecurrenceError::InvalidCertificate(
            "incorrect affine power",
        ));
    }
    // Affine maps are equal iff equal on zero and all unit vectors. The first
    // equality enforces stabilization, the second independently checks P^2=P.
    for seed in std::iter::once(0).chain((0..expected_model.bits).map(|bit| 1u64 << bit)) {
        let image = checker_evaluate(&power, seed);
        if checker_evaluate(expected_model, image) != image {
            return Err(AffineRecurrenceError::StabilizationFailed { witness: seed });
        }
        if checker_evaluate(&power, image) != image {
            return Err(AffineRecurrenceError::InvalidCertificate(
                "power is not idempotent",
            ));
        }
    }
    let basis = checker_basis(image_columns, expected_model.bits as usize);
    let rank = basis.len() as u8;
    if certificate.image_basis != basis
        || certificate.rank != rank
        || certificate.class_count.exponent != rank
    {
        return Err(AffineRecurrenceError::InvalidCertificate(
            "incorrect image basis, rank or class count",
        ));
    }
    let origin_value = checker_parity(expected_query.mask & power.offset) ^ expected_query.constant;
    let varying = basis
        .iter()
        .copied()
        .find(|&vector| checker_parity(expected_query.mask & vector));
    let outcome = match varying {
        None => AffineReadoutOutcome::Uniform {
            value: origin_value,
        },
        Some(vector) => {
            let other = power.offset ^ vector;
            let (zero_state, one_state) = if origin_value {
                (other, power.offset)
            } else {
                (power.offset, other)
            };
            if checker_evaluate(expected_model, zero_state) != zero_state
                || checker_evaluate(expected_model, one_state) != one_state
                || zero_state == one_state
                || (checker_parity(expected_query.mask & zero_state) ^ expected_query.constant)
                || !(checker_parity(expected_query.mask & one_state) ^ expected_query.constant)
            {
                return Err(AffineRecurrenceError::InvalidCertificate(
                    "readout witness is not two fixed states with opposite readouts",
                ));
            }
            AffineReadoutOutcome::Disagreement {
                zero_state,
                one_state,
            }
        }
    };
    if certificate.outcome != outcome {
        return Err(AffineRecurrenceError::InvalidCertificate(
            "incorrect readout conclusion or witnesses",
        ));
    }
    Ok(VerifiedAffineRecurrence {
        bits: expected_model.bits,
        transient_bound: expected_model.bits,
        rank,
        class_count: AffineClassCount { exponent: rank },
        outcome,
    })
}

/// Declared conservative adapter resource profile. No output/call/heap/host
/// operations exist in this profile; temporal depth is exactly one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineExtractionConfig {
    pub memory_cells: usize,
    pub memory_bounds: BoundsPolicy,
    pub max_instructions: u64,
    pub max_stack_depth: usize,
}

impl Default for AffineExtractionConfig {
    fn default() -> Self {
        Self {
            memory_cells: 64,
            memory_bounds: BoundsPolicy::Error,
            max_instructions: 4096,
            max_stack_depth: 256,
        }
    }
}

/// Exact program/config binding is separate from the model certificate. Calling
/// `check_affine_bytecode_binding` reruns the trusted extraction; the independent
/// recurrence checker proves only the resulting explicitly supplied model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineBytecodeModel {
    pub semantics_version: u16,
    pub program_sha256: [u8; 32],
    pub config: AffineExtractionConfig,
    pub model: AffineBooleanSystem,
}

// Each bit of a full word is an affine function of the n input bits. Literals
// and addresses retain all 64 bits; only PROPHECY masks the stored word.
#[derive(Clone)]
struct SymbolicWord {
    coefficients: [u64; 64],
    constant: u64,
}

impl SymbolicWord {
    fn literal(word: u64) -> Self {
        Self {
            coefficients: [0; 64],
            constant: word,
        }
    }
    fn xor(mut self, other: Self) -> Self {
        for (left, right) in self.coefficients.iter_mut().zip(other.coefficients) {
            *left ^= right;
        }
        self.constant ^= other.constant;
        self
    }
    fn masked(mut self) -> Self {
        self.coefficients[1..].fill(0);
        self.constant &= 1;
        self
    }
}

fn adapter_unsupported(reason: impl Into<String>) -> AffineRecurrenceError {
    AffineRecurrenceError::UnsupportedBytecode(reason.into())
}

fn adapter_address(
    word: SymbolicWord,
    cells: usize,
    bounds: BoundsPolicy,
) -> Result<usize, AffineRecurrenceError> {
    if word
        .coefficients
        .iter()
        .any(|&coefficient| coefficient != 0)
    {
        return Err(adapter_unsupported("memory address depends on input state"));
    }
    if word.constant < cells as u64 {
        return Ok(word.constant as usize);
    }
    match bounds {
        BoundsPolicy::Error => Err(adapter_unsupported("constant address is outside the scope")),
        BoundsPolicy::Wrap => Ok((word.constant % cells as u64) as usize),
        BoundsPolicy::Clamp => Ok(cells - 1),
    }
}

pub fn extract_affine_bytecode(
    program: &BytecodeProgram,
    config: &AffineExtractionConfig,
) -> Result<AffineBytecodeModel, AffineRecurrenceError> {
    if config.memory_cells == 0
        || config.memory_cells > MAX_ADAPTER_MEMORY
        || config.max_stack_depth > MAX_ADAPTER_STACK
    {
        return Err(AffineRecurrenceError::AdapterResourceLimit(
            "memory/stack configuration exceeds adapter ceiling",
        ));
    }
    if program.instructions.len() > MAX_ADAPTER_INSTRUCTIONS
        || program.source_map.len() > MAX_ADAPTER_INSTRUCTIONS
    {
        return Err(AffineRecurrenceError::AdapterResourceLimit(
            "instruction/source-map count exceeds adapter ceiling",
        ));
    }
    // Production VM linking requires a host even for an unused declaration.
    // That environmental setup is outside this closed finite extraction.
    if !program.foreigns.is_empty() {
        return Err(adapter_unsupported(
            "foreign declarations require excluded host linking",
        ));
    }
    if program.main.start >= program.main.end
        || program.main.end as usize > program.instructions.len()
        || program.main.end - program.main.start < 3
    {
        return Err(adapter_unsupported(
            "main must be a whole control-free scoped unit",
        ));
    }
    if (program.main.end - program.main.start) as u64 > config.max_instructions {
        return Err(AffineRecurrenceError::AdapterResourceLimit(
            "fetched instructions exceed configured gas",
        ));
    }
    // Canonical serialization structurally validates every unit, including
    // unused prelude procedures/quotes. Main cannot call
    // them in this profile; their exact bytes remain bound by the digest.
    let bytes = program
        .to_bytes()
        .map_err(|error| adapter_unsupported(error.to_string()))?;
    // This production admission gate belongs to trusted extraction only. The
    // independent model certificate checker never calls a bytecode verifier.
    crate::bytecode_verifier::verify_default(program)
        .map_err(|error| adapter_unsupported(error.to_string()))?;
    let start = program.main.start;
    let exit = program.main.end - 2;
    let cells = match program.instructions[start as usize] {
        Instruction::TemporalEnter {
            base: 0,
            size,
            cell_bits: 1,
            exit_target,
        } if exit_target == exit && (1..=64).contains(&size) => size as usize,
        _ => {
            return Err(adapter_unsupported(
                "main must begin with a base-zero 1..64-cell one-bit scope",
            ))
        }
    };
    if cells > config.memory_cells {
        return Err(adapter_unsupported("scope exceeds configured memory"));
    }
    if program.instructions[exit as usize]
        != (Instruction::TemporalExit {
            enter_target: start,
        })
        || program.instructions[exit as usize + 1] != Instruction::Return
    {
        return Err(adapter_unsupported(
            "main must end with paired temporal EXIT then RETURN",
        ));
    }
    let mut present = vec![SymbolicWord::literal(0); cells];
    let mut stack: Vec<SymbolicWord> = Vec::new();
    for instruction in &program.instructions[start as usize + 1..exit as usize] {
        let need = |stack: &[SymbolicWord], count| {
            if stack.len() < count {
                Err(adapter_unsupported("symbolic operand stack underflow"))
            } else {
                Ok(())
            }
        };
        match *instruction {
            Instruction::PushWord(word) => stack.push(SymbolicWord::literal(word)),
            Instruction::Primitive(OpCode::Nop) => {}
            Instruction::Primitive(OpCode::Pop) => {
                need(&stack, 1)?;
                stack.pop();
            }
            Instruction::Primitive(OpCode::Dup) => {
                need(&stack, 1)?;
                stack.push(stack[stack.len() - 1].clone());
            }
            Instruction::Primitive(OpCode::Swap) => {
                need(&stack, 2)?;
                let n = stack.len();
                stack.swap(n - 1, n - 2);
            }
            Instruction::Primitive(OpCode::Over) => {
                need(&stack, 2)?;
                stack.push(stack[stack.len() - 2].clone());
            }
            Instruction::Primitive(OpCode::Rot) => {
                need(&stack, 3)?;
                let word = stack.remove(stack.len() - 3);
                stack.push(word);
            }
            Instruction::Primitive(OpCode::Xor) => {
                need(&stack, 2)?;
                let right = stack.pop().unwrap();
                let left = stack.pop().unwrap();
                stack.push(left.xor(right));
            }
            Instruction::Primitive(OpCode::Oracle | OpCode::PresentRead) => {
                need(&stack, 1)?;
                let address = adapter_address(stack.pop().unwrap(), cells, config.memory_bounds)?;
                if *instruction == Instruction::Primitive(OpCode::Oracle) {
                    let mut word = SymbolicWord::literal(0);
                    word.coefficients[0] = 1u64 << address;
                    stack.push(word);
                } else {
                    stack.push(present[address].clone());
                }
            }
            Instruction::Primitive(OpCode::Prophecy) => {
                need(&stack, 2)?;
                let address = adapter_address(stack.pop().unwrap(), cells, config.memory_bounds)?;
                present[address] = stack.pop().unwrap().masked();
            }
            other => {
                return Err(adapter_unsupported(format!(
                    "instruction {other:?} is not an admitted linear primitive"
                )))
            }
        }
        if stack.len() > config.max_stack_depth {
            return Err(AffineRecurrenceError::AdapterResourceLimit(
                "symbolic stack exceeds configured limit",
            ));
        }
    }
    let model = AffineBooleanSystem {
        bits: cells as u8,
        rows: present.iter().map(|word| word.coefficients[0]).collect(),
        offset: present.iter().enumerate().fold(0, |offset, (bit, word)| {
            offset | ((word.constant & 1) << bit)
        }),
    };
    model.validate()?;
    Ok(AffineBytecodeModel {
        semantics_version: AFFINE_RECURRENCE_SEMANTICS_VERSION,
        program_sha256: Sha256::digest(bytes).into(),
        config: config.clone(),
        model,
    })
}

pub fn check_affine_bytecode_binding(
    extracted: &AffineBytecodeModel,
    expected_program: &BytecodeProgram,
    expected_config: &AffineExtractionConfig,
) -> Result<(), AffineRecurrenceError> {
    if extracted.semantics_version != AFFINE_RECURRENCE_SEMANTICS_VERSION {
        return Err(AffineRecurrenceError::BindingMismatch(
            "extraction semantics version",
        ));
    }
    if extracted.config != *expected_config {
        return Err(AffineRecurrenceError::BindingMismatch(
            "extraction resource configuration",
        ));
    }
    let expected = extract_affine_bytecode(expected_program, expected_config)?;
    if extracted.program_sha256 != expected.program_sha256 {
        return Err(AffineRecurrenceError::BindingMismatch(
            "canonical bytecode SHA-256",
        ));
    }
    if extracted.model != expected.model {
        return Err(AffineRecurrenceError::BindingMismatch(
            "extracted transition model",
        ));
    }
    Ok(())
}
