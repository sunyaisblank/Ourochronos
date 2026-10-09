//! Independently checked finite no-fixed-point certificates.
//!
//! This deliberately small profile admits one whole-main TEMPORAL scope at
//! base zero, at most twelve state bits, forward IFs, and acyclic procedures.
//! Every admitted memory write is inside that scope and masks its stored word;
//! fresh present memory starts at zero and the only instruction after scope
//! exit is main RETURN. Consequently every full-word point fixed state has
//! zero outside the scope and masked words inside it. Enumerating that finite
//! domain is complete, rather than a bounded search over arbitrary words.
//!
//! Generation uses the production VM. Checking does not: `interpret` below is
//! a separate u64 stack/store/control interpreter, with no VM, IR, solver, or
//! shared arithmetic-dispatch calls. The checker recomputes all transitions,
//! all terminal observations, and the absence of a fixed row. SHA-256 binds
//! canonical program bytes; a checksum detects corruption, not authenticity.
//!
//! The trusted boundary includes this interpreter/admission code, Rust u64
//! semantics, canonical bytecode serialization, SHA-256, and resource-profile
//! assumptions. This is no compiler-correctness or host-allocation theorem.
//! Provenance is projected away numerically. Numeric OUTPUT is admitted only
//! with a conservative retained-byte guarantee: OutputItem::Val costs 64 plus
//! 32 bytes per explicit dependency (core/value.rs), and all dependencies are
//! among the at most twelve scoped ORACLE addresses. Saturation reduces that
//! charge. No physical allocator or provenance representation is simulated.

use crate::ast::OpCode;
use crate::bytecode::{BytecodeProgram, CodeRange, Instruction};
use crate::bytecode_vm::{BytecodeVm, BytecodeVmConfig, BytecodeVmStatus};
use crate::core::{BoundsPolicy, OutputItem, PagedMemory, Value};
use sha2::{Digest, Sha256};
use std::fmt;

const MAGIC: &[u8; 8] = b"OUROFP\0\0";
pub const FINITE_PROOF_FORMAT_VERSION: u16 = 1;
/// Increment when admitted instruction semantics or resource assumptions change.
pub const FINITE_PROOF_SEMANTICS_VERSION: u16 = 1;
pub const MAX_FINITE_STATE_BITS: usize = 12;
pub const MAX_FINITE_STATES: usize = 1 << MAX_FINITE_STATE_BITS;
pub const MAX_FINITE_PROOF_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_FINITE_CHECK_WORK: u64 = 8 * 1024 * 1024;
const MAX_CODE: usize = 4096;
const MAX_PROCEDURES: usize = 64;
const MAX_MEMORY: usize = 4096;
const MAX_OBSERVATION_ITEMS: usize = 4096;
const MAX_PROGRAM_BYTES: usize = 1024 * 1024;

/// The only supported query; unknown serialized query tags are rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiniteProofQuery {
    /// No normal numeric point fixed state under the exact bound/resource profile.
    NoFixedPoint,
}

/// Exact resource profile. All input/host/effect/heap capabilities are denied.
/// Bounds refer first to the active scope, then to full configured memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiniteProofConfig {
    pub memory_cells: usize,
    pub memory_bounds: BoundsPolicy,
    pub max_instructions: u64,
    pub max_call_depth: usize,
    pub max_stack_depth: usize,
    pub max_temporal_depth: usize,
    pub max_output_items: usize,
    pub max_output_bytes: usize,
}

impl Default for FiniteProofConfig {
    fn default() -> Self {
        Self {
            memory_cells: 16,
            memory_bounds: BoundsPolicy::Error,
            max_instructions: 1024,
            max_call_depth: 64,
            max_stack_depth: 256,
            max_temporal_depth: 1,
            max_output_items: 64,
            max_output_bytes: 32 * 1024,
        }
    }
}

impl FiniteProofConfig {
    /// Production settings used by the generator, with excluded capabilities
    /// denied. The checker does not call this conversion or the production VM.
    pub fn vm_config(&self) -> BytecodeVmConfig {
        BytecodeVmConfig {
            max_instructions: self.max_instructions,
            max_call_depth: self.max_call_depth,
            max_stack_depth: self.max_stack_depth,
            max_temporal_depth: self.max_temporal_depth,
            max_output_items: self.max_output_items,
            max_output_bytes: self.max_output_bytes,
            memory_bounds: self.memory_bounds,
            max_collections: 0,
            max_collection_items: 0,
            max_dynamic_bytes: 0,
            max_file_snapshots: 0,
            max_file_snapshot_bytes: 0,
            max_endpoint_tape_bytes: 0,
            max_process_result_bytes: 0,
            max_effects: 0,
            max_effect_bytes: 0,
            ..BytecodeVmConfig::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FiniteProofScope {
    pub cells: usize,
    pub cell_bits: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiniteProofTerminal {
    Finished,
    Halted,
}

/// Row index is the implicit input, encoded cell-zero first in little-endian
/// bit fields. No input supplied by a certificate is trusted as domain coverage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiniteTransition {
    pub present: Vec<u64>,
    pub stack: Vec<u64>,
    pub output: Vec<u64>,
    pub terminal: FiniteProofTerminal,
    /// All fetched records count, including ENTER, EXIT, CALL, IF and RETURN.
    pub instructions_executed: u64,
}

/// Mutable data is untrusted until `check_certificate` succeeds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiniteNoFixedPointCertificate {
    pub semantics_version: u16,
    pub program_sha256: [u8; 32],
    pub config: FiniteProofConfig,
    pub query: FiniteProofQuery,
    pub scope: FiniteProofScope,
    pub rows: Vec<FiniteTransition>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedFiniteProof {
    pub program_sha256: [u8; 32],
    pub enumerated_states: usize,
    pub scope: FiniteProofScope,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FiniteProofError {
    Unsupported(String),
    ResourceLimit(&'static str),
    BindingMismatch(&'static str),
    InvalidEncoding(&'static str),
    CorruptEncoding,
    Execution { row: usize, reason: String },
    ObservationMismatch { row: usize },
    FixedPointFound { row: usize },
}

impl fmt::Display for FiniteProofError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(reason) => write!(f, "unsupported finite proof fragment: {reason}"),
            Self::ResourceLimit(reason) => write!(f, "finite proof resource limit: {reason}"),
            Self::BindingMismatch(field) => write!(f, "finite proof binding mismatch: {field}"),
            Self::InvalidEncoding(reason) => write!(f, "invalid finite proof encoding: {reason}"),
            Self::CorruptEncoding => write!(f, "finite proof checksum mismatch"),
            Self::Execution { row, reason } => write!(f, "finite proof row {row} failed: {reason}"),
            Self::ObservationMismatch { row } => {
                write!(f, "finite proof row {row} observation mismatch")
            }
            Self::FixedPointFound { row } => write!(f, "finite proof row {row} is a fixed point"),
        }
    }
}

impl std::error::Error for FiniteProofError {}

fn unsupported(reason: impl Into<String>) -> FiniteProofError {
    FiniteProofError::Unsupported(reason.into())
}

fn validate_config(config: &FiniteProofConfig) -> Result<(), FiniteProofError> {
    if config.memory_cells == 0 || config.memory_cells > MAX_MEMORY {
        return Err(FiniteProofError::ResourceLimit(
            "memory must contain 1..4096 cells",
        ));
    }
    if config.max_call_depth > MAX_PROCEDURES
        || config.max_stack_depth > MAX_OBSERVATION_ITEMS
        || config.max_output_items > MAX_OBSERVATION_ITEMS
        || config.max_temporal_depth > 1
        || config.max_output_bytes > MAX_FINITE_PROOF_BYTES
        || config.max_instructions > MAX_FINITE_CHECK_WORK
    {
        return Err(FiniteProofError::ResourceLimit(
            "resource configuration exceeds profile ceilings",
        ));
    }
    Ok(())
}

fn state_count(scope: FiniteProofScope) -> Result<usize, FiniteProofError> {
    let bits = scope
        .cells
        .checked_mul(scope.cell_bits as usize)
        .ok_or(FiniteProofError::ResourceLimit("scope bit count overflows"))?;
    if scope.cells == 0 || scope.cell_bits == 0 || bits > MAX_FINITE_STATE_BITS {
        return Err(unsupported("scope must contain 1..12 total state bits"));
    }
    Ok(1usize << bits)
}

fn profile_limits(
    config: &FiniteProofConfig,
    scope: FiniteProofScope,
) -> Result<usize, FiniteProofError> {
    validate_config(config)?;
    let states = state_count(scope)?;
    if scope.cells > config.memory_cells {
        return Err(unsupported("scope exceeds configured memory"));
    }
    // Include generator frame scans and row execution in the work ceiling.
    let work = config
        .max_instructions
        .checked_add(config.memory_cells as u64)
        .and_then(|per_row| per_row.checked_mul(states as u64))
        .ok_or(FiniteProofError::ResourceLimit("work estimate overflows"))?;
    if work > MAX_FINITE_CHECK_WORK {
        return Err(FiniteProofError::ResourceLimit(
            "aggregate row work exceeds ceiling",
        ));
    }
    let per_row = 32usize
        .checked_add(scope.cells * 8)
        .and_then(|n| n.checked_add(config.max_stack_depth * 8))
        .and_then(|n| n.checked_add(config.max_output_items * 8))
        .ok_or(FiniteProofError::ResourceLimit(
            "certificate estimate overflows",
        ))?;
    if states
        .checked_mul(per_row)
        .and_then(|n| n.checked_add(256))
        .is_none_or(|n| n > MAX_FINITE_PROOF_BYTES)
    {
        return Err(FiniteProofError::ResourceLimit(
            "worst-case certificate exceeds byte ceiling",
        ));
    }
    Ok(states)
}

fn admitted_opcode(op: OpCode) -> bool {
    matches!(
        op,
        OpCode::Nop
            | OpCode::Halt
            | OpCode::Pop
            | OpCode::Dup
            | OpCode::Swap
            | OpCode::Over
            | OpCode::Rot
            | OpCode::Depth
            | OpCode::Pick
            | OpCode::Roll
            | OpCode::Reverse
            | OpCode::Add
            | OpCode::Sub
            | OpCode::Mul
            | OpCode::Div
            | OpCode::Mod
            | OpCode::Neg
            | OpCode::Abs
            | OpCode::Min
            | OpCode::Max
            | OpCode::Sign
            | OpCode::Not
            | OpCode::And
            | OpCode::Or
            | OpCode::Xor
            | OpCode::Shl
            | OpCode::Shr
            | OpCode::Eq
            | OpCode::Neq
            | OpCode::Lt
            | OpCode::Gt
            | OpCode::Lte
            | OpCode::Gte
            | OpCode::Slt
            | OpCode::Sgt
            | OpCode::Slte
            | OpCode::Sgte
            | OpCode::Oracle
            | OpCode::Prophecy
            | OpCode::PresentRead
            | OpCode::Output
    )
}

struct Admitted {
    scope: FiniteProofScope,
    states: usize,
    digest: [u8; 32],
}

/// Admission checks structure for every unit and semantics for direct-call
/// reachable units. Dynamic quotation calls are excluded, so this reachability
/// boundary cannot hide an executable capability in an unused procedure.
/// Bytecode's serializer checks artifact structure; this admission independently
/// confines all controls/stores and rejects every cycle and capability outside
/// the proof profile. It does not rely on a HIR/type/solver acceptance label.
fn admit(
    program: &BytecodeProgram,
    config: &FiniteProofConfig,
) -> Result<Admitted, FiniteProofError> {
    validate_config(config)?;
    if program.instructions.len() > MAX_CODE
        || program.procedures.len() > MAX_PROCEDURES
        || program.source_map.len() > MAX_CODE
    {
        return Err(FiniteProofError::ResourceLimit(
            "program exceeds finite profile ceilings",
        ));
    }
    if !program.quotations.is_empty() || !program.foreigns.is_empty() {
        return Err(unsupported(
            "quotations and foreign declarations are excluded",
        ));
    }
    let main = program.main;
    if main.end <= main.start
        || main.end as usize > program.instructions.len()
        || main.end - main.start < 3
    {
        return Err(unsupported(
            "main must be one whole temporal scope followed by RETURN",
        ));
    }
    let exit = main.end - 2;
    let (cells, cell_bits) = match program.instructions[main.start as usize] {
        Instruction::TemporalEnter {
            base: 0,
            size,
            cell_bits,
            exit_target,
        } if exit_target == exit => (
            usize::try_from(size).map_err(|_| unsupported("scope size does not fit"))?,
            cell_bits,
        ),
        _ => {
            return Err(unsupported(
                "main must begin with a base-zero temporal scope",
            ))
        }
    };
    if program.instructions[exit as usize]
        != (Instruction::TemporalExit {
            enter_target: main.start,
        })
        || program.instructions[(main.end - 1) as usize] != Instruction::Return
    {
        return Err(unsupported("main must end with paired EXIT then RETURN"));
    }
    let scope = FiniteProofScope { cells, cell_bits };
    let states = profile_limits(config, scope)?;
    if config.max_temporal_depth == 0 {
        return Err(FiniteProofError::ResourceLimit(
            "temporal depth does not permit the scope",
        ));
    }
    let mut owners = vec![None; program.instructions.len()];
    let mut ranges = vec![main];
    ranges.extend(program.procedures.iter().map(|p| p.range));
    let mut calls = vec![Vec::new(); ranges.len()];
    let mut has_output = false;
    for (unit, range) in ranges.iter().copied().enumerate() {
        if range.start >= range.end
            || range.end as usize > owners.len()
            || program.instructions[(range.end - 1) as usize] != Instruction::Return
        {
            return Err(unsupported(
                "invalid callable range or missing final RETURN",
            ));
        }
        if unit > 0 && program.procedures[unit - 1].id.index() != unit - 1 {
            return Err(unsupported("procedure identities are not canonical"));
        }
        for pc in range.start..range.end {
            if owners[pc as usize].replace(unit).is_some() {
                return Err(unsupported("overlapping callable ranges"));
            }
            if let Instruction::CallProcedure(id) = program.instructions[pc as usize] {
                if id.index() >= program.procedures.len() {
                    return Err(unsupported("invalid procedure call identity"));
                }
                calls[unit].push(id.index() + 1);
            }
        }
    }
    if owners.iter().any(Option::is_none) {
        return Err(unsupported("instructions outside declared units"));
    }
    let mut reachable = vec![false; ranges.len()];
    let mut pending = vec![0];
    while let Some(unit) = pending.pop() {
        if reachable[unit] {
            continue;
        }
        reachable[unit] = true;
        pending.extend(calls[unit].iter().copied());
    }
    for (unit, range) in ranges.iter().copied().enumerate() {
        if !reachable[unit] {
            continue;
        }
        for pc in range.start..range.end {
            match program.instructions[pc as usize] {
                Instruction::Primitive(op) if admitted_opcode(op) => {
                    has_output |= op == OpCode::Output
                }
                Instruction::PushWord(_) => {}
                Instruction::CallProcedure(_) => {}
                Instruction::IfFalse {
                    else_target,
                    end_target,
                    ..
                } => {
                    let end_limit = if unit == 0 { exit } else { range.end - 1 };
                    if else_target <= pc || else_target > end_target || end_target > end_limit {
                        return Err(unsupported("IF escapes its scoped body or is not forward"));
                    }
                }
                Instruction::Jump { target } => {
                    let end_limit = if unit == 0 { exit } else { range.end - 1 };
                    if target <= pc || target > end_limit {
                        return Err(unsupported(
                            "jump escapes its scoped body or is not forward",
                        ));
                    }
                }
                Instruction::TemporalEnter { .. } if pc == main.start => {}
                Instruction::TemporalExit { .. } if pc == exit => {}
                Instruction::Return if pc == range.end - 1 => {}
                other => return Err(unsupported(format!("instruction {pc}: {other:?}"))),
            }
        }
    }
    fn visit(unit: usize, calls: &[Vec<usize>], colors: &mut [u8]) -> bool {
        if colors[unit] == 1 {
            return false;
        }
        if colors[unit] == 2 {
            return true;
        }
        colors[unit] = 1;
        for &callee in &calls[unit] {
            if !visit(callee, calls, colors) {
                return false;
            }
        }
        colors[unit] = 2;
        true
    }
    let mut colors = vec![0; ranges.len()];
    if !visit(0, &calls, &mut colors) {
        return Err(unsupported("recursive procedure calls"));
    }
    if has_output {
        let charge = 64 + 32 * scope.cells;
        let required =
            config
                .max_output_items
                .checked_mul(charge)
                .ok_or(FiniteProofError::ResourceLimit(
                    "numeric output byte bound overflows",
                ))?;
        if config.max_output_bytes < required {
            return Err(unsupported(
                "OUTPUT byte limit must cover all scoped dependencies for every allowed item",
            ));
        }
    }
    let bytes = program
        .to_bytes()
        .map_err(|error| unsupported(error.to_string()))?;
    if bytes.len() > MAX_PROGRAM_BYTES {
        return Err(FiniteProofError::ResourceLimit(
            "canonical program exceeds byte ceiling",
        ));
    }
    Ok(Admitted {
        scope,
        states,
        digest: Sha256::digest(&bytes).into(),
    })
}

fn input_at(row: usize, scope: FiniteProofScope) -> Vec<u64> {
    let mask = (1u64 << scope.cell_bits) - 1;
    (0..scope.cells)
        .map(|cell| ((row as u64) >> (cell * scope.cell_bits as usize)) & mask)
        .collect()
}

/// Generate a complete table, refusing every fixed row, unsupported instruction,
/// gas/resource failure, execution error or non-normal terminal status.
pub fn generate_certificate(
    program: &BytecodeProgram,
    config: &FiniteProofConfig,
    query: FiniteProofQuery,
) -> Result<FiniteNoFixedPointCertificate, FiniteProofError> {
    let admitted = admit(program, config)?;
    let vm = BytecodeVm::with_config(config.vm_config());
    let mut rows = Vec::new();
    rows.try_reserve(admitted.states)
        .map_err(|_| FiniteProofError::ResourceLimit("table allocation"))?;
    for row in 0..admitted.states {
        let input = input_at(row, admitted.scope);
        let fail = |reason: String| FiniteProofError::Execution { row, reason };
        let mut anamnesis =
            PagedMemory::with_size(config.memory_cells).map_err(|e| fail(e.to_string()))?;
        for (address, &word) in input.iter().enumerate() {
            anamnesis
                .write(address as u64, Value::new(word))
                .map_err(|e| fail(e.to_string()))?;
        }
        let execution = vm
            .run(program, &anamnesis)
            .map_err(|e| fail(e.to_string()))?;
        let terminal = match execution.status {
            BytecodeVmStatus::Finished => FiniteProofTerminal::Finished,
            BytecodeVmStatus::Halted => FiniteProofTerminal::Halted,
            BytecodeVmStatus::Paradox => {
                return Err(fail("PARADOX is not normal execution".into()))
            }
        };
        // Independently test the static frame argument on generated VM rows.
        for address in admitted.scope.cells..config.memory_cells {
            if execution
                .present
                .get(address as u64)
                .is_none_or(|v| v.val != 0)
            {
                return Err(fail("production VM violated outside-zero frame".into()));
            }
        }
        let present: Vec<_> = (0..admitted.scope.cells)
            .map(|address| {
                execution
                    .present
                    .get(address as u64)
                    .map(|v| v.val)
                    .ok_or_else(|| fail("production VM omitted scoped cell".into()))
            })
            .collect::<Result<_, _>>()?;
        if present == input {
            return Err(FiniteProofError::FixedPointFound { row });
        }
        let output = execution
            .output
            .iter()
            .map(|item| match item {
                OutputItem::Val(v) => Ok(v.val),
                OutputItem::Char(_) => {
                    Err(fail("non-numeric output is outside the profile".into()))
                }
            })
            .collect::<Result<_, _>>()?;
        rows.push(FiniteTransition {
            present,
            stack: execution.stack.iter().map(|v| v.val).collect(),
            output,
            terminal,
            instructions_executed: execution.instructions_executed,
        });
    }
    let certificate = FiniteNoFixedPointCertificate {
        semantics_version: FINITE_PROOF_SEMANTICS_VERSION,
        program_sha256: admitted.digest,
        config: config.clone(),
        query,
        scope: admitted.scope,
        rows,
    };
    // Generation may use the VM; it does not confer authority on VM output.
    check_certificate(&certificate, program, config, query)?;
    Ok(certificate)
}

/// Check against caller-supplied identity/config/query. Every transition is
/// recomputed here; neither a stored table nor its checksum proves anything.
pub fn check_certificate(
    certificate: &FiniteNoFixedPointCertificate,
    expected_program: &BytecodeProgram,
    expected_config: &FiniteProofConfig,
    expected_query: FiniteProofQuery,
) -> Result<VerifiedFiniteProof, FiniteProofError> {
    if certificate.semantics_version != FINITE_PROOF_SEMANTICS_VERSION {
        return Err(FiniteProofError::BindingMismatch("semantics version"));
    }
    if certificate.config != *expected_config {
        return Err(FiniteProofError::BindingMismatch("resource configuration"));
    }
    if certificate.query != expected_query {
        return Err(FiniteProofError::BindingMismatch("query"));
    }
    let admitted = admit(expected_program, expected_config)?;
    if certificate.program_sha256 != admitted.digest {
        return Err(FiniteProofError::BindingMismatch(
            "canonical program SHA-256",
        ));
    }
    if certificate.scope != admitted.scope {
        return Err(FiniteProofError::BindingMismatch("scope"));
    }
    validate_rows(certificate, admitted.states)?;
    for (row, claimed) in certificate.rows.iter().enumerate() {
        let input = input_at(row, admitted.scope);
        let computed = interpret(expected_program, expected_config, admitted.scope, &input)
            .map_err(|reason| FiniteProofError::Execution { row, reason })?;
        if *claimed != computed {
            return Err(FiniteProofError::ObservationMismatch { row });
        }
        if computed.present == input {
            return Err(FiniteProofError::FixedPointFound { row });
        }
    }
    Ok(VerifiedFiniteProof {
        program_sha256: admitted.digest,
        enumerated_states: admitted.states,
        scope: admitted.scope,
    })
}

fn validate_rows(c: &FiniteNoFixedPointCertificate, states: usize) -> Result<(), FiniteProofError> {
    if c.rows.len() != states {
        return Err(FiniteProofError::InvalidEncoding("incomplete domain table"));
    }
    let mask = (1u64 << c.scope.cell_bits) - 1;
    let mut bytes = 256usize;
    for row in &c.rows {
        if row.present.len() != c.scope.cells
            || row.present.iter().any(|&v| v > mask)
            || row.stack.len() > c.config.max_stack_depth
            || row.output.len() > c.config.max_output_items
            || row.instructions_executed > c.config.max_instructions
        {
            return Err(FiniteProofError::InvalidEncoding(
                "row exceeds its declared profile",
            ));
        }
        bytes = bytes
            .checked_add(32 + 8 * (row.present.len() + row.stack.len() + row.output.len()))
            .ok_or(FiniteProofError::ResourceLimit(
                "certificate byte count overflows",
            ))?;
        if bytes > MAX_FINITE_PROOF_BYTES {
            return Err(FiniteProofError::ResourceLimit("certificate byte ceiling"));
        }
    }
    Ok(())
}

// The checker interpreter begins here. It uses only bytecode data and local
// u64 operations. In particular, Value arithmetic, VM execution/preparation,
// temporal IR extraction, and solver dispatch are never called.
struct WordMachine<'a> {
    config: &'a FiniteProofConfig,
    scope: FiniteProofScope,
    stack: Vec<u64>,
    present: Vec<u64>,
    output: Vec<u64>,
    active_scope: bool,
}

impl WordMachine<'_> {
    fn need(&self, n: usize) -> Result<(), String> {
        if self.stack.len() < n {
            Err("operand stack underflow".into())
        } else {
            Ok(())
        }
    }
    fn pop(&mut self) -> Result<u64, String> {
        self.stack
            .pop()
            .ok_or_else(|| "operand stack underflow".into())
    }
    fn push(&mut self, word: u64) -> Result<(), String> {
        if self.stack.len() >= self.config.max_stack_depth {
            return Err("operand stack limit".into());
        }
        self.stack
            .try_reserve(1)
            .map_err(|_| "checker stack allocation".to_string())?;
        self.stack.push(word);
        Ok(())
    }
    fn address(&self, raw: u64) -> Result<usize, String> {
        if !self.active_scope {
            return Err("memory access outside admitted scope".into());
        }
        if raw < self.scope.cells as u64 {
            return Ok(raw as usize);
        }
        match self.config.memory_bounds {
            BoundsPolicy::Error => Err("scoped address out of bounds".into()),
            BoundsPolicy::Wrap => Ok((raw % self.scope.cells as u64) as usize),
            BoundsPolicy::Clamp => Ok(self.scope.cells - 1),
        }
    }
    fn unary(&mut self, f: impl FnOnce(u64) -> u64) -> Result<(), String> {
        let word = self.pop()?;
        self.push(f(word))
    }
    fn binary(&mut self, f: impl FnOnce(u64, u64) -> u64) -> Result<(), String> {
        self.need(2)?;
        let b = self.pop()?;
        let a = self.pop()?;
        self.push(f(a, b))
    }
    fn primitive(&mut self, op: OpCode, input: &[u64]) -> Result<(), String> {
        match op {
            OpCode::Nop => Ok(()),
            OpCode::Pop => {
                self.pop()?;
                Ok(())
            }
            OpCode::Dup => {
                self.need(1)?;
                self.push(self.stack[self.stack.len() - 1])
            }
            OpCode::Swap => {
                self.need(2)?;
                let n = self.stack.len();
                self.stack.swap(n - 1, n - 2);
                Ok(())
            }
            OpCode::Over => {
                self.need(2)?;
                self.push(self.stack[self.stack.len() - 2])
            }
            OpCode::Rot => {
                self.need(3)?;
                let word = self.stack.remove(self.stack.len() - 3);
                self.push(word)
            }
            OpCode::Depth => self.push(self.stack.len() as u64),
            OpCode::Pick | OpCode::Roll => {
                let depth = self.pop()?;
                if depth >= self.stack.len() as u64 {
                    return Err("stack selection underflow".into());
                }
                let index = self.stack.len() - 1 - depth as usize;
                let word = if op == OpCode::Pick {
                    self.stack[index]
                } else {
                    self.stack.remove(index)
                };
                self.push(word)
            }
            OpCode::Reverse => {
                let count = self.pop()?;
                if count > self.stack.len() as u64 {
                    return Err("stack reversal underflow".into());
                }
                let start = self.stack.len() - count as usize;
                self.stack[start..].reverse();
                Ok(())
            }
            OpCode::Add => self.binary(u64::wrapping_add),
            OpCode::Sub => self.binary(u64::wrapping_sub),
            OpCode::Mul => self.binary(u64::wrapping_mul),
            OpCode::Div => self.binary(|a, b| if b == 0 { 0 } else { a / b }),
            OpCode::Mod => self.binary(|a, b| if b == 0 { 0 } else { a % b }),
            OpCode::Neg => self.unary(u64::wrapping_neg),
            OpCode::Abs => self.unary(|a| (a as i64).wrapping_abs() as u64),
            OpCode::Sign => self.unary(|a| (a as i64).signum() as u64),
            OpCode::Min => self.binary(u64::min),
            OpCode::Max => self.binary(u64::max),
            OpCode::Not => self.unary(|a| u64::from(a == 0)),
            OpCode::And => self.binary(|a, b| a & b),
            OpCode::Or => self.binary(|a, b| a | b),
            OpCode::Xor => self.binary(|a, b| a ^ b),
            OpCode::Shl => self.binary(|a, b| a.wrapping_shl((b % 64) as u32)),
            OpCode::Shr => self.binary(|a, b| a.wrapping_shr((b % 64) as u32)),
            OpCode::Eq => self.binary(|a, b| u64::from(a == b)),
            OpCode::Neq => self.binary(|a, b| u64::from(a != b)),
            OpCode::Lt => self.binary(|a, b| u64::from(a < b)),
            OpCode::Gt => self.binary(|a, b| u64::from(a > b)),
            OpCode::Lte => self.binary(|a, b| u64::from(a <= b)),
            OpCode::Gte => self.binary(|a, b| u64::from(a >= b)),
            OpCode::Slt => self.binary(|a, b| u64::from((a as i64) < (b as i64))),
            OpCode::Sgt => self.binary(|a, b| u64::from((a as i64) > (b as i64))),
            OpCode::Slte => self.binary(|a, b| u64::from((a as i64) <= (b as i64))),
            OpCode::Sgte => self.binary(|a, b| u64::from((a as i64) >= (b as i64))),
            OpCode::Oracle | OpCode::PresentRead => {
                let raw = self.pop()?;
                let address = self.address(raw)?;
                self.push(if op == OpCode::Oracle {
                    input[address]
                } else {
                    self.present[address]
                })
            }
            OpCode::Prophecy => {
                self.need(2)?;
                let raw = self.pop()?;
                let word = self.pop()?;
                let address = self.address(raw)?;
                self.present[address] = word & ((1u64 << self.scope.cell_bits) - 1);
                Ok(())
            }
            OpCode::Output => {
                let word = self.pop()?;
                if self.output.len() >= self.config.max_output_items {
                    return Err("numeric output item limit".into());
                }
                self.output
                    .try_reserve(1)
                    .map_err(|_| "checker output allocation".to_string())?;
                self.output.push(word);
                Ok(())
            }
            _ => Err("unsupported primitive in checker".into()),
        }
    }
}

fn interpret(
    program: &BytecodeProgram,
    config: &FiniteProofConfig,
    scope: FiniteProofScope,
    input: &[u64],
) -> Result<FiniteTransition, String> {
    let mut machine = WordMachine {
        config,
        scope,
        stack: Vec::new(),
        present: vec![0; scope.cells],
        output: Vec::new(),
        active_scope: false,
    };
    let mut cursor = program.main;
    let mut frames: Vec<CodeRange> = Vec::new();
    let mut steps = 0;
    let terminal = loop {
        if steps >= config.max_instructions {
            return Err("instruction gas exhausted".into());
        }
        if cursor.start >= cursor.end {
            return Err("invalid program counter".into());
        }
        let pc = cursor.start;
        let instruction = *program
            .instructions
            .get(pc as usize)
            .ok_or("invalid program counter")?;
        cursor.start += 1;
        steps += 1;
        match instruction {
            Instruction::PushWord(word) => machine.push(word)?,
            Instruction::Primitive(OpCode::Halt) => break FiniteProofTerminal::Halted,
            Instruction::Primitive(op) => machine.primitive(op, input)?,
            Instruction::CallProcedure(id) => {
                if frames.len() >= config.max_call_depth {
                    return Err("procedure call depth limit".into());
                }
                frames
                    .try_reserve(1)
                    .map_err(|_| "checker call allocation".to_string())?;
                frames.push(cursor);
                cursor = program
                    .procedures
                    .get(id.index())
                    .ok_or("invalid procedure identity")?
                    .range;
            }
            Instruction::IfFalse { else_target, .. } => {
                if machine.pop()? == 0 {
                    cursor.start = else_target;
                }
            }
            Instruction::Jump { target } => cursor.start = target,
            Instruction::TemporalEnter {
                base,
                size,
                cell_bits,
                ..
            } => {
                if machine.active_scope
                    || config.max_temporal_depth == 0
                    || base != 0
                    || size != scope.cells as u64
                    || cell_bits != scope.cell_bits
                {
                    return Err("invalid temporal entry".into());
                }
                machine.active_scope = true;
            }
            Instruction::TemporalExit { enter_target } => {
                if !machine.active_scope || enter_target != program.main.start {
                    return Err("invalid temporal exit".into());
                }
                machine.active_scope = false;
            }
            Instruction::Return => {
                if let Some(caller) = frames.pop() {
                    cursor = caller;
                } else {
                    break FiniteProofTerminal::Finished;
                }
            }
            _ => return Err("unsupported control in checker".into()),
        }
    };
    Ok(FiniteTransition {
        present: machine.present,
        stack: machine.stack,
        output: machine.output,
        terminal,
        instructions_executed: steps,
    })
}

impl FiniteNoFixedPointCertificate {
    /// Deterministic bounded binary format, including a payload checksum.
    pub fn to_bytes(&self) -> Result<Vec<u8>, FiniteProofError> {
        if self.semantics_version != FINITE_PROOF_SEMANTICS_VERSION {
            return Err(FiniteProofError::BindingMismatch("semantics version"));
        }
        let states = profile_limits(&self.config, self.scope)?;
        validate_rows(self, states)?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        put16(&mut bytes, FINITE_PROOF_FORMAT_VERSION);
        put16(&mut bytes, self.semantics_version);
        bytes.extend_from_slice(&[0, 0]); // NoFixedPoint query and reserved flags.
        bytes.extend_from_slice(&self.program_sha256);
        put32(&mut bytes, self.config.memory_cells as u32);
        bytes.push(match self.config.memory_bounds {
            BoundsPolicy::Error => 0,
            BoundsPolicy::Wrap => 1,
            BoundsPolicy::Clamp => 2,
        });
        put64(&mut bytes, self.config.max_instructions);
        for n in [
            self.config.max_call_depth,
            self.config.max_stack_depth,
            self.config.max_temporal_depth,
            self.config.max_output_items,
        ] {
            put32(&mut bytes, n as u32);
        }
        put64(&mut bytes, self.config.max_output_bytes as u64);
        bytes.push(self.scope.cells as u8);
        bytes.push(self.scope.cell_bits);
        put32(&mut bytes, states as u32);
        for row in &self.rows {
            bytes.push(match row.terminal {
                FiniteProofTerminal::Finished => 0,
                FiniteProofTerminal::Halted => 1,
            });
            put64(&mut bytes, row.instructions_executed);
            for &word in &row.present {
                put64(&mut bytes, word);
            }
            for words in [&row.stack, &row.output] {
                put32(&mut bytes, words.len() as u32);
                for &word in words {
                    put64(&mut bytes, word);
                }
            }
        }
        let checksum: [u8; 32] = Sha256::digest(&bytes).into();
        bytes.extend_from_slice(&checksum);
        if bytes.len() > MAX_FINITE_PROOF_BYTES {
            return Err(FiniteProofError::ResourceLimit("serialized byte ceiling"));
        }
        Ok(bytes)
    }

    /// Verify checksum and all count/byte ceilings before row allocations.
    /// Identity against an external program remains `check_certificate`'s job.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FiniteProofError> {
        if bytes.len() > MAX_FINITE_PROOF_BYTES {
            return Err(FiniteProofError::ResourceLimit("encoded byte ceiling"));
        }
        if bytes.len() < 32 {
            return Err(FiniteProofError::InvalidEncoding("truncated checksum"));
        }
        let (payload, checksum) = bytes.split_at(bytes.len() - 32);
        if Sha256::digest(payload).as_slice() != checksum {
            return Err(FiniteProofError::CorruptEncoding);
        }
        let mut reader = Reader {
            bytes: payload,
            at: 0,
        };
        if reader.take(8)? != MAGIC {
            return Err(FiniteProofError::InvalidEncoding("magic"));
        }
        if reader.u16()? != FINITE_PROOF_FORMAT_VERSION {
            return Err(FiniteProofError::InvalidEncoding("format version"));
        }
        let semantics_version = reader.u16()?;
        if semantics_version != FINITE_PROOF_SEMANTICS_VERSION {
            return Err(FiniteProofError::InvalidEncoding("semantics version"));
        }
        if reader.u8()? != 0 {
            return Err(FiniteProofError::InvalidEncoding("unsupported query tag"));
        }
        if reader.u8()? != 0 {
            return Err(FiniteProofError::InvalidEncoding("reserved flags"));
        }
        let program_sha256 = reader.take(32)?.try_into().unwrap();
        let memory_cells = reader.u32()? as usize;
        let memory_bounds = match reader.u8()? {
            0 => BoundsPolicy::Error,
            1 => BoundsPolicy::Wrap,
            2 => BoundsPolicy::Clamp,
            _ => return Err(FiniteProofError::InvalidEncoding("bounds policy")),
        };
        let max_instructions = reader.u64()?;
        let max_call_depth = reader.u32()? as usize;
        let max_stack_depth = reader.u32()? as usize;
        let max_temporal_depth = reader.u32()? as usize;
        let max_output_items = reader.u32()? as usize;
        let max_output_bytes = usize::try_from(reader.u64()?)
            .map_err(|_| FiniteProofError::ResourceLimit("output bytes do not fit"))?;
        let config = FiniteProofConfig {
            memory_cells,
            memory_bounds,
            max_instructions,
            max_call_depth,
            max_stack_depth,
            max_temporal_depth,
            max_output_items,
            max_output_bytes,
        };
        let scope = FiniteProofScope {
            cells: reader.u8()? as usize,
            cell_bits: reader.u8()?,
        };
        let states = profile_limits(&config, scope)?;
        if reader.u32()? as usize != states {
            return Err(FiniteProofError::InvalidEncoding("incomplete domain table"));
        }
        let min_row = 17 + 8 * scope.cells;
        if reader.remaining() < states * min_row {
            return Err(FiniteProofError::InvalidEncoding("truncated row table"));
        }
        let mut rows = Vec::new();
        rows.try_reserve(states)
            .map_err(|_| FiniteProofError::ResourceLimit("decoder table allocation"))?;
        for _ in 0..states {
            let terminal = match reader.u8()? {
                0 => FiniteProofTerminal::Finished,
                1 => FiniteProofTerminal::Halted,
                _ => return Err(FiniteProofError::InvalidEncoding("non-normal terminal tag")),
            };
            let instructions_executed = reader.u64()?;
            let present = reader.words(scope.cells)?;
            let stack_count = reader.u32()? as usize;
            if stack_count > config.max_stack_depth {
                return Err(FiniteProofError::InvalidEncoding(
                    "stack count exceeds limit",
                ));
            }
            let stack = reader.words(stack_count)?;
            let output_count = reader.u32()? as usize;
            if output_count > config.max_output_items {
                return Err(FiniteProofError::InvalidEncoding(
                    "output count exceeds limit",
                ));
            }
            let output = reader.words(output_count)?;
            rows.push(FiniteTransition {
                present,
                stack,
                output,
                terminal,
                instructions_executed,
            });
        }
        if reader.remaining() != 0 {
            return Err(FiniteProofError::InvalidEncoding("trailing payload bytes"));
        }
        let certificate = Self {
            semantics_version,
            program_sha256,
            config,
            query: FiniteProofQuery::NoFixedPoint,
            scope,
            rows,
        };
        validate_rows(&certificate, states)?;
        Ok(certificate)
    }
}

fn put16(out: &mut Vec<u8>, n: u16) {
    out.extend_from_slice(&n.to_le_bytes());
}
fn put32(out: &mut Vec<u8>, n: u32) {
    out.extend_from_slice(&n.to_le_bytes());
}
fn put64(out: &mut Vec<u8>, n: u64) {
    out.extend_from_slice(&n.to_le_bytes());
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}
impl<'a> Reader<'a> {
    fn remaining(&self) -> usize {
        self.bytes.len() - self.at
    }
    fn take(&mut self, count: usize) -> Result<&'a [u8], FiniteProofError> {
        let end = self
            .at
            .checked_add(count)
            .ok_or(FiniteProofError::InvalidEncoding("offset overflow"))?;
        let value = self
            .bytes
            .get(self.at..end)
            .ok_or(FiniteProofError::InvalidEncoding("truncated payload"))?;
        self.at = end;
        Ok(value)
    }
    fn u8(&mut self) -> Result<u8, FiniteProofError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, FiniteProofError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, FiniteProofError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, FiniteProofError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn words(&mut self, count: usize) -> Result<Vec<u64>, FiniteProofError> {
        let bytes = count
            .checked_mul(8)
            .ok_or(FiniteProofError::InvalidEncoding(
                "word byte count overflow",
            ))?;
        if bytes > self.remaining() {
            return Err(FiniteProofError::InvalidEncoding("truncated word vector"));
        }
        let mut values = Vec::new();
        values
            .try_reserve(count)
            .map_err(|_| FiniteProofError::ResourceLimit("decoder word allocation"))?;
        for _ in 0..count {
            values.push(self.u64()?);
        }
        Ok(values)
    }
}
