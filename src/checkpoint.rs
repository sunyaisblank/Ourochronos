//! Real, bounded continuation of the deterministic classical bytecode core.
//!
//! Slices retain the authoritative VM's cursor and suspended call completions.
//! Reloaded bytes are an opaque, unverified image until a bounded canonical
//! prefix execution proves exact state equality. SHA256 detects corruption; it
//! is neither authentication nor a reachability proof. Reload validation pays
//! for that prefix once; subsequent slices execute only the saved suffix.

use crate::ast::OpCode;
use crate::bytecode::{BytecodeProgram, Instruction};
use crate::bytecode_vm::{
    classical_slice, BytecodeExecution, BytecodeVmConfig, BytecodeVmError, CallFrame,
    ClassicalPause, ClassicalVmProgress, ClassicalVmState, Completion, Cursor, PreparedBytecode,
};
use crate::core::{BoundsPolicy, OutputItem, PagedMemory, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashSet, VecDeque};
use std::error::Error;
use std::fmt;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

pub const CHECKPOINT_FORMAT_VERSION: u16 = 1;
/// Bump whenever the admitted instruction semantics, charging, or state change.
pub const CHECKPOINT_SEMANTICS_VERSION: u16 = 1;
pub const MAX_CHECKPOINT_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_CHECKPOINT_INSTRUCTIONS: u64 = 10_000_000;
pub const MAX_CHECKPOINT_MEMORY_CELLS: usize = 1_048_576;
const MAX_WORDS: usize = 1_000_000;
const MAX_FRAMES: usize = 4_096;
const MAGIC: &[u8; 8] = b"OUROCP\0\0";
const HEADER_BYTES: usize = 16;
const CHECKSUM_BYTES: usize = 32;

/// Exact, frozen query and cumulative resource policy. Slice allowances and
/// cancellation are deliberately separate; neither can reset these ceilings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassicalCheckpointConfig {
    pub memory_cells: usize,
    pub max_instructions: u64,
    pub max_call_depth: usize,
    pub max_stack_depth: usize,
    pub max_output_items: usize,
    pub max_output_bytes: usize,
    pub memory_bounds: BoundsPolicy,
    pub input: Vec<u64>,
}

impl Default for ClassicalCheckpointConfig {
    fn default() -> Self {
        let vm = BytecodeVmConfig::default();
        Self {
            memory_cells: 65_536,
            max_instructions: vm.max_instructions,
            max_call_depth: vm.max_call_depth,
            max_stack_depth: vm.max_stack_depth,
            max_output_items: vm.max_output_items,
            max_output_bytes: vm.max_output_bytes,
            memory_bounds: vm.memory_bounds,
            input: Vec::new(),
        }
    }
}

impl ClassicalCheckpointConfig {
    fn validate(&self) -> Result<(), CheckpointError> {
        if self.memory_cells == 0
            || self.memory_cells > MAX_CHECKPOINT_MEMORY_CELLS
            || self.max_instructions == 0
            || self.max_instructions > MAX_CHECKPOINT_INSTRUCTIONS
            || self.max_call_depth > MAX_FRAMES
            || self.max_stack_depth > MAX_WORDS
            || self.max_output_items > MAX_WORDS
            || self.max_output_bytes > MAX_CHECKPOINT_BYTES
            || self.input.len() > MAX_WORDS
        {
            return Err(CheckpointError::InvalidConfiguration);
        }
        Ok(())
    }

    fn vm(&self) -> BytecodeVmConfig {
        BytecodeVmConfig {
            max_instructions: self.max_instructions,
            max_call_depth: self.max_call_depth,
            max_stack_depth: self.max_stack_depth,
            max_output_items: self.max_output_items,
            max_output_bytes: self.max_output_bytes,
            memory_bounds: self.memory_bounds,
            input: self.input.clone(),
            ..BytecodeVmConfig::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointPauseReason {
    SliceLimit,
    InstructionBound,
    Cancelled,
}

impl From<ClassicalPause> for CheckpointPauseReason {
    fn from(value: ClassicalPause) -> Self {
        match value {
            ClassicalPause::Slice => Self::SliceLimit,
            ClassicalPause::InstructionBound => Self::InstructionBound,
            ClassicalPause::Cancelled => Self::Cancelled,
        }
    }
}

/// A private, reached runtime state tied to one immutable executable and query.
#[derive(Debug)]
pub struct ClassicalCheckpoint {
    program: Arc<PreparedBytecode>,
    vm_config: Arc<BytecodeVmConfig>,
    config: ClassicalCheckpointConfig,
    program_digest: [u8; 32],
    state: ClassicalVmState,
}

#[derive(Debug)]
pub enum CheckpointOutcome {
    /// This is UNKNOWN for unrestricted halting, including at the fixed ceiling.
    Paused {
        checkpoint: ClassicalCheckpoint,
        reason: CheckpointPauseReason,
    },
    Complete(BytecodeExecution),
    /// A fetched instruction failed. Fault states cannot be resumed.
    Fault {
        error: BytecodeVmError,
        instructions: u64,
    },
}

/// A decoded image has no public mutable state and no dispatch method. Only
/// canonical reachability validation can turn it into a runnable checkpoint.
#[derive(Debug)]
pub struct UnverifiedCheckpoint {
    config: ClassicalCheckpointConfig,
    program_digest: [u8; 32],
    state: ClassicalVmState,
}

#[derive(Debug)]
pub enum CheckpointError {
    InvalidConfiguration,
    UnsupportedOpcode(&'static str),
    InvalidVm(BytecodeVmError),
    InvalidEncoding,
    UnsupportedVersion { format: u16, semantics: u16 },
    TooLarge { size: usize, limit: usize },
    ChecksumMismatch,
    ProgramMismatch,
    ConfigurationMismatch,
    UnreachableState,
    ValidationCancelled { instructions_verified: u64 },
    AllocationFailed,
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration => f.write_str("checkpoint configuration exceeds the declared classical bounds"),
            Self::UnsupportedOpcode(op) => write!(f, "{op} is outside the deterministic classical checkpoint core"),
            Self::InvalidVm(error) => write!(f, "checkpoint VM validation failed: {error}"),
            Self::InvalidEncoding => f.write_str("invalid or truncated checkpoint encoding"),
            Self::UnsupportedVersion { format, semantics } => write!(f, "unsupported checkpoint format {format} or semantics {semantics}"),
            Self::TooLarge { size, limit } => write!(f, "checkpoint size {size} exceeds limit {limit}"),
            Self::ChecksumMismatch => f.write_str("checkpoint SHA256 checksum mismatch"),
            Self::ProgramMismatch => f.write_str("checkpoint belongs to a different exact executable"),
            Self::ConfigurationMismatch => f.write_str("checkpoint query, frozen input, or cumulative resource policy changed"),
            Self::UnreachableState => f.write_str("checkpoint does not equal the canonical reached state"),
            Self::ValidationCancelled { instructions_verified } => write!(f, "checkpoint reachability validation cancelled after {instructions_verified} instructions; result UNKNOWN"),
            Self::AllocationFailed => f.write_str("bounded checkpoint allocation failed"),
        }
    }
}

impl Error for CheckpointError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidVm(error) => Some(error),
            _ => None,
        }
    }
}

impl From<BytecodeVmError> for CheckpointError {
    fn from(value: BytecodeVmError) -> Self {
        Self::InvalidVm(value)
    }
}

fn prepare(program: &BytecodeProgram) -> Result<Arc<PreparedBytecode>, CheckpointError> {
    let prepared = PreparedBytecode::new(program.clone())?;
    if !program.foreigns.is_empty() {
        return Err(CheckpointError::UnsupportedOpcode("foreign descriptors"));
    }
    // Direct calls determine procedure reachability. Once a dynamic quotation
    // combinator is reachable, every quote is conservatively included: input
    // words and arithmetic can select any quotation identity at runtime.
    // PreparedBytecode above still validates all unused units structurally.
    let mut pending = VecDeque::from([program.main]);
    let mut visited = HashSet::new();
    while let Some(range) = pending.pop_front() {
        if !visited.insert((range.start, range.end)) {
            continue;
        }
        for instruction in &program.instructions[range.start as usize..range.end as usize] {
            match instruction {
                Instruction::CallForeign(_) => {
                    return Err(CheckpointError::UnsupportedOpcode("foreign call"))
                }
                Instruction::TemporalEnter { .. } | Instruction::TemporalExit { .. } => {
                    return Err(CheckpointError::UnsupportedOpcode("TEMPORAL"))
                }
                Instruction::CallProcedure(id) => {
                    pending.push_back(program.procedures[id.index()].range);
                }
                Instruction::Primitive(opcode) if !classical_opcode(*opcode) => {
                    return Err(CheckpointError::UnsupportedOpcode(opcode.name()))
                }
                Instruction::Primitive(
                    OpCode::Exec | OpCode::Dip | OpCode::Keep | OpCode::Bi | OpCode::Rec,
                ) => {
                    pending.extend(program.quotations.iter().map(|entry| entry.range));
                }
                _ => {}
            }
        }
    }
    Ok(Arc::new(prepared))
}

fn classical_opcode(op: OpCode) -> bool {
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
            | OpCode::Exec
            | OpCode::Dip
            | OpCode::Keep
            | OpCode::Bi
            | OpCode::Rec
            | OpCode::StrRev
            | OpCode::StrCat
            | OpCode::StrSplit
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
            | OpCode::Assert
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
            | OpCode::PresentRead
            | OpCode::Pack
            | OpCode::Unpack
            | OpCode::Index
            | OpCode::Store
            | OpCode::Input
            | OpCode::Output
            | OpCode::Emit
    )
}

fn code_digest(program: &BytecodeProgram) -> Result<[u8; 32], CheckpointError> {
    let bytes = program
        .to_bytes()
        .map_err(BytecodeVmError::InvalidArtifact)?;
    let mut hash = Sha256::new();
    hash.update(b"ourochronos.classical-checkpoint.executable/v1\0");
    hash.update(CHECKPOINT_SEMANTICS_VERSION.to_le_bytes());
    hash.update(bytes);
    Ok(hash.finalize().into())
}

impl ClassicalCheckpoint {
    pub fn start(
        program: &BytecodeProgram,
        config: ClassicalCheckpointConfig,
    ) -> Result<Self, CheckpointError> {
        config.validate()?;
        let program_digest = code_digest(program)?;
        let program = prepare(program)?;
        let vm_config = Arc::new(config.vm());
        let ClassicalVmProgress::Paused { state, .. } =
            classical_slice(&program, &vm_config, config.memory_cells, None, 0, None)?
        else {
            return Err(CheckpointError::UnreachableState);
        };
        Ok(Self {
            program,
            vm_config,
            config,
            program_digest,
            state,
        })
    }

    pub fn instructions_executed(&self) -> u64 {
        self.state.instructions_executed
    }
    pub fn config(&self) -> &ClassicalCheckpointConfig {
        &self.config
    }

    /// Stop only between fetched records. Completing HALT/RETURN wins over a
    /// slice limit reached by that same fetched record. Each call consumes the
    /// checkpoint so faults and completed runs cannot accidentally resume.
    pub fn run_slice(
        self,
        allowance: u64,
        cancellation: Option<&AtomicBool>,
    ) -> Result<CheckpointOutcome, CheckpointError> {
        let Self {
            program,
            vm_config,
            config,
            program_digest,
            state,
        } = self;
        match classical_slice(
            &program,
            &vm_config,
            config.memory_cells,
            Some(state),
            allowance,
            cancellation,
        )? {
            ClassicalVmProgress::Paused { state, reason } => Ok(CheckpointOutcome::Paused {
                checkpoint: Self {
                    program,
                    vm_config,
                    config,
                    program_digest,
                    state,
                },
                reason: reason.into(),
            }),
            ClassicalVmProgress::Complete(execution) => Ok(CheckpointOutcome::Complete(execution)),
            ClassicalVmProgress::Fault {
                error,
                instructions,
            } => Ok(CheckpointOutcome::Fault {
                error,
                instructions,
            }),
        }
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, CheckpointError> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        put_u16(&mut bytes, CHECKPOINT_FORMAT_VERSION);
        put_u16(&mut bytes, CHECKPOINT_SEMANTICS_VERSION);
        put_u32(&mut bytes, 0);
        bytes.extend_from_slice(&self.program_digest);
        encode_config(&mut bytes, &self.config);
        encode_state(&mut bytes, &self.state)?;
        let total = bytes
            .len()
            .checked_add(CHECKSUM_BYTES)
            .ok_or(CheckpointError::InvalidEncoding)?;
        check_size(total)?;
        let body_len = u32::try_from(bytes.len() - HEADER_BYTES)
            .map_err(|_| CheckpointError::InvalidEncoding)?;
        bytes[12..16].copy_from_slice(&body_len.to_le_bytes());
        bytes.extend_from_slice(&checksum(&bytes));
        Ok(bytes)
    }
}

impl UnverifiedCheckpoint {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CheckpointError> {
        check_size(bytes.len())?;
        if bytes.len() < HEADER_BYTES + CHECKSUM_BYTES || &bytes[..8] != MAGIC {
            return Err(CheckpointError::InvalidEncoding);
        }
        let mut header = Reader::new(&bytes[8..HEADER_BYTES]);
        let format = header.u16()?;
        let semantics = header.u16()?;
        if format != CHECKPOINT_FORMAT_VERSION || semantics != CHECKPOINT_SEMANTICS_VERSION {
            return Err(CheckpointError::UnsupportedVersion { format, semantics });
        }
        let length = header.u32()? as usize;
        let end = HEADER_BYTES
            .checked_add(length)
            .ok_or(CheckpointError::InvalidEncoding)?;
        if end.checked_add(CHECKSUM_BYTES) != Some(bytes.len()) {
            return Err(CheckpointError::InvalidEncoding);
        }
        if checksum(&bytes[..end]).as_slice() != &bytes[end..] {
            return Err(CheckpointError::ChecksumMismatch);
        }
        let mut reader = Reader::new(&bytes[HEADER_BYTES..end]);
        let program_digest = reader
            .take(32)?
            .try_into()
            .map_err(|_| CheckpointError::InvalidEncoding)?;
        let config = decode_config(&mut reader)?;
        let state = decode_state(&mut reader, &config)?;
        if !reader.remaining().is_empty() {
            return Err(CheckpointError::InvalidEncoding);
        }
        Ok(Self {
            config,
            program_digest,
            state,
        })
    }

    pub fn instructions_claimed(&self) -> u64 {
        self.state.instructions_executed
    }

    /// Establish that a saved image was reached. Work is bounded by the saved
    /// cumulative count, the exact query ceiling, and the absolute v1 ceiling.
    /// Cancellation leaves the image unverified and proves no halting claim.
    pub fn validate(
        self,
        program: &BytecodeProgram,
        config: &ClassicalCheckpointConfig,
        cancellation: Option<&AtomicBool>,
    ) -> Result<ClassicalCheckpoint, CheckpointError> {
        config.validate()?;
        if &self.config != config {
            return Err(CheckpointError::ConfigurationMismatch);
        }
        if self.program_digest != code_digest(program)? {
            return Err(CheckpointError::ProgramMismatch);
        }
        let program = prepare(program)?;
        validate_control(&self.state, program.program())?;
        let vm_config = Arc::new(config.vm());
        let replay = classical_slice(
            &program,
            &vm_config,
            config.memory_cells,
            None,
            self.state.instructions_executed,
            cancellation,
        )?;
        let ClassicalVmProgress::Paused { state, reason } = replay else {
            return Err(CheckpointError::UnreachableState);
        };
        if reason == ClassicalPause::Cancelled {
            return Err(CheckpointError::ValidationCancelled {
                instructions_verified: state.instructions_executed,
            });
        }
        if state != self.state {
            return Err(CheckpointError::UnreachableState);
        }
        Ok(ClassicalCheckpoint {
            program,
            vm_config,
            config: self.config,
            program_digest: self.program_digest,
            state: self.state,
        })
    }
}

fn validate_control(
    state: &ClassicalVmState,
    program: &BytecodeProgram,
) -> Result<(), CheckpointError> {
    let valid = |cursor: Cursor| {
        cursor.pc >= program.main.start
            && cursor.pc < cursor.end
            && std::iter::once(program.main)
                .chain(program.procedures.iter().map(|p| p.range))
                .chain(program.quotations.iter().map(|q| q.range))
                .any(|range| range.end == cursor.end && cursor.pc >= range.start)
    };
    if !valid(state.cursor) || state.frames.iter().any(|frame| !valid(frame.caller)
        || matches!(frame.completion, Completion::BiSecond { quotation, .. } if quotation.as_u64() >= program.quotations.len() as u64)) {
        return Err(CheckpointError::UnreachableState);
    }
    Ok(())
}

fn encode_config(bytes: &mut Vec<u8>, config: &ClassicalCheckpointConfig) {
    for value in [
        config.memory_cells as u64,
        config.max_instructions,
        config.max_call_depth as u64,
        config.max_stack_depth as u64,
        config.max_output_items as u64,
        config.max_output_bytes as u64,
    ] {
        put_u64(bytes, value);
    }
    bytes.push(match config.memory_bounds {
        BoundsPolicy::Error => 0,
        BoundsPolicy::Wrap => 1,
        BoundsPolicy::Clamp => 2,
    });
    put_words(bytes, &config.input);
}

fn decode_config(reader: &mut Reader<'_>) -> Result<ClassicalCheckpointConfig, CheckpointError> {
    let config = ClassicalCheckpointConfig {
        memory_cells: reader.usize()?,
        max_instructions: reader.u64()?,
        max_call_depth: reader.usize()?,
        max_stack_depth: reader.usize()?,
        max_output_items: reader.usize()?,
        max_output_bytes: reader.usize()?,
        memory_bounds: match reader.u8()? {
            0 => BoundsPolicy::Error,
            1 => BoundsPolicy::Wrap,
            2 => BoundsPolicy::Clamp,
            _ => return Err(CheckpointError::InvalidEncoding),
        },
        input: reader.words(MAX_WORDS)?,
    };
    config.validate()?;
    Ok(config)
}

fn encode_state(bytes: &mut Vec<u8>, state: &ClassicalVmState) -> Result<(), CheckpointError> {
    put_u64(bytes, state.instructions_executed);
    put_u32(bytes, state.cursor.pc);
    put_u32(bytes, state.cursor.end);
    for value in [
        state.maximum_call_depth,
        state.maximum_stack_depth,
        state.input_cursor,
        state.output_bytes,
    ] {
        put_u64(bytes, value as u64);
    }
    put_u32(bytes, state.frames.len() as u32);
    for frame in &state.frames {
        put_u32(bytes, frame.caller.pc);
        put_u32(bytes, frame.caller.end);
        match &frame.completion {
            Completion::None => bytes.push(0),
            Completion::Restore(value) => {
                bytes.push(1);
                put_value(bytes, value)?;
            }
            Completion::BiSecond { value, quotation } => {
                bytes.push(2);
                put_value(bytes, value)?;
                put_u64(bytes, quotation.as_u64());
            }
        }
    }
    put_u32(bytes, state.stack.len() as u32);
    for value in &state.stack {
        put_value(bytes, value)?;
    }
    let count_index = bytes.len();
    put_u32(bytes, 0);
    let mut count = 0u32;
    for address in 0..state.present.len() {
        let value = state
            .present
            .get(address as u64)
            .ok_or(CheckpointError::InvalidEncoding)?;
        if !value.prov.is_pure() {
            return Err(CheckpointError::InvalidEncoding);
        }
        if value.val != 0 {
            count += 1;
            put_u64(bytes, address as u64);
            put_u64(bytes, value.val);
        }
    }
    bytes[count_index..count_index + 4].copy_from_slice(&count.to_le_bytes());
    put_u32(bytes, state.output.len() as u32);
    for item in &state.output {
        match item {
            OutputItem::Val(value) => {
                bytes.push(0);
                put_value(bytes, value)?;
            }
            OutputItem::Char(byte) => {
                bytes.push(1);
                bytes.push(*byte);
            }
        }
    }
    Ok(())
}

fn decode_state(
    reader: &mut Reader<'_>,
    config: &ClassicalCheckpointConfig,
) -> Result<ClassicalVmState, CheckpointError> {
    let instructions_executed = reader.u64()?;
    let cursor = Cursor {
        pc: reader.u32()?,
        end: reader.u32()?,
    };
    let maximum_call_depth = reader.usize()?;
    let maximum_stack_depth = reader.usize()?;
    let input_cursor = reader.usize()?;
    let output_bytes = reader.usize()?;
    if instructions_executed > config.max_instructions
        || maximum_call_depth > config.max_call_depth
        || maximum_stack_depth > config.max_stack_depth
        || input_cursor > config.input.len()
        || output_bytes > config.max_output_bytes
    {
        return Err(CheckpointError::InvalidEncoding);
    }
    let frame_count = reader.count(config.max_call_depth, 9)?;
    let mut frames = reserved(frame_count)?;
    for _ in 0..frame_count {
        let caller = Cursor {
            pc: reader.u32()?,
            end: reader.u32()?,
        };
        let completion = match reader.u8()? {
            0 => Completion::None,
            1 => Completion::Restore(Value::new(reader.u64()?)),
            2 => Completion::BiSecond {
                value: Value::new(reader.u64()?),
                quotation: crate::ast::QuoteId::new(reader.u64()?),
            },
            _ => return Err(CheckpointError::InvalidEncoding),
        };
        frames.push(CallFrame { caller, completion });
    }
    let stack = reader
        .words(config.max_stack_depth)?
        .into_iter()
        .map(Value::new)
        .collect::<Vec<_>>();
    if frames.len() > maximum_call_depth || stack.len() > maximum_stack_depth {
        return Err(CheckpointError::InvalidEncoding);
    }
    let memory_count = reader.count(config.memory_cells, 16)?;
    let mut present = PagedMemory::with_size(config.memory_cells)
        .map_err(|_| CheckpointError::AllocationFailed)?;
    let mut previous = None;
    for _ in 0..memory_count {
        let address = reader.u64()?;
        let word = reader.u64()?;
        if address >= config.memory_cells as u64
            || word == 0
            || previous.is_some_and(|old| old >= address)
        {
            return Err(CheckpointError::InvalidEncoding);
        }
        present
            .write(address, Value::new(word))
            .map_err(|_| CheckpointError::InvalidEncoding)?;
        previous = Some(address);
    }
    let output_count = reader.count(config.max_output_items, 2)?;
    let mut output = reserved(output_count)?;
    let mut computed_bytes = 0usize;
    for _ in 0..output_count {
        let item = match reader.u8()? {
            0 => OutputItem::Val(Value::new(reader.u64()?)),
            1 => OutputItem::Char(reader.u8()?),
            _ => return Err(CheckpointError::InvalidEncoding),
        };
        computed_bytes = computed_bytes
            .checked_add(item.retained_size_charge())
            .ok_or(CheckpointError::InvalidEncoding)?;
        output.push(item);
    }
    if computed_bytes != output_bytes {
        return Err(CheckpointError::InvalidEncoding);
    }
    Ok(ClassicalVmState {
        cursor,
        frames,
        stack,
        present,
        output,
        output_bytes,
        input_cursor,
        inputs_consumed: config.input[..input_cursor].to_vec(),
        instructions_executed,
        maximum_call_depth,
        maximum_stack_depth,
    })
}

fn put_value(bytes: &mut Vec<u8>, value: &Value) -> Result<(), CheckpointError> {
    if !value.prov.is_pure() {
        return Err(CheckpointError::InvalidEncoding);
    }
    put_u64(bytes, value.val);
    Ok(())
}
fn put_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
fn put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
fn put_words(bytes: &mut Vec<u8>, words: &[u64]) {
    put_u32(bytes, words.len() as u32);
    for &word in words {
        put_u64(bytes, word);
    }
}
fn checksum(bytes: &[u8]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"ourochronos.classical-checkpoint.image/v1\0");
    hash.update(bytes);
    hash.finalize().into()
}
fn check_size(size: usize) -> Result<(), CheckpointError> {
    if size > MAX_CHECKPOINT_BYTES {
        Err(CheckpointError::TooLarge {
            size,
            limit: MAX_CHECKPOINT_BYTES,
        })
    } else {
        Ok(())
    }
}
fn reserved<T>(count: usize) -> Result<Vec<T>, CheckpointError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| CheckpointError::AllocationFailed)?;
    Ok(values)
}

struct Reader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}
impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }
    fn remaining(&self) -> &'a [u8] {
        &self.bytes[self.cursor..]
    }
    fn take(&mut self, count: usize) -> Result<&'a [u8], CheckpointError> {
        let end = self
            .cursor
            .checked_add(count)
            .ok_or(CheckpointError::InvalidEncoding)?;
        let value = self
            .bytes
            .get(self.cursor..end)
            .ok_or(CheckpointError::InvalidEncoding)?;
        self.cursor = end;
        Ok(value)
    }
    fn u8(&mut self) -> Result<u8, CheckpointError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, CheckpointError> {
        Ok(u16::from_le_bytes(
            self.take(2)?
                .try_into()
                .map_err(|_| CheckpointError::InvalidEncoding)?,
        ))
    }
    fn u32(&mut self) -> Result<u32, CheckpointError> {
        Ok(u32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| CheckpointError::InvalidEncoding)?,
        ))
    }
    fn u64(&mut self) -> Result<u64, CheckpointError> {
        Ok(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| CheckpointError::InvalidEncoding)?,
        ))
    }
    fn usize(&mut self) -> Result<usize, CheckpointError> {
        usize::try_from(self.u64()?).map_err(|_| CheckpointError::InvalidEncoding)
    }
    fn count(&mut self, limit: usize, min_bytes: usize) -> Result<usize, CheckpointError> {
        let count = self.u32()? as usize;
        if count > limit
            || count
                .checked_mul(min_bytes)
                .is_none_or(|bytes| bytes > self.remaining().len())
        {
            return Err(CheckpointError::InvalidEncoding);
        }
        Ok(count)
    }
    fn words(&mut self, limit: usize) -> Result<Vec<u64>, CheckpointError> {
        let count = self.count(limit, 8)?;
        let mut values = reserved(count)?;
        for _ in 0..count {
            values.push(self.u64()?);
        }
        Ok(values)
    }
}
