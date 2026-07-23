//! Iterative execution backend for validated Ourochronos bytecode.
//!
//! Procedure and quotation calls use an explicit frame vector.  No language
//! call, loop, or quotation combinator consumes the host call stack.  The VM
//! validates every artifact before dispatch and applies deterministic limits
//! to instructions, frames, the operand stack, temporal nesting, output, and
//! dynamic collections. CLOCK, RANDOM, file reads, socket receives, and
//! process results consume explicit frozen environment tapes. File and socket
//! handles operate only on per-run virtual images; destructive operations,
//! process launches, and sleeps become [`EffectIntent`] records and cannot
//! reach an authorized host adapter during bytecode dispatch.

use crate::ast::{OpCode, QuoteId};
use crate::bytecode::{BytecodeError, BytecodeProgram, CodeRange, Instruction};
use crate::bytecode_verifier::{verify_default as verify_bytecode, BytecodeVerificationError};
use crate::core::error::BoundsPolicy;
use crate::core::provenance::Provenance;
use crate::core::{DataStructures, OutputItem, PagedMemory, Value};
use crate::hir::ForeignId;
use crate::runtime::ffi::{ForeignHostError, ForeignHostTable};
use crate::runtime::io::{FileMode, SeekOrigin};
use crate::temporal::transaction::{EffectIntent, ObservationTranscript};
pub use crate::temporal::transaction::{
    FrozenEndpointTape, FrozenFileSnapshot, FrozenProcessResult,
};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::io::Read;
use std::sync::Arc;

/// Deterministic resource and memory policy for bytecode execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BytecodeVmConfig {
    /// Maximum number of fetched bytecode instructions.
    pub max_instructions: u64,
    /// Maximum number of suspended language call frames (the main unit is not
    /// counted).
    pub max_call_depth: usize,
    /// Maximum operand-stack depth.
    pub max_stack_depth: usize,
    /// Maximum simultaneously active temporal scopes.
    pub max_temporal_depth: usize,
    /// Maximum buffered output items.
    pub max_output_items: usize,
    /// Maximum conservative retained bytes for buffered output, including
    /// explicit provenance dependencies.
    pub max_output_bytes: usize,
    /// Maximum number of each dynamic collection kind.
    /// Buffers count as a distinct collection kind; freed handles are not
    /// reused, so stale words can never alias a later allocation.
    pub max_collections: usize,
    /// Maximum number of values in any one vector, hash table, or set, and
    /// maximum capacity/length in bytes of any one in-memory buffer.
    pub max_collection_items: usize,
    /// Maximum conservative aggregate bytes retained by all vectors, hash
    /// tables, sets, buffers, and their entries in one epoch.
    pub max_dynamic_bytes: usize,
    /// Policy for addresses outside memory or an active temporal scope.
    pub memory_bounds: BoundsPolicy,
    /// Frozen deterministic input values.  Exhaustion is an error.
    pub input: Vec<u64>,
    /// Permit INPUT to read from the live process after the configured tape.
    /// Temporal drivers may enable this only for the first candidate epoch,
    /// then must freeze the captured tape for every subsequent evaluation.
    pub allow_interactive_input: bool,
    /// Frozen millisecond timestamps returned by CLOCK. Exhaustion is an
    /// error unless live system-input capture is explicitly enabled.
    pub clock_input: Vec<u64>,
    /// Frozen words returned by RANDOM. Exhaustion is an error unless live
    /// system-input capture is explicitly enabled.
    pub random_input: Vec<u64>,
    /// Permit CLOCK and RANDOM to capture live process values after their
    /// configured tapes. Temporal drivers may enable this only for the first
    /// candidate epoch, then must freeze both captured tapes for replay.
    pub allow_live_system_inputs: bool,
    /// Explicit immutable file observations. Every path that may be read or
    /// queried must appear here unless first-candidate live capture is enabled.
    pub file_snapshots: Vec<FrozenFileSnapshot>,
    /// Exact path identities whose contents or metadata may be observed by
    /// language file primitives. Snapshots without this capability remain
    /// hidden base images for write-only virtualization.
    pub file_read_capabilities: Vec<String>,
    /// Exact path identities on which destructive virtual file operations may
    /// be staged. This grants no host access during execution.
    pub file_write_capabilities: Vec<String>,
    /// Permit missing file snapshots to be captured from the live host. A
    /// temporal driver may enable this only for the first candidate and must
    /// freeze all observations before evaluating another candidate.
    pub allow_live_file_reads: bool,
    /// Maximum number of configured or captured file snapshots and open
    /// virtual file handles.
    pub max_file_snapshots: usize,
    /// Maximum aggregate bytes retained by configured or captured snapshots.
    pub max_file_snapshot_bytes: usize,
    /// Frozen deterministic receive streams keyed by exact `host:port`.
    pub endpoint_tapes: Vec<FrozenEndpointTape>,
    /// Exact endpoint identities whose sends may be staged. Frozen endpoint
    /// tapes independently authorize deterministic virtual connect/receive;
    /// neither grants host network access during a run.
    pub network_send_capabilities: Vec<String>,
    /// Maximum aggregate bytes retained by frozen endpoint receive streams.
    pub max_endpoint_tape_bytes: usize,
    /// Frozen deterministic command results keyed by exact command spelling.
    pub process_results: Vec<FrozenProcessResult>,
    /// Exact command spellings whose post-selection spawn may be staged.
    pub process_capabilities: Vec<String>,
    /// Maximum aggregate command-identity and output bytes retained by frozen
    /// process results.
    pub max_process_result_bytes: usize,
    /// Maximum selected sleep duration that may be staged. `None` denies
    /// `SLEEP`; no candidate ever sleeps on the host.
    pub max_sleep_milliseconds: Option<u64>,
    /// Maximum staged irreversible intents in one execution.
    pub max_effects: usize,
    /// Maximum aggregate path and payload bytes in staged intents.
    pub max_effect_bytes: usize,
}

impl Default for BytecodeVmConfig {
    fn default() -> Self {
        Self {
            max_instructions: 10_000_000,
            max_call_depth: 4_096,
            max_stack_depth: 1_000_000,
            max_temporal_depth: 1,
            max_output_items: 1_000_000,
            max_output_bytes: 64 * 1024 * 1024,
            max_collections: 100_000,
            max_collection_items: 1_000_000,
            max_dynamic_bytes: 64 * 1024 * 1024,
            memory_bounds: BoundsPolicy::Error,
            input: Vec::new(),
            allow_interactive_input: false,
            clock_input: Vec::new(),
            random_input: Vec::new(),
            allow_live_system_inputs: false,
            file_snapshots: Vec::new(),
            file_read_capabilities: Vec::new(),
            file_write_capabilities: Vec::new(),
            allow_live_file_reads: false,
            max_file_snapshots: 1_024,
            max_file_snapshot_bytes: 64 * 1024 * 1024,
            endpoint_tapes: Vec::new(),
            network_send_capabilities: Vec::new(),
            max_endpoint_tape_bytes: 64 * 1024 * 1024,
            process_results: Vec::new(),
            process_capabilities: Vec::new(),
            max_process_result_bytes: 64 * 1024 * 1024,
            max_sleep_milliseconds: None,
            max_effects: 100_000,
            max_effect_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Normal terminal condition of a bytecode run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BytecodeVmStatus {
    /// The main unit returned.
    Finished,
    /// `HALT` terminated the entire run.
    Halted,
    /// `PARADOX` rejected the current epoch.
    Paradox,
}

/// Complete observable result of a successful or paradoxical run.
#[derive(Debug, Clone)]
pub struct BytecodeExecution {
    /// Final operand stack.
    pub stack: Vec<Value>,
    /// Present memory produced by the run.
    pub present: PagedMemory,
    /// Buffered numeric and character output.
    pub output: Vec<OutputItem>,
    /// Dynamic vectors, hash tables, and sets allocated by the run.
    pub data_structures: DataStructures,
    /// Normal terminal condition.
    pub status: BytecodeVmStatus,
    /// Number of fetched instructions, including control and return records.
    pub instructions_executed: u64,
    /// Frozen inputs consumed in order.
    pub inputs_consumed: Vec<u64>,
    /// Frozen CLOCK values consumed in order.
    pub clock_inputs_consumed: Vec<u64>,
    /// Frozen RANDOM values consumed in order.
    pub random_inputs_consumed: Vec<u64>,
    /// File observations consumed in first-access order, including explicit
    /// missing-path metadata and any opt-in live captures.
    pub file_snapshots_consumed: Vec<FrozenFileSnapshot>,
    /// Frozen endpoint streams consumed in first-connect order.
    pub endpoint_tapes_consumed: Vec<FrozenEndpointTape>,
    /// Frozen command results consumed in first-execution order.
    pub process_results_consumed: Vec<FrozenProcessResult>,
    /// Irreversible operations represented only as data. No host effect has
    /// occurred while producing this execution record.
    pub effects: Vec<EffectIntent>,
    /// Number of temporal-scope entries executed.
    pub temporal_entries: u64,
    /// Number of temporal-scope exits executed.
    pub temporal_exits: u64,
    /// Maximum simultaneously suspended call frames.
    pub maximum_call_depth: usize,
    /// Maximum simultaneously active temporal scopes.
    pub maximum_temporal_depth: usize,
}

impl BytecodeExecution {
    /// Exact frozen environment observations that participated in this epoch.
    pub fn observation_transcript(&self) -> ObservationTranscript {
        ObservationTranscript {
            input: self.inputs_consumed.clone(),
            clock: self.clock_inputs_consumed.clone(),
            random: self.random_inputs_consumed.clone(),
            files: self.file_snapshots_consumed.clone(),
            endpoints: self.endpoint_tapes_consumed.clone(),
            processes: self.process_results_consumed.clone(),
        }
    }

    /// Conservative deterministic charge used by bounded multi-epoch caches.
    pub(crate) fn retained_size_charge(&self) -> usize {
        let mut bytes = std::mem::size_of::<Self>()
            .saturating_add(self.present.retained_size_charge())
            .saturating_add(self.data_structures.retained_size_charge());
        for value in &self.stack {
            bytes = bytes.saturating_add(value.retained_size_charge());
        }
        for item in &self.output {
            bytes = bytes.saturating_add(item.retained_size_charge());
        }
        bytes = bytes.saturating_add(
            self.inputs_consumed
                .len()
                .saturating_add(self.clock_inputs_consumed.len())
                .saturating_add(self.random_inputs_consumed.len())
                .saturating_mul(std::mem::size_of::<u64>()),
        );
        for snapshot in &self.file_snapshots_consumed {
            bytes = bytes
                .saturating_add(snapshot.path.len())
                .saturating_add(snapshot.contents.as_ref().map_or(0, Vec::len))
                .saturating_add(1);
        }
        for tape in &self.endpoint_tapes_consumed {
            bytes = bytes
                .saturating_add(tape.endpoint.len())
                .saturating_add(tape.recv_bytes.len());
        }
        for result in &self.process_results_consumed {
            bytes = bytes
                .saturating_add(result.command.len())
                .saturating_add(result.output.len())
                .saturating_add(std::mem::size_of::<i32>());
        }
        for effect in &self.effects {
            bytes = bytes.saturating_add(64).saturating_add(effect.byte_len());
        }
        bytes
    }
}

/// A checked bytecode runtime failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BytecodeVmError {
    /// Artifact structure was invalid.  Validation happens before any dispatch.
    InvalidArtifact(BytecodeError),
    /// Independent bytecode proof rejected a structurally valid artifact.
    BytecodeVerification(BytecodeVerificationError),
    /// Execution consumed its exact instruction allowance.
    GasExhausted { limit: u64 },
    /// A call would exceed the configured language-frame limit.
    CallDepthExceeded { limit: usize },
    /// An operation would exceed the operand-stack limit.
    StackLimitExceeded { limit: usize },
    /// An operation required more operands than were present.
    StackUnderflow {
        operation: &'static str,
        needed: usize,
        available: usize,
    },
    /// A runtime quotation word did not identify a compiled quotation.
    InvalidQuote { id: u64, quote_count: usize },
    /// The artifact's complete foreign table could not be linked before run.
    ForeignLink(ForeignHostError),
    /// A linked process-host callback failed or violated its return contract.
    ForeignCall(ForeignHostError),
    /// The backend deliberately does not implement this primitive.
    UnsupportedPrimitive(OpCode),
    /// Deterministic input was exhausted.
    InputExhausted { consumed: usize },
    /// Opt-in interactive INPUT reached end of file before a token.
    InteractiveInputEof,
    /// Opt-in interactive INPUT exceeded its deterministic byte bound.
    InteractiveInputTooLong { limit: usize },
    /// Opt-in interactive INPUT was not a valid unsigned word.
    InteractiveInputInvalid,
    /// Opt-in interactive INPUT failed at the process I/O boundary.
    InteractiveInputIo { message: String },
    /// The frozen CLOCK tape was exhausted.
    ClockInputExhausted { consumed: usize },
    /// The frozen RANDOM tape was exhausted.
    RandomInputExhausted { consumed: usize },
    /// A file observation was not present in the frozen snapshot set.
    FileSnapshotUnavailable { path: String },
    /// Configured frozen file metadata violated deterministic bounds.
    InvalidFileSnapshot { message: String },
    /// Configured frozen endpoint metadata violated deterministic bounds.
    InvalidEndpointTape { message: String },
    /// A virtual connect had no exact frozen endpoint stream.
    EndpointTapeUnavailable { endpoint: String },
    /// A virtual connect or staged send lacked an exact endpoint capability.
    NetworkCapabilityDenied { endpoint: String },
    /// A word did not identify a live virtual socket handle.
    InvalidSocketHandle { handle: u64 },
    /// Configured frozen process metadata violated deterministic bounds.
    InvalidProcessResult { message: String },
    /// A command had no exact frozen result.
    ProcessResultUnavailable { command: String },
    /// A staged process invocation lacked an exact command capability.
    ProcessCapabilityDenied { command: String },
    /// A selected sleep exceeded or lacked its explicit duration capability.
    SleepCapabilityDenied {
        milliseconds: u64,
        maximum: Option<u64>,
    },
    /// A destructive file operation lacked an exact path capability.
    FileCapabilityDenied { path: String },
    /// A file observation lacked an exact path capability.
    FileReadCapabilityDenied { path: String },
    /// A word did not identify a live virtual file handle.
    InvalidFileHandle { handle: u64 },
    /// A virtual or opt-in capture file operation failed.
    FileOperation {
        operation: &'static str,
        path: String,
        message: String,
    },
    /// A memory address was rejected by the selected bounds policy.
    MemoryOutOfBounds { address: u64, memory_cells: usize },
    /// A temporal scope was invalid for this concrete memory or runtime limit.
    TemporalViolation(String),
    /// An assertion evaluated false.
    AssertionFailed(String),
    /// A dynamic structure handle or index was invalid.
    DataStructure(String),
    /// A word did not identify a live, correctly typed in-memory buffer.
    InvalidBufferHandle { handle: u64 },
    /// A deterministic dynamic allocation limit was reached.
    AllocationLimit { what: &'static str, limit: usize },
    /// An impossible program counter was encountered after validation.
    InvalidProgramCounter(u32),
}

impl fmt::Display for BytecodeVmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArtifact(error) => write!(formatter, "invalid bytecode: {error}"),
            Self::BytecodeVerification(error) => {
                write!(formatter, "bytecode verification failed: {error}")
            }
            Self::GasExhausted { limit } => {
                write!(formatter, "bytecode instruction limit {limit} exhausted")
            }
            Self::CallDepthExceeded { limit } => {
                write!(formatter, "bytecode call depth limit {limit} exceeded")
            }
            Self::StackLimitExceeded { limit } => {
                write!(formatter, "bytecode operand stack limit {limit} exceeded")
            }
            Self::StackUnderflow {
                operation,
                needed,
                available,
            } => write!(
                formatter,
                "{operation} requires {needed} stack values, but only {available} are available"
            ),
            Self::InvalidQuote { id, quote_count } => {
                write!(
                    formatter,
                    "quotation {id} is outside table of size {quote_count}"
                )
            }
            Self::ForeignLink(error) => {
                write!(formatter, "foreign execution is not linked: {error}")
            }
            Self::ForeignCall(error) => write!(formatter, "foreign call failed: {error}"),
            Self::UnsupportedPrimitive(opcode) => {
                write!(
                    formatter,
                    "bytecode primitive {} is unsupported",
                    opcode.name()
                )
            }
            Self::InputExhausted { consumed } => {
                write!(formatter, "frozen input exhausted after {consumed} values")
            }
            Self::InteractiveInputEof => {
                formatter.write_str("interactive input reached end of file")
            }
            Self::InteractiveInputTooLong { limit } => {
                write!(formatter, "interactive input exceeds {limit} bytes")
            }
            Self::InteractiveInputInvalid => {
                formatter.write_str("interactive input is not an unsigned 64-bit integer")
            }
            Self::InteractiveInputIo { message } => {
                write!(formatter, "interactive input I/O failed: {message}")
            }
            Self::ClockInputExhausted { consumed } => {
                write!(
                    formatter,
                    "frozen CLOCK input exhausted after {consumed} values"
                )
            }
            Self::RandomInputExhausted { consumed } => {
                write!(
                    formatter,
                    "frozen RANDOM input exhausted after {consumed} values"
                )
            }
            Self::FileSnapshotUnavailable { path } => {
                write!(formatter, "frozen file snapshot unavailable for '{path}'")
            }
            Self::InvalidFileSnapshot { message } => {
                write!(formatter, "invalid frozen file snapshot: {message}")
            }
            Self::InvalidEndpointTape { message } => {
                write!(formatter, "invalid frozen endpoint tape: {message}")
            }
            Self::EndpointTapeUnavailable { endpoint } => {
                write!(
                    formatter,
                    "frozen endpoint tape unavailable for '{endpoint}'"
                )
            }
            Self::NetworkCapabilityDenied { endpoint } => {
                write!(formatter, "network capability denied for '{endpoint}'")
            }
            Self::InvalidSocketHandle { handle } => {
                write!(formatter, "invalid virtual socket handle {handle}")
            }
            Self::InvalidProcessResult { message } => {
                write!(formatter, "invalid frozen process result: {message}")
            }
            Self::ProcessResultUnavailable { command } => {
                write!(
                    formatter,
                    "frozen process result unavailable for {command:?}"
                )
            }
            Self::ProcessCapabilityDenied { command } => {
                write!(formatter, "process capability denied for {command:?}")
            }
            Self::SleepCapabilityDenied {
                milliseconds,
                maximum,
            } => match maximum {
                Some(maximum) => write!(
                    formatter,
                    "sleep duration {milliseconds}ms exceeds capability limit {maximum}ms"
                ),
                None => write!(formatter, "sleep capability denied for {milliseconds}ms"),
            },
            Self::FileCapabilityDenied { path } => {
                write!(formatter, "file write capability denied for '{path}'")
            }
            Self::FileReadCapabilityDenied { path } => {
                write!(formatter, "file read capability denied for '{path}'")
            }
            Self::InvalidFileHandle { handle } => {
                write!(formatter, "invalid virtual file handle {handle}")
            }
            Self::FileOperation {
                operation,
                path,
                message,
            } => write!(formatter, "{operation} '{path}': {message}"),
            Self::MemoryOutOfBounds {
                address,
                memory_cells,
            } => write!(
                formatter,
                "memory address {address} is outside {memory_cells} cells"
            ),
            Self::TemporalViolation(message) => write!(formatter, "temporal scope: {message}"),
            Self::AssertionFailed(message) => write!(formatter, "assertion failed: {message}"),
            Self::DataStructure(message) => write!(formatter, "data structure: {message}"),
            Self::InvalidBufferHandle { handle } => {
                write!(formatter, "invalid buffer handle {handle}")
            }
            Self::AllocationLimit { what, limit } => {
                write!(formatter, "{what} allocation limit {limit} reached")
            }
            Self::InvalidProgramCounter(pc) => {
                write!(formatter, "invalid bytecode program counter {pc}")
            }
        }
    }
}

impl Error for BytecodeVmError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidArtifact(error) => Some(error),
            Self::BytecodeVerification(error) => Some(error),
            Self::ForeignLink(error) | Self::ForeignCall(error) => Some(error),
            _ => None,
        }
    }
}

impl From<BytecodeError> for BytecodeVmError {
    fn from(error: BytecodeError) -> Self {
        Self::InvalidArtifact(error)
    }
}

impl From<BytecodeVerificationError> for BytecodeVmError {
    fn from(error: BytecodeVerificationError) -> Self {
        Self::BytecodeVerification(error)
    }
}

/// Whether the iterative backend has concrete semantics for a primitive.
/// Only the two obsolete dynamic FFI spellings remain unsupported. Environment
/// observations require frozen typed inputs or explicit first-candidate
/// capture, and irreversible operations are staged instead of running here.
pub const fn bytecode_vm_supports(opcode: OpCode) -> bool {
    !matches!(opcode, OpCode::FFICall | OpCode::FFICallNamed)
}

/// Immutable independently verified bytecode prepared for repeated execution.
///
/// Construction performs structural validation and the independent CFG/stack/
/// temporal proof once. Because the wrapped program has no mutable access,
/// subsequent epochs can skip those redundant scans without creating a second
/// instruction representation or changing dispatch, gas, error, or observation
/// semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedBytecode {
    program: BytecodeProgram,
}

impl PreparedBytecode {
    /// Independently verify and seal one executable artifact.
    pub fn new(program: BytecodeProgram) -> Result<Self, BytecodeVmError> {
        verify_bytecode(&program)?;
        Ok(Self { program })
    }

    /// Borrow the exact validated executable representation.
    pub fn program(&self) -> &BytecodeProgram {
        &self.program
    }

    /// Recover the executable bytes. Re-preparation is required after any
    /// later mutation of the returned program.
    pub fn into_program(self) -> BytecodeProgram {
        self.program
    }
}

/// Iterative bytecode executor.
#[derive(Debug, Clone)]
pub struct BytecodeVm {
    /// Runtime limits and deterministic inputs.
    pub config: BytecodeVmConfig,
    /// Exact process-local implementations for linked foreign descriptors.
    foreign_host: Option<Arc<ForeignHostTable>>,
}

impl Default for BytecodeVm {
    fn default() -> Self {
        Self::new()
    }
}

impl BytecodeVm {
    /// Construct an executor with strict deterministic defaults.
    pub fn new() -> Self {
        Self {
            config: BytecodeVmConfig::default(),
            foreign_host: None,
        }
    }

    /// Construct an executor with explicit limits and inputs.
    pub fn with_config(config: BytecodeVmConfig) -> Self {
        Self {
            config,
            foreign_host: None,
        }
    }

    /// Attach an explicitly linked, lifetime-safe process host table.
    pub fn with_foreign_host(mut self, foreign_host: Arc<ForeignHostTable>) -> Self {
        self.foreign_host = Some(foreign_host);
        self
    }

    /// Independently verify and execute one bytecode epoch against a read-only
    /// anamnesis.
    pub fn run(
        &self,
        program: &BytecodeProgram,
        anamnesis: &PagedMemory,
    ) -> Result<BytecodeExecution, BytecodeVmError> {
        verify_bytecode(program)?;
        self.run_validated_program(program, anamnesis)
    }

    /// Execute an immutable artifact whose structural and independent semantic
    /// verification has already succeeded. This is the optimized repeated-epoch
    /// dispatch path.
    pub fn run_prepared(
        &self,
        program: &PreparedBytecode,
        anamnesis: &PagedMemory,
    ) -> Result<BytecodeExecution, BytecodeVmError> {
        self.run_validated_program(program.program(), anamnesis)
    }

    fn run_validated_program(
        &self,
        program: &BytecodeProgram,
        anamnesis: &PagedMemory,
    ) -> Result<BytecodeExecution, BytecodeVmError> {
        if let Some(first) = program.foreigns.first() {
            let host = self
                .foreign_host
                .as_deref()
                .ok_or(BytecodeVmError::ForeignLink(
                    ForeignHostError::UnknownTarget { id: first.id },
                ))?;
            host.validate_program(program)
                .map_err(BytecodeVmError::ForeignLink)?;
        }
        Runtime::new(
            program,
            anamnesis,
            &self.config,
            self.foreign_host.as_deref(),
        )?
        .run()
    }
}

#[derive(Debug, Clone, Copy)]
struct Cursor {
    pc: u32,
    end: u32,
}

#[derive(Debug, Clone)]
enum Completion {
    None,
    Restore(Value),
    BiSecond { value: Value, quotation: QuoteId },
}

#[derive(Debug, Clone)]
struct CallFrame {
    caller: Cursor,
    completion: Completion,
}

#[derive(Debug, Clone)]
struct TemporalScope {
    enter_pc: u32,
    base: u64,
    size: u64,
    cell_bits: u8,
    snapshot: Vec<Value>,
}

/// A buffer handle is deliberately distinct from vector/hash/set handles in
/// the runtime.  The language representation remains one word, but the host
/// implementation cannot accidentally use it with another handle table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BufferHandle(u64);

#[derive(Debug, Clone, PartialEq, Eq)]
struct ByteBuffer {
    data: Vec<u8>,
    position: usize,
}

impl ByteBuffer {
    fn growth_capacity(&self, limit: usize) -> usize {
        if self.data.len() < self.data.capacity() {
            return self.data.capacity();
        }
        let doubled = self.data.capacity().max(4).saturating_mul(2);
        doubled.min(limit).max(self.data.len().saturating_add(1))
    }

    fn with_capacity(capacity: usize, limit: usize) -> Result<Self, BytecodeVmError> {
        if capacity > limit {
            return Err(BytecodeVmError::AllocationLimit {
                what: "buffer byte",
                limit,
            });
        }
        let mut data = Vec::new();
        data.try_reserve_exact(capacity)
            .map_err(|_| BytecodeVmError::AllocationLimit {
                what: "buffer byte",
                limit,
            })?;
        Ok(Self { data, position: 0 })
    }

    fn from_values(values: &[Value], limit: usize) -> Result<Self, BytecodeVmError> {
        if values.len() > limit {
            return Err(BytecodeVmError::AllocationLimit {
                what: "buffer byte",
                limit,
            });
        }
        let mut data = Vec::new();
        data.try_reserve_exact(values.len())
            .map_err(|_| BytecodeVmError::AllocationLimit {
                what: "buffer byte",
                limit,
            })?;
        data.extend(values.iter().map(|value| value.val as u8));
        Ok(Self { data, position: 0 })
    }

    fn read_byte(&mut self) -> u8 {
        let Some(&byte) = self.data.get(self.position) else {
            return 0;
        };
        self.position += 1;
        byte
    }

    fn write_byte(&mut self, byte: u8, limit: usize) -> Result<(), BytecodeVmError> {
        if self.position < self.data.len() {
            self.data[self.position] = byte;
            self.position += 1;
            return Ok(());
        }
        if self.data.len() >= limit {
            return Err(BytecodeVmError::AllocationLimit {
                what: "buffer byte",
                limit,
            });
        }
        let target = self.growth_capacity(limit);
        self.data
            .try_reserve_exact(target.saturating_sub(self.data.len()))
            .map_err(|_| BytecodeVmError::AllocationLimit {
                what: "buffer byte",
                limit,
            })?;
        self.data.push(byte);
        self.position += 1;
        Ok(())
    }
}

/// Per-run, bounded buffer arena. Handles are one-based to match the legacy
/// executor. Slots are never reused: after `BUFFER_FREE`, a stale handle stays
/// invalid instead of silently acquiring the type/identity of a later buffer.
#[derive(Debug, Default)]
struct BufferStore {
    slots: Vec<Option<ByteBuffer>>,
}

impl BufferStore {
    fn allocate(
        &mut self,
        buffer: ByteBuffer,
        limit: usize,
    ) -> Result<BufferHandle, BytecodeVmError> {
        if self.slots.len() >= limit {
            return Err(BytecodeVmError::AllocationLimit {
                what: "buffer",
                limit,
            });
        }
        self.slots
            .try_reserve(1)
            .map_err(|_| BytecodeVmError::AllocationLimit {
                what: "buffer",
                limit,
            })?;
        let word = u64::try_from(self.slots.len())
            .ok()
            .and_then(|index| index.checked_add(1))
            .ok_or(BytecodeVmError::AllocationLimit {
                what: "buffer",
                limit,
            })?;
        self.slots.push(Some(buffer));
        Ok(BufferHandle(word))
    }

    fn index(handle: BufferHandle) -> Result<usize, BytecodeVmError> {
        let index = handle
            .0
            .checked_sub(1)
            .and_then(|word| usize::try_from(word).ok())
            .ok_or(BytecodeVmError::InvalidBufferHandle { handle: handle.0 })?;
        Ok(index)
    }

    fn get(&self, handle: BufferHandle) -> Result<&ByteBuffer, BytecodeVmError> {
        self.slots
            .get(Self::index(handle)?)
            .and_then(Option::as_ref)
            .ok_or(BytecodeVmError::InvalidBufferHandle { handle: handle.0 })
    }

    fn get_mut(&mut self, handle: BufferHandle) -> Result<&mut ByteBuffer, BytecodeVmError> {
        self.slots
            .get_mut(Self::index(handle)?)
            .and_then(Option::as_mut)
            .ok_or(BytecodeVmError::InvalidBufferHandle { handle: handle.0 })
    }

    fn free(&mut self, handle: BufferHandle) -> Option<ByteBuffer> {
        Self::index(handle)
            .ok()
            .and_then(|index| self.slots.get_mut(index))
            .and_then(Option::take)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VirtualFileHandle(u64);

#[derive(Debug, Clone, PartialEq, Eq)]
struct VirtualOpenFile {
    path: String,
    mode: FileMode,
    position: u64,
}

#[derive(Debug)]
struct VirtualFileStore {
    frozen: BTreeMap<String, FrozenFileSnapshot>,
    images: BTreeMap<String, Option<Vec<u8>>>,
    consumed_paths: BTreeSet<String>,
    consumed: Vec<FrozenFileSnapshot>,
    handles: Vec<Option<VirtualOpenFile>>,
    read_capabilities: BTreeSet<String>,
    write_capabilities: BTreeSet<String>,
    allow_live_reads: bool,
    max_snapshots: usize,
    max_snapshot_bytes: usize,
    max_file_bytes: usize,
    total_image_bytes: usize,
    max_effects: usize,
    max_effect_bytes: usize,
    effect_bytes: usize,
    effects: Vec<EffectIntent>,
}

impl VirtualFileStore {
    fn new(config: &BytecodeVmConfig) -> Result<Self, BytecodeVmError> {
        if config.file_snapshots.len() > config.max_file_snapshots {
            return Err(BytecodeVmError::InvalidFileSnapshot {
                message: format!(
                    "{} snapshots exceed limit {}",
                    config.file_snapshots.len(),
                    config.max_file_snapshots
                ),
            });
        }
        for (kind, capabilities) in [
            ("read", &config.file_read_capabilities),
            ("write", &config.file_write_capabilities),
        ] {
            if capabilities.len() > config.max_file_snapshots {
                return Err(BytecodeVmError::InvalidFileSnapshot {
                    message: format!(
                        "{} file {kind} capabilities exceed limit {}",
                        capabilities.len(),
                        config.max_file_snapshots
                    ),
                });
            }
            if capabilities
                .iter()
                .any(|path| path.len() > config.max_collection_items)
            {
                return Err(BytecodeVmError::InvalidFileSnapshot {
                    message: format!(
                        "file {kind} capability exceeds {} bytes",
                        config.max_collection_items
                    ),
                });
            }
        }
        let mut frozen = BTreeMap::new();
        let mut images = BTreeMap::new();
        let mut total_image_bytes = 0usize;
        for snapshot in &config.file_snapshots {
            if snapshot.path.len() > config.max_collection_items {
                return Err(BytecodeVmError::InvalidFileSnapshot {
                    message: format!("path exceeds {} bytes", config.max_collection_items),
                });
            }
            if let Some(contents) = &snapshot.contents {
                if contents.len() > config.max_collection_items {
                    return Err(BytecodeVmError::InvalidFileSnapshot {
                        message: format!(
                            "'{}' exceeds per-file limit {}",
                            snapshot.path, config.max_collection_items
                        ),
                    });
                }
                total_image_bytes =
                    total_image_bytes
                        .checked_add(contents.len())
                        .ok_or_else(|| BytecodeVmError::InvalidFileSnapshot {
                            message: "aggregate byte count overflowed".to_string(),
                        })?;
            }
            if total_image_bytes > config.max_file_snapshot_bytes {
                return Err(BytecodeVmError::InvalidFileSnapshot {
                    message: format!(
                        "aggregate bytes exceed limit {}",
                        config.max_file_snapshot_bytes
                    ),
                });
            }
            if frozen
                .insert(snapshot.path.clone(), snapshot.clone())
                .is_some()
            {
                return Err(BytecodeVmError::InvalidFileSnapshot {
                    message: format!("duplicate path '{}'", snapshot.path),
                });
            }
            images.insert(snapshot.path.clone(), snapshot.contents.clone());
        }
        Ok(Self {
            frozen,
            images,
            consumed_paths: BTreeSet::new(),
            consumed: Vec::new(),
            handles: Vec::new(),
            read_capabilities: config.file_read_capabilities.iter().cloned().collect(),
            write_capabilities: config.file_write_capabilities.iter().cloned().collect(),
            allow_live_reads: config.allow_live_file_reads,
            max_snapshots: config.max_file_snapshots,
            max_snapshot_bytes: config.max_file_snapshot_bytes,
            max_file_bytes: config.max_collection_items,
            total_image_bytes,
            max_effects: config.max_effects,
            max_effect_bytes: config.max_effect_bytes,
            effect_bytes: 0,
            effects: Vec::new(),
        })
    }

    fn ensure_snapshot(&mut self, path: &str) -> Result<(), BytecodeVmError> {
        if !self.frozen.contains_key(path) {
            if !self.allow_live_reads {
                return Err(BytecodeVmError::FileSnapshotUnavailable {
                    path: path.to_string(),
                });
            }
            if self.frozen.len() >= self.max_snapshots {
                return Err(BytecodeVmError::AllocationLimit {
                    what: "file snapshot",
                    limit: self.max_snapshots,
                });
            }
            let snapshot = self.capture(path)?;
            self.images
                .insert(path.to_string(), snapshot.contents.clone());
            self.frozen.insert(path.to_string(), snapshot);
        }
        if self.consumed_paths.insert(path.to_string()) {
            let snapshot = self
                .frozen
                .get(path)
                .expect("snapshot was inserted above")
                .clone();
            self.consumed.push(snapshot);
        }
        Ok(())
    }

    fn capture(&mut self, path: &str) -> Result<FrozenFileSnapshot, BytecodeVmError> {
        if path.len() > self.max_file_bytes {
            return Err(BytecodeVmError::AllocationLimit {
                what: "file path byte",
                limit: self.max_file_bytes,
            });
        }
        let file = match std::fs::File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(FrozenFileSnapshot {
                    path: path.to_string(),
                    contents: None,
                });
            }
            Err(error) => {
                return Err(BytecodeVmError::FileOperation {
                    operation: "capture",
                    path: path.to_string(),
                    message: error.to_string(),
                });
            }
        };
        let metadata = file
            .metadata()
            .map_err(|error| BytecodeVmError::FileOperation {
                operation: "capture",
                path: path.to_string(),
                message: error.to_string(),
            })?;
        if !metadata.is_file() {
            return Err(BytecodeVmError::FileOperation {
                operation: "capture",
                path: path.to_string(),
                message: "path is not a regular file".to_string(),
            });
        }
        let length =
            usize::try_from(metadata.len()).map_err(|_| BytecodeVmError::AllocationLimit {
                what: "file snapshot byte",
                limit: self.max_file_bytes,
            })?;
        let remaining = self
            .max_snapshot_bytes
            .saturating_sub(self.total_image_bytes);
        let limit = self.max_file_bytes.min(remaining);
        if length > limit {
            return Err(BytecodeVmError::AllocationLimit {
                what: "file snapshot byte",
                limit,
            });
        }
        let mut contents = Vec::with_capacity(length);
        file.take(u64::try_from(limit).unwrap_or(u64::MAX).saturating_add(1))
            .read_to_end(&mut contents)
            .map_err(|error| BytecodeVmError::FileOperation {
                operation: "capture",
                path: path.to_string(),
                message: error.to_string(),
            })?;
        if contents.len() > limit {
            return Err(BytecodeVmError::AllocationLimit {
                what: "file snapshot byte",
                limit,
            });
        }
        self.total_image_bytes += contents.len();
        Ok(FrozenFileSnapshot {
            path: path.to_string(),
            contents: Some(contents),
        })
    }

    fn stage(&mut self, effect: EffectIntent) -> Result<(), BytecodeVmError> {
        if self.effects.len() >= self.max_effects {
            return Err(BytecodeVmError::AllocationLimit {
                what: "effect",
                limit: self.max_effects,
            });
        }
        let additional = match &effect {
            EffectIntent::FileWrite {
                path,
                bytes,
                initial,
                ..
            } => path
                .len()
                .saturating_add(bytes.len())
                .saturating_add(initial.path.len())
                .saturating_add(initial.contents.as_ref().map_or(0, Vec::len)),
            EffectIntent::FileSetLength { path, initial, .. } => path
                .len()
                .saturating_add(initial.path.len())
                .saturating_add(initial.contents.as_ref().map_or(0, Vec::len)),
            EffectIntent::NetworkSend { endpoint, bytes } => {
                endpoint.len().saturating_add(bytes.len())
            }
            EffectIntent::ProcessSpawn { program, arguments } => arguments
                .iter()
                .fold(program.len(), |sum, value| sum.saturating_add(value.len())),
            EffectIntent::Sleep { .. } => std::mem::size_of::<u64>(),
            EffectIntent::Custom { namespace, payload } => {
                namespace.len().saturating_add(payload.len())
            }
        };
        let total =
            self.effect_bytes
                .checked_add(additional)
                .ok_or(BytecodeVmError::AllocationLimit {
                    what: "effect byte",
                    limit: self.max_effect_bytes,
                })?;
        if total > self.max_effect_bytes {
            return Err(BytecodeVmError::AllocationLimit {
                what: "effect byte",
                limit: self.max_effect_bytes,
            });
        }
        self.effect_bytes = total;
        self.effects.push(effect);
        Ok(())
    }

    fn require_write(&self, path: &str) -> Result<(), BytecodeVmError> {
        if self.write_capabilities.contains(path) {
            Ok(())
        } else {
            Err(BytecodeVmError::FileCapabilityDenied {
                path: path.to_string(),
            })
        }
    }

    fn require_read(&self, path: &str) -> Result<(), BytecodeVmError> {
        if self.read_capabilities.contains(path) {
            Ok(())
        } else {
            Err(BytecodeVmError::FileReadCapabilityDenied {
                path: path.to_string(),
            })
        }
    }

    fn ensure_frozen_base(&mut self, path: &str) -> Result<(), BytecodeVmError> {
        if !self.frozen.contains_key(path) {
            return Err(BytecodeVmError::FileSnapshotUnavailable {
                path: path.to_string(),
            });
        }
        if self.consumed_paths.insert(path.to_string()) {
            self.consumed.push(
                self.frozen
                    .get(path)
                    .expect("snapshot existence was checked above")
                    .clone(),
            );
        }
        Ok(())
    }

    fn image(&self, path: &str) -> Result<&Option<Vec<u8>>, BytecodeVmError> {
        self.images
            .get(path)
            .ok_or_else(|| BytecodeVmError::FileSnapshotUnavailable {
                path: path.to_string(),
            })
    }

    fn initial_snapshot(&self, path: &str) -> Result<FrozenFileSnapshot, BytecodeVmError> {
        self.frozen
            .get(path)
            .cloned()
            .ok_or_else(|| BytecodeVmError::FileSnapshotUnavailable {
                path: path.to_string(),
            })
    }

    fn replace_image(&mut self, path: &str, image: Option<Vec<u8>>) -> Result<(), BytecodeVmError> {
        let old_length = self
            .images
            .get(path)
            .and_then(Option::as_ref)
            .map_or(0, Vec::len);
        let new_length = image.as_ref().map_or(0, Vec::len);
        if new_length > self.max_file_bytes {
            return Err(BytecodeVmError::AllocationLimit {
                what: "virtual file byte",
                limit: self.max_file_bytes,
            });
        }
        let without_old = self
            .total_image_bytes
            .checked_sub(old_length)
            .ok_or_else(|| BytecodeVmError::InvalidFileSnapshot {
                message: "virtual file byte accounting underflowed".to_string(),
            })?;
        let total =
            without_old
                .checked_add(new_length)
                .ok_or(BytecodeVmError::AllocationLimit {
                    what: "virtual file byte",
                    limit: self.max_snapshot_bytes,
                })?;
        if total > self.max_snapshot_bytes {
            return Err(BytecodeVmError::AllocationLimit {
                what: "virtual file byte",
                limit: self.max_snapshot_bytes,
            });
        }
        self.total_image_bytes = total;
        self.images.insert(path.to_string(), image);
        Ok(())
    }

    fn allocate_handle(
        &mut self,
        file: VirtualOpenFile,
    ) -> Result<VirtualFileHandle, BytecodeVmError> {
        if self.handles.len() >= self.max_snapshots {
            return Err(BytecodeVmError::AllocationLimit {
                what: "open file",
                limit: self.max_snapshots,
            });
        }
        let handle = u64::try_from(self.handles.len())
            .ok()
            .and_then(|value| value.checked_add(1))
            .ok_or(BytecodeVmError::AllocationLimit {
                what: "open file",
                limit: self.max_snapshots,
            })?;
        self.handles.push(Some(file));
        Ok(VirtualFileHandle(handle))
    }

    fn handle(&self, handle: VirtualFileHandle) -> Result<&VirtualOpenFile, BytecodeVmError> {
        let index = handle
            .0
            .checked_sub(1)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(BytecodeVmError::InvalidFileHandle { handle: handle.0 })?;
        self.handles
            .get(index)
            .and_then(Option::as_ref)
            .ok_or(BytecodeVmError::InvalidFileHandle { handle: handle.0 })
    }

    fn handle_mut(
        &mut self,
        handle: VirtualFileHandle,
    ) -> Result<&mut VirtualOpenFile, BytecodeVmError> {
        let index = handle
            .0
            .checked_sub(1)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(BytecodeVmError::InvalidFileHandle { handle: handle.0 })?;
        self.handles
            .get_mut(index)
            .and_then(Option::as_mut)
            .ok_or(BytecodeVmError::InvalidFileHandle { handle: handle.0 })
    }

    fn close(&mut self, handle: VirtualFileHandle) -> Result<(), BytecodeVmError> {
        let index = handle
            .0
            .checked_sub(1)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(BytecodeVmError::InvalidFileHandle { handle: handle.0 })?;
        if self.handles.get_mut(index).and_then(Option::take).is_some() {
            Ok(())
        } else {
            Err(BytecodeVmError::InvalidFileHandle { handle: handle.0 })
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VirtualSocketHandle(u64);

#[derive(Debug, Clone, PartialEq, Eq)]
struct VirtualSocket {
    endpoint: String,
}

#[derive(Debug)]
struct EndpointStream {
    tape: FrozenEndpointTape,
    cursor: usize,
}

#[derive(Debug)]
struct VirtualNetworkStore {
    endpoints: BTreeMap<String, EndpointStream>,
    send_capabilities: BTreeSet<String>,
    consumed_endpoints: BTreeSet<String>,
    consumed: Vec<FrozenEndpointTape>,
    handles: Vec<Option<VirtualSocket>>,
    max_handles: usize,
}

impl VirtualNetworkStore {
    fn new(config: &BytecodeVmConfig) -> Result<Self, BytecodeVmError> {
        if config.endpoint_tapes.len() > config.max_collections {
            return Err(BytecodeVmError::InvalidEndpointTape {
                message: format!(
                    "{} endpoint tapes exceed limit {}",
                    config.endpoint_tapes.len(),
                    config.max_collections
                ),
            });
        }
        if config.network_send_capabilities.len() > config.max_collections
            || config
                .network_send_capabilities
                .iter()
                .any(|endpoint| endpoint.len() > config.max_collection_items)
        {
            return Err(BytecodeVmError::InvalidEndpointTape {
                message: format!(
                    "network send capabilities exceed count {} or identity-byte {} limit",
                    config.max_collections, config.max_collection_items
                ),
            });
        }
        let mut endpoints = BTreeMap::new();
        let mut aggregate_bytes = 0usize;
        for tape in &config.endpoint_tapes {
            if tape.endpoint.len() > config.max_collection_items {
                return Err(BytecodeVmError::InvalidEndpointTape {
                    message: format!(
                        "endpoint identity exceeds {} bytes",
                        config.max_collection_items
                    ),
                });
            }
            if tape.recv_bytes.len() > config.max_collection_items {
                return Err(BytecodeVmError::InvalidEndpointTape {
                    message: format!(
                        "'{}' exceeds per-endpoint limit {}",
                        tape.endpoint, config.max_collection_items
                    ),
                });
            }
            aggregate_bytes = aggregate_bytes
                .checked_add(tape.recv_bytes.len())
                .ok_or_else(|| BytecodeVmError::InvalidEndpointTape {
                    message: "aggregate byte count overflowed".to_string(),
                })?;
            if aggregate_bytes > config.max_endpoint_tape_bytes {
                return Err(BytecodeVmError::InvalidEndpointTape {
                    message: format!(
                        "aggregate bytes exceed limit {}",
                        config.max_endpoint_tape_bytes
                    ),
                });
            }
            if endpoints
                .insert(
                    tape.endpoint.clone(),
                    EndpointStream {
                        tape: tape.clone(),
                        cursor: 0,
                    },
                )
                .is_some()
            {
                return Err(BytecodeVmError::InvalidEndpointTape {
                    message: format!("duplicate endpoint '{}'", tape.endpoint),
                });
            }
        }
        Ok(Self {
            endpoints,
            send_capabilities: config.network_send_capabilities.iter().cloned().collect(),
            consumed_endpoints: BTreeSet::new(),
            consumed: Vec::new(),
            handles: Vec::new(),
            max_handles: config.max_collections,
        })
    }

    fn require_send(&self, endpoint: &str) -> Result<(), BytecodeVmError> {
        if self.send_capabilities.contains(endpoint) {
            Ok(())
        } else {
            Err(BytecodeVmError::NetworkCapabilityDenied {
                endpoint: endpoint.to_string(),
            })
        }
    }

    fn connect(&mut self, endpoint: String) -> Result<VirtualSocketHandle, BytecodeVmError> {
        let stream = self.endpoints.get(&endpoint).ok_or_else(|| {
            BytecodeVmError::EndpointTapeUnavailable {
                endpoint: endpoint.clone(),
            }
        })?;
        if self.consumed_endpoints.insert(endpoint.clone()) {
            self.consumed.push(stream.tape.clone());
        }
        if self.handles.len() >= self.max_handles {
            return Err(BytecodeVmError::AllocationLimit {
                what: "socket",
                limit: self.max_handles,
            });
        }
        let handle = u64::try_from(self.handles.len())
            .ok()
            .and_then(|value| value.checked_add(1))
            .ok_or(BytecodeVmError::AllocationLimit {
                what: "socket",
                limit: self.max_handles,
            })?;
        self.handles.push(Some(VirtualSocket { endpoint }));
        Ok(VirtualSocketHandle(handle))
    }

    fn socket(&self, handle: VirtualSocketHandle) -> Result<&VirtualSocket, BytecodeVmError> {
        let index = handle
            .0
            .checked_sub(1)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(BytecodeVmError::InvalidSocketHandle { handle: handle.0 })?;
        self.handles
            .get(index)
            .and_then(Option::as_ref)
            .ok_or(BytecodeVmError::InvalidSocketHandle { handle: handle.0 })
    }

    fn receive(
        &mut self,
        handle: VirtualSocketHandle,
        maximum: usize,
    ) -> Result<Vec<u8>, BytecodeVmError> {
        let endpoint = self.socket(handle)?.endpoint.clone();
        let stream = self.endpoints.get_mut(&endpoint).ok_or_else(|| {
            BytecodeVmError::EndpointTapeUnavailable {
                endpoint: endpoint.clone(),
            }
        })?;
        let end = stream
            .tape
            .recv_bytes
            .len()
            .min(stream.cursor.saturating_add(maximum));
        let bytes = stream.tape.recv_bytes[stream.cursor..end].to_vec();
        stream.cursor = end;
        Ok(bytes)
    }

    fn receive_len(
        &self,
        handle: VirtualSocketHandle,
        maximum: usize,
    ) -> Result<usize, BytecodeVmError> {
        let endpoint = &self.socket(handle)?.endpoint;
        let stream = self.endpoints.get(endpoint).ok_or_else(|| {
            BytecodeVmError::EndpointTapeUnavailable {
                endpoint: endpoint.clone(),
            }
        })?;
        Ok(stream
            .tape
            .recv_bytes
            .len()
            .min(stream.cursor.saturating_add(maximum))
            .saturating_sub(stream.cursor))
    }

    fn close(&mut self, handle: VirtualSocketHandle) -> Result<(), BytecodeVmError> {
        let index = handle
            .0
            .checked_sub(1)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(BytecodeVmError::InvalidSocketHandle { handle: handle.0 })?;
        if self.handles.get_mut(index).and_then(Option::take).is_some() {
            Ok(())
        } else {
            Err(BytecodeVmError::InvalidSocketHandle { handle: handle.0 })
        }
    }
}

#[derive(Debug)]
struct FrozenProcessStore {
    results: BTreeMap<String, FrozenProcessResult>,
    capabilities: BTreeSet<String>,
    consumed_commands: BTreeSet<String>,
    consumed: Vec<FrozenProcessResult>,
}

impl FrozenProcessStore {
    fn new(config: &BytecodeVmConfig) -> Result<Self, BytecodeVmError> {
        if config.process_results.len() > config.max_collections {
            return Err(BytecodeVmError::InvalidProcessResult {
                message: format!(
                    "{} process results exceed limit {}",
                    config.process_results.len(),
                    config.max_collections
                ),
            });
        }
        if config.process_capabilities.len() > config.max_collections
            || config
                .process_capabilities
                .iter()
                .any(|command| command.len() > config.max_collection_items)
        {
            return Err(BytecodeVmError::InvalidProcessResult {
                message: format!(
                    "process capabilities exceed count {} or command-byte {} limit",
                    config.max_collections, config.max_collection_items
                ),
            });
        }
        let mut results = BTreeMap::new();
        let mut aggregate_bytes = 0usize;
        for result in &config.process_results {
            if result.command.len() > config.max_collection_items {
                return Err(BytecodeVmError::InvalidProcessResult {
                    message: format!(
                        "command identity exceeds {} bytes",
                        config.max_collection_items
                    ),
                });
            }
            if result.output.len() > config.max_collection_items {
                return Err(BytecodeVmError::InvalidProcessResult {
                    message: format!(
                        "command {:?} output exceeds per-result limit {}",
                        result.command, config.max_collection_items
                    ),
                });
            }
            let retained_bytes = result
                .command
                .len()
                .checked_add(result.output.len())
                .ok_or_else(|| BytecodeVmError::InvalidProcessResult {
                    message: "retained byte count overflowed".to_string(),
                })?;
            aggregate_bytes = aggregate_bytes.checked_add(retained_bytes).ok_or_else(|| {
                BytecodeVmError::InvalidProcessResult {
                    message: "aggregate byte count overflowed".to_string(),
                }
            })?;
            if aggregate_bytes > config.max_process_result_bytes {
                return Err(BytecodeVmError::InvalidProcessResult {
                    message: format!(
                        "aggregate bytes exceed limit {}",
                        config.max_process_result_bytes
                    ),
                });
            }
            if results
                .insert(result.command.clone(), result.clone())
                .is_some()
            {
                return Err(BytecodeVmError::InvalidProcessResult {
                    message: format!("duplicate command {:?}", result.command),
                });
            }
        }
        Ok(Self {
            results,
            capabilities: config.process_capabilities.iter().cloned().collect(),
            consumed_commands: BTreeSet::new(),
            consumed: Vec::new(),
        })
    }

    fn execute(&mut self, command: &str) -> Result<FrozenProcessResult, BytecodeVmError> {
        if !self.capabilities.contains(command) {
            return Err(BytecodeVmError::ProcessCapabilityDenied {
                command: command.to_string(),
            });
        }
        let result = self.results.get(command).cloned().ok_or_else(|| {
            BytecodeVmError::ProcessResultUnavailable {
                command: command.to_string(),
            }
        })?;
        if self.consumed_commands.insert(command.to_string()) {
            self.consumed.push(result.clone());
        }
        Ok(result)
    }

    fn output_len(&self, command: &str) -> Result<usize, BytecodeVmError> {
        if !self.capabilities.contains(command) {
            return Err(BytecodeVmError::ProcessCapabilityDenied {
                command: command.to_string(),
            });
        }
        self.results
            .get(command)
            .map(|result| result.output.len())
            .ok_or_else(|| BytecodeVmError::ProcessResultUnavailable {
                command: command.to_string(),
            })
    }
}

struct Runtime<'a> {
    program: &'a BytecodeProgram,
    config: &'a BytecodeVmConfig,
    foreign_host: Option<&'a ForeignHostTable>,
    cursor: Cursor,
    frames: Vec<CallFrame>,
    temporal_scopes: Vec<TemporalScope>,
    stack: Vec<Value>,
    present: PagedMemory,
    anamnesis: &'a PagedMemory,
    output: Vec<OutputItem>,
    output_bytes: usize,
    data: DataStructures,
    dynamic_bytes: usize,
    buffers: BufferStore,
    files: VirtualFileStore,
    network: VirtualNetworkStore,
    processes: FrozenProcessStore,
    input_cursor: usize,
    inputs_consumed: Vec<u64>,
    clock_input_cursor: usize,
    clock_inputs_consumed: Vec<u64>,
    random_input_cursor: usize,
    random_inputs_consumed: Vec<u64>,
    instructions_executed: u64,
    temporal_entries: u64,
    temporal_exits: u64,
    maximum_call_depth: usize,
    maximum_temporal_depth: usize,
}

impl<'a> Runtime<'a> {
    fn new(
        program: &'a BytecodeProgram,
        anamnesis: &'a PagedMemory,
        config: &'a BytecodeVmConfig,
        foreign_host: Option<&'a ForeignHostTable>,
    ) -> Result<Self, BytecodeVmError> {
        let present = PagedMemory::with_size(anamnesis.len())
            .map_err(|error| BytecodeVmError::TemporalViolation(error.to_string()))?;
        let files = VirtualFileStore::new(config)?;
        let network = VirtualNetworkStore::new(config)?;
        let processes = FrozenProcessStore::new(config)?;
        Ok(Self {
            program,
            config,
            foreign_host,
            cursor: Cursor {
                pc: program.main.start,
                end: program.main.end,
            },
            frames: Vec::new(),
            temporal_scopes: Vec::new(),
            stack: Vec::new(),
            present,
            anamnesis,
            output: Vec::new(),
            output_bytes: 0,
            data: DataStructures::new(),
            dynamic_bytes: 0,
            buffers: BufferStore::default(),
            files,
            network,
            processes,
            input_cursor: 0,
            inputs_consumed: Vec::new(),
            clock_input_cursor: 0,
            clock_inputs_consumed: Vec::new(),
            random_input_cursor: 0,
            random_inputs_consumed: Vec::new(),
            instructions_executed: 0,
            temporal_entries: 0,
            temporal_exits: 0,
            maximum_call_depth: 0,
            maximum_temporal_depth: 0,
        })
    }

    fn run(mut self) -> Result<BytecodeExecution, BytecodeVmError> {
        let status = loop {
            if self.instructions_executed >= self.config.max_instructions {
                return Err(BytecodeVmError::GasExhausted {
                    limit: self.config.max_instructions,
                });
            }
            if self.cursor.pc >= self.cursor.end {
                return Err(BytecodeVmError::InvalidProgramCounter(self.cursor.pc));
            }
            let pc = self.cursor.pc;
            let instruction = *self
                .program
                .instructions
                .get(pc as usize)
                .ok_or(BytecodeVmError::InvalidProgramCounter(pc))?;
            self.cursor.pc = self
                .cursor
                .pc
                .checked_add(1)
                .ok_or(BytecodeVmError::InvalidProgramCounter(pc))?;
            self.instructions_executed += 1;

            match instruction {
                Instruction::Primitive(OpCode::Halt) => break BytecodeVmStatus::Halted,
                Instruction::Primitive(OpCode::Paradox) => {
                    self.rollback_temporal()?;
                    break BytecodeVmStatus::Paradox;
                }
                Instruction::Primitive(opcode) => self.execute_primitive(opcode)?,
                Instruction::PushWord(word) => self.push(Value::new(word))?,
                Instruction::PushQuote(id) => self.push(Value::new(id.as_u64()))?,
                Instruction::CallProcedure(id) => {
                    let entry = self
                        .program
                        .procedures
                        .get(id.index())
                        .ok_or(BytecodeVmError::InvalidProgramCounter(pc))?;
                    self.enter(entry.range, Completion::None)?;
                }
                Instruction::CallForeign(id) => {
                    self.call_foreign(id)?;
                }
                Instruction::IfFalse {
                    else_target,
                    end_target: _,
                    has_else: _,
                } => {
                    let condition = self.pop("IF")?;
                    if !condition.to_bool() {
                        self.cursor.pc = else_target;
                    }
                }
                Instruction::Jump { target } | Instruction::LoopBack { target } => {
                    self.cursor.pc = target;
                }
                Instruction::WhileFalse {
                    loop_start: _,
                    end_target,
                } => {
                    let condition = self.pop("WHILE")?;
                    if !condition.to_bool() {
                        self.cursor.pc = end_target;
                    }
                }
                Instruction::TemporalEnter {
                    base,
                    size,
                    cell_bits,
                    exit_target: _,
                } => self.enter_temporal(pc, base, size, cell_bits)?,
                Instruction::TemporalExit { enter_target } => {
                    self.exit_temporal(enter_target)?;
                }
                Instruction::Return => {
                    let Some(frame) = self.frames.pop() else {
                        break BytecodeVmStatus::Finished;
                    };
                    self.cursor = frame.caller;
                    self.finish_completion(frame.completion)?;
                }
            }
        };

        Ok(BytecodeExecution {
            stack: self.stack,
            present: self.present,
            output: self.output,
            data_structures: self.data,
            status,
            instructions_executed: self.instructions_executed,
            inputs_consumed: self.inputs_consumed,
            clock_inputs_consumed: self.clock_inputs_consumed,
            random_inputs_consumed: self.random_inputs_consumed,
            file_snapshots_consumed: self.files.consumed,
            endpoint_tapes_consumed: self.network.consumed,
            process_results_consumed: self.processes.consumed,
            effects: self.files.effects,
            temporal_entries: self.temporal_entries,
            temporal_exits: self.temporal_exits,
            maximum_call_depth: self.maximum_call_depth,
            maximum_temporal_depth: self.maximum_temporal_depth,
        })
    }

    fn enter(&mut self, range: CodeRange, completion: Completion) -> Result<(), BytecodeVmError> {
        if self.frames.len() >= self.config.max_call_depth {
            return Err(BytecodeVmError::CallDepthExceeded {
                limit: self.config.max_call_depth,
            });
        }
        self.frames.push(CallFrame {
            caller: self.cursor,
            completion,
        });
        self.maximum_call_depth = self.maximum_call_depth.max(self.frames.len());
        self.cursor = Cursor {
            pc: range.start,
            end: range.end,
        };
        Ok(())
    }

    fn enter_quote(&mut self, id: QuoteId, completion: Completion) -> Result<(), BytecodeVmError> {
        let index = usize::try_from(id.as_u64()).map_err(|_| BytecodeVmError::InvalidQuote {
            id: id.as_u64(),
            quote_count: self.program.quotations.len(),
        })?;
        let entry = self
            .program
            .quotations
            .get(index)
            .ok_or(BytecodeVmError::InvalidQuote {
                id: id.as_u64(),
                quote_count: self.program.quotations.len(),
            })?;
        self.enter(entry.range, completion)
    }

    fn enter_quote_word(&mut self, id: u64, completion: Completion) -> Result<(), BytecodeVmError> {
        let checked_id = self.check_quote_word(id)?;
        let index = checked_id.as_u64() as usize;
        let entry = self
            .program
            .quotations
            .get(index)
            .ok_or(BytecodeVmError::InvalidQuote {
                id,
                quote_count: self.program.quotations.len(),
            })?;
        self.enter(entry.range, completion)
    }

    fn check_quote_word(&self, id: u64) -> Result<QuoteId, BytecodeVmError> {
        let index = usize::try_from(id).map_err(|_| BytecodeVmError::InvalidQuote {
            id,
            quote_count: self.program.quotations.len(),
        })?;
        if index >= self.program.quotations.len() {
            return Err(BytecodeVmError::InvalidQuote {
                id,
                quote_count: self.program.quotations.len(),
            });
        }
        Ok(QuoteId::new(id))
    }

    fn finish_completion(&mut self, completion: Completion) -> Result<(), BytecodeVmError> {
        match completion {
            Completion::None => Ok(()),
            Completion::Restore(value) => self.push(value),
            Completion::BiSecond { value, quotation } => {
                self.push(value)?;
                self.enter_quote(quotation, Completion::None)
            }
        }
    }

    fn enter_temporal(
        &mut self,
        enter_pc: u32,
        base: u64,
        size: u64,
        cell_bits: u8,
    ) -> Result<(), BytecodeVmError> {
        if self.temporal_scopes.len() >= self.config.max_temporal_depth {
            return Err(BytecodeVmError::TemporalViolation(format!(
                "nesting exceeds configured depth {}",
                self.config.max_temporal_depth
            )));
        }
        let end = base.checked_add(size).ok_or_else(|| {
            BytecodeVmError::TemporalViolation("address range overflows u64".to_string())
        })?;
        if end > self.present.len() as u64 {
            return Err(BytecodeVmError::TemporalViolation(format!(
                "region [{base}..{end}) exceeds {} memory cells",
                self.present.len()
            )));
        }
        let snapshot = (base..end)
            .map(|address| read_cell(&self.present, address))
            .collect::<Result<Vec<_>, _>>()?;
        self.temporal_scopes.push(TemporalScope {
            enter_pc,
            base,
            size,
            cell_bits,
            snapshot,
        });
        self.temporal_entries = self.temporal_entries.saturating_add(1);
        self.maximum_temporal_depth = self.maximum_temporal_depth.max(self.temporal_scopes.len());
        Ok(())
    }

    fn exit_temporal(&mut self, enter_target: u32) -> Result<(), BytecodeVmError> {
        let scope = self.temporal_scopes.pop().ok_or_else(|| {
            BytecodeVmError::TemporalViolation("exit without active scope".to_string())
        })?;
        if scope.enter_pc != enter_target {
            return Err(BytecodeVmError::TemporalViolation(format!(
                "exit for instruction {enter_target} closes instruction {}",
                scope.enter_pc
            )));
        }
        self.temporal_exits = self.temporal_exits.saturating_add(1);
        Ok(())
    }

    fn rollback_temporal(&mut self) -> Result<(), BytecodeVmError> {
        while let Some(scope) = self.temporal_scopes.pop() {
            for (offset, value) in scope.snapshot.into_iter().enumerate() {
                write_cell(&mut self.present, scope.base + offset as u64, value)?;
            }
        }
        Ok(())
    }

    fn need(&self, operation: &'static str, count: usize) -> Result<(), BytecodeVmError> {
        if self.stack.len() < count {
            Err(BytecodeVmError::StackUnderflow {
                operation,
                needed: count,
                available: self.stack.len(),
            })
        } else {
            Ok(())
        }
    }

    fn pop(&mut self, operation: &'static str) -> Result<Value, BytecodeVmError> {
        self.need(operation, 1)?;
        self.stack.pop().ok_or(BytecodeVmError::StackUnderflow {
            operation,
            needed: 1,
            available: 0,
        })
    }

    fn reserve_stack(&mut self, additional: usize) -> Result<(), BytecodeVmError> {
        let new_depth = self.stack.len().checked_add(additional).ok_or(
            BytecodeVmError::StackLimitExceeded {
                limit: self.config.max_stack_depth,
            },
        )?;
        if new_depth > self.config.max_stack_depth {
            return Err(BytecodeVmError::StackLimitExceeded {
                limit: self.config.max_stack_depth,
            });
        }
        self.stack
            .try_reserve(additional)
            .map_err(|_| BytecodeVmError::StackLimitExceeded {
                limit: self.config.max_stack_depth,
            })?;
        Ok(())
    }

    fn push(&mut self, value: Value) -> Result<(), BytecodeVmError> {
        self.reserve_stack(1)?;
        self.stack.push(value);
        Ok(())
    }

    fn push_many(
        &mut self,
        values: impl IntoIterator<Item = Value>,
    ) -> Result<(), BytecodeVmError> {
        let mut values = values.into_iter();
        let (lower, upper) = values.size_hint();
        if upper == Some(lower) {
            self.reserve_stack(lower)?;
            self.stack.extend(values);
            return Ok(());
        }
        for value in values.by_ref() {
            self.push(value)?;
        }
        Ok(())
    }

    fn call_foreign(&mut self, id: ForeignId) -> Result<(), BytecodeVmError> {
        let descriptor = self
            .program
            .foreigns
            .get(id.index())
            .cloned()
            .ok_or(BytecodeVmError::InvalidProgramCounter(self.cursor.pc - 1))?;
        let argument_count = descriptor.parameters.len();
        self.need("foreign call", argument_count)?;

        let retained = self.stack.len() - argument_count;
        let final_depth = retained + usize::from(descriptor.result.is_some());
        if final_depth > self.config.max_stack_depth {
            return Err(BytecodeVmError::StackLimitExceeded {
                limit: self.config.max_stack_depth,
            });
        }
        let arguments = self.stack[retained..]
            .iter()
            .map(|value| value.val)
            .collect::<Vec<_>>();
        let host = self.foreign_host.ok_or(BytecodeVmError::ForeignLink(
            ForeignHostError::UnknownTarget { id },
        ))?;
        let result = host
            .call(&descriptor, &arguments)
            .map_err(BytecodeVmError::ForeignCall)?;
        self.stack.truncate(retained);
        if let Some(value) = result {
            self.stack.push(Value::new(value));
        }
        Ok(())
    }

    fn execute_primitive(&mut self, opcode: OpCode) -> Result<(), BytecodeVmError> {
        match opcode {
            OpCode::Nop | OpCode::Halt | OpCode::Paradox => {}
            OpCode::Pop => {
                self.pop("POP")?;
            }
            OpCode::Dup => {
                self.need("DUP", 1)?;
                self.push(self.stack[self.stack.len() - 1].clone())?;
            }
            OpCode::Swap => {
                self.need("SWAP", 2)?;
                let length = self.stack.len();
                self.stack.swap(length - 1, length - 2);
            }
            OpCode::Over => {
                self.need("OVER", 2)?;
                self.push(self.stack[self.stack.len() - 2].clone())?;
            }
            OpCode::Rot => {
                self.need("ROT", 3)?;
                let index = self.stack.len() - 3;
                let value = self.stack.remove(index);
                self.push(value)?;
            }
            OpCode::Depth => self.push(Value::new(self.stack.len() as u64))?,
            OpCode::Pick => {
                let depth = self.pop("PICK")?.val;
                if depth >= self.stack.len() as u64 {
                    return Err(BytecodeVmError::StackUnderflow {
                        operation: "PICK",
                        needed: usize::try_from(depth)
                            .unwrap_or(usize::MAX)
                            .saturating_add(1),
                        available: self.stack.len(),
                    });
                }
                let index = self.stack.len() - 1 - depth as usize;
                self.push(self.stack[index].clone())?;
            }
            OpCode::Roll => {
                let depth = self.pop("ROLL")?.val;
                if depth >= self.stack.len() as u64 {
                    return Err(BytecodeVmError::StackUnderflow {
                        operation: "ROLL",
                        needed: usize::try_from(depth)
                            .unwrap_or(usize::MAX)
                            .saturating_add(1),
                        available: self.stack.len(),
                    });
                }
                let index = self.stack.len() - 1 - depth as usize;
                let value = self.stack.remove(index);
                self.push(value)?;
            }
            OpCode::Reverse => {
                let count_word = self.pop("REVERSE")?.val;
                if count_word > self.stack.len() as u64 {
                    return Err(BytecodeVmError::StackUnderflow {
                        operation: "REVERSE",
                        needed: usize::try_from(count_word).unwrap_or(usize::MAX),
                        available: self.stack.len(),
                    });
                }
                let count = count_word as usize;
                let start = self.stack.len() - count;
                self.stack[start..].reverse();
            }
            OpCode::Exec => {
                let quotation = self.pop("EXEC")?.val;
                self.enter_quote_word(quotation, Completion::None)?;
            }
            OpCode::Dip => {
                self.need("DIP", 2)?;
                let quotation = self.pop("DIP")?.val;
                let hidden = self.pop("DIP")?;
                self.enter_quote_word(quotation, Completion::Restore(hidden))?;
            }
            OpCode::Keep => {
                self.need("KEEP", 2)?;
                let quotation = self.pop("KEEP")?.val;
                let kept = self.pop("KEEP")?;
                self.push(kept.clone())?;
                self.enter_quote_word(quotation, Completion::Restore(kept))?;
            }
            OpCode::Bi => {
                self.need("BI", 3)?;
                let second = self.pop("BI")?.val;
                let first = self.pop("BI")?.val;
                let second = self.check_quote_word(second)?;
                self.check_quote_word(first)?;
                let value = self.pop("BI")?;
                self.push(value.clone())?;
                self.enter_quote_word(
                    first,
                    Completion::BiSecond {
                        value,
                        quotation: second,
                    },
                )?;
            }
            OpCode::Rec => {
                let quotation = self.pop("REC")?;
                let id = quotation.val;
                self.push(quotation)?;
                self.enter_quote_word(id, Completion::None)?;
            }
            OpCode::StrRev => self.string_reverse()?,
            OpCode::StrCat => self.string_concat()?,
            OpCode::StrSplit => self.string_split()?,
            OpCode::Add => self.binary("ADD", |a, b| a + b)?,
            OpCode::Sub => self.binary("SUB", |a, b| a - b)?,
            OpCode::Mul => self.binary("MUL", |a, b| a * b)?,
            OpCode::Div => self.binary("DIV", |a, b| a / b)?,
            OpCode::Mod => self.binary("MOD", |a, b| a % b)?,
            OpCode::Neg => {
                let value = self.pop("NEG")?;
                self.push(Value {
                    val: value.val.wrapping_neg(),
                    prov: value.prov,
                })?;
            }
            OpCode::Abs => {
                let value = self.pop("ABS")?;
                self.push(value.abs())?;
            }
            OpCode::Min => self.binary("MIN", |a, b| Value {
                val: a.val.min(b.val),
                prov: a.prov.merge(&b.prov),
            })?,
            OpCode::Max => self.binary("MAX", |a, b| Value {
                val: a.val.max(b.val),
                prov: a.prov.merge(&b.prov),
            })?,
            OpCode::Sign => {
                let value = self.pop("SIGN")?;
                self.push(value.signum())?;
            }
            OpCode::Assert => self.assert()?,
            OpCode::Not => {
                let value = self.pop("NOT")?;
                self.push(Value {
                    val: u64::from(value.val == 0),
                    prov: value.prov,
                })?;
            }
            OpCode::And => self.binary("AND", |a, b| a & b)?,
            OpCode::Or => self.binary("OR", |a, b| a | b)?,
            OpCode::Xor => self.binary("XOR", |a, b| a ^ b)?,
            OpCode::Shl => self.shift("SHL", true)?,
            OpCode::Shr => self.shift("SHR", false)?,
            OpCode::Eq => self.compare("EQ", |a, b| a == b)?,
            OpCode::Neq => self.compare("NEQ", |a, b| a != b)?,
            OpCode::Lt => self.compare("LT", |a, b| a < b)?,
            OpCode::Gt => self.compare("GT", |a, b| a > b)?,
            OpCode::Lte => self.compare("LTE", |a, b| a <= b)?,
            OpCode::Gte => self.compare("GTE", |a, b| a >= b)?,
            OpCode::Slt => self.signed_compare("SLT", |a, b| a < b)?,
            OpCode::Sgt => self.signed_compare("SGT", |a, b| a > b)?,
            OpCode::Slte => self.signed_compare("SLTE", |a, b| a <= b)?,
            OpCode::Sgte => self.signed_compare("SGTE", |a, b| a >= b)?,
            OpCode::Oracle => self.oracle()?,
            OpCode::Prophecy => self.prophecy()?,
            OpCode::PresentRead => self.present_read()?,
            OpCode::Pack => self.pack()?,
            OpCode::Unpack => self.unpack()?,
            OpCode::Index => self.index()?,
            OpCode::Store => self.store()?,
            OpCode::Input => self.input()?,
            OpCode::Clock => self.clock_input()?,
            OpCode::Random => self.random_input()?,
            OpCode::FileOpen => self.file_open()?,
            OpCode::FileRead => self.file_read()?,
            OpCode::FileWrite => self.file_write()?,
            OpCode::FileSeek => self.file_seek()?,
            OpCode::FileFlush => self.file_flush()?,
            OpCode::FileClose => self.file_close()?,
            OpCode::FileExists => self.file_exists()?,
            OpCode::FileSize => self.file_size()?,
            OpCode::TcpConnect => self.tcp_connect()?,
            OpCode::SocketSend => self.socket_send()?,
            OpCode::SocketRecv => self.socket_recv()?,
            OpCode::SocketClose => self.socket_close()?,
            OpCode::ProcExec => self.proc_exec()?,
            OpCode::Sleep => self.sleep()?,
            OpCode::Output => {
                let value = self.pop("OUTPUT")?;
                self.push_output(OutputItem::Val(value))?;
            }
            OpCode::Emit => {
                let value = self.pop("EMIT")?;
                self.push_output(OutputItem::Char((value.val % 256) as u8))?;
            }
            OpCode::VecNew => self.vec_new()?,
            OpCode::VecPush => self.vec_push()?,
            OpCode::VecPop => self.vec_pop()?,
            OpCode::VecGet => self.vec_get()?,
            OpCode::VecSet => self.vec_set()?,
            OpCode::VecLen => self.vec_len()?,
            OpCode::HashNew => self.hash_new()?,
            OpCode::HashPut => self.hash_put()?,
            OpCode::HashGet => self.hash_get()?,
            OpCode::HashDel => self.hash_del()?,
            OpCode::HashHas => self.hash_has()?,
            OpCode::HashLen => self.hash_len()?,
            OpCode::SetNew => self.set_new()?,
            OpCode::SetAdd => self.set_add()?,
            OpCode::SetHas => self.set_has()?,
            OpCode::SetDel => self.set_del()?,
            OpCode::SetLen => self.set_len()?,
            OpCode::BufferNew => self.buffer_new()?,
            OpCode::BufferFromStack => self.buffer_from_stack()?,
            OpCode::BufferToStack => self.buffer_to_stack()?,
            OpCode::BufferLen => self.buffer_len()?,
            OpCode::BufferReadByte => self.buffer_read_byte()?,
            OpCode::BufferWriteByte => self.buffer_write_byte()?,
            OpCode::BufferFree => self.buffer_free()?,
            OpCode::FFICall | OpCode::FFICallNamed => {
                return Err(BytecodeVmError::UnsupportedPrimitive(opcode));
            }
        }
        Ok(())
    }

    fn binary(
        &mut self,
        operation: &'static str,
        apply: impl FnOnce(Value, Value) -> Value,
    ) -> Result<(), BytecodeVmError> {
        self.need(operation, 2)?;
        let right = self.pop(operation)?;
        let left = self.pop(operation)?;
        self.push(apply(left, right))
    }

    fn compare(
        &mut self,
        operation: &'static str,
        compare: impl FnOnce(u64, u64) -> bool,
    ) -> Result<(), BytecodeVmError> {
        self.binary(operation, |left, right| {
            Value::from_bool_with_prov(compare(left.val, right.val), left.prov.merge(&right.prov))
        })
    }

    fn signed_compare(
        &mut self,
        operation: &'static str,
        compare: impl FnOnce(i64, i64) -> bool,
    ) -> Result<(), BytecodeVmError> {
        self.binary(operation, |left, right| {
            Value::from_bool_with_prov(
                compare(left.as_signed(), right.as_signed()),
                left.prov.merge(&right.prov),
            )
        })
    }

    fn shift(&mut self, operation: &'static str, left: bool) -> Result<(), BytecodeVmError> {
        self.binary(operation, |value, count| Value {
            val: if left {
                value.val.wrapping_shl((count.val % 64) as u32)
            } else {
                value.val.wrapping_shr((count.val % 64) as u32)
            },
            prov: value.prov.merge(&count.prov),
        })
    }

    fn string_reverse(&mut self) -> Result<(), BytecodeVmError> {
        let length_value = self.pop("STR_REV")?;
        let length = self.checked_stack_count("STR_REV", length_value.val)?;
        let start = self.stack.len() - length;
        self.stack[start..].reverse();
        self.push(length_value)
    }

    fn string_concat(&mut self) -> Result<(), BytecodeVmError> {
        let length2_value = self.pop("STR_CAT")?;
        let length2 = self.checked_stack_count("STR_CAT", length2_value.val)?;
        let split = self.stack.len() - length2;
        let second = self.stack.split_off(split);
        let length1_value = self.pop("STR_CAT")?;
        self.checked_stack_count("STR_CAT", length1_value.val)?;
        self.push_many(second)?;
        self.push(Value {
            val: length1_value.val.wrapping_add(length2_value.val),
            prov: length1_value.prov.merge(&length2_value.prov),
        })
    }

    fn string_split(&mut self) -> Result<(), BytecodeVmError> {
        self.need("STR_SPLIT", 2)?;
        let delimiter = self.pop("STR_SPLIT")?;
        let length_value = self.pop("STR_SPLIT")?;
        let length = self.checked_stack_count("STR_SPLIT", length_value.val)?;
        let split = self.stack.len() - length;
        let separators = self.stack[split..]
            .iter()
            .filter(|character| character.val == delimiter.val)
            .count();
        let part_count = separators
            .checked_add(1)
            .ok_or(BytecodeVmError::StackLimitExceeded {
                limit: self.config.max_stack_depth,
            })?;
        let output_count = length
            .checked_sub(separators)
            .and_then(|characters| characters.checked_add(part_count))
            .and_then(|values| values.checked_add(1))
            .ok_or(BytecodeVmError::StackLimitExceeded {
                limit: self.config.max_stack_depth,
            })?;
        let final_depth =
            split
                .checked_add(output_count)
                .ok_or(BytecodeVmError::StackLimitExceeded {
                    limit: self.config.max_stack_depth,
                })?;
        if final_depth > self.config.max_stack_depth {
            return Err(BytecodeVmError::StackLimitExceeded {
                limit: self.config.max_stack_depth,
            });
        }
        if final_depth > self.stack.len() {
            self.stack
                .try_reserve(final_depth - self.stack.len())
                .map_err(|_| BytecodeVmError::StackLimitExceeded {
                    limit: self.config.max_stack_depth,
                })?;
        }
        let mut values = Vec::new();
        values.try_reserve_exact(output_count).map_err(|_| {
            BytecodeVmError::StackLimitExceeded {
                limit: self.config.max_stack_depth,
            }
        })?;
        let mut part_length = 0usize;
        for character in self.stack.drain(split..) {
            if character.val == delimiter.val {
                values.push(Value::new(part_length as u64));
                part_length = 0;
            } else {
                values.push(character);
                part_length += 1;
            }
        }
        values.push(Value::new(part_length as u64));
        values.push(Value::new(part_count as u64));
        debug_assert_eq!(values.len(), output_count);
        self.stack.extend(values);
        Ok(())
    }

    fn assert(&mut self) -> Result<(), BytecodeVmError> {
        let length_word = self.pop("ASSERT")?.val;
        let length = self.checked_stack_count("ASSERT", length_word)?;
        let split = self.stack.len() - length;
        let characters = self.stack.split_off(split);
        let condition = self.pop("ASSERT")?;
        if condition.to_bool() {
            Ok(())
        } else {
            let bytes: Vec<u8> = characters
                .into_iter()
                .map(|character| character.val as u8)
                .collect();
            Err(BytecodeVmError::AssertionFailed(
                String::from_utf8_lossy(&bytes).into_owned(),
            ))
        }
    }

    fn checked_stack_count(
        &self,
        operation: &'static str,
        count: u64,
    ) -> Result<usize, BytecodeVmError> {
        if count > self.stack.len() as u64 {
            return Err(BytecodeVmError::StackUnderflow {
                operation,
                needed: usize::try_from(count).unwrap_or(usize::MAX),
                available: self.stack.len(),
            });
        }
        Ok(count as usize)
    }

    fn active_scope(&self) -> Option<&TemporalScope> {
        self.temporal_scopes.last()
    }

    fn resolve_address(&self, raw: u64) -> Result<u64, BytecodeVmError> {
        let translated = if let Some(scope) = self.active_scope() {
            let local = self.apply_bounds(raw, scope.size as usize)?;
            scope.base.checked_add(local).ok_or_else(|| {
                BytecodeVmError::TemporalViolation("translated address overflows u64".to_string())
            })?
        } else {
            raw
        };
        self.apply_bounds(translated, self.present.len())
    }

    fn apply_bounds(&self, raw: u64, length: usize) -> Result<u64, BytecodeVmError> {
        if raw < length as u64 {
            return Ok(raw);
        }
        match self.config.memory_bounds {
            BoundsPolicy::Wrap => Ok(raw % length as u64),
            BoundsPolicy::Clamp => Ok(length.saturating_sub(1) as u64),
            BoundsPolicy::Error => Err(BytecodeVmError::MemoryOutOfBounds {
                address: raw,
                memory_cells: length,
            }),
        }
    }

    fn scoped_value(&self, mut value: Value) -> Value {
        if let Some(scope) = self.active_scope() {
            if scope.cell_bits < 64 {
                value.val &= (1u64 << scope.cell_bits) - 1;
            }
        }
        value
    }

    fn oracle(&mut self) -> Result<(), BytecodeVmError> {
        let address_value = self.pop("ORACLE")?;
        let address = self.resolve_address(address_value.val)?;
        let mut value = read_cell(self.anamnesis, address)?;
        value.prov = value
            .prov
            .merge(&Provenance::single(address))
            .merge(&address_value.prov);
        self.push(value)
    }

    fn prophecy(&mut self) -> Result<(), BytecodeVmError> {
        self.need("PROPHECY", 2)?;
        let address_value = self.pop("PROPHECY")?;
        let value = self.pop("PROPHECY")?;
        let address = self.resolve_address(address_value.val)?;
        let value = self.scoped_value(value);
        write_cell(&mut self.present, address, value)?;
        Ok(())
    }

    fn present_read(&mut self) -> Result<(), BytecodeVmError> {
        let address_value = self.pop("PRESENT")?;
        let address = self.resolve_address(address_value.val)?;
        let mut value = read_cell(&self.present, address)?;
        value.prov = value.prov.merge(&address_value.prov);
        self.push(value)
    }

    fn pack(&mut self) -> Result<(), BytecodeVmError> {
        self.need("PACK", 2)?;
        let count_word = self.pop("PACK")?.val;
        let base = self.pop("PACK")?.val;
        let count = self.checked_stack_count("PACK", count_word)?;
        let split = self.stack.len() - count;
        let values = self.stack.split_off(split);
        for (offset, value) in values.into_iter().enumerate() {
            let address = self.resolve_address(base.wrapping_add(offset as u64))?;
            let value = self.scoped_value(value);
            write_cell(&mut self.present, address, value)?;
        }
        Ok(())
    }

    fn unpack(&mut self) -> Result<(), BytecodeVmError> {
        self.need("UNPACK", 2)?;
        let count_word = self.pop("UNPACK")?.val;
        let base = self.pop("UNPACK")?.val;
        let count =
            usize::try_from(count_word).map_err(|_| BytecodeVmError::StackLimitExceeded {
                limit: self.config.max_stack_depth,
            })?;
        self.reserve_stack(count)?;
        for offset in 0..count {
            let address = self.resolve_address(base.wrapping_add(offset as u64))?;
            self.stack.push(read_cell(&self.present, address)?);
        }
        Ok(())
    }

    fn index(&mut self) -> Result<(), BytecodeVmError> {
        self.need("INDEX", 2)?;
        let offset = self.pop("INDEX")?;
        let base = self.pop("INDEX")?;
        let address = self.resolve_address(base.val.wrapping_add(offset.val))?;
        self.push(read_cell(&self.present, address)?)
    }

    fn store(&mut self) -> Result<(), BytecodeVmError> {
        self.need("STORE", 3)?;
        let offset = self.pop("STORE")?;
        let base = self.pop("STORE")?;
        let value = self.pop("STORE")?;
        let address = self.resolve_address(base.val.wrapping_add(offset.val))?;
        let value = self.scoped_value(value);
        write_cell(&mut self.present, address, value)?;
        Ok(())
    }

    fn input(&mut self) -> Result<(), BytecodeVmError> {
        let value = if let Some(value) = self.config.input.get(self.input_cursor).copied() {
            value
        } else if self.config.allow_interactive_input {
            crate::vm::executor::read_input_interactive().map_err(|error| match error {
                crate::vm::executor::InteractiveInputError::EndOfFile => {
                    BytecodeVmError::InteractiveInputEof
                }
                crate::vm::executor::InteractiveInputError::TooLong { limit } => {
                    BytecodeVmError::InteractiveInputTooLong { limit }
                }
                crate::vm::executor::InteractiveInputError::InvalidValue => {
                    BytecodeVmError::InteractiveInputInvalid
                }
                crate::vm::executor::InteractiveInputError::Io(message) => {
                    BytecodeVmError::InteractiveInputIo { message }
                }
            })?
        } else {
            return Err(BytecodeVmError::InputExhausted {
                consumed: self.input_cursor,
            });
        };
        self.input_cursor += 1;
        self.inputs_consumed.push(value);
        self.push(Value::new(value))
    }

    fn clock_input(&mut self) -> Result<(), BytecodeVmError> {
        let value = if let Some(value) = self
            .config
            .clock_input
            .get(self.clock_input_cursor)
            .copied()
        {
            value
        } else if self.config.allow_live_system_inputs {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64
        } else {
            return Err(BytecodeVmError::ClockInputExhausted {
                consumed: self.clock_input_cursor,
            });
        };
        self.clock_input_cursor += 1;
        self.clock_inputs_consumed.push(value);
        self.push(Value::new(value))
    }

    fn random_input(&mut self) -> Result<(), BytecodeVmError> {
        let value = if let Some(value) = self
            .config
            .random_input
            .get(self.random_input_cursor)
            .copied()
        {
            value
        } else if self.config.allow_live_system_inputs {
            crate::vm::executor::random_u64()
        } else {
            return Err(BytecodeVmError::RandomInputExhausted {
                consumed: self.random_input_cursor,
            });
        };
        self.random_input_cursor += 1;
        self.random_inputs_consumed.push(value);
        self.push(Value::new(value))
    }

    fn pop_path(&mut self, operation: &'static str) -> Result<String, BytecodeVmError> {
        let length_word = self.pop(operation)?.val;
        let length =
            usize::try_from(length_word).map_err(|_| BytecodeVmError::AllocationLimit {
                what: "file path byte",
                limit: self.config.max_collection_items,
            })?;
        if length > self.config.max_collection_items {
            return Err(BytecodeVmError::AllocationLimit {
                what: "file path byte",
                limit: self.config.max_collection_items,
            });
        }
        self.need(operation, length)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| BytecodeVmError::AllocationLimit {
                what: "file path byte",
                limit: self.config.max_collection_items,
            })?;
        for _ in 0..length {
            bytes.push(self.pop(operation)?.val as u8);
        }
        bytes.reverse();
        Ok(bytes.into_iter().map(char::from).collect())
    }

    fn file_open(&mut self) -> Result<(), BytecodeVmError> {
        let mode = FileMode::from_u64(self.pop("FILE_OPEN")?.val);
        let path = self.pop_path("FILE_OPEN")?;
        if mode.can_read() || mode.0 == 0 {
            self.files.require_read(&path)?;
            self.files.ensure_snapshot(&path)?;
        } else {
            // A write-only capability may use an eagerly frozen base image to
            // preserve offsets without gaining language-visible read access.
            self.files.ensure_frozen_base(&path)?;
        }
        if mode.is_destructive() {
            self.files.require_write(&path)?;
        }

        let initial = self.files.initial_snapshot(&path)?;
        let was_missing = self.files.image(&path)?.is_none();
        if was_missing && !mode.can_create() {
            return Err(BytecodeVmError::FileOperation {
                operation: "FILE_OPEN",
                path,
                message: "path does not exist in the frozen snapshot".to_string(),
            });
        }
        if mode.can_create() && !mode.can_write() {
            return Err(BytecodeVmError::FileOperation {
                operation: "FILE_OPEN",
                path,
                message: "CREATE requires WRITE or APPEND".to_string(),
            });
        }
        let truncate = mode.0 & FileMode::TRUNCATE.0 != 0;
        if truncate && !mode.can_write() {
            return Err(BytecodeVmError::FileOperation {
                operation: "FILE_OPEN",
                path,
                message: "TRUNCATE requires WRITE or APPEND".to_string(),
            });
        }
        if was_missing || truncate {
            self.files.replace_image(&path, Some(Vec::new()))?;
            self.files.stage(EffectIntent::FileSetLength {
                path: path.clone(),
                length: 0,
                initial,
            })?;
        }
        let position = if mode.0 & FileMode::APPEND.0 != 0 {
            self.files
                .image(&path)?
                .as_ref()
                .map_or(0, |contents| contents.len() as u64)
        } else {
            0
        };
        let handle = self.files.allocate_handle(VirtualOpenFile {
            path,
            mode,
            position,
        })?;
        self.push(Value::new(handle.0))
    }

    fn file_read(&mut self) -> Result<(), BytecodeVmError> {
        let max_bytes_word = self.pop("FILE_READ")?.val;
        let max_bytes =
            usize::try_from(max_bytes_word).map_err(|_| BytecodeVmError::AllocationLimit {
                what: "file read byte",
                limit: self.config.max_collection_items,
            })?;
        if max_bytes > self.config.max_collection_items {
            return Err(BytecodeVmError::AllocationLimit {
                what: "file read byte",
                limit: self.config.max_collection_items,
            });
        }
        let handle = VirtualFileHandle(self.pop("FILE_READ")?.val);
        let open = self.files.handle(handle)?.clone();
        self.files.require_read(&open.path)?;
        if !open.mode.can_read() && open.mode.0 != 0 {
            return Err(BytecodeVmError::FileOperation {
                operation: "FILE_READ",
                path: open.path,
                message: "file was not opened for reading".to_string(),
            });
        }
        let start = usize::try_from(open.position).unwrap_or(usize::MAX);
        let (end, bytes_read) = {
            let contents = self.files.image(&open.path)?.as_ref().ok_or_else(|| {
                BytecodeVmError::FileOperation {
                    operation: "FILE_READ",
                    path: open.path.clone(),
                    message: "path is missing".to_string(),
                }
            })?;
            let end = contents.len().min(start.saturating_add(max_bytes));
            (end, end.saturating_sub(start.min(end)))
        };
        self.preflight_buffer_allocation(bytes_read)?;
        let data = if bytes_read == 0 {
            Vec::new()
        } else {
            self.files
                .image(&open.path)?
                .as_ref()
                .expect("checked above")[start..end]
                .to_vec()
        };
        let new_position = open.position.saturating_add(bytes_read as u64);
        self.files.handle_mut(handle)?.position = new_position;
        let buffer = self.allocate_buffer(ByteBuffer { data, position: 0 })?;
        self.reserve_stack(3)?;
        self.stack.push(Value::new(handle.0));
        self.stack.push(Value::new(buffer.0));
        self.stack.push(Value::new(bytes_read as u64));
        Ok(())
    }

    fn file_write(&mut self) -> Result<(), BytecodeVmError> {
        let buffer = BufferHandle(self.pop("FILE_WRITE")?.val);
        let handle = VirtualFileHandle(self.pop("FILE_WRITE")?.val);
        let bytes = self.buffers.get(buffer)?.data.clone();
        let open = self.files.handle(handle)?.clone();
        if !open.mode.can_write() {
            return Err(BytecodeVmError::FileOperation {
                operation: "FILE_WRITE",
                path: open.path,
                message: "file was not opened for writing".to_string(),
            });
        }
        self.files.require_write(&open.path)?;
        let mut contents = self
            .files
            .image(&open.path)?
            .as_ref()
            .cloned()
            .ok_or_else(|| BytecodeVmError::FileOperation {
                operation: "FILE_WRITE",
                path: open.path.clone(),
                message: "path is missing".to_string(),
            })?;
        let offset = if open.mode.0 & FileMode::APPEND.0 != 0 {
            contents.len() as u64
        } else {
            open.position
        };
        let start = usize::try_from(offset).map_err(|_| BytecodeVmError::AllocationLimit {
            what: "virtual file byte",
            limit: self.config.max_collection_items,
        })?;
        let end = start
            .checked_add(bytes.len())
            .ok_or(BytecodeVmError::AllocationLimit {
                what: "virtual file byte",
                limit: self.config.max_collection_items,
            })?;
        if end > self.config.max_collection_items {
            return Err(BytecodeVmError::AllocationLimit {
                what: "virtual file byte",
                limit: self.config.max_collection_items,
            });
        }
        if contents.len() < end {
            contents.resize(end, 0);
        }
        contents[start..end].copy_from_slice(&bytes);
        self.files.replace_image(&open.path, Some(contents))?;
        self.files.stage(EffectIntent::FileWrite {
            path: open.path.clone(),
            offset,
            bytes: bytes.clone(),
            initial: self.files.initial_snapshot(&open.path)?,
        })?;
        self.files.handle_mut(handle)?.position = end as u64;
        self.reserve_stack(2)?;
        self.stack.push(Value::new(handle.0));
        self.stack.push(Value::new(bytes.len() as u64));
        Ok(())
    }

    fn file_seek(&mut self) -> Result<(), BytecodeVmError> {
        let origin = SeekOrigin::from_u64(self.pop("FILE_SEEK")?.val);
        let offset = self.pop("FILE_SEEK")?.val as i64;
        let handle = VirtualFileHandle(self.pop("FILE_SEEK")?.val);
        let open = self.files.handle(handle)?.clone();
        if origin != SeekOrigin::Start {
            self.files.require_read(&open.path)?;
        }
        let end = self
            .files
            .image(&open.path)?
            .as_ref()
            .map_or(0u64, |contents| contents.len() as u64);
        let position = match origin {
            SeekOrigin::Start => Some(offset as u64),
            SeekOrigin::Current => checked_seek(open.position, offset),
            SeekOrigin::End => checked_seek(end, offset),
        }
        .ok_or_else(|| BytecodeVmError::FileOperation {
            operation: "FILE_SEEK",
            path: open.path,
            message: "seek would move before the beginning of the file".to_string(),
        })?;
        self.files.handle_mut(handle)?.position = position;
        self.reserve_stack(2)?;
        self.stack.push(Value::new(handle.0));
        self.stack.push(Value::new(position));
        Ok(())
    }

    fn file_flush(&mut self) -> Result<(), BytecodeVmError> {
        let handle = VirtualFileHandle(self.pop("FILE_FLUSH")?.val);
        self.files.handle(handle)?;
        self.push(Value::new(handle.0))
    }

    fn file_close(&mut self) -> Result<(), BytecodeVmError> {
        let handle = VirtualFileHandle(self.pop("FILE_CLOSE")?.val);
        self.files.close(handle)
    }

    fn file_exists(&mut self) -> Result<(), BytecodeVmError> {
        let path = self.pop_path("FILE_EXISTS")?;
        self.files.require_read(&path)?;
        self.files.ensure_snapshot(&path)?;
        self.push(Value::new(u64::from(self.files.image(&path)?.is_some())))
    }

    fn file_size(&mut self) -> Result<(), BytecodeVmError> {
        let path = self.pop_path("FILE_SIZE")?;
        self.files.require_read(&path)?;
        self.files.ensure_snapshot(&path)?;
        let size = self
            .files
            .image(&path)?
            .as_ref()
            .map_or(0, |contents| contents.len() as u64);
        self.push(Value::new(size))
    }

    fn pop_text(
        &mut self,
        operation: &'static str,
        allocation: &'static str,
    ) -> Result<String, BytecodeVmError> {
        let length_word = self.pop(operation)?.val;
        let length =
            usize::try_from(length_word).map_err(|_| BytecodeVmError::AllocationLimit {
                what: allocation,
                limit: self.config.max_collection_items,
            })?;
        if length > self.config.max_collection_items {
            return Err(BytecodeVmError::AllocationLimit {
                what: allocation,
                limit: self.config.max_collection_items,
            });
        }
        self.need(operation, length)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| BytecodeVmError::AllocationLimit {
                what: allocation,
                limit: self.config.max_collection_items,
            })?;
        for _ in 0..length {
            bytes.push(self.pop(operation)?.val as u8);
        }
        bytes.reverse();
        Ok(bytes.into_iter().map(char::from).collect())
    }

    fn tcp_connect(&mut self) -> Result<(), BytecodeVmError> {
        self.need("TCP_CONNECT", 2)?;
        let port = self.pop("TCP_CONNECT")?.val as u16;
        let host = self.pop_text("TCP_CONNECT", "endpoint byte")?;
        let endpoint = format!("{host}:{port}");
        let handle = self.network.connect(endpoint)?;
        self.push(Value::new(handle.0))
    }

    fn socket_send(&mut self) -> Result<(), BytecodeVmError> {
        self.need("SOCKET_SEND", 2)?;
        let buffer = BufferHandle(self.pop("SOCKET_SEND")?.val);
        let socket = VirtualSocketHandle(self.pop("SOCKET_SEND")?.val);
        let bytes = self.buffers.get(buffer)?.data.clone();
        let endpoint = self.network.socket(socket)?.endpoint.clone();
        self.network.require_send(&endpoint)?;
        self.files.stage(EffectIntent::NetworkSend {
            endpoint,
            bytes: bytes.clone(),
        })?;
        self.push_many([Value::new(socket.0), Value::new(bytes.len() as u64)])
    }

    fn socket_recv(&mut self) -> Result<(), BytecodeVmError> {
        self.need("SOCKET_RECV", 2)?;
        let maximum_word = self.pop("SOCKET_RECV")?.val;
        let maximum =
            usize::try_from(maximum_word).map_err(|_| BytecodeVmError::AllocationLimit {
                what: "socket receive byte",
                limit: self.config.max_collection_items,
            })?;
        if maximum > self.config.max_collection_items {
            return Err(BytecodeVmError::AllocationLimit {
                what: "socket receive byte",
                limit: self.config.max_collection_items,
            });
        }
        let socket = VirtualSocketHandle(self.pop("SOCKET_RECV")?.val);
        let expected = self.network.receive_len(socket, maximum)?;
        self.preflight_buffer_allocation(expected)?;
        let bytes = self.network.receive(socket, maximum)?;
        let received = bytes.len();
        let buffer = self.allocate_buffer(ByteBuffer {
            data: bytes,
            position: 0,
        })?;
        self.push_many([
            Value::new(socket.0),
            Value::new(buffer.0),
            Value::new(received as u64),
        ])
    }

    fn socket_close(&mut self) -> Result<(), BytecodeVmError> {
        let socket = VirtualSocketHandle(self.pop("SOCKET_CLOSE")?.val);
        self.network.close(socket)
    }

    fn proc_exec(&mut self) -> Result<(), BytecodeVmError> {
        let command = self.pop_text("PROC_EXEC", "command byte")?;
        let output_len = self.processes.output_len(&command)?;
        self.preflight_buffer_allocation(output_len)?;
        let result = self.processes.execute(&command)?;
        let output = self.allocate_buffer(ByteBuffer {
            data: result.output,
            position: 0,
        })?;
        let (program, arguments) = shell_invocation(command);
        self.files
            .stage(EffectIntent::ProcessSpawn { program, arguments })?;
        self.push_many([Value::new(output.0), Value::new(result.exit_code as u64)])
    }

    fn sleep(&mut self) -> Result<(), BytecodeVmError> {
        let milliseconds = self.pop("SLEEP")?.val;
        if self
            .config
            .max_sleep_milliseconds
            .is_none_or(|maximum| milliseconds > maximum)
        {
            return Err(BytecodeVmError::SleepCapabilityDenied {
                milliseconds,
                maximum: self.config.max_sleep_milliseconds,
            });
        }
        self.files.stage(EffectIntent::Sleep { milliseconds })
    }

    fn push_output(&mut self, item: OutputItem) -> Result<(), BytecodeVmError> {
        if self.output.len() >= self.config.max_output_items {
            return Err(BytecodeVmError::AllocationLimit {
                what: "output",
                limit: self.config.max_output_items,
            });
        }
        let charge = item.retained_size_charge();
        let next_bytes =
            self.output_bytes
                .checked_add(charge)
                .ok_or(BytecodeVmError::AllocationLimit {
                    what: "output byte",
                    limit: self.config.max_output_bytes,
                })?;
        if next_bytes > self.config.max_output_bytes {
            return Err(BytecodeVmError::AllocationLimit {
                what: "output byte",
                limit: self.config.max_output_bytes,
            });
        }
        self.output.push(item);
        self.output_bytes = next_bytes;
        Ok(())
    }

    fn check_collection_count(
        count: usize,
        config: &BytecodeVmConfig,
        what: &'static str,
    ) -> Result<(), BytecodeVmError> {
        if count >= config.max_collections {
            Err(BytecodeVmError::AllocationLimit {
                what,
                limit: config.max_collections,
            })
        } else {
            Ok(())
        }
    }

    fn retain_dynamic_bytes(&mut self, charge: usize) -> Result<(), BytecodeVmError> {
        let next =
            self.dynamic_bytes
                .checked_add(charge)
                .ok_or(BytecodeVmError::AllocationLimit {
                    what: "aggregate dynamic byte",
                    limit: self.config.max_dynamic_bytes,
                })?;
        if next > self.config.max_dynamic_bytes {
            return Err(BytecodeVmError::AllocationLimit {
                what: "aggregate dynamic byte",
                limit: self.config.max_dynamic_bytes,
            });
        }
        self.dynamic_bytes = next;
        Ok(())
    }

    fn release_dynamic_bytes(&mut self, charge: usize) {
        self.dynamic_bytes = self.dynamic_bytes.saturating_sub(charge);
    }

    fn preflight_buffer_allocation(&self, capacity: usize) -> Result<(), BytecodeVmError> {
        if self.buffers.slots.len() >= self.config.max_collections {
            return Err(BytecodeVmError::AllocationLimit {
                what: "buffer",
                limit: self.config.max_collections,
            });
        }
        let charge = capacity.saturating_add(64);
        if self
            .dynamic_bytes
            .checked_add(charge)
            .is_none_or(|bytes| bytes > self.config.max_dynamic_bytes)
        {
            return Err(BytecodeVmError::AllocationLimit {
                what: "aggregate dynamic byte",
                limit: self.config.max_dynamic_bytes,
            });
        }
        Ok(())
    }

    fn allocate_buffer(&mut self, buffer: ByteBuffer) -> Result<BufferHandle, BytecodeVmError> {
        self.preflight_buffer_allocation(buffer.data.capacity())?;
        self.retain_dynamic_bytes(buffer.data.capacity().saturating_add(64))?;
        self.buffers.allocate(buffer, self.config.max_collections)
    }

    fn vec_new(&mut self) -> Result<(), BytecodeVmError> {
        Self::check_collection_count(self.data.vectors.count(), self.config, "vector")?;
        self.retain_dynamic_bytes(64)?;
        let handle = self.data.vectors.alloc();
        self.push(Value::new(handle))
    }

    fn vec_push(&mut self) -> Result<(), BytecodeVmError> {
        self.need("VEC_PUSH", 2)?;
        let value = self.pop("VEC_PUSH")?;
        let handle = self.pop("VEC_PUSH")?.val;
        let length = self
            .data
            .vectors
            .len(handle)
            .map_err(|error| BytecodeVmError::DataStructure(error.to_string()))?;
        if length >= self.config.max_collection_items as u64 {
            return Err(BytecodeVmError::AllocationLimit {
                what: "vector item",
                limit: self.config.max_collection_items,
            });
        }
        let (capacity_growth, reserve_additional) = {
            let vector = self.data.vectors.get(handle).ok_or_else(|| {
                BytecodeVmError::DataStructure(format!("invalid vector handle: {handle}"))
            })?;
            if vector.len() < vector.capacity() {
                (0, 0)
            } else {
                let target = vector
                    .capacity()
                    .max(2)
                    .saturating_mul(2)
                    .min(self.config.max_collection_items)
                    .max(vector.len().saturating_add(1));
                (
                    target
                        .saturating_sub(vector.capacity())
                        .saturating_mul(std::mem::size_of::<Value>()),
                    target.saturating_sub(vector.len()),
                )
            }
        };
        let charge = capacity_growth.saturating_add(value.provenance_retained_size_charge());
        self.retain_dynamic_bytes(charge)?;
        let vector = self.data.vectors.get_mut(handle).ok_or_else(|| {
            BytecodeVmError::DataStructure(format!("invalid vector handle: {handle}"))
        })?;
        if reserve_additional != 0 {
            vector.try_reserve_exact(reserve_additional).map_err(|_| {
                BytecodeVmError::AllocationLimit {
                    what: "aggregate dynamic byte",
                    limit: self.config.max_dynamic_bytes,
                }
            })?;
        }
        vector.push(value);
        self.push(Value::new(handle))
    }

    fn vec_pop(&mut self) -> Result<(), BytecodeVmError> {
        let handle = self.pop("VEC_POP")?.val;
        let value = self
            .data
            .vectors
            .pop(handle)
            .map_err(|error| BytecodeVmError::DataStructure(error.to_string()))?;
        self.release_dynamic_bytes(value.provenance_retained_size_charge());
        self.push_many([Value::new(handle), value])
    }

    fn vec_get(&mut self) -> Result<(), BytecodeVmError> {
        self.need("VEC_GET", 2)?;
        let index = self.pop("VEC_GET")?.val;
        let handle = self.pop("VEC_GET")?.val;
        let value = self
            .data
            .vectors
            .get_at(handle, index)
            .map_err(|error| BytecodeVmError::DataStructure(error.to_string()))?;
        self.push_many([Value::new(handle), value])
    }

    fn vec_set(&mut self) -> Result<(), BytecodeVmError> {
        self.need("VEC_SET", 3)?;
        let index = self.pop("VEC_SET")?.val;
        let value = self.pop("VEC_SET")?;
        let handle = self.pop("VEC_SET")?.val;
        let old = self
            .data
            .vectors
            .get(handle)
            .and_then(|vector| vector.get(index as usize))
            .cloned()
            .ok_or_else(|| {
                BytecodeVmError::DataStructure(format!(
                    "vector index {index} is invalid for handle {handle}"
                ))
            })?;
        let old_charge = old.provenance_retained_size_charge();
        let new_charge = value.provenance_retained_size_charge();
        if new_charge > old_charge {
            self.retain_dynamic_bytes(new_charge - old_charge)?;
        }
        self.data
            .vectors
            .set_at(handle, index, value)
            .map_err(|error| BytecodeVmError::DataStructure(error.to_string()))?;
        if old_charge > new_charge {
            self.release_dynamic_bytes(old_charge - new_charge);
        }
        self.push(Value::new(handle))
    }

    fn vec_len(&mut self) -> Result<(), BytecodeVmError> {
        let handle = self.pop("VEC_LEN")?.val;
        let length = self
            .data
            .vectors
            .len(handle)
            .map_err(|error| BytecodeVmError::DataStructure(error.to_string()))?;
        self.push_many([Value::new(handle), Value::new(length)])
    }

    fn hash_new(&mut self) -> Result<(), BytecodeVmError> {
        Self::check_collection_count(self.data.hashes.count(), self.config, "hash table")?;
        self.retain_dynamic_bytes(64)?;
        let handle = self.data.hashes.alloc();
        self.push(Value::new(handle))
    }

    fn hash_put(&mut self) -> Result<(), BytecodeVmError> {
        self.need("HASH_PUT", 3)?;
        let value = self.pop("HASH_PUT")?;
        let key = self.pop("HASH_PUT")?.val;
        let handle = self.pop("HASH_PUT")?.val;
        let old = self
            .data
            .hashes
            .get(handle)
            .and_then(|table| table.get(&key))
            .cloned();
        let exists = old.is_some();
        let length = self
            .data
            .hashes
            .len(handle)
            .map_err(|error| BytecodeVmError::DataStructure(error.to_string()))?;
        if !exists && length >= self.config.max_collection_items as u64 {
            return Err(BytecodeVmError::AllocationLimit {
                what: "hash item",
                limit: self.config.max_collection_items,
            });
        }
        let old_charge = old
            .as_ref()
            .map_or(0, Value::provenance_retained_size_charge);
        let new_charge = value
            .provenance_retained_size_charge()
            .saturating_add(usize::from(!exists).saturating_mul(256));
        if new_charge > old_charge {
            self.retain_dynamic_bytes(new_charge - old_charge)?;
        }
        let table = self.data.hashes.get_mut(handle).ok_or_else(|| {
            BytecodeVmError::DataStructure(format!("invalid hash handle: {handle}"))
        })?;
        if !exists {
            table
                .try_reserve(1)
                .map_err(|_| BytecodeVmError::AllocationLimit {
                    what: "aggregate dynamic byte",
                    limit: self.config.max_dynamic_bytes,
                })?;
        }
        table.insert(key, value);
        if old_charge > new_charge {
            self.release_dynamic_bytes(old_charge - new_charge);
        }
        self.push(Value::new(handle))
    }

    fn hash_get(&mut self) -> Result<(), BytecodeVmError> {
        self.need("HASH_GET", 2)?;
        let key = self.pop("HASH_GET")?.val;
        let handle = self.pop("HASH_GET")?.val;
        let (value, found) = self
            .data
            .hashes
            .get_key(handle, key)
            .map_err(|error| BytecodeVmError::DataStructure(error.to_string()))?;
        self.push_many([Value::new(handle), value, Value::new(u64::from(found))])
    }

    fn hash_del(&mut self) -> Result<(), BytecodeVmError> {
        self.need("HASH_DEL", 2)?;
        let key = self.pop("HASH_DEL")?.val;
        let handle = self.pop("HASH_DEL")?.val;
        let removed = self
            .data
            .hashes
            .get(handle)
            .and_then(|table| table.get(&key))
            .cloned();
        self.data
            .hashes
            .delete(handle, key)
            .map_err(|error| BytecodeVmError::DataStructure(error.to_string()))?;
        if let Some(value) = removed {
            // Buckets retain their high-water capacity after deletion. Only
            // the removed value's provenance allocation is released.
            self.release_dynamic_bytes(value.provenance_retained_size_charge());
        }
        self.push(Value::new(handle))
    }

    fn hash_has(&mut self) -> Result<(), BytecodeVmError> {
        self.need("HASH_HAS", 2)?;
        let key = self.pop("HASH_HAS")?.val;
        let handle = self.pop("HASH_HAS")?.val;
        let found = self
            .data
            .hashes
            .has(handle, key)
            .map_err(|error| BytecodeVmError::DataStructure(error.to_string()))?;
        self.push_many([Value::new(handle), Value::new(u64::from(found))])
    }

    fn hash_len(&mut self) -> Result<(), BytecodeVmError> {
        let handle = self.pop("HASH_LEN")?.val;
        let length = self
            .data
            .hashes
            .len(handle)
            .map_err(|error| BytecodeVmError::DataStructure(error.to_string()))?;
        self.push_many([Value::new(handle), Value::new(length)])
    }

    fn set_new(&mut self) -> Result<(), BytecodeVmError> {
        Self::check_collection_count(self.data.sets.count(), self.config, "set")?;
        self.retain_dynamic_bytes(64)?;
        let handle = self.data.sets.alloc();
        self.push(Value::new(handle))
    }

    fn set_add(&mut self) -> Result<(), BytecodeVmError> {
        self.need("SET_ADD", 2)?;
        let value = self.pop("SET_ADD")?.val;
        let handle = self.pop("SET_ADD")?.val;
        let exists = self
            .data
            .sets
            .has(handle, value)
            .map_err(|error| BytecodeVmError::DataStructure(error.to_string()))?;
        let length = self
            .data
            .sets
            .len(handle)
            .map_err(|error| BytecodeVmError::DataStructure(error.to_string()))?;
        if !exists && length >= self.config.max_collection_items as u64 {
            return Err(BytecodeVmError::AllocationLimit {
                what: "set item",
                limit: self.config.max_collection_items,
            });
        }
        if !exists {
            self.retain_dynamic_bytes(128)?;
            let set = self.data.sets.get_mut(handle).ok_or_else(|| {
                BytecodeVmError::DataStructure(format!("invalid set handle: {handle}"))
            })?;
            set.try_reserve(1)
                .map_err(|_| BytecodeVmError::AllocationLimit {
                    what: "aggregate dynamic byte",
                    limit: self.config.max_dynamic_bytes,
                })?;
            set.insert(value);
        }
        self.push(Value::new(handle))
    }

    fn set_has(&mut self) -> Result<(), BytecodeVmError> {
        self.need("SET_HAS", 2)?;
        let value = self.pop("SET_HAS")?.val;
        let handle = self.pop("SET_HAS")?.val;
        let found = self
            .data
            .sets
            .has(handle, value)
            .map_err(|error| BytecodeVmError::DataStructure(error.to_string()))?;
        self.push_many([Value::new(handle), Value::new(u64::from(found))])
    }

    fn set_del(&mut self) -> Result<(), BytecodeVmError> {
        self.need("SET_DEL", 2)?;
        let value = self.pop("SET_DEL")?.val;
        let handle = self.pop("SET_DEL")?.val;
        self.data
            .sets
            .delete(handle, value)
            .map_err(|error| BytecodeVmError::DataStructure(error.to_string()))?;
        // HashSet retains its high-water bucket capacity after deletion, so
        // the entry allocation charge remains until the epoch ends.
        self.push(Value::new(handle))
    }

    fn set_len(&mut self) -> Result<(), BytecodeVmError> {
        let handle = self.pop("SET_LEN")?.val;
        let length = self
            .data
            .sets
            .len(handle)
            .map_err(|error| BytecodeVmError::DataStructure(error.to_string()))?;
        self.push_many([Value::new(handle), Value::new(length)])
    }

    fn buffer_new(&mut self) -> Result<(), BytecodeVmError> {
        let capacity_word = self.pop("BUFFER_NEW")?.val;
        let capacity =
            usize::try_from(capacity_word).map_err(|_| BytecodeVmError::AllocationLimit {
                what: "buffer byte",
                limit: self.config.max_collection_items,
            })?;
        self.preflight_buffer_allocation(capacity)?;
        let buffer = ByteBuffer::with_capacity(capacity, self.config.max_collection_items)?;
        let handle = self.allocate_buffer(buffer)?;
        self.push(Value::new(handle.0))
    }

    fn buffer_from_stack(&mut self) -> Result<(), BytecodeVmError> {
        let count_word = self.pop("BUFFER_FROM_STACK")?.val;
        let count = usize::try_from(count_word).map_err(|_| BytecodeVmError::AllocationLimit {
            what: "buffer byte",
            limit: self.config.max_collection_items,
        })?;
        if count > self.config.max_collection_items {
            return Err(BytecodeVmError::AllocationLimit {
                what: "buffer byte",
                limit: self.config.max_collection_items,
            });
        }
        self.need("BUFFER_FROM_STACK", count)?;
        self.preflight_buffer_allocation(count)?;
        let start = self.stack.len() - count;
        let buffer =
            ByteBuffer::from_values(&self.stack[start..], self.config.max_collection_items)?;
        let handle = self.allocate_buffer(buffer)?;
        self.stack.truncate(start);
        self.push(Value::new(handle.0))
    }

    fn buffer_to_stack(&mut self) -> Result<(), BytecodeVmError> {
        let handle = BufferHandle(self.pop("BUFFER_TO_STACK")?.val);
        let bytes = self.buffers.get(handle)?.data.clone();
        let length = bytes.len();
        let outputs = length
            .checked_add(1)
            .ok_or(BytecodeVmError::StackLimitExceeded {
                limit: self.config.max_stack_depth,
            })?;
        self.reserve_stack(outputs)?;
        self.push_many(
            bytes
                .into_iter()
                .map(|byte| Value::new(byte as u64))
                .chain(std::iter::once(Value::new(length as u64))),
        )
    }

    fn buffer_len(&mut self) -> Result<(), BytecodeVmError> {
        let handle = BufferHandle(self.pop("BUFFER_LEN")?.val);
        let length = self.buffers.get(handle)?.data.len();
        self.push_many([Value::new(handle.0), Value::new(length as u64)])
    }

    fn buffer_read_byte(&mut self) -> Result<(), BytecodeVmError> {
        let handle = BufferHandle(self.pop("BUFFER_READ_BYTE")?.val);
        let byte = self.buffers.get_mut(handle)?.read_byte();
        self.push_many([Value::new(handle.0), Value::new(byte as u64)])
    }

    fn buffer_write_byte(&mut self) -> Result<(), BytecodeVmError> {
        self.need("BUFFER_WRITE_BYTE", 2)?;
        let byte = self.pop("BUFFER_WRITE_BYTE")?.val as u8;
        let handle = BufferHandle(self.pop("BUFFER_WRITE_BYTE")?.val);
        let (old_capacity, growth_bound) = {
            let buffer = self.buffers.get(handle)?;
            let needs_growth =
                buffer.position >= buffer.data.len() && buffer.data.len() == buffer.data.capacity();
            let target = if needs_growth {
                buffer.growth_capacity(self.config.max_collection_items)
            } else {
                buffer.data.capacity()
            };
            (buffer.data.capacity(), target - buffer.data.capacity())
        };
        if growth_bound != 0 {
            self.retain_dynamic_bytes(growth_bound)?;
        }
        self.buffers
            .get_mut(handle)?
            .write_byte(byte, self.config.max_collection_items)?;
        let new_capacity = self.buffers.get(handle)?.data.capacity();
        let growth = new_capacity.saturating_sub(old_capacity);
        match growth.cmp(&growth_bound) {
            std::cmp::Ordering::Greater => {
                self.retain_dynamic_bytes(growth - growth_bound)?;
            }
            std::cmp::Ordering::Less => {
                self.release_dynamic_bytes(growth_bound - growth);
            }
            std::cmp::Ordering::Equal => {}
        }
        self.push(Value::new(handle.0))
    }

    fn buffer_free(&mut self) -> Result<(), BytecodeVmError> {
        let handle = BufferHandle(self.pop("BUFFER_FREE")?.val);
        // The legacy primitive deliberately treats freeing an unknown handle
        // as a no-op. All operations which dereference the word return the
        // typed InvalidBufferHandle error instead.
        if let Some(buffer) = self.buffers.free(handle) {
            // Handle slots are intentionally never reused, so their arena
            // charge remains until the epoch ends. Only the dropped payload
            // capacity is released.
            self.release_dynamic_bytes(buffer.data.capacity());
        }
        Ok(())
    }
}

fn read_cell(memory: &PagedMemory, address: u64) -> Result<Value, BytecodeVmError> {
    memory
        .get(address)
        .cloned()
        .ok_or(BytecodeVmError::MemoryOutOfBounds {
            address,
            memory_cells: memory.len(),
        })
}

fn checked_seek(base: u64, offset: i64) -> Option<u64> {
    if offset >= 0 {
        base.checked_add(offset as u64)
    } else {
        base.checked_sub(offset.unsigned_abs())
    }
}

#[cfg(not(windows))]
fn shell_invocation(command: String) -> (String, Vec<String>) {
    ("sh".to_string(), vec!["-c".to_string(), command])
}

#[cfg(windows)]
fn shell_invocation(command: String) -> (String, Vec<String>) {
    ("cmd".to_string(), vec!["/C".to_string(), command])
}

fn write_cell(memory: &mut PagedMemory, address: u64, value: Value) -> Result<(), BytecodeVmError> {
    memory
        .write(address, value)
        .map_err(|error| BytecodeVmError::TemporalViolation(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Program, Stmt};
    use crate::bytecode::{
        ForeignEffects, ForeignEntry, ForeignScalarType, ProcedureEntry, QuoteEntry,
    };
    use crate::core::Memory;
    use crate::hir::ProcedureId;
    use crate::runtime::ffi::ForeignHostTable;
    use crate::vm::executor::{Executor, VmState};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn program(
        main: Vec<Instruction>,
        quotations: Vec<Vec<Instruction>>,
        procedures: Vec<Vec<Instruction>>,
    ) -> BytecodeProgram {
        let mut instructions = main;
        let main = CodeRange {
            start: 0,
            end: instructions.len() as u32,
        };
        let mut quotation_entries = Vec::new();
        for (index, body) in quotations.into_iter().enumerate() {
            let start = instructions.len() as u32;
            instructions.extend(body);
            quotation_entries.push(QuoteEntry {
                id: QuoteId::new(index as u64),
                range: CodeRange {
                    start,
                    end: instructions.len() as u32,
                },
            });
        }
        let mut procedure_entries = Vec::new();
        for (index, body) in procedures.into_iter().enumerate() {
            let start = instructions.len() as u32;
            instructions.extend(body);
            procedure_entries.push(ProcedureEntry {
                id: ProcedureId::try_from_index(index).unwrap(),
                range: CodeRange {
                    start,
                    end: instructions.len() as u32,
                },
            });
        }
        BytecodeProgram {
            instructions,
            main,
            quotations: quotation_entries,
            procedures: procedure_entries,
            foreigns: Vec::new(),
            source_map: Vec::new(),
        }
    }

    fn memory() -> PagedMemory {
        PagedMemory::with_size(16).unwrap()
    }

    fn scalar_foreign() -> ForeignEntry {
        ForeignEntry {
            id: ForeignId::try_from_index(0).unwrap(),
            library: "process".into(),
            symbol: "add_signed".into(),
            parameters: vec![ForeignScalarType::U64, ForeignScalarType::I64],
            result: Some(ForeignScalarType::I64),
            effects: ForeignEffects::from_bits(ForeignEffects::IO).unwrap(),
        }
    }

    fn foreign_program(main: Vec<Instruction>) -> BytecodeProgram {
        let mut program = program(main, Vec::new(), Vec::new());
        program.foreigns.push(scalar_foreign());
        program
    }

    fn push_path(instructions: &mut Vec<Instruction>, path: &str) {
        instructions.extend(path.bytes().map(|byte| Instruction::PushWord(byte as u64)));
        instructions.push(Instruction::PushWord(path.len() as u64));
    }

    fn unique_file(name: &str) -> std::path::PathBuf {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        std::env::temp_dir().join(format!(
            "ouro-bytecode-file-{}-{}-{name}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn assert_buffer_differential(instructions: Vec<Instruction>, expected: &[u64]) {
        let source = Program {
            body: instructions
                .iter()
                .map(|instruction| match instruction {
                    Instruction::PushWord(word) => Stmt::Push(Value::new(*word)),
                    Instruction::Primitive(opcode) => Stmt::Op(*opcode),
                    other => panic!("buffer differential received {other:?}"),
                })
                .collect(),
            ..Program::default()
        };
        let mut legacy = Executor::new();
        let mut legacy_state = VmState::new(Memory::new());
        legacy
            .execute_ast_oracle(&mut legacy_state, &source)
            .expect("legacy buffer execution");

        let mut bytecode_instructions = instructions;
        bytecode_instructions.push(Instruction::Return);
        let result = BytecodeVm::new()
            .run(
                &program(bytecode_instructions, Vec::new(), Vec::new()),
                &memory(),
            )
            .expect("bytecode buffer execution");

        assert_eq!(result.stack, legacy_state.stack);
        assert_eq!(
            result
                .stack
                .iter()
                .map(|value| value.val)
                .collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn buffer_primitives_match_legacy_stack_and_cursor_semantics() {
        assert_buffer_differential(
            vec![
                Instruction::PushWord(4),
                Instruction::Primitive(OpCode::BufferNew),
                Instruction::Primitive(OpCode::Dup),
                Instruction::Primitive(OpCode::BufferLen),
            ],
            &[1, 1, 0],
        );

        assert_buffer_differential(
            vec![
                Instruction::PushWord(65),
                Instruction::PushWord(300),
                Instruction::PushWord(255),
                Instruction::PushWord(3),
                Instruction::Primitive(OpCode::BufferFromStack),
                Instruction::Primitive(OpCode::Dup),
                Instruction::Primitive(OpCode::BufferLen),
                Instruction::Primitive(OpCode::Pop),
                Instruction::Primitive(OpCode::BufferReadByte),
                Instruction::Primitive(OpCode::Pop),
                Instruction::Primitive(OpCode::BufferReadByte),
                Instruction::Primitive(OpCode::Pop),
                Instruction::PushWord(90),
                Instruction::Primitive(OpCode::BufferWriteByte),
                Instruction::Primitive(OpCode::BufferReadByte),
                Instruction::Primitive(OpCode::Pop),
                Instruction::Primitive(OpCode::BufferToStack),
            ],
            &[1, 65, 44, 90, 3],
        );

        // BUFFER_FREE intentionally preserves the legacy no-op behavior for
        // an unknown handle while still consuming exactly one stack word.
        assert_buffer_differential(
            vec![
                Instruction::PushWord(0),
                Instruction::Primitive(OpCode::BufferNew),
                Instruction::Primitive(OpCode::BufferFree),
                Instruction::PushWord(999),
                Instruction::Primitive(OpCode::BufferFree),
            ],
            &[],
        );
    }

    #[test]
    fn buffer_handles_and_storage_are_deterministically_bounded() {
        let one_buffer = BytecodeVmConfig {
            max_collections: 1,
            ..BytecodeVmConfig::default()
        };
        let allocation_after_free = program(
            vec![
                Instruction::PushWord(0),
                Instruction::Primitive(OpCode::BufferNew),
                Instruction::Primitive(OpCode::BufferFree),
                Instruction::PushWord(0),
                Instruction::Primitive(OpCode::BufferNew),
                Instruction::Return,
            ],
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(
            BytecodeVm::with_config(one_buffer)
                .run(&allocation_after_free, &memory())
                .unwrap_err(),
            BytecodeVmError::AllocationLimit {
                what: "buffer",
                limit: 1
            }
        );

        let two_bytes = BytecodeVmConfig {
            max_collection_items: 2,
            ..BytecodeVmConfig::default()
        };
        for main in [
            vec![
                Instruction::PushWord(3),
                Instruction::Primitive(OpCode::BufferNew),
                Instruction::Return,
            ],
            vec![
                Instruction::PushWord(1),
                Instruction::PushWord(2),
                Instruction::PushWord(3),
                Instruction::PushWord(3),
                Instruction::Primitive(OpCode::BufferFromStack),
                Instruction::Return,
            ],
            vec![
                Instruction::PushWord(0),
                Instruction::Primitive(OpCode::BufferNew),
                Instruction::PushWord(1),
                Instruction::Primitive(OpCode::BufferWriteByte),
                Instruction::PushWord(2),
                Instruction::Primitive(OpCode::BufferWriteByte),
                Instruction::PushWord(3),
                Instruction::Primitive(OpCode::BufferWriteByte),
                Instruction::Return,
            ],
        ] {
            assert_eq!(
                BytecodeVm::with_config(two_bytes.clone())
                    .run(&program(main, Vec::new(), Vec::new()), &memory())
                    .unwrap_err(),
                BytecodeVmError::AllocationLimit {
                    what: "buffer byte",
                    limit: 2
                }
            );
        }
    }

    #[test]
    fn aggregate_dynamic_bytes_bound_many_individually_valid_buffers() {
        let config = BytecodeVmConfig {
            max_dynamic_bytes: 143,
            ..BytecodeVmConfig::default()
        };
        let two_live_buffers = program(
            vec![
                Instruction::PushWord(8),
                Instruction::Primitive(OpCode::BufferNew),
                Instruction::Primitive(OpCode::Pop),
                Instruction::PushWord(8),
                Instruction::Primitive(OpCode::BufferNew),
                Instruction::Return,
            ],
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(
            BytecodeVm::with_config(config.clone())
                .run(&two_live_buffers, &memory())
                .unwrap_err(),
            BytecodeVmError::AllocationLimit {
                what: "aggregate dynamic byte",
                limit: 143
            }
        );

        let freed_then_reused = program(
            vec![
                Instruction::PushWord(8),
                Instruction::Primitive(OpCode::BufferNew),
                Instruction::Primitive(OpCode::BufferFree),
                Instruction::PushWord(8),
                Instruction::Primitive(OpCode::BufferNew),
                Instruction::Return,
            ],
            Vec::new(),
            Vec::new(),
        );
        assert!(BytecodeVm::with_config(config)
            .run(&freed_then_reused, &memory())
            .is_ok());
    }

    #[test]
    fn output_byte_budget_is_independent_of_output_item_count() {
        let bytecode = program(
            vec![
                Instruction::PushWord(7),
                Instruction::Primitive(OpCode::Output),
                Instruction::Return,
            ],
            Vec::new(),
            Vec::new(),
        );
        let error = BytecodeVm::with_config(BytecodeVmConfig {
            max_output_items: usize::MAX,
            max_output_bytes: 63,
            ..BytecodeVmConfig::default()
        })
        .run(&bytecode, &memory())
        .unwrap_err();
        assert_eq!(
            error,
            BytecodeVmError::AllocationLimit {
                what: "output byte",
                limit: 63
            }
        );
    }

    #[test]
    fn buffer_dereference_errors_are_typed_and_stale_handles_never_alias() {
        for main in [
            vec![
                Instruction::PushWord(77),
                Instruction::Primitive(OpCode::BufferLen),
                Instruction::Return,
            ],
            vec![
                Instruction::PushWord(77),
                Instruction::Primitive(OpCode::BufferReadByte),
                Instruction::Return,
            ],
            vec![
                Instruction::PushWord(77),
                Instruction::Primitive(OpCode::BufferToStack),
                Instruction::Return,
            ],
            vec![
                Instruction::PushWord(77),
                Instruction::PushWord(1),
                Instruction::Primitive(OpCode::BufferWriteByte),
                Instruction::Return,
            ],
        ] {
            assert_eq!(
                BytecodeVm::new()
                    .run(&program(main, Vec::new(), Vec::new()), &memory())
                    .unwrap_err(),
                BytecodeVmError::InvalidBufferHandle { handle: 77 }
            );
        }

        let stale = program(
            vec![
                Instruction::PushWord(0),
                Instruction::Primitive(OpCode::BufferNew),
                Instruction::Primitive(OpCode::Dup),
                Instruction::Primitive(OpCode::BufferFree),
                Instruction::Primitive(OpCode::BufferLen),
                Instruction::Return,
            ],
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(
            BytecodeVm::new().run(&stale, &memory()).unwrap_err(),
            BytecodeVmError::InvalidBufferHandle { handle: 1 }
        );
    }

    #[test]
    fn buffer_from_stack_checks_its_dynamic_operand_count() {
        let bytecode = program(
            vec![
                Instruction::PushWord(1),
                Instruction::PushWord(3),
                Instruction::Primitive(OpCode::BufferFromStack),
                Instruction::Return,
            ],
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(
            BytecodeVm::new().run(&bytecode, &memory()).unwrap_err(),
            BytecodeVmError::StackUnderflow {
                operation: "BUFFER_FROM_STACK",
                needed: 3,
                available: 1
            }
        );
    }

    #[test]
    fn frozen_file_reads_match_legacy_stack_and_buffer_semantics() {
        let path = unique_file("read-parity");
        std::fs::write(&path, b"abcd").unwrap();
        let path = path.display().to_string();
        let mut instructions = Vec::new();
        push_path(&mut instructions, &path);
        instructions.push(Instruction::Primitive(OpCode::FileExists));
        push_path(&mut instructions, &path);
        instructions.push(Instruction::Primitive(OpCode::FileSize));
        push_path(&mut instructions, &path);
        instructions.push(Instruction::PushWord(FileMode::READ.0 as u64));
        instructions.push(Instruction::Primitive(OpCode::FileOpen));
        instructions.push(Instruction::PushWord((-2i64) as u64));
        instructions.push(Instruction::PushWord(2));
        instructions.push(Instruction::Primitive(OpCode::FileSeek));
        instructions.push(Instruction::Primitive(OpCode::Pop));
        instructions.push(Instruction::PushWord(3));
        instructions.push(Instruction::Primitive(OpCode::FileRead));
        instructions.push(Instruction::Primitive(OpCode::Pop));
        instructions.push(Instruction::Primitive(OpCode::BufferToStack));

        let source = Program {
            body: instructions
                .iter()
                .map(|instruction| match instruction {
                    Instruction::PushWord(word) => Stmt::Push(Value::new(*word)),
                    Instruction::Primitive(opcode) => Stmt::Op(*opcode),
                    other => panic!("file differential received {other:?}"),
                })
                .collect(),
            ..Program::default()
        };
        let mut legacy = Executor::new();
        let mut legacy_state = VmState::new(Memory::new());
        legacy
            .execute_ast_oracle(&mut legacy_state, &source)
            .unwrap();

        instructions.push(Instruction::Return);
        let execution = BytecodeVm::with_config(BytecodeVmConfig {
            file_snapshots: vec![FrozenFileSnapshot {
                path: path.clone(),
                contents: Some(b"abcd".to_vec()),
            }],
            file_read_capabilities: vec![path.clone()],
            ..BytecodeVmConfig::default()
        })
        .run(&program(instructions, Vec::new(), Vec::new()), &memory())
        .unwrap();
        assert_eq!(execution.stack, legacy_state.stack);
        assert_eq!(
            execution
                .stack
                .iter()
                .map(|value| value.val)
                .collect::<Vec<_>>(),
            vec![1, 4, 1, b'c' as u64, b'd' as u64, 2]
        );
        assert_eq!(
            execution.file_snapshots_consumed,
            vec![FrozenFileSnapshot {
                path: path.clone(),
                contents: Some(b"abcd".to_vec())
            }]
        );
        assert!(execution.effects.is_empty());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn file_mutations_are_virtual_until_selected_effect_commit() {
        use crate::temporal::effect_adapter::NativeEffectAdapter;
        use crate::temporal::transaction::{
            CommitLog, CommitToken, TemporalTransaction, TransactionLimits,
        };

        let path = unique_file("selected-write");
        std::fs::write(&path, b"host-original").unwrap();
        let path_text = path.display().to_string();
        let mut instructions = vec![
            Instruction::PushWord(b'n' as u64),
            Instruction::PushWord(b'e' as u64),
            Instruction::PushWord(b'w' as u64),
            Instruction::PushWord(3),
            Instruction::Primitive(OpCode::BufferFromStack),
        ];
        push_path(&mut instructions, &path_text);
        instructions.push(Instruction::PushWord(
            (FileMode::WRITE.0 | FileMode::TRUNCATE.0) as u64,
        ));
        instructions.push(Instruction::Primitive(OpCode::FileOpen));
        instructions.push(Instruction::Primitive(OpCode::Swap));
        instructions.push(Instruction::Primitive(OpCode::FileWrite));
        instructions.push(Instruction::Return);
        let execution = BytecodeVm::with_config(BytecodeVmConfig {
            file_snapshots: vec![FrozenFileSnapshot {
                path: path_text.clone(),
                contents: Some(b"host-original".to_vec()),
            }],
            file_write_capabilities: vec![path_text.clone()],
            ..BytecodeVmConfig::default()
        })
        .run(&program(instructions, Vec::new(), Vec::new()), &memory())
        .unwrap();

        assert_eq!(std::fs::read(&path).unwrap(), b"host-original");
        assert_eq!(
            execution
                .stack
                .iter()
                .map(|value| value.val)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
        assert_eq!(
            execution.effects,
            vec![
                EffectIntent::FileSetLength {
                    path: path_text.clone(),
                    length: 0,
                    initial: FrozenFileSnapshot {
                        path: path_text.clone(),
                        contents: Some(b"host-original".to_vec()),
                    },
                },
                EffectIntent::FileWrite {
                    path: path_text.clone(),
                    offset: 0,
                    bytes: b"new".to_vec(),
                    initial: FrozenFileSnapshot {
                        path: path_text.clone(),
                        contents: Some(b"host-original".to_vec()),
                    },
                },
            ]
        );

        let mut transaction =
            TemporalTransaction::new(Vec::new(), TransactionLimits::default()).unwrap();
        let mut context = transaction.begin_candidate().unwrap();
        for effect in &execution.effects {
            context.stage_effect(effect.clone()).unwrap();
        }
        let candidate = context.finish(Vec::new()).unwrap();
        let timeline = transaction.stage_candidate(candidate).unwrap();
        transaction.select(timeline).unwrap();
        let mut adapter = NativeEffectAdapter::new();
        adapter.allow_file(path.clone());
        let mut log = CommitLog::default();
        transaction
            .commit_selected_with_adapter(CommitToken(7), &mut log, &mut adapter)
            .unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn file_access_fails_closed_without_read_snapshot_and_write_capabilities() {
        let path = "missing-frozen-file";
        let mut exists = Vec::new();
        push_path(&mut exists, path);
        exists.extend([
            Instruction::Primitive(OpCode::FileExists),
            Instruction::Return,
        ]);
        assert_eq!(
            BytecodeVm::new()
                .run(&program(exists.clone(), Vec::new(), Vec::new()), &memory())
                .unwrap_err(),
            BytecodeVmError::FileReadCapabilityDenied {
                path: path.to_string()
            }
        );
        assert_eq!(
            BytecodeVm::with_config(BytecodeVmConfig {
                file_read_capabilities: vec![path.to_string()],
                ..BytecodeVmConfig::default()
            })
            .run(&program(exists.clone(), Vec::new(), Vec::new()), &memory())
            .unwrap_err(),
            BytecodeVmError::FileSnapshotUnavailable {
                path: path.to_string()
            }
        );
        assert_eq!(
            BytecodeVm::with_config(BytecodeVmConfig {
                file_snapshots: vec![FrozenFileSnapshot {
                    path: path.to_string(),
                    contents: Some(b"hidden".to_vec()),
                }],
                file_write_capabilities: vec![path.to_string()],
                ..BytecodeVmConfig::default()
            })
            .run(&program(exists, Vec::new(), Vec::new()), &memory())
            .unwrap_err(),
            BytecodeVmError::FileReadCapabilityDenied {
                path: path.to_string()
            }
        );

        let mut open = Vec::new();
        push_path(&mut open, path);
        open.extend([
            Instruction::PushWord((FileMode::WRITE.0 | FileMode::CREATE.0) as u64),
            Instruction::Primitive(OpCode::FileOpen),
            Instruction::Return,
        ]);
        assert_eq!(
            BytecodeVm::with_config(BytecodeVmConfig {
                file_snapshots: vec![FrozenFileSnapshot {
                    path: path.to_string(),
                    contents: None,
                }],
                ..BytecodeVmConfig::default()
            })
            .run(&program(open, Vec::new(), Vec::new()), &memory())
            .unwrap_err(),
            BytecodeVmError::FileCapabilityDenied {
                path: path.to_string()
            }
        );
    }

    #[test]
    fn live_file_capture_records_an_exact_replay_snapshot() {
        let path = unique_file("capture-replay");
        std::fs::write(&path, b"first").unwrap();
        let path_text = path.display().to_string();
        let mut instructions = Vec::new();
        push_path(&mut instructions, &path_text);
        instructions.push(Instruction::PushWord(FileMode::READ.0 as u64));
        instructions.push(Instruction::Primitive(OpCode::FileOpen));
        instructions.push(Instruction::PushWord(16));
        instructions.push(Instruction::Primitive(OpCode::FileRead));
        instructions.push(Instruction::Primitive(OpCode::Pop));
        instructions.push(Instruction::Primitive(OpCode::BufferToStack));
        instructions.push(Instruction::Return);
        let bytecode = program(instructions, Vec::new(), Vec::new());
        let captured = BytecodeVm::with_config(BytecodeVmConfig {
            file_read_capabilities: vec![path_text.clone()],
            allow_live_file_reads: true,
            ..BytecodeVmConfig::default()
        })
        .run(&bytecode, &memory())
        .unwrap();
        assert_eq!(captured.file_snapshots_consumed.len(), 1);
        assert_eq!(
            captured.file_snapshots_consumed[0].contents.as_deref(),
            Some(b"first".as_slice())
        );

        std::fs::write(&path, b"second").unwrap();
        let replayed = BytecodeVm::with_config(BytecodeVmConfig {
            file_snapshots: captured.file_snapshots_consumed.clone(),
            file_read_capabilities: vec![path_text],
            ..BytecodeVmConfig::default()
        })
        .run(&bytecode, &memory())
        .unwrap();
        assert_eq!(replayed.stack, captured.stack);
        assert_eq!(
            replayed.instructions_executed,
            captured.instructions_executed
        );
        assert_eq!(
            replayed.file_snapshots_consumed,
            captured.file_snapshots_consumed
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn live_file_capture_rejects_oversized_handle_before_reading() {
        let path = unique_file("capture-bounded");
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(4_097).unwrap();
        let path_text = path.display().to_string();
        let mut instructions = Vec::new();
        push_path(&mut instructions, &path_text);
        instructions.push(Instruction::PushWord(FileMode::READ.0 as u64));
        instructions.push(Instruction::Primitive(OpCode::FileOpen));
        instructions.push(Instruction::Return);

        let error = BytecodeVm::with_config(BytecodeVmConfig {
            file_read_capabilities: vec![path_text],
            allow_live_file_reads: true,
            max_file_snapshot_bytes: 4_096,
            max_collection_items: 4_096,
            ..BytecodeVmConfig::default()
        })
        .run(&program(instructions, Vec::new(), Vec::new()), &memory())
        .unwrap_err();
        assert!(matches!(
            error,
            BytecodeVmError::AllocationLimit {
                what: "file snapshot byte",
                limit: 4_096
            }
        ));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn frozen_file_snapshot_and_effect_storage_are_bounded() {
        let duplicate = BytecodeVmConfig {
            file_snapshots: vec![
                FrozenFileSnapshot {
                    path: "same".to_string(),
                    contents: None,
                },
                FrozenFileSnapshot {
                    path: "same".to_string(),
                    contents: Some(Vec::new()),
                },
            ],
            ..BytecodeVmConfig::default()
        };
        let bytecode = program(vec![Instruction::Return], Vec::new(), Vec::new());
        assert!(matches!(
            BytecodeVm::with_config(duplicate).run(&bytecode, &memory()),
            Err(BytecodeVmError::InvalidFileSnapshot { .. })
        ));

        let oversized = BytecodeVmConfig {
            file_snapshots: vec![FrozenFileSnapshot {
                path: "large".to_string(),
                contents: Some(vec![0; 3]),
            }],
            max_file_snapshot_bytes: 2,
            ..BytecodeVmConfig::default()
        };
        assert!(matches!(
            BytecodeVm::with_config(oversized).run(&bytecode, &memory()),
            Err(BytecodeVmError::InvalidFileSnapshot { .. })
        ));

        let path = "bounded-effect";
        let mut write = vec![
            Instruction::PushWord(1),
            Instruction::PushWord(1),
            Instruction::Primitive(OpCode::BufferFromStack),
        ];
        push_path(&mut write, path);
        write.extend([
            Instruction::PushWord(FileMode::APPEND.0 as u64),
            Instruction::Primitive(OpCode::FileOpen),
            Instruction::Primitive(OpCode::Swap),
            Instruction::Primitive(OpCode::FileWrite),
            Instruction::Return,
        ]);
        let error = BytecodeVm::with_config(BytecodeVmConfig {
            file_snapshots: vec![FrozenFileSnapshot {
                path: path.to_string(),
                contents: Some(Vec::new()),
            }],
            file_write_capabilities: vec![path.to_string()],
            max_effects: 0,
            ..BytecodeVmConfig::default()
        })
        .run(&program(write, Vec::new(), Vec::new()), &memory())
        .unwrap_err();
        assert_eq!(
            error,
            BytecodeVmError::AllocationLimit {
                what: "effect",
                limit: 0
            }
        );
    }

    #[test]
    fn linked_scalar_host_call_marshals_exact_word_bits() {
        let target = ForeignId::try_from_index(0).unwrap();
        let bytecode = foreign_program(vec![
            Instruction::PushWord(50),
            Instruction::PushWord((-8i64) as u64),
            Instruction::CallForeign(target),
            Instruction::Return,
        ]);
        let mut host = ForeignHostTable::new();
        host.bind(scalar_foreign(), |args| {
            let result = args[0] as i64 + args[1] as i64;
            Ok(Some(result as u64))
        })
        .unwrap();

        let result = BytecodeVm::new()
            .with_foreign_host(Arc::new(host))
            .run(&bytecode, &memory())
            .unwrap();

        assert_eq!(result.stack[0].val as i64, 42);
        assert_eq!(result.instructions_executed, 4);
    }

    #[test]
    fn foreign_link_rejects_missing_and_mismatched_symbols_before_dispatch() {
        let target = ForeignId::try_from_index(0).unwrap();
        let bytecode = foreign_program(vec![Instruction::Return]);
        assert!(matches!(
            BytecodeVm::new().run(&bytecode, &memory()),
            Err(BytecodeVmError::ForeignLink(ForeignHostError::UnknownTarget { id }))
                if id == target
        ));

        let mut wrong = scalar_foreign();
        wrong.symbol = "other".into();
        let mut host = ForeignHostTable::new();
        host.bind(wrong, |_| Ok(Some(0))).unwrap();
        assert!(matches!(
            BytecodeVm::new()
                .with_foreign_host(Arc::new(host))
                .run(&bytecode, &memory()),
            Err(BytecodeVmError::ForeignLink(ForeignHostError::SignatureMismatch { id }))
                if id == target
        ));
    }

    #[test]
    fn foreign_underflow_and_return_shape_are_checked() {
        let target = ForeignId::try_from_index(0).unwrap();
        let bytecode = foreign_program(vec![
            Instruction::PushWord(1),
            Instruction::CallForeign(target),
            Instruction::Return,
        ]);
        let mut host = ForeignHostTable::new();
        host.bind(scalar_foreign(), |_| Ok(Some(0))).unwrap();
        let vm = BytecodeVm::new().with_foreign_host(Arc::new(host));
        assert!(matches!(
            vm.run(&bytecode, &memory()),
            Err(BytecodeVmError::BytecodeVerification(
                BytecodeVerificationError::MainEntryUnderflow { missing: 1, .. }
            ))
        ));

        let bytecode = foreign_program(vec![
            Instruction::PushWord(1),
            Instruction::PushWord(2),
            Instruction::CallForeign(target),
            Instruction::Return,
        ]);
        let mut host = ForeignHostTable::new();
        host.bind(scalar_foreign(), |_| Ok(None)).unwrap();
        assert!(matches!(
            BytecodeVm::new()
                .with_foreign_host(Arc::new(host))
                .run(&bytecode, &memory()),
            Err(BytecodeVmError::ForeignCall(ForeignHostError::ReturnMismatch {
                id,
                expected_value: true
            })) if id == target
        ));
    }

    #[test]
    fn gas_exhaustion_happens_before_the_external_callback() {
        let target = ForeignId::try_from_index(0).unwrap();
        let bytecode = foreign_program(vec![
            Instruction::PushWord(1),
            Instruction::PushWord(2),
            Instruction::CallForeign(target),
            Instruction::Return,
        ]);
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&calls);
        let mut host = ForeignHostTable::new();
        host.bind(scalar_foreign(), move |_| {
            observed.fetch_add(1, Ordering::SeqCst);
            Ok(Some(3))
        })
        .unwrap();
        let config = BytecodeVmConfig {
            max_instructions: 2,
            ..BytecodeVmConfig::default()
        };

        assert_eq!(
            BytecodeVm::with_config(config)
                .with_foreign_host(Arc::new(host))
                .run(&bytecode, &memory())
                .unwrap_err(),
            BytecodeVmError::GasExhausted { limit: 2 }
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn executes_procedure_and_quote_calls_with_explicit_frames() {
        let procedure = ProcedureId::try_from_index(0).unwrap();
        let bytecode = program(
            vec![
                Instruction::CallProcedure(procedure),
                Instruction::PushQuote(QuoteId::new(0)),
                Instruction::Primitive(OpCode::Exec),
                Instruction::Return,
            ],
            vec![vec![Instruction::PushWord(7), Instruction::Return]],
            vec![vec![
                Instruction::PushWord(2),
                Instruction::PushWord(3),
                Instruction::Primitive(OpCode::Add),
                Instruction::Return,
            ]],
        );

        let result = BytecodeVm::new().run(&bytecode, &memory()).unwrap();

        assert_eq!(result.status, BytecodeVmStatus::Finished);
        assert_eq!(
            result
                .stack
                .iter()
                .map(|value| value.val)
                .collect::<Vec<_>>(),
            [5, 7]
        );
        assert_eq!(result.maximum_call_depth, 1);
    }

    #[test]
    fn bi_uses_iterative_continuations_for_both_quotations() {
        let bytecode = program(
            vec![
                Instruction::PushWord(5),
                Instruction::PushQuote(QuoteId::new(0)),
                Instruction::PushQuote(QuoteId::new(1)),
                Instruction::Primitive(OpCode::Bi),
                Instruction::Return,
            ],
            vec![
                vec![
                    Instruction::PushWord(1),
                    Instruction::Primitive(OpCode::Add),
                    Instruction::Return,
                ],
                vec![
                    Instruction::PushWord(2),
                    Instruction::Primitive(OpCode::Mul),
                    Instruction::Return,
                ],
            ],
            Vec::new(),
        );

        let result = BytecodeVm::new().run(&bytecode, &memory()).unwrap();

        assert_eq!(
            result
                .stack
                .iter()
                .map(|value| value.val)
                .collect::<Vec<_>>(),
            [6, 10]
        );
    }

    #[test]
    fn structured_if_and_while_dispatch_by_program_counter() {
        let bytecode = program(
            vec![
                Instruction::PushWord(0),
                Instruction::IfFalse {
                    else_target: 4,
                    end_target: 5,
                    has_else: true,
                },
                Instruction::PushWord(111),
                Instruction::Jump { target: 5 },
                Instruction::PushWord(3),
                Instruction::Primitive(OpCode::Dup),
                Instruction::WhileFalse {
                    loop_start: 5,
                    end_target: 10,
                },
                Instruction::PushWord(1),
                Instruction::Primitive(OpCode::Sub),
                Instruction::LoopBack { target: 5 },
                Instruction::Return,
            ],
            Vec::new(),
            Vec::new(),
        );

        let result = BytecodeVm::new().run(&bytecode, &memory()).unwrap();

        assert_eq!(result.stack.len(), 1);
        assert_eq!(result.stack[0].val, 0);
    }

    #[test]
    fn temporal_scope_translates_truncates_and_accounts() {
        let bytecode = program(
            vec![
                Instruction::TemporalEnter {
                    base: 2,
                    size: 1,
                    cell_bits: 4,
                    exit_target: 6,
                },
                Instruction::PushWord(42),
                Instruction::PushWord(0),
                Instruction::Primitive(OpCode::Prophecy),
                Instruction::PushWord(0),
                Instruction::Primitive(OpCode::PresentRead),
                Instruction::TemporalExit { enter_target: 0 },
                Instruction::Return,
            ],
            Vec::new(),
            Vec::new(),
        );

        let result = BytecodeVm::new().run(&bytecode, &memory()).unwrap();

        assert_eq!(result.stack[0].val, 10);
        assert_eq!(result.present.get(2).unwrap().val, 10);
        assert_eq!(result.temporal_entries, 1);
        assert_eq!(result.temporal_exits, 1);
        assert_eq!(result.maximum_temporal_depth, 1);
    }

    #[test]
    fn paradox_rolls_back_an_open_temporal_scope() {
        let bytecode = program(
            vec![
                Instruction::TemporalEnter {
                    base: 2,
                    size: 1,
                    cell_bits: 64,
                    exit_target: 4,
                },
                Instruction::PushWord(99),
                Instruction::PushWord(0),
                Instruction::Primitive(OpCode::Prophecy),
                Instruction::TemporalExit { enter_target: 0 },
                Instruction::Return,
            ],
            Vec::new(),
            Vec::new(),
        );
        let mut bytecode = bytecode;
        bytecode
            .instructions
            .insert(4, Instruction::Primitive(OpCode::Paradox));
        if let Instruction::TemporalEnter { exit_target, .. } = &mut bytecode.instructions[0] {
            *exit_target = 5;
        }
        if let Instruction::TemporalExit { enter_target } = &mut bytecode.instructions[5] {
            *enter_target = 0;
        }
        bytecode.main.end += 1;

        let result = BytecodeVm::new().run(&bytecode, &memory()).unwrap();

        assert_eq!(result.status, BytecodeVmStatus::Paradox);
        assert_eq!(result.present.get(2).unwrap().val, 0);
        assert_eq!(result.temporal_entries, 1);
        assert_eq!(result.temporal_exits, 0);
    }

    #[test]
    fn recursive_procedure_hits_language_depth_not_host_stack() {
        let procedure = ProcedureId::try_from_index(0).unwrap();
        let bytecode = program(
            vec![Instruction::CallProcedure(procedure), Instruction::Return],
            Vec::new(),
            vec![vec![
                Instruction::CallProcedure(procedure),
                Instruction::Return,
            ]],
        );
        let config = BytecodeVmConfig {
            max_call_depth: 64,
            ..BytecodeVmConfig::default()
        };

        let error = BytecodeVm::with_config(config)
            .run(&bytecode, &memory())
            .unwrap_err();

        assert_eq!(error, BytecodeVmError::CallDepthExceeded { limit: 64 });
    }

    #[test]
    fn recursive_quotation_hits_language_depth_not_host_stack() {
        let bytecode = program(
            vec![
                Instruction::PushQuote(QuoteId::new(0)),
                Instruction::Primitive(OpCode::Exec),
                Instruction::Return,
            ],
            vec![vec![
                Instruction::PushQuote(QuoteId::new(0)),
                Instruction::Primitive(OpCode::Rec),
                Instruction::Return,
            ]],
            Vec::new(),
        );
        let config = BytecodeVmConfig {
            max_call_depth: 128,
            ..BytecodeVmConfig::default()
        };

        let error = BytecodeVm::with_config(config)
            .run(&bytecode, &memory())
            .unwrap_err();

        assert_eq!(error, BytecodeVmError::CallDepthExceeded { limit: 128 });
    }

    #[test]
    fn infinite_loop_stops_at_exact_gas_bound() {
        let bytecode = program(
            vec![
                Instruction::PushWord(1),
                Instruction::WhileFalse {
                    loop_start: 0,
                    end_target: 4,
                },
                Instruction::Primitive(OpCode::Nop),
                Instruction::LoopBack { target: 0 },
                Instruction::Return,
            ],
            Vec::new(),
            Vec::new(),
        );
        let config = BytecodeVmConfig {
            max_instructions: 17,
            ..BytecodeVmConfig::default()
        };

        let error = BytecodeVm::with_config(config)
            .run(&bytecode, &memory())
            .unwrap_err();

        assert_eq!(error, BytecodeVmError::GasExhausted { limit: 17 });
    }

    #[test]
    fn stack_growth_is_bounded() {
        let bytecode = program(
            vec![
                Instruction::PushWord(1),
                Instruction::PushWord(2),
                Instruction::Return,
            ],
            Vec::new(),
            Vec::new(),
        );
        let config = BytecodeVmConfig {
            max_stack_depth: 1,
            ..BytecodeVmConfig::default()
        };

        let error = BytecodeVm::with_config(config)
            .run(&bytecode, &memory())
            .unwrap_err();

        assert_eq!(error, BytecodeVmError::StackLimitExceeded { limit: 1 });
    }

    #[test]
    fn input_is_deterministic_by_default_and_records_the_frozen_tape() {
        let bytecode = program(
            vec![Instruction::Primitive(OpCode::Input), Instruction::Return],
            Vec::new(),
            Vec::new(),
        );

        let error = BytecodeVm::new().run(&bytecode, &memory()).unwrap_err();
        assert_eq!(error, BytecodeVmError::InputExhausted { consumed: 0 });

        let result = BytecodeVm::with_config(BytecodeVmConfig {
            input: vec![73],
            ..BytecodeVmConfig::default()
        })
        .run(&bytecode, &memory())
        .unwrap();
        assert_eq!(result.stack[0].val, 73);
        assert_eq!(result.inputs_consumed, vec![73]);
    }

    #[test]
    fn clock_and_random_use_independent_frozen_tapes() {
        let bytecode = program(
            vec![
                Instruction::Primitive(OpCode::Clock),
                Instruction::Primitive(OpCode::Random),
                Instruction::Primitive(OpCode::Clock),
                Instruction::Primitive(OpCode::Random),
                Instruction::Return,
            ],
            Vec::new(),
            Vec::new(),
        );
        let vm = BytecodeVm::with_config(BytecodeVmConfig {
            clock_input: vec![1_000, 2_000],
            random_input: vec![0xA5, 0x5A],
            ..BytecodeVmConfig::default()
        });
        let result = vm.run(&bytecode, &memory()).unwrap();

        assert_eq!(
            result
                .stack
                .iter()
                .map(|value| value.val)
                .collect::<Vec<_>>(),
            vec![1_000, 0xA5, 2_000, 0x5A]
        );
        assert_eq!(result.clock_inputs_consumed, vec![1_000, 2_000]);
        assert_eq!(result.random_inputs_consumed, vec![0xA5, 0x5A]);
        assert_eq!(result.instructions_executed, 5);

        let mut different_anamnesis = memory();
        different_anamnesis.write(0, Value::new(u64::MAX)).unwrap();
        let repeated = vm.run(&bytecode, &different_anamnesis).unwrap();
        assert_eq!(repeated.stack, result.stack);
        assert_eq!(repeated.clock_inputs_consumed, result.clock_inputs_consumed);
        assert_eq!(
            repeated.random_inputs_consumed,
            result.random_inputs_consumed
        );
        assert_eq!(repeated.instructions_executed, result.instructions_executed);
    }

    #[test]
    fn system_inputs_exhaust_by_default_with_typed_errors() {
        for (opcode, expected) in [
            (
                OpCode::Clock,
                BytecodeVmError::ClockInputExhausted { consumed: 0 },
            ),
            (
                OpCode::Random,
                BytecodeVmError::RandomInputExhausted { consumed: 0 },
            ),
        ] {
            let bytecode = program(
                vec![Instruction::Primitive(opcode), Instruction::Return],
                Vec::new(),
                Vec::new(),
            );
            assert_eq!(
                BytecodeVm::new().run(&bytecode, &memory()).unwrap_err(),
                expected
            );
        }
    }

    #[test]
    fn live_system_inputs_are_recorded_for_exact_replay() {
        let bytecode = program(
            vec![
                Instruction::Primitive(OpCode::Clock),
                Instruction::Primitive(OpCode::Random),
                Instruction::Primitive(OpCode::Clock),
                Instruction::Primitive(OpCode::Random),
                Instruction::Return,
            ],
            Vec::new(),
            Vec::new(),
        );
        let captured = BytecodeVm::with_config(BytecodeVmConfig {
            allow_live_system_inputs: true,
            ..BytecodeVmConfig::default()
        })
        .run(&bytecode, &memory())
        .unwrap();
        assert_eq!(captured.clock_inputs_consumed.len(), 2);
        assert_eq!(captured.random_inputs_consumed.len(), 2);

        let replayed = BytecodeVm::with_config(BytecodeVmConfig {
            clock_input: captured.clock_inputs_consumed.clone(),
            random_input: captured.random_inputs_consumed.clone(),
            ..BytecodeVmConfig::default()
        })
        .run(&bytecode, &memory())
        .unwrap();
        assert_eq!(replayed.stack, captured.stack);
        assert_eq!(
            replayed.clock_inputs_consumed,
            captured.clock_inputs_consumed
        );
        assert_eq!(
            replayed.random_inputs_consumed,
            captured.random_inputs_consumed
        );
        assert_eq!(
            replayed.instructions_executed,
            captured.instructions_executed
        );
    }

    #[test]
    fn system_input_dispatch_preserves_exact_gas_accounting() {
        let bytecode = program(
            vec![
                Instruction::Primitive(OpCode::Clock),
                Instruction::Primitive(OpCode::Random),
                Instruction::Return,
            ],
            Vec::new(),
            Vec::new(),
        );
        let error = BytecodeVm::with_config(BytecodeVmConfig {
            max_instructions: 2,
            clock_input: vec![7],
            random_input: vec![11],
            ..BytecodeVmConfig::default()
        })
        .run(&bytecode, &memory())
        .unwrap_err();
        assert_eq!(error, BytecodeVmError::GasExhausted { limit: 2 });
    }

    #[test]
    fn virtual_socket_uses_frozen_receive_bytes_and_stages_send() {
        let host = "frozen.example";
        let endpoint = format!("{host}:4242");
        let mut main = Vec::new();
        push_path(&mut main, host);
        main.push(Instruction::PushWord(4242));
        main.push(Instruction::Primitive(OpCode::TcpConnect));
        main.extend([
            Instruction::PushWord(b'A' as u64),
            Instruction::PushWord(b'B' as u64),
            Instruction::PushWord(2),
            Instruction::Primitive(OpCode::BufferFromStack),
            Instruction::Primitive(OpCode::SocketSend),
            Instruction::Primitive(OpCode::Pop),
            Instruction::PushWord(2),
            Instruction::Primitive(OpCode::SocketRecv),
            Instruction::Primitive(OpCode::Pop),
            Instruction::Primitive(OpCode::BufferToStack),
            Instruction::Return,
        ]);
        let bytecode = program(main, Vec::new(), Vec::new());
        let vm = BytecodeVm::with_config(BytecodeVmConfig {
            endpoint_tapes: vec![FrozenEndpointTape {
                endpoint: endpoint.clone(),
                recv_bytes: b"xyz".to_vec(),
            }],
            network_send_capabilities: vec![endpoint.clone()],
            ..BytecodeVmConfig::default()
        });

        let first = vm.run(&bytecode, &memory()).unwrap();
        let replay = vm.run(&bytecode, &memory()).unwrap();
        assert_eq!(
            first
                .stack
                .iter()
                .map(|value| value.val)
                .collect::<Vec<_>>(),
            vec![1, b'x' as u64, b'y' as u64, 2]
        );
        assert_eq!(replay.stack, first.stack);
        assert_eq!(
            first.endpoint_tapes_consumed,
            vec![FrozenEndpointTape {
                endpoint: endpoint.clone(),
                recv_bytes: b"xyz".to_vec(),
            }]
        );
        assert_eq!(
            first.effects,
            vec![EffectIntent::NetworkSend {
                endpoint,
                bytes: b"AB".to_vec(),
            }]
        );
    }

    #[test]
    fn virtual_socket_resources_fail_closed_and_reject_stale_handles() {
        let host = "denied.example";
        let endpoint = format!("{host}:80");
        let mut connect = Vec::new();
        push_path(&mut connect, host);
        connect.extend([
            Instruction::PushWord(80),
            Instruction::Primitive(OpCode::TcpConnect),
            Instruction::Return,
        ]);
        let connect = program(connect, Vec::new(), Vec::new());
        assert_eq!(
            BytecodeVm::new().run(&connect, &memory()).unwrap_err(),
            BytecodeVmError::EndpointTapeUnavailable {
                endpoint: endpoint.clone()
            }
        );

        let mut denied_send = Vec::new();
        push_path(&mut denied_send, host);
        denied_send.extend([
            Instruction::PushWord(80),
            Instruction::Primitive(OpCode::TcpConnect),
            Instruction::PushWord(0),
            Instruction::Primitive(OpCode::BufferNew),
            Instruction::Primitive(OpCode::SocketSend),
            Instruction::Return,
        ]);
        assert_eq!(
            BytecodeVm::with_config(BytecodeVmConfig {
                endpoint_tapes: vec![FrozenEndpointTape {
                    endpoint: endpoint.clone(),
                    recv_bytes: Vec::new(),
                }],
                ..BytecodeVmConfig::default()
            })
            .run(&program(denied_send, Vec::new(), Vec::new()), &memory())
            .unwrap_err(),
            BytecodeVmError::NetworkCapabilityDenied {
                endpoint: endpoint.clone()
            }
        );

        let mut stale = Vec::new();
        push_path(&mut stale, host);
        stale.extend([
            Instruction::PushWord(80),
            Instruction::Primitive(OpCode::TcpConnect),
            Instruction::Primitive(OpCode::Dup),
            Instruction::Primitive(OpCode::SocketClose),
            Instruction::PushWord(1),
            Instruction::Primitive(OpCode::SocketRecv),
            Instruction::Return,
        ]);
        assert_eq!(
            BytecodeVm::with_config(BytecodeVmConfig {
                endpoint_tapes: vec![FrozenEndpointTape {
                    endpoint: endpoint.clone(),
                    recv_bytes: Vec::new(),
                }],
                ..BytecodeVmConfig::default()
            })
            .run(&program(stale, Vec::new(), Vec::new()), &memory())
            .unwrap_err(),
            BytecodeVmError::InvalidSocketHandle { handle: 1 }
        );
    }

    #[test]
    fn process_exec_uses_exact_frozen_result_and_stages_shell_spawn() {
        let command = "printf deterministic";
        let mut main = Vec::new();
        push_path(&mut main, command);
        main.extend([
            Instruction::Primitive(OpCode::ProcExec),
            Instruction::Primitive(OpCode::Swap),
            Instruction::Primitive(OpCode::BufferToStack),
            Instruction::Return,
        ]);
        let bytecode = program(main, Vec::new(), Vec::new());
        let vm = BytecodeVm::with_config(BytecodeVmConfig {
            process_results: vec![FrozenProcessResult {
                command: command.to_string(),
                output: b"ok".to_vec(),
                exit_code: -7,
            }],
            process_capabilities: vec![command.to_string()],
            ..BytecodeVmConfig::default()
        });

        let execution = vm.run(&bytecode, &memory()).unwrap();
        assert_eq!(
            execution
                .stack
                .iter()
                .map(|value| value.val)
                .collect::<Vec<_>>(),
            vec![(-7i32) as u64, b'o' as u64, b'k' as u64, 2]
        );
        assert_eq!(execution.process_results_consumed.len(), 1);
        let (program, arguments) = shell_invocation(command.to_string());
        assert_eq!(
            execution.effects,
            vec![EffectIntent::ProcessSpawn { program, arguments }]
        );
    }

    #[test]
    fn process_result_buffers_share_the_aggregate_dynamic_budget() {
        let command = "bounded-result";
        let mut main = Vec::new();
        for _ in 0..2 {
            push_path(&mut main, command);
            main.extend([
                Instruction::Primitive(OpCode::ProcExec),
                Instruction::Primitive(OpCode::Pop),
                Instruction::Primitive(OpCode::Pop),
            ]);
        }
        main.push(Instruction::Return);
        let error = BytecodeVm::with_config(BytecodeVmConfig {
            process_results: vec![FrozenProcessResult {
                command: command.to_string(),
                output: b"12345678".to_vec(),
                exit_code: 0,
            }],
            process_capabilities: vec![command.to_string()],
            max_dynamic_bytes: 143,
            ..BytecodeVmConfig::default()
        })
        .run(&program(main, Vec::new(), Vec::new()), &memory())
        .unwrap_err();
        assert_eq!(
            error,
            BytecodeVmError::AllocationLimit {
                what: "aggregate dynamic byte",
                limit: 143
            }
        );
    }

    #[test]
    fn process_and_sleep_require_explicit_capabilities() {
        let command = "never-run";
        let mut process_main = Vec::new();
        push_path(&mut process_main, command);
        process_main.extend([
            Instruction::Primitive(OpCode::ProcExec),
            Instruction::Return,
        ]);
        let process = program(process_main, Vec::new(), Vec::new());
        assert_eq!(
            BytecodeVm::new().run(&process, &memory()).unwrap_err(),
            BytecodeVmError::ProcessCapabilityDenied {
                command: command.to_string()
            }
        );
        assert_eq!(
            BytecodeVm::with_config(BytecodeVmConfig {
                process_capabilities: vec![command.to_string()],
                ..BytecodeVmConfig::default()
            })
            .run(&process, &memory())
            .unwrap_err(),
            BytecodeVmError::ProcessResultUnavailable {
                command: command.to_string()
            }
        );

        let sleep = program(
            vec![
                Instruction::PushWord(1),
                Instruction::Primitive(OpCode::Sleep),
                Instruction::Return,
            ],
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(
            BytecodeVm::new().run(&sleep, &memory()).unwrap_err(),
            BytecodeVmError::SleepCapabilityDenied {
                milliseconds: 1,
                maximum: None
            }
        );
        let execution = BytecodeVm::with_config(BytecodeVmConfig {
            max_sleep_milliseconds: Some(1),
            ..BytecodeVmConfig::default()
        })
        .run(&sleep, &memory())
        .unwrap();
        assert_eq!(
            execution.effects,
            vec![EffectIntent::Sleep { milliseconds: 1 }]
        );
    }

    #[test]
    fn endpoint_process_and_sleep_storage_obey_deterministic_limits() {
        let empty = program(vec![Instruction::Return], Vec::new(), Vec::new());
        assert!(matches!(
            BytecodeVm::with_config(BytecodeVmConfig {
                endpoint_tapes: vec![FrozenEndpointTape {
                    endpoint: "bounded:1".to_string(),
                    recv_bytes: vec![1, 2, 3],
                }],
                max_endpoint_tape_bytes: 2,
                ..BytecodeVmConfig::default()
            })
            .run(&empty, &memory()),
            Err(BytecodeVmError::InvalidEndpointTape { .. })
        ));
        assert!(matches!(
            BytecodeVm::with_config(BytecodeVmConfig {
                process_results: vec![
                    FrozenProcessResult {
                        command: "same".to_string(),
                        output: Vec::new(),
                        exit_code: 0,
                    },
                    FrozenProcessResult {
                        command: "same".to_string(),
                        output: Vec::new(),
                        exit_code: 1,
                    },
                ],
                ..BytecodeVmConfig::default()
            })
            .run(&empty, &memory()),
            Err(BytecodeVmError::InvalidProcessResult { .. })
        ));

        let sleep = program(
            vec![
                Instruction::PushWord(0),
                Instruction::Primitive(OpCode::Sleep),
                Instruction::Return,
            ],
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(
            BytecodeVm::with_config(BytecodeVmConfig {
                max_sleep_milliseconds: Some(0),
                max_effect_bytes: 7,
                ..BytecodeVmConfig::default()
            })
            .run(&sleep, &memory())
            .unwrap_err(),
            BytecodeVmError::AllocationLimit {
                what: "effect byte",
                limit: 7
            }
        );
    }

    #[test]
    fn prepared_dispatch_is_observationally_and_gas_equivalent() {
        let bytecode = program(
            vec![
                Instruction::PushWord(40),
                Instruction::PushWord(2),
                Instruction::Primitive(OpCode::Add),
                Instruction::Primitive(OpCode::Dup),
                Instruction::Primitive(OpCode::Output),
                Instruction::Return,
            ],
            Vec::new(),
            Vec::new(),
        );
        let prepared = PreparedBytecode::new(bytecode.clone()).unwrap();
        let vm = BytecodeVm::new();
        let reference = vm.run(&bytecode, &memory()).unwrap();
        let optimized = vm.run_prepared(&prepared, &memory()).unwrap();

        assert_eq!(optimized.stack, reference.stack);
        assert_eq!(optimized.present, reference.present);
        assert_eq!(optimized.output, reference.output);
        assert_eq!(optimized.status, reference.status);
        assert_eq!(
            optimized.instructions_executed,
            reference.instructions_executed
        );
        assert_eq!(optimized.inputs_consumed, reference.inputs_consumed);
        assert_eq!(
            optimized.clock_inputs_consumed,
            reference.clock_inputs_consumed
        );
        assert_eq!(
            optimized.random_inputs_consumed,
            reference.random_inputs_consumed
        );
        assert_eq!(
            optimized.file_snapshots_consumed,
            reference.file_snapshots_consumed
        );
        assert_eq!(
            optimized.endpoint_tapes_consumed,
            reference.endpoint_tapes_consumed
        );
        assert_eq!(
            optimized.process_results_consumed,
            reference.process_results_consumed
        );
        assert_eq!(optimized.effects, reference.effects);

        let bounded = BytecodeVm::with_config(BytecodeVmConfig {
            max_instructions: 3,
            ..BytecodeVmConfig::default()
        });
        assert_eq!(
            bounded.run(&bytecode, &memory()).unwrap_err(),
            bounded.run_prepared(&prepared, &memory()).unwrap_err()
        );
    }

    #[test]
    fn prepared_bytecode_cannot_seal_a_malformed_artifact() {
        let malformed = BytecodeProgram {
            instructions: vec![Instruction::PushWord(1)],
            main: CodeRange { start: 0, end: 1 },
            quotations: Vec::new(),
            procedures: Vec::new(),
            foreigns: Vec::new(),
            source_map: Vec::new(),
        };
        assert!(PreparedBytecode::new(malformed).is_err());
    }

    #[test]
    fn raw_and_prepared_entry_points_reject_verifier_invalid_bytecode() {
        let underflow = program(
            vec![Instruction::Primitive(OpCode::Add), Instruction::Return],
            Vec::new(),
            Vec::new(),
        );

        assert!(matches!(
            PreparedBytecode::new(underflow.clone()),
            Err(BytecodeVmError::BytecodeVerification(
                BytecodeVerificationError::MainEntryUnderflow { .. }
            ))
        ));
        assert!(matches!(
            BytecodeVm::new().run(&underflow, &memory()),
            Err(BytecodeVmError::BytecodeVerification(
                BytecodeVerificationError::MainEntryUnderflow { .. }
            ))
        ));
    }

    #[test]
    fn all_unsupported_primitives_fail_with_the_typed_error() {
        let unsupported = [OpCode::FFICall, OpCode::FFICallNamed];
        assert_eq!(OpCode::ALL.len(), 99);
        assert_eq!(OpCode::ALL.len() - unsupported.len(), 97);
        for &opcode in OpCode::ALL {
            assert_eq!(bytecode_vm_supports(opcode), !unsupported.contains(&opcode));
        }

        for opcode in unsupported {
            let bytecode = program(
                vec![
                    Instruction::PushWord(0),
                    Instruction::Primitive(opcode),
                    Instruction::Return,
                ],
                Vec::new(),
                Vec::new(),
            );
            let error = BytecodeVm::new().run(&bytecode, &memory()).unwrap_err();
            assert_eq!(error, BytecodeVmError::UnsupportedPrimitive(opcode));
        }
    }

    #[test]
    fn invalid_artifact_is_rejected_before_execution() {
        let malformed = BytecodeProgram {
            instructions: vec![Instruction::Primitive(OpCode::Clock)],
            main: CodeRange { start: 0, end: 1 },
            quotations: Vec::new(),
            procedures: Vec::new(),
            foreigns: Vec::new(),
            source_map: Vec::new(),
        };

        let error = BytecodeVm::new().run(&malformed, &memory()).unwrap_err();

        assert!(matches!(
            error,
            BytecodeVmError::BytecodeVerification(BytecodeVerificationError::Structural(_))
        ));
    }
}
