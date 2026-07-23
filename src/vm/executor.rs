//! Virtual Machine for OUROCHRONOS epoch execution.
//!
//! The VM executes a single epoch: given an anamnesis (read-only "future" memory),
//! it produces a present (read-write "current" memory) and output.
//!
//! The fixed-point search (in timeloop.rs) repeatedly runs epochs until
//! Present = Anamnesis (temporal consistency achieved).

use crate::ast::Program;
#[cfg(test)]
use crate::ast::{OpCode, Stmt};
use crate::bytecode::{BytecodeProgram, Instruction};
use crate::bytecode_verifier::verify_default as verify_bytecode;
use crate::bytecode_vm::{
    BytecodeExecution, BytecodeVm, BytecodeVmConfig, BytecodeVmError, BytecodeVmStatus,
    PreparedBytecode,
};
#[cfg(test)]
use crate::core::error::{BoundsPolicy, MemoryOperation, SourceLocation};
use crate::core::error::{DivisionByZeroPolicy, ErrorConfig, StackPolicy};
#[cfg(test)]
use crate::core::provenance::Provenance;
#[cfg(test)]
use crate::core::Address;
use crate::core::{DataStructures, Memory, OutputItem, PagedMemory, Value};
use crate::hir::HirProgram;
#[cfg(test)]
use crate::runtime::ffi::FFICaller;
use crate::runtime::ffi::FFIContext;
use crate::runtime::io::IOContext;
#[cfg(test)]
use crate::runtime::io::{FileMode, SeekOrigin};
use crate::semantics::check as check_semantics;
use crate::types::type_check;
use std::io::{self, BufRead, Read, Write};

/// Status of epoch execution.
#[derive(Debug, Clone, PartialEq)]
pub enum EpochStatus {
    /// Epoch is still running.
    Running,
    /// Epoch completed normally (reached end or HALT).
    Finished,
    /// Epoch terminated due to explicit PARADOX instruction.
    Paradox,
    /// Epoch terminated due to runtime error.
    Error(String),
}

/// Complete state of the VM during epoch execution.
#[derive(Debug, Clone)]
pub struct VmState {
    /// The operand stack.
    pub stack: Vec<Value>,
    /// Present memory (being constructed this epoch).
    pub present: Memory,
    /// Anamnesis memory (read-only, from the "future").
    pub anamnesis: Memory,
    /// Output buffer.
    pub output: Vec<OutputItem>,
    /// Execution status.
    pub status: EpochStatus,
    /// Instruction count (for gas limiting).
    pub instructions_executed: u64,
    /// Dynamic data structures (vectors, hash tables, sets).
    pub data_structures: DataStructures,
}

impl VmState {
    /// Create a new VM state for an epoch.
    pub fn new(anamnesis: Memory) -> Self {
        let memory_cells = anamnesis.len();
        Self {
            stack: Vec::new(),
            present: Memory::with_size(memory_cells),
            anamnesis,
            output: Vec::new(),
            status: EpochStatus::Running,
            instructions_executed: 0,
            data_structures: DataStructures::new(),
        }
    }
}

/// Result of executing a single epoch.
#[derive(Debug)]
pub struct EpochResult {
    /// The final present memory.
    pub present: Memory,
    /// Output produced during the epoch.
    pub output: Vec<OutputItem>,
    /// Terminal status.
    pub status: EpochStatus,
    /// Number of instructions executed.
    pub instructions_executed: u64,
    /// Input values consumed during this epoch.
    /// Used for input freezing to ensure temporal consistency.
    pub inputs_consumed: Vec<u64>,
}

/// Configuration for the executor.
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Maximum instructions per epoch (gas limit).
    pub max_instructions: u64,
    /// Historical immediate-output switch. Authoritative execution always
    /// buffers output and never performs this host effect.
    pub immediate_output: bool,
    /// Input values (simulated input).
    pub input: Vec<u64>,
    /// Error handling policy for bounds checking, division, etc.
    pub error_config: ErrorConfig,
    /// Where these epochs are being executed. The context carries the
    /// effects policy with it, so "not in a search but with a policy set"
    /// is unrepresentable.
    pub context: EpochContext,
}

/// Where an epoch runs, which determines whether the effect gate applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EpochContext {
    /// A single evaluation whose result is the timeline itself (trivially
    /// consistent programmes and direct executor use). Bytecode capabilities
    /// still default to denied.
    #[default]
    SingleEpoch,
    /// One iteration of a fixed-point search, with its effects policy.
    FixedPointSearch(EffectsPolicy),
}

impl EpochContext {
    /// True when the search context declines non-pure opcodes.
    #[cfg(test)]
    fn declines_effects(&self) -> bool {
        matches!(self, EpochContext::FixedPointSearch(EffectsPolicy::Decline))
    }

    /// True inside a fixed-point search under any policy.
    #[cfg(test)]
    fn in_search(&self) -> bool {
        matches!(self, EpochContext::FixedPointSearch(_))
    }
}

/// Policy for opcodes with an EffectClass other than Pure during a
/// fixed-point search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EffectsPolicy {
    /// Decline over fabricate: executing the opcode is an epoch error that
    /// names the opcode. The default.
    #[default]
    Decline,
    /// Permit everything, accepting that external effects repeat per search
    /// epoch and non-determinism may prevent convergence. Retained only as a
    /// legacy configuration value; `Executor` rejects it because unrestricted
    /// host authority has no exact bytecode capability mapping.
    Unrestricted,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_instructions: 10_000_000,
            immediate_output: true,
            input: Vec::new(),
            error_config: ErrorConfig::default(),
            context: EpochContext::default(),
        }
    }
}

/// Compatibility facade for executing source programs through authoritative
/// typed, independently verified bytecode.
///
/// `Executor` retains the historical source-level API, but it is not an AST
/// interpreter. Every epoch resolves names into HIR, runs mandatory semantic
/// analysis, lowers and independently verifies bytecode, and dispatches only
/// through [`BytecodeVm`]. Legacy host contexts cannot be represented by the
/// capability-oriented bytecode runtime and are rejected explicitly.
#[cfg_attr(test, allow(dead_code))]
pub struct Executor {
    pub config: ExecutorConfig,
    #[cfg(test)]
    input_cursor: usize,
    /// Inputs consumed during the current epoch (for capture).
    #[cfg(test)]
    inputs_consumed: Vec<u64>,
    /// Retained legacy I/O context. If present, authoritative epoch execution
    /// rejects the configuration instead of exposing mutable host resources.
    pub io_context: Option<IOContext>,
    /// Retained legacy FFI context. If present, authoritative epoch execution
    /// rejects the configuration; typed bytecode foreigns use `ForeignHostTable`.
    pub ffi_context: Option<FFIContext>,
    /// Active finite temporal scope. Nested scopes are deliberately rejected
    /// so address translation is unambiguous across all backends.
    #[cfg(test)]
    temporal_scope: Option<RuntimeTemporalScope>,
}

#[cfg(test)]
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
struct RuntimeTemporalScope {
    base: u64,
    size: u64,
    cell_bits: u8,
}

#[cfg_attr(test, allow(dead_code))]
impl Executor {
    /// Create a new executor with default configuration.
    pub fn new() -> Self {
        Self {
            config: ExecutorConfig::default(),
            #[cfg(test)]
            input_cursor: 0,
            #[cfg(test)]
            inputs_consumed: Vec::new(),
            io_context: None,
            ffi_context: None,
            #[cfg(test)]
            temporal_scope: None,
        }
    }

    /// Create an executor with custom configuration.
    pub fn with_config(config: ExecutorConfig) -> Self {
        Self {
            config,
            #[cfg(test)]
            input_cursor: 0,
            #[cfg(test)]
            inputs_consumed: Vec::new(),
            io_context: None,
            ffi_context: None,
            #[cfg(test)]
            temporal_scope: None,
        }
    }

    /// Retain legacy I/O and FFI contexts for source compatibility.
    ///
    /// Epoch execution rejects this configuration explicitly. Use
    /// [`BytecodeVm`] frozen observations/capabilities and a typed
    /// `ForeignHostTable` instead of these mutable host facades.
    pub fn with_contexts(
        config: ExecutorConfig,
        io_context: IOContext,
        ffi_context: FFIContext,
    ) -> Self {
        Self {
            config,
            #[cfg(test)]
            input_cursor: 0,
            #[cfg(test)]
            inputs_consumed: Vec::new(),
            io_context: Some(io_context),
            ffi_context: Some(ffi_context),
            #[cfg(test)]
            temporal_scope: None,
        }
    }

    /// Run a single epoch through typed and independently verified bytecode.
    ///
    /// Compilation, semantic, verification, capability, and runtime failures
    /// are represented as [`EpochStatus::Error`]. Output is always buffered in
    /// the result; the retained `immediate_output` compatibility flag no longer
    /// permits execution to perform a host effect.
    pub fn run_epoch(&mut self, program: &Program, anamnesis: &Memory) -> EpochResult {
        match self.run_bytecode(program, anamnesis) {
            Ok(execution) => epoch_result(execution),
            Err(error) => EpochResult {
                present: Memory::with_size(anamnesis.len()),
                output: Vec::new(),
                status: EpochStatus::Error(error.message),
                instructions_executed: error.instructions_executed,
                inputs_consumed: Vec::new(),
            },
        }
    }

    /// Execute into a pristine compatibility state through bytecode.
    ///
    /// The bytecode VM deliberately has no mutable mid-epoch entry point.
    /// Therefore a state carrying a pre-existing stack, present writes,
    /// output, collections, gas, or terminal status is rejected rather than
    /// being assigned ambiguous resume semantics.
    pub fn execute(&mut self, state: &mut VmState, program: &Program) -> Result<(), String> {
        if !state_is_pristine(state) {
            return Err(
                "Executor::execute requires a pristine VmState; bytecode execution cannot resume a legacy mid-epoch state"
                    .to_string(),
            );
        }

        let execution = self
            .run_bytecode(program, &state.anamnesis)
            .map_err(|error| error.message)?;
        state.stack = execution.stack;
        state.present = paged_to_memory(&execution.present);
        state.output = execution.output;
        state.status = bytecode_status(execution.status);
        state.instructions_executed = execution.instructions_executed;
        state.data_structures = execution.data_structures;
        Ok(())
    }

    fn run_bytecode(
        &self,
        program: &Program,
        anamnesis: &Memory,
    ) -> Result<BytecodeExecution, ExecutorFailure> {
        if self.io_context.is_some() || self.ffi_context.is_some() {
            return Err(ExecutorFailure::admission(
                "legacy IOContext/FFIContext execution is unavailable; use BytecodeVm with explicit frozen inputs and capabilities",
            ));
        }
        if matches!(
            self.config.context,
            EpochContext::FixedPointSearch(EffectsPolicy::Unrestricted)
        ) {
            return Err(ExecutorFailure::admission(
                "EffectsPolicy::Unrestricted has no bytecode capability mapping; use BytecodeTimeLoop with explicit capabilities",
            ));
        }
        if self.config.error_config.stack_underflow != StackPolicy::Error {
            return Err(ExecutorFailure::admission(
                "StackPolicy::ReturnZero is not an authoritative bytecode policy",
            ));
        }
        if !self.config.error_config.track_provenance {
            return Err(ExecutorFailure::admission(
                "disabling provenance is not supported by authoritative bytecode execution",
            ));
        }

        let type_report = type_check(program);
        if !type_report.is_valid {
            return Err(ExecutorFailure::admission(format!(
                "semantic analysis failed during type checking: {:?}",
                type_report.errors
            )));
        }
        let hir = HirProgram::resolve(program).map_err(|errors| {
            ExecutorFailure::admission(format!("HIR resolution failed: {errors:?}"))
        })?;
        let semantics = check_semantics(&hir);
        if !semantics.is_accepted_for_interpreter() {
            return Err(ExecutorFailure::admission(format!(
                "semantic analysis failed: {:?}",
                semantics.errors
            )));
        }
        let bytecode = BytecodeProgram::compile(&hir).map_err(|error| {
            ExecutorFailure::admission(format!("bytecode lowering failed: {error}"))
        })?;

        if self.config.error_config.division_by_zero != DivisionByZeroPolicy::ReturnZero
            && bytecode.instructions.iter().any(|instruction| {
                matches!(
                    instruction,
                    Instruction::Primitive(crate::ast::OpCode::Div | crate::ast::OpCode::Mod)
                )
            })
        {
            return Err(ExecutorFailure::admission(
                "the selected division-by-zero policy is not implemented by authoritative bytecode",
            ));
        }

        verify_bytecode(&bytecode).map_err(|error| {
            ExecutorFailure::admission(format!("independent bytecode verification failed: {error}"))
        })?;
        let prepared = PreparedBytecode::new(bytecode).map_err(|error| {
            ExecutorFailure::admission(format!("bytecode preparation failed: {error}"))
        })?;
        let memory = memory_to_paged(anamnesis).map_err(ExecutorFailure::admission)?;
        let max_stack_depth = match self.config.error_config.max_stack_depth {
            0 => usize::MAX,
            limit => limit,
        };
        let vm = BytecodeVm::with_config(BytecodeVmConfig {
            max_instructions: self.config.max_instructions,
            max_stack_depth,
            memory_bounds: self.config.error_config.memory_bounds,
            input: self.config.input.clone(),
            allow_interactive_input: false,
            ..BytecodeVmConfig::default()
        });
        let execution = vm
            .run_prepared(&prepared, &memory)
            .map_err(ExecutorFailure::runtime)?;
        if !execution.effects.is_empty() {
            return Err(ExecutorFailure::admission(
                "bytecode produced staged effects that Executor::run_epoch cannot expose; use BytecodeVm or BytecodeTimeLoop",
            ));
        }
        Ok(execution)
    }

    /// Test-only differential oracle for the retired direct AST walker.
    /// Production builds do not contain this entry point or its implementation.
    #[cfg(test)]
    pub(crate) fn execute_ast_oracle(
        &mut self,
        state: &mut VmState,
        program: &Program,
    ) -> Result<(), String> {
        self.execute_block(&program.body, state, &program.quotes)
    }

    /// Execute a block of statements.
    #[cfg(test)]
    fn execute_block(
        &mut self,
        stmts: &[Stmt],
        state: &mut VmState,
        quotes: &[Vec<Stmt>],
    ) -> Result<(), String> {
        for stmt in stmts {
            // Check for termination conditions
            match &state.status {
                EpochStatus::Paradox | EpochStatus::Finished | EpochStatus::Error(_) => {
                    return Ok(());
                }
                EpochStatus::Running => {}
            }

            // Check gas limit
            if state.instructions_executed >= self.config.max_instructions {
                state.status = EpochStatus::Error("Instruction limit exceeded".to_string());
                return Ok(());
            }

            self.execute_stmt(stmt, state, quotes)?;
        }
        Ok(())
    }

    /// Execute a single statement.
    #[cfg(test)]
    fn execute_stmt(
        &mut self,
        stmt: &Stmt,
        state: &mut VmState,
        quotes: &[Vec<Stmt>],
    ) -> Result<(), String> {
        state.instructions_executed += 1;

        match stmt {
            Stmt::Op(op) => self.execute_op(*op, state, quotes),

            Stmt::Push(v) => {
                state.stack.push(v.clone());
                Ok(())
            }

            Stmt::PushConstant { value, .. } => {
                state.stack.push(Value::new(*value));
                Ok(())
            }

            Stmt::ReadTemporal { address, .. } => {
                state.stack.push(Value::new(*address));
                self.execute_op(OpCode::Oracle, state, quotes)
            }

            Stmt::PushQuote(id) => {
                state.stack.push(Value::new(id.as_u64()));
                Ok(())
            }

            Stmt::Block(stmts) => self.execute_block(stmts, state, quotes),

            Stmt::If {
                then_branch,
                else_branch,
            } => {
                let cond = self.pop(state)?;
                if cond.to_bool() {
                    self.execute_block(then_branch, state, quotes)
                } else if let Some(else_stmts) = else_branch {
                    self.execute_block(else_stmts, state, quotes)
                } else {
                    Ok(())
                }
            }

            Stmt::While { cond, body } => {
                loop {
                    // Check termination
                    if state.status != EpochStatus::Running {
                        break;
                    }

                    // Evaluate condition
                    self.execute_block(cond, state, quotes)?;
                    let result = self.pop(state)?;

                    if !result.to_bool() {
                        break;
                    }

                    // Execute body
                    self.execute_block(body, state, quotes)?;

                    // Gas check for infinite loop prevention
                    if state.instructions_executed >= self.config.max_instructions {
                        state.status = EpochStatus::Error("Instruction limit in loop".to_string());
                        break;
                    }
                }
                Ok(())
            }

            Stmt::Call { name } => {
                // Note: Procedure lookup happens at runtime via the program
                // For now, this is a placeholder - actual inlining happens in run_epoch
                Err(format!(
                    "Procedure call '{}' not inlined - ensure program is preprocessed",
                    name
                ))
            }

            Stmt::TemporalScope {
                base,
                size,
                cell_bits,
                body,
            } => {
                // Temporal scoping: create an isolated memory region
                //
                // 1. Snapshot the memory region [base, base+size)
                // 2. Execute body with Oracle/Prophecy relative to base
                // 3. On success, propagate changes to parent
                // 4. On paradox/error, discard changes

                if self.temporal_scope.is_some() {
                    return Err("nested TEMPORAL scopes are not supported".to_string());
                }
                if *size == 0 || !(1..=64).contains(cell_bits) {
                    return Err(
                        "TEMPORAL scope requires a positive size and BITS in 1..=64".to_string()
                    );
                }
                let end = base
                    .checked_add(*size)
                    .ok_or_else(|| "TEMPORAL scope address range overflows u64".to_string())?;
                if end > state.present.len() as u64 {
                    return Err(format!(
                        "TEMPORAL scope [{}..{}) exceeds configured memory width {}",
                        base,
                        end,
                        state.present.len()
                    ));
                }

                // Snapshot present, not anamnesis: rollback restores the state
                // as it was immediately before entering the region.
                let snapshot: Vec<Value> = (*base..end)
                    .map(|address| state.present.read(address))
                    .collect();
                self.temporal_scope = Some(RuntimeTemporalScope {
                    base: *base,
                    size: *size,
                    cell_bits: *cell_bits,
                });
                let result = self.execute_block(body, state, quotes);
                self.temporal_scope = None;

                if result.is_err()
                    || matches!(state.status, EpochStatus::Paradox | EpochStatus::Error(_))
                {
                    for (offset, value) in snapshot.into_iter().enumerate() {
                        state.present.write(base + offset as u64, value);
                    }
                }
                result
            }
        }
    }

    #[cfg(test)]
    fn resolve_memory_address(&self, raw: u64, operation: MemoryOperation) -> Result<u64, String> {
        let Some(scope) = self.temporal_scope else {
            return Ok(raw);
        };
        let local = match self.config.error_config.memory_bounds {
            BoundsPolicy::Wrap => raw % scope.size,
            BoundsPolicy::Clamp => raw.min(scope.size - 1),
            BoundsPolicy::Error if raw >= scope.size => {
                return Err(format!(
                    "{} address {} is outside active TEMPORAL scope 0..{}",
                    operation, raw, scope.size
                ));
            }
            BoundsPolicy::Error => raw,
        };
        Ok(scope.base + local)
    }

    #[cfg(test)]
    fn truncate_scoped_value(&self, mut value: Value) -> Value {
        if let Some(scope) = self.temporal_scope {
            if scope.cell_bits < 64 {
                value.val &= (1u64 << scope.cell_bits) - 1;
            }
        }
        value
    }

    /// Execute a single opcode.
    #[cfg(test)]
    fn execute_op(
        &mut self,
        op: OpCode,
        state: &mut VmState,
        quotes: &[Vec<Stmt>],
    ) -> Result<(), String> {
        // Effect gate: inside a fixed-point search the epoch function must be
        // a fixed function of the memory state, and external effects would
        // fire once per search epoch rather than once per timeline. Checked
        // at dispatch, before operands are popped, so the decline is clean.
        if self.config.context.declines_effects() {
            match op.effect_class() {
                crate::ast::EffectClass::External => {
                    return Err(format!(
                        "{} is an external effect inside a fixed-point search; \
each epoch of the search would perform it again. \
Run with --effects unrestricted to permit this.",
                        op.name()
                    ));
                }
                crate::ast::EffectClass::NonDeterministic => {
                    return Err(format!(
                        "{} is non-deterministic inside a fixed-point search; \
the epoch function must return the same state for the same anamnesis. \
Run with --effects unrestricted to permit this.",
                        op.name()
                    ));
                }
                crate::ast::EffectClass::Pure => {}
            }
        }
        match op {
            // ═══════════════════════════════════════════════════════════
            // Stack Manipulation
            // ═══════════════════════════════════════════════════════════
            OpCode::Nop => {}

            OpCode::Halt => {
                state.status = EpochStatus::Finished;
            }

            OpCode::Pop => {
                self.pop(state)?;
            }

            OpCode::Dup => {
                let a = self.peek(state)?;
                state.stack.push(a);
            }
            OpCode::Pick => {
                let n_val = self.pop(state)?;
                let n = n_val.val;
                let len = state.stack.len() as u64;
                if n >= len {
                    return Err(format!("Pick out of bounds: index {} but depth {}", n, len));
                }
                let idx = (len - 1 - n) as usize;
                let val = state.stack[idx].clone();
                state.stack.push(val);
            }
            OpCode::Roll => {
                let n_val = self.pop(state)?;
                let n = n_val.val;
                let len = state.stack.len() as u64;
                if n >= len {
                    return Err(format!("Roll out of bounds: index {} but depth {}", n, len));
                }
                let idx = (len - 1 - n) as usize;
                let val = state.stack.remove(idx);
                state.stack.push(val);
            }
            OpCode::Reverse => {
                let n_val = self.pop(state)?;
                let count = n_val.val as usize;
                let len = state.stack.len();
                if count > len {
                    return Err(format!(
                        "Reverse out of bounds: count {} but depth {}",
                        count, len
                    ));
                }
                if count > 1 {
                    let start = len - count;
                    state.stack[start..].reverse();
                }
            }

            OpCode::Exec => {
                let id_val = self.pop(state)?;
                let id = id_val.val as usize;

                if id >= quotes.len() {
                    return Err(format!("Exec: Invalid quote ID {}", id));
                }

                // Recursively execute quote
                // Use recursion guard? (not strict requirement yet, gas limit handles it)
                self.execute_block(&quotes[id], state, quotes)?;
            }

            OpCode::Dip => {
                let id_val = self.pop(state)?;
                let id = id_val.val as usize;

                if id >= quotes.len() {
                    return Err(format!("Dip: Invalid quote ID {}", id));
                }

                // Pop X
                let x = self.pop(state)?;

                // Execute quote
                self.execute_block(&quotes[id], state, quotes)?;

                // Restore X
                state.stack.push(x);
            }

            OpCode::Keep => {
                let id_val = self.pop(state)?;
                let id = id_val.val as usize;

                if id >= quotes.len() {
                    return Err(format!("Keep: Invalid quote ID {}", id));
                }

                // Pop X (to be kept)
                let x = self.pop(state)?;

                // Restore X (it stays on stack)
                state.stack.push(x.clone());

                // Execute quote (it sees X)
                self.execute_block(&quotes[id], state, quotes)?;

                // Keep: ( x q -- .. x ).
                // Wait. Keep usually means preserve x across call.
                // Standard: ( x q -- .. x )
                // Apply q to x.
                // Stack: x. q.
                // q runs. Consumes x? Maybe.
                // If q consumes x, we need to restore it.
                // "Keep" implies we KEEP a copy of x for later use?
                // Or we keep x on top?
                // Factor `keep`: ( x q -- .. x )
                // "Call quote with x on stack, restoring x afterwards"

                // Implementation:
                // Pop q. Pop x.
                // Push x.
                // Run q.
                // Push x.
                // But if q consumes x, we need to ensure it was there.
                // My implementation:
                // Pop q, x.
                // Push x.
                // Execute q. (q sees x).
                // Push x. (Restores x).
                state.stack.push(x);
            }

            OpCode::Bi => {
                let q_val = self.pop(state)?;
                let q_id = q_val.val as usize;

                let p_val = self.pop(state)?;
                let p_id = p_val.val as usize;

                if q_id >= quotes.len() || p_id >= quotes.len() {
                    return Err("Bi: Invalid quote ID".to_string());
                }

                let x = self.pop(state)?;

                // Apply P to X
                state.stack.push(x.clone());
                self.execute_block(&quotes[p_id], state, quotes)?;

                // Apply Q to X
                state.stack.push(x);
                self.execute_block(&quotes[q_id], state, quotes)?;
            }

            OpCode::Rec => {
                let id_val = self.pop(state)?;
                let id = id_val.val as usize;

                if id >= quotes.len() {
                    return Err(format!("Rec: Invalid quote ID {}", id));
                }

                // Push quote back (for recursion inside)
                state.stack.push(id_val);

                // Execute quote
                self.execute_block(&quotes[id], state, quotes)?;
            }

            OpCode::StrRev => {
                let len_val = self.pop(state)?;
                let len = len_val.val as usize;
                if len > state.stack.len() {
                    return Err("StrRev: length exceeds stack depth".to_string());
                }
                if len > 0 {
                    let start = state.stack.len() - len;
                    state.stack[start..].reverse();
                }
                state.stack.push(len_val);
            }

            OpCode::Assert => {
                let len_val = self.pop(state)?;
                let len = len_val.val as usize;

                let mut chars: Vec<u8> = Vec::with_capacity(len);
                for _ in 0..len {
                    let val = self.pop(state)?;
                    chars.push(val.val as u8);
                }
                chars.reverse(); // Stack order is reverse of string order?
                                 // Wait, parser pushes char 0, then char 1... then len.
                                 // So top of stack is len, then char N, ..., char 0.
                                 // popping gives char N, ..., char 0.
                                 // So we need to reverse to get 0..N.
                                 // Actually parse logic:
                                 // "ABC" -> Push 'A', Push 'B', Push 'C', Push 3.
                                 // Stack: [..., A, B, C, 3] (Top is right)
                                 // Pop 3. Stack: [..., A, B, C]
                                 // Pop C, Pop B, Pop A.
                                 // Result C, B, A. Reverse -> A, B, C. Correct.

                let cond_val = self.pop(state)?;
                if cond_val.val == 0 {
                    let msg = String::from_utf8_lossy(&chars);
                    return Err(format!("Assertion failed: {}", msg));
                }
            }

            OpCode::StrCat => {
                let len2_val = self.pop(state)?;
                let len2 = len2_val.val as usize;

                // Pop string 2 chars
                let mut s2: Vec<Value> = Vec::with_capacity(len2);
                for _ in 0..len2 {
                    s2.push(self.pop(state)?);
                }
                s2.reverse(); // Popped in reverse order

                // Pop len1
                let len1_val = self.pop(state)?;
                let len1 = len1_val.val as usize;

                // s1 is already on stack. Just verify depth.
                if len1 > state.stack.len() {
                    return Err("StrCat: string 1 length exceeds stack depth".to_string());
                }

                // Push s2 back
                for v in s2 {
                    state.stack.push(v);
                }

                // Push total length
                state.stack.push(Value {
                    val: (len1 + len2) as u64,
                    prov: len1_val.prov.merge(&len2_val.prov),
                });
            }

            OpCode::StrSplit => {
                let delim_val = self.pop(state)?;
                let delim = delim_val.val;

                let len_val = self.pop(state)?;
                let len = len_val.val as usize;

                // Pop string chars
                let mut chars: Vec<Value> = Vec::with_capacity(len);
                for _ in 0..len {
                    chars.push(self.pop(state)?);
                }
                chars.reverse();

                // Split logic
                let mut parts: Vec<Vec<Value>> = Vec::new();
                let mut current: Vec<Value> = Vec::new();

                for c in chars {
                    if c.val == delim {
                        parts.push(current);
                        current = Vec::new();
                    } else {
                        current.push(c);
                    }
                }
                parts.push(current); // Last part

                // Push parts back to stack
                let count = parts.len();
                for part in parts {
                    let part_len = part.len() as u64;
                    for c in part {
                        state.stack.push(c);
                    }
                    state.stack.push(Value::new(part_len));
                }

                state.stack.push(Value::new(count as u64));
            }

            OpCode::Swap => {
                let a = self.pop(state)?;
                let b = self.pop(state)?;
                state.stack.push(a);
                state.stack.push(b);
            }

            OpCode::Over => {
                if state.stack.len() < 2 {
                    return Err("Stack underflow: OVER requires 2 elements".to_string());
                }
                let val = state.stack[state.stack.len() - 2].clone();
                state.stack.push(val);
            }

            OpCode::Rot => {
                if state.stack.len() < 3 {
                    return Err("Stack underflow: ROT requires 3 elements".to_string());
                }
                let len = state.stack.len();
                let c = state.stack.remove(len - 3);
                state.stack.push(c);
            }

            OpCode::Depth => {
                state.stack.push(Value::new(state.stack.len() as u64));
            }

            // ═══════════════════════════════════════════════════════════
            // Arithmetic
            // ═══════════════════════════════════════════════════════════
            OpCode::Add => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                state.stack.push(a + b);
            }

            OpCode::Sub => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                state.stack.push(a - b);
            }

            OpCode::Mul => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                state.stack.push(a * b);
            }

            OpCode::Div => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                state.stack.push(a / b); // Returns 0 for div by 0
            }

            OpCode::Mod => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                state.stack.push(a % b); // Returns 0 for mod by 0
            }

            OpCode::Neg => {
                let a = self.pop(state)?;
                state.stack.push(Value {
                    val: a.val.wrapping_neg(),
                    prov: a.prov,
                });
            }

            OpCode::Abs => {
                let a = self.pop(state)?;
                state.stack.push(a.abs());
            }

            OpCode::Min => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                let prov = a.prov.merge(&b.prov);
                let min_val = if a.val < b.val { a.val } else { b.val };
                state.stack.push(Value { val: min_val, prov });
            }

            OpCode::Max => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                let prov = a.prov.merge(&b.prov);
                let max_val = if a.val > b.val { a.val } else { b.val };
                state.stack.push(Value { val: max_val, prov });
            }

            OpCode::Sign => {
                let a = self.pop(state)?;
                state.stack.push(a.signum());
            }

            // ═══════════════════════════════════════════════════════════
            // Bitwise Logic
            // ═══════════════════════════════════════════════════════════
            OpCode::Not => {
                let a = self.pop(state)?;
                // Logical NOT: 0 -> 1, nonzero -> 0
                let result = if a.val == 0 { 1 } else { 0 };
                state.stack.push(Value {
                    val: result,
                    prov: a.prov,
                });
            }

            OpCode::And => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                state.stack.push(a & b);
            }

            OpCode::Or => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                state.stack.push(a | b);
            }

            OpCode::Xor => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                state.stack.push(a ^ b);
            }

            OpCode::Shl => {
                let n = self.pop(state)?;
                let a = self.pop(state)?;
                let shift = (n.val % 64) as u32;
                state.stack.push(Value {
                    val: a.val.wrapping_shl(shift),
                    prov: a.prov.merge(&n.prov),
                });
            }

            OpCode::Shr => {
                let n = self.pop(state)?;
                let a = self.pop(state)?;
                let shift = (n.val % 64) as u32;
                state.stack.push(Value {
                    val: a.val.wrapping_shr(shift),
                    prov: a.prov.merge(&n.prov),
                });
            }

            // ═══════════════════════════════════════════════════════════
            // Comparison
            // ═══════════════════════════════════════════════════════════
            OpCode::Eq => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                let prov = a.prov.merge(&b.prov);
                state
                    .stack
                    .push(Value::from_bool_with_prov(a.val == b.val, prov));
            }

            OpCode::Neq => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                let prov = a.prov.merge(&b.prov);
                state
                    .stack
                    .push(Value::from_bool_with_prov(a.val != b.val, prov));
            }

            OpCode::Lt => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                let prov = a.prov.merge(&b.prov);
                state
                    .stack
                    .push(Value::from_bool_with_prov(a.val < b.val, prov));
            }

            OpCode::Gt => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                let prov = a.prov.merge(&b.prov);
                state
                    .stack
                    .push(Value::from_bool_with_prov(a.val > b.val, prov));
            }

            OpCode::Lte => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                let prov = a.prov.merge(&b.prov);
                state
                    .stack
                    .push(Value::from_bool_with_prov(a.val <= b.val, prov));
            }

            OpCode::Gte => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                let prov = a.prov.merge(&b.prov);
                state
                    .stack
                    .push(Value::from_bool_with_prov(a.val >= b.val, prov));
            }

            // ═══════════════════════════════════════════════════════════
            // Signed Comparison
            // ═══════════════════════════════════════════════════════════
            OpCode::Slt => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                state.stack.push(a.signed_lt(b));
            }

            OpCode::Sgt => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                state.stack.push(a.signed_gt(b));
            }

            OpCode::Slte => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                state.stack.push(a.signed_lte(b));
            }

            OpCode::Sgte => {
                let b = self.pop(state)?;
                let a = self.pop(state)?;
                state.stack.push(a.signed_gte(b));
            }

            // ═══════════════════════════════════════════════════════════
            // Temporal Operations
            // ═══════════════════════════════════════════════════════════
            OpCode::Oracle => {
                let addr_val = self.pop(state)?;
                let location = SourceLocation::at_stmt(state.instructions_executed as usize);
                let policy = self.config.error_config.memory_bounds;
                let address = self.resolve_memory_address(addr_val.val, MemoryOperation::Oracle)?;

                // Bounds-checked read from anamnesis
                let mut val = state
                    .anamnesis
                    .oracle_read(address, policy, location)
                    .map_err(|e| e.to_string())?;

                // Inject oracle provenance using the validated address
                let addr = state
                    .anamnesis
                    .validate_address_in(
                        address,
                        policy,
                        MemoryOperation::Oracle,
                        SourceLocation::default(),
                    )
                    .unwrap_or(address as Address);
                let oracle_prov = Provenance::single(addr);
                val.prov = val.prov.merge(&oracle_prov).merge(&addr_val.prov);

                state.stack.push(val);
            }

            OpCode::Prophecy => {
                let addr_val = self.pop(state)?;
                let popped = self.pop(state)?;
                let val = self.truncate_scoped_value(popped);
                let location = SourceLocation::at_stmt(state.instructions_executed as usize);
                let policy = self.config.error_config.memory_bounds;
                let address =
                    self.resolve_memory_address(addr_val.val, MemoryOperation::Prophecy)?;

                // Bounds-checked write to present
                state
                    .present
                    .prophecy_write(address, val, policy, location)
                    .map_err(|e| e.to_string())?;
            }

            OpCode::PresentRead => {
                let addr_val = self.pop(state)?;
                let location = SourceLocation::at_stmt(state.instructions_executed as usize);
                let policy = self.config.error_config.memory_bounds;
                let address = self.resolve_memory_address(addr_val.val, MemoryOperation::Read)?;

                // Bounds-checked read from present
                let mut val = state
                    .present
                    .read_checked(address, policy, location)
                    .map_err(|e| e.to_string())?;
                val.prov = val.prov.merge(&addr_val.prov);

                state.stack.push(val);
            }

            OpCode::Paradox => {
                state.status = EpochStatus::Paradox;
            }

            // ═══════════════════════════════════════════════════════════
            // I/O
            // ═══════════════════════════════════════════════════════════
            OpCode::Input => {
                let val = if self.input_cursor < self.config.input.len() {
                    let v = self.config.input[self.input_cursor];
                    self.input_cursor += 1;
                    self.inputs_consumed.push(v);
                    v
                } else if self.config.context.in_search() && !self.config.input.is_empty() {
                    // The input stream is frozen (Temporal Input Invariant);
                    // reading past its end would re-open the live stdin the
                    // freeze exists to shut, making F non-deterministic.
                    return Err(format!(
                        "INPUT beyond the {} frozen input value(s) inside a \
fixed-point search; the input stream must be fixed for the search to converge",
                        self.config.input.len()
                    ));
                } else {
                    // First consuming epoch of a search (nothing frozen yet)
                    // or a single-epoch run: read interactively.
                    let v = self.read_input_interactive()?;
                    self.inputs_consumed.push(v);
                    v
                };
                state.stack.push(Value::new(val));
            }

            OpCode::Output => {
                let val = self.pop(state)?;

                if self.config.immediate_output {
                    println!("OUTPUT: {} [deps: {:?}]", val.val, val.prov);
                }

                state.output.push(OutputItem::Val(val));
            }

            OpCode::Emit => {
                let val = self.pop(state)?;
                let char_val = (val.val % 256) as u8;

                if self.config.immediate_output {
                    // For immediate output in debug mode, maybe show explicit EMIT
                    println!("EMIT: '{}' ({})", char_val as char, char_val);
                }

                state.output.push(OutputItem::Char(char_val));
            }

            // Array/Memory Operations
            OpCode::Pack => {
                // Pack n values into contiguous memory at base address
                let n = self.pop(state)?.val as usize;
                let base_val = self.pop(state)?;
                let location = SourceLocation::at_stmt(state.instructions_executed as usize);
                let policy = self.config.error_config.memory_bounds;
                let mut values = Vec::with_capacity(n);
                for _ in 0..n {
                    values.push(self.pop(state)?);
                }
                values.reverse();
                for (offset, value) in values.into_iter().enumerate() {
                    let address = self.resolve_memory_address(
                        base_val.val.wrapping_add(offset as u64),
                        MemoryOperation::Pack,
                    )?;
                    let value = self.truncate_scoped_value(value);
                    state
                        .present
                        .write_checked(address, value, policy, location.clone())
                        .map_err(|e| e.to_string())?;
                }
            }

            OpCode::Unpack => {
                // Unpack n values from contiguous memory at base address
                let n = self.pop(state)?.val as usize;
                let base_val = self.pop(state)?;
                let location = SourceLocation::at_stmt(state.instructions_executed as usize);
                let policy = self.config.error_config.memory_bounds;
                for offset in 0..n {
                    let address = self.resolve_memory_address(
                        base_val.val.wrapping_add(offset as u64),
                        MemoryOperation::Unpack,
                    )?;
                    let val = state
                        .present
                        .read_checked(address, policy, location.clone())
                        .map_err(|e| e.to_string())?;
                    state.stack.push(val);
                }
            }

            OpCode::Index => {
                // Read from base + index
                let index_val = self.pop(state)?;
                let base_val = self.pop(state)?;
                let location = SourceLocation::at_stmt(state.instructions_executed as usize);
                let policy = self.config.error_config.memory_bounds;
                let address = self.resolve_memory_address(
                    base_val.val.wrapping_add(index_val.val),
                    MemoryOperation::Index,
                )?;
                let val = state
                    .present
                    .read_checked(address, policy, location)
                    .map_err(|e| e.to_string())?;
                state.stack.push(val);
            }

            OpCode::Store => {
                // Store value at base + index
                let index_val = self.pop(state)?;
                let base_val = self.pop(state)?;
                let popped = self.pop(state)?;
                let val = self.truncate_scoped_value(popped);
                let location = SourceLocation::at_stmt(state.instructions_executed as usize);
                let policy = self.config.error_config.memory_bounds;
                let address = self.resolve_memory_address(
                    base_val.val.wrapping_add(index_val.val),
                    MemoryOperation::Store,
                )?;
                state
                    .present
                    .write_checked(address, val, policy, location)
                    .map_err(|e| e.to_string())?;
            }

            // ═══════════════════════════════════════════════════════════
            // Vector Operations
            // ═══════════════════════════════════════════════════════════
            OpCode::VecNew => {
                let handle = state.data_structures.vectors.alloc();
                state.stack.push(Value::new(handle));
            }

            OpCode::VecPush => {
                let value = self.pop(state)?;
                let handle = self.pop(state)?.val;
                state
                    .data_structures
                    .vectors
                    .push(handle, value)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
            }

            OpCode::VecPop => {
                let handle = self.pop(state)?.val;
                let value = state
                    .data_structures
                    .vectors
                    .pop(handle)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
                state.stack.push(value);
            }

            OpCode::VecGet => {
                let index = self.pop(state)?.val;
                let handle = self.pop(state)?.val;
                let value = state
                    .data_structures
                    .vectors
                    .get_at(handle, index)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
                state.stack.push(value);
            }

            OpCode::VecSet => {
                let index = self.pop(state)?.val;
                let value = self.pop(state)?;
                let handle = self.pop(state)?.val;
                state
                    .data_structures
                    .vectors
                    .set_at(handle, index, value)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
            }

            OpCode::VecLen => {
                let handle = self.pop(state)?.val;
                let length = state
                    .data_structures
                    .vectors
                    .len(handle)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
                state.stack.push(Value::new(length));
            }

            // ═══════════════════════════════════════════════════════════
            // Hash Table Operations
            // ═══════════════════════════════════════════════════════════
            OpCode::HashNew => {
                let handle = state.data_structures.hashes.alloc();
                state.stack.push(Value::new(handle));
            }

            OpCode::HashPut => {
                let value = self.pop(state)?;
                let key = self.pop(state)?.val;
                let handle = self.pop(state)?.val;
                state
                    .data_structures
                    .hashes
                    .put(handle, key, value)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
            }

            OpCode::HashGet => {
                let key = self.pop(state)?.val;
                let handle = self.pop(state)?.val;
                let (value, found) = state
                    .data_structures
                    .hashes
                    .get_key(handle, key)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
                state.stack.push(value);
                state.stack.push(Value::new(if found { 1 } else { 0 }));
            }

            OpCode::HashDel => {
                let key = self.pop(state)?.val;
                let handle = self.pop(state)?.val;
                state
                    .data_structures
                    .hashes
                    .delete(handle, key)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
            }

            OpCode::HashHas => {
                let key = self.pop(state)?.val;
                let handle = self.pop(state)?.val;
                let found = state
                    .data_structures
                    .hashes
                    .has(handle, key)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
                state.stack.push(Value::new(if found { 1 } else { 0 }));
            }

            OpCode::HashLen => {
                let handle = self.pop(state)?.val;
                let count = state
                    .data_structures
                    .hashes
                    .len(handle)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
                state.stack.push(Value::new(count));
            }

            // ═══════════════════════════════════════════════════════════
            // Set Operations
            // ═══════════════════════════════════════════════════════════
            OpCode::SetNew => {
                let handle = state.data_structures.sets.alloc();
                state.stack.push(Value::new(handle));
            }

            OpCode::SetAdd => {
                let value = self.pop(state)?.val;
                let handle = self.pop(state)?.val;
                state
                    .data_structures
                    .sets
                    .add(handle, value)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
            }

            OpCode::SetHas => {
                let value = self.pop(state)?.val;
                let handle = self.pop(state)?.val;
                let found = state
                    .data_structures
                    .sets
                    .has(handle, value)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
                state.stack.push(Value::new(if found { 1 } else { 0 }));
            }

            OpCode::SetDel => {
                let value = self.pop(state)?.val;
                let handle = self.pop(state)?.val;
                state
                    .data_structures
                    .sets
                    .delete(handle, value)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
            }

            OpCode::SetLen => {
                let handle = self.pop(state)?.val;
                let count = state
                    .data_structures
                    .sets
                    .len(handle)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
                state.stack.push(Value::new(count));
            }

            // ═══════════════════════════════════════════════════════════
            // FFI Operations
            // ═══════════════════════════════════════════════════════════
            OpCode::FFICall => {
                // ( arg1..argN function_id -- result1..resultM )
                let id_val = self.pop(state)?;
                let id = id_val.val as u32;
                let ctx = self.ffi_ctx()?;
                FFICaller::call_by_id(ctx, state, id, SourceLocation::default())
                    .map_err(|e| e.to_string())?;
            }

            OpCode::FFICallNamed => {
                // ( arg1..argN name_len name_chars... -- result1..resultM )
                // Pop name length and characters to build function name
                let name_len = self.pop(state)?.val as usize;
                let mut name_chars = Vec::with_capacity(name_len);
                for _ in 0..name_len {
                    name_chars.push(self.pop(state)?.val as u8 as char);
                }
                name_chars.reverse();
                let name: String = name_chars.into_iter().collect();

                let ctx = self.ffi_ctx()?;
                FFICaller::call_by_name(ctx, state, &name, SourceLocation::default())
                    .map_err(|e| e.to_string())?;
            }

            // ═══════════════════════════════════════════════════════════
            // File I/O Operations
            // ═══════════════════════════════════════════════════════════
            OpCode::FileOpen => {
                // ( path_len path_chars... mode -- file_handle )
                let mode_val = self.pop(state)?.val;
                let mode = FileMode::from_u64(mode_val);

                // The effect gate cannot classify FILE_OPEN statically
                // because destructiveness depends on this operand: a
                // create/truncate open would rewrite the file on every
                // epoch of the search.
                if self.config.context.declines_effects() && mode.is_destructive() {
                    return Err(
                        "FILE_OPEN with a write, append, create, or truncate mode is an \
external effect inside a fixed-point search; each epoch of the search would \
reapply it. Run with --effects unrestricted to permit this."
                            .to_string(),
                    );
                }

                let path_len = self.pop(state)?.val as usize;
                let mut path_chars = Vec::with_capacity(path_len);
                for _ in 0..path_len {
                    path_chars.push(self.pop(state)?.val as u8 as char);
                }
                path_chars.reverse();
                let path: String = path_chars.into_iter().collect();

                let handle = self
                    .io_ctx()?
                    .file_open(&path, mode)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
            }

            OpCode::FileRead => {
                // ( file_handle max_bytes -- file_handle buffer_handle bytes_read )
                let max_bytes = self.pop(state)?.val as usize;
                let file_handle = self.pop(state)?.val;

                let (buffer_handle, bytes_read) = self
                    .io_ctx()?
                    .file_read(file_handle, max_bytes)
                    .map_err(|e| e.to_string())?;

                state.stack.push(Value::new(file_handle));
                state.stack.push(Value::new(buffer_handle));
                state.stack.push(Value::new(bytes_read as u64));
            }

            OpCode::FileWrite => {
                // ( file_handle buffer_handle -- file_handle bytes_written )
                let buffer_handle = self.pop(state)?.val;
                let file_handle = self.pop(state)?.val;

                let bytes_written = self
                    .io_ctx()?
                    .file_write(file_handle, buffer_handle)
                    .map_err(|e| e.to_string())?;

                state.stack.push(Value::new(file_handle));
                state.stack.push(Value::new(bytes_written as u64));
            }

            OpCode::FileSeek => {
                // ( file_handle offset origin -- file_handle new_position )
                let origin_val = self.pop(state)?.val;
                let offset = self.pop(state)?.val as i64;
                let file_handle = self.pop(state)?.val;

                let origin = SeekOrigin::from_u64(origin_val);
                let new_pos = self
                    .io_ctx()?
                    .file_seek(file_handle, origin, offset)
                    .map_err(|e| e.to_string())?;

                state.stack.push(Value::new(file_handle));
                state.stack.push(Value::new(new_pos));
            }

            OpCode::FileFlush => {
                // ( file_handle -- file_handle )
                let file_handle = self.pop(state)?.val;

                self.io_ctx()?
                    .file_flush(file_handle)
                    .map_err(|e| e.to_string())?;

                state.stack.push(Value::new(file_handle));
            }

            OpCode::FileClose => {
                // ( file_handle -- )
                let file_handle = self.pop(state)?.val;

                self.io_ctx()?
                    .file_close(file_handle)
                    .map_err(|e| e.to_string())?;
            }

            OpCode::FileExists => {
                // ( path_len path_chars... -- exists )
                let path_len = self.pop(state)?.val as usize;
                let mut path_chars = Vec::with_capacity(path_len);
                for _ in 0..path_len {
                    path_chars.push(self.pop(state)?.val as u8 as char);
                }
                path_chars.reverse();
                let path: String = path_chars.into_iter().collect();

                let exists = self.io_ctx()?.file_exists(&path);
                state.stack.push(Value::new(if exists { 1 } else { 0 }));
            }

            OpCode::FileSize => {
                // ( path_len path_chars... -- size )
                let path_len = self.pop(state)?.val as usize;
                let mut path_chars = Vec::with_capacity(path_len);
                for _ in 0..path_len {
                    path_chars.push(self.pop(state)?.val as u8 as char);
                }
                path_chars.reverse();
                let path: String = path_chars.into_iter().collect();

                let size = self.io_ctx()?.file_size(&path).unwrap_or(0);
                state.stack.push(Value::new(size));
            }

            // ═══════════════════════════════════════════════════════════
            // Buffer Operations
            // ═══════════════════════════════════════════════════════════
            OpCode::BufferNew => {
                // ( capacity -- buffer_handle )
                let capacity = self.pop(state)?.val as usize;
                let handle = self.io_ctx()?.buffer_new(capacity);
                state.stack.push(Value::new(handle));
            }

            OpCode::BufferFromStack => {
                // ( byte1..byteN n -- buffer_handle )
                let n = self.pop(state)?.val as usize;
                let mut bytes = Vec::with_capacity(n);
                for _ in 0..n {
                    bytes.push(self.pop(state)?);
                }
                bytes.reverse();
                let handle = self.io_ctx()?.buffer_from_values(&bytes);
                state.stack.push(Value::new(handle));
            }

            OpCode::BufferToStack => {
                // ( buffer_handle -- byte1..byteN n )
                let handle = self.pop(state)?.val;
                let values = self
                    .io_ctx()?
                    .buffer_to_values(handle)
                    .map_err(|e| e.to_string())?;
                let len = values.len();
                for v in values {
                    state.stack.push(v);
                }
                state.stack.push(Value::new(len as u64));
            }

            OpCode::BufferLen => {
                // ( buffer_handle -- buffer_handle length )
                let handle = self.pop(state)?.val;
                let length = self
                    .io_ctx()?
                    .buffer_len(handle)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
                state.stack.push(Value::new(length as u64));
            }

            OpCode::BufferReadByte => {
                // ( buffer_handle -- buffer_handle byte )
                let handle = self.pop(state)?.val;
                let byte = self
                    .io_ctx()?
                    .buffer_read_byte(handle)
                    .map_err(|e| e.to_string())?
                    .unwrap_or(0);
                state.stack.push(Value::new(handle));
                state.stack.push(Value::new(byte as u64));
            }

            OpCode::BufferWriteByte => {
                // ( buffer_handle byte -- buffer_handle )
                let byte = self.pop(state)?.val as u8;
                let handle = self.pop(state)?.val;
                self.io_ctx()?
                    .buffer_write_byte(handle, byte)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
            }

            OpCode::BufferFree => {
                // ( buffer_handle -- )
                let handle = self.pop(state)?.val;
                self.io_ctx()?.buffer_free(handle);
            }

            // ═══════════════════════════════════════════════════════════
            // Network Operations
            // ═══════════════════════════════════════════════════════════
            OpCode::TcpConnect => {
                // ( host_len host_chars... port -- socket_handle )
                let port = self.pop(state)?.val as u16;
                let host_len = self.pop(state)?.val as usize;
                let mut host_chars = Vec::with_capacity(host_len);
                for _ in 0..host_len {
                    host_chars.push(self.pop(state)?.val as u8 as char);
                }
                host_chars.reverse();
                let host: String = host_chars.into_iter().collect();

                let handle = self
                    .io_ctx()?
                    .tcp_connect(&host, port)
                    .map_err(|e| e.to_string())?;
                state.stack.push(Value::new(handle));
            }

            OpCode::SocketSend => {
                // ( socket_handle buffer_handle -- socket_handle bytes_sent )
                let buffer_handle = self.pop(state)?.val;
                let socket_handle = self.pop(state)?.val;

                let bytes_sent = self
                    .io_ctx()?
                    .socket_send(socket_handle, buffer_handle)
                    .map_err(|e| e.to_string())?;

                state.stack.push(Value::new(socket_handle));
                state.stack.push(Value::new(bytes_sent as u64));
            }

            OpCode::SocketRecv => {
                // ( socket_handle max_bytes -- socket_handle buffer_handle bytes_recv )
                let max_bytes = self.pop(state)?.val as usize;
                let socket_handle = self.pop(state)?.val;

                let (buffer_handle, bytes_recv) = self
                    .io_ctx()?
                    .socket_recv(socket_handle, max_bytes)
                    .map_err(|e| e.to_string())?;

                state.stack.push(Value::new(socket_handle));
                state.stack.push(Value::new(buffer_handle));
                state.stack.push(Value::new(bytes_recv as u64));
            }

            OpCode::SocketClose => {
                // ( socket_handle -- )
                let socket_handle = self.pop(state)?.val;

                self.io_ctx()?
                    .socket_close(socket_handle)
                    .map_err(|e| e.to_string())?;
            }

            // ═══════════════════════════════════════════════════════════
            // Process Operations
            // ═══════════════════════════════════════════════════════════
            OpCode::ProcExec => {
                // ( cmd_len cmd_chars... -- output_handle exit_code )
                let cmd_len = self.pop(state)?.val as usize;
                let mut cmd_chars = Vec::with_capacity(cmd_len);
                for _ in 0..cmd_len {
                    cmd_chars.push(self.pop(state)?.val as u8 as char);
                }
                cmd_chars.reverse();
                let command: String = cmd_chars.into_iter().collect();

                let (output_handle, exit_code) =
                    self.io_ctx()?.exec(&command).map_err(|e| e.to_string())?;

                state.stack.push(Value::new(output_handle));
                state.stack.push(Value::new(exit_code as u64));
            }

            // ═══════════════════════════════════════════════════════════
            // System Operations
            // ═══════════════════════════════════════════════════════════
            OpCode::Clock => {
                // ( -- timestamp )
                use std::time::{SystemTime, UNIX_EPOCH};
                let duration = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default();
                state.stack.push(Value::new(duration.as_millis() as u64));
            }

            OpCode::Sleep => {
                // ( milliseconds -- )
                let ms = self.pop(state)?.val;
                std::thread::sleep(std::time::Duration::from_millis(ms));
            }

            OpCode::Random => {
                // ( -- random_value )
                state.stack.push(Value::new(random_u64()));
            }
        }

        Ok(())
    }

    /// Pop a value from the stack.
    #[cfg(test)]
    fn pop(&self, state: &mut VmState) -> Result<Value, String> {
        state
            .stack
            .pop()
            .ok_or_else(|| "Stack underflow".to_string())
    }

    /// Peek at the top of the stack.
    #[cfg(test)]
    fn peek(&self, state: &VmState) -> Result<Value, String> {
        state
            .stack
            .last()
            .cloned()
            .ok_or_else(|| "Stack underflow".to_string())
    }

    /// Get a mutable reference to the I/O context, lazily initialising if absent.
    #[cfg(test)]
    fn io_ctx(&mut self) -> Result<&mut IOContext, String> {
        if self.io_context.is_none() {
            self.io_context = Some(IOContext::new());
        }
        Ok(self.io_context.as_mut().unwrap())
    }

    /// Get a mutable reference to the FFI context, lazily initialising if absent.
    #[cfg(test)]
    fn ffi_ctx(&mut self) -> Result<&mut FFIContext, String> {
        if self.ffi_context.is_none() {
            self.ffi_context = Some(FFIContext::new());
        }
        Ok(self.ffi_context.as_mut().unwrap())
    }

    /// Read input interactively.
    #[cfg(test)]
    fn read_input_interactive(&self) -> Result<u64, String> {
        read_input_interactive().map_err(|error| error.to_string())
    }
}

#[derive(Debug)]
struct ExecutorFailure {
    message: String,
    instructions_executed: u64,
}

impl ExecutorFailure {
    fn admission(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            instructions_executed: 0,
        }
    }

    fn runtime(error: BytecodeVmError) -> Self {
        let instructions_executed = match &error {
            BytecodeVmError::GasExhausted { limit } => *limit,
            _ => 0,
        };
        Self {
            message: format!("bytecode execution failed: {error}"),
            instructions_executed,
        }
    }
}

fn state_is_pristine(state: &VmState) -> bool {
    state.stack.is_empty()
        && state.present.exact_sparse_state().cells().is_empty()
        && state.output.is_empty()
        && state.status == EpochStatus::Running
        && state.instructions_executed == 0
        && state.data_structures.vectors.count() == 0
        && state.data_structures.hashes.count() == 0
        && state.data_structures.sets.count() == 0
}

fn memory_to_paged(memory: &Memory) -> Result<PagedMemory, String> {
    let mut paged = PagedMemory::with_size(memory.len())
        .map_err(|error| format!("bytecode memory construction failed: {error}"))?;
    for (address, value) in memory.exact_sparse_state().cells() {
        paged
            .write(*address, value.clone())
            .map_err(|error| format!("bytecode memory transfer failed: {error}"))?;
    }
    Ok(paged)
}

fn paged_to_memory(memory: &PagedMemory) -> Memory {
    let mut dense = Memory::with_size(memory.len());
    for (address, value) in memory.iter_sparse() {
        dense.write(address, value.clone());
    }
    dense
}

fn bytecode_status(status: BytecodeVmStatus) -> EpochStatus {
    match status {
        BytecodeVmStatus::Finished | BytecodeVmStatus::Halted => EpochStatus::Finished,
        BytecodeVmStatus::Paradox => EpochStatus::Paradox,
    }
}

fn epoch_result(execution: BytecodeExecution) -> EpochResult {
    EpochResult {
        present: paged_to_memory(&execution.present),
        output: execution.output,
        status: bytecode_status(execution.status),
        instructions_executed: execution.instructions_executed,
        inputs_consumed: execution.inputs_consumed,
    }
}

/// Maximum bytes accepted for one interactive numeric token, including its
/// optional line ending. The bounded reader consumes at most one extra byte to
/// distinguish a full valid line from an oversized one.
pub(crate) const MAX_INTERACTIVE_INPUT_BYTES: usize = 128;

/// Typed failure from the opt-in live INPUT reader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum InteractiveInputError {
    EndOfFile,
    TooLong { limit: usize },
    InvalidValue,
    Io(String),
}

impl std::fmt::Display for InteractiveInputError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EndOfFile => formatter.write_str("interactive input reached end of file"),
            Self::TooLong { limit } => {
                write!(formatter, "interactive input exceeds {limit} bytes")
            }
            Self::InvalidValue => {
                formatter.write_str("interactive input is not an unsigned 64-bit integer")
            }
            Self::Io(message) => write!(formatter, "interactive input I/O failed: {message}"),
        }
    }
}

fn read_bounded_input(reader: &mut impl BufRead) -> Result<u64, InteractiveInputError> {
    let read_limit = u64::try_from(MAX_INTERACTIVE_INPUT_BYTES)
        .unwrap_or(u64::MAX)
        .saturating_add(1);
    let mut bytes = Vec::with_capacity(MAX_INTERACTIVE_INPUT_BYTES + 1);
    Read::take(reader, read_limit)
        .read_until(b'\n', &mut bytes)
        .map_err(|error| InteractiveInputError::Io(error.to_string()))?;
    if bytes.is_empty() {
        return Err(InteractiveInputError::EndOfFile);
    }
    if bytes.len() > MAX_INTERACTIVE_INPUT_BYTES {
        return Err(InteractiveInputError::TooLong {
            limit: MAX_INTERACTIVE_INPUT_BYTES,
        });
    }
    let text = std::str::from_utf8(&bytes).map_err(|_| InteractiveInputError::InvalidValue)?;
    text.trim()
        .parse()
        .map_err(|_| InteractiveInputError::InvalidValue)
}

/// Read one bounded line from stdin as a `u64`.
/// Shared by both VMs so opt-in live INPUT has one typed failure semantics.
pub(crate) fn read_input_interactive() -> Result<u64, InteractiveInputError> {
    print!("INPUT> ");
    io::stdout()
        .flush()
        .map_err(|error| InteractiveInputError::Io(error.to_string()))?;

    let stdin = io::stdin();
    read_bounded_input(&mut stdin.lock())
}

/// Time-seeded pseudo-random value. Shared by both VMs so RANDOM has a
/// single (deliberately non-deterministic) semantics.
///
/// SplitMix64 over nanosecond time plus a process-wide draw counter: the
/// counter guarantees consecutive draws differ even within one timer tick,
/// and the finaliser spreads results across the full u64 range (the previous
/// subsec_nanos value never exceeded a billion).
pub(crate) fn random_u64() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static DRAWS: AtomicU64 = AtomicU64::new(0);

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let draw = DRAWS.fetch_add(1, Ordering::Relaxed);
    let mut z = nanos.wrapping_add(draw.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn random_draws_are_distinct_within_one_tick() {
        // The draw counter must separate values even when the clock does
        // not advance between calls.
        let a = super::random_u64();
        let b = super::random_u64();
        assert_ne!(a, b);
    }

    use super::*;
    use crate::parser::parse;
    use std::io::Cursor;

    #[test]
    fn interactive_input_reader_is_bounded_and_typed() {
        let mut valid = Cursor::new(b"  18446744073709551615\n".to_vec());
        assert_eq!(read_bounded_input(&mut valid), Ok(u64::MAX));

        let mut oversized = Cursor::new(vec![b'7'; MAX_INTERACTIVE_INPUT_BYTES + 1]);
        assert_eq!(
            read_bounded_input(&mut oversized),
            Err(InteractiveInputError::TooLong {
                limit: MAX_INTERACTIVE_INPUT_BYTES
            })
        );

        let mut invalid = Cursor::new(b"not-a-word\n".to_vec());
        assert_eq!(
            read_bounded_input(&mut invalid),
            Err(InteractiveInputError::InvalidValue)
        );

        let mut eof = Cursor::new(Vec::<u8>::new());
        assert_eq!(
            read_bounded_input(&mut eof),
            Err(InteractiveInputError::EndOfFile)
        );
    }

    #[test]
    fn public_executor_runs_quotations_through_bytecode() {
        let program = parse("[ 40 2 ADD OUTPUT ] EXEC").unwrap();
        let mut executor = Executor::new();
        let result = executor.run_epoch(&program, &Memory::new());

        assert_eq!(result.status, EpochStatus::Finished);
        assert!(matches!(
            result.output.as_slice(),
            [OutputItem::Val(value)] if value.val == 42
        ));
    }

    #[test]
    fn recursive_procedure_is_bounded_by_bytecode_gas_without_host_recursion() {
        let program = parse("PROCEDURE recur { recur } recur").unwrap();
        let mut executor = Executor::with_config(ExecutorConfig {
            max_instructions: 31,
            immediate_output: false,
            ..ExecutorConfig::default()
        });
        let result = executor.run_epoch(&program, &Memory::new());

        match result.status {
            EpochStatus::Error(message) => {
                assert!(
                    message.contains("instruction limit 31 exhausted"),
                    "{message}"
                );
                assert_eq!(result.instructions_executed, 31);
            }
            status => panic!("expected bytecode gas failure, got {status:?}"),
        }
    }

    #[test]
    fn single_epoch_sleep_is_denied_without_a_bytecode_capability() {
        let program = parse("1 SLEEP").unwrap();
        let mut executor = Executor::new();
        let result = executor.run_epoch(&program, &Memory::new());

        match result.status {
            EpochStatus::Error(message) => {
                assert!(message.contains("sleep capability denied"), "{message}");
            }
            status => panic!("expected capability denial, got {status:?}"),
        }
    }

    #[test]
    fn definite_semantic_errors_are_rejected_before_dispatch() {
        let program = parse("ADD").unwrap();
        let mut executor = Executor::new();
        let result = executor.run_epoch(&program, &Memory::new());

        match result.status {
            EpochStatus::Error(message) => {
                assert!(message.contains("semantic analysis failed"), "{message}");
            }
            status => panic!("expected semantic rejection, got {status:?}"),
        }
        assert_eq!(result.instructions_executed, 0);
    }

    #[test]
    fn execute_rejects_ambiguous_mid_epoch_state() {
        let program = parse("1").unwrap();
        let mut state = VmState::new(Memory::new());
        state.stack.push(Value::new(9));
        let error = Executor::new().execute(&mut state, &program).unwrap_err();
        assert!(error.contains("pristine VmState"));
        assert_eq!(state.stack[0].val, 9);
    }

    #[test]
    fn test_simple_arithmetic() {
        let program = parse("10 20 ADD").unwrap();
        let mut executor = Executor::new();
        let result = executor.run_epoch(&program, &Memory::new());

        assert_eq!(result.status, EpochStatus::Finished);
    }

    #[test]
    fn test_oracle_prophecy() {
        let program = parse("0 ORACLE 0 PROPHECY").unwrap();
        let mut anamnesis = Memory::new();
        anamnesis.write(0, Value::new(42));

        let mut executor = Executor::new();
        let result = executor.run_epoch(&program, &anamnesis);

        assert_eq!(result.status, EpochStatus::Finished);
        assert_eq!(result.present.read(0).val, 42);
    }

    #[test]
    fn test_halt() {
        let program = parse("10 HALT 20").unwrap();
        let mut executor = Executor::new();
        let result = executor.run_epoch(&program, &Memory::new());

        assert_eq!(result.status, EpochStatus::Finished);
    }

    #[test]
    fn test_paradox() {
        let program = parse("PARADOX").unwrap();
        let mut executor = Executor::new();
        let result = executor.run_epoch(&program, &Memory::new());

        assert_eq!(result.status, EpochStatus::Paradox);
    }

    #[test]
    fn test_division_by_zero() {
        let program = parse("10 0 DIV").unwrap();
        let mut executor = Executor::new();
        let result = executor.run_epoch(&program, &Memory::new());

        // Should not error; returns 0
        assert_eq!(result.status, EpochStatus::Finished);
    }

    #[test]
    fn test_math_opcodes() {
        // Test MIN
        let program = parse("10 20 MIN").unwrap();
        let mut executor = Executor::new();
        let result = executor.run_epoch(&program, &Memory::new());
        assert_eq!(result.status, EpochStatus::Finished);
        assert_eq!(result.present.read(0).val, 0); // Stack result

        // Test MAX
        let program = parse("10 20 MAX").unwrap();
        let mut executor = Executor::new();
        let result = executor.run_epoch(&program, &Memory::new());
        assert_eq!(result.status, EpochStatus::Finished);

        // Test ABS (positive)
        let program = parse("42 ABS OUTPUT").unwrap();
        let mut executor = Executor::new();
        let result = executor.run_epoch(&program, &Memory::new());
        assert_eq!(result.status, EpochStatus::Finished);
        assert_eq!(result.output.len(), 1);

        // Test SIGN
        let program = parse("0 SIGN OUTPUT").unwrap();
        let mut executor = Executor::new();
        let result = executor.run_epoch(&program, &Memory::new());
        assert_eq!(result.status, EpochStatus::Finished);
    }

    #[test]
    fn test_signed_comparison() {
        // Testing signed comparisons with two's complement
        // -1 (u64::MAX) should be less than 0 in signed comparison
        let program = parse("18446744073709551615 0 SLT OUTPUT").unwrap(); // u64::MAX = -1 signed
        let mut executor = Executor::new();
        let result = executor.run_epoch(&program, &Memory::new());
        assert_eq!(result.status, EpochStatus::Finished);
        // -1 < 0 in signed => 1
        if let Some(crate::core::OutputItem::Val(v)) = result.output.first() {
            assert_eq!(v.val, 1);
        }

        // But in unsigned, u64::MAX > 0
        let program = parse("18446744073709551615 0 LT OUTPUT").unwrap();
        let mut executor = Executor::new();
        let result = executor.run_epoch(&program, &Memory::new());
        assert_eq!(result.status, EpochStatus::Finished);
        // u64::MAX < 0 in unsigned => 0
        if let Some(crate::core::OutputItem::Val(v)) = result.output.first() {
            assert_eq!(v.val, 0);
        }
    }

    #[test]
    fn test_vector_operations() {
        // Test VEC_NEW, VEC_PUSH, VEC_LEN
        let program = parse("VEC_NEW 42 VEC_PUSH 99 VEC_PUSH VEC_LEN OUTPUT").unwrap();
        let mut executor = Executor::new();
        executor.config.immediate_output = false;
        let result = executor.run_epoch(&program, &Memory::new());
        assert_eq!(result.status, EpochStatus::Finished);
        if let Some(crate::core::OutputItem::Val(v)) = result.output.first() {
            assert_eq!(v.val, 2); // Vector has 2 elements
        }
    }

    #[test]
    fn test_vector_get_pop() {
        // Test VEC_GET and VEC_POP
        let program = parse("VEC_NEW 100 VEC_PUSH 200 VEC_PUSH 0 VEC_GET OUTPUT").unwrap();
        let mut executor = Executor::new();
        executor.config.immediate_output = false;
        let result = executor.run_epoch(&program, &Memory::new());
        assert_eq!(result.status, EpochStatus::Finished);
        // First element is 100
        if let Some(crate::core::OutputItem::Val(v)) = result.output.first() {
            assert_eq!(v.val, 100);
        }
    }

    #[test]
    fn test_hash_operations() {
        // Test HASH_NEW, HASH_PUT, HASH_GET
        let program = parse("HASH_NEW 123 456 HASH_PUT 123 HASH_GET OUTPUT").unwrap();
        let mut executor = Executor::new();
        executor.config.immediate_output = false;
        let result = executor.run_epoch(&program, &Memory::new());
        assert_eq!(result.status, EpochStatus::Finished);
        // Found flag should be 1
        if let Some(crate::core::OutputItem::Val(v)) = result.output.first() {
            assert_eq!(v.val, 1);
        }
    }

    #[test]
    fn test_hash_has() {
        // Test HASH_HAS for existing and non-existing keys
        let program =
            parse("HASH_NEW 42 999 HASH_PUT 42 HASH_HAS OUTPUT 100 HASH_HAS OUTPUT").unwrap();
        let mut executor = Executor::new();
        executor.config.immediate_output = false;
        let result = executor.run_epoch(&program, &Memory::new());
        assert_eq!(result.status, EpochStatus::Finished);
        // First output: 42 exists => 1
        if let Some(crate::core::OutputItem::Val(v)) = result.output.first() {
            assert_eq!(v.val, 1);
        }
        // Second output: 100 doesn't exist => 0
        if let Some(crate::core::OutputItem::Val(v)) = result.output.get(1) {
            assert_eq!(v.val, 0);
        }
    }

    #[test]
    fn test_set_operations() {
        // Test SET_NEW, SET_ADD, SET_HAS
        let program =
            parse("SET_NEW 42 SET_ADD 99 SET_ADD 42 SET_HAS OUTPUT 100 SET_HAS OUTPUT").unwrap();
        let mut executor = Executor::new();
        executor.config.immediate_output = false;
        let result = executor.run_epoch(&program, &Memory::new());
        assert_eq!(result.status, EpochStatus::Finished);
        // 42 exists => 1
        if let Some(crate::core::OutputItem::Val(v)) = result.output.first() {
            assert_eq!(v.val, 1);
        }
        // 100 doesn't exist => 0
        if let Some(crate::core::OutputItem::Val(v)) = result.output.get(1) {
            assert_eq!(v.val, 0);
        }
    }

    #[test]
    fn test_set_len() {
        // Test SET_LEN
        let program = parse("SET_NEW 1 SET_ADD 2 SET_ADD 1 SET_ADD SET_LEN OUTPUT").unwrap();
        let mut executor = Executor::new();
        executor.config.immediate_output = false;
        let result = executor.run_epoch(&program, &Memory::new());
        assert_eq!(result.status, EpochStatus::Finished);
        // Set has 2 unique elements (1, 2) - adding 1 again doesn't change size
        if let Some(crate::core::OutputItem::Val(v)) = result.output.first() {
            assert_eq!(v.val, 2);
        }
    }

    #[test]
    fn temporal_scope_addresses_are_relative_and_values_are_typed() {
        let program = parse("TEMPORAL 2 1 BITS 2 { 7 0 PROPHECY }").unwrap();
        let mut executor = Executor::new();
        executor.config.immediate_output = false;
        let result = executor.run_epoch(&program, &Memory::with_size(4));
        assert_eq!(result.status, EpochStatus::Finished);
        assert_eq!(result.present.read(0).val, 0);
        assert_eq!(result.present.read(2).val, 3);
    }

    #[test]
    fn strict_temporal_scope_rejects_local_out_of_bounds_access() {
        let program = parse("TEMPORAL 2 1 BITS 2 { 1 ORACLE POP }").unwrap();
        let mut executor = Executor::with_config(ExecutorConfig {
            immediate_output: false,
            error_config: ErrorConfig::strict(),
            ..ExecutorConfig::default()
        });
        let result = executor.run_epoch(&program, &Memory::with_size(4));
        match result.status {
            EpochStatus::Error(message) => assert!(message.contains("outside"), "{message}"),
            status => panic!("expected scope bounds error, got {:?}", status),
        }
    }
}
