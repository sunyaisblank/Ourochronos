//! Deterministic orbit policies over the iterative bytecode VM.
//!
//! Standard point-fixed-state search, diagnostic trajectory analysis, and
//! deterministic Deutsch stationary-cycle semantics all execute the same
//! validated bytecode epoch transition. Global, all-fixed, recurrent, and
//! action-guided policies remain distinct APIs and must not be silently
//! approximated by this driver. Live environment capture, when explicitly
//! enabled, is confined to the first candidate and converted to frozen tapes.
//! Candidate file mutations remain data; `run_with_adapter` and its diagnostic
//! counterpart expose the only authorized post-selection host commit boundary.

use crate::bytecode::BytecodeProgram;
use crate::bytecode_vm::{
    BytecodeVm, BytecodeVmConfig, BytecodeVmStatus, FrozenEndpointTape, FrozenFileSnapshot,
    FrozenProcessResult, PreparedBytecode,
};
use crate::core::{Address, Memory, PagedMemory, Value};
use crate::temporal::timeloop::{ConvergenceStatus, Direction, ParadoxDiagnosis};
use crate::temporal::transaction::{
    CommitLog, CommitOutcome, CommitToken, EffectCommitAdapter, ObservationTranscript,
    TemporalTransaction, TransactionLimits,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

/// Hard aggregate bound for collision journals, persistent orbit snapshots,
/// and Deutsch per-epoch payloads retained by one temporal search.
pub const MAX_RETAINED_ORBIT_BYTES: usize = 256 * 1024 * 1024;

/// Deterministic standard-orbit limits for bytecode execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BytecodeTimeLoopConfig {
    /// Maximum epoch transformations before returning UNKNOWN/timeout.
    pub max_epochs: usize,
    /// Number of temporal memory cells.
    pub memory_cells: usize,
    /// Optional deterministic nonzero seed used for the first 16 cells.
    pub seed: u64,
    /// Explicit sparse initial anamnesis. This is used by packaged,
    /// replay-verified witnesses; addresses must be unique and in range.
    pub initial_state: Vec<(Address, u64)>,
    /// Per-epoch bytecode runtime configuration.
    pub vm: BytecodeVmConfig,
}

impl Default for BytecodeTimeLoopConfig {
    fn default() -> Self {
        Self {
            max_epochs: crate::temporal::timeloop::DEFAULT_MAX_EPOCHS,
            memory_cells: crate::core::MEMORY_SIZE,
            seed: 0,
            initial_state: Vec::new(),
            vm: BytecodeVmConfig::default(),
        }
    }
}

/// Iterative point-fixed-state driver backed only by validated bytecode.
#[derive(Debug, Clone)]
pub struct BytecodeTimeLoop {
    /// Runtime and orbit limits.
    pub config: BytecodeTimeLoopConfig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FrozenVmInputTapes {
    input: Vec<u64>,
    clock: Vec<u64>,
    random: Vec<u64>,
    files: Vec<FrozenFileSnapshot>,
    endpoints: Vec<FrozenEndpointTape>,
    processes: Vec<FrozenProcessResult>,
}

impl FrozenVmInputTapes {
    fn from_config(config: &BytecodeVmConfig) -> Self {
        Self {
            input: config.input.clone(),
            clock: config.clock_input.clone(),
            random: config.random_input.clone(),
            files: config.file_snapshots.clone(),
            endpoints: config.endpoint_tapes.clone(),
            processes: config.process_results.clone(),
        }
    }
}

impl BytecodeTimeLoop {
    /// Construct a standard bytecode orbit driver.
    pub fn new(config: BytecodeTimeLoopConfig) -> Result<Self, String> {
        if config.max_epochs == 0 {
            return Err("max_epochs must be greater than zero".to_string());
        }
        if config.memory_cells == 0 {
            return Err("memory_cells must be greater than zero".to_string());
        }
        if config.memory_cells > crate::core::MAX_DENSE_MEMORY_CELLS {
            return Err(format!(
                "memory_cells {} exceeds dense result limit {}",
                config.memory_cells,
                crate::core::MAX_DENSE_MEMORY_CELLS
            ));
        }
        // Enforce the complete paged-memory allocation contract before a
        // runtime can be constructed. The temporary zero snapshot shares one
        // physical page across its bounded page table and is dropped here.
        PagedMemory::with_size(config.memory_cells).map_err(|error| error.to_string())?;
        if config.vm.max_instructions == 0 {
            return Err("max_instructions must be greater than zero".to_string());
        }
        if config.seed != 0 && !config.initial_state.is_empty() {
            return Err("seed and explicit initial_state are mutually exclusive".to_string());
        }
        let mut previous = None;
        for (address, value) in &config.initial_state {
            if *address >= config.memory_cells as u64 {
                return Err(format!(
                    "initial state address {address} is outside {} cells",
                    config.memory_cells
                ));
            }
            if previous.is_some_and(|prior| prior >= *address) {
                return Err("initial_state addresses must be strictly increasing".to_string());
            }
            if *value == 0 {
                return Err("initial_state must use canonical nonzero cells".to_string());
            }
            previous = Some(*address);
        }
        Ok(Self { config })
    }

    /// Iterate `F(anamnesis) = present` until a point fixed state, explicit
    /// paradox, verified repeated orbit state, runtime error, or epoch limit.
    pub fn run(&self, program: &BytecodeProgram) -> ConvergenceStatus {
        self.run_standard(program, None)
    }

    /// Resolve a standard point-fixed timeline and, only after exact selection,
    /// ledger and apply its staged intents through an explicitly authorized
    /// adapter. Candidate epochs never receive or invoke the adapter. `token`
    /// is the caller's durable idempotency identity for this selected batch;
    /// retaining `log` makes repeated calls with the same exact token and batch
    /// idempotent without invoking the adapter again.
    pub fn run_with_adapter(
        &self,
        program: &BytecodeProgram,
        token: CommitToken,
        log: &mut CommitLog,
        adapter: &mut dyn EffectCommitAdapter,
    ) -> ConvergenceStatus {
        self.run_standard(program, Some((token, log, adapter)))
    }

    fn run_standard(
        &self,
        program: &BytecodeProgram,
        mut commit: Option<(CommitToken, &mut CommitLog, &mut dyn EffectCommitAdapter)>,
    ) -> ConvergenceStatus {
        let prepared = match PreparedBytecode::new(program.clone()) {
            Ok(prepared) => prepared,
            Err(error) => {
                return ConvergenceStatus::Error {
                    message: format!("invalid bytecode: {error}"),
                    epoch: 0,
                };
            }
        };
        let mut anamnesis = match self.initial_anamnesis() {
            Ok(memory) => memory,
            Err(message) => return ConvergenceStatus::Error { message, epoch: 0 },
        };
        // Provenance is retained by the VM and transaction, but the temporal
        // state S and orbit relation are numeric. Hash collisions are still
        // verified against the complete sparse numeric state.
        let mut seen = HashMap::<u64, Vec<(Vec<(Address, u64)>, usize)>>::new();
        let mut trajectory = Vec::<PagedMemory>::new();
        let mut retained_orbit_bytes = 0usize;
        let mut vm_config = self.config.vm.clone();
        let mut frozen_inputs = FrozenVmInputTapes::from_config(&vm_config);

        for epoch in 0..self.config.max_epochs {
            let sparse = anamnesis.numeric_sparse_state();
            let hash = numeric_state_hash(anamnesis.len(), &sparse);
            if let Some(previous) = seen.get(&hash).and_then(|candidates| {
                candidates
                    .iter()
                    .find_map(|(candidate, prior)| (candidate == &sparse).then_some(*prior))
            }) {
                let cycle = &trajectory[previous..epoch];
                let period = epoch - previous;
                let oscillating_cells = oscillating_cells(cycle, self.config.memory_cells);
                let cycle = cycle
                    .iter()
                    .map(PagedMemory::numeric_sparse_state)
                    .collect();
                return ConvergenceStatus::Oscillation {
                    period,
                    oscillating_cells,
                    diagnosis: ParadoxDiagnosis::Oscillation { cycle },
                };
            }
            retained_orbit_bytes =
                match retain_orbit_state(retained_orbit_bytes, &anamnesis, &sparse) {
                    Ok(bytes) => bytes,
                    Err(message) => return ConvergenceStatus::Error { message, epoch },
                };
            seen.entry(hash).or_default().push((sparse, epoch));
            trajectory.push(anamnesis.clone());

            let result =
                match self.run_epoch(&prepared, &anamnesis, &mut vm_config, &mut frozen_inputs) {
                    Ok(result) => result,
                    Err(message) => {
                        return ConvergenceStatus::Error {
                            message,
                            epoch: epoch + 1,
                        };
                    }
                };
            match result.status {
                BytecodeVmStatus::Finished | BytecodeVmStatus::Halted => {
                    if result.present.numeric_values_equal(&anamnesis) {
                        let output = match self.commit_selected_output(
                            &result.present,
                            &result.output,
                            &result.effects,
                            &result.observation_transcript(),
                            &frozen_inputs.input,
                            commit.take(),
                        ) {
                            Ok(output) => output,
                            Err(message) => {
                                return ConvergenceStatus::Error {
                                    message,
                                    epoch: epoch + 1,
                                };
                            }
                        };
                        return ConvergenceStatus::Consistent {
                            memory: dense_memory(&result.present),
                            output,
                            epochs: epoch + 1,
                        };
                    }
                    anamnesis = result.present;
                }
                BytecodeVmStatus::Paradox => {
                    return ConvergenceStatus::Paradox {
                        message: "Explicit PARADOX instruction".to_string(),
                        epoch: epoch + 1,
                    };
                }
            }
        }
        ConvergenceStatus::Timeout {
            max_epochs: self.config.max_epochs,
        }
    }

    /// Run uncached point iteration without revisit/cycle diagnosis.
    ///
    /// This preserves the compatibility `ExecutionMode::Pure` contract: its
    /// orbit is not bounded by `max_epochs`, but by the historical
    /// 1,000,001-candidate implementation safety ceiling. Each candidate still
    /// executes the same prepared bytecode and frozen observation tapes as the
    /// other policies; only memoization and trajectory classification differ.
    pub fn run_pure(&self, program: &BytecodeProgram) -> ConvergenceStatus {
        if is_trivially_consistent(program) {
            return self.run_trivial(program, None);
        }
        let prepared = match PreparedBytecode::new(program.clone()) {
            Ok(prepared) => prepared,
            Err(error) => {
                return ConvergenceStatus::Error {
                    message: format!("invalid bytecode: {error}"),
                    epoch: 0,
                };
            }
        };
        let mut anamnesis = match self.initial_anamnesis() {
            Ok(memory) => memory,
            Err(message) => return ConvergenceStatus::Error { message, epoch: 0 },
        };
        let mut vm_config = self.config.vm.clone();
        let mut frozen_inputs = FrozenVmInputTapes::from_config(&vm_config);

        for epoch in 0..=1_000_000usize {
            let result =
                match self.run_epoch(&prepared, &anamnesis, &mut vm_config, &mut frozen_inputs) {
                    Ok(result) => result,
                    Err(message) => {
                        return ConvergenceStatus::Error {
                            message,
                            epoch: epoch + 1,
                        };
                    }
                };
            match result.status {
                BytecodeVmStatus::Finished | BytecodeVmStatus::Halted => {
                    if result.present.numeric_values_equal(&anamnesis) {
                        let output = match self.commit_selected_output(
                            &result.present,
                            &result.output,
                            &result.effects,
                            &result.observation_transcript(),
                            &frozen_inputs.input,
                            None,
                        ) {
                            Ok(output) => output,
                            Err(message) => {
                                return ConvergenceStatus::Error {
                                    message,
                                    epoch: epoch + 1,
                                };
                            }
                        };
                        return ConvergenceStatus::Consistent {
                            memory: dense_memory(&result.present),
                            output,
                            epochs: epoch + 1,
                        };
                    }
                    anamnesis = result.present;
                }
                BytecodeVmStatus::Paradox => {
                    return ConvergenceStatus::Paradox {
                        message: "Explicit PARADOX instruction".to_string(),
                        epoch: epoch + 1,
                    };
                }
            }
        }

        ConvergenceStatus::Timeout {
            max_epochs: 1_000_001,
        }
    }

    /// Run the standard search while retaining the complete trajectory for
    /// verified oscillation diagnosis and finite-prefix divergence evidence.
    pub fn run_diagnostic(&self, program: &BytecodeProgram) -> ConvergenceStatus {
        self.run_diagnostic_inner(program, None)
    }

    /// Diagnostic point-fixed resolution with exactly-once selected effect
    /// dispatch. Divergence, oscillation, paradox, and timeout never invoke the
    /// adapter.
    pub fn run_diagnostic_with_adapter(
        &self,
        program: &BytecodeProgram,
        token: CommitToken,
        log: &mut CommitLog,
        adapter: &mut dyn EffectCommitAdapter,
    ) -> ConvergenceStatus {
        self.run_diagnostic_inner(program, Some((token, log, adapter)))
    }

    fn run_diagnostic_inner(
        &self,
        program: &BytecodeProgram,
        mut commit: Option<(CommitToken, &mut CommitLog, &mut dyn EffectCommitAdapter)>,
    ) -> ConvergenceStatus {
        if is_trivially_consistent(program) {
            return self.run_trivial(program, commit);
        }
        let prepared = match PreparedBytecode::new(program.clone()) {
            Ok(prepared) => prepared,
            Err(error) => {
                return ConvergenceStatus::Error {
                    message: format!("invalid bytecode: {error}"),
                    epoch: 0,
                };
            }
        };
        let mut anamnesis = match self.initial_anamnesis() {
            Ok(memory) => memory,
            Err(message) => return ConvergenceStatus::Error { message, epoch: 0 },
        };
        let mut seen = HashMap::<u64, Vec<(Vec<(Address, u64)>, usize)>>::new();
        let mut trajectory = Vec::<PagedMemory>::new();
        let mut retained_orbit_bytes = 0usize;
        let mut vm_config = self.config.vm.clone();
        let mut frozen_inputs = FrozenVmInputTapes::from_config(&vm_config);

        for epoch in 0..self.config.max_epochs {
            let sparse = anamnesis.numeric_sparse_state();
            let hash = numeric_state_hash(anamnesis.len(), &sparse);
            if let Some(previous) = seen.get(&hash).and_then(|candidates| {
                candidates
                    .iter()
                    .find_map(|(candidate, prior)| (candidate == &sparse).then_some(*prior))
            }) {
                let cycle = &trajectory[previous..epoch];
                return ConvergenceStatus::Oscillation {
                    period: epoch - previous,
                    oscillating_cells: oscillating_cells(cycle, self.config.memory_cells),
                    diagnosis: oscillation_diagnosis(cycle),
                };
            }
            retained_orbit_bytes =
                match retain_orbit_state(retained_orbit_bytes, &anamnesis, &sparse) {
                    Ok(bytes) => bytes,
                    Err(message) => return ConvergenceStatus::Error { message, epoch },
                };
            seen.entry(hash).or_default().push((sparse, epoch));
            trajectory.push(anamnesis.clone());

            // This deliberately matches the established diagnostic policy:
            // only prefixes containing at least twelve visited states are
            // eligible for monotone-trend evidence.
            if epoch > 10 {
                if let Some((diverging_cells, direction)) = detect_divergence(&trajectory) {
                    return ConvergenceStatus::Divergence {
                        diverging_cells,
                        direction,
                    };
                }
            }

            let result =
                match self.run_epoch(&prepared, &anamnesis, &mut vm_config, &mut frozen_inputs) {
                    Ok(result) => result,
                    Err(message) => {
                        return ConvergenceStatus::Error {
                            message,
                            epoch: epoch + 1,
                        };
                    }
                };
            match result.status {
                BytecodeVmStatus::Finished | BytecodeVmStatus::Halted => {
                    if result.present.numeric_values_equal(&anamnesis) {
                        let output = match self.commit_selected_output(
                            &result.present,
                            &result.output,
                            &result.effects,
                            &result.observation_transcript(),
                            &frozen_inputs.input,
                            commit.take(),
                        ) {
                            Ok(output) => output,
                            Err(message) => {
                                return ConvergenceStatus::Error {
                                    message,
                                    epoch: epoch + 1,
                                };
                            }
                        };
                        return ConvergenceStatus::Consistent {
                            memory: dense_memory(&result.present),
                            output,
                            epochs: epoch + 1,
                        };
                    }
                    anamnesis = result.present;
                }
                BytecodeVmStatus::Paradox => {
                    return ConvergenceStatus::Paradox {
                        message: "Explicit PARADOX instruction".to_string(),
                        epoch: epoch + 1,
                    };
                }
            }
        }

        ConvergenceStatus::Timeout {
            max_epochs: self.config.max_epochs,
        }
    }

    /// Find the recurrent class reachable from the configured seed and return
    /// its uniform deterministic Deutsch stationary ensemble. A point fixed
    /// state is represented as a period-one ensemble, matching the established
    /// deterministic Deutsch policy.
    pub fn run_deutsch(&self, program: &BytecodeProgram) -> ConvergenceStatus {
        self.run_deutsch_inner(program, None)
    }

    /// Deutsch resolution with exactly-once dispatch for a uniquely selected
    /// trivial or period-one timeline. Effects on longer recurrent cycles are
    /// still rejected because no single chronology has been selected.
    pub fn run_deutsch_with_adapter(
        &self,
        program: &BytecodeProgram,
        token: CommitToken,
        log: &mut CommitLog,
        adapter: &mut dyn EffectCommitAdapter,
    ) -> ConvergenceStatus {
        self.run_deutsch_inner(program, Some((token, log, adapter)))
    }

    fn run_deutsch_inner(
        &self,
        program: &BytecodeProgram,
        mut commit: Option<(CommitToken, &mut CommitLog, &mut dyn EffectCommitAdapter)>,
    ) -> ConvergenceStatus {
        if is_trivially_consistent(program) {
            return self.run_trivial(program, commit);
        }
        let prepared = match PreparedBytecode::new(program.clone()) {
            Ok(prepared) => prepared,
            Err(error) => {
                return ConvergenceStatus::Error {
                    message: format!("invalid bytecode: {error}"),
                    epoch: 0,
                };
            }
        };
        let mut anamnesis = match self.initial_anamnesis() {
            Ok(memory) => memory,
            Err(message) => return ConvergenceStatus::Error { message, epoch: 0 },
        };
        let mut seen = HashMap::<u64, Vec<(Vec<(Address, u64)>, usize)>>::new();
        let mut trajectory = Vec::<PagedMemory>::new();
        let mut epoch_outputs = Vec::<Vec<crate::core::OutputItem>>::new();
        let mut epoch_effects = Vec::<Vec<crate::temporal::transaction::EffectIntent>>::new();
        let mut vm_config = self.config.vm.clone();
        let configured_output_bytes = vm_config.max_output_bytes;
        let configured_effect_bytes = vm_config.max_effect_bytes;
        let mut retained_orbit_bytes = 0usize;
        let mut frozen_inputs = FrozenVmInputTapes::from_config(&vm_config);

        for epoch in 0..self.config.max_epochs {
            let sparse = anamnesis.numeric_sparse_state();
            let hash = numeric_state_hash(anamnesis.len(), &sparse);
            if let Some(first_epoch) = seen.get(&hash).and_then(|candidates| {
                candidates
                    .iter()
                    .find_map(|(candidate, prior)| (candidate == &sparse).then_some(*prior))
            }) {
                if epoch_effects[first_epoch..epoch]
                    .iter()
                    .any(|effects| !effects.is_empty())
                {
                    return ConvergenceStatus::Error {
                        message: "Deutsch recurrent cycles with staged effects require an explicit batch-selection policy; no host effects were committed".to_string(),
                        epoch,
                    };
                }
                let cycle = trajectory[first_epoch..epoch]
                    .iter()
                    .map(dense_memory)
                    .collect::<Vec<_>>();
                let outputs = epoch_outputs[first_epoch..epoch].to_vec();
                let period = cycle.len();
                let unanimous_output = outputs.first().and_then(|first| {
                    outputs
                        .iter()
                        .all(|candidate| observable_outputs_equal(first, candidate))
                        .then(|| first.clone())
                });
                return ConvergenceStatus::DeutschConsistent {
                    cycle,
                    outputs,
                    period,
                    transient_epochs: first_epoch,
                    unanimous_output,
                };
            }
            retained_orbit_bytes =
                match retain_orbit_state(retained_orbit_bytes, &anamnesis, &sparse) {
                    Ok(bytes) => bytes,
                    Err(message) => return ConvergenceStatus::Error { message, epoch },
                };
            seen.entry(hash).or_default().push((sparse, epoch));
            trajectory.push(anamnesis.clone());

            let remaining = MAX_RETAINED_ORBIT_BYTES - retained_orbit_bytes;
            vm_config.max_output_bytes = configured_output_bytes.min(remaining);
            vm_config.max_effect_bytes = configured_effect_bytes.min(remaining);

            let result =
                match self.run_epoch(&prepared, &anamnesis, &mut vm_config, &mut frozen_inputs) {
                    Ok(result) => result,
                    Err(message) => {
                        return ConvergenceStatus::Error {
                            message,
                            epoch: epoch + 1,
                        };
                    }
                };
            match result.status {
                BytecodeVmStatus::Finished | BytecodeVmStatus::Halted => {
                    if result.present.numeric_values_equal(&anamnesis) {
                        let output = match self.commit_selected_output(
                            &result.present,
                            &result.output,
                            &result.effects,
                            &result.observation_transcript(),
                            &frozen_inputs.input,
                            commit.take(),
                        ) {
                            Ok(output) => output,
                            Err(message) => {
                                return ConvergenceStatus::Error {
                                    message,
                                    epoch: epoch + 1,
                                };
                            }
                        };
                        return ConvergenceStatus::DeutschConsistent {
                            cycle: vec![dense_memory(&result.present)],
                            outputs: vec![output.clone()],
                            period: 1,
                            transient_epochs: epoch,
                            unanimous_output: Some(output),
                        };
                    }
                    let payload_charge = result
                        .output
                        .iter()
                        .fold(0usize, |total, item| {
                            total.saturating_add(item.retained_size_charge())
                        })
                        .saturating_add(result.effects.iter().fold(0usize, |total, effect| {
                            total.saturating_add(64).saturating_add(effect.byte_len())
                        }));
                    retained_orbit_bytes = match retained_orbit_bytes.checked_add(payload_charge) {
                        Some(bytes) if bytes <= MAX_RETAINED_ORBIT_BYTES => bytes,
                        _ => {
                            return ConvergenceStatus::Error {
                                message: format!(
                                    "Deutsch retained orbit payload exceeds hard limit {MAX_RETAINED_ORBIT_BYTES} bytes"
                                ),
                                epoch: epoch + 1,
                            }
                        }
                    };
                    epoch_outputs.push(result.output);
                    epoch_effects.push(result.effects);
                    anamnesis = result.present;
                }
                BytecodeVmStatus::Paradox => {
                    return ConvergenceStatus::Paradox {
                        message: "Explicit PARADOX instruction".to_string(),
                        epoch: epoch + 1,
                    };
                }
            }
        }

        ConvergenceStatus::Timeout {
            max_epochs: self.config.max_epochs,
        }
    }

    fn initial_anamnesis(&self) -> Result<PagedMemory, String> {
        let mut memory =
            PagedMemory::with_size(self.config.memory_cells).map_err(|error| error.to_string())?;
        if !self.config.initial_state.is_empty() {
            for (address, value) in &self.config.initial_state {
                memory
                    .write(*address, Value::new(*value))
                    .map_err(|error| error.to_string())?;
            }
        } else if self.config.seed != 0 {
            for index in 0..16.min(self.config.memory_cells) {
                memory
                    .write(
                        index as Address,
                        Value::new(self.config.seed.wrapping_mul(index as u64 + 1)),
                    )
                    .map_err(|error| error.to_string())?;
            }
        }
        Ok(memory)
    }

    fn run_epoch(
        &self,
        program: &PreparedBytecode,
        anamnesis: &PagedMemory,
        vm_config: &mut BytecodeVmConfig,
        frozen_inputs: &mut FrozenVmInputTapes,
    ) -> Result<crate::bytecode_vm::BytecodeExecution, String> {
        // Resource cursors are per-candidate. Reinstall the same immutable
        // endpoint and process observations before every epoch.
        vm_config.endpoint_tapes = frozen_inputs.endpoints.clone();
        vm_config.process_results = frozen_inputs.processes.clone();
        let result = BytecodeVm::with_config(vm_config.clone())
            .run_prepared(program, anamnesis)
            .map_err(|error| error.to_string())?;
        if vm_config.allow_interactive_input {
            if result.inputs_consumed.len() > frozen_inputs.input.len() {
                frozen_inputs.input = result.inputs_consumed.clone();
            }
            vm_config.input = frozen_inputs.input.clone();
            vm_config.allow_interactive_input = false;
        }
        if vm_config.allow_live_system_inputs {
            if result.clock_inputs_consumed.len() > frozen_inputs.clock.len() {
                frozen_inputs.clock = result.clock_inputs_consumed.clone();
            }
            if result.random_inputs_consumed.len() > frozen_inputs.random.len() {
                frozen_inputs.random = result.random_inputs_consumed.clone();
            }
            vm_config.clock_input = frozen_inputs.clock.clone();
            vm_config.random_input = frozen_inputs.random.clone();
            vm_config.allow_live_system_inputs = false;
        }
        if vm_config.allow_live_file_reads {
            for snapshot in &result.file_snapshots_consumed {
                if let Some(existing) = frozen_inputs
                    .files
                    .iter()
                    .find(|frozen| frozen.path == snapshot.path)
                {
                    if existing != snapshot {
                        return Err(format!(
                            "captured file snapshot for '{}' disagrees with frozen input",
                            snapshot.path
                        ));
                    }
                } else {
                    frozen_inputs.files.push(snapshot.clone());
                }
            }
            vm_config.file_snapshots = frozen_inputs.files.clone();
            vm_config.allow_live_file_reads = false;
        }
        Ok(result)
    }

    /// Preserve the established mode-independent shortcut for programs that
    /// never read anamnesis: one zero-seeded candidate is already the selected
    /// consistent timeline, even when a nonzero orbit seed was configured.
    fn run_trivial(
        &self,
        program: &BytecodeProgram,
        commit: Option<(CommitToken, &mut CommitLog, &mut dyn EffectCommitAdapter)>,
    ) -> ConvergenceStatus {
        let prepared = match PreparedBytecode::new(program.clone()) {
            Ok(prepared) => prepared,
            Err(error) => {
                return ConvergenceStatus::Error {
                    message: format!("invalid bytecode: {error}"),
                    epoch: 0,
                };
            }
        };
        let memory = match PagedMemory::with_size(self.config.memory_cells) {
            Ok(memory) => memory,
            Err(error) => {
                return ConvergenceStatus::Error {
                    message: error.to_string(),
                    epoch: 0,
                };
            }
        };
        let mut vm_config = self.config.vm.clone();
        let mut frozen_inputs = FrozenVmInputTapes::from_config(&vm_config);
        let result = match self.run_epoch(&prepared, &memory, &mut vm_config, &mut frozen_inputs) {
            Ok(result) => result,
            Err(message) => return ConvergenceStatus::Error { message, epoch: 1 },
        };
        match result.status {
            BytecodeVmStatus::Finished | BytecodeVmStatus::Halted => {
                let output = match self.commit_selected_output(
                    &result.present,
                    &result.output,
                    &result.effects,
                    &result.observation_transcript(),
                    &frozen_inputs.input,
                    commit,
                ) {
                    Ok(output) => output,
                    Err(message) => return ConvergenceStatus::Error { message, epoch: 1 },
                };
                ConvergenceStatus::Consistent {
                    memory: dense_memory(&result.present),
                    output,
                    epochs: 1,
                }
            }
            BytecodeVmStatus::Paradox => ConvergenceStatus::Paradox {
                message: "Explicit PARADOX in trivial program".to_string(),
                epoch: 1,
            },
        }
    }

    fn commit_selected_output(
        &self,
        present: &PagedMemory,
        output: &[crate::core::OutputItem],
        effects: &[crate::temporal::transaction::EffectIntent],
        observations: &ObservationTranscript,
        frozen_input: &[u64],
        commit: Option<(CommitToken, &mut CommitLog, &mut dyn EffectCommitAdapter)>,
    ) -> Result<Vec<crate::core::OutputItem>, String> {
        let mut transaction =
            TemporalTransaction::new(frozen_input.to_vec(), transaction_limits(&self.config.vm))
                .map_err(|error| error.to_string())?;
        let mut context = transaction
            .begin_candidate()
            .map_err(|error| error.to_string())?;
        for expected in &observations.input {
            let frozen = context.read_input().map_err(|error| error.to_string())?;
            if frozen != *expected {
                return Err("bytecode input consumption disagrees with frozen tape".to_string());
            }
        }
        for item in output {
            context
                .stage_output(item.clone())
                .map_err(|error| error.to_string())?;
        }
        for effect in effects {
            context
                .stage_effect(effect.clone())
                .map_err(|error| error.to_string())?;
        }
        context
            .set_observation_transcript(observations.clone())
            .map_err(|error| error.to_string())?;
        let state = present
            .iter_sparse()
            .map(|(address, value)| (address, value.clone()))
            .collect();
        let candidate = context.finish(state).map_err(|error| error.to_string())?;
        let timeline = transaction
            .stage_candidate(candidate)
            .map_err(|error| error.to_string())?;
        transaction
            .select(timeline)
            .map_err(|error| error.to_string())?;
        if let Some((token, log, adapter)) = commit {
            let outcome = transaction
                .commit_selected_with_adapter(token, log, adapter)
                .map_err(|error| error.to_string())?;
            committed_output(log, &outcome)
        } else {
            let mut log = CommitLog::default();
            let outcome = transaction
                .commit_selected(CommitToken(u128::from(timeline.0)), &mut log)
                .map_err(|error| error.to_string())?;
            committed_output(&log, &outcome)
        }
    }
}

pub(crate) fn transaction_limits(config: &BytecodeVmConfig) -> TransactionLimits {
    let instruction_observation_bytes = usize::try_from(config.max_instructions)
        .unwrap_or(usize::MAX)
        .saturating_mul(std::mem::size_of::<u64>());
    TransactionLimits {
        max_inputs: usize::try_from(config.max_instructions).unwrap_or(usize::MAX),
        max_outputs: config.max_output_items,
        max_output_bytes: config.max_output_bytes,
        max_effects: config.max_effects,
        max_effect_bytes: config.max_effect_bytes,
        max_observation_bytes: config
            .max_file_snapshot_bytes
            .saturating_add(config.max_endpoint_tape_bytes)
            .saturating_add(config.max_process_result_bytes)
            .saturating_add(config.max_effect_bytes)
            .saturating_add(instruction_observation_bytes),
        ..TransactionLimits::default()
    }
}

fn committed_output(
    log: &CommitLog,
    outcome: &CommitOutcome,
) -> Result<Vec<crate::core::OutputItem>, String> {
    let receipt = match outcome {
        CommitOutcome::Committed(receipt) | CommitOutcome::AlreadyCommitted(receipt) => receipt,
    };
    log.batches()
        .get(receipt.sequence)
        .map(|batch| batch.output.clone())
        .ok_or_else(|| "selected timeline produced no commit record".to_string())
}

fn oscillating_cells(cycle: &[PagedMemory], memory_cells: usize) -> Vec<Address> {
    if cycle.len() < 2 {
        return Vec::new();
    }
    (0..memory_cells)
        .filter_map(|index| {
            let first = cycle[0].get(index as Address)?.val;
            cycle[1..]
                .iter()
                .any(|state| {
                    state
                        .get(index as Address)
                        .is_some_and(|value| value.val != first)
                })
                .then_some(index as Address)
        })
        .collect()
}

/// Accelerator for numeric orbit buckets. The complete sparse numeric state is
/// always compared after a hash hit, so a collision cannot fabricate a cycle.
fn numeric_state_hash(width: usize, state: &[(Address, u64)]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    width.hash(&mut hasher);
    state.hash(&mut hasher);
    hasher.finish()
}

pub(crate) fn is_trivially_consistent(program: &BytecodeProgram) -> bool {
    let mut pending = VecDeque::from([program.main]);
    let mut visited = HashSet::new();
    let mut all_quotations_added = false;
    while let Some(range) = pending.pop_front() {
        if !visited.insert((range.start, range.end)) {
            continue;
        }
        let Some(instructions) = program
            .instructions
            .get(range.start as usize..range.end as usize)
        else {
            return false;
        };
        for instruction in instructions {
            match instruction {
                crate::bytecode::Instruction::Primitive(crate::ast::OpCode::Oracle) => {
                    return false;
                }
                crate::bytecode::Instruction::CallProcedure(id) => {
                    let Some(entry) = program.procedures.get(id.index()) else {
                        return false;
                    };
                    pending.push_back(entry.range);
                }
                crate::bytecode::Instruction::Primitive(
                    crate::ast::OpCode::Exec
                    | crate::ast::OpCode::Dip
                    | crate::ast::OpCode::Keep
                    | crate::ast::OpCode::Bi
                    | crate::ast::OpCode::Rec,
                ) if !all_quotations_added => {
                    // Quotation identities are runtime words. Once a dynamic
                    // quote combinator is reachable, every quotation is a
                    // conservative possible target.
                    pending.extend(program.quotations.iter().map(|entry| entry.range));
                    all_quotations_added = true;
                }
                _ => {}
            }
        }
    }
    true
}

fn numeric_differences(left: &PagedMemory, right: &PagedMemory) -> Vec<Address> {
    debug_assert_eq!(left.len(), right.len());
    let mut candidates = std::collections::HashSet::new();
    candidates.extend(
        left.iter_nonzero()
            .map(|(address, _)| address)
            .chain(right.iter_nonzero().map(|(address, _)| address)),
    );
    let mut differences = candidates
        .into_iter()
        .filter(|address| {
            left.get(*address).map(|value| value.val) != right.get(*address).map(|value| value.val)
        })
        .collect::<Vec<_>>();
    differences.sort_unstable();
    differences
}

fn oscillation_diagnosis(cycle: &[PagedMemory]) -> ParadoxDiagnosis {
    if cycle.len() == 2 {
        let differences = numeric_differences(&cycle[0], &cycle[1]);
        if differences.len() == 1 {
            let address = differences[0];
            let first = cycle[0]
                .get(address)
                .expect("cycle states have one configured width")
                .val;
            let second = cycle[1]
                .get(address)
                .expect("cycle states have one configured width")
                .val;
            if first == !second || (first == 0 && second != 0) || (first != 0 && second == 0) {
                return ParadoxDiagnosis::NegativeLoop {
                    cells: vec![address],
                    explanation: format!(
                        "Cell {address} oscillates between {first} and {second}. \
                         This is a grandfather paradox: the cell's value \
                         determines its own opposite."
                    ),
                };
            }
        }
    }

    ParadoxDiagnosis::Oscillation {
        cycle: cycle
            .iter()
            .map(PagedMemory::numeric_sparse_state)
            .collect(),
    }
}

fn detect_divergence(trajectory: &[PagedMemory]) -> Option<(Vec<Address>, Direction)> {
    if trajectory.len() < 5 {
        return None;
    }

    let mut changed = std::collections::HashSet::new();
    for window in trajectory.windows(2) {
        changed.extend(numeric_differences(&window[0], &window[1]));
    }

    let mut diverging = Vec::new();
    let mut direction = Direction::Increasing;
    for address in changed {
        let values = trajectory
            .iter()
            .map(|memory| {
                memory
                    .get(address)
                    .expect("trajectory states have one configured width")
                    .val
            })
            .collect::<Vec<_>>();
        let increasing = values.windows(2).all(|pair| pair[1] > pair[0]);
        let decreasing = values.windows(2).all(|pair| pair[1] < pair[0]);
        if increasing {
            diverging.push(address);
            direction = Direction::Increasing;
        } else if decreasing {
            diverging.push(address);
            direction = Direction::Decreasing;
        }
    }
    diverging.sort_unstable();
    (!diverging.is_empty()).then_some((diverging, direction))
}

fn observable_outputs_equal(
    left: &[crate::core::OutputItem],
    right: &[crate::core::OutputItem],
) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| match (left, right) {
                (crate::core::OutputItem::Val(left), crate::core::OutputItem::Val(right)) => {
                    left.val == right.val
                }
                (crate::core::OutputItem::Char(left), crate::core::OutputItem::Char(right)) => {
                    left == right
                }
                _ => false,
            })
}

fn dense_memory(memory: &PagedMemory) -> Memory {
    let mut dense = Memory::with_size(memory.len());
    for (address, value) in memory.iter_sparse() {
        dense.write(address, value.clone());
    }
    dense
}

pub(crate) fn retain_orbit_state(
    retained: usize,
    memory: &PagedMemory,
    sparse: &[(Address, u64)],
) -> Result<usize, String> {
    retain_orbit_state_with_limit(retained, memory, sparse, MAX_RETAINED_ORBIT_BYTES)
}

fn retain_orbit_state_with_limit(
    retained: usize,
    memory: &PagedMemory,
    sparse: &[(Address, u64)],
    limit: usize,
) -> Result<usize, String> {
    let charge = memory
        .retained_size_charge()
        .saturating_add(std::mem::size_of::<Vec<(Address, u64)>>())
        .saturating_add(
            sparse
                .len()
                .saturating_mul(std::mem::size_of::<(Address, u64)>()),
        );
    match retained.checked_add(charge) {
        Some(bytes) if bytes <= limit => Ok(bytes),
        _ => Err(format!(
            "retained temporal orbit exceeds hard limit {limit} bytes"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{OpCode, Program, Stmt};
    use crate::bytecode::BytecodeProgram;
    use crate::core::{Provenance, Value};
    use crate::hir::HirProgram;
    use crate::temporal::timeloop::{ExecutionMode, TimeLoop, TimeLoopConfig};

    #[test]
    fn retained_orbit_budget_rejects_before_journal_growth() {
        let memory = PagedMemory::with_size(4).unwrap();
        let sparse = memory.numeric_sparse_state();
        let charge = memory
            .retained_size_charge()
            .saturating_add(std::mem::size_of::<Vec<(Address, u64)>>());
        assert_eq!(
            retain_orbit_state_with_limit(0, &memory, &sparse, charge).unwrap(),
            charge
        );
        assert!(retain_orbit_state_with_limit(0, &memory, &sparse, charge - 1).is_err());
    }

    fn compile(statements: Vec<Stmt>) -> BytecodeProgram {
        let mut source = Program::new();
        source.body = statements;
        BytecodeProgram::compile(&HirProgram::resolve(&source).unwrap()).unwrap()
    }

    fn source_and_bytecode(statements: Vec<Stmt>) -> (Program, BytecodeProgram) {
        let mut source = Program::new();
        source.body = statements;
        let bytecode = BytecodeProgram::compile(&HirProgram::resolve(&source).unwrap()).unwrap();
        (source, bytecode)
    }

    fn run_source(program: &Program, mode: ExecutionMode, seed: u64) -> ConvergenceStatus {
        TimeLoop::new(TimeLoopConfig {
            mode,
            seed,
            memory_cells: 4,
            ..TimeLoopConfig::default()
        })
        .unwrap()
        .run(program)
    }

    fn numeric_output(output: &[crate::core::OutputItem]) -> Vec<(u8, u64)> {
        output
            .iter()
            .map(|item| match item {
                crate::core::OutputItem::Val(value) => (0, value.val),
                crate::core::OutputItem::Char(value) => (1, u64::from(*value)),
            })
            .collect()
    }

    fn push_path(statements: &mut Vec<Stmt>, path: &str) {
        statements.extend(path.bytes().map(|byte| Stmt::Push(Value::new(byte as u64))));
        statements.push(Stmt::Push(Value::new(path.len() as u64)));
    }

    fn unique_file(name: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        std::env::temp_dir().join(format!(
            "ouro-bytecode-timeloop-{}-{}-{name}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn driver() -> BytecodeTimeLoop {
        let config = BytecodeTimeLoopConfig {
            memory_cells: 4,
            ..BytecodeTimeLoopConfig::default()
        };
        BytecodeTimeLoop::new(config).unwrap()
    }

    #[test]
    fn pure_mode_is_uncached_and_uses_its_compatibility_safety_ceiling() {
        let program = compile(vec![
            Stmt::Push(Value::new(1)),
            Stmt::Push(Value::ZERO),
            Stmt::Op(OpCode::Prophecy),
            Stmt::Push(Value::ZERO),
            Stmt::Op(OpCode::Oracle),
            Stmt::Op(OpCode::Pop),
        ]);
        let driver = BytecodeTimeLoop::new(BytecodeTimeLoopConfig {
            max_epochs: 1,
            memory_cells: 4,
            ..BytecodeTimeLoopConfig::default()
        })
        .unwrap();

        assert!(matches!(
            driver.run(&program),
            ConvergenceStatus::Timeout { max_epochs: 1 }
        ));
        assert!(matches!(
            driver.run_pure(&program),
            ConvergenceStatus::Consistent { epochs: 2, .. }
        ));
    }

    #[test]
    fn reaches_and_replays_a_point_fixed_state() {
        let program = compile(vec![
            Stmt::Push(Value::new(42)),
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Prophecy),
        ]);
        match driver().run(&program) {
            ConvergenceStatus::Consistent { memory, epochs, .. } => {
                assert_eq!(epochs, 2);
                assert_eq!(memory.read(0).val, 42);
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn detects_a_verified_two_state_orbit() {
        let program = compile(vec![
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Oracle),
            Stmt::Op(OpCode::Not),
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Prophecy),
        ]);
        match driver().run(&program) {
            ConvergenceStatus::Oscillation {
                period,
                oscillating_cells,
                ..
            } => {
                assert_eq!(period, 2);
                assert_eq!(oscillating_cells, vec![0]);
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn publishes_only_the_selected_fixed_epoch_output() {
        let program = compile(vec![
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Oracle),
            Stmt::Op(OpCode::Output),
            Stmt::Push(Value::new(42)),
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Prophecy),
        ]);
        match driver().run(&program) {
            ConvergenceStatus::Consistent { output, epochs, .. } => {
                assert_eq!(epochs, 2);
                assert_eq!(
                    output,
                    vec![crate::core::OutputItem::Val(Value::with_provenance(
                        42,
                        Provenance::single(0),
                    ))]
                );
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn freezes_input_identically_across_candidate_epochs() {
        let program = compile(vec![
            Stmt::Op(OpCode::Input),
            Stmt::Op(OpCode::Output),
            Stmt::Push(Value::new(7)),
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Prophecy),
        ]);
        let driver = BytecodeTimeLoop::new(BytecodeTimeLoopConfig {
            memory_cells: 4,
            vm: BytecodeVmConfig {
                input: vec![99],
                ..BytecodeVmConfig::default()
            },
            ..BytecodeTimeLoopConfig::default()
        })
        .unwrap();
        match driver.run(&program) {
            ConvergenceStatus::Consistent { output, epochs, .. } => {
                assert_eq!(epochs, 2);
                assert_eq!(output, vec![crate::core::OutputItem::Val(Value::new(99))]);
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn captures_live_system_inputs_once_and_replays_both_tapes() {
        let program = compile(vec![
            Stmt::Op(OpCode::Clock),
            Stmt::Op(OpCode::Dup),
            Stmt::Op(OpCode::Output),
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Prophecy),
            Stmt::Op(OpCode::Random),
            Stmt::Op(OpCode::Dup),
            Stmt::Op(OpCode::Output),
            Stmt::Push(Value::new(1)),
            Stmt::Op(OpCode::Prophecy),
        ]);
        let driver = BytecodeTimeLoop::new(BytecodeTimeLoopConfig {
            memory_cells: 4,
            vm: BytecodeVmConfig {
                allow_live_system_inputs: true,
                ..BytecodeVmConfig::default()
            },
            ..BytecodeTimeLoopConfig::default()
        })
        .unwrap();

        match driver.run(&program) {
            ConvergenceStatus::Consistent {
                memory,
                output,
                epochs,
            } => {
                assert_eq!(epochs, 2);
                assert_eq!(output.len(), 2);
                let clock = match &output[0] {
                    crate::core::OutputItem::Val(value) => value.val,
                    other => panic!("unexpected CLOCK output: {other:?}"),
                };
                let random = match &output[1] {
                    crate::core::OutputItem::Val(value) => value.val,
                    other => panic!("unexpected RANDOM output: {other:?}"),
                };
                assert_eq!(memory.read(0).val, clock);
                assert_eq!(memory.read(1).val, random);
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn live_system_capture_is_not_deferred_past_the_first_candidate() {
        let program = compile(vec![
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Oracle),
            Stmt::If {
                then_branch: vec![Stmt::Op(OpCode::Random), Stmt::Op(OpCode::Pop)],
                else_branch: None,
            },
            Stmt::Push(Value::new(1)),
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Prophecy),
        ]);
        let driver = BytecodeTimeLoop::new(BytecodeTimeLoopConfig {
            memory_cells: 4,
            vm: BytecodeVmConfig {
                allow_live_system_inputs: true,
                ..BytecodeVmConfig::default()
            },
            ..BytecodeTimeLoopConfig::default()
        })
        .unwrap();

        match driver.run(&program) {
            ConvergenceStatus::Error { message, epoch } => {
                assert_eq!(epoch, 2);
                assert!(
                    message.contains("frozen RANDOM input exhausted"),
                    "{message}"
                );
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn captures_live_file_snapshot_on_first_candidate_and_replays_it() {
        let path = unique_file("capture");
        std::fs::write(&path, b"1234567").unwrap();
        let path_text = path.display().to_string();
        let mut statements = Vec::new();
        push_path(&mut statements, &path_text);
        statements.extend([
            Stmt::Op(OpCode::FileSize),
            Stmt::Op(OpCode::Dup),
            Stmt::Op(OpCode::Output),
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Prophecy),
        ]);
        let program = compile(statements);
        let driver = BytecodeTimeLoop::new(BytecodeTimeLoopConfig {
            memory_cells: 4,
            vm: BytecodeVmConfig {
                file_read_capabilities: vec![path_text],
                allow_live_file_reads: true,
                ..BytecodeVmConfig::default()
            },
            ..BytecodeTimeLoopConfig::default()
        })
        .unwrap();
        match driver.run(&program) {
            ConvergenceStatus::Consistent {
                memory,
                output,
                epochs,
            } => {
                assert_eq!(epochs, 2);
                assert_eq!(memory.read(0).val, 7);
                assert_eq!(numeric_output(&output), vec![(0, 7)]);
            }
            other => panic!("unexpected result: {other:?}"),
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn live_file_capture_is_not_deferred_past_the_first_candidate() {
        let path = unique_file("late-capture");
        std::fs::write(&path, b"data").unwrap();
        let path_text = path.display().to_string();
        let mut then_branch = Vec::new();
        push_path(&mut then_branch, &path_text);
        then_branch.extend([Stmt::Op(OpCode::FileSize), Stmt::Op(OpCode::Pop)]);
        let program = compile(vec![
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Oracle),
            Stmt::If {
                then_branch,
                else_branch: None,
            },
            Stmt::Push(Value::new(1)),
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Prophecy),
        ]);
        let driver = BytecodeTimeLoop::new(BytecodeTimeLoopConfig {
            memory_cells: 4,
            vm: BytecodeVmConfig {
                file_read_capabilities: vec![path_text],
                allow_live_file_reads: true,
                ..BytecodeVmConfig::default()
            },
            ..BytecodeTimeLoopConfig::default()
        })
        .unwrap();
        match driver.run(&program) {
            ConvergenceStatus::Error { message, epoch } => {
                assert_eq!(epoch, 2);
                assert!(
                    message.contains("frozen file snapshot unavailable"),
                    "{message}"
                );
            }
            other => panic!("unexpected result: {other:?}"),
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn frozen_endpoint_cursor_resets_for_every_candidate_epoch() {
        let host = "timeloop.invalid";
        let endpoint = format!("{host}:3030");
        let mut statements = Vec::new();
        push_path(&mut statements, host);
        statements.extend([
            Stmt::Push(Value::new(3030)),
            Stmt::Op(OpCode::TcpConnect),
            Stmt::Push(Value::ONE),
            Stmt::Op(OpCode::SocketRecv),
            Stmt::Op(OpCode::Pop),
            Stmt::Op(OpCode::BufferToStack),
            Stmt::Op(OpCode::Pop),
            Stmt::Push(Value::ZERO),
            Stmt::Op(OpCode::Prophecy),
            Stmt::Op(OpCode::SocketClose),
        ]);
        let program = compile(statements);
        let driver = BytecodeTimeLoop::new(BytecodeTimeLoopConfig {
            memory_cells: 4,
            vm: BytecodeVmConfig {
                endpoint_tapes: vec![FrozenEndpointTape {
                    endpoint: endpoint.clone(),
                    recv_bytes: vec![b'Q'],
                }],
                ..BytecodeVmConfig::default()
            },
            ..BytecodeTimeLoopConfig::default()
        })
        .unwrap();

        match driver.run(&program) {
            ConvergenceStatus::Consistent { memory, epochs, .. } => {
                assert_eq!(epochs, 2);
                assert_eq!(memory.read(0).val, b'Q' as u64);
            }
            other => panic!("frozen endpoint was not replayed: {other:?}"),
        }
    }

    #[test]
    fn selected_sleep_reaches_adapter_once_and_never_runs_in_candidates() {
        #[derive(Default)]
        struct RecordingAdapter {
            calls: usize,
            effects: Vec<crate::temporal::transaction::EffectIntent>,
        }
        impl EffectCommitAdapter for RecordingAdapter {
            fn apply_selected(
                &mut self,
                _receipt: &crate::temporal::transaction::CommitReceipt,
                effects: &[crate::temporal::transaction::EffectIntent],
            ) -> Result<(), String> {
                self.calls += 1;
                self.effects = effects.to_vec();
                Ok(())
            }
        }

        let program = compile(vec![Stmt::Push(Value::ZERO), Stmt::Op(OpCode::Sleep)]);
        let driver = BytecodeTimeLoop::new(BytecodeTimeLoopConfig {
            memory_cells: 4,
            vm: BytecodeVmConfig {
                max_sleep_milliseconds: Some(0),
                ..BytecodeVmConfig::default()
            },
            ..BytecodeTimeLoopConfig::default()
        })
        .unwrap();
        let mut adapter = RecordingAdapter::default();
        let mut log = CommitLog::default();
        assert!(matches!(
            driver.run_with_adapter(&program, CommitToken(3030), &mut log, &mut adapter),
            ConvergenceStatus::Consistent { epochs: 1, .. }
        ));
        assert!(matches!(
            driver.run_with_adapter(&program, CommitToken(3030), &mut log, &mut adapter),
            ConvergenceStatus::Consistent { .. }
        ));
        assert_eq!(adapter.calls, 1);
        assert_eq!(
            adapter.effects,
            vec![crate::temporal::transaction::EffectIntent::Sleep { milliseconds: 0 }]
        );
    }

    #[test]
    fn selected_file_effects_remain_staged_without_an_authorized_adapter() {
        let path = unique_file("staged-only");
        std::fs::write(&path, b"host").unwrap();
        let path_text = path.display().to_string();
        let mut statements = vec![
            Stmt::Push(Value::new(b'x' as u64)),
            Stmt::Push(Value::new(1)),
            Stmt::Op(OpCode::BufferFromStack),
        ];
        push_path(&mut statements, &path_text);
        statements.extend([
            Stmt::Push(Value::new(
                (crate::runtime::io::FileMode::WRITE.0 | crate::runtime::io::FileMode::TRUNCATE.0)
                    as u64,
            )),
            Stmt::Op(OpCode::FileOpen),
            Stmt::Op(OpCode::Swap),
            Stmt::Op(OpCode::FileWrite),
            Stmt::Op(OpCode::Pop),
            Stmt::Op(OpCode::FileClose),
            Stmt::Push(Value::new(9)),
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Prophecy),
        ]);
        let program = compile(statements);
        let driver = BytecodeTimeLoop::new(BytecodeTimeLoopConfig {
            memory_cells: 4,
            vm: BytecodeVmConfig {
                file_snapshots: vec![FrozenFileSnapshot {
                    path: path_text.clone(),
                    contents: Some(b"frozen".to_vec()),
                }],
                file_write_capabilities: vec![path_text],
                ..BytecodeVmConfig::default()
            },
            ..BytecodeTimeLoopConfig::default()
        })
        .unwrap();
        assert!(matches!(
            driver.run(&program),
            ConvergenceStatus::Consistent { .. }
        ));
        assert_eq!(std::fs::read(&path).unwrap(), b"host");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn authorized_adapter_receives_only_the_selected_file_batch_once() {
        #[derive(Default)]
        struct RecordingAdapter {
            calls: usize,
            effects: Vec<crate::temporal::transaction::EffectIntent>,
        }
        impl EffectCommitAdapter for RecordingAdapter {
            fn apply_selected(
                &mut self,
                _receipt: &crate::temporal::transaction::CommitReceipt,
                effects: &[crate::temporal::transaction::EffectIntent],
            ) -> Result<(), String> {
                self.calls += 1;
                self.effects = effects.to_vec();
                Ok(())
            }
        }

        let path = "virtual-selected-file";
        let mut statements = vec![
            Stmt::Push(Value::new(b'z' as u64)),
            Stmt::Push(Value::new(1)),
            Stmt::Op(OpCode::BufferFromStack),
        ];
        push_path(&mut statements, path);
        statements.extend([
            Stmt::Push(Value::new(crate::runtime::io::FileMode::APPEND.0 as u64)),
            Stmt::Op(OpCode::FileOpen),
            Stmt::Op(OpCode::Swap),
            Stmt::Op(OpCode::FileWrite),
            Stmt::Op(OpCode::Pop),
            Stmt::Op(OpCode::FileClose),
        ]);
        push_path(&mut statements, path);
        statements.extend([
            Stmt::Op(OpCode::FileSize),
            Stmt::Op(OpCode::Dup),
            Stmt::Op(OpCode::Output),
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Prophecy),
        ]);
        let program = compile(statements);
        let driver = BytecodeTimeLoop::new(BytecodeTimeLoopConfig {
            memory_cells: 4,
            vm: BytecodeVmConfig {
                file_snapshots: vec![FrozenFileSnapshot {
                    path: path.to_string(),
                    contents: Some(b"old".to_vec()),
                }],
                file_read_capabilities: vec![path.to_string()],
                file_write_capabilities: vec![path.to_string()],
                ..BytecodeVmConfig::default()
            },
            ..BytecodeTimeLoopConfig::default()
        })
        .unwrap();
        let mut adapter = RecordingAdapter::default();
        let mut log = CommitLog::default();
        let result = driver.run_with_adapter(&program, CommitToken(99), &mut log, &mut adapter);
        assert!(matches!(
            result,
            ConvergenceStatus::Consistent {
                epochs: 2,
                ref output,
                ..
            } if numeric_output(output) == vec![(0, 4)]
        ));
        let replay = driver.run_with_adapter(&program, CommitToken(99), &mut log, &mut adapter);
        assert!(matches!(replay, ConvergenceStatus::Consistent { .. }));
        assert_eq!(adapter.calls, 1);
        assert_eq!(log.batches().len(), 1);
        assert_eq!(
            log.batches()[0].observations.files,
            vec![FrozenFileSnapshot {
                path: path.to_string(),
                contents: Some(b"old".to_vec()),
            }]
        );
        assert_eq!(adapter.effects.len(), 1);
        assert!(matches!(
            &adapter.effects[0],
            crate::temporal::transaction::EffectIntent::FileWrite { offset: 3, bytes, .. }
                if bytes == b"z"
        ));
    }

    #[test]
    fn deutsch_period_one_dispatches_its_selected_effect_batch_once() {
        #[derive(Default)]
        struct RecordingAdapter {
            calls: usize,
            effects: Vec<crate::temporal::transaction::EffectIntent>,
        }
        impl EffectCommitAdapter for RecordingAdapter {
            fn apply_selected(
                &mut self,
                _receipt: &crate::temporal::transaction::CommitReceipt,
                effects: &[crate::temporal::transaction::EffectIntent],
            ) -> Result<(), String> {
                self.calls += 1;
                self.effects = effects.to_vec();
                Ok(())
            }
        }

        let program = compile(vec![Stmt::Push(Value::ONE), Stmt::Op(OpCode::Sleep)]);
        let driver = BytecodeTimeLoop::new(BytecodeTimeLoopConfig {
            memory_cells: 4,
            vm: BytecodeVmConfig {
                max_sleep_milliseconds: Some(1),
                ..BytecodeVmConfig::default()
            },
            ..BytecodeTimeLoopConfig::default()
        })
        .unwrap();
        let mut adapter = RecordingAdapter::default();
        let mut log = CommitLog::default();
        assert!(matches!(
            driver.run_deutsch_with_adapter(&program, CommitToken(4040), &mut log, &mut adapter,),
            ConvergenceStatus::Consistent { epochs: 1, .. }
        ));
        assert_eq!(adapter.calls, 1);
        assert!(matches!(
            adapter.effects.as_slice(),
            [crate::temporal::transaction::EffectIntent::Sleep { milliseconds: 1 }]
        ));
    }

    #[test]
    fn deutsch_cycle_with_effects_is_withheld_without_batch_selection_policy() {
        let path = "virtual-deutsch-file";
        let mut statements = vec![
            Stmt::Push(Value::new(b'z' as u64)),
            Stmt::Push(Value::new(1)),
            Stmt::Op(OpCode::BufferFromStack),
        ];
        push_path(&mut statements, path);
        statements.extend([
            Stmt::Push(Value::new(crate::runtime::io::FileMode::APPEND.0 as u64)),
            Stmt::Op(OpCode::FileOpen),
            Stmt::Op(OpCode::Swap),
            Stmt::Op(OpCode::FileWrite),
            Stmt::Op(OpCode::Pop),
            Stmt::Op(OpCode::FileClose),
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Oracle),
            Stmt::Op(OpCode::Not),
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Prophecy),
        ]);
        let program = compile(statements);
        let driver = BytecodeTimeLoop::new(BytecodeTimeLoopConfig {
            memory_cells: 4,
            vm: BytecodeVmConfig {
                file_snapshots: vec![FrozenFileSnapshot {
                    path: path.to_string(),
                    contents: Some(b"old".to_vec()),
                }],
                file_write_capabilities: vec![path.to_string()],
                ..BytecodeVmConfig::default()
            },
            ..BytecodeTimeLoopConfig::default()
        })
        .unwrap();
        match driver.run_deutsch(&program) {
            ConvergenceStatus::Error { message, epoch } => {
                assert_eq!(epoch, 2);
                assert!(message.contains("staged effects"), "{message}");
                assert!(
                    message.contains("no host effects were committed"),
                    "{message}"
                );
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn propagates_paradox_and_runtime_limits_without_host_recursion() {
        let paradox = compile(vec![Stmt::Op(OpCode::Paradox)]);
        assert!(matches!(
            driver().run(&paradox),
            ConvergenceStatus::Paradox { epoch: 1, .. }
        ));

        let infinite = compile(vec![Stmt::While {
            cond: vec![Stmt::Push(Value::ONE)],
            body: vec![Stmt::Op(OpCode::Nop)],
        }]);
        let config = BytecodeTimeLoopConfig {
            memory_cells: 4,
            vm: BytecodeVmConfig {
                max_instructions: 8,
                ..BytecodeVmConfig::default()
            },
            ..BytecodeTimeLoopConfig::default()
        };
        let result = BytecodeTimeLoop::new(config).unwrap().run(&infinite);
        assert!(matches!(result, ConvergenceStatus::Error { epoch: 1, .. }));
    }

    #[test]
    fn construction_rejects_memory_outside_dense_result_limit() {
        let error = BytecodeTimeLoop::new(BytecodeTimeLoopConfig {
            memory_cells: crate::core::MAX_DENSE_MEMORY_CELLS + 1,
            ..BytecodeTimeLoopConfig::default()
        })
        .unwrap_err();
        assert!(error.contains("dense result limit"), "{error}");
    }

    #[test]
    fn diagnostic_negative_loop_matches_source_policy() {
        let (source, bytecode) = source_and_bytecode(vec![
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Oracle),
            Stmt::Op(OpCode::Not),
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Prophecy),
        ]);
        let source_status = run_source(&source, ExecutionMode::Diagnostic, 0);
        let bytecode_status = driver().run_diagnostic(&bytecode);

        match (source_status, bytecode_status) {
            (
                ConvergenceStatus::Oscillation {
                    period: source_period,
                    oscillating_cells: source_cells,
                    diagnosis:
                        ParadoxDiagnosis::NegativeLoop {
                            cells: source_diagnosis_cells,
                            ..
                        },
                },
                ConvergenceStatus::Oscillation {
                    period: bytecode_period,
                    oscillating_cells: bytecode_cells,
                    diagnosis:
                        ParadoxDiagnosis::NegativeLoop {
                            cells: bytecode_diagnosis_cells,
                            ..
                        },
                },
            ) => {
                assert_eq!(bytecode_period, source_period);
                assert_eq!(bytecode_cells, source_cells);
                assert_eq!(bytecode_diagnosis_cells, source_diagnosis_cells);
            }
            other => panic!("diagnostic policies diverged: {other:?}"),
        }
    }

    #[test]
    fn diagnostic_divergence_matches_source_policy() {
        let (source, bytecode) = source_and_bytecode(vec![
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Oracle),
            Stmt::Push(Value::new(1)),
            Stmt::Op(OpCode::Add),
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Prophecy),
        ]);
        let source_status = run_source(&source, ExecutionMode::Diagnostic, 0);
        let bytecode_status = driver().run_diagnostic(&bytecode);

        match (source_status, bytecode_status) {
            (
                ConvergenceStatus::Divergence {
                    diverging_cells: source_cells,
                    direction: source_direction,
                },
                ConvergenceStatus::Divergence {
                    diverging_cells: bytecode_cells,
                    direction: bytecode_direction,
                },
            ) => {
                assert_eq!(bytecode_cells, source_cells);
                assert_eq!(bytecode_direction, source_direction);
            }
            other => panic!("diagnostic policies diverged: {other:?}"),
        }
    }

    #[test]
    fn diagnostic_commits_only_the_selected_fixed_epoch_output() {
        let program = compile(vec![
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Oracle),
            Stmt::Op(OpCode::Output),
            Stmt::Push(Value::new(42)),
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Prophecy),
        ]);
        match driver().run_diagnostic(&program) {
            ConvergenceStatus::Consistent { output, epochs, .. } => {
                assert_eq!(epochs, 2);
                assert_eq!(numeric_output(&output), vec![(0, 42)]);
            }
            other => panic!("unexpected diagnostic result: {other:?}"),
        }
    }

    #[test]
    fn deutsch_transient_cycle_and_outputs_match_source_policy() {
        let (source, bytecode) = source_and_bytecode(vec![
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Oracle),
            Stmt::Op(OpCode::Dup),
            Stmt::Op(OpCode::Output),
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Eq),
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Prophecy),
        ]);
        let source_status = run_source(&source, ExecutionMode::Deutsch, 7);
        let bytecode_driver = BytecodeTimeLoop::new(BytecodeTimeLoopConfig {
            memory_cells: 4,
            seed: 7,
            ..BytecodeTimeLoopConfig::default()
        })
        .unwrap();
        let bytecode_status = bytecode_driver.run_deutsch(&bytecode);

        match (source_status, bytecode_status) {
            (
                ConvergenceStatus::DeutschConsistent {
                    cycle: source_cycle,
                    outputs: source_outputs,
                    period: source_period,
                    transient_epochs: source_transient,
                    unanimous_output: source_unanimous,
                },
                ConvergenceStatus::DeutschConsistent {
                    cycle: bytecode_cycle,
                    outputs: bytecode_outputs,
                    period: bytecode_period,
                    transient_epochs: bytecode_transient,
                    unanimous_output: bytecode_unanimous,
                },
            ) => {
                assert_eq!(bytecode_period, source_period);
                assert_eq!(bytecode_transient, source_transient);
                assert_eq!(
                    bytecode_cycle
                        .iter()
                        .map(Memory::sparse_state)
                        .collect::<Vec<_>>(),
                    source_cycle
                        .iter()
                        .map(Memory::sparse_state)
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    bytecode_outputs
                        .iter()
                        .map(|output| numeric_output(output))
                        .collect::<Vec<_>>(),
                    source_outputs
                        .iter()
                        .map(|output| numeric_output(output))
                        .collect::<Vec<_>>()
                );
                assert_eq!(bytecode_unanimous.is_some(), source_unanimous.is_some());
            }
            other => panic!("Deutsch policies diverged: {other:?}"),
        }
    }

    #[test]
    fn deutsch_point_state_commits_its_selected_output() {
        let program = compile(vec![
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Oracle),
            Stmt::Op(OpCode::Output),
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Oracle),
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Prophecy),
        ]);
        match driver().run_deutsch(&program) {
            ConvergenceStatus::DeutschConsistent {
                period,
                transient_epochs,
                outputs,
                unanimous_output,
                ..
            } => {
                assert_eq!(period, 1);
                assert_eq!(transient_epochs, 0);
                assert_eq!(numeric_output(&outputs[0]), vec![(0, 0)]);
                assert_eq!(
                    numeric_output(unanimous_output.as_ref().unwrap()),
                    vec![(0, 0)]
                );
            }
            other => panic!("unexpected Deutsch result: {other:?}"),
        }
    }

    #[test]
    fn diagnostic_and_deutsch_preserve_the_trivial_program_shortcut() {
        let (source, bytecode) = source_and_bytecode(vec![
            Stmt::Push(Value::new(42)),
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Prophecy),
            Stmt::Push(Value::new(9)),
            Stmt::Op(OpCode::Output),
        ]);
        let bytecode_driver = BytecodeTimeLoop::new(BytecodeTimeLoopConfig {
            memory_cells: 4,
            seed: 7,
            ..BytecodeTimeLoopConfig::default()
        })
        .unwrap();

        for (source_status, bytecode_status) in [
            (
                run_source(&source, ExecutionMode::Diagnostic, 7),
                bytecode_driver.run_diagnostic(&bytecode),
            ),
            (
                run_source(&source, ExecutionMode::Deutsch, 7),
                bytecode_driver.run_deutsch(&bytecode),
            ),
        ] {
            match (source_status, bytecode_status) {
                (
                    ConvergenceStatus::Consistent {
                        memory: source_memory,
                        output: source_output,
                        epochs: source_epochs,
                    },
                    ConvergenceStatus::Consistent {
                        memory: bytecode_memory,
                        output: bytecode_output,
                        epochs: bytecode_epochs,
                    },
                ) => {
                    assert_eq!(bytecode_epochs, source_epochs);
                    assert_eq!(bytecode_memory.sparse_state(), source_memory.sparse_state());
                    assert_eq!(
                        numeric_output(&bytecode_output),
                        numeric_output(&source_output)
                    );
                }
                other => panic!("trivial shortcut policies diverged: {other:?}"),
            }
        }
    }
}
