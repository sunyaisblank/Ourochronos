//! Complete deterministic transition-graph analysis for small finite domains.
//!
//! Unlike orbit-based cycle detection, this module evaluates every state in a
//! declared finite domain.  It refuses to return an analysis when the program
//! is partial, escapes the domain, or the requested graph exceeds its state
//! limit.  Consequently every reported recurrent class and basin is complete
//! for that domain.

use crate::ast::Program;
use crate::bytecode::BytecodeProgram;
use crate::bytecode_vm::{
    BytecodeVm, BytecodeVmConfig, BytecodeVmError, BytecodeVmStatus, PreparedBytecode,
};
use crate::core::{BoundsPolicy, Memory, OutputItem, PagedMemory, Value};
use crate::hir::HirProgram;
use crate::linker::{link, ObjectModule};
use std::fmt;

/// Hard ceiling for complete recurrent-domain enumeration, independent of a
/// caller's softer `max_states` request.
pub const MAX_RECURRENT_STATES: usize = 262_144;

/// Hard ceiling on the aggregate instructions fetched while enumerating one
/// complete transition graph.
pub const MAX_RECURRENT_INSTRUCTIONS: u64 = 100_000_000;

/// Hard ceiling on language-visible output retained across all source states.
pub const MAX_RECURRENT_OUTPUT_ITEMS: usize = 1_000_000;

/// Hard ceiling on conservative retained output bytes across all source states.
pub const MAX_RECURRENT_OUTPUT_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeterministicTransitionGraph {
    successors: Vec<usize>,
}

impl DeterministicTransitionGraph {
    pub fn new(successors: Vec<usize>) -> Result<Self, TransitionGraphError> {
        if successors.is_empty() {
            return Err(TransitionGraphError::InvalidGraph(
                "a transition graph must contain at least one state".to_string(),
            ));
        }
        let state_count = successors.len();
        for (source, &target) in successors.iter().enumerate() {
            if target >= state_count {
                return Err(TransitionGraphError::InvalidGraph(format!(
                    "state {} targets {}, outside 0..{}",
                    source, target, state_count
                )));
            }
        }
        Ok(Self { successors })
    }

    pub fn state_count(&self) -> usize {
        self.successors.len()
    }

    pub fn successor(&self, state: usize) -> usize {
        self.successors[state]
    }

    pub fn successors(&self) -> &[usize] {
        &self.successors
    }

    /// Classify every state in this finite functional graph.
    pub fn analyze(&self) -> RecurrentAnalysis {
        let count = self.successors.len();
        let mut class_of = vec![None; count];
        let mut distance_to_recurrence = vec![0usize; count];
        let mut recurrent_classes: Vec<Vec<usize>> = Vec::new();
        let mut visit_generation = vec![0usize; count];
        let mut position = vec![0usize; count];
        let mut path = Vec::new();

        for start in 0..count {
            if class_of[start].is_some() {
                continue;
            }

            path.clear();
            let generation = start + 1;
            let mut current = start;
            while class_of[current].is_none() && visit_generation[current] != generation {
                visit_generation[current] = generation;
                position[current] = path.len();
                path.push(current);
                current = self.successors[current];
            }

            if let Some(existing_class) = class_of[current] {
                for (distance, &state) in
                    (distance_to_recurrence[current] + 1..).zip(path.iter().rev())
                {
                    class_of[state] = Some(existing_class);
                    distance_to_recurrence[state] = distance;
                }
                continue;
            }

            let cycle_start = position[current];
            let mut cycle = path[cycle_start..].to_vec();
            rotate_to_smallest(&mut cycle);
            let class = recurrent_classes.len();
            for &state in &cycle {
                class_of[state] = Some(class);
                distance_to_recurrence[state] = 0;
            }
            recurrent_classes.push(cycle);

            for (distance, &state) in (1..).zip(path[..cycle_start].iter().rev()) {
                class_of[state] = Some(class);
                distance_to_recurrence[state] = distance;
            }
        }

        let state_class: Vec<usize> = class_of
            .into_iter()
            .map(|class| class.expect("functional graph traversal assigns every state"))
            .collect();
        let mut basin_sizes = vec![0usize; recurrent_classes.len()];
        for &class in &state_class {
            basin_sizes[class] += 1;
        }
        let fixed_states = recurrent_classes
            .iter()
            .filter(|class| class.len() == 1)
            .map(|class| class[0])
            .collect();

        RecurrentAnalysis {
            recurrent_classes,
            fixed_states,
            state_class,
            distance_to_recurrence,
            basin_sizes,
        }
    }
}

fn rotate_to_smallest(cycle: &mut [usize]) {
    if let Some((index, _)) = cycle.iter().enumerate().min_by_key(|(_, state)| *state) {
        cycle.rotate_left(index);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecurrentAnalysis {
    /// Every directed cycle, canonically rotated to its smallest state.
    pub recurrent_classes: Vec<Vec<usize>>,
    pub fixed_states: Vec<usize>,
    /// The recurrent-class index eventually reached from each state.
    pub state_class: Vec<usize>,
    /// Number of transitions from each state to its recurrent class.
    pub distance_to_recurrence: Vec<usize>,
    /// Number of recurrent and transient states attracted to each class.
    pub basin_sizes: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
pub struct ProgramGraphConfig {
    pub memory_cells: usize,
    pub cell_bits: u8,
    pub max_states: usize,
    pub max_instructions: u64,
    pub bounds_policy: BoundsPolicy,
}

impl Default for ProgramGraphConfig {
    fn default() -> Self {
        Self {
            memory_cells: 1,
            cell_bits: 1,
            max_states: 65_536,
            max_instructions: 10_000_000,
            bounds_policy: BoundsPolicy::Wrap,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProgramGraphAnalysis {
    pub graph: DeterministicTransitionGraph,
    pub recurrent: RecurrentAnalysis,
    /// Buffered bytecode-VM output for each source state.
    pub outputs: Vec<Vec<OutputItem>>,
    pub memory_cells: usize,
    pub cell_bits: u8,
}

impl ProgramGraphAnalysis {
    pub fn state_memory(&self, state: usize) -> Memory {
        decode_state(state, self.memory_cells, self.cell_bits)
    }

    /// A recurrent class has a stable readout exactly when every state on the
    /// cycle produces the same buffered output.
    pub fn class_has_unanimous_output(&self, class: usize) -> bool {
        let states = &self.recurrent.recurrent_classes[class];
        match states.split_first() {
            None => true,
            Some((first, rest)) => rest.iter().all(|state| {
                observable_outputs_equal(&self.outputs[*state], &self.outputs[*first])
            }),
        }
    }

    /// Graphviz DOT for the complete transition graph. Recurrent states use
    /// double circles; every state and edge is included.
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph ourochronos_recurrence {\n  rankdir=LR;\n");
        for state in 0..self.graph.state_count() {
            let memory = format!("{:?}", self.state_memory(state))
                .replace('\\', "\\\\")
                .replace('\"', "\\\"");
            let shape = if self.recurrent.distance_to_recurrence[state] == 0 {
                "doublecircle"
            } else {
                "circle"
            };
            dot.push_str(&format!(
                "  n{} [label=\"{}\\n{}\", shape={}];\n",
                state, state, memory, shape
            ));
        }
        for (source, target) in self.graph.successors().iter().enumerate() {
            dot.push_str(&format!("  n{} -> n{};\n", source, target));
        }
        dot.push_str("}\n");
        dot
    }
}

/// Compare language-visible output while ignoring provenance metadata.
fn observable_outputs_equal(left: &[OutputItem], right: &[OutputItem]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| match (left, right) {
                (OutputItem::Val(left), OutputItem::Val(right)) => left.val == right.val,
                (OutputItem::Char(left), OutputItem::Char(right)) => left == right,
                _ => false,
            })
}

pub struct ProgramTransitionAnalyzer;

impl ProgramTransitionAnalyzer {
    /// Compile and link a source program, then analyze the resulting bytecode.
    ///
    /// This source-facing entry point is retained for API compatibility. New
    /// compiler and package integrations should use [`BytecodeTransitionAnalyzer`]
    /// so the already-linked artifact remains the single execution authority.
    pub fn analyze(
        program: &Program,
        config: ProgramGraphConfig,
    ) -> Result<ProgramGraphAnalysis, TransitionGraphError> {
        let program = linked_bytecode(program)?;
        BytecodeTransitionAnalyzer::analyze(&program, config)
    }

    /// Analyze an already-linked bytecode artifact.
    ///
    /// This convenience method is equivalent to
    /// [`BytecodeTransitionAnalyzer::analyze`].
    pub fn analyze_bytecode(
        program: &BytecodeProgram,
        config: ProgramGraphConfig,
    ) -> Result<ProgramGraphAnalysis, TransitionGraphError> {
        BytecodeTransitionAnalyzer::analyze(program, config)
    }
}

/// Complete transition-graph analysis over validated linked bytecode.
pub struct BytecodeTransitionAnalyzer;

impl BytecodeTransitionAnalyzer {
    /// Execute the linked artifact once for every state in the bounded domain.
    ///
    /// Every epoch uses [`BytecodeVm`] with a [`PagedMemory`] anamnesis. The
    /// method returns an analysis only when the artifact is valid, every
    /// transition terminates normally, and every produced word remains inside
    /// the declared cell-width domain.
    pub fn analyze(
        program: &BytecodeProgram,
        config: ProgramGraphConfig,
    ) -> Result<ProgramGraphAnalysis, TransitionGraphError> {
        let state_count = checked_state_count(config)?;
        let prepared = PreparedBytecode::new(program.clone())
            .map_err(|error| TransitionGraphError::Unsupported(error.to_string()))?;
        let mask = cell_mask(config.cell_bits);
        let mut successors = Vec::new();
        successors.try_reserve_exact(state_count).map_err(|_| {
            TransitionGraphError::ResourceLimit(format!(
                "cannot reserve successors for {state_count} recurrent states"
            ))
        })?;
        let mut outputs = Vec::new();
        outputs.try_reserve_exact(state_count).map_err(|_| {
            TransitionGraphError::ResourceLimit(format!(
                "cannot reserve output tables for {state_count} recurrent states"
            ))
        })?;

        let mut vm = BytecodeVm::with_config(BytecodeVmConfig {
            max_instructions: config.max_instructions,
            memory_bounds: config.bounds_policy,
            input: Vec::new(),
            ..BytecodeVmConfig::default()
        });
        let mut aggregate_instructions = 0u64;
        let mut aggregate_outputs = 0usize;
        let mut aggregate_output_bytes = 0usize;
        for state in 0..state_count {
            let remaining_instructions = MAX_RECURRENT_INSTRUCTIONS - aggregate_instructions;
            let epoch_instruction_limit = config.max_instructions.min(remaining_instructions);
            vm.config.max_instructions = epoch_instruction_limit;
            let remaining_outputs = MAX_RECURRENT_OUTPUT_ITEMS - aggregate_outputs;
            vm.config.max_output_items = remaining_outputs;
            let remaining_output_bytes = MAX_RECURRENT_OUTPUT_BYTES - aggregate_output_bytes;
            vm.config.max_output_bytes = remaining_output_bytes;
            let memory = decode_paged_state(state, config.memory_cells, config.cell_bits)?;
            let epoch = vm.run_prepared(&prepared, &memory).map_err(|error| match error {
                BytecodeVmError::GasExhausted { .. }
                    if epoch_instruction_limit < config.max_instructions =>
                {
                    TransitionGraphError::ResourceLimit(format!(
                        "aggregate recurrent-analysis instructions reached hard ceiling {MAX_RECURRENT_INSTRUCTIONS}"
                    ))
                }
                BytecodeVmError::AllocationLimit { what: "output", .. } => {
                    TransitionGraphError::ResourceLimit(format!(
                        "aggregate recurrent-analysis output reached hard ceiling {MAX_RECURRENT_OUTPUT_ITEMS}"
                    ))
                }
                BytecodeVmError::AllocationLimit {
                    what: "output byte",
                    ..
                } => TransitionGraphError::ResourceLimit(format!(
                    "aggregate recurrent-analysis output bytes reached hard ceiling {MAX_RECURRENT_OUTPUT_BYTES}"
                )),
                error => TransitionGraphError::UndefinedTransition {
                    state,
                    reason: error.to_string(),
                },
            })?;
            aggregate_instructions = aggregate_instructions
                .checked_add(epoch.instructions_executed)
                .ok_or_else(|| {
                    TransitionGraphError::ResourceLimit(
                        "aggregate recurrent-analysis instruction count overflowed".to_string(),
                    )
                })?;
            if aggregate_instructions > MAX_RECURRENT_INSTRUCTIONS {
                return Err(TransitionGraphError::ResourceLimit(format!(
                    "aggregate recurrent-analysis instructions exceed hard ceiling {MAX_RECURRENT_INSTRUCTIONS}"
                )));
            }
            aggregate_outputs = aggregate_outputs
                .checked_add(epoch.output.len())
                .ok_or_else(|| {
                    TransitionGraphError::ResourceLimit(
                        "aggregate recurrent-analysis output count overflowed".to_string(),
                    )
                })?;
            if aggregate_outputs > MAX_RECURRENT_OUTPUT_ITEMS {
                return Err(TransitionGraphError::ResourceLimit(format!(
                    "aggregate recurrent-analysis output exceeds hard ceiling {MAX_RECURRENT_OUTPUT_ITEMS}"
                )));
            }
            let epoch_output_bytes = epoch.output.iter().try_fold(0usize, |total, item| {
                total.checked_add(item.retained_size_charge())
            });
            aggregate_output_bytes = aggregate_output_bytes
                .checked_add(epoch_output_bytes.ok_or_else(|| {
                    TransitionGraphError::ResourceLimit(
                        "recurrent-analysis output byte charge overflowed".to_string(),
                    )
                })?)
                .ok_or_else(|| {
                    TransitionGraphError::ResourceLimit(
                        "aggregate recurrent-analysis output bytes overflowed".to_string(),
                    )
                })?;
            if aggregate_output_bytes > MAX_RECURRENT_OUTPUT_BYTES {
                return Err(TransitionGraphError::ResourceLimit(format!(
                    "aggregate recurrent-analysis output bytes exceed hard ceiling {MAX_RECURRENT_OUTPUT_BYTES}"
                )));
            }
            match epoch.status {
                BytecodeVmStatus::Finished | BytecodeVmStatus::Halted => {}
                BytecodeVmStatus::Paradox => {
                    return Err(TransitionGraphError::UndefinedTransition {
                        state,
                        reason: "program executed PARADOX".to_string(),
                    });
                }
            }

            if let Some((address, value)) = epoch
                .present
                .iter()
                .find(|(_, value)| value.val & !mask != 0)
            {
                return Err(TransitionGraphError::DomainNotClosed {
                    state,
                    address,
                    value: value.val,
                    cell_bits: config.cell_bits,
                });
            }
            successors.push(encode_state(&epoch.present, config.cell_bits));
            outputs.push(epoch.output);
        }

        let graph = DeterministicTransitionGraph::new(successors)?;
        let recurrent = graph.analyze();
        Ok(ProgramGraphAnalysis {
            graph,
            recurrent,
            outputs,
            memory_cells: config.memory_cells,
            cell_bits: config.cell_bits,
        })
    }
}

fn linked_bytecode(program: &Program) -> Result<BytecodeProgram, TransitionGraphError> {
    let hir = HirProgram::resolve(program).map_err(|errors| {
        TransitionGraphError::Unsupported(format!(
            "typed name resolution failed: {}",
            errors
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; ")
        ))
    })?;
    let code = BytecodeProgram::compile(&hir)
        .map_err(|error| TransitionGraphError::Unsupported(error.to_string()))?;
    link(&[ObjectModule::new("transition-graph-source", code)])
        .map_err(|error| TransitionGraphError::Unsupported(error.to_string()))
}

fn checked_state_count(config: ProgramGraphConfig) -> Result<usize, TransitionGraphError> {
    if config.memory_cells == 0 {
        return Err(TransitionGraphError::InvalidDomain(
            "the state domain requires at least one memory cell".to_string(),
        ));
    }
    if config.cell_bits == 0 || config.cell_bits > 64 {
        return Err(TransitionGraphError::InvalidDomain(
            "cell bit width must be in 1..=64".to_string(),
        ));
    }
    if config.max_states == 0 {
        return Err(TransitionGraphError::InvalidDomain(
            "the state limit must be greater than zero".to_string(),
        ));
    }
    if config.max_states > MAX_RECURRENT_STATES {
        return Err(TransitionGraphError::InvalidDomain(format!(
            "the state limit {} exceeds hard ceiling {MAX_RECURRENT_STATES}",
            config.max_states
        )));
    }
    let exponent = config
        .memory_cells
        .checked_mul(config.cell_bits as usize)
        .ok_or(TransitionGraphError::DomainTooLarge {
            exponent: usize::MAX,
            max_states: config.max_states,
        })?;
    let state_count = if exponent < usize::BITS as usize {
        1usize << exponent
    } else {
        return Err(TransitionGraphError::DomainTooLarge {
            exponent,
            max_states: config.max_states,
        });
    };
    if state_count > config.max_states {
        return Err(TransitionGraphError::DomainTooLarge {
            exponent,
            max_states: config.max_states,
        });
    }
    Ok(state_count)
}

fn cell_mask(bits: u8) -> u64 {
    if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

fn decode_state(mut state: usize, memory_cells: usize, cell_bits: u8) -> Memory {
    let mut memory = Memory::with_size(memory_cells);
    let mask = cell_mask(cell_bits) as usize;
    for address in 0..memory_cells {
        let value = (state & mask) as u64;
        if value != 0 {
            memory.write(address as u64, Value::new(value));
        }
        state >>= cell_bits;
    }
    memory
}

fn decode_paged_state(
    mut state: usize,
    memory_cells: usize,
    cell_bits: u8,
) -> Result<PagedMemory, TransitionGraphError> {
    let mut memory = PagedMemory::with_size(memory_cells).map_err(|error| {
        TransitionGraphError::InvalidDomain(format!(
            "cannot construct the recurrent state memory: {error}"
        ))
    })?;
    let mask = cell_mask(cell_bits) as usize;
    for address in 0..memory_cells {
        let value = (state & mask) as u64;
        if value != 0 {
            memory
                .write(address as u64, Value::new(value))
                .map_err(|error| {
                    TransitionGraphError::InvalidDomain(format!(
                        "cannot initialize recurrent state memory: {error}"
                    ))
                })?;
        }
        state >>= cell_bits;
    }
    Ok(memory)
}

fn encode_state(memory: &PagedMemory, cell_bits: u8) -> usize {
    let mut state = 0usize;
    for (address, value) in memory.iter() {
        state |= (value.val as usize) << (address as usize * cell_bits as usize);
    }
    state
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransitionGraphError {
    InvalidGraph(String),
    InvalidDomain(String),
    ResourceLimit(String),
    DomainTooLarge {
        exponent: usize,
        max_states: usize,
    },
    Unsupported(String),
    UndefinedTransition {
        state: usize,
        reason: String,
    },
    DomainNotClosed {
        state: usize,
        address: u64,
        value: u64,
        cell_bits: u8,
    },
}

impl fmt::Display for TransitionGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGraph(reason)
            | Self::InvalidDomain(reason)
            | Self::ResourceLimit(reason)
            | Self::Unsupported(reason) => f.write_str(reason),
            Self::DomainTooLarge {
                exponent,
                max_states,
            } => write!(
                f,
                "requested domain has 2^{} states, exceeding the complete-analysis limit {}",
                exponent, max_states
            ),
            Self::UndefinedTransition { state, reason } => {
                write!(
                    f,
                    "transition from state {} is undefined: {}",
                    state, reason
                )
            }
            Self::DomainNotClosed {
                state,
                address,
                value,
                cell_bits,
            } => write!(
                f,
                "transition from state {} writes {} to cell {}, outside the {}-bit domain",
                state, value, address, cell_bits
            ),
        }
    }
}

impl std::error::Error for TransitionGraphError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Provenance;
    use crate::parser::parse;

    fn bytecode(source: &str) -> BytecodeProgram {
        linked_bytecode(&parse(source).unwrap()).unwrap()
    }

    #[test]
    fn functional_graph_classifies_every_cycle_and_basin() {
        let graph = DeterministicTransitionGraph::new(vec![1, 0, 2, 2]).unwrap();
        let analysis = graph.analyze();
        assert_eq!(analysis.recurrent_classes, vec![vec![0, 1], vec![2]]);
        assert_eq!(analysis.fixed_states, vec![2]);
        assert_eq!(analysis.state_class, vec![0, 0, 1, 1]);
        assert_eq!(analysis.distance_to_recurrence, vec![0, 0, 0, 1]);
        assert_eq!(analysis.basin_sizes, vec![2, 2]);
    }

    #[test]
    fn complete_one_bit_analysis_finds_the_not_cycle() {
        let program = parse("0 ORACLE NOT 0 PROPHECY").unwrap();
        let analysis =
            ProgramTransitionAnalyzer::analyze(&program, ProgramGraphConfig::default()).unwrap();
        assert_eq!(analysis.graph.successors(), &[1, 0]);
        assert_eq!(analysis.recurrent.recurrent_classes, vec![vec![0, 1]]);
        assert!(analysis.recurrent.fixed_states.is_empty());
    }

    #[test]
    fn complete_two_bit_analysis_finds_a_four_cycle() {
        let program = parse("0 ORACLE 1 ADD 3 AND 0 PROPHECY").unwrap();
        let analysis = ProgramTransitionAnalyzer::analyze(
            &program,
            ProgramGraphConfig {
                cell_bits: 2,
                ..ProgramGraphConfig::default()
            },
        )
        .unwrap();
        assert_eq!(analysis.recurrent.recurrent_classes, vec![vec![0, 1, 2, 3]]);
    }

    #[test]
    fn source_and_bytecode_apis_have_identical_observations_and_dot() {
        let source = parse("PROCEDURE step { 0 ORACLE NOT DUP OUTPUT 0 PROPHECY } step").unwrap();
        let bytecode = linked_bytecode(&source).unwrap();
        let source_analysis =
            ProgramTransitionAnalyzer::analyze(&source, ProgramGraphConfig::default()).unwrap();
        let bytecode_analysis =
            BytecodeTransitionAnalyzer::analyze(&bytecode, ProgramGraphConfig::default()).unwrap();

        assert_eq!(source_analysis.graph, bytecode_analysis.graph);
        assert_eq!(source_analysis.recurrent, bytecode_analysis.recurrent);
        assert_eq!(source_analysis.outputs, bytecode_analysis.outputs);
        assert_eq!(source_analysis.to_dot(), bytecode_analysis.to_dot());
    }

    #[test]
    fn bytecode_gas_limit_is_exact() {
        let bytecode = bytecode("0 ORACLE 0 PROPHECY");
        let required = u64::from(bytecode.main.end - bytecode.main.start);
        let accepted = BytecodeTransitionAnalyzer::analyze(
            &bytecode,
            ProgramGraphConfig {
                max_instructions: required,
                ..ProgramGraphConfig::default()
            },
        )
        .unwrap();
        assert_eq!(accepted.graph.successors(), &[0, 1]);

        let error = BytecodeTransitionAnalyzer::analyze(
            &bytecode,
            ProgramGraphConfig {
                max_instructions: required - 1,
                ..ProgramGraphConfig::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            TransitionGraphError::UndefinedTransition { state: 0, ref reason }
                if reason.contains("instruction limit")
        ));
    }

    #[test]
    fn malformed_bytecode_is_rejected_before_enumeration() {
        let mut bytecode = bytecode("0 ORACLE 0 PROPHECY");
        bytecode.main.end -= 1;
        let error = BytecodeTransitionAnalyzer::analyze(&bytecode, ProgramGraphConfig::default())
            .unwrap_err();
        assert!(matches!(error, TransitionGraphError::Unsupported(_)));
    }

    #[test]
    fn bytecode_environment_inputs_must_be_frozen_for_recurrent_analysis() {
        let clock = bytecode("CLOCK POP 0 ORACLE 0 PROPHECY");
        let error =
            BytecodeTransitionAnalyzer::analyze(&clock, ProgramGraphConfig::default()).unwrap_err();
        assert!(matches!(
            error,
            TransitionGraphError::UndefinedTransition { state: 0, ref reason }
                if reason.contains("frozen CLOCK input exhausted")
        ));

        let file = bytecode("\"proof-missing\" FILE_EXISTS POP 0 ORACLE 0 PROPHECY");
        let error =
            BytecodeTransitionAnalyzer::analyze(&file, ProgramGraphConfig::default()).unwrap_err();
        assert!(matches!(
            error,
            TransitionGraphError::UndefinedTransition { state: 0, ref reason }
                if reason.contains("file read capability denied")
        ));
    }

    #[test]
    fn analysis_rejects_a_domain_that_is_not_closed() {
        let program = parse("0 ORACLE 1 ADD 0 PROPHECY").unwrap();
        let error = ProgramTransitionAnalyzer::analyze(
            &program,
            ProgramGraphConfig {
                cell_bits: 2,
                ..ProgramGraphConfig::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            TransitionGraphError::DomainNotClosed { state: 3, .. }
        ));
    }

    #[test]
    fn state_limit_prevents_accidental_exponential_enumeration() {
        let program = parse("0 ORACLE 0 PROPHECY").unwrap();
        let error = ProgramTransitionAnalyzer::analyze(
            &program,
            ProgramGraphConfig {
                memory_cells: 3,
                cell_bits: 8,
                max_states: 100,
                ..ProgramGraphConfig::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            TransitionGraphError::DomainTooLarge { exponent: 24, .. }
        ));
    }

    #[test]
    fn hard_state_ceiling_rejects_untrusted_caller_limits_before_allocation() {
        let error = BytecodeTransitionAnalyzer::analyze(
            &bytecode("0 ORACLE 0 PROPHECY"),
            ProgramGraphConfig {
                max_states: usize::MAX,
                ..ProgramGraphConfig::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            TransitionGraphError::InvalidDomain(ref reason)
                if reason.contains("hard ceiling")
        ));
    }

    #[test]
    fn dot_export_contains_every_state_and_transition() {
        let program = parse("0 ORACLE NOT 0 PROPHECY").unwrap();
        let analysis =
            ProgramTransitionAnalyzer::analyze(&program, ProgramGraphConfig::default()).unwrap();
        let dot = analysis.to_dot();
        assert!(dot.contains("n0 ["));
        assert!(dot.contains("n1 ["));
        assert!(dot.contains("n0 -> n1"));
        assert!(dot.contains("n1 -> n0"));
        assert!(dot.contains("shape=doublecircle"));
    }

    #[test]
    fn unanimous_output_ignores_value_provenance() {
        let graph = DeterministicTransitionGraph::new(vec![1, 0]).unwrap();
        let analysis = ProgramGraphAnalysis {
            recurrent: graph.analyze(),
            graph,
            outputs: vec![
                vec![OutputItem::Val(Value::new(7)), OutputItem::Char(b'x')],
                vec![
                    OutputItem::Val(Value::with_provenance(7, Provenance::single(0))),
                    OutputItem::Char(b'x'),
                ],
            ],
            memory_cells: 1,
            cell_bits: 1,
        };

        assert!(analysis.class_has_unanimous_output(0));
    }
}
