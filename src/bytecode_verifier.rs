//! Adversarial stack and control-flow verification for flat bytecode.
//!
//! [`BytecodeProgram::validate`](crate::bytecode::BytecodeProgram::validate)
//! proves artifact structure.  This module deliberately performs a separate
//! data-flow proof: every reachable control-flow join must agree on relative
//! stack height and active temporal scopes, and every statically fixed
//! primitive must have enough input words.  Value-dependent operations are
//! never assigned a fabricated stack effect.  They produce explicit
//! [`VerificationObligation`] records and stop the proof on that path.

use crate::ast::{OpCode, QuoteId};
use crate::bytecode::{BytecodeError, BytecodeProgram, CodeRange, Instruction};
use crate::hir::ProcedureId;
use crate::source::SourceSpan;
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::error::Error;
use std::fmt;

/// Default upper bound on individual CFG work-list transitions.
pub const DEFAULT_MAX_CONTROL_FLOW_STEPS: usize = 4_000_000;
/// Default upper bound on call-graph and SCC traversal steps.
pub const DEFAULT_MAX_CALL_GRAPH_STEPS: usize = 12_000_000;

/// Deterministic denial-of-service bounds for bytecode verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerificationLimits {
    /// Maximum number of instruction visits and CFG edge checks.
    pub max_control_flow_steps: usize,
    /// Maximum number of call-graph scans, DFS edges, and component edges.
    pub max_call_graph_steps: usize,
}

impl Default for VerificationLimits {
    fn default() -> Self {
        Self {
            max_control_flow_steps: DEFAULT_MAX_CONTROL_FLOW_STEPS,
            max_call_graph_steps: DEFAULT_MAX_CALL_GRAPH_STEPS,
        }
    }
}

/// Typed owner of an independently verified bytecode unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BytecodeUnit {
    /// Top-level program body.
    Main,
    /// Anonymous quotation entry point.
    Quote(QuoteId),
    /// Named procedure entry point.
    Procedure(ProcedureId),
}

/// Exact bytecode location attached to a verifier finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerificationSite {
    /// Independently verified owner.
    pub unit: BytecodeUnit,
    /// Flat instruction index.
    pub instruction: u32,
    /// Source location when the artifact retained one for this instruction.
    pub source: Option<SourceSpan>,
}

/// Whether the complete unit has a static stack/control proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitProofStatus {
    /// Every reachable continuation was proved.
    Proven,
    /// At least one path stopped at an explicit runtime/link obligation.
    RuntimeValidationRequired,
}

/// Verified symbolic stack effect for one typed entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitVerification {
    /// Typed unit identity.
    pub unit: BytecodeUnit,
    /// Flat code occupied by the unit.
    pub range: CodeRange,
    /// Minimum number of words needed at unit entry on proved prefixes.
    pub minimum_entry_height: usize,
    /// Common relative stack delta at every reachable `RETURN`.
    ///
    /// This is `None` when no return is reachable or a deferred path prevents
    /// a complete return proof.
    pub return_delta: Option<i64>,
    /// Whether at least one reachable path returns to its caller.
    pub can_return: bool,
    /// Whether a fixed terminal primitive can end execution from this unit.
    pub may_terminate: bool,
    /// Number of instructions reached by the verifier work list.
    pub reachable_instructions: usize,
    /// Complete static proof status.
    pub status: UnitProofStatus,
}

impl UnitVerification {
    /// Returns whether this summary is safe to substitute at a typed call.
    pub const fn is_proven(&self) -> bool {
        matches!(self.status, UnitProofStatus::Proven)
    }
}

/// Runtime or linker fact intentionally outside the static bytecode proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationObligationKind {
    /// A primitive has a value-dependent stack row.
    DynamicPrimitive {
        /// Primitive whose complete effect is not statically fixed.
        opcode: OpCode,
        /// Operand words definitely required before dynamic behavior begins.
        minimum_inputs: usize,
    },
    /// A typed procedure call did not have a proven callee summary.
    UnverifiedProcedureCall {
        /// Typed procedure target.
        target: ProcedureId,
    },
    /// A strongly connected procedure component requires a runtime recursion
    /// discipline and cannot be summarized by this finite proof.
    RecursiveProcedureCycle {
        /// Exact SCC members in typed identity order.
        members: Vec<ProcedureId>,
    },
    /// A terminal operation can bypass lexical temporal exits.  A runtime may
    /// discharge this only by specifying and implementing scope unwinding.
    TerminalTemporalUnwind {
        /// Number of active scopes requiring unwinding.
        depth: usize,
    },
}

/// One explicit boundary of the static proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationObligation {
    /// Location of the boundary.  SCC obligations are attached to the first
    /// member's entry instruction.
    pub site: VerificationSite,
    /// Fact required before execution can be called fully verified.
    pub kind: VerificationObligationKind,
}

/// Successful verifier report, including honest gaps in its proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BytecodeVerificationReport {
    /// Main entry-point verification.
    pub main: UnitVerification,
    /// Quotation verifications in typed identity order.
    pub quotations: Vec<UnitVerification>,
    /// Procedure verifications in typed identity order.
    pub procedures: Vec<UnitVerification>,
    /// All deferred facts in deterministic unit/instruction order.
    pub obligations: Vec<VerificationObligation>,
    /// CFG work charged against [`VerificationLimits`].
    pub control_flow_steps: usize,
    /// Call-graph/SCC work charged against [`VerificationLimits`].
    pub call_graph_steps: usize,
}

impl BytecodeVerificationReport {
    /// True only when every typed entry point and the empty main entry row are
    /// completely proved with no deferred dynamic, recursion, unwind, or link
    /// fact.
    pub fn is_fully_verified(&self) -> bool {
        self.main.minimum_entry_height == 0
            && self.main.is_proven()
            && self.quotations.iter().all(UnitVerification::is_proven)
            && self.procedures.iter().all(UnitVerification::is_proven)
            && self.obligations.is_empty()
    }
}

/// Static contradiction or deterministic verifier resource failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BytecodeVerificationError {
    /// Structural artifact validation failed before semantic verification.
    Structural(BytecodeError),
    /// Top-level execution starts with an empty stack but needs input words.
    MainEntryUnderflow {
        /// Minimum missing word count.
        missing: usize,
        /// Instruction that established the maximum requirement.
        site: VerificationSite,
    },
    /// Two CFG predecessors disagree on the relative stack row at a join.
    StackJoinMismatch {
        /// Join instruction.
        site: VerificationSite,
        /// Previously established relative height.
        established: i64,
        /// New predecessor's relative height.
        incoming: i64,
        /// Instruction supplying the conflicting edge.
        incoming_from: u32,
    },
    /// Two CFG predecessors disagree on the active temporal-scope stack.
    TemporalJoinMismatch {
        /// Join instruction.
        site: VerificationSite,
        /// Previously established open-scope enter instructions.
        established_enters: Vec<u32>,
        /// Incoming open-scope enter instructions.
        incoming_enters: Vec<u32>,
        /// Instruction supplying the conflicting edge.
        incoming_from: u32,
    },
    /// A reachable temporal exit does not close the innermost active scope.
    TemporalExitMismatch {
        /// Exit instruction.
        site: VerificationSite,
        /// Enter instruction named by the bytecode.
        encoded_enter: u32,
        /// Actual innermost enter instruction, if a scope is active.
        active_enter: Option<u32>,
    },
    /// A normal return leaves temporal scopes active.
    TemporalScopeLeak {
        /// Return instruction.
        site: VerificationSite,
        /// Still-active temporal enter instructions.
        active_enters: Vec<u32>,
    },
    /// A relative stack calculation exceeded its representation.
    StackArithmeticOverflow {
        /// Instruction whose effect overflowed.
        site: VerificationSite,
    },
    /// A deterministic verifier work limit was exhausted.
    WorkLimitExceeded {
        /// Verifier phase being bounded.
        phase: &'static str,
        /// Configured maximum steps.
        limit: usize,
    },
    /// A post-validation invariant was unavailable without indexing or panic.
    InternalInvariant {
        /// Precise invariant description.
        message: String,
    },
}

impl fmt::Display for BytecodeVerificationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Structural(error) => write!(formatter, "structural bytecode error: {error}"),
            Self::MainEntryUnderflow { missing, site } => write!(
                formatter,
                "main entry needs {missing} stack word(s) at instruction {}",
                site.instruction
            ),
            Self::StackJoinMismatch {
                site,
                established,
                incoming,
                incoming_from,
            } => write!(
                formatter,
                "stack-height mismatch at instruction {}: established {established}, incoming {incoming} from {incoming_from}",
                site.instruction
            ),
            Self::TemporalJoinMismatch {
                site,
                incoming_from,
                ..
            } => write!(
                formatter,
                "temporal-scope mismatch at instruction {} from predecessor {incoming_from}",
                site.instruction
            ),
            Self::TemporalExitMismatch {
                site,
                encoded_enter,
                active_enter,
            } => write!(
                formatter,
                "temporal exit {} names enter {encoded_enter}, but active enter is {active_enter:?}",
                site.instruction
            ),
            Self::TemporalScopeLeak {
                site,
                active_enters,
            } => write!(
                formatter,
                "return {} leaves temporal scopes {active_enters:?} active",
                site.instruction
            ),
            Self::StackArithmeticOverflow { site } => write!(
                formatter,
                "stack arithmetic overflow at instruction {}",
                site.instruction
            ),
            Self::WorkLimitExceeded { phase, limit } => {
                write!(formatter, "{phase} verifier work exceeds limit {limit}")
            }
            Self::InternalInvariant { message } => {
                write!(formatter, "bytecode verifier invariant failed: {message}")
            }
        }
    }
}

impl Error for BytecodeVerificationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Structural(error) => Some(error),
            _ => None,
        }
    }
}

impl From<BytecodeError> for BytecodeVerificationError {
    fn from(value: BytecodeError) -> Self {
        Self::Structural(value)
    }
}

/// Independently verify bytecode with conservative default limits.
pub fn verify(
    program: &BytecodeProgram,
) -> Result<BytecodeVerificationReport, BytecodeVerificationError> {
    verify_with_limits(program, VerificationLimits::default())
}

/// Independently verify bytecode with caller-selected denial-of-service
/// limits.
pub fn verify_with_limits(
    program: &BytecodeProgram,
    limits: VerificationLimits,
) -> Result<BytecodeVerificationReport, BytecodeVerificationError> {
    program.validate()?;
    Verifier::new(program, limits).verify()
}

/// Compatibility alias spelling out that the default limits are used.
pub fn verify_default(
    program: &BytecodeProgram,
) -> Result<BytecodeVerificationReport, BytecodeVerificationError> {
    verify(program)
}

#[derive(Debug, Clone, Copy)]
enum OpcodeShape {
    Fixed {
        inputs: usize,
        outputs: usize,
        terminal: bool,
    },
    Dynamic {
        minimum_inputs: usize,
    },
}

/// Exhaustive verifier-local classification.  This cannot reuse the HIR
/// checker classifier because that function is intentionally private.
fn opcode_shape(opcode: OpCode) -> OpcodeShape {
    use OpCode::*;
    match opcode {
        Pick | Roll | Reverse | Exec | Rec | StrRev | BufferFromStack | BufferToStack
        | FileExists | FileSize | ProcExec | FFICall | FFICallNamed => {
            OpcodeShape::Dynamic { minimum_inputs: 1 }
        }
        Dip | Keep | StrCat | StrSplit | Assert | Pack | Unpack | FileOpen | TcpConnect => {
            OpcodeShape::Dynamic { minimum_inputs: 2 }
        }
        Bi => OpcodeShape::Dynamic { minimum_inputs: 3 },

        Nop => fixed(0, 0),
        Halt | Paradox => terminal(0, 0),
        Pop | Output | Emit | FileClose | BufferFree | SocketClose | Sleep => fixed(1, 0),
        Dup => fixed(1, 2),
        Swap => fixed(2, 2),
        Over => fixed(2, 3),
        Rot => fixed(3, 3),
        Depth | Input | VecNew | HashNew | SetNew | Clock | Random => fixed(0, 1),
        Neg | Abs | Sign | Not | Oracle | PresentRead | FileFlush | BufferNew => fixed(1, 1),
        Add | Sub | Mul | Div | Mod | Min | Max | And | Or | Xor | Shl | Shr | Eq | Neq | Lt
        | Gt | Lte | Gte | Slt | Sgt | Slte | Sgte | Index => fixed(2, 1),
        Prophecy => fixed(2, 0),
        Store => fixed(3, 0),
        VecPush | HashDel | SetAdd | SetDel | BufferWriteByte => fixed(2, 1),
        VecPop | VecLen | HashHas | SetHas | SetLen | BufferLen | BufferReadByte => fixed(1, 2),
        VecGet | FileWrite | SocketSend => fixed(2, 2),
        VecSet | HashPut => fixed(3, 1),
        HashGet | FileRead | SocketRecv => fixed(2, 3),
        HashLen => fixed(1, 2),
        FileSeek => fixed(3, 2),
    }
}

const fn fixed(inputs: usize, outputs: usize) -> OpcodeShape {
    OpcodeShape::Fixed {
        inputs,
        outputs,
        terminal: false,
    }
}

const fn terminal(inputs: usize, outputs: usize) -> OpcodeShape {
    OpcodeShape::Fixed {
        inputs,
        outputs,
        terminal: true,
    }
}

struct Verifier<'a> {
    program: &'a BytecodeProgram,
    limits: VerificationLimits,
    control_flow_steps: usize,
    call_graph_steps: usize,
}

impl<'a> Verifier<'a> {
    fn new(program: &'a BytecodeProgram, limits: VerificationLimits) -> Self {
        Self {
            program,
            limits,
            control_flow_steps: 0,
            call_graph_steps: 0,
        }
    }

    fn verify(mut self) -> Result<BytecodeVerificationReport, BytecodeVerificationError> {
        let adjacency = self.procedure_adjacency()?;
        let components = self.strongly_connected_components(&adjacency)?;
        let component_of = component_index(adjacency.len(), &components)?;
        let component_order =
            self.component_dependency_order(&adjacency, &components, &component_of)?;

        let mut procedure_proofs = vec![None::<UnitVerification>; self.program.procedures.len()];
        let mut procedure_results = vec![None::<Analysis>; self.program.procedures.len()];
        let mut cycle_obligations = Vec::new();

        for component in component_order {
            let members = components.get(component).ok_or_else(|| {
                BytecodeVerificationError::InternalInvariant {
                    message: format!("missing procedure component {component}"),
                }
            })?;
            let cyclic = members.len() > 1
                || members.first().is_some_and(|member| {
                    adjacency
                        .get(*member)
                        .is_some_and(|edges| edges.binary_search(member).is_ok())
                });
            if cyclic {
                let ids = members
                    .iter()
                    .map(|index| self.procedure_id(*index))
                    .collect::<Result<Vec<_>, _>>()?;
                let first = *members.first().ok_or_else(|| {
                    BytecodeVerificationError::InternalInvariant {
                        message: "empty SCC".to_string(),
                    }
                })?;
                let entry = self.procedure_entry(first)?;
                cycle_obligations.push(VerificationObligation {
                    site: self.site(BytecodeUnit::Procedure(entry.id), entry.range.start),
                    kind: VerificationObligationKind::RecursiveProcedureCycle { members: ids },
                });
            }

            for &index in members {
                let entry = self.procedure_entry(index)?;
                let analysis = self.analyze_unit(
                    BytecodeUnit::Procedure(entry.id),
                    entry.range,
                    &procedure_proofs,
                )?;
                if !cyclic && analysis.verification.is_proven() {
                    procedure_proofs[index] = Some(analysis.verification.clone());
                }
                procedure_results[index] = Some(analysis);
            }
        }

        let mut obligations = cycle_obligations;
        let mut procedures = Vec::with_capacity(procedure_results.len());
        for (index, result) in procedure_results.into_iter().enumerate() {
            let result = result.ok_or_else(|| BytecodeVerificationError::InternalInvariant {
                message: format!("procedure {index} was not analyzed"),
            })?;
            procedures.push(result.verification);
            obligations.extend(result.obligations);
        }

        let mut quotations = Vec::with_capacity(self.program.quotations.len());
        for entry in &self.program.quotations {
            let analysis = self.analyze_unit(
                BytecodeUnit::Quote(entry.id),
                entry.range,
                &procedure_proofs,
            )?;
            quotations.push(analysis.verification);
            obligations.extend(analysis.obligations);
        }

        let main_analysis =
            self.analyze_unit(BytecodeUnit::Main, self.program.main, &procedure_proofs)?;
        if main_analysis.verification.minimum_entry_height != 0 {
            return Err(BytecodeVerificationError::MainEntryUnderflow {
                missing: main_analysis.verification.minimum_entry_height,
                site: main_analysis.required_site,
            });
        }
        obligations.extend(main_analysis.obligations);

        obligations.sort_by_key(|obligation| {
            (
                obligation.site.unit,
                obligation.site.instruction,
                obligation_kind_order(&obligation.kind),
            )
        });

        Ok(BytecodeVerificationReport {
            main: main_analysis.verification,
            quotations,
            procedures,
            obligations,
            control_flow_steps: self.control_flow_steps,
            call_graph_steps: self.call_graph_steps,
        })
    }

    fn procedure_adjacency(&mut self) -> Result<Vec<Vec<usize>>, BytecodeVerificationError> {
        let mut adjacency = vec![Vec::new(); self.program.procedures.len()];
        for (index, entry) in self.program.procedures.iter().enumerate() {
            let state_count =
                usize::try_from(entry.range.end - entry.range.start).map_err(|_| {
                    BytecodeVerificationError::InternalInvariant {
                        message: format!(
                            "procedure range {}..{} does not fit usize",
                            entry.range.start, entry.range.end
                        ),
                    }
                })?;
            let mut visited = vec![false; state_count];
            let mut work = VecDeque::from([entry.range.start]);
            let mut edges = Vec::new();
            while let Some(pc) = work.pop_front() {
                self.charge_call_graph()?;
                let offset = self.unit_index(entry.range, pc)?;
                let seen = visited.get_mut(offset).ok_or_else(|| {
                    BytecodeVerificationError::InternalInvariant {
                        message: format!("missing call-graph reachability state for {pc}"),
                    }
                })?;
                if *seen {
                    continue;
                }
                *seen = true;
                match *self.instruction(pc)? {
                    Instruction::Primitive(opcode)
                        if matches!(
                            opcode_shape(opcode),
                            OpcodeShape::Fixed { terminal: true, .. }
                        ) => {}
                    Instruction::Return => {}
                    Instruction::CallProcedure(target) => {
                        edges.push(target.index());
                        enqueue_call_graph_fallthrough(entry.range, pc, &mut work)?;
                    }
                    Instruction::IfFalse { else_target, .. }
                    | Instruction::WhileFalse {
                        end_target: else_target,
                        ..
                    } => {
                        enqueue_call_graph_fallthrough(entry.range, pc, &mut work)?;
                        enqueue_call_graph_target(entry.range, else_target, &mut work)?;
                    }
                    Instruction::Jump { target } | Instruction::LoopBack { target } => {
                        enqueue_call_graph_target(entry.range, target, &mut work)?;
                    }
                    _ => enqueue_call_graph_fallthrough(entry.range, pc, &mut work)?,
                }
            }
            edges.sort_unstable();
            edges.dedup();
            *adjacency.get_mut(index).ok_or_else(|| {
                BytecodeVerificationError::InternalInvariant {
                    message: format!("missing adjacency row {index}"),
                }
            })? = edges;
        }
        Ok(adjacency)
    }

    fn strongly_connected_components(
        &mut self,
        adjacency: &[Vec<usize>],
    ) -> Result<Vec<Vec<usize>>, BytecodeVerificationError> {
        let mut visited = vec![false; adjacency.len()];
        let mut finish = Vec::with_capacity(adjacency.len());
        for start in 0..adjacency.len() {
            if visited[start] {
                continue;
            }
            visited[start] = true;
            let mut stack = vec![(start, 0usize)];
            while let Some((node, edge_index)) = stack.last_mut() {
                self.charge_call_graph()?;
                let edges = adjacency.get(*node).ok_or_else(|| {
                    BytecodeVerificationError::InternalInvariant {
                        message: format!("missing call graph node {node}"),
                    }
                })?;
                if *edge_index < edges.len() {
                    let next = edges[*edge_index];
                    *edge_index += 1;
                    let seen = visited.get_mut(next).ok_or_else(|| {
                        BytecodeVerificationError::InternalInvariant {
                            message: format!("call edge references procedure {next}"),
                        }
                    })?;
                    if !*seen {
                        *seen = true;
                        stack.push((next, 0));
                    }
                } else {
                    let (finished, _) = stack.pop().ok_or_else(|| {
                        BytecodeVerificationError::InternalInvariant {
                            message: "DFS stack unexpectedly empty".to_string(),
                        }
                    })?;
                    finish.push(finished);
                }
            }
        }

        let mut reverse = vec![Vec::new(); adjacency.len()];
        for (from, edges) in adjacency.iter().enumerate() {
            for &to in edges {
                self.charge_call_graph()?;
                reverse
                    .get_mut(to)
                    .ok_or_else(|| BytecodeVerificationError::InternalInvariant {
                        message: format!("reverse edge references procedure {to}"),
                    })?
                    .push(from);
            }
        }
        for edges in &mut reverse {
            edges.sort_unstable();
            edges.dedup();
        }

        let mut assigned = vec![false; adjacency.len()];
        let mut components = Vec::new();
        for &start in finish.iter().rev() {
            if assigned[start] {
                continue;
            }
            assigned[start] = true;
            let mut stack = vec![start];
            let mut component = Vec::new();
            while let Some(node) = stack.pop() {
                self.charge_call_graph()?;
                component.push(node);
                let edges = reverse.get(node).ok_or_else(|| {
                    BytecodeVerificationError::InternalInvariant {
                        message: format!("missing reverse node {node}"),
                    }
                })?;
                for &next in edges.iter().rev() {
                    let seen = assigned.get_mut(next).ok_or_else(|| {
                        BytecodeVerificationError::InternalInvariant {
                            message: format!("reverse edge references procedure {next}"),
                        }
                    })?;
                    if !*seen {
                        *seen = true;
                        stack.push(next);
                    }
                }
            }
            component.sort_unstable();
            components.push(component);
        }
        components.sort_by_key(|component| component.first().copied().unwrap_or(usize::MAX));
        Ok(components)
    }

    fn component_dependency_order(
        &mut self,
        adjacency: &[Vec<usize>],
        components: &[Vec<usize>],
        component_of: &[usize],
    ) -> Result<Vec<usize>, BytecodeVerificationError> {
        let mut dependencies = vec![BTreeSet::new(); components.len()];
        let mut callers = vec![BTreeSet::new(); components.len()];
        for (from, edges) in adjacency.iter().enumerate() {
            let from_component = *component_of.get(from).ok_or_else(|| {
                BytecodeVerificationError::InternalInvariant {
                    message: format!("procedure {from} has no component"),
                }
            })?;
            for &to in edges {
                self.charge_call_graph()?;
                let to_component = *component_of.get(to).ok_or_else(|| {
                    BytecodeVerificationError::InternalInvariant {
                        message: format!("procedure {to} has no component"),
                    }
                })?;
                if from_component != to_component
                    && dependencies[from_component].insert(to_component)
                {
                    callers[to_component].insert(from_component);
                }
            }
        }

        let mut pending = dependencies.iter().map(BTreeSet::len).collect::<Vec<_>>();
        let mut ready = pending
            .iter()
            .enumerate()
            .filter_map(|(index, count)| (*count == 0).then_some(index))
            .collect::<BTreeSet<_>>();
        let mut order = Vec::with_capacity(components.len());
        while let Some(component) = ready.pop_first() {
            self.charge_call_graph()?;
            order.push(component);
            for &caller in callers.get(component).ok_or_else(|| {
                BytecodeVerificationError::InternalInvariant {
                    message: format!("missing component caller row {component}"),
                }
            })? {
                let count = pending.get_mut(caller).ok_or_else(|| {
                    BytecodeVerificationError::InternalInvariant {
                        message: format!("missing pending component {caller}"),
                    }
                })?;
                *count = count.checked_sub(1).ok_or_else(|| {
                    BytecodeVerificationError::InternalInvariant {
                        message: format!("component {caller} dependency underflow"),
                    }
                })?;
                if *count == 0 {
                    ready.insert(caller);
                }
            }
        }
        if order.len() != components.len() {
            return Err(BytecodeVerificationError::InternalInvariant {
                message: "procedure component graph is cyclic".to_string(),
            });
        }
        Ok(order)
    }

    fn analyze_unit(
        &mut self,
        unit: BytecodeUnit,
        range: CodeRange,
        procedure_proofs: &[Option<UnitVerification>],
    ) -> Result<Analysis, BytecodeVerificationError> {
        let state_count = usize::try_from(range.end - range.start).map_err(|_| {
            BytecodeVerificationError::InternalInvariant {
                message: format!(
                    "unit range {}..{} does not fit usize",
                    range.start, range.end
                ),
            }
        })?;
        let mut states = vec![None::<FlowState>; state_count];
        let mut work = VecDeque::new();
        let mut scopes = ScopeArena::new();
        self.enqueue(
            unit,
            range,
            range.start,
            FlowState { delta: 0, scope: 0 },
            range.start,
            &mut states,
            &mut work,
            &scopes,
        )?;

        let mut required = 0usize;
        let mut required_site = self.site(unit, range.start);
        let mut return_delta = None::<i64>;
        let mut can_return = false;
        let mut may_terminate = false;
        let mut reachable = 0usize;
        let mut obligations = Vec::new();

        while let Some(pc) = work.pop_front() {
            self.charge_control_flow()?;
            reachable = reachable.checked_add(1).ok_or_else(|| {
                BytecodeVerificationError::InternalInvariant {
                    message: "reachable instruction counter overflow".to_string(),
                }
            })?;
            let index = self.unit_index(range, pc)?;
            let state = states.get(index).and_then(|state| *state).ok_or_else(|| {
                BytecodeVerificationError::InternalInvariant {
                    message: format!("instruction {pc} was queued without a state"),
                }
            })?;
            let instruction = *self.instruction(pc)?;

            match instruction {
                Instruction::Primitive(opcode) => match opcode_shape(opcode) {
                    OpcodeShape::Dynamic { minimum_inputs } => {
                        self.record_requirement(
                            unit,
                            pc,
                            state.delta,
                            minimum_inputs,
                            &mut required,
                            &mut required_site,
                        )?;
                        obligations.push(VerificationObligation {
                            site: self.site(unit, pc),
                            kind: VerificationObligationKind::DynamicPrimitive {
                                opcode,
                                minimum_inputs,
                            },
                        });
                    }
                    OpcodeShape::Fixed {
                        inputs,
                        outputs,
                        terminal,
                    } => {
                        let next = self.apply_effect(
                            unit,
                            pc,
                            state,
                            inputs,
                            outputs,
                            &mut required,
                            &mut required_site,
                        )?;
                        if terminal {
                            may_terminate = true;
                            let depth = scopes.depth(next.scope)?;
                            if depth != 0 {
                                obligations.push(VerificationObligation {
                                    site: self.site(unit, pc),
                                    kind: VerificationObligationKind::TerminalTemporalUnwind {
                                        depth,
                                    },
                                });
                            }
                        } else {
                            self.enqueue_fallthrough(
                                unit,
                                range,
                                pc,
                                next,
                                &mut states,
                                &mut work,
                                &scopes,
                            )?;
                        }
                    }
                },
                Instruction::PushWord(_) | Instruction::PushQuote(_) => {
                    let next = self.apply_effect(
                        unit,
                        pc,
                        state,
                        0,
                        1,
                        &mut required,
                        &mut required_site,
                    )?;
                    self.enqueue_fallthrough(
                        unit,
                        range,
                        pc,
                        next,
                        &mut states,
                        &mut work,
                        &scopes,
                    )?;
                }
                Instruction::CallProcedure(target) => {
                    let proof = procedure_proofs
                        .get(target.index())
                        .and_then(Option::as_ref);
                    let Some(proof) = proof else {
                        obligations.push(VerificationObligation {
                            site: self.site(unit, pc),
                            kind: VerificationObligationKind::UnverifiedProcedureCall { target },
                        });
                        continue;
                    };
                    self.record_requirement(
                        unit,
                        pc,
                        state.delta,
                        proof.minimum_entry_height,
                        &mut required,
                        &mut required_site,
                    )?;
                    if proof.may_terminate {
                        may_terminate = true;
                        let depth = scopes.depth(state.scope)?;
                        if depth != 0 {
                            obligations.push(VerificationObligation {
                                site: self.site(unit, pc),
                                kind: VerificationObligationKind::TerminalTemporalUnwind { depth },
                            });
                        }
                    }
                    if proof.can_return {
                        let delta = proof.return_delta.ok_or_else(|| {
                            BytecodeVerificationError::InternalInvariant {
                                message: format!(
                                    "returning procedure {} lacks a return delta",
                                    target.as_u32()
                                ),
                            }
                        })?;
                        let next_delta = state.delta.checked_add(delta).ok_or_else(|| {
                            BytecodeVerificationError::StackArithmeticOverflow {
                                site: self.site(unit, pc),
                            }
                        })?;
                        self.enqueue_fallthrough(
                            unit,
                            range,
                            pc,
                            FlowState {
                                delta: next_delta,
                                scope: state.scope,
                            },
                            &mut states,
                            &mut work,
                            &scopes,
                        )?;
                    }
                }
                Instruction::CallForeign(target) => {
                    let signature = &self.program.foreigns[target.index()];
                    let next = self.apply_effect(
                        unit,
                        pc,
                        state,
                        signature.parameters.len(),
                        usize::from(signature.result.is_some()),
                        &mut required,
                        &mut required_site,
                    )?;
                    self.enqueue_fallthrough(
                        unit,
                        range,
                        pc,
                        next,
                        &mut states,
                        &mut work,
                        &scopes,
                    )?;
                }
                Instruction::IfFalse { else_target, .. }
                | Instruction::WhileFalse {
                    end_target: else_target,
                    ..
                } => {
                    let next = self.apply_effect(
                        unit,
                        pc,
                        state,
                        1,
                        0,
                        &mut required,
                        &mut required_site,
                    )?;
                    self.enqueue_fallthrough(
                        unit,
                        range,
                        pc,
                        next,
                        &mut states,
                        &mut work,
                        &scopes,
                    )?;
                    self.enqueue(
                        unit,
                        range,
                        else_target,
                        next,
                        pc,
                        &mut states,
                        &mut work,
                        &scopes,
                    )?;
                }
                Instruction::Jump { target } | Instruction::LoopBack { target } => {
                    self.enqueue(
                        unit,
                        range,
                        target,
                        state,
                        pc,
                        &mut states,
                        &mut work,
                        &scopes,
                    )?;
                }
                Instruction::TemporalEnter { .. } => {
                    let scope = scopes.enter(state.scope, pc)?;
                    self.enqueue_fallthrough(
                        unit,
                        range,
                        pc,
                        FlowState {
                            delta: state.delta,
                            scope,
                        },
                        &mut states,
                        &mut work,
                        &scopes,
                    )?;
                }
                Instruction::TemporalExit { enter_target } => {
                    let active = scopes.enter_pc(state.scope)?;
                    if active != Some(enter_target) {
                        return Err(BytecodeVerificationError::TemporalExitMismatch {
                            site: self.site(unit, pc),
                            encoded_enter: enter_target,
                            active_enter: active,
                        });
                    }
                    let parent = scopes.parent(state.scope)?;
                    self.enqueue_fallthrough(
                        unit,
                        range,
                        pc,
                        FlowState {
                            delta: state.delta,
                            scope: parent,
                        },
                        &mut states,
                        &mut work,
                        &scopes,
                    )?;
                }
                Instruction::Return => {
                    let active_enters = scopes.path(state.scope)?;
                    if !active_enters.is_empty() {
                        return Err(BytecodeVerificationError::TemporalScopeLeak {
                            site: self.site(unit, pc),
                            active_enters,
                        });
                    }
                    can_return = true;
                    match return_delta {
                        None => return_delta = Some(state.delta),
                        Some(established) if established == state.delta => {}
                        Some(established) => {
                            return Err(BytecodeVerificationError::StackJoinMismatch {
                                site: self.site(unit, pc),
                                established,
                                incoming: state.delta,
                                incoming_from: pc,
                            });
                        }
                    }
                }
            }
        }

        let status = if obligations.is_empty() {
            UnitProofStatus::Proven
        } else {
            UnitProofStatus::RuntimeValidationRequired
        };
        if !matches!(status, UnitProofStatus::Proven) {
            return_delta = None;
        }
        Ok(Analysis {
            verification: UnitVerification {
                unit,
                range,
                minimum_entry_height: required,
                return_delta,
                can_return,
                may_terminate,
                reachable_instructions: reachable,
                status,
            },
            obligations,
            required_site,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_effect(
        &self,
        unit: BytecodeUnit,
        pc: u32,
        state: FlowState,
        inputs: usize,
        outputs: usize,
        required: &mut usize,
        required_site: &mut VerificationSite,
    ) -> Result<FlowState, BytecodeVerificationError> {
        self.record_requirement(unit, pc, state.delta, inputs, required, required_site)?;
        let inputs = i64::try_from(inputs).map_err(|_| {
            BytecodeVerificationError::StackArithmeticOverflow {
                site: self.site(unit, pc),
            }
        })?;
        let outputs = i64::try_from(outputs).map_err(|_| {
            BytecodeVerificationError::StackArithmeticOverflow {
                site: self.site(unit, pc),
            }
        })?;
        let delta = state
            .delta
            .checked_sub(inputs)
            .and_then(|delta| delta.checked_add(outputs))
            .ok_or_else(|| BytecodeVerificationError::StackArithmeticOverflow {
                site: self.site(unit, pc),
            })?;
        Ok(FlowState { delta, ..state })
    }

    #[allow(clippy::too_many_arguments)]
    fn record_requirement(
        &self,
        unit: BytecodeUnit,
        pc: u32,
        delta: i64,
        inputs: usize,
        required: &mut usize,
        required_site: &mut VerificationSite,
    ) -> Result<(), BytecodeVerificationError> {
        let needed = (inputs as i128 - i128::from(delta)).max(0);
        let needed = usize::try_from(needed).map_err(|_| {
            BytecodeVerificationError::StackArithmeticOverflow {
                site: self.site(unit, pc),
            }
        })?;
        if needed > *required {
            *required = needed;
            *required_site = self.site(unit, pc);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_fallthrough(
        &mut self,
        unit: BytecodeUnit,
        range: CodeRange,
        pc: u32,
        state: FlowState,
        states: &mut [Option<FlowState>],
        work: &mut VecDeque<u32>,
        scopes: &ScopeArena,
    ) -> Result<(), BytecodeVerificationError> {
        let target =
            pc.checked_add(1)
                .ok_or_else(|| BytecodeVerificationError::InternalInvariant {
                    message: format!("fallthrough overflows after instruction {pc}"),
                })?;
        self.enqueue(unit, range, target, state, pc, states, work, scopes)
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue(
        &mut self,
        unit: BytecodeUnit,
        range: CodeRange,
        target: u32,
        incoming: FlowState,
        incoming_from: u32,
        states: &mut [Option<FlowState>],
        work: &mut VecDeque<u32>,
        scopes: &ScopeArena,
    ) -> Result<(), BytecodeVerificationError> {
        self.charge_control_flow()?;
        let index = self.unit_index(range, target)?;
        let slot =
            states
                .get_mut(index)
                .ok_or_else(|| BytecodeVerificationError::InternalInvariant {
                    message: format!("CFG target {target} is outside state table"),
                })?;
        match *slot {
            None => {
                *slot = Some(incoming);
                work.push_back(target);
            }
            Some(established) if established == incoming => {}
            Some(established) if established.delta != incoming.delta => {
                return Err(BytecodeVerificationError::StackJoinMismatch {
                    site: self.site(unit, target),
                    established: established.delta,
                    incoming: incoming.delta,
                    incoming_from,
                });
            }
            Some(established) => {
                return Err(BytecodeVerificationError::TemporalJoinMismatch {
                    site: self.site(unit, target),
                    established_enters: scopes.path(established.scope)?,
                    incoming_enters: scopes.path(incoming.scope)?,
                    incoming_from,
                });
            }
        }
        Ok(())
    }

    fn unit_index(&self, range: CodeRange, pc: u32) -> Result<usize, BytecodeVerificationError> {
        if pc < range.start || pc >= range.end {
            return Err(BytecodeVerificationError::InternalInvariant {
                message: format!(
                    "CFG target {pc} escapes unit {}..{}",
                    range.start, range.end
                ),
            });
        }
        usize::try_from(pc - range.start).map_err(|_| {
            BytecodeVerificationError::InternalInvariant {
                message: format!("instruction offset for {pc} does not fit usize"),
            }
        })
    }

    fn instruction(&self, pc: u32) -> Result<&Instruction, BytecodeVerificationError> {
        self.program.instructions.get(pc as usize).ok_or_else(|| {
            BytecodeVerificationError::InternalInvariant {
                message: format!("instruction {pc} disappeared after structural validation"),
            }
        })
    }

    fn procedure_entry(
        &self,
        index: usize,
    ) -> Result<crate::bytecode::ProcedureEntry, BytecodeVerificationError> {
        self.program.procedures.get(index).copied().ok_or_else(|| {
            BytecodeVerificationError::InternalInvariant {
                message: format!("missing procedure entry {index}"),
            }
        })
    }

    fn procedure_id(&self, index: usize) -> Result<ProcedureId, BytecodeVerificationError> {
        Ok(self.procedure_entry(index)?.id)
    }

    fn site(&self, unit: BytecodeUnit, instruction: u32) -> VerificationSite {
        let source = self
            .program
            .source_map
            .binary_search_by_key(&instruction, |entry| entry.instruction)
            .ok()
            .and_then(|index| self.program.source_map.get(index))
            .map(|entry| entry.span);
        VerificationSite {
            unit,
            instruction,
            source,
        }
    }

    fn charge_control_flow(&mut self) -> Result<(), BytecodeVerificationError> {
        charge(
            &mut self.control_flow_steps,
            self.limits.max_control_flow_steps,
            "control-flow",
        )
    }

    fn charge_call_graph(&mut self) -> Result<(), BytecodeVerificationError> {
        charge(
            &mut self.call_graph_steps,
            self.limits.max_call_graph_steps,
            "call-graph",
        )
    }
}

fn component_index(
    procedure_count: usize,
    components: &[Vec<usize>],
) -> Result<Vec<usize>, BytecodeVerificationError> {
    let mut result = vec![usize::MAX; procedure_count];
    for (component, members) in components.iter().enumerate() {
        for &member in members {
            let slot = result.get_mut(member).ok_or_else(|| {
                BytecodeVerificationError::InternalInvariant {
                    message: format!("SCC contains missing procedure {member}"),
                }
            })?;
            if *slot != usize::MAX {
                return Err(BytecodeVerificationError::InternalInvariant {
                    message: format!("procedure {member} appears in two SCCs"),
                });
            }
            *slot = component;
        }
    }
    if let Some(missing) = result.iter().position(|entry| *entry == usize::MAX) {
        return Err(BytecodeVerificationError::InternalInvariant {
            message: format!("procedure {missing} appears in no SCC"),
        });
    }
    Ok(result)
}

fn enqueue_call_graph_fallthrough(
    range: CodeRange,
    pc: u32,
    work: &mut VecDeque<u32>,
) -> Result<(), BytecodeVerificationError> {
    let target = pc
        .checked_add(1)
        .ok_or_else(|| BytecodeVerificationError::InternalInvariant {
            message: format!("call-graph fallthrough overflows after instruction {pc}"),
        })?;
    enqueue_call_graph_target(range, target, work)
}

fn enqueue_call_graph_target(
    range: CodeRange,
    target: u32,
    work: &mut VecDeque<u32>,
) -> Result<(), BytecodeVerificationError> {
    if target < range.start || target >= range.end {
        return Err(BytecodeVerificationError::InternalInvariant {
            message: format!(
                "call-graph target {target} escapes unit {}..{}",
                range.start, range.end
            ),
        });
    }
    work.push_back(target);
    Ok(())
}

fn obligation_kind_order(kind: &VerificationObligationKind) -> u8 {
    match kind {
        VerificationObligationKind::DynamicPrimitive { .. } => 0,
        VerificationObligationKind::UnverifiedProcedureCall { .. } => 2,
        VerificationObligationKind::RecursiveProcedureCycle { .. } => 3,
        VerificationObligationKind::TerminalTemporalUnwind { .. } => 4,
    }
}

fn charge(
    counter: &mut usize,
    limit: usize,
    phase: &'static str,
) -> Result<(), BytecodeVerificationError> {
    if *counter >= limit {
        return Err(BytecodeVerificationError::WorkLimitExceeded { phase, limit });
    }
    *counter = counter
        .checked_add(1)
        .ok_or(BytecodeVerificationError::WorkLimitExceeded { phase, limit })?;
    Ok(())
}

#[derive(Debug, Clone)]
struct Analysis {
    verification: UnitVerification,
    obligations: Vec<VerificationObligation>,
    required_site: VerificationSite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FlowState {
    delta: i64,
    scope: usize,
}

#[derive(Debug, Clone, Copy)]
struct ScopeNode {
    parent: usize,
    enter_pc: Option<u32>,
    depth: usize,
}

struct ScopeArena {
    nodes: Vec<ScopeNode>,
    interned: HashMap<(usize, u32), usize>,
}

impl ScopeArena {
    fn new() -> Self {
        Self {
            nodes: vec![ScopeNode {
                parent: 0,
                enter_pc: None,
                depth: 0,
            }],
            interned: HashMap::new(),
        }
    }

    fn enter(&mut self, parent: usize, enter_pc: u32) -> Result<usize, BytecodeVerificationError> {
        if let Some(existing) = self.interned.get(&(parent, enter_pc)) {
            return Ok(*existing);
        }
        let parent_depth = self
            .nodes
            .get(parent)
            .ok_or_else(|| BytecodeVerificationError::InternalInvariant {
                message: format!("missing temporal parent state {parent}"),
            })?
            .depth;
        let depth = parent_depth.checked_add(1).ok_or_else(|| {
            BytecodeVerificationError::InternalInvariant {
                message: "temporal scope depth overflow".to_string(),
            }
        })?;
        let id = self.nodes.len();
        self.nodes.push(ScopeNode {
            parent,
            enter_pc: Some(enter_pc),
            depth,
        });
        self.interned.insert((parent, enter_pc), id);
        Ok(id)
    }

    fn parent(&self, scope: usize) -> Result<usize, BytecodeVerificationError> {
        self.nodes
            .get(scope)
            .map(|node| node.parent)
            .ok_or_else(|| BytecodeVerificationError::InternalInvariant {
                message: format!("missing temporal scope state {scope}"),
            })
    }

    fn enter_pc(&self, scope: usize) -> Result<Option<u32>, BytecodeVerificationError> {
        self.nodes
            .get(scope)
            .map(|node| node.enter_pc)
            .ok_or_else(|| BytecodeVerificationError::InternalInvariant {
                message: format!("missing temporal scope state {scope}"),
            })
    }

    fn depth(&self, scope: usize) -> Result<usize, BytecodeVerificationError> {
        self.nodes.get(scope).map(|node| node.depth).ok_or_else(|| {
            BytecodeVerificationError::InternalInvariant {
                message: format!("missing temporal scope state {scope}"),
            }
        })
    }

    fn path(&self, mut scope: usize) -> Result<Vec<u32>, BytecodeVerificationError> {
        let mut reversed = Vec::new();
        loop {
            let node = self.nodes.get(scope).ok_or_else(|| {
                BytecodeVerificationError::InternalInvariant {
                    message: format!("missing temporal scope state {scope}"),
                }
            })?;
            let Some(enter_pc) = node.enter_pc else {
                break;
            };
            reversed.push(enter_pc);
            scope = node.parent;
        }
        reversed.reverse();
        Ok(reversed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Procedure, Program, Stmt};
    use crate::core::Value;
    use crate::hir::{ForeignId, HirProgram};

    fn procedure(name: &str, body: Vec<Stmt>) -> Procedure {
        Procedure {
            name: name.to_string(),
            params: Vec::new(),
            returns: 0,
            effects: Vec::new(),
            body,
        }
    }

    fn bytecode(program: &Program) -> BytecodeProgram {
        let hir = HirProgram::resolve(program).expect("resolve test program");
        BytecodeProgram::compile(&hir).expect("compile test program")
    }

    #[test]
    fn proves_fixed_stack_effects_from_an_empty_main_row() {
        let mut program = Program::new();
        program.body = vec![
            Stmt::Push(Value::new(2)),
            Stmt::Push(Value::new(3)),
            Stmt::Op(OpCode::Add),
        ];
        let report = verify_default(&bytecode(&program)).expect("verify");
        assert!(report.is_fully_verified());
        assert_eq!(report.main.minimum_entry_height, 0);
        assert_eq!(report.main.return_delta, Some(1));
    }

    #[test]
    fn rejects_definite_main_underflow() {
        let mut program = Program::new();
        program.body = vec![Stmt::Op(OpCode::Swap)];
        assert!(matches!(
            verify_default(&bytecode(&program)),
            Err(BytecodeVerificationError::MainEntryUnderflow { missing: 2, .. })
        ));
    }

    #[test]
    fn rejects_if_and_while_stack_row_drift() {
        let mut branch = Program::new();
        branch.body = vec![
            Stmt::Push(Value::ONE),
            Stmt::If {
                then_branch: vec![Stmt::Push(Value::ONE)],
                else_branch: Some(Vec::new()),
            },
        ];
        assert!(matches!(
            verify_default(&bytecode(&branch)),
            Err(BytecodeVerificationError::StackJoinMismatch { .. })
        ));

        let mut loop_drift = Program::new();
        loop_drift.body = vec![Stmt::While {
            cond: vec![Stmt::Push(Value::ONE)],
            body: vec![Stmt::Push(Value::ONE)],
        }];
        assert!(matches!(
            verify_default(&bytecode(&loop_drift)),
            Err(BytecodeVerificationError::StackJoinMismatch { .. })
        ));
    }

    #[test]
    fn substitutes_only_proven_typed_procedure_summaries() {
        let mut program = Program::new();
        program
            .procedures
            .push(procedure("copy", vec![Stmt::Op(OpCode::Dup)]));
        program.body = vec![
            Stmt::Push(Value::new(7)),
            Stmt::Call {
                name: "copy".to_string(),
            },
        ];
        let report = verify_default(&bytecode(&program)).expect("verify");
        assert!(report.is_fully_verified());
        assert_eq!(report.procedures[0].minimum_entry_height, 1);
        assert_eq!(report.procedures[0].return_delta, Some(1));
        assert_eq!(report.main.return_delta, Some(2));
    }

    #[test]
    fn independently_verifies_typed_quote_entries() {
        let mut program = Program::new();
        program.quotes.push(vec![Stmt::Op(OpCode::Dup)]);
        program.body = vec![Stmt::PushQuote(QuoteId::new(0))];
        let report = verify_default(&bytecode(&program)).expect("verify");
        assert!(report.is_fully_verified());
        assert_eq!(
            report.quotations[0].unit,
            BytecodeUnit::Quote(QuoteId::new(0))
        );
        assert_eq!(report.quotations[0].minimum_entry_height, 1);
        assert_eq!(report.quotations[0].return_delta, Some(1));
    }

    #[test]
    fn dynamic_primitive_is_an_obligation_not_a_proof() {
        let mut program = Program::new();
        program.quotes.push(Vec::new());
        program.body = vec![Stmt::PushQuote(QuoteId::new(0)), Stmt::Op(OpCode::Exec)];
        let report = verify_default(&bytecode(&program)).expect("verify with obligation");
        assert!(!report.is_fully_verified());
        assert!(matches!(
            report.obligations.as_slice(),
            [VerificationObligation {
                kind: VerificationObligationKind::DynamicPrimitive {
                    opcode: OpCode::Exec,
                    minimum_inputs: 1
                },
                ..
            }]
        ));
        assert_eq!(
            report.main.status,
            UnitProofStatus::RuntimeValidationRequired
        );

        let mut underflow = Program::new();
        underflow.body = vec![Stmt::Op(OpCode::Exec)];
        assert!(matches!(
            verify_default(&bytecode(&underflow)),
            Err(BytecodeVerificationError::MainEntryUnderflow { missing: 1, .. })
        ));
    }

    #[test]
    fn foreign_call_uses_the_retained_scalar_signature() {
        let target = ForeignId::try_from_index(0).expect("foreign test identity");
        let artifact = BytecodeProgram {
            instructions: vec![
                Instruction::PushWord(9),
                Instruction::CallForeign(target),
                Instruction::Return,
            ],
            main: CodeRange { start: 0, end: 3 },
            quotations: Vec::new(),
            procedures: Vec::new(),
            foreigns: vec![crate::bytecode::ForeignEntry {
                id: target,
                library: "host".into(),
                symbol: "foreign".into(),
                parameters: vec![crate::bytecode::ForeignScalarType::U64],
                result: Some(crate::bytecode::ForeignScalarType::U64),
                effects: crate::bytecode::ForeignEffects::from_bits(
                    crate::bytecode::ForeignEffects::IO,
                )
                .unwrap(),
            }],
            source_map: Vec::new(),
        };
        let report = verify_default(&artifact).expect("foreign signature is exact");
        assert!(report.obligations.is_empty());
        assert!(report.is_fully_verified());
        assert_eq!(report.main.return_delta, Some(1));
    }

    #[test]
    fn recursive_scc_is_reported_without_recursive_verifier_calls() {
        let mut program = Program::new();
        program.procedures = vec![
            procedure(
                "left",
                vec![Stmt::Call {
                    name: "right".to_string(),
                }],
            ),
            procedure(
                "right",
                vec![Stmt::Call {
                    name: "left".to_string(),
                }],
            ),
        ];
        let report = verify_default(&bytecode(&program)).expect("bounded cycle report");
        assert!(!report.is_fully_verified());
        assert!(report.obligations.iter().any(|obligation| matches!(
            &obligation.kind,
            VerificationObligationKind::RecursiveProcedureCycle { members }
                if members.iter().map(|id| id.index()).collect::<Vec<_>>() == vec![0, 1]
        )));
    }

    #[test]
    fn unreachable_call_after_terminal_does_not_fabricate_a_cycle() {
        let mut program = Program::new();
        program.procedures.push(procedure(
            "terminal",
            vec![
                Stmt::Op(OpCode::Halt),
                Stmt::Call {
                    name: "terminal".to_string(),
                },
            ],
        ));
        program.body = vec![Stmt::Call {
            name: "terminal".to_string(),
        }];
        let report = verify_default(&bytecode(&program)).expect("terminal proof");
        assert!(report.is_fully_verified());
        assert!(report.procedures[0].may_terminate);
        assert!(!report.procedures[0].can_return);
    }

    #[test]
    fn balanced_temporal_scope_is_proven_but_terminal_unwind_is_explicit() {
        let mut balanced = Program::new();
        balanced.body = vec![Stmt::TemporalScope {
            base: 0,
            size: 1,
            cell_bits: 64,
            body: vec![Stmt::Push(Value::ONE), Stmt::Op(OpCode::Pop)],
        }];
        assert!(verify_default(&bytecode(&balanced))
            .expect("balanced temporal scope")
            .is_fully_verified());

        let mut terminal = Program::new();
        terminal.body = vec![Stmt::TemporalScope {
            base: 0,
            size: 1,
            cell_bits: 64,
            body: vec![Stmt::Op(OpCode::Paradox)],
        }];
        let report = verify_default(&bytecode(&terminal)).expect("unwind obligation");
        assert!(report.obligations.iter().any(|obligation| matches!(
            obligation.kind,
            VerificationObligationKind::TerminalTemporalUnwind { depth: 1 }
        )));
    }

    #[test]
    fn deterministic_work_limits_reject_before_unbounded_processing() {
        let program = bytecode(&Program::new());
        let error = verify_with_limits(
            &program,
            VerificationLimits {
                max_control_flow_steps: 0,
                max_call_graph_steps: DEFAULT_MAX_CALL_GRAPH_STEPS,
            },
        )
        .expect_err("zero work limit");
        assert_eq!(
            error,
            BytecodeVerificationError::WorkLimitExceeded {
                phase: "control-flow",
                limit: 0
            }
        );
    }

    #[test]
    fn opcode_classifier_is_exhaustive_and_has_the_reviewed_boundary() {
        let mut fixed_count = 0usize;
        let mut dynamic = Vec::new();
        for &opcode in OpCode::ALL {
            match opcode_shape(opcode) {
                OpcodeShape::Fixed { .. } => fixed_count += 1,
                OpcodeShape::Dynamic { .. } => dynamic.push(opcode),
            }
        }
        assert_eq!(OpCode::ALL.len(), 99);
        assert_eq!(fixed_count, 76);
        assert_eq!(dynamic.len(), 23);
        assert_eq!(
            dynamic
                .iter()
                .filter(|opcode| **opcode == OpCode::Exec)
                .count(),
            1
        );
    }
}
