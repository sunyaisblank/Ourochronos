//! Conservative temporal-verification readiness analysis for bytecode.
//!
//! The existing [`TemporalIrCompiler`]
//! still consumes the source AST.  This module does not pretend that running
//! that compiler is bytecode lowering.  Instead it proves, directly over the
//! validated bytecode CFG, when every executable instruction is in the exact
//! subset understood by that compiler.  The resulting report is suitable for
//! gating the legacy source lowering while a native bytecode IR builder is
//! introduced.
//!
//! A report is deliberately stricter than structural bytecode validation:
//! quotations have no finite-IR sort, foreign and runtime-only primitives
//! have no symbolic semantics, nested temporal scopes are not implemented by
//! the legacy compiler, and recursive calls are not total.  Bounded loops are
//! compatible with legacy lowering, but always make the result unknown for a
//! global UNSAT claim.

use crate::ast::{OpCode, QuoteId};
use crate::bytecode::{
    BytecodeError, BytecodeProgram, CodeRange, Instruction, StackSemanticStatus,
};
use crate::bytecode_verifier::{BytecodeUnit, VerificationSite};
use crate::core::BoundsPolicy;
use crate::hir::{ForeignId, ProcedureId};
use crate::temporal::ir::{
    CompareOp, ExprId, IrCompleteness, IrExpr, IrExprKind, IrObservation, IrType, ObservationKind,
    TemporalIr, TemporalIrCompiler, TemporalIrConfig, WordBinaryOp, WordUnaryOp,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;

/// Default upper bound on reachable bytecode CFG transitions.
pub const DEFAULT_MAX_TEMPORAL_CFG_STEPS: usize = 4_000_000;
/// Default upper bound on call-graph traversal and SCC operations.
pub const DEFAULT_MAX_TEMPORAL_CALL_GRAPH_STEPS: usize = 8_000_000;
/// Conservative recursion depth accepted by the legacy source IR compiler.
pub const DEFAULT_MAX_LEGACY_CALL_DEPTH: usize = 256;
/// Conservative structured-control depth accepted by the legacy compiler.
pub const DEFAULT_MAX_LEGACY_CONTROL_DEPTH: usize = 256;
/// Default maximum number of native symbolic-lowering task steps.
pub const DEFAULT_MAX_TEMPORAL_LOWERING_STEPS: usize = 4_000_000;
/// Default maximum number of expressions emitted by native lowering.
pub const DEFAULT_MAX_TEMPORAL_IR_EXPRESSIONS: usize = 8_000_000;
/// Default maximum number of buffered symbolic observations.
pub const DEFAULT_MAX_TEMPORAL_IR_OBSERVATIONS: usize = 1_000_000;

/// Deterministic resource and legacy-lowering limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BytecodeTemporalLimits {
    /// Maximum reachable-instruction and CFG-edge operations.
    pub max_cfg_steps: usize,
    /// Maximum call-graph and SCC operations.
    pub max_call_graph_steps: usize,
    /// Maximum acyclic procedure-call depth delegated to the recursive legacy
    /// source compiler.
    pub max_legacy_call_depth: usize,
    /// Maximum nested IF/WHILE/TEMPORAL depth delegated to the recursive
    /// legacy source compiler.
    pub max_legacy_control_depth: usize,
}

impl Default for BytecodeTemporalLimits {
    fn default() -> Self {
        Self {
            max_cfg_steps: DEFAULT_MAX_TEMPORAL_CFG_STEPS,
            max_call_graph_steps: DEFAULT_MAX_TEMPORAL_CALL_GRAPH_STEPS,
            max_legacy_call_depth: DEFAULT_MAX_LEGACY_CALL_DEPTH,
            max_legacy_control_depth: DEFAULT_MAX_LEGACY_CONTROL_DEPTH,
        }
    }
}

/// Deterministic bounds for native bytecode-to-temporal-IR lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BytecodeTemporalLoweringLimits {
    /// Readiness/CFG analysis limits applied before symbolic lowering.
    pub analysis: BytecodeTemporalLimits,
    /// Maximum explicit task and bytecode-instruction steps.
    pub max_lowering_steps: usize,
    /// Maximum typed expressions in the resulting arena.
    pub max_expressions: usize,
    /// Maximum buffered OUTPUT/EMIT observations.
    pub max_observations: usize,
}

impl Default for BytecodeTemporalLoweringLimits {
    fn default() -> Self {
        Self {
            analysis: BytecodeTemporalLimits::default(),
            max_lowering_steps: DEFAULT_MAX_TEMPORAL_LOWERING_STEPS,
            max_expressions: DEFAULT_MAX_TEMPORAL_IR_EXPRESSIONS,
            max_observations: DEFAULT_MAX_TEMPORAL_IR_OBSERVATIONS,
        }
    }
}

/// Broad reason why a primitive cannot enter the finite temporal IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveBoundary {
    /// Stack shape or executable code depends on runtime word values.
    Dynamic,
    /// Mutable heap/string/container behavior has no finite-IR model.
    RuntimeState,
    /// Host I/O, processes, networking, FFI, or nondeterminism is external to
    /// the point-fixed-state formula.
    ExternalEffect,
}

/// A precise reason that complete bytecode temporal lowering is unavailable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BytecodeTemporalIssueKind {
    /// A WHILE back edge is represented only up to the configured unroll
    /// bound by the legacy IR compiler.
    BoundedLoop {
        /// First instruction of the loop condition.
        loop_start: u32,
        /// First instruction after the loop.
        end_target: u32,
        /// Legacy symbolic unroll bound.
        unroll_limit: usize,
    },
    /// A reachable strongly connected procedure component is not total in the
    /// finite IR.
    RecursiveProcedures {
        /// Exact SCC members in typed identity order.
        members: Vec<ProcedureId>,
    },
    /// An acyclic call chain is semantically finite but unsafe to hand to the
    /// recursive legacy source lowerer at the configured depth.
    LegacyCallDepthExceeded {
        /// First observed depth beyond the configured boundary.
        depth: usize,
        /// Configured legacy boundary.
        limit: usize,
    },
    /// Structured bytecode nesting is semantically finite but unsafe to hand
    /// to the recursive legacy source lowerer at the configured depth.
    LegacyControlDepthExceeded {
        /// Nesting depth at the construct marker.
        depth: usize,
        /// Configured legacy boundary.
        limit: usize,
    },
    /// The finite temporal IR has no quotation value sort or typed quote call.
    QuotationValue {
        /// Exact quotation identity pushed by the bytecode.
        target: QuoteId,
    },
    /// Foreign identities are retained by bytecode, but the finite IR has no
    /// linked foreign semantics.
    ForeignCall {
        /// Exact foreign target.
        target: ForeignId,
    },
    /// A primitive has no total deterministic finite-IR lowering.
    UnsupportedPrimitive {
        /// Primitive encountered on an executable path.
        opcode: OpCode,
        /// Class of missing semantics.
        boundary: PrimitiveBoundary,
    },
    /// The legacy IR supports only one active temporal scope.
    NestedTemporalScope {
        /// Innermost already-active scope.
        outer_enter: u32,
    },
    /// A structurally valid scope falls outside the configured dense memory.
    TemporalScopeExceedsMemory {
        /// Scope base cell.
        base: u64,
        /// Scope cell count.
        size: u64,
        /// Configured dense memory width.
        memory_cells: usize,
    },
}

impl BytecodeTemporalIssueKind {
    fn is_unknown(&self) -> bool {
        matches!(
            self,
            Self::BoundedLoop { .. }
                | Self::RecursiveProcedures { .. }
                | Self::LegacyCallDepthExceeded { .. }
                | Self::LegacyControlDepthExceeded { .. }
        )
    }

    fn legacy_compatible(&self) -> bool {
        matches!(self, Self::BoundedLoop { .. })
    }
}

/// One bytecode/source-located verification boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BytecodeTemporalIssue {
    /// Executable instruction at which the boundary becomes observable.
    pub site: VerificationSite,
    /// Exact missing semantic or completeness fact.
    pub kind: BytecodeTemporalIssueKind,
}

/// Overall bytecode temporal-verification disposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BytecodeTemporalDisposition {
    /// Every executable path is finite and has legacy-compatible semantics.
    Ready,
    /// Lowering is supported only as a bounded/incomplete execution set.
    Unknown,
    /// At least one executable instruction has no finite-IR semantics.
    Unsupported,
}

/// Successful executable-CFG support and completeness report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BytecodeTemporalAnalysis {
    /// Main plus transitively called procedures, in deterministic discovery
    /// order.  Quote bodies are not executable merely because a quote value is
    /// pushed.
    pub reachable_units: Vec<BytecodeUnit>,
    /// Typed quotation identities observed on executable paths.
    pub referenced_quotations: Vec<QuoteId>,
    /// Number of distinct executable instructions visited.
    pub reachable_instructions: usize,
    /// Deterministic CFG work charged during analysis.
    pub cfg_steps: usize,
    /// Deterministic call-graph work charged during analysis.
    pub call_graph_steps: usize,
    /// Stack/type semantics remain the responsibility of the mandatory HIR
    /// and bytecode-verifier phases; this CFG support pass does not duplicate
    /// that proof.
    pub stack_semantics: StackSemanticStatus,
    /// Findings in stable unit/instruction order.
    pub issues: Vec<BytecodeTemporalIssue>,
}

impl BytecodeTemporalAnalysis {
    /// Conservative summary suitable for CLI/API policy decisions.
    pub fn disposition(&self) -> BytecodeTemporalDisposition {
        if self.issues.iter().any(|issue| !issue.kind.is_unknown()) {
            BytecodeTemporalDisposition::Unsupported
        } else if self.issues.is_empty() {
            BytecodeTemporalDisposition::Ready
        } else {
            BytecodeTemporalDisposition::Unknown
        }
    }

    /// True only when delegating to the source-bound IR compiler preserves the
    /// analyzed bytecode subset.  Bounded loops are allowed because that
    /// compiler records [`crate::temporal::ir::IrCompleteness::BoundedLoops`].
    pub fn legacy_lowering_compatible(&self) -> bool {
        self.issues
            .iter()
            .all(|issue| issue.kind.legacy_compatible())
    }

    /// Whether this report covers a complete supported CFG.  This is a
    /// necessary, not sufficient, condition for global UNSAT: callers must
    /// also require successful stack verification and typed-IR construction.
    pub fn has_complete_supported_cfg(&self) -> bool {
        self.disposition() == BytecodeTemporalDisposition::Ready
    }
}

/// Structural, configuration, invariant, or resource failure during analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BytecodeTemporalError {
    /// The input artifact was not structurally valid.
    Structural(BytecodeError),
    /// The finite temporal memory configuration was empty.
    InvalidConfiguration(&'static str),
    /// A deterministic analysis work limit was exhausted.
    WorkLimitExceeded {
        /// Analysis phase.
        phase: &'static str,
        /// Configured maximum.
        limit: usize,
    },
    /// A validated invariant could not be recovered without unchecked access.
    InternalInvariant(String),
}

impl fmt::Display for BytecodeTemporalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Structural(error) => write!(formatter, "structural bytecode error: {error}"),
            Self::InvalidConfiguration(message) => formatter.write_str(message),
            Self::WorkLimitExceeded { phase, limit } => {
                write!(formatter, "{phase} work limit {limit} exceeded")
            }
            Self::InternalInvariant(message) => {
                write!(
                    formatter,
                    "validated bytecode invariant unavailable: {message}"
                )
            }
        }
    }
}

impl Error for BytecodeTemporalError {}

impl From<BytecodeError> for BytecodeTemporalError {
    fn from(value: BytecodeError) -> Self {
        Self::Structural(value)
    }
}

/// Typed reason native bytecode lowering could not produce temporal IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BytecodeTemporalLoweringErrorKind {
    /// Readiness/structural/configuration analysis failed before lowering.
    Analysis(BytecodeTemporalError),
    /// A reachable bytecode boundary has no complete native lowering.
    Unsupported(BytecodeTemporalIssueKind),
    /// A supported operation did not have enough symbolic input words.
    StackUnderflow {
        /// Operation or structured construct being lowered.
        operation: String,
        /// Required input count.
        required: usize,
        /// Available symbolic words.
        available: usize,
    },
    /// Symbolic control-flow branches do not rejoin at one stack height.
    StackJoinMismatch {
        /// Structured construct.
        construct: &'static str,
        /// Then/continuing path height.
        when_true: usize,
        /// Else/exiting path height.
        when_false: usize,
    },
    /// Only one side of symbolic control flow terminates the epoch.
    PathDependentHalt {
        /// Structured construct containing the terminal path.
        construct: &'static str,
    },
    /// HALT occurred in a symbolic WHILE condition or body, which the bounded
    /// formula cannot merge faithfully.
    HaltInSymbolicLoop {
        /// Loop portion containing HALT.
        portion: &'static str,
    },
    /// A call inherited an already-active temporal scope and entered another.
    NestedTemporalScope {
        /// Already-active scope enter instruction.
        outer_enter: u32,
    },
    /// The flat structured-control metadata was encountered outside the block
    /// boundary that owns it.
    UnexpectedControlInstruction,
    /// A deterministic native-lowering resource limit was exhausted.
    LimitExceeded {
        /// Limited resource.
        what: &'static str,
        /// Configured maximum.
        limit: usize,
    },
    /// The constructed expression graph violated typed arena invariants.
    InvalidIr(String),
    /// A post-validation bytecode invariant could not be recovered safely.
    InternalInvariant(String),
}

/// Source-located native bytecode temporal-lowering failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BytecodeTemporalLoweringError {
    /// Exact bytecode/source site when the failure belongs to an instruction.
    pub site: Option<VerificationSite>,
    /// Typed failure reason.
    pub kind: BytecodeTemporalLoweringErrorKind,
}

impl BytecodeTemporalLoweringError {
    fn at(site: VerificationSite, kind: BytecodeTemporalLoweringErrorKind) -> Self {
        Self {
            site: Some(site),
            kind,
        }
    }

    fn global(kind: BytecodeTemporalLoweringErrorKind) -> Self {
        Self { site: None, kind }
    }
}

impl fmt::Display for BytecodeTemporalLoweringError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(site) = self.site {
            write!(formatter, "instruction {}: ", site.instruction)?;
        }
        match &self.kind {
            BytecodeTemporalLoweringErrorKind::Analysis(error) => error.fmt(formatter),
            BytecodeTemporalLoweringErrorKind::Unsupported(kind) => {
                write!(
                    formatter,
                    "unsupported temporal bytecode boundary: {kind:?}"
                )
            }
            BytecodeTemporalLoweringErrorKind::StackUnderflow {
                operation,
                required,
                available,
            } => write!(
                formatter,
                "stack underflow lowering {operation}: need {required}, have {available}"
            ),
            BytecodeTemporalLoweringErrorKind::StackJoinMismatch {
                construct,
                when_true,
                when_false,
            } => write!(
                formatter,
                "{construct} paths have stack heights {when_true} and {when_false}"
            ),
            BytecodeTemporalLoweringErrorKind::PathDependentHalt { construct } => {
                write!(formatter, "{construct} contains path-dependent HALT")
            }
            BytecodeTemporalLoweringErrorKind::HaltInSymbolicLoop { portion } => {
                write!(formatter, "HALT in symbolic WHILE {portion} is unsupported")
            }
            BytecodeTemporalLoweringErrorKind::NestedTemporalScope { outer_enter } => write!(
                formatter,
                "nested temporal scope entered while instruction {outer_enter} is active"
            ),
            BytecodeTemporalLoweringErrorKind::UnexpectedControlInstruction => formatter
                .write_str("structured control instruction escaped its owning bytecode block"),
            BytecodeTemporalLoweringErrorKind::LimitExceeded { what, limit } => {
                write!(formatter, "native temporal {what} limit {limit} exceeded")
            }
            BytecodeTemporalLoweringErrorKind::InvalidIr(message) => {
                write!(formatter, "invalid native temporal IR: {message}")
            }
            BytecodeTemporalLoweringErrorKind::InternalInvariant(message) => {
                write!(
                    formatter,
                    "validated bytecode invariant unavailable: {message}"
                )
            }
        }
    }
}

impl Error for BytecodeTemporalLoweringError {}

/// Lower validated linked bytecode directly into the typed finite temporal IR.
pub fn lower_bytecode_temporal_ir(
    program: &BytecodeProgram,
    config: TemporalIrConfig,
) -> Result<TemporalIr, BytecodeTemporalLoweringError> {
    lower_bytecode_temporal_ir_with_limits(
        program,
        config,
        BytecodeTemporalLoweringLimits::default(),
    )
}

/// Lower bytecode directly with explicit deterministic resource limits.
pub fn lower_bytecode_temporal_ir_with_limits(
    program: &BytecodeProgram,
    config: TemporalIrConfig,
    limits: BytecodeTemporalLoweringLimits,
) -> Result<TemporalIr, BytecodeTemporalLoweringError> {
    analyze_bytecode_temporal_with_limits(program, config, limits.analysis).map_err(|error| {
        BytecodeTemporalLoweringError::global(BytecodeTemporalLoweringErrorKind::Analysis(error))
    })?;
    NativeCompiler::new(program, config, limits)?.compile()
}

#[derive(Debug, Clone)]
struct SymbolicState {
    stack: Vec<ExprId>,
    present: ExprId,
    valid: ExprId,
    stopped: bool,
    calls: Vec<ProcedureId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ActiveScope {
    enter: u32,
    base: u64,
    size: u64,
    cell_bits: u8,
}

#[derive(Debug, Clone, Copy)]
struct LoopShape {
    marker: u32,
    end_target: u32,
}

enum LowerTask {
    Block {
        unit: BytecodeUnit,
        pc: u32,
        end: u32,
        state: SymbolicState,
        scope: Option<ActiveScope>,
    },
    ContinueBlock {
        unit: BytecodeUnit,
        pc: u32,
        end: u32,
        scope: Option<ActiveScope>,
    },
    ReturnFromCall {
        unit: BytecodeUnit,
        pc: u32,
        end: u32,
        scope: Option<ActiveScope>,
        target: ProcedureId,
    },
    LowerIf {
        unit: BytecodeUnit,
        marker: u32,
        state: SymbolicState,
        scope: Option<ActiveScope>,
    },
    AfterIfThen {
        unit: BytecodeUnit,
        marker: u32,
        condition: ExprId,
        else_state: SymbolicState,
        else_range: Option<(u32, u32)>,
        scope: Option<ActiveScope>,
    },
    AfterIfElse {
        unit: BytecodeUnit,
        marker: u32,
        condition: ExprId,
        then_state: SymbolicState,
    },
    LowerWhile {
        unit: BytecodeUnit,
        loop_start: u32,
        shape: LoopShape,
        state: SymbolicState,
        scope: Option<ActiveScope>,
        remaining: usize,
        count_loop: bool,
    },
    AfterWhileCondition {
        unit: BytecodeUnit,
        loop_start: u32,
        shape: LoopShape,
        scope: Option<ActiveScope>,
        remaining: usize,
    },
    AfterWhileBody {
        unit: BytecodeUnit,
        loop_start: u32,
        shape: LoopShape,
        scope: Option<ActiveScope>,
        remaining: usize,
        condition: ExprId,
        exit_state: SymbolicState,
    },
    AfterWhileContinue {
        unit: BytecodeUnit,
        marker: u32,
        condition: ExprId,
        exit_state: SymbolicState,
    },
}

struct NativeCompiler<'a> {
    program: &'a BytecodeProgram,
    config: TemporalIrConfig,
    limits: BytecodeTemporalLoweringLimits,
    work: WorkCounter,
    expressions: Vec<IrExpr>,
    observations: Vec<IrObservation>,
    loops: BTreeMap<(BytecodeUnit, u32), LoopShape>,
    loop_count: usize,
    anamnesis: ExprId,
}

impl<'a> NativeCompiler<'a> {
    fn new(
        program: &'a BytecodeProgram,
        config: TemporalIrConfig,
        limits: BytecodeTemporalLoweringLimits,
    ) -> Result<Self, BytecodeTemporalLoweringError> {
        let loops = collect_loop_shapes(program)?;
        Ok(Self {
            program,
            config,
            limits,
            work: WorkCounter::new("lowering step", limits.max_lowering_steps),
            expressions: Vec::new(),
            observations: Vec::new(),
            loops,
            loop_count: 0,
            anamnesis: 0,
        })
    }

    fn compile(mut self) -> Result<TemporalIr, BytecodeTemporalLoweringError> {
        self.reserve_expressions(4, None)?;
        self.anamnesis = self.add(IrType::Memory, IrExprKind::Anamnesis);
        let zero_memory = self.add(IrType::Memory, IrExprKind::ZeroMemory);
        let valid = self.bool_const(true);
        let initial = SymbolicState {
            stack: Vec::new(),
            present: zero_memory,
            valid,
            stopped: false,
            calls: Vec::new(),
        };
        let main_end = self.program.main.end.checked_sub(1).ok_or_else(|| {
            BytecodeTemporalLoweringError::global(
                BytecodeTemporalLoweringErrorKind::InternalInvariant(
                    "main bytecode range has no RETURN".to_string(),
                ),
            )
        })?;
        let mut tasks = vec![LowerTask::Block {
            unit: BytecodeUnit::Main,
            pc: self.program.main.start,
            end: main_end,
            state: initial,
            scope: None,
        }];
        let mut last = None::<SymbolicState>;

        while let Some(task) = tasks.pop() {
            self.work.charge(1).map_err(|error| {
                BytecodeTemporalLoweringError::global(BytecodeTemporalLoweringErrorKind::Analysis(
                    error,
                ))
            })?;
            match task {
                LowerTask::Block {
                    unit,
                    pc,
                    end,
                    state,
                    scope,
                } => self.lower_block(unit, pc, end, state, scope, &mut tasks, &mut last)?,
                LowerTask::ContinueBlock {
                    unit,
                    pc,
                    end,
                    scope,
                } => {
                    let state = take_state(&mut last, "block continuation")?;
                    if state.stopped {
                        last = Some(state);
                    } else {
                        tasks.push(LowerTask::Block {
                            unit,
                            pc,
                            end,
                            state,
                            scope,
                        });
                    }
                }
                LowerTask::ReturnFromCall {
                    unit,
                    pc,
                    end,
                    scope,
                    target,
                } => {
                    let mut state = take_state(&mut last, "procedure return")?;
                    let active = state.calls.pop().ok_or_else(|| {
                        BytecodeTemporalLoweringError::global(
                            BytecodeTemporalLoweringErrorKind::InternalInvariant(format!(
                                "procedure {} returned without an active call",
                                target.as_u32()
                            )),
                        )
                    })?;
                    if active != target {
                        return Err(BytecodeTemporalLoweringError::global(
                            BytecodeTemporalLoweringErrorKind::InternalInvariant(format!(
                                "procedure {} returned while {} was active",
                                target.as_u32(),
                                active.as_u32()
                            )),
                        ));
                    }
                    if state.stopped {
                        last = Some(state);
                    } else {
                        tasks.push(LowerTask::Block {
                            unit,
                            pc,
                            end,
                            state,
                            scope,
                        });
                    }
                }
                LowerTask::LowerIf {
                    unit,
                    marker,
                    state,
                    scope,
                } => self.start_if(unit, marker, state, scope, &mut tasks)?,
                LowerTask::AfterIfThen {
                    unit,
                    marker,
                    condition,
                    else_state,
                    else_range,
                    scope,
                } => {
                    let then_state = take_state(&mut last, "IF then branch")?;
                    self.reserve_expressions(4, Some(self.site(unit, marker)))?;
                    let mut else_state = else_state;
                    let not_condition = self.bool_not(condition);
                    else_state.valid = self.bool_and(else_state.valid, not_condition);
                    if let Some((start, end)) = else_range {
                        tasks.push(LowerTask::AfterIfElse {
                            unit,
                            marker,
                            condition,
                            then_state,
                        });
                        tasks.push(LowerTask::Block {
                            unit,
                            pc: start,
                            end,
                            state: else_state,
                            scope,
                        });
                    } else {
                        last = Some(self.merge_states(
                            self.site(unit, marker),
                            condition,
                            then_state,
                            else_state,
                            "IF",
                        )?);
                    }
                }
                LowerTask::AfterIfElse {
                    unit,
                    marker,
                    condition,
                    then_state,
                } => {
                    let else_state = take_state(&mut last, "IF else branch")?;
                    last = Some(self.merge_states(
                        self.site(unit, marker),
                        condition,
                        then_state,
                        else_state,
                        "IF",
                    )?);
                }
                LowerTask::LowerWhile {
                    unit,
                    loop_start,
                    shape,
                    state,
                    scope,
                    remaining,
                    count_loop,
                } => {
                    if count_loop {
                        self.loop_count = self.loop_count.checked_add(1).ok_or_else(|| {
                            BytecodeTemporalLoweringError::at(
                                self.site(unit, shape.marker),
                                BytecodeTemporalLoweringErrorKind::InternalInvariant(
                                    "lowered loop count overflowed usize".to_string(),
                                ),
                            )
                        })?;
                    }
                    tasks.push(LowerTask::AfterWhileCondition {
                        unit,
                        loop_start,
                        shape,
                        scope,
                        remaining,
                    });
                    tasks.push(LowerTask::Block {
                        unit,
                        pc: loop_start,
                        end: shape.marker,
                        state,
                        scope,
                    });
                }
                LowerTask::AfterWhileCondition {
                    unit,
                    loop_start,
                    shape,
                    scope,
                    remaining,
                } => {
                    let mut state = take_state(&mut last, "WHILE condition")?;
                    let site = self.site(unit, shape.marker);
                    if state.stopped {
                        return Err(BytecodeTemporalLoweringError::at(
                            site,
                            BytecodeTemporalLoweringErrorKind::HaltInSymbolicLoop {
                                portion: "condition",
                            },
                        ));
                    }
                    let condition_word = self.pop(&mut state, site, "WHILE condition")?;
                    self.reserve_expressions(8, Some(site))?;
                    let condition = self.word_to_bool(condition_word);
                    let mut exit_state = state.clone();
                    let not_condition = self.bool_not(condition);
                    exit_state.valid = self.bool_and(exit_state.valid, not_condition);
                    if remaining == 0 {
                        last = Some(exit_state);
                    } else {
                        let mut body_state = state;
                        body_state.valid = self.bool_and(body_state.valid, condition);
                        tasks.push(LowerTask::AfterWhileBody {
                            unit,
                            loop_start,
                            shape,
                            scope,
                            remaining,
                            condition,
                            exit_state,
                        });
                        tasks.push(LowerTask::Block {
                            unit,
                            pc: shape.marker + 1,
                            end: shape.end_target - 1,
                            state: body_state,
                            scope,
                        });
                    }
                }
                LowerTask::AfterWhileBody {
                    unit,
                    loop_start,
                    shape,
                    scope,
                    remaining,
                    condition,
                    exit_state,
                } => {
                    let body_state = take_state(&mut last, "WHILE body")?;
                    if body_state.stopped {
                        return Err(BytecodeTemporalLoweringError::at(
                            self.site(unit, shape.marker),
                            BytecodeTemporalLoweringErrorKind::HaltInSymbolicLoop {
                                portion: "body",
                            },
                        ));
                    }
                    tasks.push(LowerTask::AfterWhileContinue {
                        unit,
                        marker: shape.marker,
                        condition,
                        exit_state,
                    });
                    tasks.push(LowerTask::LowerWhile {
                        unit,
                        loop_start,
                        shape,
                        state: body_state,
                        scope,
                        remaining: remaining - 1,
                        count_loop: false,
                    });
                }
                LowerTask::AfterWhileContinue {
                    unit,
                    marker,
                    condition,
                    exit_state,
                } => {
                    let continue_state = take_state(&mut last, "WHILE continuation")?;
                    last = Some(self.merge_states(
                        self.site(unit, marker),
                        condition,
                        continue_state,
                        exit_state,
                        "WHILE",
                    )?);
                }
            }
        }

        let state = take_state(&mut last, "main result")?;
        let completeness = if self.loop_count == 0 {
            IrCompleteness::Complete
        } else {
            IrCompleteness::BoundedLoops {
                loop_count: self.loop_count,
                unroll_limit: self.config.loop_unroll_limit,
            }
        };
        self.validate_ir(state.present, state.valid)?;
        Ok(TemporalIr {
            expressions: self.expressions,
            anamnesis: self.anamnesis,
            final_memory: state.present,
            valid: state.valid,
            observations: self.observations,
            memory_cells: self.config.memory_cells,
            bounds_policy: self.config.bounds_policy,
            completeness,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_block(
        &mut self,
        unit: BytecodeUnit,
        mut pc: u32,
        end: u32,
        mut state: SymbolicState,
        scope: Option<ActiveScope>,
        tasks: &mut Vec<LowerTask>,
        last: &mut Option<SymbolicState>,
    ) -> Result<(), BytecodeTemporalLoweringError> {
        while pc < end && !state.stopped {
            self.work.charge(1).map_err(|error| {
                BytecodeTemporalLoweringError::at(
                    self.site(unit, pc),
                    BytecodeTemporalLoweringErrorKind::Analysis(error),
                )
            })?;
            self.reserve_expressions(32, Some(self.site(unit, pc)))?;

            if let Some(shape) = self
                .loops
                .get(&(unit, pc))
                .copied()
                .filter(|shape| shape.end_target <= end)
            {
                tasks.push(LowerTask::ContinueBlock {
                    unit,
                    pc: shape.end_target,
                    end,
                    scope,
                });
                tasks.push(LowerTask::LowerWhile {
                    unit,
                    loop_start: pc,
                    shape,
                    state,
                    scope,
                    remaining: self.config.loop_unroll_limit,
                    count_loop: true,
                });
                return Ok(());
            }

            let instruction = self.instruction(pc)?;
            let site = self.site(unit, pc);
            match instruction {
                Instruction::Primitive(opcode) => {
                    if primitive_boundary(opcode).is_some() {
                        return Err(BytecodeTemporalLoweringError::at(
                            site,
                            BytecodeTemporalLoweringErrorKind::Unsupported(
                                BytecodeTemporalIssueKind::UnsupportedPrimitive {
                                    opcode,
                                    boundary: primitive_boundary(opcode).ok_or_else(|| {
                                        BytecodeTemporalLoweringError::at(
                                            site,
                                            BytecodeTemporalLoweringErrorKind::InternalInvariant(
                                                "unsupported primitive lost its boundary"
                                                    .to_string(),
                                            ),
                                        )
                                    })?,
                                },
                            ),
                        ));
                    }
                    self.lower_op(opcode, &mut state, scope, site)?;
                    pc += 1;
                }
                Instruction::PushWord(value) => {
                    let value = self.word_const(value);
                    state.stack.push(value);
                    pc += 1;
                }
                Instruction::CallProcedure(target) => {
                    if let Some(cycle_start) =
                        state.calls.iter().position(|active| *active == target)
                    {
                        let members = state.calls.get(cycle_start..).ok_or_else(|| {
                            BytecodeTemporalLoweringError::at(
                                site,
                                BytecodeTemporalLoweringErrorKind::InternalInvariant(
                                    "recursive call suffix is absent".to_string(),
                                ),
                            )
                        })?;
                        return Err(BytecodeTemporalLoweringError::at(
                            site,
                            BytecodeTemporalLoweringErrorKind::Unsupported(
                                BytecodeTemporalIssueKind::RecursiveProcedures {
                                    members: members.to_vec(),
                                },
                            ),
                        ));
                    }
                    let range = self.procedure_range(target, site)?;
                    let body_end = range.end.checked_sub(1).ok_or_else(|| {
                        BytecodeTemporalLoweringError::at(
                            site,
                            BytecodeTemporalLoweringErrorKind::InternalInvariant(format!(
                                "procedure {} has no RETURN",
                                target.as_u32()
                            )),
                        )
                    })?;
                    state.calls.push(target);
                    tasks.push(LowerTask::ReturnFromCall {
                        unit,
                        pc: pc + 1,
                        end,
                        scope,
                        target,
                    });
                    tasks.push(LowerTask::Block {
                        unit: BytecodeUnit::Procedure(target),
                        pc: range.start,
                        end: body_end,
                        state,
                        scope,
                    });
                    return Ok(());
                }
                Instruction::IfFalse { end_target, .. } => {
                    tasks.push(LowerTask::ContinueBlock {
                        unit,
                        pc: end_target,
                        end,
                        scope,
                    });
                    tasks.push(LowerTask::LowerIf {
                        unit,
                        marker: pc,
                        state,
                        scope,
                    });
                    return Ok(());
                }
                Instruction::TemporalEnter {
                    base,
                    size,
                    cell_bits,
                    exit_target,
                } => {
                    if let Some(outer) = scope {
                        return Err(BytecodeTemporalLoweringError::at(
                            site,
                            BytecodeTemporalLoweringErrorKind::NestedTemporalScope {
                                outer_enter: outer.enter,
                            },
                        ));
                    }
                    let exceeds_memory = match base.checked_add(size) {
                        Some(region_end) => region_end > self.config.memory_cells as u64,
                        None => true,
                    };
                    if exceeds_memory {
                        return Err(BytecodeTemporalLoweringError::at(
                            site,
                            BytecodeTemporalLoweringErrorKind::Unsupported(
                                BytecodeTemporalIssueKind::TemporalScopeExceedsMemory {
                                    base,
                                    size,
                                    memory_cells: self.config.memory_cells,
                                },
                            ),
                        ));
                    }
                    tasks.push(LowerTask::ContinueBlock {
                        unit,
                        pc: exit_target + 1,
                        end,
                        scope,
                    });
                    tasks.push(LowerTask::Block {
                        unit,
                        pc: pc + 1,
                        end: exit_target,
                        state,
                        scope: Some(ActiveScope {
                            enter: pc,
                            base,
                            size,
                            cell_bits,
                        }),
                    });
                    return Ok(());
                }
                Instruction::PushQuote(target) => {
                    return Err(BytecodeTemporalLoweringError::at(
                        site,
                        BytecodeTemporalLoweringErrorKind::Unsupported(
                            BytecodeTemporalIssueKind::QuotationValue { target },
                        ),
                    ));
                }
                Instruction::CallForeign(target) => {
                    return Err(BytecodeTemporalLoweringError::at(
                        site,
                        BytecodeTemporalLoweringErrorKind::Unsupported(
                            BytecodeTemporalIssueKind::ForeignCall { target },
                        ),
                    ));
                }
                Instruction::Jump { .. }
                | Instruction::WhileFalse { .. }
                | Instruction::LoopBack { .. }
                | Instruction::TemporalExit { .. }
                | Instruction::Return => {
                    return Err(BytecodeTemporalLoweringError::at(
                        site,
                        BytecodeTemporalLoweringErrorKind::UnexpectedControlInstruction,
                    ));
                }
            }
        }
        *last = Some(state);
        Ok(())
    }

    fn start_if(
        &mut self,
        unit: BytecodeUnit,
        marker: u32,
        mut state: SymbolicState,
        scope: Option<ActiveScope>,
        tasks: &mut Vec<LowerTask>,
    ) -> Result<(), BytecodeTemporalLoweringError> {
        let site = self.site(unit, marker);
        let Instruction::IfFalse {
            else_target,
            end_target,
            has_else,
        } = self.instruction(marker)?
        else {
            return Err(BytecodeTemporalLoweringError::at(
                site,
                BytecodeTemporalLoweringErrorKind::InternalInvariant(
                    "IF marker is not IF_FALSE".to_string(),
                ),
            ));
        };
        self.reserve_expressions(8, Some(site))?;
        let condition_word = self.pop(&mut state, site, "IF")?;
        let condition = self.word_to_bool(condition_word);
        let parent_valid = state.valid;
        let mut then_state = state.clone();
        then_state.valid = self.bool_and(parent_valid, condition);
        let mut else_state = state;
        else_state.valid = parent_valid;

        let then_end = if has_else {
            else_target.checked_sub(1).ok_or_else(|| {
                BytecodeTemporalLoweringError::at(
                    site,
                    BytecodeTemporalLoweringErrorKind::InternalInvariant(
                        "IF else boundary underflows".to_string(),
                    ),
                )
            })?
        } else {
            end_target
        };
        tasks.push(LowerTask::AfterIfThen {
            unit,
            marker,
            condition,
            else_state,
            else_range: has_else.then_some((else_target, end_target)),
            scope,
        });
        tasks.push(LowerTask::Block {
            unit,
            pc: marker + 1,
            end: then_end,
            state: then_state,
            scope,
        });
        Ok(())
    }

    fn merge_states(
        &mut self,
        site: VerificationSite,
        condition: ExprId,
        when_true: SymbolicState,
        when_false: SymbolicState,
        construct: &'static str,
    ) -> Result<SymbolicState, BytecodeTemporalLoweringError> {
        self.reserve_expressions(when_true.stack.len().saturating_add(4), Some(site))?;
        let true_impossible = self.is_bool_const(when_true.valid, false);
        let false_impossible = self.is_bool_const(when_false.valid, false);
        let stack = if true_impossible {
            when_false.stack.clone()
        } else if false_impossible {
            when_true.stack.clone()
        } else {
            if when_true.stack.len() != when_false.stack.len() {
                return Err(BytecodeTemporalLoweringError::at(
                    site,
                    BytecodeTemporalLoweringErrorKind::StackJoinMismatch {
                        construct,
                        when_true: when_true.stack.len(),
                        when_false: when_false.stack.len(),
                    },
                ));
            }
            when_true
                .stack
                .iter()
                .zip(&when_false.stack)
                .map(|(&left, &right)| self.ite(condition, left, right))
                .collect()
        };
        if when_true.stopped != when_false.stopped && !true_impossible && !false_impossible {
            return Err(BytecodeTemporalLoweringError::at(
                site,
                BytecodeTemporalLoweringErrorKind::PathDependentHalt { construct },
            ));
        }
        if when_true.calls != when_false.calls {
            return Err(BytecodeTemporalLoweringError::at(
                site,
                BytecodeTemporalLoweringErrorKind::InternalInvariant(format!(
                    "{construct} paths rejoin with different active procedure calls"
                )),
            ));
        }
        let present = self.ite(condition, when_true.present, when_false.present);
        let valid = self.bool_or(when_true.valid, when_false.valid);
        let calls = when_true.calls.clone();
        Ok(SymbolicState {
            stack,
            present,
            valid,
            stopped: if true_impossible {
                when_false.stopped
            } else {
                when_true.stopped
            },
            calls,
        })
    }

    fn pop(
        &self,
        state: &mut SymbolicState,
        site: VerificationSite,
        operation: &str,
    ) -> Result<ExprId, BytecodeTemporalLoweringError> {
        state.stack.pop().ok_or_else(|| {
            BytecodeTemporalLoweringError::at(
                site,
                BytecodeTemporalLoweringErrorKind::StackUnderflow {
                    operation: operation.to_string(),
                    required: 1,
                    available: 0,
                },
            )
        })
    }

    fn require_stack(
        &self,
        state: &SymbolicState,
        site: VerificationSite,
        operation: &str,
        required: usize,
    ) -> Result<(), BytecodeTemporalLoweringError> {
        if state.stack.len() < required {
            return Err(BytecodeTemporalLoweringError::at(
                site,
                BytecodeTemporalLoweringErrorKind::StackUnderflow {
                    operation: operation.to_string(),
                    required,
                    available: state.stack.len(),
                },
            ));
        }
        Ok(())
    }

    fn lower_op(
        &mut self,
        op: OpCode,
        state: &mut SymbolicState,
        scope: Option<ActiveScope>,
        site: VerificationSite,
    ) -> Result<(), BytecodeTemporalLoweringError> {
        match op {
            OpCode::Nop => {}
            OpCode::Halt => state.stopped = true,
            OpCode::Pop => {
                self.pop(state, site, op.name())?;
            }
            OpCode::Dup => {
                let value = self.pop(state, site, op.name())?;
                state.stack.push(value);
                state.stack.push(value);
            }
            OpCode::Swap => {
                self.require_stack(state, site, op.name(), 2)?;
                let right = self.pop(state, site, op.name())?;
                let left = self.pop(state, site, op.name())?;
                state.stack.push(right);
                state.stack.push(left);
            }
            OpCode::Over => {
                self.require_stack(state, site, op.name(), 2)?;
                let index = state.stack.len() - 2;
                let value = state.stack.get(index).copied().ok_or_else(|| {
                    BytecodeTemporalLoweringError::at(
                        site,
                        BytecodeTemporalLoweringErrorKind::InternalInvariant(
                            "OVER input disappeared after height check".to_string(),
                        ),
                    )
                })?;
                state.stack.push(value);
            }
            OpCode::Rot => {
                self.require_stack(state, site, op.name(), 3)?;
                let index = state.stack.len() - 3;
                let value = state.stack.remove(index);
                state.stack.push(value);
            }
            OpCode::Depth => {
                let depth = self.word_const(state.stack.len() as u64);
                state.stack.push(depth);
            }
            OpCode::Add
            | OpCode::Sub
            | OpCode::Mul
            | OpCode::Div
            | OpCode::Mod
            | OpCode::Min
            | OpCode::Max
            | OpCode::And
            | OpCode::Or
            | OpCode::Xor
            | OpCode::Shl
            | OpCode::Shr => {
                self.require_stack(state, site, op.name(), 2)?;
                let rhs = self.pop(state, site, op.name())?;
                let lhs = self.pop(state, site, op.name())?;
                let operation = match op {
                    OpCode::Add => WordBinaryOp::Add,
                    OpCode::Sub => WordBinaryOp::Sub,
                    OpCode::Mul => WordBinaryOp::Mul,
                    OpCode::Div => WordBinaryOp::UnsignedDivOrZero,
                    OpCode::Mod => WordBinaryOp::UnsignedRemOrZero,
                    OpCode::Min => WordBinaryOp::MinUnsigned,
                    OpCode::Max => WordBinaryOp::MaxUnsigned,
                    OpCode::And => WordBinaryOp::BitAnd,
                    OpCode::Or => WordBinaryOp::BitOr,
                    OpCode::Xor => WordBinaryOp::BitXor,
                    OpCode::Shl => WordBinaryOp::ShiftLeftModulo64,
                    OpCode::Shr => WordBinaryOp::ShiftRightModulo64,
                    _ => {
                        return Err(BytecodeTemporalLoweringError::at(
                            site,
                            BytecodeTemporalLoweringErrorKind::InternalInvariant(
                                "binary opcode classification drifted".to_string(),
                            ),
                        ));
                    }
                };
                let result = self.add(
                    IrType::Word64,
                    IrExprKind::WordBinary {
                        op: operation,
                        lhs,
                        rhs,
                    },
                );
                state.stack.push(result);
            }
            OpCode::Neg | OpCode::Abs | OpCode::Sign | OpCode::Not => {
                let value = self.pop(state, site, op.name())?;
                let operation = match op {
                    OpCode::Neg => WordUnaryOp::Neg,
                    OpCode::Abs => WordUnaryOp::AbsSigned,
                    OpCode::Sign => WordUnaryOp::SignSigned,
                    OpCode::Not => WordUnaryOp::LogicalNot,
                    _ => {
                        return Err(BytecodeTemporalLoweringError::at(
                            site,
                            BytecodeTemporalLoweringErrorKind::InternalInvariant(
                                "unary opcode classification drifted".to_string(),
                            ),
                        ));
                    }
                };
                let result = self.add(
                    IrType::Word64,
                    IrExprKind::WordUnary {
                        op: operation,
                        value,
                    },
                );
                state.stack.push(result);
            }
            OpCode::Eq
            | OpCode::Neq
            | OpCode::Lt
            | OpCode::Gt
            | OpCode::Lte
            | OpCode::Gte
            | OpCode::Slt
            | OpCode::Sgt
            | OpCode::Slte
            | OpCode::Sgte => {
                self.require_stack(state, site, op.name(), 2)?;
                let rhs = self.pop(state, site, op.name())?;
                let lhs = self.pop(state, site, op.name())?;
                let operation = match op {
                    OpCode::Eq => CompareOp::Eq,
                    OpCode::Neq => CompareOp::Ne,
                    OpCode::Lt => CompareOp::Ult,
                    OpCode::Gt => CompareOp::Ugt,
                    OpCode::Lte => CompareOp::Ule,
                    OpCode::Gte => CompareOp::Uge,
                    OpCode::Slt => CompareOp::Slt,
                    OpCode::Sgt => CompareOp::Sgt,
                    OpCode::Slte => CompareOp::Sle,
                    OpCode::Sgte => CompareOp::Sge,
                    _ => {
                        return Err(BytecodeTemporalLoweringError::at(
                            site,
                            BytecodeTemporalLoweringErrorKind::InternalInvariant(
                                "comparison opcode classification drifted".to_string(),
                            ),
                        ));
                    }
                };
                let boolean = self.add(
                    IrType::Bool,
                    IrExprKind::Compare {
                        op: operation,
                        lhs,
                        rhs,
                    },
                );
                let one = self.word_const(1);
                let zero = self.word_const(0);
                let result = self.ite(boolean, one, zero);
                state.stack.push(result);
            }
            OpCode::Oracle => {
                let raw_address = self.pop(state, site, op.name())?;
                let address = self.resolve_address(state, raw_address, scope);
                let value = self.add(
                    IrType::Word64,
                    IrExprKind::Select {
                        memory: self.anamnesis,
                        address,
                    },
                );
                state.stack.push(value);
            }
            OpCode::PresentRead => {
                let raw_address = self.pop(state, site, op.name())?;
                let address = self.resolve_address(state, raw_address, scope);
                let value = self.add(
                    IrType::Word64,
                    IrExprKind::Select {
                        memory: state.present,
                        address,
                    },
                );
                state.stack.push(value);
            }
            OpCode::Prophecy => {
                self.require_stack(state, site, op.name(), 2)?;
                let raw_address = self.pop(state, site, op.name())?;
                let value = self.pop(state, site, op.name())?;
                let address = self.resolve_address(state, raw_address, scope);
                let value = self.truncate_to_scope(value, scope);
                state.present = self.add(
                    IrType::Memory,
                    IrExprKind::Store {
                        memory: state.present,
                        address,
                        value,
                    },
                );
            }
            OpCode::Index => {
                self.require_stack(state, site, op.name(), 2)?;
                let index = self.pop(state, site, op.name())?;
                let base = self.pop(state, site, op.name())?;
                let address = self.add(
                    IrType::Word64,
                    IrExprKind::WordBinary {
                        op: WordBinaryOp::Add,
                        lhs: base,
                        rhs: index,
                    },
                );
                let address = self.resolve_address(state, address, scope);
                let value = self.add(
                    IrType::Word64,
                    IrExprKind::Select {
                        memory: state.present,
                        address,
                    },
                );
                state.stack.push(value);
            }
            OpCode::Store => {
                self.require_stack(state, site, op.name(), 3)?;
                let index = self.pop(state, site, op.name())?;
                let base = self.pop(state, site, op.name())?;
                let value = self.pop(state, site, op.name())?;
                let address = self.add(
                    IrType::Word64,
                    IrExprKind::WordBinary {
                        op: WordBinaryOp::Add,
                        lhs: base,
                        rhs: index,
                    },
                );
                let address = self.resolve_address(state, address, scope);
                let value = self.truncate_to_scope(value, scope);
                state.present = self.add(
                    IrType::Memory,
                    IrExprKind::Store {
                        memory: state.present,
                        address,
                        value,
                    },
                );
            }
            OpCode::Output | OpCode::Emit => {
                let value = self.pop(state, site, op.name())?;
                if self.observations.len() >= self.limits.max_observations {
                    return Err(BytecodeTemporalLoweringError::at(
                        site,
                        BytecodeTemporalLoweringErrorKind::LimitExceeded {
                            what: "observation",
                            limit: self.limits.max_observations,
                        },
                    ));
                }
                self.observations.push(IrObservation {
                    guard: state.valid,
                    value,
                    kind: if op == OpCode::Emit {
                        ObservationKind::Character
                    } else {
                        ObservationKind::Value
                    },
                });
            }
            OpCode::Paradox => {
                state.valid = self.bool_const(false);
                state.stopped = true;
            }
            _ => {
                let boundary = primitive_boundary(op).ok_or_else(|| {
                    BytecodeTemporalLoweringError::at(
                        site,
                        BytecodeTemporalLoweringErrorKind::InternalInvariant(
                            "supported opcode missing native lowering".to_string(),
                        ),
                    )
                })?;
                return Err(BytecodeTemporalLoweringError::at(
                    site,
                    BytecodeTemporalLoweringErrorKind::Unsupported(
                        BytecodeTemporalIssueKind::UnsupportedPrimitive {
                            opcode: op,
                            boundary,
                        },
                    ),
                ));
            }
        }
        Ok(())
    }

    fn resolve_address(
        &mut self,
        state: &mut SymbolicState,
        address: ExprId,
        scope: Option<ActiveScope>,
    ) -> ExprId {
        let Some(scope) = scope else {
            self.require_address_in_bounds(state, address, self.config.memory_cells as u64);
            return address;
        };
        let local = match self.config.bounds_policy {
            BoundsPolicy::Wrap => {
                let size = self.word_const(scope.size);
                self.add(
                    IrType::Word64,
                    IrExprKind::WordBinary {
                        op: WordBinaryOp::UnsignedRemOrZero,
                        lhs: address,
                        rhs: size,
                    },
                )
            }
            BoundsPolicy::Clamp => {
                let size = self.word_const(scope.size);
                let in_bounds = self.add(
                    IrType::Bool,
                    IrExprKind::Compare {
                        op: CompareOp::Ult,
                        lhs: address,
                        rhs: size,
                    },
                );
                let last = self.word_const(scope.size - 1);
                self.ite(in_bounds, address, last)
            }
            BoundsPolicy::Error => {
                self.require_address_in_bounds(state, address, scope.size);
                address
            }
        };
        if scope.base == 0 {
            local
        } else {
            let base = self.word_const(scope.base);
            self.add(
                IrType::Word64,
                IrExprKind::WordBinary {
                    op: WordBinaryOp::Add,
                    lhs: base,
                    rhs: local,
                },
            )
        }
    }

    fn truncate_to_scope(&mut self, value: ExprId, scope: Option<ActiveScope>) -> ExprId {
        let Some(scope) = scope else {
            return value;
        };
        if scope.cell_bits == 64 {
            return value;
        }
        let mask = self.word_const((1u64 << scope.cell_bits) - 1);
        self.add(
            IrType::Word64,
            IrExprKind::WordBinary {
                op: WordBinaryOp::BitAnd,
                lhs: value,
                rhs: mask,
            },
        )
    }

    fn require_address_in_bounds(
        &mut self,
        state: &mut SymbolicState,
        address: ExprId,
        limit_value: u64,
    ) {
        if self.config.bounds_policy != BoundsPolicy::Error {
            return;
        }
        let limit = self.word_const(limit_value);
        let in_bounds = self.add(
            IrType::Bool,
            IrExprKind::Compare {
                op: CompareOp::Ult,
                lhs: address,
                rhs: limit,
            },
        );
        state.valid = self.bool_and(state.valid, in_bounds);
    }

    fn reserve_expressions(
        &self,
        additional: usize,
        site: Option<VerificationSite>,
    ) -> Result<(), BytecodeTemporalLoweringError> {
        let exceeds = self
            .expressions
            .len()
            .checked_add(additional)
            .is_none_or(|count| count > self.limits.max_expressions);
        if exceeds {
            let kind = BytecodeTemporalLoweringErrorKind::LimitExceeded {
                what: "expression",
                limit: self.limits.max_expressions,
            };
            return Err(match site {
                Some(site) => BytecodeTemporalLoweringError::at(site, kind),
                None => BytecodeTemporalLoweringError::global(kind),
            });
        }
        Ok(())
    }

    fn add(&mut self, ty: IrType, kind: IrExprKind) -> ExprId {
        let id = self.expressions.len();
        self.expressions.push(IrExpr { ty, kind });
        id
    }

    fn bool_const(&mut self, value: bool) -> ExprId {
        self.add(IrType::Bool, IrExprKind::BoolConst(value))
    }

    fn word_const(&mut self, value: u64) -> ExprId {
        self.add(IrType::Word64, IrExprKind::WordConst(value))
    }

    fn is_bool_const(&self, id: ExprId, expected: bool) -> bool {
        self.expressions
            .get(id)
            .is_some_and(|expression| matches!(expression.kind, IrExprKind::BoolConst(value) if value == expected))
    }

    fn bool_not(&mut self, value: ExprId) -> ExprId {
        if self.is_bool_const(value, true) {
            return self.bool_const(false);
        }
        if self.is_bool_const(value, false) {
            return self.bool_const(true);
        }
        self.add(IrType::Bool, IrExprKind::BoolNot(value))
    }

    fn bool_and(&mut self, lhs: ExprId, rhs: ExprId) -> ExprId {
        if self.is_bool_const(lhs, false) || self.is_bool_const(rhs, false) {
            return self.bool_const(false);
        }
        if self.is_bool_const(lhs, true) {
            return rhs;
        }
        if self.is_bool_const(rhs, true) {
            return lhs;
        }
        self.add(IrType::Bool, IrExprKind::BoolAnd(lhs, rhs))
    }

    fn bool_or(&mut self, lhs: ExprId, rhs: ExprId) -> ExprId {
        if self.is_bool_const(lhs, true) || self.is_bool_const(rhs, true) {
            return self.bool_const(true);
        }
        if self.is_bool_const(lhs, false) {
            return rhs;
        }
        if self.is_bool_const(rhs, false) {
            return lhs;
        }
        self.add(IrType::Bool, IrExprKind::BoolOr(lhs, rhs))
    }

    fn ite(&mut self, condition: ExprId, when_true: ExprId, when_false: ExprId) -> ExprId {
        if self.is_bool_const(condition, true) {
            return when_true;
        }
        if self.is_bool_const(condition, false) {
            return when_false;
        }
        if when_true == when_false {
            return when_true;
        }
        let ty = self
            .expressions
            .get(when_true)
            .map_or(IrType::Word64, |expression| expression.ty);
        self.add(
            ty,
            IrExprKind::Ite {
                condition,
                when_true,
                when_false,
            },
        )
    }

    fn word_to_bool(&mut self, word: ExprId) -> ExprId {
        let zero = self.word_const(0);
        self.add(
            IrType::Bool,
            IrExprKind::Compare {
                op: CompareOp::Ne,
                lhs: word,
                rhs: zero,
            },
        )
    }

    fn instruction(&self, pc: u32) -> Result<Instruction, BytecodeTemporalLoweringError> {
        self.program
            .instructions
            .get(pc as usize)
            .copied()
            .ok_or_else(|| {
                BytecodeTemporalLoweringError::global(
                    BytecodeTemporalLoweringErrorKind::InternalInvariant(format!(
                        "instruction {pc} is absent"
                    )),
                )
            })
    }

    fn procedure_range(
        &self,
        id: ProcedureId,
        site: VerificationSite,
    ) -> Result<CodeRange, BytecodeTemporalLoweringError> {
        self.program
            .procedures
            .get(id.index())
            .map(|entry| entry.range)
            .ok_or_else(|| {
                BytecodeTemporalLoweringError::at(
                    site,
                    BytecodeTemporalLoweringErrorKind::InternalInvariant(format!(
                        "procedure {} has no entry",
                        id.as_u32()
                    )),
                )
            })
    }

    fn site(&self, unit: BytecodeUnit, instruction: u32) -> VerificationSite {
        verification_site(self.program, unit, instruction)
    }

    fn validate_ir(
        &self,
        final_memory: ExprId,
        valid: ExprId,
    ) -> Result<(), BytecodeTemporalLoweringError> {
        for (id, expression) in self.expressions.iter().enumerate() {
            let child = |child: ExprId, expected: IrType| {
                let Some(child_expression) = self.expressions.get(child) else {
                    return Err(BytecodeTemporalLoweringError::global(
                        BytecodeTemporalLoweringErrorKind::InvalidIr(format!(
                            "expression e{id} references absent child e{child}"
                        )),
                    ));
                };
                if child >= id {
                    return Err(BytecodeTemporalLoweringError::global(
                        BytecodeTemporalLoweringErrorKind::InvalidIr(format!(
                            "expression e{id} has non-prior child e{child}"
                        )),
                    ));
                }
                if child_expression.ty != expected {
                    return Err(BytecodeTemporalLoweringError::global(
                        BytecodeTemporalLoweringErrorKind::InvalidIr(format!(
                            "expression e{id} expected {expected:?} child e{child}, found {:?}",
                            child_expression.ty
                        )),
                    ));
                }
                Ok(())
            };
            match expression.kind {
                IrExprKind::BoolConst(_)
                | IrExprKind::WordConst(_)
                | IrExprKind::Anamnesis
                | IrExprKind::ZeroMemory => {}
                IrExprKind::Select { memory, address } => {
                    child(memory, IrType::Memory)?;
                    child(address, IrType::Word64)?;
                }
                IrExprKind::Store {
                    memory,
                    address,
                    value,
                } => {
                    child(memory, IrType::Memory)?;
                    child(address, IrType::Word64)?;
                    child(value, IrType::Word64)?;
                }
                IrExprKind::WordUnary { value, .. } => child(value, IrType::Word64)?,
                IrExprKind::WordBinary { lhs, rhs, .. } | IrExprKind::Compare { lhs, rhs, .. } => {
                    child(lhs, IrType::Word64)?;
                    child(rhs, IrType::Word64)?;
                }
                IrExprKind::BoolNot(value) => child(value, IrType::Bool)?,
                IrExprKind::BoolAnd(lhs, rhs) | IrExprKind::BoolOr(lhs, rhs) => {
                    child(lhs, IrType::Bool)?;
                    child(rhs, IrType::Bool)?;
                }
                IrExprKind::Ite {
                    condition,
                    when_true,
                    when_false,
                } => {
                    child(condition, IrType::Bool)?;
                    child(when_true, expression.ty)?;
                    child(when_false, expression.ty)?;
                }
            }
        }
        let final_memory_ty = self.expressions.get(final_memory).map(|expr| expr.ty);
        let valid_ty = self.expressions.get(valid).map(|expr| expr.ty);
        let anamnesis_ty = self.expressions.get(self.anamnesis).map(|expr| expr.ty);
        if final_memory_ty != Some(IrType::Memory)
            || valid_ty != Some(IrType::Bool)
            || anamnesis_ty != Some(IrType::Memory)
        {
            return Err(BytecodeTemporalLoweringError::global(
                BytecodeTemporalLoweringErrorKind::InvalidIr(
                    "IR roots have incorrect or absent types".to_string(),
                ),
            ));
        }
        for (index, observation) in self.observations.iter().enumerate() {
            let guard_ty = self.expressions.get(observation.guard).map(|expr| expr.ty);
            let value_ty = self.expressions.get(observation.value).map(|expr| expr.ty);
            if guard_ty != Some(IrType::Bool) || value_ty != Some(IrType::Word64) {
                return Err(BytecodeTemporalLoweringError::global(
                    BytecodeTemporalLoweringErrorKind::InvalidIr(format!(
                        "observation {index} has incorrect or absent guard/value types"
                    )),
                ));
            }
        }
        Ok(())
    }
}

fn take_state(
    last: &mut Option<SymbolicState>,
    context: &str,
) -> Result<SymbolicState, BytecodeTemporalLoweringError> {
    last.take().ok_or_else(|| {
        BytecodeTemporalLoweringError::global(BytecodeTemporalLoweringErrorKind::InternalInvariant(
            format!("{context} has no symbolic result"),
        ))
    })
}

fn collect_loop_shapes(
    program: &BytecodeProgram,
) -> Result<BTreeMap<(BytecodeUnit, u32), LoopShape>, BytecodeTemporalLoweringError> {
    let mut loops = BTreeMap::new();
    let mut units = Vec::with_capacity(1 + program.procedures.len());
    units.push((BytecodeUnit::Main, program.main));
    units.extend(
        program
            .procedures
            .iter()
            .map(|entry| (BytecodeUnit::Procedure(entry.id), entry.range)),
    );
    for (unit, range) in units {
        for pc in range.start..range.end {
            let instruction = program.instructions.get(pc as usize).ok_or_else(|| {
                BytecodeTemporalLoweringError::global(
                    BytecodeTemporalLoweringErrorKind::InternalInvariant(format!(
                        "instruction {pc} is absent while indexing loops"
                    )),
                )
            })?;
            if let Instruction::WhileFalse {
                loop_start,
                end_target,
            } = *instruction
            {
                if loops
                    .insert(
                        (unit, loop_start),
                        LoopShape {
                            marker: pc,
                            end_target,
                        },
                    )
                    .is_some()
                {
                    return Err(BytecodeTemporalLoweringError::at(
                        verification_site(program, unit, pc),
                        BytecodeTemporalLoweringErrorKind::InternalInvariant(format!(
                            "multiple WHILE markers share loop start {loop_start}"
                        )),
                    ));
                }
            }
        }
    }
    Ok(loops)
}

fn verification_site(
    program: &BytecodeProgram,
    unit: BytecodeUnit,
    instruction: u32,
) -> VerificationSite {
    let source = program
        .source_map
        .binary_search_by_key(&instruction, |entry| entry.instruction)
        .ok()
        .and_then(|index| program.source_map.get(index))
        .map(|entry| entry.span);
    VerificationSite {
        unit,
        instruction,
        source,
    }
}

/// Analyze temporal-verification readiness with default deterministic limits.
pub fn analyze_bytecode_temporal(
    program: &BytecodeProgram,
    config: TemporalIrConfig,
) -> Result<BytecodeTemporalAnalysis, BytecodeTemporalError> {
    analyze_bytecode_temporal_with_limits(program, config, BytecodeTemporalLimits::default())
}

/// Analyze the executable bytecode CFG without source-AST reinterpretation.
pub fn analyze_bytecode_temporal_with_limits(
    program: &BytecodeProgram,
    config: TemporalIrConfig,
    limits: BytecodeTemporalLimits,
) -> Result<BytecodeTemporalAnalysis, BytecodeTemporalError> {
    program.validate()?;
    if config.memory_cells == 0 {
        return Err(BytecodeTemporalError::InvalidConfiguration(
            "temporal bytecode analysis requires at least one memory cell",
        ));
    }
    if config.memory_cells > crate::core::MAX_DENSE_MEMORY_CELLS {
        return Err(BytecodeTemporalError::InvalidConfiguration(
            "temporal bytecode analysis memory width exceeds dense result limit",
        ));
    }

    let analyzer = Analyzer {
        program,
        config,
        limits,
        cfg_work: WorkCounter::new("temporal CFG", limits.max_cfg_steps),
        graph_work: WorkCounter::new("temporal call graph", limits.max_call_graph_steps),
        unit_queue: VecDeque::from([BytecodeUnit::Main]),
        scheduled_units: BTreeSet::from([BytecodeUnit::Main]),
        reachable_units: Vec::new(),
        referenced_quotations: BTreeSet::new(),
        reachable_instructions: 0,
        call_edges: Vec::new(),
        issues: Vec::new(),
    };
    analyzer.run()
}

#[derive(Debug, Clone, Copy)]
struct CallEdge {
    caller: BytecodeUnit,
    callee: ProcedureId,
    site: VerificationSite,
}

struct Analyzer<'a> {
    program: &'a BytecodeProgram,
    config: TemporalIrConfig,
    limits: BytecodeTemporalLimits,
    cfg_work: WorkCounter,
    graph_work: WorkCounter,
    unit_queue: VecDeque<BytecodeUnit>,
    scheduled_units: BTreeSet<BytecodeUnit>,
    reachable_units: Vec<BytecodeUnit>,
    referenced_quotations: BTreeSet<QuoteId>,
    reachable_instructions: usize,
    call_edges: Vec<CallEdge>,
    issues: Vec<BytecodeTemporalIssue>,
}

impl Analyzer<'_> {
    fn run(mut self) -> Result<BytecodeTemporalAnalysis, BytecodeTemporalError> {
        while let Some(unit) = self.unit_queue.pop_front() {
            self.reachable_units.push(unit);
            self.scan_unit(unit)?;
        }

        self.analyze_call_graph()?;
        self.issues.sort_by_key(|issue| {
            (
                issue.site.unit,
                issue.site.instruction,
                u8::from(!issue.kind.is_unknown()),
            )
        });

        Ok(BytecodeTemporalAnalysis {
            reachable_units: self.reachable_units,
            referenced_quotations: self.referenced_quotations.into_iter().collect(),
            reachable_instructions: self.reachable_instructions,
            cfg_steps: self.cfg_work.used,
            call_graph_steps: self.graph_work.used,
            stack_semantics: StackSemanticStatus::NotChecked,
            issues: self.issues,
        })
    }

    fn scan_unit(&mut self, unit: BytecodeUnit) -> Result<(), BytecodeTemporalError> {
        let range = self.unit_range(unit)?;
        let lexical = self.lexical_context(range)?;
        let control_depths = self.control_depths(range)?;
        let mut queue = VecDeque::from([range.start]);
        let mut reached = BTreeSet::<u32>::new();

        while let Some(pc) = queue.pop_front() {
            if !reached.insert(pc) {
                continue;
            }
            self.cfg_work.charge(1)?;
            self.reachable_instructions =
                self.reachable_instructions.checked_add(1).ok_or_else(|| {
                    BytecodeTemporalError::InternalInvariant(
                        "reachable instruction count overflowed usize".to_string(),
                    )
                })?;

            let instruction = *self.instruction(pc)?;
            let site = self.site(unit, pc);
            if let Some(depth) = control_depths.get(&pc).copied() {
                if depth > self.limits.max_legacy_control_depth {
                    self.issues.push(BytecodeTemporalIssue {
                        site,
                        kind: BytecodeTemporalIssueKind::LegacyControlDepthExceeded {
                            depth,
                            limit: self.limits.max_legacy_control_depth,
                        },
                    });
                }
            }

            match instruction {
                Instruction::Primitive(opcode) => {
                    if let Some(boundary) = primitive_boundary(opcode) {
                        self.issues.push(BytecodeTemporalIssue {
                            site,
                            kind: BytecodeTemporalIssueKind::UnsupportedPrimitive {
                                opcode,
                                boundary,
                            },
                        });
                    }
                    if !matches!(opcode, OpCode::Halt | OpCode::Paradox) {
                        self.enqueue_fallthrough(pc, range, &mut queue)?;
                    }
                }
                Instruction::PushWord(_) => {
                    self.enqueue_fallthrough(pc, range, &mut queue)?;
                }
                Instruction::PushQuote(target) => {
                    self.referenced_quotations.insert(target);
                    self.issues.push(BytecodeTemporalIssue {
                        site,
                        kind: BytecodeTemporalIssueKind::QuotationValue { target },
                    });
                    self.enqueue_fallthrough(pc, range, &mut queue)?;
                }
                Instruction::CallProcedure(target) => {
                    self.call_edges.push(CallEdge {
                        caller: unit,
                        callee: target,
                        site,
                    });
                    let target_unit = BytecodeUnit::Procedure(target);
                    if self.scheduled_units.insert(target_unit) {
                        self.unit_queue.push_back(target_unit);
                    }
                    self.enqueue_fallthrough(pc, range, &mut queue)?;
                }
                Instruction::CallForeign(target) => {
                    self.issues.push(BytecodeTemporalIssue {
                        site,
                        kind: BytecodeTemporalIssueKind::ForeignCall { target },
                    });
                    self.enqueue_fallthrough(pc, range, &mut queue)?;
                }
                Instruction::IfFalse { else_target, .. } => {
                    self.enqueue_target(pc, pc.checked_add(1), range, &mut queue)?;
                    self.enqueue_target(pc, Some(else_target), range, &mut queue)?;
                }
                Instruction::Jump { target } => {
                    self.enqueue_target(pc, Some(target), range, &mut queue)?;
                }
                Instruction::WhileFalse {
                    loop_start,
                    end_target,
                } => {
                    self.issues.push(BytecodeTemporalIssue {
                        site,
                        kind: BytecodeTemporalIssueKind::BoundedLoop {
                            loop_start,
                            end_target,
                            unroll_limit: self.config.loop_unroll_limit,
                        },
                    });
                    self.enqueue_target(pc, pc.checked_add(1), range, &mut queue)?;
                    self.enqueue_target(pc, Some(end_target), range, &mut queue)?;
                }
                Instruction::LoopBack { target } => {
                    self.enqueue_target(pc, Some(target), range, &mut queue)?;
                }
                Instruction::TemporalEnter {
                    base,
                    size,
                    exit_target: _,
                    cell_bits: _,
                } => {
                    if let Some(Some(outer_enter)) = lexical.get(&pc) {
                        self.issues.push(BytecodeTemporalIssue {
                            site,
                            kind: BytecodeTemporalIssueKind::NestedTemporalScope {
                                outer_enter: *outer_enter,
                            },
                        });
                    }
                    let exceeds_memory = match base.checked_add(size) {
                        Some(end) => end > self.config.memory_cells as u64,
                        None => true,
                    };
                    if exceeds_memory {
                        self.issues.push(BytecodeTemporalIssue {
                            site,
                            kind: BytecodeTemporalIssueKind::TemporalScopeExceedsMemory {
                                base,
                                size,
                                memory_cells: self.config.memory_cells,
                            },
                        });
                    }
                    self.enqueue_fallthrough(pc, range, &mut queue)?;
                }
                Instruction::TemporalExit { .. } => {
                    self.enqueue_fallthrough(pc, range, &mut queue)?;
                }
                Instruction::Return => {}
            }
        }
        Ok(())
    }

    fn analyze_call_graph(&mut self) -> Result<(), BytecodeTemporalError> {
        let reachable = self
            .scheduled_units
            .iter()
            .filter_map(|unit| match unit {
                BytecodeUnit::Procedure(id) => Some(id.index()),
                BytecodeUnit::Main | BytecodeUnit::Quote(_) => None,
            })
            .collect::<BTreeSet<_>>();
        let mut adjacency = reachable
            .iter()
            .copied()
            .map(|id| (id, Vec::<usize>::new()))
            .collect::<BTreeMap<_, _>>();
        let mut reverse = adjacency.clone();

        for edge in &self.call_edges {
            self.graph_work.charge(1)?;
            let BytecodeUnit::Procedure(caller) = edge.caller else {
                continue;
            };
            let outgoing = adjacency.get_mut(&caller.index()).ok_or_else(|| {
                BytecodeTemporalError::InternalInvariant(format!(
                    "reachable procedure {} missing from call graph",
                    caller.as_u32()
                ))
            })?;
            outgoing.push(edge.callee.index());
            let incoming = reverse.get_mut(&edge.callee.index()).ok_or_else(|| {
                BytecodeTemporalError::InternalInvariant(format!(
                    "reachable callee {} missing from reverse call graph",
                    edge.callee.as_u32()
                ))
            })?;
            incoming.push(caller.index());
        }
        for edges in adjacency.values_mut().chain(reverse.values_mut()) {
            edges.sort_unstable();
            edges.dedup();
        }

        let finish_order = self.finish_order(&reachable, &adjacency)?;
        let components = self.reverse_components(&finish_order, &reverse)?;
        let mut component_by_node = BTreeMap::<usize, usize>::new();
        for (component_id, component) in components.iter().enumerate() {
            for member in component {
                component_by_node.insert(*member, component_id);
            }
        }

        let mut recursive_components = BTreeSet::<usize>::new();
        for (component_id, component) in components.iter().enumerate() {
            let self_recursive = component.first().is_some_and(|node| {
                adjacency
                    .get(node)
                    .is_some_and(|targets| targets.binary_search(node).is_ok())
            });
            if component.len() > 1 || self_recursive {
                recursive_components.insert(component_id);
            }
        }

        for component_id in &recursive_components {
            let component = components.get(*component_id).ok_or_else(|| {
                BytecodeTemporalError::InternalInvariant(format!(
                    "recursive component {component_id} is absent"
                ))
            })?;
            let member_set = component.iter().copied().collect::<BTreeSet<_>>();
            let site = self
                .call_edges
                .iter()
                .filter_map(|edge| {
                    let BytecodeUnit::Procedure(caller) = edge.caller else {
                        return None;
                    };
                    (member_set.contains(&caller.index())
                        && member_set.contains(&edge.callee.index()))
                    .then_some(edge.site)
                })
                .min_by_key(|site| (site.unit, site.instruction))
                .ok_or_else(|| {
                    BytecodeTemporalError::InternalInvariant(format!(
                        "recursive component {component_id} has no internal call edge"
                    ))
                })?;
            let members = component
                .iter()
                .map(|index| {
                    ProcedureId::try_from_index(*index).ok_or_else(|| {
                        BytecodeTemporalError::InternalInvariant(format!(
                            "procedure index {index} cannot be represented"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.issues.push(BytecodeTemporalIssue {
                site,
                kind: BytecodeTemporalIssueKind::RecursiveProcedures { members },
            });
        }

        if recursive_components.is_empty() {
            self.check_legacy_call_depth(&finish_order, &component_by_node)?;
        }
        Ok(())
    }

    fn finish_order(
        &mut self,
        reachable: &BTreeSet<usize>,
        adjacency: &BTreeMap<usize, Vec<usize>>,
    ) -> Result<Vec<usize>, BytecodeTemporalError> {
        let mut state = reachable
            .iter()
            .copied()
            .map(|node| (node, 0u8))
            .collect::<BTreeMap<_, _>>();
        let mut order = Vec::with_capacity(reachable.len());

        for root in reachable {
            if state.get(root).copied() != Some(0) {
                continue;
            }
            let mut stack = vec![(*root, false)];
            while let Some((node, expanded)) = stack.pop() {
                self.graph_work.charge(1)?;
                let node_state = state.get(&node).copied().ok_or_else(|| {
                    BytecodeTemporalError::InternalInvariant(format!(
                        "call graph node {node} has no traversal state"
                    ))
                })?;
                if expanded {
                    if node_state != 2 {
                        state.insert(node, 2);
                        order.push(node);
                    }
                    continue;
                }
                if node_state != 0 {
                    continue;
                }
                state.insert(node, 1);
                stack.push((node, true));
                let targets = adjacency.get(&node).ok_or_else(|| {
                    BytecodeTemporalError::InternalInvariant(format!(
                        "call graph node {node} has no adjacency list"
                    ))
                })?;
                for target in targets.iter().rev() {
                    self.graph_work.charge(1)?;
                    if state.get(target).copied() == Some(0) {
                        stack.push((*target, false));
                    }
                }
            }
        }
        Ok(order)
    }

    fn reverse_components(
        &mut self,
        finish_order: &[usize],
        reverse: &BTreeMap<usize, Vec<usize>>,
    ) -> Result<Vec<Vec<usize>>, BytecodeTemporalError> {
        let mut assigned = BTreeSet::<usize>::new();
        let mut components = Vec::<Vec<usize>>::new();
        for root in finish_order.iter().rev() {
            if !assigned.insert(*root) {
                continue;
            }
            let mut component = Vec::new();
            let mut stack = vec![*root];
            while let Some(node) = stack.pop() {
                self.graph_work.charge(1)?;
                component.push(node);
                let incoming = reverse.get(&node).ok_or_else(|| {
                    BytecodeTemporalError::InternalInvariant(format!(
                        "call graph node {node} has no reverse adjacency list"
                    ))
                })?;
                for predecessor in incoming.iter().rev() {
                    self.graph_work.charge(1)?;
                    if assigned.insert(*predecessor) {
                        stack.push(*predecessor);
                    }
                }
            }
            component.sort_unstable();
            components.push(component);
        }
        Ok(components)
    }

    fn check_legacy_call_depth(
        &mut self,
        finish_order: &[usize],
        component_by_node: &BTreeMap<usize, usize>,
    ) -> Result<(), BytecodeTemporalError> {
        let mut depths = BTreeMap::<usize, usize>::new();
        let mut first_excess = None::<(usize, VerificationSite)>;

        for edge in self
            .call_edges
            .iter()
            .filter(|edge| edge.caller == BytecodeUnit::Main)
        {
            self.graph_work.charge(1)?;
            let depth = 1usize;
            depths
                .entry(edge.callee.index())
                .and_modify(|known| *known = (*known).max(depth))
                .or_insert(depth);
            if depth > self.limits.max_legacy_call_depth {
                first_excess = choose_excess(first_excess, depth, edge.site);
            }
        }

        for caller_index in finish_order.iter().rev() {
            self.graph_work.charge(1)?;
            if !component_by_node.contains_key(caller_index) {
                continue;
            }
            let Some(caller_depth) = depths.get(caller_index).copied() else {
                continue;
            };
            let caller_id = ProcedureId::try_from_index(*caller_index).ok_or_else(|| {
                BytecodeTemporalError::InternalInvariant(format!(
                    "procedure index {caller_index} cannot be represented"
                ))
            })?;
            for edge in self
                .call_edges
                .iter()
                .filter(|edge| edge.caller == BytecodeUnit::Procedure(caller_id))
            {
                self.graph_work.charge(1)?;
                let depth = caller_depth.checked_add(1).ok_or_else(|| {
                    BytecodeTemporalError::InternalInvariant(
                        "legacy call depth overflowed usize".to_string(),
                    )
                })?;
                depths
                    .entry(edge.callee.index())
                    .and_modify(|known| *known = (*known).max(depth))
                    .or_insert(depth);
                if depth > self.limits.max_legacy_call_depth {
                    first_excess = choose_excess(first_excess, depth, edge.site);
                }
            }
        }

        if let Some((depth, site)) = first_excess {
            self.issues.push(BytecodeTemporalIssue {
                site,
                kind: BytecodeTemporalIssueKind::LegacyCallDepthExceeded {
                    depth,
                    limit: self.limits.max_legacy_call_depth,
                },
            });
        }
        Ok(())
    }

    fn lexical_context(
        &self,
        range: CodeRange,
    ) -> Result<BTreeMap<u32, Option<u32>>, BytecodeTemporalError> {
        let mut active = Vec::<u32>::new();
        let mut context = BTreeMap::new();
        for pc in range.start..range.end {
            context.insert(pc, active.last().copied());
            match self.instruction(pc)? {
                Instruction::TemporalEnter { .. } => active.push(pc),
                Instruction::TemporalExit { enter_target } => {
                    let actual = active.pop().ok_or_else(|| {
                        BytecodeTemporalError::InternalInvariant(format!(
                            "temporal exit {pc} has no active enter"
                        ))
                    })?;
                    if actual != *enter_target {
                        return Err(BytecodeTemporalError::InternalInvariant(format!(
                            "temporal exit {pc} closes {enter_target}, active enter is {actual}"
                        )));
                    }
                }
                _ => {}
            }
        }
        if let Some(open) = active.last() {
            return Err(BytecodeTemporalError::InternalInvariant(format!(
                "temporal enter {open} remains open"
            )));
        }
        Ok(context)
    }

    fn control_depths(
        &self,
        range: CodeRange,
    ) -> Result<BTreeMap<u32, usize>, BytecodeTemporalError> {
        let mut starts = BTreeMap::<u32, Vec<(u32, u32)>>::new();
        for pc in range.start..range.end {
            let interval = match self.instruction(pc)? {
                Instruction::IfFalse { end_target, .. } => Some((pc, *end_target, pc)),
                Instruction::WhileFalse {
                    loop_start,
                    end_target,
                } => Some((*loop_start, *end_target, pc)),
                Instruction::TemporalEnter { exit_target, .. } => {
                    let end = exit_target.checked_add(1).ok_or_else(|| {
                        BytecodeTemporalError::InternalInvariant(format!(
                            "temporal exit target {exit_target} overflows"
                        ))
                    })?;
                    Some((pc, end, pc))
                }
                _ => None,
            };
            if let Some((start, end, marker)) = interval {
                starts.entry(start).or_default().push((end, marker));
            }
        }
        for intervals in starts.values_mut() {
            intervals.sort_by(|left, right| right.0.cmp(&left.0).then(left.1.cmp(&right.1)));
        }

        let mut active = Vec::<(u32, u32)>::new();
        let mut marker_depths = BTreeMap::new();
        for pc in range.start..range.end {
            while active.last().is_some_and(|(end, _)| *end <= pc) {
                active.pop();
            }
            if let Some(intervals) = starts.get(&pc) {
                for &(end, marker) in intervals {
                    active.push((end, marker));
                    marker_depths.insert(marker, active.len());
                }
            }
        }
        Ok(marker_depths)
    }

    fn enqueue_fallthrough(
        &mut self,
        pc: u32,
        range: CodeRange,
        queue: &mut VecDeque<u32>,
    ) -> Result<(), BytecodeTemporalError> {
        self.enqueue_target(pc, pc.checked_add(1), range, queue)
    }

    fn enqueue_target(
        &mut self,
        from: u32,
        target: Option<u32>,
        range: CodeRange,
        queue: &mut VecDeque<u32>,
    ) -> Result<(), BytecodeTemporalError> {
        self.cfg_work.charge(1)?;
        let target = target.ok_or_else(|| {
            BytecodeTemporalError::InternalInvariant(format!(
                "successor of instruction {from} overflows u32"
            ))
        })?;
        if target < range.start || target >= range.end {
            return Err(BytecodeTemporalError::InternalInvariant(format!(
                "instruction {from} has out-of-unit successor {target}"
            )));
        }
        queue.push_back(target);
        Ok(())
    }

    fn unit_range(&self, unit: BytecodeUnit) -> Result<CodeRange, BytecodeTemporalError> {
        match unit {
            BytecodeUnit::Main => Ok(self.program.main),
            BytecodeUnit::Quote(id) => self
                .program
                .quotations
                .get(id.as_u64() as usize)
                .map(|entry| entry.range)
                .ok_or_else(|| {
                    BytecodeTemporalError::InternalInvariant(format!(
                        "quotation {} has no code range",
                        id.as_u64()
                    ))
                }),
            BytecodeUnit::Procedure(id) => self
                .program
                .procedures
                .get(id.index())
                .map(|entry| entry.range)
                .ok_or_else(|| {
                    BytecodeTemporalError::InternalInvariant(format!(
                        "procedure {} has no code range",
                        id.as_u32()
                    ))
                }),
        }
    }

    fn instruction(&self, pc: u32) -> Result<&Instruction, BytecodeTemporalError> {
        self.program.instructions.get(pc as usize).ok_or_else(|| {
            BytecodeTemporalError::InternalInvariant(format!(
                "instruction {pc} is absent after structural validation"
            ))
        })
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
}

#[derive(Debug, Clone, Copy)]
struct WorkCounter {
    phase: &'static str,
    limit: usize,
    used: usize,
}

impl WorkCounter {
    const fn new(phase: &'static str, limit: usize) -> Self {
        Self {
            phase,
            limit,
            used: 0,
        }
    }

    fn charge(&mut self, amount: usize) -> Result<(), BytecodeTemporalError> {
        let Some(next) = self.used.checked_add(amount) else {
            return Err(BytecodeTemporalError::WorkLimitExceeded {
                phase: self.phase,
                limit: self.limit,
            });
        };
        if next > self.limit {
            return Err(BytecodeTemporalError::WorkLimitExceeded {
                phase: self.phase,
                limit: self.limit,
            });
        }
        self.used = next;
        Ok(())
    }
}

fn choose_excess(
    current: Option<(usize, VerificationSite)>,
    depth: usize,
    site: VerificationSite,
) -> Option<(usize, VerificationSite)> {
    match current {
        None => Some((depth, site)),
        Some((known_depth, known_site)) => {
            let known_key = (known_depth, known_site.unit, known_site.instruction);
            let candidate_key = (depth, site.unit, site.instruction);
            if candidate_key < known_key {
                Some((depth, site))
            } else {
                Some((known_depth, known_site))
            }
        }
    }
}

fn primitive_boundary(opcode: OpCode) -> Option<PrimitiveBoundary> {
    if TemporalIrCompiler::supports_opcode(opcode) {
        return None;
    }
    Some(match opcode {
        OpCode::Pick
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
        | OpCode::Assert
        | OpCode::Pack
        | OpCode::Unpack => PrimitiveBoundary::Dynamic,
        OpCode::VecNew
        | OpCode::VecPush
        | OpCode::VecPop
        | OpCode::VecGet
        | OpCode::VecSet
        | OpCode::VecLen
        | OpCode::HashNew
        | OpCode::HashPut
        | OpCode::HashGet
        | OpCode::HashDel
        | OpCode::HashHas
        | OpCode::HashLen
        | OpCode::SetNew
        | OpCode::SetAdd
        | OpCode::SetHas
        | OpCode::SetDel
        | OpCode::SetLen
        | OpCode::BufferNew
        | OpCode::BufferFromStack
        | OpCode::BufferToStack
        | OpCode::BufferLen
        | OpCode::BufferReadByte
        | OpCode::BufferWriteByte
        | OpCode::BufferFree => PrimitiveBoundary::RuntimeState,
        OpCode::Input
        | OpCode::FFICall
        | OpCode::FFICallNamed
        | OpCode::FileOpen
        | OpCode::FileRead
        | OpCode::FileWrite
        | OpCode::FileSeek
        | OpCode::FileFlush
        | OpCode::FileClose
        | OpCode::FileExists
        | OpCode::FileSize
        | OpCode::TcpConnect
        | OpCode::SocketSend
        | OpCode::SocketRecv
        | OpCode::SocketClose
        | OpCode::ProcExec
        | OpCode::Clock
        | OpCode::Sleep
        | OpCode::Random => PrimitiveBoundary::ExternalEffect,
        OpCode::Nop
        | OpCode::Halt
        | OpCode::Pop
        | OpCode::Dup
        | OpCode::Swap
        | OpCode::Over
        | OpCode::Rot
        | OpCode::Depth
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
        | OpCode::Paradox
        | OpCode::Index
        | OpCode::Store
        | OpCode::Output
        | OpCode::Emit => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Procedure, Program, Stmt};
    use crate::bytecode_vm::BytecodeVm;
    use crate::core::{OutputItem, PagedMemory, Value};
    use crate::hir::HirProgram;
    use crate::source::{SourceId, SourceSpan, TextRange};

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
        let hir = HirProgram::resolve(program).expect("test HIR resolves");
        BytecodeProgram::compile(&hir).expect("test bytecode compiles")
    }

    fn assert_ir_parity(program: &Program, config: TemporalIrConfig) -> TemporalIr {
        let source = TemporalIrCompiler::compile(program, config).expect("source IR lowers");
        let native = lower_bytecode_temporal_ir(&bytecode(program), config)
            .expect("native bytecode IR lowers");
        assert_eq!(native.to_smt2(false), source.to_smt2(false));
        assert_eq!(native.expressions, source.expressions);
        assert_eq!(native.observations, source.observations);
        assert_eq!(native.completeness, source.completeness);
        native
    }

    #[test]
    fn acyclic_calls_control_and_scope_are_complete() {
        let mut program = Program::new();
        program.procedures.push(procedure(
            "read",
            vec![
                Stmt::Push(Value::new(0)),
                Stmt::Op(OpCode::Oracle),
                Stmt::Op(OpCode::Pop),
            ],
        ));
        program.body = vec![
            Stmt::Call {
                name: "read".to_string(),
            },
            Stmt::Push(Value::ONE),
            Stmt::If {
                then_branch: vec![Stmt::TemporalScope {
                    base: 0,
                    size: 1,
                    cell_bits: 64,
                    body: vec![Stmt::Op(OpCode::Nop)],
                }],
                else_branch: Some(vec![Stmt::Op(OpCode::Nop)]),
            },
        ];

        let analysis =
            analyze_bytecode_temporal(&bytecode(&program), TemporalIrConfig::default()).unwrap();
        assert_eq!(analysis.disposition(), BytecodeTemporalDisposition::Ready);
        assert!(analysis.legacy_lowering_compatible());
        assert!(analysis.has_complete_supported_cfg());
        assert_eq!(analysis.stack_semantics, StackSemanticStatus::NotChecked);
        assert_eq!(analysis.reachable_units.len(), 2);
    }

    #[test]
    fn bounded_loop_is_unknown_but_legacy_compatible() {
        let mut program = Program::new();
        program.body.push(Stmt::While {
            cond: vec![Stmt::Push(Value::ZERO)],
            body: vec![Stmt::Op(OpCode::Nop)],
        });
        let analysis =
            analyze_bytecode_temporal(&bytecode(&program), TemporalIrConfig::default()).unwrap();
        assert_eq!(analysis.disposition(), BytecodeTemporalDisposition::Unknown);
        assert!(analysis.legacy_lowering_compatible());
        assert!(!analysis.has_complete_supported_cfg());
        assert!(matches!(
            analysis.issues.as_slice(),
            [BytecodeTemporalIssue {
                kind: BytecodeTemporalIssueKind::BoundedLoop { .. },
                ..
            }]
        ));
    }

    #[test]
    fn unreachable_external_effect_after_halt_is_not_executable() {
        let mut program = Program::new();
        program.body = vec![Stmt::Op(OpCode::Halt), Stmt::Op(OpCode::Clock)];
        let analysis =
            analyze_bytecode_temporal(&bytecode(&program), TemporalIrConfig::default()).unwrap();
        assert_eq!(analysis.disposition(), BytecodeTemporalDisposition::Ready);
        assert_eq!(analysis.reachable_instructions, 1);
    }

    #[test]
    fn reachable_unsupported_primitive_retains_source_site() {
        let span = SourceSpan::new(SourceId::new(4), TextRange::new(8, 13));
        let mut artifact = bytecode(&Program::new());
        artifact.instructions = vec![Instruction::Primitive(OpCode::Clock), Instruction::Return];
        artifact.main = CodeRange { start: 0, end: 2 };
        artifact.source_map = vec![crate::bytecode::SourceMapEntry {
            instruction: 0,
            span,
        }];
        let analysis = analyze_bytecode_temporal(&artifact, TemporalIrConfig::default()).unwrap();
        assert_eq!(
            analysis.disposition(),
            BytecodeTemporalDisposition::Unsupported
        );
        assert_eq!(analysis.issues[0].site.source, Some(span));
        assert!(matches!(
            analysis.issues[0].kind,
            BytecodeTemporalIssueKind::UnsupportedPrimitive {
                opcode: OpCode::Clock,
                boundary: PrimitiveBoundary::ExternalEffect
            }
        ));
        let error = lower_bytecode_temporal_ir(&artifact, TemporalIrConfig::default()).unwrap_err();
        assert_eq!(error.site.and_then(|site| site.source), Some(span));
        assert!(matches!(
            error.kind,
            BytecodeTemporalLoweringErrorKind::Unsupported(
                BytecodeTemporalIssueKind::UnsupportedPrimitive {
                    opcode: OpCode::Clock,
                    boundary: PrimitiveBoundary::ExternalEffect
                }
            )
        ));
    }

    #[test]
    fn quotation_identity_is_preserved_as_unsupported_value() {
        let mut program = Program::new();
        program.quotes.push(vec![Stmt::Op(OpCode::Nop)]);
        program.body.push(Stmt::PushQuote(QuoteId::new(0)));
        let analysis =
            analyze_bytecode_temporal(&bytecode(&program), TemporalIrConfig::default()).unwrap();
        assert_eq!(analysis.referenced_quotations, vec![QuoteId::new(0)]);
        assert!(matches!(
            analysis.issues[0].kind,
            BytecodeTemporalIssueKind::QuotationValue { target }
                if target == QuoteId::new(0)
        ));
    }

    #[test]
    fn recursive_procedure_component_is_unknown_and_not_delegated() {
        let mut program = Program::new();
        program.procedures.push(procedure(
            "again",
            vec![Stmt::Call {
                name: "again".to_string(),
            }],
        ));
        program.body.push(Stmt::Call {
            name: "again".to_string(),
        });
        let analysis =
            analyze_bytecode_temporal(&bytecode(&program), TemporalIrConfig::default()).unwrap();
        assert_eq!(analysis.disposition(), BytecodeTemporalDisposition::Unknown);
        assert!(!analysis.legacy_lowering_compatible());
        assert!(analysis.issues.iter().any(|issue| matches!(
            &issue.kind,
            BytecodeTemporalIssueKind::RecursiveProcedures { members }
                if members.len() == 1 && members[0].index() == 0
        )));
        let error = lower_bytecode_temporal_ir(&bytecode(&program), TemporalIrConfig::default())
            .unwrap_err();
        assert!(matches!(
            error.kind,
            BytecodeTemporalLoweringErrorKind::Unsupported(
                BytecodeTemporalIssueKind::RecursiveProcedures { ref members }
            ) if members.len() == 1 && members[0].index() == 0
        ));
        assert!(error.site.is_some());
    }

    #[test]
    fn nested_and_out_of_memory_temporal_scopes_are_explicit() {
        let mut program = Program::new();
        program.body.push(Stmt::TemporalScope {
            base: 0,
            size: 1,
            cell_bits: 64,
            body: vec![Stmt::TemporalScope {
                base: 1,
                size: 2,
                cell_bits: 64,
                body: vec![Stmt::Op(OpCode::Nop)],
            }],
        });
        let config = TemporalIrConfig {
            memory_cells: 2,
            ..TemporalIrConfig::default()
        };
        let analysis = analyze_bytecode_temporal(&bytecode(&program), config).unwrap();
        assert_eq!(
            analysis.disposition(),
            BytecodeTemporalDisposition::Unsupported
        );
        assert!(analysis.issues.iter().any(|issue| matches!(
            issue.kind,
            BytecodeTemporalIssueKind::NestedTemporalScope { outer_enter: 0 }
        )));
        assert!(analysis.issues.iter().any(|issue| matches!(
            issue.kind,
            BytecodeTemporalIssueKind::TemporalScopeExceedsMemory {
                base: 1,
                size: 2,
                memory_cells: 2
            }
        )));
    }

    #[test]
    fn legacy_depth_and_work_limits_fail_closed() {
        let mut program = Program::new();
        program.procedures.push(procedure("leaf", Vec::new()));
        program.procedures.push(procedure(
            "middle",
            vec![Stmt::Call {
                name: "leaf".to_string(),
            }],
        ));
        program.body.push(Stmt::Call {
            name: "middle".to_string(),
        });
        let artifact = bytecode(&program);
        let limits = BytecodeTemporalLimits {
            max_legacy_call_depth: 1,
            ..BytecodeTemporalLimits::default()
        };
        let analysis =
            analyze_bytecode_temporal_with_limits(&artifact, TemporalIrConfig::default(), limits)
                .unwrap();
        assert!(analysis.issues.iter().any(|issue| matches!(
            issue.kind,
            BytecodeTemporalIssueKind::LegacyCallDepthExceeded { depth: 2, limit: 1 }
        )));

        let tiny = BytecodeTemporalLimits {
            max_cfg_steps: 0,
            ..BytecodeTemporalLimits::default()
        };
        assert!(matches!(
            analyze_bytecode_temporal_with_limits(&artifact, TemporalIrConfig::default(), tiny),
            Err(BytecodeTemporalError::WorkLimitExceeded {
                phase: "temporal CFG",
                limit: 0
            })
        ));
    }

    #[test]
    fn primitive_boundary_tracks_the_ir_compiler_exhaustively() {
        for &opcode in OpCode::ALL {
            assert_eq!(
                primitive_boundary(opcode).is_none(),
                TemporalIrCompiler::supports_opcode(opcode),
                "temporal boundary drifted for {}",
                opcode.name()
            );
        }
    }

    #[test]
    fn native_ir_matches_source_for_calls_if_and_inherited_temporal_scope() {
        let mut program = Program::new();
        program.procedures.push(procedure(
            "copy_cell",
            vec![
                Stmt::Push(Value::ZERO),
                Stmt::Op(OpCode::Oracle),
                Stmt::Push(Value::ZERO),
                Stmt::Op(OpCode::Prophecy),
            ],
        ));
        program.body = vec![
            Stmt::TemporalScope {
                base: 4,
                size: 2,
                cell_bits: 8,
                body: vec![Stmt::Call {
                    name: "copy_cell".to_string(),
                }],
            },
            Stmt::Push(Value::ONE),
            Stmt::If {
                then_branch: vec![Stmt::Push(Value::new(65)), Stmt::Op(OpCode::Emit)],
                else_branch: Some(vec![Stmt::Push(Value::new(66)), Stmt::Op(OpCode::Emit)]),
            },
        ];
        let config = TemporalIrConfig {
            memory_cells: 16,
            loop_unroll_limit: 3,
            bounds_policy: BoundsPolicy::Wrap,
        };
        let ir = assert_ir_parity(&program, config);
        assert_eq!(ir.completeness, IrCompleteness::Complete);
        assert_eq!(ir.observations.len(), 2);
    }

    #[test]
    fn native_ir_matches_source_bounded_while_exactly() {
        let mut program = Program::new();
        program.body = vec![
            Stmt::Push(Value::ZERO),
            Stmt::While {
                cond: vec![
                    Stmt::Op(OpCode::Dup),
                    Stmt::Push(Value::new(2)),
                    Stmt::Op(OpCode::Lt),
                ],
                body: vec![Stmt::Push(Value::ONE), Stmt::Op(OpCode::Add)],
            },
            Stmt::Op(OpCode::Pop),
        ];
        let config = TemporalIrConfig {
            memory_cells: 8,
            loop_unroll_limit: 2,
            bounds_policy: BoundsPolicy::Wrap,
        };
        let ir = assert_ir_parity(&program, config);
        assert_eq!(
            ir.completeness,
            IrCompleteness::BoundedLoops {
                loop_count: 1,
                unroll_limit: 2
            }
        );
    }

    #[test]
    fn native_lowering_rejects_stack_join_and_cross_call_nested_scope() {
        let mut mismatch = Program::new();
        mismatch.body = vec![
            Stmt::Push(Value::ONE),
            Stmt::If {
                then_branch: vec![Stmt::Push(Value::ONE)],
                else_branch: Some(Vec::new()),
            },
        ];
        let error = lower_bytecode_temporal_ir(&bytecode(&mismatch), TemporalIrConfig::default())
            .unwrap_err();
        assert!(matches!(
            error.kind,
            BytecodeTemporalLoweringErrorKind::StackJoinMismatch {
                construct: "IF",
                when_true: 1,
                when_false: 0
            }
        ));

        let mut nested = Program::new();
        nested.procedures.push(procedure(
            "inner",
            vec![Stmt::TemporalScope {
                base: 1,
                size: 1,
                cell_bits: 64,
                body: vec![Stmt::Op(OpCode::Nop)],
            }],
        ));
        nested.body.push(Stmt::TemporalScope {
            base: 0,
            size: 1,
            cell_bits: 64,
            body: vec![Stmt::Call {
                name: "inner".to_string(),
            }],
        });
        let error = lower_bytecode_temporal_ir(&bytecode(&nested), TemporalIrConfig::default())
            .unwrap_err();
        assert!(matches!(
            error.kind,
            BytecodeTemporalLoweringErrorKind::NestedTemporalScope { .. }
        ));
        assert!(error.site.is_some());
    }

    #[test]
    fn native_lowering_and_bytecode_vm_agree_on_concrete_store_and_output() {
        let mut program = Program::new();
        program.body = vec![
            Stmt::Push(Value::new(42)),
            Stmt::Push(Value::new(3)),
            Stmt::Op(OpCode::Prophecy),
            Stmt::Push(Value::new(3)),
            Stmt::Op(OpCode::PresentRead),
            Stmt::Op(OpCode::Output),
        ];
        let artifact = bytecode(&program);
        let config = TemporalIrConfig {
            memory_cells: 16,
            loop_unroll_limit: 2,
            bounds_policy: BoundsPolicy::Error,
        };
        let ir = lower_bytecode_temporal_ir(&artifact, config).unwrap();
        let execution = BytecodeVm::new()
            .run(&artifact, &PagedMemory::with_size(16).unwrap())
            .unwrap();
        assert_eq!(execution.present.get(3).map(|value| value.val), Some(42));
        assert_eq!(execution.output, vec![OutputItem::Val(Value::new(42))]);

        let IrExprKind::Store { address, value, .. } = ir.expressions[ir.final_memory].kind else {
            panic!("native final memory is not a symbolic store");
        };
        assert!(matches!(
            ir.expressions[address].kind,
            IrExprKind::WordConst(3)
        ));
        assert!(matches!(
            ir.expressions[value].kind,
            IrExprKind::WordConst(42)
        ));
        assert_eq!(ir.observations.len(), 1);
    }

    #[test]
    fn native_lowering_has_source_parity_for_every_supported_primitive() {
        for &opcode in OpCode::ALL {
            let required = match opcode {
                OpCode::Nop | OpCode::Halt | OpCode::Depth | OpCode::Paradox => Some(0),
                OpCode::Pop
                | OpCode::Dup
                | OpCode::Neg
                | OpCode::Abs
                | OpCode::Sign
                | OpCode::Not
                | OpCode::Oracle
                | OpCode::PresentRead
                | OpCode::Output
                | OpCode::Emit => Some(1),
                OpCode::Swap
                | OpCode::Over
                | OpCode::Add
                | OpCode::Sub
                | OpCode::Mul
                | OpCode::Div
                | OpCode::Mod
                | OpCode::Min
                | OpCode::Max
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
                | OpCode::Prophecy
                | OpCode::Index => Some(2),
                OpCode::Rot | OpCode::Store => Some(3),
                _ => None,
            };
            let Some(required) = required else {
                assert!(!TemporalIrCompiler::supports_opcode(opcode));
                continue;
            };
            let mut program = Program::new();
            for value in 1..=required {
                program.body.push(Stmt::Push(Value::new(value as u64)));
            }
            program.body.push(Stmt::Op(opcode));
            assert_ir_parity(
                &program,
                TemporalIrConfig {
                    memory_cells: 16,
                    ..TemporalIrConfig::default()
                },
            );
        }
    }

    #[test]
    fn native_bounded_loop_lowering_is_iterative_at_large_unroll_depth() {
        let mut program = Program::new();
        program.body.push(Stmt::While {
            cond: vec![Stmt::Push(Value::ZERO)],
            body: vec![Stmt::Op(OpCode::Nop)],
        });
        let ir = lower_bytecode_temporal_ir(
            &bytecode(&program),
            TemporalIrConfig {
                memory_cells: 8,
                loop_unroll_limit: 10_000,
                bounds_policy: BoundsPolicy::Wrap,
            },
        )
        .unwrap();
        assert_eq!(
            ir.completeness,
            IrCompleteness::BoundedLoops {
                loop_count: 1,
                unroll_limit: 10_000
            }
        );
    }

    #[test]
    fn native_lowering_does_not_reject_unreachable_effect_after_terminal_call() {
        let mut program = Program::new();
        program
            .procedures
            .push(procedure("stop", vec![Stmt::Op(OpCode::Halt)]));
        program.body = vec![
            Stmt::Call {
                name: "stop".to_string(),
            },
            Stmt::Op(OpCode::Clock),
        ];
        let artifact = bytecode(&program);
        let readiness = analyze_bytecode_temporal(&artifact, TemporalIrConfig::default()).unwrap();
        assert_eq!(
            readiness.disposition(),
            BytecodeTemporalDisposition::Unsupported,
            "the conservative readiness pass intentionally over-approximates call returns"
        );
        let native = lower_bytecode_temporal_ir(&artifact, TemporalIrConfig::default()).unwrap();
        let source = TemporalIrCompiler::compile(&program, TemporalIrConfig::default()).unwrap();
        assert_eq!(native.to_smt2(false), source.to_smt2(false));
    }
}
