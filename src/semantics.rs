//! Mandatory structural stack semantics for resolved HIR.
//!
//! This checker deliberately separates facts proved from source structure from
//! operations that require bytecode/runtime validation.  In particular, a
//! dynamic operation never receives a fabricated fixed stack effect: its
//! minimum operand requirement is checked, and the continuation becomes
//! explicitly not proven.

use crate::ast::{OpCode, QuoteId};
use crate::hir::{CallTarget, ForeignId, HirOwner, HirProgram, HirStmt, HirStmtKind, ProcedureId};
use crate::source::SourceSpan;
use std::collections::VecDeque;

/// Whether a stack summary covers every possible continuation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackCertainty {
    /// Every continuation has a statically known structural stack effect.
    Proven,
    /// A dynamic operation or unresolved recursive call requires validation.
    RuntimeValidationRequired,
}

/// Structural stack effect inferred for one statement list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackSummary {
    /// Minimum words required at entry before the first unknown boundary.
    pub required: usize,
    /// Net effect when proven; otherwise the effect of the proven prefix only.
    pub delta: i64,
    /// Whether the code can continue past the statement list.
    ///
    /// For a non-proven summary this is conservative: `true` means that a
    /// falling-through runtime path may exist.
    pub falls_through: bool,
    /// Static proof status for the complete statement list.
    pub certainty: StackCertainty,
}

impl StackSummary {
    const fn empty() -> Self {
        Self {
            required: 0,
            delta: 0,
            falls_through: true,
            certainty: StackCertainty::Proven,
        }
    }

    const fn unknown(required: usize, proven_prefix_delta: i64) -> Self {
        Self {
            required,
            delta: proven_prefix_delta,
            falls_through: true,
            certainty: StackCertainty::RuntimeValidationRequired,
        }
    }

    /// Returns whether the whole summary has a single static stack effect.
    pub const fn is_proven(self) -> bool {
        matches!(self.certainty, StackCertainty::Proven)
    }
}

/// A breadcrumb within a statement owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextFrame {
    /// Zero-based statement index in the current list.
    Statement(usize),
    /// The true arm of an `IF`.
    ThenBranch,
    /// The false arm of an `IF`.
    ElseBranch,
    /// The condition block of a `WHILE`.
    WhileCondition,
    /// The body block of a `WHILE`.
    WhileBody,
    /// The child list of a scoped block.
    BlockBody,
    /// The child list of a temporal scope.
    TemporalBody,
}

/// Owner and precise structural context attached to a diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticSite {
    /// Independently checked code owner.
    pub owner: HirOwner,
    /// Structural route from the owner root to the statement.
    pub context: Vec<ContextFrame>,
    /// Source location, once the located-HIR migration has attached one.
    pub location: Option<SourceSpan>,
}

impl SemanticSite {
    fn owner_root(owner: HirOwner) -> Self {
        Self {
            owner,
            context: Vec::new(),
            location: None,
        }
    }
}

/// Kind of statically definite semantic failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticErrorKind {
    /// The main body needs words that are not present at program entry.
    DefiniteMainUnderflow {
        /// Number of missing entry words.
        missing: usize,
    },
    /// Two falling-through `IF` arms return different stack rows.
    BranchRowMismatch {
        /// Net effect of the true arm.
        then_delta: i64,
        /// Net effect of the false arm.
        else_delta: i64,
    },
    /// A falling loop condition did not add exactly one condition word.
    LoopConditionDrift {
        /// Net condition-block effect; the required value is `1`.
        delta: i64,
    },
    /// A loop body does not preserve its carried stack row.
    LoopBodyDrift {
        /// Net body effect; the required value is `0`.
        delta: i64,
    },
    /// A typed HIR reference does not name a table entry.
    InvalidReference {
        /// Human-readable reference category and value.
        reference: String,
    },
}

/// A hard semantic error with owner and structural context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticError {
    /// Specific static contradiction.
    pub kind: SemanticErrorKind,
    /// Location in resolved code.
    pub site: SemanticSite,
}

/// Structured control construct requiring later validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlConstruct {
    /// An `IF` whose complete branch row is not statically known.
    If,
    /// A `WHILE` whose carried row is not statically known.
    While,
}

/// Kind of explicit runtime, bytecode-validation, or link obligation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObligationKind {
    /// A primitive has value-dependent stack behaviour.
    DynamicOpcode {
        /// Dynamic primitive.
        opcode: OpCode,
        /// Operand words definitely consumed before its dynamic behaviour.
        minimum_inputs: usize,
    },
    /// A recursive procedure SCC cannot be assigned an acyclic summary.
    RecursiveProcedures {
        /// All members of the recursive SCC, in stable table order.
        members: Vec<ProcedureId>,
    },
    /// A call targets a procedure without a proven summary.
    ProcedureCallValidation {
        /// Called procedure.
        target: ProcedureId,
    },
    /// Structured control depends on a non-proven child summary.
    StructuredControlValidation {
        /// Construct requiring bytecode/runtime validation.
        construct: ControlConstruct,
    },
    /// The foreign signature is structurally exact, but execution/link support
    /// must be established before compiled code can run.
    ForeignLink {
        /// Foreign declaration table entry.
        target: ForeignId,
        /// Source-level foreign name.
        name: String,
        /// Exact signature input width.
        inputs: usize,
        /// Exact signature output width.
        outputs: usize,
    },
}

/// A non-static semantic obligation with owner and structural context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticObligation {
    /// Required later validation.
    pub kind: ObligationKind,
    /// Location in resolved code, or the procedure root for an SCC.
    pub site: SemanticSite,
}

/// Summary for a stable procedure table entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcedureSummary {
    /// Procedure identity.
    pub id: ProcedureId,
    /// Inferred structural effect.
    pub stack: StackSummary,
}

/// Summary for a stable quotation table entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuoteSummary {
    /// Quotation identity.
    pub id: QuoteId,
    /// Inferred structural effect when invoked with a runtime-provided row.
    pub stack: StackSummary,
}

/// Complete mandatory semantic-check report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticsReport {
    /// Top-level body summary.
    pub main: StackSummary,
    /// Procedure summaries in HIR table order.
    pub procedures: Vec<ProcedureSummary>,
    /// Quotation summaries in HIR table order.
    pub quotes: Vec<QuoteSummary>,
    /// Definite errors that make interpretation unsafe.
    pub errors: Vec<SemanticError>,
    /// Honest validation/link obligations that prevent a fully static result.
    pub obligations: Vec<SemanticObligation>,
}

impl SemanticsReport {
    /// Interpreter admission rejects only definite semantic errors.
    pub fn is_accepted_for_interpreter(&self) -> bool {
        self.errors.is_empty()
    }

    /// A fully static program has neither hard errors nor deferred obligations.
    pub fn is_fully_static(&self) -> bool {
        self.errors.is_empty() && self.obligations.is_empty()
    }
}

/// Perform mandatory structural semantic checking over resolved HIR.
pub fn check(program: &HirProgram) -> SemanticsReport {
    Checker::new(program).check()
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

/// Exhaustive stack classifier for every primitive.
///
/// The 23 dynamic cases are intentionally not delegated to
/// `OpCode::stack_effect`: its values are implementation minima, not proofs.
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

struct Checker<'a> {
    program: &'a HirProgram,
    errors: Vec<SemanticError>,
    obligations: Vec<SemanticObligation>,
    procedure_summaries: Vec<Option<StackSummary>>,
}

impl<'a> Checker<'a> {
    fn new(program: &'a HirProgram) -> Self {
        Self {
            program,
            errors: Vec::new(),
            obligations: Vec::new(),
            procedure_summaries: vec![None; program.procedures.len()],
        }
    }

    fn check(mut self) -> SemanticsReport {
        self.check_table_identities();
        let graph = procedure_graph(self.program, &mut self.errors);
        let components = strongly_connected_components(&graph);
        let mut recursive = vec![false; graph.len()];

        for mut component in components {
            component.sort_unstable();
            let is_recursive = component.len() > 1
                || component
                    .first()
                    .is_some_and(|node| graph[*node].contains(node));
            if !is_recursive {
                continue;
            }

            let members: Vec<_> = component
                .iter()
                .filter_map(|index| ProcedureId::try_from_index(*index))
                .collect();
            for index in &component {
                recursive[*index] = true;
                self.procedure_summaries[*index] = Some(StackSummary::unknown(0, 0));
            }
            let owner = members
                .first()
                .copied()
                .map(HirOwner::Procedure)
                .unwrap_or(HirOwner::ProgramBody);
            self.obligations.push(SemanticObligation {
                kind: ObligationKind::RecursiveProcedures { members },
                site: SemanticSite::owner_root(owner),
            });
        }

        self.infer_acyclic_procedures(&graph, &recursive);

        // Recursive bodies still receive independent local checking after all
        // non-recursive callees have summaries.  Their SCC summary remains
        // unknown; no result here can be used as a hard proof at a call site.
        for (index, procedure) in self.program.procedures.iter().enumerate() {
            if recursive[index] {
                let owner = HirOwner::Procedure(procedure.id);
                let _ = self.analyze_block(&procedure.body, owner, &[]);
            }
        }

        let mut quotes = Vec::with_capacity(self.program.quotes.len());
        for quote in &self.program.quotes {
            let owner = HirOwner::Quote(quote.id);
            let stack = self.analyze_block(&quote.body, owner, &[]);
            quotes.push(QuoteSummary {
                id: quote.id,
                stack,
            });
        }

        let main = self.analyze_block(&self.program.body, HirOwner::ProgramBody, &[]);
        if main.required > 0 {
            self.errors.push(SemanticError {
                kind: SemanticErrorKind::DefiniteMainUnderflow {
                    missing: main.required,
                },
                site: SemanticSite::owner_root(HirOwner::ProgramBody),
            });
        }

        let procedures = self
            .program
            .procedures
            .iter()
            .enumerate()
            .map(|(index, procedure)| ProcedureSummary {
                id: procedure.id,
                stack: self.procedure_summaries[index]
                    .unwrap_or_else(|| StackSummary::unknown(0, 0)),
            })
            .collect();

        SemanticsReport {
            main,
            procedures,
            quotes,
            errors: self.errors,
            obligations: self.obligations,
        }
    }

    fn check_table_identities(&mut self) {
        for (index, procedure) in self.program.procedures.iter().enumerate() {
            if procedure.id.index() != index {
                self.errors.push(SemanticError {
                    kind: SemanticErrorKind::InvalidReference {
                        reference: format!(
                            "procedure table entry {index} stores ID {}",
                            procedure.id
                        ),
                    },
                    site: SemanticSite::owner_root(HirOwner::Procedure(procedure.id)),
                });
            }
        }
        for (index, foreign) in self.program.foreigns.iter().enumerate() {
            if foreign.id.index() != index {
                self.errors.push(SemanticError {
                    kind: SemanticErrorKind::InvalidReference {
                        reference: format!("foreign table entry {index} stores ID {}", foreign.id),
                    },
                    site: SemanticSite::owner_root(HirOwner::ProgramBody),
                });
            }
        }
        for (index, quote) in self.program.quotes.iter().enumerate() {
            if quote.id.as_u64() != index as u64 {
                self.errors.push(SemanticError {
                    kind: SemanticErrorKind::InvalidReference {
                        reference: format!(
                            "quotation table entry {index} stores ID {}",
                            quote.id.as_u64()
                        ),
                    },
                    site: SemanticSite::owner_root(HirOwner::Quote(quote.id)),
                });
            }
        }
    }

    fn infer_acyclic_procedures(&mut self, graph: &[Vec<usize>], recursive: &[bool]) {
        let count = graph.len();
        let mut dependents = vec![Vec::new(); count];
        let mut pending = vec![0usize; count];

        for (caller, dependencies) in graph.iter().enumerate() {
            if recursive[caller] {
                continue;
            }
            pending[caller] = dependencies
                .iter()
                .filter(|callee| !recursive[**callee])
                .count();
            for callee in dependencies {
                if !recursive[*callee] {
                    dependents[*callee].push(caller);
                }
            }
        }

        let mut ready = VecDeque::new();
        for index in 0..count {
            if !recursive[index] && pending[index] == 0 {
                ready.push_back(index);
            }
        }

        while let Some(index) = ready.pop_front() {
            let procedure = &self.program.procedures[index];
            let owner = HirOwner::Procedure(procedure.id);
            let summary = self.analyze_block(&procedure.body, owner, &[]);
            self.procedure_summaries[index] = Some(summary);

            for dependent in dependents[index].iter().copied() {
                pending[dependent] -= 1;
                if pending[dependent] == 0 {
                    ready.push_back(dependent);
                }
            }
        }

        // Every remaining node indicates malformed graph bookkeeping.  Keep
        // it explicitly unknown rather than inventing a callable effect.
        for index in 0..count {
            if self.procedure_summaries[index].is_none() {
                self.procedure_summaries[index] = Some(StackSummary::unknown(0, 0));
            }
        }
    }

    fn analyze_block(
        &mut self,
        statements: &[HirStmt],
        owner: HirOwner,
        context: &[ContextFrame],
    ) -> StackSummary {
        let mut result = StackSummary::empty();

        for (index, statement) in statements.iter().enumerate() {
            let mut statement_context = context.to_vec();
            statement_context.push(ContextFrame::Statement(index));
            let statement_summary = self.analyze_statement(statement, owner, &statement_context);

            if result.falls_through && result.is_proven() {
                result = compose(result, statement_summary);
            }
        }
        result
    }

    fn analyze_statement(
        &mut self,
        statement: &HirStmt,
        owner: HirOwner,
        context: &[ContextFrame],
    ) -> StackSummary {
        match &statement.kind {
            HirStmtKind::Op(opcode) => match opcode_shape(*opcode) {
                OpcodeShape::Fixed {
                    inputs,
                    outputs,
                    terminal,
                } => StackSummary {
                    required: inputs,
                    delta: usize_delta(outputs, inputs),
                    falls_through: !terminal,
                    certainty: StackCertainty::Proven,
                },
                OpcodeShape::Dynamic { minimum_inputs } => {
                    self.obligations.push(SemanticObligation {
                        kind: ObligationKind::DynamicOpcode {
                            opcode: *opcode,
                            minimum_inputs,
                        },
                        site: site(owner, context, statement.location),
                    });
                    StackSummary::unknown(minimum_inputs, -(minimum_inputs as i64))
                }
            },
            HirStmtKind::Push(_)
            | HirStmtKind::PushConstant { .. }
            | HirStmtKind::ReadTemporal { .. }
            | HirStmtKind::PushQuote(_) => StackSummary {
                required: 0,
                delta: 1,
                falls_through: true,
                certainty: StackCertainty::Proven,
            },
            HirStmtKind::Block(body) => {
                let child = child_context(context, ContextFrame::BlockBody);
                self.analyze_block(body, owner, &child)
            }
            HirStmtKind::TemporalScope { body, .. } => {
                let child = child_context(context, ContextFrame::TemporalBody);
                self.analyze_block(body, owner, &child)
            }
            HirStmtKind::If {
                then_branch,
                else_branch,
            } => self.analyze_if(
                statement,
                then_branch,
                else_branch.as_deref(),
                owner,
                context,
            ),
            HirStmtKind::While { cond, body } => {
                self.analyze_while(statement, cond, body, owner, context)
            }
            HirStmtKind::Call { target } => self.analyze_call(statement, *target, owner, context),
        }
    }

    fn analyze_if(
        &mut self,
        statement: &HirStmt,
        then_branch: &[HirStmt],
        else_branch: Option<&[HirStmt]>,
        owner: HirOwner,
        context: &[ContextFrame],
    ) -> StackSummary {
        let then_context = child_context(context, ContextFrame::ThenBranch);
        let then_summary = self.analyze_block(then_branch, owner, &then_context);
        let else_summary = if let Some(else_branch) = else_branch {
            let else_context = child_context(context, ContextFrame::ElseBranch);
            self.analyze_block(else_branch, owner, &else_context)
        } else {
            StackSummary::empty()
        };

        let required = 1usize
            .max(then_summary.required.saturating_add(1))
            .max(else_summary.required.saturating_add(1));

        if then_summary.is_proven()
            && else_summary.is_proven()
            && then_summary.falls_through
            && else_summary.falls_through
            && then_summary.delta != else_summary.delta
        {
            self.errors.push(SemanticError {
                kind: SemanticErrorKind::BranchRowMismatch {
                    then_delta: then_summary.delta,
                    else_delta: else_summary.delta,
                },
                site: site(owner, context, statement.location),
            });
        }

        if !then_summary.is_proven() || !else_summary.is_proven() {
            self.obligations.push(SemanticObligation {
                kind: ObligationKind::StructuredControlValidation {
                    construct: ControlConstruct::If,
                },
                site: site(owner, context, statement.location),
            });
            return StackSummary::unknown(required, -1);
        }

        let (falls_through, branch_delta) =
            match (then_summary.falls_through, else_summary.falls_through) {
                (true, true) => (true, then_summary.delta),
                (true, false) => (true, then_summary.delta),
                (false, true) => (true, else_summary.delta),
                (false, false) => (false, 0),
            };

        StackSummary {
            required,
            delta: -1 + branch_delta,
            falls_through,
            certainty: StackCertainty::Proven,
        }
    }

    fn analyze_while(
        &mut self,
        statement: &HirStmt,
        cond: &[HirStmt],
        body: &[HirStmt],
        owner: HirOwner,
        context: &[ContextFrame],
    ) -> StackSummary {
        let cond_context = child_context(context, ContextFrame::WhileCondition);
        let cond_summary = self.analyze_block(cond, owner, &cond_context);
        let body_context = child_context(context, ContextFrame::WhileBody);
        let body_summary = self.analyze_block(body, owner, &body_context);

        if cond_summary.is_proven() && cond_summary.falls_through && cond_summary.delta != 1 {
            self.errors.push(SemanticError {
                kind: SemanticErrorKind::LoopConditionDrift {
                    delta: cond_summary.delta,
                },
                site: site(owner, context, statement.location),
            });
        }
        if body_summary.is_proven() && body_summary.delta != 0 {
            self.errors.push(SemanticError {
                kind: SemanticErrorKind::LoopBodyDrift {
                    delta: body_summary.delta,
                },
                site: site(owner, context, statement.location),
            });
        }

        if !cond_summary.is_proven()
            || !body_summary.is_proven()
            || (cond_summary.falls_through && cond_summary.delta != 1)
            || body_summary.delta != 0
        {
            self.obligations.push(SemanticObligation {
                kind: ObligationKind::StructuredControlValidation {
                    construct: ControlConstruct::While,
                },
                site: site(owner, context, statement.location),
            });
            let required = if cond_summary.is_proven() && cond_summary.delta == 1 {
                cond_summary.required.max(body_summary.required)
            } else {
                cond_summary.required
            };
            return StackSummary::unknown(required, 0);
        }

        if !cond_summary.falls_through {
            return cond_summary;
        }

        StackSummary {
            required: cond_summary.required.max(body_summary.required),
            delta: 0,
            falls_through: true,
            certainty: StackCertainty::Proven,
        }
    }

    fn analyze_call(
        &mut self,
        statement: &HirStmt,
        target: CallTarget,
        owner: HirOwner,
        context: &[ContextFrame],
    ) -> StackSummary {
        match target {
            CallTarget::Procedure(target) => {
                let Some(summary) = self
                    .procedure_summaries
                    .get(target.index())
                    .and_then(|summary| *summary)
                else {
                    self.errors.push(SemanticError {
                        kind: SemanticErrorKind::InvalidReference {
                            reference: format!("procedure {target}"),
                        },
                        site: site(owner, context, statement.location),
                    });
                    return StackSummary::unknown(0, 0);
                };
                if !summary.is_proven() {
                    self.obligations.push(SemanticObligation {
                        kind: ObligationKind::ProcedureCallValidation { target },
                        site: site(owner, context, statement.location),
                    });
                }
                summary
            }
            CallTarget::Foreign(target) => {
                let Some(foreign) = self.program.foreigns.get(target.index()) else {
                    self.errors.push(SemanticError {
                        kind: SemanticErrorKind::InvalidReference {
                            reference: format!("foreign {target}"),
                        },
                        site: site(owner, context, statement.location),
                    });
                    return StackSummary::unknown(0, 0);
                };
                let signature = &foreign.declaration.signature;
                let inputs = signature.input_stack_size();
                let outputs = signature.output_stack_size();
                self.obligations.push(SemanticObligation {
                    kind: ObligationKind::ForeignLink {
                        target,
                        name: signature.name.clone(),
                        inputs,
                        outputs,
                    },
                    site: site(owner, context, statement.location),
                });
                StackSummary {
                    required: inputs,
                    delta: usize_delta(outputs, inputs),
                    falls_through: true,
                    certainty: StackCertainty::Proven,
                }
            }
        }
    }
}

fn site(owner: HirOwner, context: &[ContextFrame], location: Option<SourceSpan>) -> SemanticSite {
    SemanticSite {
        owner,
        context: context.to_vec(),
        location,
    }
}

fn child_context(context: &[ContextFrame], frame: ContextFrame) -> Vec<ContextFrame> {
    let mut child = context.to_vec();
    child.push(frame);
    child
}

fn usize_delta(outputs: usize, inputs: usize) -> i64 {
    let outputs = i64::try_from(outputs).unwrap_or(i64::MAX);
    let inputs = i64::try_from(inputs).unwrap_or(i64::MAX);
    outputs.saturating_sub(inputs)
}

fn compose(first: StackSummary, second: StackSummary) -> StackSummary {
    debug_assert!(first.falls_through && first.is_proven());
    let shifted_requirement = if first.delta >= 0 {
        second.required.saturating_sub(first.delta as usize)
    } else {
        second
            .required
            .saturating_add(first.delta.unsigned_abs() as usize)
    };
    StackSummary {
        required: first.required.max(shifted_requirement),
        delta: first.delta.saturating_add(second.delta),
        falls_through: second.falls_through,
        certainty: second.certainty,
    }
}

fn procedure_graph(program: &HirProgram, errors: &mut Vec<SemanticError>) -> Vec<Vec<usize>> {
    let mut graph = vec![Vec::new(); program.procedures.len()];
    for (caller, procedure) in program.procedures.iter().enumerate() {
        collect_procedure_calls(
            &procedure.body,
            HirOwner::Procedure(procedure.id),
            &mut graph[caller],
            program.procedures.len(),
            errors,
            &[],
        );
        graph[caller].sort_unstable();
        graph[caller].dedup();
    }
    graph
}

fn collect_procedure_calls(
    statements: &[HirStmt],
    owner: HirOwner,
    output: &mut Vec<usize>,
    procedure_count: usize,
    errors: &mut Vec<SemanticError>,
    context: &[ContextFrame],
) {
    for (index, statement) in statements.iter().enumerate() {
        let mut statement_context = context.to_vec();
        statement_context.push(ContextFrame::Statement(index));
        match &statement.kind {
            HirStmtKind::Call {
                target: CallTarget::Procedure(target),
            } => {
                if target.index() < procedure_count {
                    output.push(target.index());
                } else {
                    errors.push(SemanticError {
                        kind: SemanticErrorKind::InvalidReference {
                            reference: format!("procedure {target}"),
                        },
                        site: site(owner, &statement_context, statement.location),
                    });
                }
            }
            HirStmtKind::If {
                then_branch,
                else_branch,
            } => {
                let then_context = child_context(&statement_context, ContextFrame::ThenBranch);
                collect_procedure_calls(
                    then_branch,
                    owner,
                    output,
                    procedure_count,
                    errors,
                    &then_context,
                );
                if let Some(else_branch) = else_branch {
                    let else_context = child_context(&statement_context, ContextFrame::ElseBranch);
                    collect_procedure_calls(
                        else_branch,
                        owner,
                        output,
                        procedure_count,
                        errors,
                        &else_context,
                    );
                }
            }
            HirStmtKind::While { cond, body } => {
                let cond_context = child_context(&statement_context, ContextFrame::WhileCondition);
                collect_procedure_calls(
                    cond,
                    owner,
                    output,
                    procedure_count,
                    errors,
                    &cond_context,
                );
                let body_context = child_context(&statement_context, ContextFrame::WhileBody);
                collect_procedure_calls(
                    body,
                    owner,
                    output,
                    procedure_count,
                    errors,
                    &body_context,
                );
            }
            HirStmtKind::Block(body) => {
                let body_context = child_context(&statement_context, ContextFrame::BlockBody);
                collect_procedure_calls(
                    body,
                    owner,
                    output,
                    procedure_count,
                    errors,
                    &body_context,
                );
            }
            HirStmtKind::TemporalScope { body, .. } => {
                let body_context = child_context(&statement_context, ContextFrame::TemporalBody);
                collect_procedure_calls(
                    body,
                    owner,
                    output,
                    procedure_count,
                    errors,
                    &body_context,
                );
            }
            HirStmtKind::Op(_)
            | HirStmtKind::Push(_)
            | HirStmtKind::PushConstant { .. }
            | HirStmtKind::ReadTemporal { .. }
            | HirStmtKind::PushQuote(_)
            | HirStmtKind::Call {
                target: CallTarget::Foreign(_),
            } => {}
        }
    }
}

/// Iterative Kosaraju SCC decomposition, avoiding host recursion on large
/// procedure graphs.
fn strongly_connected_components(graph: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut reverse = vec![Vec::new(); graph.len()];
    for (from, edges) in graph.iter().enumerate() {
        for to in edges {
            reverse[*to].push(from);
        }
    }

    let mut seen = vec![false; graph.len()];
    let mut order = Vec::with_capacity(graph.len());
    for root in 0..graph.len() {
        if seen[root] {
            continue;
        }
        seen[root] = true;
        let mut work = vec![(root, 0usize)];
        while let Some((node, next_edge)) = work.pop() {
            if next_edge < graph[node].len() {
                work.push((node, next_edge + 1));
                let next = graph[node][next_edge];
                if !seen[next] {
                    seen[next] = true;
                    work.push((next, 0));
                }
            } else {
                order.push(node);
            }
        }
    }

    let mut assigned = vec![false; graph.len()];
    let mut components = Vec::new();
    for root in order.into_iter().rev() {
        if assigned[root] {
            continue;
        }
        assigned[root] = true;
        let mut component = Vec::new();
        let mut work = vec![root];
        while let Some(node) = work.pop() {
            component.push(node);
            for next in reverse[node].iter().copied() {
                if !assigned[next] {
                    assigned[next] = true;
                    work.push(next);
                }
            }
        }
        components.push(component);
    }
    components
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Procedure, Program, Stmt};
    use crate::core::Value;
    use crate::runtime::ffi::{FFIEffect, FFISignature, FFIType};

    fn resolved(body: Vec<Stmt>) -> HirProgram {
        let mut program = Program::new();
        program.body = body;
        HirProgram::resolve(&program).expect("test program resolves")
    }

    fn procedure(name: &str, params: usize, returns: usize, body: Vec<Stmt>) -> Procedure {
        Procedure {
            name: name.to_string(),
            params: (0..params).map(|index| format!("p{index}")).collect(),
            returns,
            effects: Vec::new(),
            body,
        }
    }

    fn has_error(report: &SemanticsReport, predicate: impl Fn(&SemanticErrorKind) -> bool) -> bool {
        report.errors.iter().any(|error| predicate(&error.kind))
    }

    #[test]
    fn rejects_definite_main_underflow_for_add_and_one_word_swap() {
        let add = check(&resolved(vec![Stmt::Op(OpCode::Add)]));
        assert!(has_error(&add, |kind| matches!(
            kind,
            SemanticErrorKind::DefiniteMainUnderflow { missing: 2 }
        )));

        let swap = check(&resolved(vec![
            Stmt::Push(Value::ONE),
            Stmt::Op(OpCode::Swap),
        ]));
        assert!(has_error(&swap, |kind| matches!(
            kind,
            SemanticErrorKind::DefiniteMainUnderflow { missing: 1 }
        )));
    }

    #[test]
    fn enforces_equal_falling_if_rows() {
        let equal = check(&resolved(vec![
            Stmt::Push(Value::ONE),
            Stmt::If {
                then_branch: vec![Stmt::Push(Value::ONE)],
                else_branch: Some(vec![Stmt::Push(Value::ZERO)]),
            },
        ]));
        assert!(equal.is_fully_static());

        let unequal = check(&resolved(vec![
            Stmt::Push(Value::ONE),
            Stmt::If {
                then_branch: vec![Stmt::Push(Value::ONE)],
                else_branch: Some(Vec::new()),
            },
        ]));
        assert!(has_error(&unequal, |kind| matches!(
            kind,
            SemanticErrorKind::BranchRowMismatch { .. }
        )));
    }

    #[test]
    fn terminal_if_arm_is_exempt_from_row_equality() {
        let report = check(&resolved(vec![
            Stmt::Push(Value::ONE),
            Stmt::If {
                then_branch: vec![Stmt::Push(Value::ONE), Stmt::Op(OpCode::Halt)],
                else_branch: Some(Vec::new()),
            },
        ]));
        assert!(report.is_fully_static());
        assert_eq!(report.main.delta, 0);
    }

    #[test]
    fn rejects_loop_condition_and_body_drift() {
        let bad_condition = check(&resolved(vec![Stmt::While {
            cond: Vec::new(),
            body: Vec::new(),
        }]));
        assert!(has_error(&bad_condition, |kind| matches!(
            kind,
            SemanticErrorKind::LoopConditionDrift { delta: 0 }
        )));

        let bad_body = check(&resolved(vec![Stmt::While {
            cond: vec![Stmt::Push(Value::ONE)],
            body: vec![Stmt::Push(Value::ONE)],
        }]));
        assert!(has_error(&bad_body, |kind| matches!(
            kind,
            SemanticErrorKind::LoopBodyDrift { delta: 1 }
        )));
    }

    #[test]
    fn procedure_requirements_flow_through_typed_calls() {
        let mut valid = Program::new();
        valid
            .procedures
            .push(procedure("sum", 2, 1, vec![Stmt::Op(OpCode::Add)]));
        valid.body = vec![
            Stmt::Push(Value::ONE),
            Stmt::Push(Value::ONE),
            Stmt::Call {
                name: "sum".to_string(),
            },
        ];
        let valid = check(&HirProgram::resolve(&valid).unwrap());
        assert!(valid.is_fully_static());
        assert_eq!(valid.procedures[0].stack.required, 2);
        assert_eq!(valid.procedures[0].stack.delta, -1);

        let mut invalid = Program::new();
        invalid
            .procedures
            .push(procedure("sum", 2, 1, vec![Stmt::Op(OpCode::Add)]));
        invalid.body = vec![
            Stmt::Push(Value::ONE),
            Stmt::Call {
                name: "sum".to_string(),
            },
        ];
        let invalid = check(&HirProgram::resolve(&invalid).unwrap());
        assert!(has_error(&invalid, |kind| matches!(
            kind,
            SemanticErrorKind::DefiniteMainUnderflow { missing: 1 }
        )));
    }

    #[test]
    fn dynamic_boundary_is_an_obligation_not_a_false_proof() {
        let report = check(&resolved(vec![
            Stmt::Push(Value::ZERO),
            Stmt::Op(OpCode::Pick),
            Stmt::Op(OpCode::Add),
        ]));
        assert!(report.is_accepted_for_interpreter());
        assert!(!report.is_fully_static());
        assert_eq!(
            report.main.certainty,
            StackCertainty::RuntimeValidationRequired
        );
        assert!(report.obligations.iter().any(|obligation| matches!(
            obligation.kind,
            ObligationKind::DynamicOpcode {
                opcode: OpCode::Pick,
                minimum_inputs: 1
            }
        )));

        let definite_underflow = check(&resolved(vec![Stmt::Op(OpCode::Pick)]));
        assert!(!definite_underflow.is_accepted_for_interpreter());
    }

    #[test]
    fn recursive_scc_has_no_fabricated_summary() {
        let mut program = Program::new();
        program.procedures.push(procedure(
            "recur",
            0,
            0,
            vec![Stmt::Call {
                name: "recur".to_string(),
            }],
        ));
        program.body.push(Stmt::Call {
            name: "recur".to_string(),
        });
        let report = check(&HirProgram::resolve(&program).unwrap());
        assert!(report.is_accepted_for_interpreter());
        assert!(!report.procedures[0].stack.is_proven());
        assert!(report.obligations.iter().any(|obligation| matches!(
            obligation.kind,
            ObligationKind::RecursiveProcedures { .. }
        )));
    }

    #[test]
    fn classifier_partitions_all_99_opcodes_with_23_dynamic() {
        let dynamic = OpCode::ALL
            .iter()
            .filter(|opcode| matches!(opcode_shape(**opcode), OpcodeShape::Dynamic { .. }))
            .count();
        assert_eq!(OpCode::ALL.len(), 99);
        assert_eq!(dynamic, 23);
        assert_eq!(OpCode::ALL.len() - dynamic, 76);
    }

    #[test]
    fn broad_resolve_and_check_smoke_includes_quotes_scopes_calls_and_foreigns() {
        let mut program = Program::new();
        program.procedures.push(procedure(
            "identity",
            1,
            1,
            vec![Stmt::Op(OpCode::Dup), Stmt::Op(OpCode::Pop)],
        ));
        program.quotes.push(vec![Stmt::Op(OpCode::Dup)]);
        program
            .ffi_declarations
            .push(crate::parser::FFIDeclaration {
                signature: FFISignature::new("native", "test")
                    .param("left", FFIType::U64)
                    .param("right", FFIType::Str)
                    .returns_type(FFIType::Bool)
                    .effects(vec![FFIEffect::Pure]),
                symbol_name: Some("native_symbol".to_string()),
            });
        program.body = vec![
            Stmt::Block(vec![Stmt::Push(Value::ONE)]),
            Stmt::Call {
                name: "identity".to_string(),
            },
            Stmt::Push(Value::ONE),
            Stmt::Push(Value::ONE),
            Stmt::Call {
                name: "native".to_string(),
            },
            Stmt::TemporalScope {
                base: 0,
                size: 1,
                cell_bits: 64,
                body: vec![Stmt::Op(OpCode::Pop)],
            },
        ];

        let report = check(&HirProgram::resolve(&program).unwrap());
        assert!(report.is_accepted_for_interpreter());
        assert!(!report.is_fully_static());
        assert_eq!(report.quotes[0].stack.required, 1);
        assert!(report.obligations.iter().any(|obligation| matches!(
            obligation.kind,
            ObligationKind::ForeignLink {
                inputs: 3,
                outputs: 1,
                ..
            }
        )));
    }
}
