// ═══════════════════════════════════════════════════════════════════════════
// Layer 0: Core (No internal dependencies)
// ═══════════════════════════════════════════════════════════════════════════
pub mod core;

// ═══════════════════════════════════════════════════════════════════════════
// Layer 1: Compiler (depends on core)
// ═══════════════════════════════════════════════════════════════════════════
pub mod ast;
pub mod bytecode;
pub mod bytecode_action;
pub mod bytecode_temporal;
pub mod bytecode_timeloop;
pub mod bytecode_verifier;
pub mod bytecode_vm;
pub mod hir;
pub mod launcher;
pub mod lexer;
pub mod linker;
pub mod module_graph;
pub mod object_compiler;
pub mod package;
pub mod parser;
pub mod semantics;
pub mod source;

// ═══════════════════════════════════════════════════════════════════════════
// Layer 2: Temporal & Types (depends on core, compiler)
// ═══════════════════════════════════════════════════════════════════════════
pub mod temporal;
pub mod types;

// ═══════════════════════════════════════════════════════════════════════════
// Layer 3: VM (depends on core, compiler, temporal)
// ═══════════════════════════════════════════════════════════════════════════
pub mod vm;

// ═══════════════════════════════════════════════════════════════════════════
// Layer 4: Runtime (depends on core, vm)
// ═══════════════════════════════════════════════════════════════════════════
pub mod runtime;

// ═══════════════════════════════════════════════════════════════════════════
// Layer 5: Tooling (depends on all)
// ═══════════════════════════════════════════════════════════════════════════
pub mod tooling;

// ═══════════════════════════════════════════════════════════════════════════
// Standard Library & Cross-cutting
// ═══════════════════════════════════════════════════════════════════════════
pub mod audit;
pub mod complexity;
pub mod halting;
pub mod stdlib;

// ═══════════════════════════════════════════════════════════════════════════
// Public API Re-exports
// ═══════════════════════════════════════════════════════════════════════════

// Core types
pub use core::error::{
    BoundsPolicy, DivisionByZeroPolicy, ErrorCategory, ErrorCollector, ErrorConfig,
    MemoryOperation, OuroError, OuroResult, SourceLocation, StackPolicy,
};
pub use core::provenance::{Provenance, DEFAULT_PROVENANCE_SATURATION_LIMIT};
pub use core::Handle;
pub use core::{
    Address, ExactMemoryState, Memory, OutputItem, PagedMemory, PagedMemoryError,
    PagedMemoryLimits, Stack, Value, MAX_DENSE_MEMORY_CELLS, MEMORY_SIZE, PAGE_CELLS,
};

// AST and parser
pub use ast::{
    ComplexRationalLiteral, LocatedProgram, ManifestDeclaration, MarkovDeclaration, OpCode,
    PolynomialDeclaration, ProcedureLocations, Program, ProgramLocations, PropertyCell,
    PropertyComparison, PropertyPredicate, PspaceFamilyDeclaration, QuantumChannelDeclaration,
    QuoteId, RationalLiteral, SignedRationalLiteral, Stmt, StmtLocations, TemporalDeclaration,
    TemporalPropertyDeclaration,
};
pub use bytecode::{
    BytecodeError, BytecodeProgram, BytecodeValidationReport, CodeRange, ConstantReference,
    ForeignEffects, ForeignEntry, ForeignScalarType, Instruction, ProcedureEntry, QuoteEntry,
    SourceMapEntry, StackSemanticStatus, MAX_FOREIGN_NAME_BYTES, MAX_FOREIGN_PARAMETERS,
};
pub use bytecode_action::{
    BytecodeActionConfig, BytecodeActionReport, BytecodeActionRunner, BytecodeActionStats,
    MAX_ACTION_SEEDS,
};
pub use bytecode_temporal::{
    analyze_bytecode_temporal, analyze_bytecode_temporal_with_limits, lower_bytecode_temporal_ir,
    lower_bytecode_temporal_ir_with_limits, BytecodeTemporalAnalysis, BytecodeTemporalDisposition,
    BytecodeTemporalError, BytecodeTemporalIssue, BytecodeTemporalIssueKind,
    BytecodeTemporalLimits, BytecodeTemporalLoweringError, BytecodeTemporalLoweringErrorKind,
    BytecodeTemporalLoweringLimits, PrimitiveBoundary,
};
pub use bytecode_timeloop::{BytecodeTimeLoop, BytecodeTimeLoopConfig, MAX_RETAINED_ORBIT_BYTES};
pub use bytecode_verifier::{
    verify as verify_bytecode, verify_default as verify_bytecode_default,
    verify_with_limits as verify_bytecode_with_limits, BytecodeUnit, BytecodeVerificationError,
    BytecodeVerificationReport, UnitProofStatus, UnitVerification, VerificationLimits,
    VerificationObligation, VerificationObligationKind, VerificationSite,
};
pub use bytecode_vm::{
    bytecode_vm_supports, BytecodeExecution, BytecodeVm, BytecodeVmConfig, BytecodeVmError,
    BytecodeVmStatus, FrozenEndpointTape, FrozenFileSnapshot, FrozenProcessResult,
    PreparedBytecode,
};
pub use hir::{
    CallTarget, CallableKind, ConstantId, ForeignId, HirConstant, HirError, HirErrorKind,
    HirForeign, HirOwner, HirProcedure, HirProgram, HirProperty, HirPropertyCell, HirQuote,
    HirStmt, HirStmtKind, HirTemporal, IndexKind, ProcedureId, PropertyId, TemporalId,
};
pub use launcher::{build_native_launcher, embedded_package, LauncherError, MAX_LAUNCHER_BYTES};
pub use lexer::{lex, LexError, LexErrorKind, LiteralKind, LocatedToken, Token, MAX_SOURCE_TOKENS};
pub use linker::{
    link, link_with_metadata, LinkError, LinkedProgram, ObjectEffectSummary, ObjectEffects,
    ObjectError, ObjectExport, ObjectImport, ObjectMetadata, ObjectModule, ObjectNamedTemporal,
    ObjectProperty, ObjectRelocation, ObjectSourceFile, ObjectTarget, ObjectTemporalRegion,
    ObjectVerificationArtifact, RelocationKind, SymbolKind, SymbolSignature, SymbolTarget,
    VerificationArtifactKind, MAX_LINK_OBJECTS, MAX_OBJECT_BYTES, MAX_OBJECT_CONSTANTS,
    MAX_OBJECT_EXPORTS, MAX_OBJECT_IMPORTS, MAX_OBJECT_NAMED_TEMPORALS, MAX_OBJECT_NAME_BYTES,
    MAX_OBJECT_PROPERTIES, MAX_OBJECT_RELOCATIONS, MAX_OBJECT_SOURCE_FILES,
    MAX_OBJECT_SOURCE_NAME_BYTES, MAX_OBJECT_TEMPORAL_REGIONS, MAX_OBJECT_VERIFICATION_BYTES,
};
pub use module_graph::{
    ImportEdge, ModuleError, ModuleGraph, SourceModule, MAX_GRAPH_EXPANDED_STATEMENTS,
    MAX_GRAPH_SOURCE_TOKENS,
};
pub use object_compiler::{compile_objects, ObjectCompileError};
pub use package::{
    PackageError, PackageManifest, PackageResolutionPolicy, PackageSolverDependency,
    PackageSolverDescriptor, PackageWitness, PortablePackage, CURRENT_RESOLUTION_POLICY_VERSION,
    CURRENT_RUNTIME_ABI, CURRENT_Z3_SOLVER_CONTRACT_VERSION, CURRENT_Z3_SOLVER_DEPENDENCY,
    MAX_PACKAGE_BYTES, MAX_PACKAGE_INSTRUCTIONS, MAX_PACKAGE_MEMORY_CELLS, MAX_PACKAGE_NAME_BYTES,
    MAX_PACKAGE_WITNESS_CELLS,
};
pub use parser::{
    tokenize, Parser, MAX_EXPANDED_STATEMENTS, MAX_MARKOV_STATES, MAX_PARSER_NESTING,
};
pub use runtime::ffi::{ForeignHostError, ForeignHostFn, ForeignHostTable};
pub use semantics::{
    check as check_semantics, ContextFrame, ControlConstruct, ObligationKind, ProcedureSummary,
    QuoteSummary, SemanticError, SemanticErrorKind, SemanticObligation, SemanticSite,
    SemanticsReport, StackCertainty, StackSummary,
};
pub use source::{
    LineColumn, LspPosition, SourceError, SourceFile, SourceId, SourceManager, SourceSpan,
    TextRange,
};

// Temporal
pub use temporal::action::{ActionConfig, ActionPrinciple};
pub use temporal::effect_adapter::NativeEffectAdapter;
pub use temporal::global_solver::{
    FixedPointWitness, GlobalFixedPointSolver, GlobalSolveConfig, GlobalSolveResult,
    GlobalUniquenessResult, PropertyVerificationResult, UnsatCertificate,
};
pub use temporal::ir::{
    CompareOp, ExprId, IrCompleteness, IrExpr, IrExprKind, IrObservation, IrType, ObservationKind,
    TemporalIr, TemporalIrCompiler, TemporalIrConfig, TemporalIrError, WordBinaryOp, WordUnaryOp,
};
pub use temporal::quantum::{
    analyze_quantum_declaration, Complex64, ComplexMatrix, QuantumChannel, QuantumFixedPoint,
    QubitFixedSpaceAnalysis,
};
pub use temporal::region::{TemporalRegionInfo, TemporalRegionIssue, TemporalRegionReport};
pub use temporal::smt_encoder::SmtEncoder;
pub use temporal::stochastic::{
    analyze_declaration, Rational, RationalMarkovChain, StationaryAnalysis, StationaryDecision,
    StationaryFamily,
};
pub use temporal::timeloop::{
    ConvergenceStatus, ExecutionMode, TimeLoop, TimeLoopConfig as Config,
};
pub use temporal::transaction::{
    CandidateContext, CommitLog, CommitOutcome, CommitReceipt, CommitToken, CommittedBatch,
    EffectCommitAdapter, EffectIntent, ObservationTranscript, TemporalTransaction,
    TimelineCandidate, TimelineId, TransactionError, TransactionLimits,
};
pub use temporal::transition_graph::{
    BytecodeTransitionAnalyzer, DeterministicTransitionGraph, ProgramGraphAnalysis,
    ProgramGraphConfig, ProgramTransitionAnalyzer, RecurrentAnalysis, TransitionGraphError,
    MAX_RECURRENT_INSTRUCTIONS, MAX_RECURRENT_OUTPUT_BYTES, MAX_RECURRENT_OUTPUT_ITEMS,
    MAX_RECURRENT_STATES,
};

// Type system
pub use types::{type_check, TemporalType, TypeChecker};

// VM
pub use vm::{EpochStatus, Executor, FastExecutor};

// Tooling
pub use tooling::repl::{Repl, ReplConfig};

// Standard library
pub use complexity::{ConcreteResourceProfile, PolynomialBound, PspaceFamilyContract};
pub use halting::{BoundedHaltingAnalyzer, BoundedHaltingResult};
pub use stdlib::StdLib;

mod determinism_tests;
mod property_tests;
mod tests;
