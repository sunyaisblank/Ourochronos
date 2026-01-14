pub mod error;
pub mod core_types;
pub mod ast;
pub mod parser;
pub mod vm;
pub mod fast_vm;
pub mod tracing_jit;
pub mod timeloop;
pub mod provenance;
pub mod smt_encoder;
pub mod action;
pub mod types;
pub mod module;
pub mod memo;
pub mod jit;
pub mod optimizer;
pub mod aot;
pub mod lsp;
pub mod repl;
pub mod debugger;
pub mod profiler;
pub mod stdlib;
pub mod package;
pub mod ffi;
pub mod io;

pub use error::{OuroError, OuroResult, ErrorConfig, ErrorCategory, SourceLocation,
                 DivisionByZeroPolicy, BoundsPolicy, StackPolicy, MemoryOperation, ErrorCollector};
pub use core_types::{Value, Address, Memory, OutputItem, Stack, MEMORY_SIZE};
pub use ast::{OpCode, Stmt, Program};
pub use parser::{tokenize, Parser};
pub use vm::{Executor, EpochStatus};
pub use fast_vm::{FastExecutor, FastStack, is_program_pure, execute_with_fast_path};
pub use tracing_jit::{TracingJit, JitFastExecutor, Trace, TracePattern, JitStats,
                     TraceOp, TraceAnalysis, PatternAnalysis, PatternCategory,
                     JitConfig, ProfileGuidedJit, PgoJitStats, PgoRecommendation,
                     PatternRegistry, PatternHandler,
                     // Temporal pattern optimization
                     ConvergenceProfile, SpeculativeCandidate, CandidateSource,
                     EpochSpeculator, SpeculationStats,
                     TemporalPatternOptimizer, TemporalOptimizationHints, TemporalOptimizerStats,
                     // Temporal execution context
                     TemporalContext, JitTemporalStats, TemporalExecutionResult};
pub use timeloop::{TimeLoop, ConvergenceStatus, TimeLoopConfig as Config, ExecutionMode};
pub use smt_encoder::SmtEncoder;
pub use action::{ActionPrinciple, ActionConfig, FixedPointSelector, FixedPointCandidate, ProvenanceMap, SeedStrategy, SeedGenerator, SelectionRule};
pub use types::{TemporalType, TypeChecker, TypeCheckResult, type_check, ComputedEffect, EffectSet, EffectViolation};
pub use module::{Module, ModuleRegistry};
pub use memo::{EpochCache, CacheStats, DeltaTracker, MemoizedResult, SpeculativeExecutor,
               SpeculativeBranch, SpeculativeResult, SpeculativeStats, ParallelConfig,
               SharedEpochCache, ParallelEvaluator, ParallelStats, FixedPointAccelerator,
               AccelerationMethod, AcceleratorStats};
pub use jit::{JitCompiler, CompiledFunction, CompileStats, JitError, JitResult};
pub use optimizer::{Optimizer, OptLevel, OptInstr, OptStats, TieredExecutor, TieredStats, InlineCache, ProfileData};
pub use aot::{AotCompiler, AotStats, ObjectFile};
pub use lsp::{LanguageAnalyzer, Diagnostic, Severity, CompletionItem, CompletionKind,
               HoverInfo, Document, Location, DocumentSymbol, SymbolKind,
               SemanticToken, SemanticTokenType, DiagnosticRelated};
pub use repl::{Repl, ReplConfig};
pub use debugger::{Debugger, DebugEvent, EpochSnapshot, Breakpoint, Watchpoint, WatchType, BreakCondition};
pub use profiler::{Profiler, ProfilerConfig, EpochProfile, InstructionStats, MemoryProfile, TemporalStats, ProfileSummary, OptimizationHints};
pub use stdlib::StdLib;
pub use package::{PackageManager, PackageManifest, Package, Dependency};
pub use ffi::{FFIRegistry, FFIContext, FFISignature, FFIFunction, FFIEffect, FFIType, FFICaller};
pub use io::{IOContext, FileMode, SeekOrigin, Buffer, IOStats};
pub mod analysis;

mod tests;
mod determinism_tests;
mod property_tests;


