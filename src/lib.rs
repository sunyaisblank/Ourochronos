// ═══════════════════════════════════════════════════════════════════════════
// Layer 0: Core (No internal dependencies)
// ═══════════════════════════════════════════════════════════════════════════
pub mod core;

// ═══════════════════════════════════════════════════════════════════════════
// Layer 1: Compiler (depends on core)
// ═══════════════════════════════════════════════════════════════════════════
pub mod ast;
pub mod parser;

// ═══════════════════════════════════════════════════════════════════════════
// Layer 2: Temporal & Types (depends on core, compiler)
// ═══════════════════════════════════════════════════════════════════════════
pub mod temporal;
pub mod types;

// Backward compatibility: re-export temporal submodules at crate root
pub use temporal::timeloop;
pub use temporal::action;
pub use temporal::smt_encoder;

// ═══════════════════════════════════════════════════════════════════════════
// Layer 3: VM (depends on core, compiler, temporal)
// ═══════════════════════════════════════════════════════════════════════════
pub mod vm;

// ═══════════════════════════════════════════════════════════════════════════
// Layer 4: Optimization (depends on vm, compiler)
// ═══════════════════════════════════════════════════════════════════════════
pub mod optimization;

// ═══════════════════════════════════════════════════════════════════════════
// Layer 5: Runtime (depends on core, vm)
// ═══════════════════════════════════════════════════════════════════════════
pub mod runtime;

// ═══════════════════════════════════════════════════════════════════════════
// Layer 6: Tooling (depends on all)
// ═══════════════════════════════════════════════════════════════════════════
pub mod tooling;

// ═══════════════════════════════════════════════════════════════════════════
// Standard Library & Cross-cutting
// ═══════════════════════════════════════════════════════════════════════════
pub mod stdlib;
pub mod audit;

// ═══════════════════════════════════════════════════════════════════════════
// Backward Compatibility: Re-exports from core module
// ═══════════════════════════════════════════════════════════════════════════

// Re-export error types from core
pub use core::error::{OuroError, OuroResult, ErrorConfig, ErrorCategory, SourceLocation,
                 DivisionByZeroPolicy, BoundsPolicy, StackPolicy, MemoryOperation, ErrorCollector};

// Re-export core types - maintain old module path for compatibility
pub use core::{Value, Address, Memory, OutputItem, Stack, MEMORY_SIZE};

// Re-export Handle from core for data structures
pub use core::Handle;

// Re-export provenance from core
pub use core::provenance::Provenance;

// Re-export data structures from core
pub use core::data_structures::{VecStore, HashStore, SetStore, DataStructures};

// Deprecated module aliases for backward compatibility
#[deprecated(since = "0.3.0", note = "Use ourochronos::core::error instead")]
pub mod error {
    pub use crate::core::error::*;
}

#[deprecated(since = "0.3.0", note = "Use ourochronos::core instead")]
pub mod core_types {
    pub use crate::core::{Value, Address, Memory, OutputItem, Stack, MEMORY_SIZE, Handle};
    pub use crate::core::data_structures::{VecStore, HashStore, SetStore, DataStructures};
}

#[deprecated(since = "0.3.0", note = "Use ourochronos::core::provenance instead")]
pub mod provenance {
    pub use crate::core::provenance::*;
}

#[deprecated(since = "0.3.0", note = "Use ourochronos::vm::fast_vm instead")]
pub mod fast_vm {
    pub use crate::vm::fast_vm::*;
}

#[deprecated(since = "0.3.0", note = "Use ourochronos::optimization::tracing_jit instead")]
pub mod tracing_jit {
    pub use crate::optimization::tracing_jit::*;
}

#[deprecated(since = "0.3.0", note = "Use ourochronos::optimization::memo instead")]
pub mod memo {
    pub use crate::optimization::memo::*;
}

#[deprecated(since = "0.3.0", note = "Use ourochronos::optimization::jit instead")]
pub mod jit {
    pub use crate::optimization::jit::*;
}

#[deprecated(since = "0.3.0", note = "Use ourochronos::optimization::optimizer instead")]
pub mod optimizer {
    pub use crate::optimization::optimizer::*;
}

#[deprecated(since = "0.3.0", note = "Use ourochronos::optimization::aot instead")]
pub mod aot {
    pub use crate::optimization::aot::*;
}

#[deprecated(since = "0.3.0", note = "Use ourochronos::runtime::module instead")]
pub mod module {
    pub use crate::runtime::module::*;
}

#[deprecated(since = "0.3.0", note = "Use ourochronos::runtime::ffi instead")]
pub mod ffi {
    pub use crate::runtime::ffi::*;
}

#[deprecated(since = "0.3.0", note = "Use ourochronos::runtime::io instead")]
pub mod io {
    pub use crate::runtime::io::*;
}

#[deprecated(since = "0.3.0", note = "Use ourochronos::runtime::package instead")]
pub mod package {
    pub use crate::runtime::package::*;
}

#[deprecated(since = "0.3.0", note = "Use ourochronos::tooling::lsp instead")]
pub mod lsp {
    pub use crate::tooling::lsp::*;
}

#[deprecated(since = "0.3.0", note = "Use ourochronos::tooling::debugger instead")]
pub mod debugger {
    pub use crate::tooling::debugger::*;
}

#[deprecated(since = "0.3.0", note = "Use ourochronos::tooling::profiler instead")]
pub mod profiler {
    pub use crate::tooling::profiler::*;
}

#[deprecated(since = "0.3.0", note = "Use ourochronos::tooling::repl instead")]
pub mod repl {
    pub use crate::tooling::repl::*;
}

pub use ast::{OpCode, Stmt, Program};
pub use parser::{tokenize, Parser};
pub use vm::{Executor, EpochStatus, FastExecutor, FastStack, is_program_pure, execute_with_fast_path};
pub use optimization::tracing_jit::{TracingJit, JitFastExecutor, Trace, TracePattern, JitStats,
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
pub use runtime::module::{Module, ModuleRegistry};
pub use optimization::memo::{EpochCache, CacheStats, DeltaTracker, MemoizedResult, SpeculativeExecutor,
               SpeculativeBranch, SpeculativeResult, SpeculativeStats, ParallelConfig,
               SharedEpochCache, ParallelEvaluator, ParallelStats, FixedPointAccelerator,
               AccelerationMethod, AcceleratorStats};
pub use optimization::jit::{JitCompiler, CompiledFunction, CompileStats, JitError, JitResult};
pub use optimization::optimizer::{Optimizer, OptLevel, OptInstr, OptStats, TieredExecutor, TieredStats, InlineCache, ProfileData};
pub use optimization::aot::{AotCompiler, AotStats, ObjectFile};
pub use tooling::lsp::{LanguageAnalyzer, Diagnostic, Severity, CompletionItem, CompletionKind,
               HoverInfo, Document, Location, DocumentSymbol, SymbolKind,
               SemanticToken, SemanticTokenType, DiagnosticRelated};
pub use tooling::repl::{Repl, ReplConfig};
pub use tooling::debugger::{Debugger, DebugEvent, EpochSnapshot, Breakpoint, Watchpoint, WatchType, BreakCondition};
pub use tooling::profiler::{Profiler, ProfilerConfig, EpochProfile, InstructionStats, MemoryProfile, TemporalStats, ProfileSummary, OptimizationHints};
pub use stdlib::StdLib;
pub use runtime::package::{PackageManager, PackageManifest, Package, Dependency};
pub use runtime::ffi::{FFIRegistry, FFIContext, FFISignature, FFIFunction, FFIEffect, FFIType, FFICaller};
pub use runtime::io::{IOContext, FileMode, SeekOrigin, Buffer, IOStats};
pub mod analysis;

mod tests;
mod determinism_tests;
mod property_tests;


