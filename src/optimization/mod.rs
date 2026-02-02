//! Optimization layer for OUROCHRONOS.
//!
//! This module provides various optimization strategies for temporal programs:
//!
//! - **Tracing JIT**: Hot path detection and compiled trace execution
//! - **Memoization**: Epoch result caching and speculative execution
//! - **Optimizer**: Multi-level instruction optimization and tiered execution
//! - **JIT Compiler**: Cranelift-based JIT compilation (feature-gated)
//! - **AOT Compiler**: Ahead-of-time compilation (feature-gated)

pub mod tracing_jit;
pub mod memo;
pub mod optimizer;
pub mod jit;
pub mod aot;

// Re-export from tracing_jit
pub use tracing_jit::{
    TracingJit, JitFastExecutor, Trace, TracePattern, JitStats,
    TraceOp, TraceAnalysis, PatternAnalysis, PatternCategory,
    JitConfig, ProfileGuidedJit, PgoJitStats, PgoRecommendation,
    PatternRegistry, PatternHandler,
    // Temporal pattern optimization
    ConvergenceProfile, SpeculativeCandidate, CandidateSource,
    EpochSpeculator, SpeculationStats,
    TemporalPatternOptimizer, TemporalOptimizationHints, TemporalOptimizerStats,
    // Temporal execution context
    TemporalContext, JitTemporalStats, TemporalExecutionResult,
};

// Re-export from memo
pub use memo::{
    EpochCache, CacheStats, DeltaTracker, MemoizedResult, SpeculativeExecutor,
    SpeculativeBranch, SpeculativeResult, SpeculativeStats, ParallelConfig,
    SharedEpochCache, ParallelEvaluator, ParallelStats, FixedPointAccelerator,
    AccelerationMethod, AcceleratorStats,
};

// Re-export from optimizer
pub use optimizer::{
    Optimizer, OptLevel, OptInstr, OptStats, TieredExecutor, TieredStats,
    InlineCache, ProfileData,
};

// Re-export from jit
pub use jit::{JitCompiler, CompiledFunction, CompileStats, JitError, JitResult};

// Re-export from aot
pub use aot::{AotCompiler, AotStats, ObjectFile};
