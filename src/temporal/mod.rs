//! Temporal computation engine for OUROCHRONOS.
//!
//! This module implements the fixed-point search algorithm that finds
//! self-consistent states in closed timelike curve computations.
//!
//! # Key Components
//!
//! - **TimeLoop**: The main driver for epoch-based fixed-point search
//! - **Action Principle**: Selection mechanism for choosing among multiple fixed points
//! - **SMT Encoder**: Generation of SMT-LIB2 formulas for constraint solving

pub mod action;
pub mod cache;
pub mod effect_adapter;
pub mod global_solver;
pub mod ir;
pub mod quantum;
pub mod region;
pub mod smt_encoder;
pub mod stochastic;
pub mod timeloop;
pub mod transaction;
pub mod transition_graph;

pub use action::{
    ActionConfig, ActionPrinciple, FixedPointCandidate, FixedPointSelector, ProvenanceMap,
    SeedGenerator, SeedStrategy, SelectionRule,
};
pub use global_solver::{
    FixedPointWitness, GlobalFixedPointSolver, GlobalSolveConfig, GlobalSolveResult,
    GlobalUniquenessResult, PropertyVerificationResult, UnsatCertificate,
};
pub use ir::{
    CompareOp, ExprId, IrCompleteness, IrExpr, IrExprKind, IrObservation, IrType, ObservationKind,
    TemporalIr, TemporalIrCompiler, TemporalIrConfig, TemporalIrError, WordBinaryOp, WordUnaryOp,
};
pub use quantum::{
    analyze_quantum_declaration, Complex64, ComplexMatrix, QuantumChannel, QuantumFixedPoint,
    QubitFixedSpaceAnalysis,
};
pub use region::{TemporalRegionInfo, TemporalRegionIssue, TemporalRegionReport};
pub use smt_encoder::SmtEncoder;
pub use stochastic::{
    analyze_declaration, Rational, RationalMarkovChain, StationaryAnalysis, StationaryDecision,
    StationaryFamily,
};
pub use timeloop::{ConvergenceStatus, ExecutionMode, ParadoxDiagnosis, TimeLoop, TimeLoopConfig};
pub use transition_graph::{
    BytecodeTransitionAnalyzer, DeterministicTransitionGraph, ProgramGraphAnalysis,
    ProgramGraphConfig, ProgramTransitionAnalyzer, RecurrentAnalysis, TransitionGraphError,
    MAX_RECURRENT_INSTRUCTIONS, MAX_RECURRENT_OUTPUT_BYTES, MAX_RECURRENT_OUTPUT_ITEMS,
    MAX_RECURRENT_STATES,
};
