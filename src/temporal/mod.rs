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

pub mod timeloop;
pub mod action;
pub mod smt_encoder;

pub use timeloop::{TimeLoop, ConvergenceStatus, TimeLoopConfig, ExecutionMode, ParadoxDiagnosis};
pub use action::{ActionPrinciple, ActionConfig, FixedPointSelector, FixedPointCandidate,
                 ProvenanceMap, SeedStrategy, SeedGenerator, SelectionRule};
pub use smt_encoder::SmtEncoder;
