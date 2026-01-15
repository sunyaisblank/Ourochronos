//! Ourochronos Integration Test Suite
//!
//! This file serves as the entry point for integration tests.
//! It imports and re-exports all integration test modules.
//!
//! ## Test Categories
//!
//! - **common**: Shared test utilities and helpers
//! - **integration**: Cross-component integration tests
//!   - timeloop: Fixed-point computation and temporal execution
//!   - vm: Virtual machine operations
//!   - temporal_semantics: Core temporal mechanics
//! - **benchmark**: Performance comparison tests
//!   - VM vs FastVM performance
//!   - JIT compilation benchmarks
//!
//! ## Running Tests
//!
//! ```bash
//! # Run all integration tests
//! cargo test --test main
//!
//! # Run specific test module
//! cargo test --test main timeloop
//!
//! # Run benchmarks (release mode recommended)
//! cargo test --test main benchmark --release -- --nocapture
//!
//! # Run with output
//! cargo test --test main -- --nocapture
//! ```

mod common;
mod integration;
mod benchmark;

// Re-export test modules for test organisation
pub use integration::*;
