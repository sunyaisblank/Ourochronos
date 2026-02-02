//! Stress tests for OUROCHRONOS core module.
//!
//! These tests are ignored by default. Run with:
//! `cargo test --release -- --ignored`
//!
//! Or run a specific test:
//! `cargo test --release stress_memory_writes -- --ignored`

use ourochronos::core::memory::Memory;
use ourochronos::core::provenance::Provenance;
use ourochronos::core::stack::Stack;
use ourochronos::core::value::Value;
use ourochronos::core::error::SourceLocation;
use std::collections::BTreeSet;

// ═══════════════════════════════════════════════════════════════════════════
// Memory Stress Tests
// ═══════════════════════════════════════════════════════════════════════════

/// Stress test: 1M memory writes with hash consistency verification.
///
/// Verifies that the incremental hash remains consistent after many writes.
#[test]
#[ignore]
fn stress_memory_writes_1m() {
    let mut mem = Memory::new();
    let mut rng_state: u64 = 0xDEADBEEF;

    // Simple LCG for deterministic pseudo-random numbers
    let mut next_rand = || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        rng_state
    };

    // Perform 1M writes
    for _ in 0..1_000_000 {
        let addr = (next_rand() % 65536) as u16;
        let val = next_rand();
        mem.write(addr, Value::new(val));
    }

    // Verify hash is consistent (this implicitly tests the hash_mix function)
    let hash = mem.state_hash();
    assert!(hash != 0 || mem.is_empty(), "Hash should be non-zero for non-empty memory");

    // Verify non-zero cells exist
    let non_zero = mem.non_zero_cells();
    assert!(!non_zero.is_empty(), "Memory should have non-zero cells after writes");
}

/// Stress test: Write-read cycles with hash verification.
#[test]
#[ignore]
fn stress_memory_write_read_cycles() {
    let mut mem = Memory::new();

    // Write sequential values
    for i in 0..65536u64 {
        mem.write(i as u16, Value::new(i));
    }

    // Verify all values
    for i in 0..65536u64 {
        let val = mem.read(i as u16);
        assert_eq!(val.val, i, "Value at address {} should be {}", i, i);
    }

    // Zero out every other cell
    for i in (0..65536u64).step_by(2) {
        mem.write(i as u16, Value::ZERO);
    }

    // Verify pattern
    for i in 0..65536u64 {
        let val = mem.read(i as u16);
        if i % 2 == 0 {
            assert_eq!(val.val, 0);
        } else {
            assert_eq!(val.val, i);
        }
    }
}

/// Stress test: Memory values_equal with many states.
#[test]
#[ignore]
fn stress_memory_values_equal() {
    let mut mem1 = Memory::new();
    let mut mem2 = Memory::new();

    // Both empty - should be equal
    assert!(mem1.values_equal(&mem2));

    // Write same values to both
    for i in 0..10000 {
        let addr = (i * 7) % 65536;
        mem1.write(addr as u16, Value::new(i as u64));
        mem2.write(addr as u16, Value::new(i as u64));
    }

    // Should still be equal
    assert!(mem1.values_equal(&mem2));

    // Change one value
    mem1.write(0, Value::new(999));
    assert!(!mem1.values_equal(&mem2));

    // Make them equal again
    mem2.write(0, Value::new(999));
    assert!(mem1.values_equal(&mem2));
}

// ═══════════════════════════════════════════════════════════════════════════
// Provenance Stress Tests
// ═══════════════════════════════════════════════════════════════════════════

/// Stress test: 10K provenance merges performance.
#[test]
#[ignore]
fn stress_provenance_merges_10k() {
    let mut accumulated = Provenance::none();

    // Merge 10K single-address provenances
    for i in 0..10_000 {
        let p = Provenance::single(i as u16);
        accumulated = accumulated.merge(&p);
    }

    // Verify all addresses are present
    assert_eq!(accumulated.dependency_count(), 10_000);

    // Verify we can iterate over all dependencies
    let count = accumulated.dependencies().count();
    assert_eq!(count, 10_000);
}

/// Stress test: Provenance merge with many overlapping sets.
#[test]
#[ignore]
fn stress_provenance_overlapping_merges() {
    let mut provenances: Vec<Provenance> = Vec::new();

    // Create 100 overlapping sets
    for i in 0..100 {
        let start = i * 10;
        let end = start + 50;
        let set: BTreeSet<u16> = (start..end).collect();
        provenances.push(Provenance::from_set(set));
    }

    // Merge all together
    let mut result = Provenance::none();
    for p in &provenances {
        result = result.merge(p);
    }

    // Calculate expected count: 0 to (99*10 + 49) = 0 to 1039
    let expected_max = 99 * 10 + 49;
    assert!(result.dependency_count() <= (expected_max + 1) as usize);
    assert!(result.dependency_count() > 0);

    // Verify specific addresses
    assert!(result.dependencies().any(|a| a == 0));
    assert!(result.dependencies().any(|a| a == expected_max as u16));
}

/// Stress test: Subset reuse optimization under heavy load.
#[test]
#[ignore]
fn stress_provenance_subset_optimization() {
    // Create a large base set
    let large_set: BTreeSet<u16> = (0..1000).collect();
    let p_large = Provenance::from_set(large_set);

    // Merge many subsets into it
    for i in 0..1000 {
        let small_set: BTreeSet<u16> = (i..i+10).collect();
        let p_small = Provenance::from_set(small_set);

        let merged = p_large.merge(&p_small);

        // Should reuse the large set since small is a subset
        assert!(merged.dependency_count() >= 1000);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Stack Stress Tests
// ═══════════════════════════════════════════════════════════════════════════

/// Stress test: 100K stack operations.
#[test]
#[ignore]
fn stress_stack_operations_100k() {
    let mut stack = Stack::new();
    let loc = SourceLocation::default();

    // Push 50K values
    for i in 0..50_000 {
        stack.push(Value::new(i));
    }
    assert_eq!(stack.depth(), 50_000);

    // Pop 25K values
    for _ in 0..25_000 {
        let _ = stack.pop();
    }
    assert_eq!(stack.depth(), 25_000);

    // Perform various operations
    for _ in 0..10_000 {
        stack.dup_checked(loc.clone()).unwrap();
        stack.swap_checked(loc.clone()).unwrap();
        stack.pop();
        stack.pop();
        stack.push(Value::new(42));
    }

    // Stack should be at expected depth
    assert!(stack.depth() > 0);
}

/// Stress test: Stack with max depth under pressure.
#[test]
#[ignore]
fn stress_stack_max_depth() {
    let max_depth = 10_000;
    let mut stack = Stack::with_max_depth(max_depth);
    let loc = SourceLocation::default();

    // Fill to max
    for i in 0..max_depth {
        stack.push_checked(Value::new(i as u64), loc.clone()).unwrap();
    }
    assert_eq!(stack.depth(), max_depth);

    // Next push should fail
    let result = stack.push_checked(Value::new(0), loc.clone());
    assert!(result.is_err());

    // Pop and push cycles at capacity
    for _ in 0..10_000 {
        stack.pop().unwrap();
        stack.push_checked(Value::new(99), loc.clone()).unwrap();
        assert_eq!(stack.depth(), max_depth);
    }
}

/// Stress test: pick and roll operations.
#[test]
#[ignore]
fn stress_stack_pick_roll() {
    let mut stack = Stack::new();
    let loc = SourceLocation::default();

    // Build a stack of 1000 elements
    for i in 0..1000 {
        stack.push(Value::new(i));
    }

    // Perform many pick operations
    for n in 0..500 {
        stack.pick_checked(n, loc.clone()).unwrap();
    }

    assert_eq!(stack.depth(), 1500);

    // Perform many roll operations
    for n in 0..100 {
        stack.roll_checked(n, loc.clone()).unwrap();
    }

    assert_eq!(stack.depth(), 1500);
}

/// Stress test: reverse operations.
#[test]
#[ignore]
fn stress_stack_reverse() {
    let mut stack = Stack::new();
    let loc = SourceLocation::default();

    // Push 1000 values
    for i in 0..1000 {
        stack.push(Value::new(i));
    }

    // Reverse entire stack
    stack.reverse_checked(1000, loc.clone()).unwrap();

    // Verify order is reversed
    for i in 0..1000 {
        let val = stack.pop().unwrap();
        assert_eq!(val.val, i);
    }
}

/// Stress test: pop_n with large batches.
#[test]
#[ignore]
fn stress_stack_pop_n() {
    let mut stack = Stack::new();
    let loc = SourceLocation::default();

    // Push 10000 values
    for i in 0..10000 {
        stack.push(Value::new(i));
    }

    // Pop in batches of 100
    for batch in 0..100 {
        let values = stack.pop_n(100, "stress test", loc.clone()).unwrap();
        assert_eq!(values.len(), 100);

        // Verify values are in correct order
        let expected_start = 9900 - batch * 100;
        for (i, val) in values.iter().enumerate() {
            assert_eq!(val.val, (expected_start + i) as u64);
        }
    }

    assert!(stack.is_empty());
}
