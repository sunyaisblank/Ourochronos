//! Integration tests for memory operations (INDEX, STORE, PACK, UNPACK).
//!
//! These test the full parse → execute pipeline for computed memory access.

#![cfg(test)]

use crate::common::*;

// =============================================================================
// INDEX Tests
// =============================================================================

mod index_tests {
    use super::*;

    #[test]
    fn index_reads_from_base_plus_offset() {
        // Write 42 to address 10, then INDEX with base=10, offset=0
        let result = run("42 10 PROPHECY 10 0 INDEX 0 PROPHECY");
        assert_consistent(&result);
        assert_memory_value(&result, 0, 42);
    }

    #[test]
    fn index_reads_with_nonzero_offset() {
        // Write 99 to address 13, then INDEX with base=10, offset=3
        let result = run("99 13 PROPHECY 10 3 INDEX 0 PROPHECY");
        assert_consistent(&result);
        assert_memory_value(&result, 0, 99);
    }
}

// =============================================================================
// STORE Tests
// =============================================================================

mod store_tests {
    use super::*;

    #[test]
    fn store_writes_to_base_plus_offset() {
        // STORE: pops index, pops base, pops value → writes value to present[base + index]
        let result = run("77 20 0 STORE 20 0 INDEX 0 PROPHECY");
        assert_consistent(&result);
        assert_memory_value(&result, 0, 77);
    }

    #[test]
    fn store_with_nonzero_offset() {
        let result = run("55 20 5 STORE 20 5 INDEX 0 PROPHECY");
        assert_consistent(&result);
        assert_memory_value(&result, 0, 55);
    }
}

// =============================================================================
// PACK Tests
// =============================================================================

mod pack_tests {
    use super::*;

    #[test]
    fn pack_writes_values_to_contiguous_memory() {
        // PACK: pops n, pops base, then pops n values → writes to present[base..base+n]
        // Push 10 20 30, then base=50, n=3 → writes to addresses 50, 51, 52
        let result = run("10 20 30 50 3 PACK 50 0 INDEX 0 PROPHECY");
        assert_consistent(&result);
        assert_memory_value(&result, 0, 10);
    }

    #[test]
    fn pack_then_index_second_element() {
        let result = run("10 20 30 50 3 PACK 50 1 INDEX 0 PROPHECY");
        assert_consistent(&result);
        assert_memory_value(&result, 0, 20);
    }
}

// =============================================================================
// UNPACK Tests
// =============================================================================

mod unpack_tests {
    use super::*;

    #[test]
    fn unpack_reads_contiguous_memory_onto_stack() {
        // Write values with PACK, then UNPACK them back onto stack
        // Pack [10, 20, 30] at address 50, then unpack 3 from address 50
        // After UNPACK, stack has 10 20 30 (bottom to top)
        // The top of stack (30) gets written to address 0
        let result = run("10 20 30 50 3 PACK 50 3 UNPACK 0 PROPHECY");
        assert_consistent(&result);
        assert_memory_value(&result, 0, 30);
    }

    #[test]
    fn unpack_first_element() {
        // Pack [10, 20] at address 50, unpack 2, pop top (20), write bottom (10)
        let result = run("10 20 50 2 PACK 50 2 UNPACK POP 0 PROPHECY");
        assert_consistent(&result);
        assert_memory_value(&result, 0, 10);
    }
}
