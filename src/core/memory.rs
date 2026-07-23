//! Memory type for the OUROCHRONOS virtual machine.
//!
//! Memory has a configurable positive number of cells, each holding a Value.
//! `MEMORY_SIZE` (65,536) is the backwards-compatible default.

use super::address::{Address, MEMORY_SIZE};
use super::error::{BoundsPolicy, MemoryOperation, OuroError, OuroResult, SourceLocation};
use super::provenance::Provenance;
use super::value::Value;

/// Largest dense memory returned by public convergence and solver APIs.
/// Execution itself may use a wider sparse [`PagedMemory`](super::PagedMemory),
/// but materializing larger dense snapshots would make a small program able
/// to force disproportionate host allocation.
pub const MAX_DENSE_MEMORY_CELLS: usize = 1_048_576;
use std::fmt;

/// Canonical sparse identity of a [`Memory`] snapshot.
///
/// Width, numeric values, and provenance are all retained. Pure zero cells are
/// omitted, while zero-valued cells with provenance remain present. Temporal
/// hash tables use this value to verify hash-bucket candidates exactly.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExactMemoryState {
    width: usize,
    cells: Vec<(Address, Value)>,
}

impl ExactMemoryState {
    /// Width of the source memory snapshot.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Exact non-default cells in increasing address order.
    pub fn cells(&self) -> &[(Address, Value)] {
        &self.cells
    }
}

/// A snapshot of the memory state.
///
/// Memory contains a run-configured number of cells (65,536 by default).
/// Initially all cells are zero with no provenance.
///
/// Memory states are ordered lexicographically by cell values (address 0 first).
/// This ordering enables deterministic selection among equal-action fixed points.
///
/// The numeric hash is maintained incrementally for O(1) bucket lookups.
/// Exact state users must verify candidates with [`Memory::exact_sparse_state`].
#[derive(Clone, PartialEq, Eq)]
pub struct Memory {
    cells: Vec<Value>,
    /// Incrementally updated hash of the memory state.
    /// Uses XOR-based incremental updates on write for O(1) retrieval.
    cached_hash: u64,
}

impl PartialOrd for Memory {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Memory {
    /// Exact lexicographic comparison of memory states by cell values.
    ///
    /// Numeric words are the primary key; [`Value::cmp`] uses complete
    /// provenance as the deterministic tie-break. Consequently `cmp == Equal`
    /// exactly when `Eq` is true, as required by the `Ord` contract.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        for (left, right) in self.cells.iter().zip(&other.cells) {
            let ordering = left.val.cmp(&right.val);
            if ordering != std::cmp::Ordering::Equal {
                return ordering;
            }
        }
        let width = self.cells.len().cmp(&other.cells.len());
        if width != std::cmp::Ordering::Equal {
            return width;
        }
        self.cells.cmp(&other.cells)
    }
}

impl Default for Memory {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Memory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Only show non-zero cells
        let nonzero: Vec<_> = self
            .cells
            .iter()
            .enumerate()
            .filter(|(_, v)| v.val != 0)
            .collect();

        if nonzero.is_empty() {
            write!(f, "Memory{{all zero}}")
        } else {
            write!(f, "Memory{{")?;
            for (i, (addr, val)) in nonzero.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "[{}]={}", addr, val.val)?;
            }
            write!(f, "}}")
        }
    }
}

impl Memory {
    /// Mixing function for incremental hashing.
    /// Combines address and value into a well-distributed hash contribution.
    /// Returns 0 for zero values to maintain consistency (zeros contribute nothing).
    ///
    /// Uses improved mixing that properly incorporates address into the hash
    /// rather than just adding it, ensuring better distribution.
    #[inline]
    fn hash_mix(addr: Address, val: u64) -> u64 {
        if val == 0 {
            return 0; // Zero values contribute nothing to hash
        }
        // Combine address and value with multiplicative mixing for better distribution
        let combined = addr.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(val);
        // Apply finalization mixing (MurmurHash3-style)
        let mut h = combined.wrapping_mul(0x517cc1b727220a95);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        h
    }

    /// Create a new memory state with all cells set to zero.
    pub fn new() -> Self {
        Self::with_size(MEMORY_SIZE)
    }

    /// Create an all-zero memory state with an explicit finite width.
    ///
    /// The language-level family semantics permits any positive finite width;
    /// a concrete allocation remains limited by the host process.
    pub fn with_size(memory_cells: usize) -> Self {
        assert!(memory_cells > 0, "memory must contain at least one cell");
        Self {
            cells: vec![Value::ZERO; memory_cells],
            cached_hash: 0, // All zeros contribute 0 to hash
        }
    }

    /// Read the value at the given address.
    pub fn read(&self, addr: Address) -> Value {
        self.cells[addr as usize].clone()
    }

    /// Write a value to the given address.
    ///
    /// Updates the cached hash incrementally using XOR.
    pub fn write(&mut self, addr: Address, val: Value) {
        let old_val = self.cells[addr as usize].val;
        let new_val = val.val;

        // XOR out old contribution, XOR in new contribution
        if old_val != new_val {
            self.cached_hash ^= Self::hash_mix(addr, old_val);
            self.cached_hash ^= Self::hash_mix(addr, new_val);
        }

        self.cells[addr as usize] = val;

        // Debug assertion: verify incremental hash is correct
        #[cfg(debug_assertions)]
        debug_assert_eq!(
            self.cached_hash,
            self.recompute_hash_internal(),
            "Hash invariant violated after write to address {}",
            addr
        );
    }

    /// Internal method to recompute hash from scratch.
    /// Used for debug assertions to verify incremental hash correctness.
    #[cfg(debug_assertions)]
    fn recompute_hash_internal(&self) -> u64 {
        let mut hash = 0u64;
        for (addr, cell) in self.cells.iter().enumerate() {
            if cell.val != 0 {
                hash ^= Self::hash_mix(addr as Address, cell.val);
            }
        }
        hash
    }

    /// Check if two memory states are numerically equal, ignoring provenance.
    /// This is the fixed-point check: P_final = A_initial.
    ///
    /// Uses cached hash for O(1) early rejection when states differ (common case).
    /// Falls back to O(N) cell-by-cell comparison only when hashes match.
    pub fn numeric_values_equal(&self, other: &Memory) -> bool {
        if self.cells.len() != other.cells.len() {
            return false;
        }
        // Fast path: different hashes guarantee different content
        if self.cached_hash != other.cached_hash {
            return false;
        }
        // Same hash: verify cell-by-cell (hash collision possible but rare)
        self.cells
            .iter()
            .zip(other.cells.iter())
            .all(|(v1, v2)| v1.val == v2.val)
    }

    /// Backwards-compatible alias for [`Self::numeric_values_equal`].
    ///
    /// Temporal cache and cycle identity must use [`Self::exact_sparse_state`]
    /// instead. Keeping the numeric operation explicitly named prevents the
    /// language's fixed-point rule from being confused with state-key equality.
    pub fn values_equal(&self, other: &Memory) -> bool {
        self.numeric_values_equal(other)
    }

    /// Find addresses where values differ between two memory states.
    pub fn diff(&self, other: &Memory) -> Vec<Address> {
        let common = self.cells.len().min(other.cells.len());
        let mut result: Vec<Address> = self.cells[..common]
            .iter()
            .zip(other.cells[..common].iter())
            .enumerate()
            .filter(|(_, (v1, v2))| v1.val != v2.val)
            .map(|(i, _)| i as Address)
            .collect();
        result.extend((common..self.cells.len().max(other.cells.len())).map(|i| i as Address));
        result
    }

    /// Find all non-zero cells.
    pub fn non_zero_cells(&self) -> Vec<(Address, &Value)> {
        self.cells
            .iter()
            .enumerate()
            .filter(|(_, v)| v.val != 0)
            .map(|(i, v)| (i as Address, v))
            .collect()
    }

    /// Get a numeric-state hash for cycle-detection buckets.
    ///
    /// Provenance is intentionally not part of this backwards-compatible hash,
    /// so exact users must compare [`Self::exact_sparse_state`] on every match.
    /// O(1) retrieval of the incrementally maintained hash.
    #[inline]
    pub fn state_hash(&self) -> u64 {
        self.cached_hash
    }

    /// Recompute the full hash from scratch.
    /// Used for verification that incremental hash is correct.
    #[cfg(test)]
    pub fn recompute_hash(&self) -> u64 {
        let mut hash = 0u64;
        for (addr, cell) in self.cells.iter().enumerate() {
            if cell.val != 0 {
                hash ^= Self::hash_mix(addr as Address, cell.val);
            }
        }
        hash
    }

    /// Collect all provenance information from written cells.
    pub fn collect_provenance(&self) -> Provenance {
        let mut result = Provenance::none();
        for cell in &self.cells {
            if cell.prov.is_temporal() {
                result = result.merge(&cell.prov);
            }
        }
        result
    }

    /// Iterate over non-zero cells, yielding (address, value) pairs.
    pub fn iter_nonzero(&self) -> impl Iterator<Item = (Address, &Value)> {
        self.cells
            .iter()
            .enumerate()
            .filter(|(_, v)| v.val != 0)
            .map(|(i, v)| (i as Address, v))
    }

    /// Numeric-only sparse projection in address order.
    ///
    /// This intentionally discards provenance and must not be used for cache,
    /// journal, or other exact state identity.
    pub fn sparse_state(&self) -> Vec<(Address, u64)> {
        self.iter_nonzero()
            .map(|(addr, value)| (addr, value.val))
            .collect()
    }

    /// Canonical exact sparse identity for temporal caches and journals.
    pub fn exact_sparse_state(&self) -> ExactMemoryState {
        ExactMemoryState {
            width: self.cells.len(),
            cells: self
                .cells
                .iter()
                .enumerate()
                .filter(|(_, value)| **value != Value::ZERO)
                .map(|(index, value)| (index as Address, value.clone()))
                .collect(),
        }
    }

    /// Get the total number of memory cells.
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Check if memory is empty (all zeros).
    ///
    /// Uses cached hash for O(1) early rejection when non-empty (common case).
    /// Falls back to O(N) scan only when hash is zero.
    pub fn is_empty(&self) -> bool {
        // Fast path: non-zero hash means at least one non-zero cell
        if self.cached_hash != 0 {
            return false;
        }
        // Zero hash: verify cell-by-cell (all-zero state has hash 0)
        self.cells.iter().all(|v| v.val == 0)
    }

    /// Iterate over all cells, yielding (address, value) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (Address, &Value)> {
        self.cells
            .iter()
            .enumerate()
            .map(|(i, v)| (i as Address, v))
    }

    // ═══════════════════════════════════════════════════════════════════
    // Bounds-Checked Operations
    // ═══════════════════════════════════════════════════════════════════

    /// Validate an address against the default memory bounds.
    ///
    /// Returns `Ok(addr as Address)` if within bounds, or handles according to policy.
    #[inline]
    pub fn validate_address(
        addr: u64,
        policy: BoundsPolicy,
        operation: MemoryOperation,
        location: SourceLocation,
    ) -> OuroResult<Address> {
        Self::validate_address_for_len(addr, MEMORY_SIZE, policy, operation, location)
    }

    /// Validate an address against this memory snapshot's configured width.
    pub fn validate_address_in(
        &self,
        addr: u64,
        policy: BoundsPolicy,
        operation: MemoryOperation,
        location: SourceLocation,
    ) -> OuroResult<Address> {
        Self::validate_address_for_len(addr, self.cells.len(), policy, operation, location)
    }

    fn validate_address_for_len(
        addr: u64,
        memory_cells: usize,
        policy: BoundsPolicy,
        operation: MemoryOperation,
        location: SourceLocation,
    ) -> OuroResult<Address> {
        let width = memory_cells as u64;
        if addr < width {
            Ok(addr)
        } else {
            match policy {
                BoundsPolicy::Wrap => Ok(addr % width),
                BoundsPolicy::Clamp => Ok(width - 1),
                BoundsPolicy::Error => Err(OuroError::MemoryBoundsViolation {
                    address: addr,
                    max_address: width - 1,
                    operation,
                    location,
                }),
            }
        }
    }

    /// Bounds-checked read with configurable policy.
    ///
    /// # Arguments
    /// * `addr` - The raw address (u64) to read from
    /// * `policy` - How to handle out-of-bounds access
    /// * `location` - Source location for error reporting
    pub fn read_checked(
        &self,
        addr: u64,
        policy: BoundsPolicy,
        location: SourceLocation,
    ) -> OuroResult<Value> {
        let validated = self.validate_address_in(addr, policy, MemoryOperation::Read, location)?;
        Ok(self.cells[validated as usize].clone())
    }

    /// Bounds-checked write with configurable policy.
    ///
    /// # Arguments
    /// * `addr` - The raw address (u64) to write to
    /// * `val` - The value to write
    /// * `policy` - How to handle out-of-bounds access
    /// * `location` - Source location for error reporting
    pub fn write_checked(
        &mut self,
        addr: u64,
        val: Value,
        policy: BoundsPolicy,
        location: SourceLocation,
    ) -> OuroResult<()> {
        let validated = self.validate_address_in(addr, policy, MemoryOperation::Write, location)?;

        // Update incremental hash
        let old_val = self.cells[validated as usize].val;
        let new_val = val.val;
        if old_val != new_val {
            self.cached_hash ^= Self::hash_mix(validated, old_val);
            self.cached_hash ^= Self::hash_mix(validated, new_val);
        }

        self.cells[validated as usize] = val;
        Ok(())
    }

    /// Bounds-checked indexed read: read from base + offset.
    pub fn read_indexed(
        &self,
        base: u64,
        offset: u64,
        policy: BoundsPolicy,
        location: SourceLocation,
    ) -> OuroResult<Value> {
        let addr = base.wrapping_add(offset);
        self.read_checked(addr, policy, location)
    }

    /// Bounds-checked indexed write: write to base + offset.
    pub fn write_indexed(
        &mut self,
        base: u64,
        offset: u64,
        val: Value,
        policy: BoundsPolicy,
        location: SourceLocation,
    ) -> OuroResult<()> {
        let addr = base.wrapping_add(offset);
        self.write_checked(addr, val, policy, location)
    }

    /// Bounds-checked oracle read with temporal provenance.
    pub fn oracle_read(
        &self,
        addr: u64,
        policy: BoundsPolicy,
        location: SourceLocation,
    ) -> OuroResult<Value> {
        let validated =
            self.validate_address_in(addr, policy, MemoryOperation::Oracle, location)?;
        Ok(self.cells[validated as usize].clone())
    }

    /// Bounds-checked prophecy write.
    pub fn prophecy_write(
        &mut self,
        addr: u64,
        val: Value,
        policy: BoundsPolicy,
        location: SourceLocation,
    ) -> OuroResult<()> {
        let validated =
            self.validate_address_in(addr, policy, MemoryOperation::Prophecy, location)?;

        // Update incremental hash
        let old_val = self.cells[validated as usize].val;
        let new_val = val.val;
        if old_val != new_val {
            self.cached_hash ^= Self::hash_mix(validated, old_val);
            self.cached_hash ^= Self::hash_mix(validated, new_val);
        }

        self.cells[validated as usize] = val;
        Ok(())
    }

    /// Pack multiple values into contiguous memory with bounds checking.
    pub fn pack_checked(
        &mut self,
        base: u64,
        values: &[Value],
        policy: BoundsPolicy,
        location: SourceLocation,
    ) -> OuroResult<()> {
        for (i, val) in values.iter().enumerate() {
            let addr = base.wrapping_add(i as u64);
            let validated =
                self.validate_address_in(addr, policy, MemoryOperation::Pack, location.clone())?;

            let old_val = self.cells[validated as usize].val;
            let new_val = val.val;
            if old_val != new_val {
                self.cached_hash ^= Self::hash_mix(validated, old_val);
                self.cached_hash ^= Self::hash_mix(validated, new_val);
            }

            self.cells[validated as usize] = val.clone();
        }
        Ok(())
    }

    /// Unpack multiple values from contiguous memory with bounds checking.
    pub fn unpack_checked(
        &self,
        base: u64,
        count: usize,
        policy: BoundsPolicy,
        location: SourceLocation,
    ) -> OuroResult<Vec<Value>> {
        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let addr = base.wrapping_add(i as u64);
            let validated =
                self.validate_address_in(addr, policy, MemoryOperation::Unpack, location.clone())?;
            result.push(self.cells[validated as usize].clone());
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn test_memory_values_equal() {
        let mut m1 = Memory::new();
        let mut m2 = Memory::new();

        assert!(m1.values_equal(&m2));

        m1.write(100, Value::new(42));
        assert!(!m1.values_equal(&m2));

        m2.write(100, Value::new(42));
        assert!(m1.values_equal(&m2));
    }

    #[test]
    fn configured_width_controls_bounds_and_state_identity() {
        let mut wide = Memory::with_size(70_000);
        let narrow = Memory::new();
        let loc = SourceLocation::default();

        wide.write_checked(65_536, Value::new(42), BoundsPolicy::Error, loc.clone())
            .expect("address is valid in wide memory");
        assert_eq!(wide.read(65_536).val, 42);
        assert!(narrow
            .read_checked(65_536, BoundsPolicy::Error, loc)
            .is_err());
        assert!(!wide.values_equal(&narrow));
        assert_ne!(wide.exact_sparse_state(), narrow.exact_sparse_state());
    }

    #[test]
    fn ordering_uses_provenance_to_break_numeric_ties() {
        let mut pure = Memory::with_size(4);
        pure.write(1, Value::new(7));
        let mut temporal = Memory::with_size(4);
        temporal.write(1, Value::with_provenance(7, Provenance::single(2)));

        assert!(pure.numeric_values_equal(&temporal));
        assert_ne!(pure, temporal);
        assert_ne!(pure.cmp(&temporal), std::cmp::Ordering::Equal);
        let ordered = BTreeSet::from([pure, temporal]);
        assert_eq!(ordered.len(), 2);
    }

    #[test]
    fn ordering_keeps_the_complete_numeric_state_as_the_primary_key() {
        let mut lower_numeric = Memory::with_size(2);
        lower_numeric.write(0, Value::with_provenance(0, Provenance::single(9)));
        lower_numeric.write(1, Value::new(1));
        let mut higher_numeric = Memory::with_size(2);
        higher_numeric.write(1, Value::new(2));

        assert!(lower_numeric < higher_numeric);
    }

    #[test]
    fn exact_sparse_state_retains_zero_value_provenance_and_width() {
        let pure = Memory::with_size(4);
        let mut temporal = pure.clone();
        temporal.write(2, Value::with_provenance(0, Provenance::single(3)));

        assert!(temporal.sparse_state().is_empty());
        assert_eq!(temporal.exact_sparse_state().cells().len(), 1);
        assert_ne!(pure.exact_sparse_state(), temporal.exact_sparse_state());
        assert_ne!(
            Memory::with_size(4).exact_sparse_state(),
            Memory::with_size(5).exact_sparse_state()
        );
    }

    #[test]
    fn test_memory_diff() {
        let mut m1 = Memory::new();
        let mut m2 = Memory::new();

        m1.write(5, Value::new(10));
        m2.write(5, Value::new(20));
        m2.write(10, Value::new(30));

        let diff = m1.diff(&m2);
        assert!(diff.contains(&5));
        assert!(diff.contains(&10));
    }

    #[test]
    fn test_incremental_hash_correctness() {
        let mut mem = Memory::new();

        // After writes, cached hash should match recomputed hash
        mem.write(0, Value::new(42));
        assert_eq!(mem.state_hash(), mem.recompute_hash());

        mem.write(100, Value::new(123));
        assert_eq!(mem.state_hash(), mem.recompute_hash());

        // Overwriting should work correctly
        mem.write(0, Value::new(99));
        assert_eq!(mem.state_hash(), mem.recompute_hash());

        // Setting back to zero
        mem.write(0, Value::new(0));
        assert_eq!(mem.state_hash(), mem.recompute_hash());

        // Multiple addresses
        for i in 0..20 {
            mem.write(i, Value::new(i * 7));
        }
        assert_eq!(mem.state_hash(), mem.recompute_hash());
    }

    #[test]
    fn test_bounds_check_wrap_policy() {
        let mut mem = Memory::new();
        let loc = SourceLocation::default();

        // Write at valid address
        let result = mem.write_checked(100, Value::new(42), BoundsPolicy::Wrap, loc.clone());
        assert!(result.is_ok());
        assert_eq!(mem.read(100).val, 42);

        // Write at out-of-bounds address should wrap
        let result = mem.write_checked(65536, Value::new(99), BoundsPolicy::Wrap, loc.clone());
        assert!(result.is_ok());
        assert_eq!(mem.read(0).val, 99); // 65536 % 65536 = 0

        // Read at wrapped address
        let result = mem.read_checked(65536, BoundsPolicy::Wrap, loc);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().val, 99);
    }

    #[test]
    fn test_bounds_check_error_policy() {
        let mut mem = Memory::new();
        let loc = SourceLocation::default();

        // Valid address should work
        let result = mem.write_checked(100, Value::new(42), BoundsPolicy::Error, loc.clone());
        assert!(result.is_ok());

        // Invalid address should error
        let result = mem.write_checked(65536, Value::new(99), BoundsPolicy::Error, loc.clone());
        assert!(result.is_err());

        // Read at invalid address should error
        let result = mem.read_checked(70000, BoundsPolicy::Error, loc);
        assert!(result.is_err());
    }

    #[test]
    fn test_bounds_check_clamp_policy() {
        let mut mem = Memory::new();
        let loc = SourceLocation::default();

        // Write at out-of-bounds should clamp to max valid address
        let result = mem.write_checked(100000, Value::new(42), BoundsPolicy::Clamp, loc.clone());
        assert!(result.is_ok());
        assert_eq!(mem.read(65535).val, 42); // Clamped to max address

        // Read at clamped address
        let result = mem.read_checked(100000, BoundsPolicy::Clamp, loc);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().val, 42);
    }

    #[test]
    fn test_indexed_operations_checked() {
        let mut mem = Memory::new();
        let loc = SourceLocation::default();

        // Write with base + offset
        let result = mem.write_indexed(100, 5, Value::new(42), BoundsPolicy::Error, loc.clone());
        assert!(result.is_ok());
        assert_eq!(mem.read(105).val, 42);

        // Read with base + offset
        let result = mem.read_indexed(100, 5, BoundsPolicy::Error, loc);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().val, 42);
    }

    #[test]
    fn test_pack_unpack_checked() {
        let mut mem = Memory::new();
        let loc = SourceLocation::default();
        let values = vec![Value::new(10), Value::new(20), Value::new(30)];

        // Pack values
        let result = mem.pack_checked(100, &values, BoundsPolicy::Error, loc.clone());
        assert!(result.is_ok());
        assert_eq!(mem.read(100).val, 10);
        assert_eq!(mem.read(101).val, 20);
        assert_eq!(mem.read(102).val, 30);

        // Unpack values
        let result = mem.unpack_checked(100, 3, BoundsPolicy::Error, loc);
        assert!(result.is_ok());
        let unpacked = result.unwrap();
        assert_eq!(unpacked.len(), 3);
        assert_eq!(unpacked[0].val, 10);
        assert_eq!(unpacked[1].val, 20);
        assert_eq!(unpacked[2].val, 30);
    }

    #[test]
    fn test_validate_address() {
        let loc = SourceLocation::default();

        // Valid address
        let result =
            Memory::validate_address(100, BoundsPolicy::Error, MemoryOperation::Read, loc.clone());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 100);

        // Max valid address
        let result = Memory::validate_address(
            65535,
            BoundsPolicy::Error,
            MemoryOperation::Read,
            loc.clone(),
        );
        assert!(result.is_ok());

        // First invalid address
        let result =
            Memory::validate_address(65536, BoundsPolicy::Error, MemoryOperation::Read, loc);
        assert!(result.is_err());
    }
}
