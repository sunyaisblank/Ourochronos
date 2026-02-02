//! Memory type for the OUROCHRONOS virtual machine.
//!
//! Memory consists of MEMORY_SIZE (65536) cells, each holding a Value.
//! Initially all cells are zero with no provenance.

use std::fmt;
use super::address::{Address, MEMORY_SIZE};
use super::value::Value;
use super::provenance::Provenance;
use super::error::{OuroError, OuroResult, BoundsPolicy, MemoryOperation, SourceLocation};

/// A snapshot of the memory state.
///
/// Memory consists of MEMORY_SIZE (65536) cells, each holding a Value.
/// Initially all cells are zero with no provenance.
///
/// Memory states are ordered lexicographically by cell values (address 0 first).
/// This ordering enables deterministic selection among equal-action fixed points.
///
/// The hash is maintained incrementally for O(1) state_hash() lookups.
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
    /// Lexicographic comparison of memory states by cell values.
    ///
    /// Compares cells from address 0 upward, returning on first difference.
    /// This provides a total ordering for deterministic fixed-point selection.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare only values, not provenance (for determinism)
        for (a, b) in self.cells.iter().zip(other.cells.iter()) {
            match a.val.cmp(&b.val) {
                std::cmp::Ordering::Equal => continue,
                other => return other,
            }
        }
        std::cmp::Ordering::Equal
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
        let nonzero: Vec<_> = self.cells.iter()
            .enumerate()
            .filter(|(_, v)| v.val != 0)
            .collect();

        if nonzero.is_empty() {
            write!(f, "Memory{{all zero}}")
        } else {
            write!(f, "Memory{{")?;
            for (i, (addr, val)) in nonzero.iter().enumerate() {
                if i > 0 { write!(f, ", ")?; }
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
        let combined = (addr as u64).wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(val);
        // Apply finalization mixing (MurmurHash3-style)
        let mut h = combined.wrapping_mul(0x517cc1b727220a95);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        h
    }

    /// Create a new memory state with all cells set to zero.
    pub fn new() -> Self {
        Self {
            cells: vec![Value::ZERO; MEMORY_SIZE],
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

    /// Check if two memory states are equal (by value, ignoring provenance).
    /// This is the fixed-point check: P_final = A_initial.
    ///
    /// Uses cached hash for O(1) early rejection when states differ (common case).
    /// Falls back to O(N) cell-by-cell comparison only when hashes match.
    pub fn values_equal(&self, other: &Memory) -> bool {
        // Fast path: different hashes guarantee different content
        if self.cached_hash != other.cached_hash {
            return false;
        }
        // Same hash: verify cell-by-cell (hash collision possible but rare)
        self.cells.iter()
            .zip(other.cells.iter())
            .all(|(v1, v2)| v1.val == v2.val)
    }

    /// Find addresses where values differ between two memory states.
    pub fn diff(&self, other: &Memory) -> Vec<Address> {
        self.cells.iter()
            .zip(other.cells.iter())
            .enumerate()
            .filter(|(_, (v1, v2))| v1.val != v2.val)
            .map(|(i, _)| i as Address)
            .collect()
    }

    /// Find all non-zero cells.
    pub fn non_zero_cells(&self) -> Vec<(Address, &Value)> {
        self.cells.iter()
            .enumerate()
            .filter(|(_, v)| v.val != 0)
            .map(|(i, v)| (i as Address, v))
            .collect()
    }

    /// Get a hash of the memory state (for cycle detection).
    ///
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
        self.cells.iter()
            .enumerate()
            .filter(|(_, v)| v.val != 0)
            .map(|(i, v)| (i as Address, v))
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
        self.cells.iter()
            .enumerate()
            .map(|(i, v)| (i as Address, v))
    }

    // ═══════════════════════════════════════════════════════════════════
    // Bounds-Checked Operations
    // ═══════════════════════════════════════════════════════════════════

    /// Validate an address against memory bounds.
    ///
    /// Returns `Ok(addr as Address)` if within bounds, or handles according to policy.
    #[inline]
    pub fn validate_address(
        addr: u64,
        policy: BoundsPolicy,
        operation: MemoryOperation,
        location: SourceLocation,
    ) -> OuroResult<Address> {
        if addr < MEMORY_SIZE as u64 {
            Ok(addr as Address)
        } else {
            match policy {
                BoundsPolicy::Wrap => {
                    Ok((addr % MEMORY_SIZE as u64) as Address)
                }
                BoundsPolicy::Clamp => {
                    Ok((MEMORY_SIZE - 1) as Address)
                }
                BoundsPolicy::Error => {
                    Err(OuroError::MemoryBoundsViolation {
                        address: addr,
                        max_address: (MEMORY_SIZE - 1) as Address,
                        operation,
                        location,
                    })
                }
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
        let validated = Self::validate_address(addr, policy, MemoryOperation::Read, location)?;
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
        let validated = Self::validate_address(addr, policy, MemoryOperation::Write, location)?;

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
        let validated = Self::validate_address(addr, policy, MemoryOperation::Oracle, location)?;
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
        let validated = Self::validate_address(addr, policy, MemoryOperation::Prophecy, location)?;

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
            let validated = Self::validate_address(addr, policy, MemoryOperation::Pack, location.clone())?;

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
            let validated = Self::validate_address(addr, policy, MemoryOperation::Unpack, location.clone())?;
            result.push(self.cells[validated as usize].clone());
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
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
            mem.write(i, Value::new(i as u64 * 7));
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
        let result = Memory::validate_address(100, BoundsPolicy::Error, MemoryOperation::Read, loc.clone());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 100);

        // Max valid address
        let result = Memory::validate_address(65535, BoundsPolicy::Error, MemoryOperation::Read, loc.clone());
        assert!(result.is_ok());

        // First invalid address
        let result = Memory::validate_address(65536, BoundsPolicy::Error, MemoryOperation::Read, loc);
        assert!(result.is_err());
    }
}
