//! Provenance tracking for causal dependency analysis.
//!
//! Each value in OUROCHRONOS carries provenance metadata indicating which
//! anamnesis cells influenced its computation. This enables:
//! - Causal graph construction
//! - Temporal core identification
//! - Paradox diagnosis via dependency analysis

use std::collections::BTreeSet;
use std::sync::Arc;
use std::fmt;

use super::address::Address;

/// Provenance tracks which anamnesis cells a value depends on.
///
/// The provenance lattice:
/// - ⊥ (none): No temporal dependency
/// - Oracle(A): Depends on anamnesis cells in set A
/// - Computed: Union of input provenances
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct Provenance {
    /// The set of anamnesis addresses this value depends on.
    /// None represents ⊥ (no temporal dependency).
    /// Some(set) represents Oracle(set).
    pub deps: Option<Arc<BTreeSet<Address>>>,
}

impl fmt::Debug for Provenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.deps {
            None => write!(f, "⊥"),
            Some(set) if set.is_empty() => write!(f, "⊥"),
            Some(set) => {
                write!(f, "Oracle{{")?;
                for (i, addr) in set.iter().enumerate() {
                    if i > 0 { write!(f, ",")?; }
                    write!(f, "{}", addr)?;
                }
                write!(f, "}}")
            }
        }
    }
}

impl Provenance {
    /// Create provenance with no temporal dependency (⊥).
    pub const fn none() -> Self {
        Self { deps: None }
    }

    /// Create provenance depending on a single anamnesis cell.
    pub fn single(addr: Address) -> Self {
        let mut set = BTreeSet::new();
        set.insert(addr);
        Self { deps: Some(Arc::new(set)) }
    }

    /// Create provenance depending on multiple anamnesis cells.
    pub fn from_set(addrs: BTreeSet<Address>) -> Self {
        if addrs.is_empty() {
            Self::none()
        } else {
            Self { deps: Some(Arc::new(addrs)) }
        }
    }

    /// Merge two provenances (lattice join: ⊔).
    /// The result depends on all cells that either input depends on.
    ///
    /// This is the hot path for all arithmetic operations. Optimised for
    /// the common case where both operands are pure (no temporal dependency).
    #[inline(always)]
    pub fn merge(&self, other: &Self) -> Self {
        // Fast path: both pure (no deps) - this is the common case in non-temporal code
        // CRITICAL: This check must be as cheap as possible since it runs on EVERY operation
        if self.deps.is_none() && other.deps.is_none() {
            // Return a static constant to avoid any allocation or construction
            return Self::NONE;
        }
        // Slow path: at least one operand has temporal dependency
        self.merge_slow(other)
    }

    /// Slow path for merge when at least one operand has dependencies.
    /// Marked cold to help branch prediction on the fast path.
    ///
    /// Optimizations:
    /// - Returns existing Arc if both point to same set (Arc::ptr_eq)
    /// - Reuses larger set without cloning if smaller is a subset
    /// - Only allocates new set when truly necessary
    #[cold]
    #[inline(never)]
    fn merge_slow(&self, other: &Self) -> Self {
        match (&self.deps, &other.deps) {
            (None, None) => Self::NONE, // Shouldn't reach here, but handle it
            (Some(d), None) | (None, Some(d)) => Self { deps: Some(d.clone()) },
            (Some(d1), Some(d2)) => {
                // Same Arc, no need to clone
                if Arc::ptr_eq(d1, d2) {
                    return Self { deps: Some(d1.clone()) };
                }
                // Reuse larger set if smaller is a subset (avoids allocation)
                if d1.len() >= d2.len() && d2.iter().all(|x| d1.contains(x)) {
                    return Self { deps: Some(d1.clone()) };
                }
                if d2.len() > d1.len() && d1.iter().all(|x| d2.contains(x)) {
                    return Self { deps: Some(d2.clone()) };
                }
                // Must create new set
                let mut new_set = (**d1).clone();
                new_set.extend(d2.iter().copied());
                Self { deps: Some(Arc::new(new_set)) }
            }
        }
    }

    /// Static constant for the pure (no dependency) provenance.
    /// Used to avoid allocation in the fast path.
    pub const NONE: Self = Self { deps: None };

    /// Check if this value has any temporal dependency.
    pub fn is_temporal(&self) -> bool {
        match &self.deps {
            None => false,
            Some(set) => !set.is_empty(),
        }
    }

    /// Check if this value is temporally pure (no oracle dependency).
    pub fn is_pure(&self) -> bool {
        !self.is_temporal()
    }

    /// Get the set of addresses this value depends on.
    pub fn dependencies(&self) -> impl Iterator<Item = Address> + '_ {
        self.deps.iter().flat_map(|set| set.iter().copied())
    }

    /// Count the number of dependencies.
    pub fn dependency_count(&self) -> usize {
        self.deps.as_ref().map(|s| s.len()).unwrap_or(0)
    }

    /// Merge with algebraic awareness for improved precision.
    ///
    /// This method exploits algebraic identities to reduce false positives
    /// in provenance tracking. For example:
    /// - `0 × X = 0` regardless of X's provenance
    /// - `X + 0 = X` preserves only X's provenance
    /// - `0 & X = 0` regardless of X's provenance
    ///
    /// # Sound but may over-approximate
    ///
    /// This optimization is sound (never under-approximates dependencies)
    /// but may still over-approximate in complex control flow scenarios.
    #[inline]
    pub fn merge_algebraic(
        &self,
        other: &Self,
        op: AlgebraicOp,
        self_val: u64,
        other_val: u64,
    ) -> Self {
        // Fast path: both pure - no algebraic optimization needed
        if self.is_pure() && other.is_pure() {
            return Self::NONE;
        }

        // Apply algebraic simplification rules
        match op {
            // Multiplicative zero: 0 × X = 0 (result is pure regardless of X)
            AlgebraicOp::Mul if self_val == 0 || other_val == 0 => Self::NONE,

            // Bitwise AND zero: 0 & X = 0 (result is pure regardless of X)
            AlgebraicOp::And if self_val == 0 || other_val == 0 => Self::NONE,

            // Additive identity: 0 + X = X (preserves X's provenance only)
            AlgebraicOp::Add if self_val == 0 => other.clone(),
            AlgebraicOp::Add if other_val == 0 => self.clone(),

            // Subtractive identity: X - 0 = X (preserves X's provenance only)
            AlgebraicOp::Sub if other_val == 0 => self.clone(),

            // Bitwise OR with all-ones: X | MAX = MAX (result has only MAX's provenance)
            AlgebraicOp::Or if self_val == u64::MAX => self.clone(),
            AlgebraicOp::Or if other_val == u64::MAX => other.clone(),

            // Bitwise XOR with zero: X ^ 0 = X (preserves X's provenance only)
            AlgebraicOp::Xor if self_val == 0 => other.clone(),
            AlgebraicOp::Xor if other_val == 0 => self.clone(),

            // Division by one: X / 1 = X (preserves X's provenance only)
            AlgebraicOp::Div if other_val == 1 => self.clone(),

            // Modulo by one: X % 1 = 0 (result is always 0, so pure)
            AlgebraicOp::Mod if other_val == 1 => Self::NONE,

            // Shift by zero: X << 0 = X, X >> 0 = X
            AlgebraicOp::Shl | AlgebraicOp::Shr if other_val == 0 => self.clone(),

            // Comparison with self (if both have same provenance and value)
            // This is tricky - skip for now as we don't have value equality info

            // Conservative fallback: union of dependencies
            _ => self.merge(other),
        }
    }
}

/// Algebraic operation type for provenance optimization.
///
/// Used to identify which algebraic simplification rules apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlgebraicOp {
    /// Addition: `X + 0 = X`
    Add,
    /// Subtraction: `X - 0 = X`
    Sub,
    /// Multiplication: `0 × X = 0`
    Mul,
    /// Division: `X / 1 = X`
    Div,
    /// Modulo: `X % 1 = 0`
    Mod,
    /// Bitwise AND: `0 & X = 0`
    And,
    /// Bitwise OR: `X | MAX = MAX`
    Or,
    /// Bitwise XOR: `X ^ 0 = X`
    Xor,
    /// Left shift: `X << 0 = X`
    Shl,
    /// Right shift: `X >> 0 = X`
    Shr,
    /// Operations with no algebraic simplification
    Other,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_none_is_pure() {
        let p = Provenance::none();
        assert!(p.is_pure());
        assert!(!p.is_temporal());
    }

    #[test]
    fn test_single_is_temporal() {
        let p = Provenance::single(42);
        assert!(!p.is_pure());
        assert!(p.is_temporal());
        assert_eq!(p.dependency_count(), 1);
    }

    #[test]
    fn test_merge() {
        let p1 = Provenance::single(1);
        let p2 = Provenance::single(2);
        let merged = p1.merge(&p2);
        assert_eq!(merged.dependency_count(), 2);
    }

    #[test]
    fn test_merge_with_none() {
        let p1 = Provenance::single(1);
        let p2 = Provenance::none();
        let merged = p1.merge(&p2);
        assert_eq!(merged.dependency_count(), 1);
    }

    // Algebraic optimization tests

    #[test]
    fn test_algebraic_mul_zero() {
        // 0 × temporal = 0 (pure result)
        let p_pure = Provenance::none();
        let p_temporal = Provenance::single(42);

        // 0 * X = 0 (pure zero times temporal is pure)
        let result = p_pure.merge_algebraic(&p_temporal, AlgebraicOp::Mul, 0, 5);
        assert!(result.is_pure(), "0 * temporal should be pure");

        // X * 0 = 0 (temporal times pure zero is pure)
        let result = p_temporal.merge_algebraic(&p_pure, AlgebraicOp::Mul, 5, 0);
        assert!(result.is_pure(), "temporal * 0 should be pure");
    }

    #[test]
    fn test_algebraic_add_identity() {
        // X + 0 = X (preserves only X's provenance)
        let p_temporal = Provenance::single(42);
        let p_pure = Provenance::none();

        // temporal + 0 = temporal
        let result = p_temporal.merge_algebraic(&p_pure, AlgebraicOp::Add, 5, 0);
        assert!(result.is_temporal());
        assert_eq!(result.dependency_count(), 1);

        // 0 + temporal = temporal
        let result = p_pure.merge_algebraic(&p_temporal, AlgebraicOp::Add, 0, 5);
        assert!(result.is_temporal());
        assert_eq!(result.dependency_count(), 1);
    }

    #[test]
    fn test_algebraic_and_zero() {
        // 0 & X = 0 (pure result)
        let p_temporal = Provenance::single(42);
        let p_pure = Provenance::none();

        let result = p_pure.merge_algebraic(&p_temporal, AlgebraicOp::And, 0, 0xFF);
        assert!(result.is_pure(), "0 & temporal should be pure");

        let result = p_temporal.merge_algebraic(&p_pure, AlgebraicOp::And, 0xFF, 0);
        assert!(result.is_pure(), "temporal & 0 should be pure");
    }

    #[test]
    fn test_algebraic_sub_identity() {
        // X - 0 = X (preserves X's provenance)
        let p_temporal = Provenance::single(42);
        let p_pure = Provenance::none();

        let result = p_temporal.merge_algebraic(&p_pure, AlgebraicOp::Sub, 5, 0);
        assert!(result.is_temporal());
        assert_eq!(result.dependency_count(), 1);
    }

    #[test]
    fn test_algebraic_xor_zero() {
        // X ^ 0 = X
        let p_temporal = Provenance::single(42);
        let p_pure = Provenance::none();

        let result = p_temporal.merge_algebraic(&p_pure, AlgebraicOp::Xor, 5, 0);
        assert!(result.is_temporal());

        let result = p_pure.merge_algebraic(&p_temporal, AlgebraicOp::Xor, 0, 5);
        assert!(result.is_temporal());
    }

    #[test]
    fn test_algebraic_div_one() {
        // X / 1 = X
        let p_temporal = Provenance::single(42);
        let p_pure = Provenance::none();

        let result = p_temporal.merge_algebraic(&p_pure, AlgebraicOp::Div, 10, 1);
        assert!(result.is_temporal());
        assert_eq!(result.dependency_count(), 1);
    }

    #[test]
    fn test_algebraic_mod_one() {
        // X % 1 = 0 (always pure)
        let p_temporal = Provenance::single(42);
        let p_pure = Provenance::none();

        let result = p_temporal.merge_algebraic(&p_pure, AlgebraicOp::Mod, 10, 1);
        assert!(result.is_pure(), "X % 1 should be pure (result is always 0)");
    }

    #[test]
    fn test_algebraic_shift_zero() {
        // X << 0 = X, X >> 0 = X
        let p_temporal = Provenance::single(42);
        let p_pure = Provenance::none();

        let result = p_temporal.merge_algebraic(&p_pure, AlgebraicOp::Shl, 5, 0);
        assert!(result.is_temporal());
        assert_eq!(result.dependency_count(), 1);

        let result = p_temporal.merge_algebraic(&p_pure, AlgebraicOp::Shr, 5, 0);
        assert!(result.is_temporal());
        assert_eq!(result.dependency_count(), 1);
    }

    #[test]
    fn test_algebraic_fallback() {
        // Non-identity cases should still merge provenances
        let p1 = Provenance::single(1);
        let p2 = Provenance::single(2);

        let result = p1.merge_algebraic(&p2, AlgebraicOp::Add, 3, 4);
        assert!(result.is_temporal());
        assert_eq!(result.dependency_count(), 2);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Property Tests for Large Provenance Sets
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_large_provenance_merge_preserves_all() {
        // Property: merge preserves all dependencies (|merge(A,B)| >= max(|A|,|B|))
        let mut set1 = BTreeSet::new();
        let mut set2 = BTreeSet::new();

        // Create two sets of 50 addresses each, with some overlap
        for i in 0..50 {
            set1.insert(i as Address);
        }
        for i in 25..75 {
            set2.insert(i as Address);
        }

        let p1 = Provenance::from_set(set1.clone());
        let p2 = Provenance::from_set(set2.clone());

        let merged = p1.merge(&p2);

        // Should have all unique addresses
        assert_eq!(merged.dependency_count(), 75); // 0-74 inclusive

        // Verify all original dependencies are preserved
        for addr in set1.iter() {
            assert!(
                merged.dependencies().any(|a| a == *addr),
                "Address {} from set1 should be in merged provenance",
                addr
            );
        }
        for addr in set2.iter() {
            assert!(
                merged.dependencies().any(|a| a == *addr),
                "Address {} from set2 should be in merged provenance",
                addr
            );
        }
    }

    #[test]
    fn test_large_provenance_chain_merges() {
        // Property: chained merges preserve all dependencies
        let mut accumulated = Provenance::none();
        let mut expected_addrs = BTreeSet::new();

        // Merge 100 single-address provenances
        for i in 0..100 {
            let addr = i as Address;
            let p = Provenance::single(addr);
            accumulated = accumulated.merge(&p);
            expected_addrs.insert(addr);

            // Verify count after each merge
            assert_eq!(
                accumulated.dependency_count(),
                expected_addrs.len(),
                "After merging {} addresses, count should match",
                i + 1
            );
        }

        // Verify all 100 addresses are present
        assert_eq!(accumulated.dependency_count(), 100);
        for addr in expected_addrs.iter() {
            assert!(
                accumulated.dependencies().any(|a| a == *addr),
                "Address {} should be preserved through merge chain",
                addr
            );
        }
    }

    #[test]
    fn test_provenance_merge_commutativity() {
        // Property: merge(A, B) == merge(B, A) in terms of dependency set
        let set1: BTreeSet<Address> = (0..30).collect();
        let set2: BTreeSet<Address> = (20..50).collect();

        let p1 = Provenance::from_set(set1);
        let p2 = Provenance::from_set(set2);

        let merge_1_2 = p1.merge(&p2);
        let merge_2_1 = p2.merge(&p1);

        assert_eq!(
            merge_1_2.dependency_count(),
            merge_2_1.dependency_count(),
            "merge(A,B) and merge(B,A) should have same size"
        );

        // Verify same dependencies
        let deps_1_2: BTreeSet<_> = merge_1_2.dependencies().collect();
        let deps_2_1: BTreeSet<_> = merge_2_1.dependencies().collect();
        assert_eq!(deps_1_2, deps_2_1, "merge should be commutative");
    }

    #[test]
    fn test_provenance_merge_associativity() {
        // Property: merge(merge(A, B), C) == merge(A, merge(B, C))
        let p1 = Provenance::from_set((0..20).collect());
        let p2 = Provenance::from_set((10..30).collect());
        let p3 = Provenance::from_set((20..40).collect());

        let left_assoc = p1.merge(&p2).merge(&p3);
        let right_assoc = p1.merge(&p2.merge(&p3));

        assert_eq!(
            left_assoc.dependency_count(),
            right_assoc.dependency_count(),
            "merge should be associative in size"
        );

        let deps_left: BTreeSet<_> = left_assoc.dependencies().collect();
        let deps_right: BTreeSet<_> = right_assoc.dependencies().collect();
        assert_eq!(deps_left, deps_right, "merge should be associative");
    }

    #[test]
    fn test_subset_reuse_optimization() {
        // Property: when one set is subset of another, no new allocation needed
        let small_set: BTreeSet<Address> = (0..10).collect();
        let large_set: BTreeSet<Address> = (0..100).collect();

        let p_small = Provenance::from_set(small_set);
        let p_large = Provenance::from_set(large_set.clone());

        // Merge small into large - should reuse large's Arc
        let merged = p_small.merge(&p_large);
        assert_eq!(merged.dependency_count(), 100);

        // Merge large into small - should also reuse large's Arc
        let merged2 = p_large.merge(&p_small);
        assert_eq!(merged2.dependency_count(), 100);

        // Both should have the same dependencies
        let deps: BTreeSet<_> = merged.dependencies().collect();
        assert_eq!(deps, large_set);
    }

    #[test]
    fn test_arc_sharing_same_set() {
        // Property: merging identical Arcs should reuse the Arc
        let set: BTreeSet<Address> = (0..50).collect();
        let p1 = Provenance::from_set(set.clone());
        let p2 = p1.clone(); // Shares the same Arc

        let merged = p1.merge(&p2);
        assert_eq!(merged.dependency_count(), 50);
    }

    #[test]
    fn test_merge_with_empty_returns_other() {
        // Property: merge(empty, X) == X, merge(X, empty) == X
        let set: BTreeSet<Address> = (0..100).collect();
        let p_temporal = Provenance::from_set(set.clone());
        let p_empty = Provenance::none();

        let merged1 = p_empty.merge(&p_temporal);
        let merged2 = p_temporal.merge(&p_empty);

        assert_eq!(merged1.dependency_count(), 100);
        assert_eq!(merged2.dependency_count(), 100);
    }
}
