//! Provenance tracking for causal dependency analysis.
//!
//! Each value in OUROCHRONOS carries provenance metadata indicating which
//! anamnesis cells influenced its computation. This enables:
//! - Causal graph construction
//! - Temporal core identification  
//! - Paradox diagnosis via dependency analysis

use std::collections::BTreeSet;
use std::rc::Rc;
use std::fmt;

/// Address type (16-bit index into memory)
pub type Address = u16;

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
    pub deps: Option<Rc<BTreeSet<Address>>>,
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
        Self { deps: Some(Rc::new(set)) }
    }
    
    /// Create provenance depending on multiple anamnesis cells.
    pub fn from_set(addrs: BTreeSet<Address>) -> Self {
        if addrs.is_empty() {
            Self::none()
        } else {
            Self { deps: Some(Rc::new(addrs)) }
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
    #[cold]
    #[inline(never)]
    fn merge_slow(&self, other: &Self) -> Self {
        match (&self.deps, &other.deps) {
            (None, None) => Self::NONE, // Shouldn't reach here, but handle it
            (Some(d), None) | (None, Some(d)) => Self { deps: Some(d.clone()) },
            (Some(d1), Some(d2)) => {
                if Rc::ptr_eq(d1, d2) {
                    // Same Rc, no need to clone
                    return Self { deps: Some(d1.clone()) };
                }
                let mut new_set = (**d1).clone();
                new_set.extend(d2.iter());
                Self { deps: Some(Rc::new(new_set)) }
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
}
