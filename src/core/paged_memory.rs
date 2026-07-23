//! Persistent, page-granular copy-on-write memory snapshots.
//!
//! [`PagedMemory`] is intended as the state representation used by temporal
//! search, replay caches, and serialized state journals. Cloning a snapshot is
//! O(1). The first write to a shared snapshot clones its page table and only
//! the page containing the written cell; all other pages remain shared.
//!
//! State equality is exact: both the numeric component and the complete
//! provenance representation of every [`Value`] participate. Cached hashes are
//! accelerators only. A hash match is always followed by exact comparison.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use super::address::{Address, MEMORY_SIZE};
use super::error::{BoundsPolicy, MemoryOperation, OuroError, OuroResult, SourceLocation};
use super::provenance::Provenance;
use super::value::Value;

/// Number of cells in a physical copy-on-write page.
///
/// A page is deliberately small enough that a random write has bounded copy
/// cost, while remaining large enough that page-table overhead is modest.
pub const PAGE_CELLS: usize = 256;

/// Default maximum virtual cell count accepted by the paged implementation.
pub const DEFAULT_MAX_CELLS: usize = 16_777_216;

/// Default maximum page-table entries accepted by the paged implementation.
pub const DEFAULT_MAX_PAGES: usize = DEFAULT_MAX_CELLS / PAGE_CELLS;

/// Explicit construction limits for a [`PagedMemory`] allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedMemoryLimits {
    /// Maximum virtual cells in a snapshot.
    pub max_cells: usize,
    /// Maximum entries in the snapshot's page table.
    pub max_pages: usize,
}

impl Default for PagedMemoryLimits {
    fn default() -> Self {
        Self {
            max_cells: DEFAULT_MAX_CELLS,
            max_pages: DEFAULT_MAX_PAGES,
        }
    }
}

/// Failure to construct or materialize a bounded paged-memory operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PagedMemoryError {
    /// A memory width must contain at least one cell.
    ZeroWidth,
    /// The configured cell limit is itself invalid.
    ZeroCellLimit,
    /// The configured page limit is itself invalid.
    ZeroPageLimit,
    /// The requested width exceeds the explicit virtual-cell limit.
    CellLimitExceeded { requested: usize, limit: usize },
    /// The requested width requires more page entries than allowed.
    PageLimitExceeded { requested: usize, limit: usize },
    /// The host allocator rejected the bounded page-table allocation.
    AllocationFailed { requested_pages: usize },
    /// A caller-supplied bound prevented materializing a complete diff.
    DifferenceLimitExceeded { limit: usize },
}

impl fmt::Display for PagedMemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroWidth => write!(f, "paged memory must contain at least one cell"),
            Self::ZeroCellLimit => write!(f, "paged-memory cell limit must be positive"),
            Self::ZeroPageLimit => write!(f, "paged-memory page limit must be positive"),
            Self::CellLimitExceeded { requested, limit } => write!(
                f,
                "paged-memory width {requested} exceeds the cell limit {limit}"
            ),
            Self::PageLimitExceeded { requested, limit } => write!(
                f,
                "paged-memory page count {requested} exceeds the page limit {limit}"
            ),
            Self::AllocationFailed { requested_pages } => write!(
                f,
                "could not allocate a paged-memory table with {requested_pages} entries"
            ),
            Self::DifferenceLimitExceeded { limit } => {
                write!(f, "memory diff contains more than {limit} addresses")
            }
        }
    }
}

impl std::error::Error for PagedMemoryError {}

#[derive(Clone)]
struct Page {
    cells: Box<[Value]>,
    /// XOR of exact, absolute-address cell contributions in this page.
    cached_hash: u64,
}

impl Page {
    fn zero() -> Result<Self, PagedMemoryError> {
        let mut cells = Vec::new();
        cells
            .try_reserve_exact(PAGE_CELLS)
            .map_err(|_| PagedMemoryError::AllocationFailed { requested_pages: 1 })?;
        cells.resize(PAGE_CELLS, Value::ZERO);
        Ok(Self {
            cells: cells.into_boxed_slice(),
            cached_hash: 0,
        })
    }
}

/// A persistent finite memory snapshot with page-level copy-on-write writes.
///
/// The page table and every page are reference counted. Cloning this value does
/// not copy the table or cells. A subsequent write detaches the table (O(pages))
/// and one page (O(`PAGE_CELLS`)); unchanged pages remain shared.
#[derive(Clone)]
pub struct PagedMemory {
    width: usize,
    pages: Arc<Vec<Arc<Page>>>,
    /// Width contribution XOR exact contributions of all non-default cells.
    cached_hash: u64,
}

impl Default for PagedMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl PagedMemory {
    /// Construct the backwards-compatible 65,536-cell memory.
    pub fn new() -> Self {
        Self::with_size(MEMORY_SIZE).expect("the default memory size is within default limits")
    }

    /// Construct an all-zero snapshot with a positive, explicitly bounded width.
    pub fn with_size(width: usize) -> Result<Self, PagedMemoryError> {
        Self::with_limits(width, PagedMemoryLimits::default())
    }

    /// Construct an all-zero snapshot under caller-supplied allocation limits.
    pub fn with_limits(width: usize, limits: PagedMemoryLimits) -> Result<Self, PagedMemoryError> {
        if width == 0 {
            return Err(PagedMemoryError::ZeroWidth);
        }
        if limits.max_cells == 0 {
            return Err(PagedMemoryError::ZeroCellLimit);
        }
        if limits.max_pages == 0 {
            return Err(PagedMemoryError::ZeroPageLimit);
        }
        if width > limits.max_cells {
            return Err(PagedMemoryError::CellLimitExceeded {
                requested: width,
                limit: limits.max_cells,
            });
        }

        let page_count =
            width
                .checked_add(PAGE_CELLS - 1)
                .ok_or(PagedMemoryError::CellLimitExceeded {
                    requested: width,
                    limit: limits.max_cells,
                })?
                / PAGE_CELLS;
        if page_count > limits.max_pages {
            return Err(PagedMemoryError::PageLimitExceeded {
                requested: page_count,
                limit: limits.max_pages,
            });
        }

        // Every initially-zero virtual page points to the same physical page.
        let zero_page = Arc::new(Page::zero()?);
        let mut pages = Vec::new();
        pages
            .try_reserve_exact(page_count)
            .map_err(|_| PagedMemoryError::AllocationFailed {
                requested_pages: page_count,
            })?;
        pages.resize(page_count, zero_page);

        Ok(Self {
            width,
            pages: Arc::new(pages),
            cached_hash: hash_width(width),
        })
    }

    /// Number of addressable cells.
    #[inline]
    pub fn len(&self) -> usize {
        self.width
    }

    /// A paged memory is never empty because construction requires positive width.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Number of virtual entries in the page table.
    #[inline]
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Conservative charge for retaining this snapshot independently. Shared
    /// pages inside the snapshot are counted once; sharing with another
    /// snapshot is deliberately not assumed by aggregate resource budgets.
    pub fn retained_size_charge(&self) -> usize {
        let mut bytes = std::mem::size_of::<Self>().saturating_add(
            self.pages
                .len()
                .saturating_mul(std::mem::size_of::<Arc<Page>>()),
        );
        let mut seen = HashSet::new();
        for page in self.pages.iter() {
            let identity = Arc::as_ptr(page);
            if seen.insert(identity) {
                bytes = bytes.saturating_add(std::mem::size_of::<Page>());
                for value in page.cells.iter() {
                    bytes = bytes.saturating_add(value.retained_size_charge());
                }
            }
        }
        bytes
    }

    /// Borrow a cell when the address is in range.
    #[inline]
    pub fn get(&self, address: Address) -> Option<&Value> {
        let index = usize::try_from(address).ok()?;
        if index >= self.width {
            return None;
        }
        Some(self.cell_at(index))
    }

    /// Read under strict, wrapping, or clamping bounds semantics.
    pub fn read_checked(
        &self,
        address: u64,
        policy: BoundsPolicy,
        location: SourceLocation,
    ) -> OuroResult<Value> {
        let address = self.validate_address(address, policy, MemoryOperation::Read, location)?;
        Ok(self.cell_at(address as usize).clone())
    }

    /// Write under strict, wrapping, or clamping bounds semantics.
    pub fn write_checked(
        &mut self,
        address: u64,
        value: Value,
        policy: BoundsPolicy,
        location: SourceLocation,
    ) -> OuroResult<()> {
        let address = self.validate_address(address, policy, MemoryOperation::Write, location)?;
        self.write_validated(address as usize, value);
        Ok(())
    }

    /// Strict write convenience method.
    pub fn write(&mut self, address: Address, value: Value) -> OuroResult<()> {
        self.write_checked(
            address,
            value,
            BoundsPolicy::Error,
            SourceLocation::default(),
        )
    }

    /// Read an oracle cell using checked bounds semantics.
    pub fn oracle_read(
        &self,
        address: u64,
        policy: BoundsPolicy,
        location: SourceLocation,
    ) -> OuroResult<Value> {
        let address = self.validate_address(address, policy, MemoryOperation::Oracle, location)?;
        Ok(self.cell_at(address as usize).clone())
    }

    /// Write a prophecy cell using checked bounds semantics.
    pub fn prophecy_write(
        &mut self,
        address: u64,
        value: Value,
        policy: BoundsPolicy,
        location: SourceLocation,
    ) -> OuroResult<()> {
        let address =
            self.validate_address(address, policy, MemoryOperation::Prophecy, location)?;
        self.write_validated(address as usize, value);
        Ok(())
    }

    /// Iterate every cell in canonical serialization order: increasing address.
    ///
    /// This iterator includes pure zero cells, making it appropriate for dense,
    /// versioned serializers that need the configured width reflected exactly.
    pub fn iter_serialization(&self) -> impl ExactSizeIterator<Item = (Address, &Value)> + '_ {
        (0..self.width).map(move |index| (index as Address, self.cell_at(index)))
    }

    /// Alias for canonical dense iteration.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (Address, &Value)> + '_ {
        self.iter_serialization()
    }

    /// Iterate cells whose numeric value is nonzero, in increasing address order.
    ///
    /// A zero-valued cell with provenance is intentionally not returned here;
    /// use [`Self::iter_sparse`] when exact state reconstruction is required.
    pub fn iter_nonzero(&self) -> impl Iterator<Item = (Address, &Value)> + '_ {
        self.iter().filter(|(_, value)| value.val != 0)
    }

    /// Iterate every cell that differs exactly from pure zero.
    ///
    /// Unlike numeric nonzero iteration, this includes zero-valued cells that
    /// carry provenance. The result is sufficient for exact reconstruction.
    pub fn iter_sparse(&self) -> impl Iterator<Item = (Address, &Value)> + '_ {
        self.iter().filter(|(_, value)| **value != Value::ZERO)
    }

    /// Materialize the exact sparse state used by collision-safe temporal caches.
    pub fn sparse_state(&self) -> Vec<(Address, Value)> {
        self.iter_sparse()
            .map(|(address, value)| (address, value.clone()))
            .collect()
    }

    /// Materialize the legacy numeric-only sparse projection.
    ///
    /// This projection must not be used as an exact state identity because it
    /// intentionally discards provenance.
    pub fn numeric_sparse_state(&self) -> Vec<(Address, u64)> {
        self.iter_nonzero()
            .map(|(address, value)| (address, value.val))
            .collect()
    }

    /// Iterate addresses whose exact cells differ, in increasing order.
    ///
    /// Width is part of state identity. Therefore every address present in only
    /// one snapshot is reported, even when the present cell is pure zero.
    pub fn iter_differences<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = Address> + 'a {
        let max_width = self.width.max(other.width);
        (0..max_width).filter_map(move |index| {
            let left = (index < self.width).then(|| self.cell_at(index));
            let right = (index < other.width).then(|| other.cell_at(index));
            (left != right).then_some(index as Address)
        })
    }

    /// Materialize every exact differing address.
    pub fn diff(&self, other: &Self) -> Vec<Address> {
        self.iter_differences(other).collect()
    }

    /// Materialize an exact diff while enforcing a caller-selected result limit.
    pub fn diff_limited(
        &self,
        other: &Self,
        limit: usize,
    ) -> Result<Vec<Address>, PagedMemoryError> {
        let mut differences = Vec::new();
        differences
            .try_reserve(limit.min(self.width.max(other.width)))
            .map_err(|_| PagedMemoryError::AllocationFailed {
                requested_pages: self.page_count().max(other.page_count()),
            })?;
        for address in self.iter_differences(other) {
            if differences.len() == limit {
                return Err(PagedMemoryError::DifferenceLimitExceeded { limit });
            }
            differences.push(address);
        }
        Ok(differences)
    }

    /// Exact state equality with cached-hash rejection followed by full checking.
    #[inline]
    pub fn state_equal(&self, other: &Self) -> bool {
        self == other
    }

    /// Numeric-only equality for language rules that explicitly ignore provenance.
    ///
    /// This method does not define [`Eq`] or temporal cache identity.
    pub fn numeric_values_equal(&self, other: &Self) -> bool {
        self.width == other.width
            && self
                .iter()
                .zip(other.iter())
                .all(|((_, left), (_, right))| left.val == right.val)
    }

    /// Deterministic exact-state fingerprint for hash tables and cycle buckets.
    ///
    /// Collisions are possible for every finite hash. Callers must retain and
    /// compare the exact snapshot (or [`Self::sparse_state`]) on hash matches.
    #[inline]
    pub fn state_hash(&self) -> u64 {
        self.cached_hash
    }

    /// Merge provenance from every non-default cell.
    pub fn collect_provenance(&self) -> Provenance {
        self.iter_sparse()
            .fold(Provenance::none(), |acc, (_, value)| acc.merge(&value.prov))
    }

    fn validate_address(
        &self,
        address: u64,
        policy: BoundsPolicy,
        operation: MemoryOperation,
        location: SourceLocation,
    ) -> OuroResult<Address> {
        let width = self.width as u64;
        if address < width {
            return Ok(address);
        }
        match policy {
            BoundsPolicy::Wrap => Ok(address % width),
            BoundsPolicy::Clamp => Ok(width - 1),
            BoundsPolicy::Error => Err(OuroError::MemoryBoundsViolation {
                address,
                max_address: width - 1,
                operation,
                location,
            }),
        }
    }

    #[inline]
    fn cell_at(&self, index: usize) -> &Value {
        let page = index / PAGE_CELLS;
        let offset = index % PAGE_CELLS;
        &self.pages[page].cells[offset]
    }

    fn write_validated(&mut self, index: usize, value: Value) {
        let page_index = index / PAGE_CELLS;
        let offset = index % PAGE_CELLS;
        let old_value = &self.pages[page_index].cells[offset];
        if *old_value == value {
            return;
        }

        let address = index as Address;
        let old_contribution = hash_cell(address, old_value);
        let new_contribution = hash_cell(address, &value);

        let pages = Arc::make_mut(&mut self.pages);
        let page = Arc::make_mut(&mut pages[page_index]);
        page.cells[offset] = value;
        page.cached_hash ^= old_contribution ^ new_contribution;
        self.cached_hash ^= old_contribution ^ new_contribution;

        debug_assert_eq!(page.cached_hash, recompute_page_hash(page, page_index));
        debug_assert_eq!(self.cached_hash, self.recompute_hash());
    }

    fn recompute_hash(&self) -> u64 {
        self.pages
            .iter()
            .fold(hash_width(self.width), |hash, page| hash ^ page.cached_hash)
    }
}

impl PartialEq for PagedMemory {
    fn eq(&self, other: &Self) -> bool {
        if self.width != other.width || self.cached_hash != other.cached_hash {
            return false;
        }
        if Arc::ptr_eq(&self.pages, &other.pages) {
            return true;
        }
        self.pages
            .iter()
            .zip(other.pages.iter())
            .all(|(left, right)| {
                Arc::ptr_eq(left, right)
                    || (left.cached_hash == right.cached_hash && left.cells == right.cells)
            })
    }
}

impl Eq for PagedMemory {}

impl PartialOrd for PagedMemory {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PagedMemory {
    fn cmp(&self, other: &Self) -> Ordering {
        let common_width = self.width.min(other.width);
        for index in 0..common_width {
            let ordering = compare_value_exact(self.cell_at(index), other.cell_at(index));
            if ordering != Ordering::Equal {
                return ordering;
            }
        }
        self.width.cmp(&other.width)
    }
}

impl Hash for PagedMemory {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // HashMap still calls exact Eq inside a colliding bucket.
        state.write_usize(self.width);
        state.write_u64(self.cached_hash);
    }
}

impl fmt::Debug for PagedMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sparse: Vec<_> = self.iter_sparse().take(16).collect();
        let total = self.iter_sparse().count();
        f.debug_struct("PagedMemory")
            .field("width", &self.width)
            .field("pages", &self.page_count())
            .field("state_hash", &format_args!("{:#018x}", self.cached_hash))
            .field("sparse_prefix", &sparse)
            .field("sparse_cells", &total)
            .finish()
    }
}

fn compare_value_exact(left: &Value, right: &Value) -> Ordering {
    left.val
        .cmp(&right.val)
        .then_with(|| left.prov.saturated.cmp(&right.prov.saturated))
        .then_with(|| match (&left.prov.deps, &right.prov.deps) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Less,
            (Some(_), None) => Ordering::Greater,
            (Some(left), Some(right)) => left.iter().cmp(right.iter()),
        })
}

fn hash_width(width: usize) -> u64 {
    avalanche((width as u64) ^ 0x89e1_82d6_4ad8_7f53)
}

fn hash_cell(address: Address, value: &Value) -> u64 {
    if *value == Value::ZERO {
        return 0;
    }

    let mut hash = avalanche(address ^ 0xd6e8_feb8_6659_fd93);
    hash = hash_word(hash, value.val);
    hash = hash_word(hash, u64::from(value.prov.saturated));
    match &value.prov.deps {
        None => hash_word(hash, 0x243f_6a88_85a3_08d3),
        Some(dependencies) => {
            hash = hash_word(hash, 0x1319_8a2e_0370_7344);
            hash = hash_word(hash, dependencies.len() as u64);
            for dependency in dependencies.iter() {
                hash = hash_word(hash, *dependency);
            }
            hash
        }
    }
}

fn hash_word(hash: u64, word: u64) -> u64 {
    avalanche(hash ^ word.wrapping_mul(0x9e37_79b9_7f4a_7c15))
}

fn avalanche(mut value: u64) -> u64 {
    value ^= value >> 33;
    value = value.wrapping_mul(0xff51_afd7_ed55_8ccd);
    value ^= value >> 33;
    value = value.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    value ^ (value >> 33)
}

fn recompute_page_hash(page: &Page, page_index: usize) -> u64 {
    page.cells
        .iter()
        .enumerate()
        .fold(0, |hash, (offset, value)| {
            let address = (page_index * PAGE_CELLS + offset) as Address;
            hash ^ hash_cell(address, value)
        })
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashSet};

    use super::*;

    fn provenance(addresses: &[Address]) -> Provenance {
        Provenance::from_set(addresses.iter().copied().collect::<BTreeSet<_>>())
    }

    #[test]
    fn default_65_536_cell_snapshot_is_sparse_and_addressable() {
        let mut memory = PagedMemory::new();
        assert_eq!(memory.len(), 65_536);
        assert_eq!(memory.page_count(), 256);
        assert_eq!(memory.iter().len(), 65_536);
        assert!(memory.iter_sparse().next().is_none());

        for (address, number) in [(0, 11), (255, 22), (256, 33), (65_535, 44)] {
            memory.write(address, Value::new(number)).unwrap();
        }
        assert_eq!(
            memory.numeric_sparse_state(),
            vec![(0, 11), (255, 22), (256, 33), (65_535, 44)]
        );
    }

    #[test]
    fn million_cell_snapshot_keeps_zero_pages_shared_and_sparse_ordered() {
        let mut memory = PagedMemory::with_size(1_000_000).unwrap();
        assert_eq!(memory.page_count(), 3_907);
        let shared_zero = memory.pages[0].clone();
        assert!(memory
            .pages
            .iter()
            .all(|page| Arc::ptr_eq(page, &shared_zero)));

        for (address, number) in [(999_999, 3), (500_000, 2), (17, 1)] {
            memory.write(address, Value::new(number)).unwrap();
        }
        assert_eq!(
            memory.numeric_sparse_state(),
            vec![(17, 1), (500_000, 2), (999_999, 3)]
        );

        // Only the three written pages detach from the common zero page.
        assert_eq!(
            memory
                .pages
                .iter()
                .filter(|page| !Arc::ptr_eq(page, &shared_zero))
                .count(),
            3
        );
    }

    #[test]
    fn clone_is_constant_time_and_write_detaches_only_one_page() {
        let mut original = PagedMemory::with_size(PAGE_CELLS * 4).unwrap();
        original.write(1, Value::new(10)).unwrap();
        let mut clone = original.clone();
        assert!(Arc::ptr_eq(&original.pages, &clone.pages));

        clone
            .write((PAGE_CELLS + 7) as Address, Value::new(20))
            .unwrap();
        assert!(!Arc::ptr_eq(&original.pages, &clone.pages));
        assert!(Arc::ptr_eq(&original.pages[0], &clone.pages[0]));
        assert!(!Arc::ptr_eq(&original.pages[1], &clone.pages[1]));
        assert!(Arc::ptr_eq(&original.pages[2], &clone.pages[2]));
        assert!(Arc::ptr_eq(&original.pages[3], &clone.pages[3]));
        assert_eq!(
            original.get((PAGE_CELLS + 7) as Address),
            Some(&Value::ZERO)
        );
        assert_eq!(clone.get((PAGE_CELLS + 7) as Address).unwrap().val, 20);
    }

    #[test]
    fn exact_equality_sparse_state_and_order_preserve_provenance() {
        let mut pure = PagedMemory::with_size(8).unwrap();
        let mut temporal = pure.clone();
        pure.write(3, Value::new(0)).unwrap();
        temporal
            .write(3, Value::with_provenance(0, provenance(&[7, 11])))
            .unwrap();

        assert_ne!(pure, temporal);
        assert!(!pure.state_equal(&temporal));
        assert!(pure.numeric_values_equal(&temporal));
        assert!(temporal.iter_nonzero().next().is_none());
        assert_eq!(
            temporal.iter_sparse().map(|(a, _)| a).collect::<Vec<_>>(),
            vec![3]
        );
        assert_ne!(pure.cmp(&temporal), Ordering::Equal);
        assert_eq!(pure.diff(&temporal), vec![3]);
    }

    #[test]
    fn state_hash_collision_still_falls_back_to_exact_cells() {
        let mut left = PagedMemory::with_size(8).unwrap();
        let mut right = PagedMemory::with_size(8).unwrap();
        left.write(1, Value::new(41)).unwrap();
        right.write(1, Value::new(42)).unwrap();

        // Simulate an adversarial collision at both accelerator layers.
        right.cached_hash = left.cached_hash;
        let mut colliding_page = right.pages[0].as_ref().clone();
        colliding_page.cached_hash = left.pages[0].cached_hash;
        Arc::make_mut(&mut right.pages)[0] = Arc::new(colliding_page);
        assert_ne!(left, right);

        // Hash-table identity remains collision safe because Eq compares cells.
        let mut states = HashSet::new();
        states.insert(left);
        states.insert(right);
        assert_eq!(states.len(), 2);
    }

    #[test]
    fn exact_order_handles_adversarial_provenance_representations() {
        let absent = Provenance {
            deps: None,
            saturated: false,
        };
        let explicit_empty = Provenance {
            deps: Some(Arc::new(BTreeSet::new())),
            saturated: false,
        };
        let saturated_with_deps = Provenance {
            deps: Some(Arc::new(BTreeSet::from([9]))),
            saturated: true,
        };

        let mut memories = Vec::new();
        for prov in [saturated_with_deps, explicit_empty, absent] {
            let mut memory = PagedMemory::with_size(1).unwrap();
            memory.write(0, Value::with_provenance(5, prov)).unwrap();
            memories.push(memory);
        }
        memories.sort();
        assert!(memories.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(memories.windows(2).all(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn strict_wrap_and_clamp_access_are_checked_against_configured_width() {
        let mut memory = PagedMemory::with_size(10).unwrap();
        let location = SourceLocation::default();
        assert!(memory
            .write_checked(10, Value::new(1), BoundsPolicy::Error, location.clone())
            .is_err());

        memory
            .write_checked(12, Value::new(2), BoundsPolicy::Wrap, location.clone())
            .unwrap();
        memory
            .write_checked(99, Value::new(3), BoundsPolicy::Clamp, location.clone())
            .unwrap();
        assert_eq!(
            memory
                .read_checked(2, BoundsPolicy::Error, location.clone())
                .unwrap()
                .val,
            2
        );
        assert_eq!(
            memory
                .read_checked(99, BoundsPolicy::Clamp, location.clone())
                .unwrap()
                .val,
            3
        );
        assert_eq!(
            memory
                .read_checked(12, BoundsPolicy::Wrap, location)
                .unwrap()
                .val,
            2
        );
    }

    #[test]
    fn configured_limits_reject_invalid_or_excessive_tables() {
        assert_eq!(PagedMemory::with_size(0), Err(PagedMemoryError::ZeroWidth));
        assert_eq!(
            PagedMemory::with_limits(
                257,
                PagedMemoryLimits {
                    max_cells: 256,
                    max_pages: 2,
                },
            ),
            Err(PagedMemoryError::CellLimitExceeded {
                requested: 257,
                limit: 256,
            })
        );
        assert_eq!(
            PagedMemory::with_limits(
                257,
                PagedMemoryLimits {
                    max_cells: 1_000,
                    max_pages: 1,
                },
            ),
            Err(PagedMemoryError::PageLimitExceeded {
                requested: 2,
                limit: 1,
            })
        );
    }

    #[test]
    fn serialization_and_diff_iteration_are_deterministic() {
        let mut first = PagedMemory::with_size(600).unwrap();
        let mut second = first.clone();
        for (address, value) in [(511, 1), (0, 2), (256, 3), (255, 4)] {
            first.write(address, Value::new(value)).unwrap();
        }
        for (address, value) in [(255, 9), (599, 8)] {
            second.write(address, Value::new(value)).unwrap();
        }

        assert_eq!(
            first
                .iter_sparse()
                .map(|(address, _)| address)
                .collect::<Vec<_>>(),
            vec![0, 255, 256, 511]
        );
        assert_eq!(first.diff(&second), vec![0, 255, 256, 511, 599]);
        assert_eq!(
            first.diff_limited(&second, 4),
            Err(PagedMemoryError::DifferenceLimitExceeded { limit: 4 })
        );
        assert_eq!(first.iter_serialization().next().unwrap().0, 0);
        assert_eq!(first.iter_serialization().last().unwrap().0, 599);
    }

    #[test]
    fn incremental_hash_matches_full_recomputation_with_provenance() {
        let mut memory = PagedMemory::with_size(700).unwrap();
        for address in [0, 255, 256, 699] {
            memory
                .write(
                    address,
                    Value::with_provenance(address * 3, provenance(&[address, address + 1])),
                )
                .unwrap();
            assert_eq!(memory.state_hash(), memory.recompute_hash());
        }
        memory.write(256, Value::ZERO).unwrap();
        assert_eq!(memory.state_hash(), memory.recompute_hash());
    }
}
