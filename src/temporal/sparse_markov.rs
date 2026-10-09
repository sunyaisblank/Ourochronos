//! Exact sparse finite stochastic stationary families with bounded resources.
//!
//! Rows and extremal distributions store only nonzero entries. Every closed
//! strongly connected class is included, independent of any initial state;
//! their invariant distributions span the full stationary convex hull.
//! Arithmetic uses arbitrary precision, never floating point or approximation.
//!
//! Limits cover input counts, integer bits, live elimination entries and
//! charged operations. They are not wall-clock or allocator guarantees.
//! Arithmetic preflights conservative *unreduced* intermediate bit bounds;
//! an operation can therefore be rejected even when cancellation would make
//! its reduced result small. All such rejection is explicit and typed.

pub use num_rational::BigRational as ExactRational;
use num_traits::{One, Signed, Zero};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

pub const MAX_SPARSE_MARKOV_STATES: usize = 16_384;
pub const MAX_SPARSE_MARKOV_RAW_EDGES: usize = 262_144;
pub const MAX_SPARSE_MARKOV_INTEGER_BITS: u64 = 4096;
pub const MAX_SPARSE_MARKOV_MATRIX_ENTRIES: usize = 1_000_000;
pub const MAX_SPARSE_MARKOV_OPERATIONS: u64 = 25_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseMarkovLimits {
    pub max_states: usize,
    /// Counts all submitted entries, including duplicate targets and zeros.
    pub max_raw_edges: usize,
    /// Applies to each signed numerator magnitude and positive denominator,
    /// including conservative unreduced intermediate bounds.
    pub max_integer_bits: u64,
    pub max_matrix_entries: usize,
    /// One unit per explicitly charged graph scan, sparse mutation, rational
    /// normalization, arithmetic operation or comparison. Integer internals
    /// and collection sorting are bounded by the other caps, not timed here.
    pub max_operations: u64,
}

impl Default for SparseMarkovLimits {
    fn default() -> Self {
        Self {
            max_states: MAX_SPARSE_MARKOV_STATES,
            max_raw_edges: MAX_SPARSE_MARKOV_RAW_EDGES,
            max_integer_bits: MAX_SPARSE_MARKOV_INTEGER_BITS,
            max_matrix_entries: MAX_SPARSE_MARKOV_MATRIX_ENTRIES,
            max_operations: MAX_SPARSE_MARKOV_OPERATIONS,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseMarkovResource {
    States,
    RawEdges,
    IntegerBits,
    MatrixEntries,
    Operations,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseMarkovError {
    EmptyChain,
    InvalidLimit {
        resource: SparseMarkovResource,
        requested: u64,
        maximum: u64,
    },
    ResourceLimit {
        resource: SparseMarkovResource,
        limit: u64,
        required: u64,
    },
    InvalidTarget {
        row: usize,
        target: usize,
        states: usize,
    },
    ZeroDenominator {
        context: &'static str,
        index: usize,
    },
    NegativeProbability {
        row: usize,
        target: usize,
    },
    InvalidRowSum {
        row: usize,
        sum: ExactRational,
    },
    InvalidReadoutLength {
        expected: usize,
        actual: usize,
    },
    InvalidThresholds,
    InvalidDistribution {
        reason: &'static str,
    },
    InternalInvariant {
        reason: &'static str,
    },
}

impl fmt::Display for SparseMarkovError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyChain => f.write_str("sparse Markov chain must contain a state"),
            Self::InvalidLimit {
                resource,
                requested,
                maximum,
            } => write!(
                f,
                "sparse Markov {resource:?} limit {requested} must be in 1..={maximum}"
            ),
            Self::ResourceLimit {
                resource,
                limit,
                required,
            } => write!(
                f,
                "sparse Markov {resource:?} requires {required}, exceeding limit {limit}"
            ),
            Self::InvalidTarget {
                row,
                target,
                states,
            } => write!(
                f,
                "sparse Markov row {row} target {target} is outside 0..{states}"
            ),
            Self::ZeroDenominator { context, index } => {
                write!(f, "sparse Markov {context} {index} has a zero denominator")
            }
            Self::NegativeProbability { row, target } => write!(
                f,
                "sparse Markov row {row} target {target} has a negative probability"
            ),
            Self::InvalidRowSum { row, sum } => write!(
                f,
                "sparse Markov row {row} sums to {sum}, expected exactly 1"
            ),
            Self::InvalidReadoutLength { expected, actual } => write!(
                f,
                "sparse Markov readout has length {actual}, expected {expected}"
            ),
            Self::InvalidThresholds => {
                f.write_str("sparse Markov thresholds must satisfy 0 <= reject <= accept <= 1")
            }
            Self::InvalidDistribution { reason } | Self::InternalInvariant { reason } => {
                f.write_str(reason)
            }
        }
    }
}
impl Error for SparseMarkovError {}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SparseMarkovStats {
    pub charged_operations: u64,
    /// Largest admitted input/result or conservative intermediate bit bound.
    pub peak_integer_bits: u64,
    pub peak_matrix_entries: usize,
}

/// Exact identities for a candidate: sum(weights) and nonzero entries of
/// weights*P - weights. A normalized stationary distribution has sum 1 and
/// an empty residual vector. Entries are sorted by state identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactStationaryResidual {
    pub normalization: ExactRational,
    pub residual: Vec<(usize, ExactRational)>,
}

impl ExactStationaryResidual {
    pub fn is_stationary(&self) -> bool {
        self.normalization.is_one() && self.residual.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseStationaryDistribution {
    pub recurrent_class: Vec<usize>,
    pub weights: Vec<(usize, ExactRational)>,
    pub certificate: ExactStationaryResidual,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseStationaryFamily {
    pub states: usize,
    /// Deterministic lexicographic class order; all weights are positive and
    /// sum exactly to one. No transient state has stationary weight.
    pub extremal: Vec<SparseStationaryDistribution>,
    /// Budget usage through family construction. During analyze(), this also
    /// includes threshold validation performed before constructing the family.
    pub stats: SparseMarkovStats,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseStationaryDecision {
    Accept,
    Reject,
    Ambiguous,
}

/// Names an extremal distribution in the returned family, allowing its exact
/// probability and invariant weights to be inspected as a readout witness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassReadoutWitness {
    pub class_index: usize,
    pub probability: ExactRational,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseStationaryAnalysis {
    pub family: SparseStationaryFamily,
    pub acceptance_probabilities: Vec<ExactRational>,
    pub minimum: ClassReadoutWitness,
    pub maximum: ClassReadoutWitness,
    /// One extremal inside the open promise gap, when one exists. Even when
    /// no extremal is in the gap, disagreeing classes make readout ambiguous.
    pub gap_witness: Option<ClassReadoutWitness>,
    pub decision: SparseStationaryDecision,
    pub stats: SparseMarkovStats,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseMarkovChain {
    rows: Vec<Vec<(usize, ExactRational)>>,
    edges: usize,
    limits: SparseMarkovLimits,
    admission_stats: SparseMarkovStats,
}

impl SparseMarkovChain {
    /// Validate supplied rows before constructing canonical sparse storage.
    /// Caller-owned vectors already exist; their state/raw-entry counts are
    /// checked before any coalescing, SCC or elimination allocation here.
    /// Even BigRational::new_raw inputs are checked and normalized safely.
    pub fn new(
        rows: Vec<Vec<(usize, ExactRational)>>,
        limits: SparseMarkovLimits,
    ) -> Result<Self, SparseMarkovError> {
        limits.validate()?;
        let states = rows.len();
        if states == 0 {
            return Err(SparseMarkovError::EmptyChain);
        }
        check_limit(
            SparseMarkovResource::States,
            states as u64,
            limits.max_states as u64,
        )?;
        let raw_edges = rows
            .iter()
            .try_fold(0_u64, |count, row| count.checked_add(row.len() as u64))
            .ok_or(SparseMarkovError::ResourceLimit {
                resource: SparseMarkovResource::RawEdges,
                limit: limits.max_raw_edges as u64,
                required: u64::MAX,
            })?;
        check_limit(
            SparseMarkovResource::RawEdges,
            raw_edges,
            limits.max_raw_edges as u64,
        )?;
        // Check every input before allocating canonical rows, including zeros
        // and duplicate entries which will later disappear.
        let mut budget = Budget::new(limits);
        for (row, entries) in rows.iter().enumerate() {
            for (target, probability) in entries {
                budget.charge(1)?;
                if *target >= states {
                    return Err(SparseMarkovError::InvalidTarget {
                        row,
                        target: *target,
                        states,
                    });
                }
                budget.check_rational(probability)?;
                if probability.denom().is_zero() {
                    return Err(SparseMarkovError::ZeroDenominator {
                        context: "row",
                        index: row,
                    });
                }
            }
        }
        let mut canonical = Vec::with_capacity(states);
        let mut edges = 0;
        for (row, entries) in rows.into_iter().enumerate() {
            let mut merged = BTreeMap::<usize, ExactRational>::new();
            for (target, probability) in entries {
                let probability = budget.normalize(probability, "row", row)?;
                if probability.is_negative() {
                    return Err(SparseMarkovError::NegativeProbability { row, target });
                }
                if !probability.is_zero() {
                    let prior = merged.remove(&target).unwrap_or_else(ExactRational::zero);
                    merged.insert(target, budget.add(&prior, &probability)?);
                }
            }
            let mut sum = ExactRational::zero();
            for probability in merged.values() {
                sum = budget.add(&sum, probability)?;
            }
            if !sum.is_one() {
                return Err(SparseMarkovError::InvalidRowSum { row, sum });
            }
            edges += merged.len();
            canonical.push(merged.into_iter().collect());
        }
        Ok(Self {
            rows: canonical,
            edges,
            limits,
            admission_stats: budget.stats,
        })
    }

    pub fn states(&self) -> usize {
        self.rows.len()
    }
    pub fn edges(&self) -> usize {
        self.edges
    }
    pub fn rows(&self) -> &[Vec<(usize, ExactRational)>] {
        &self.rows
    }
    pub fn limits(&self) -> SparseMarkovLimits {
        self.limits
    }
    pub fn admission_stats(&self) -> SparseMarkovStats {
        self.admission_stats
    }

    /// Solve each closed SCC exactly under one fresh operation budget.
    pub fn stationary_family(&self) -> Result<SparseStationaryFamily, SparseMarkovError> {
        self.solve(&mut Budget::new(self.limits))
    }

    /// Independently recompute exact residual identities for a sparse candidate.
    /// Candidates must use strictly increasing state IDs and nonnegative weights;
    /// the returned identities expose normalization or stationarity failures.
    pub fn residual(
        &self,
        weights: &[(usize, ExactRational)],
    ) -> Result<ExactStationaryResidual, SparseMarkovError> {
        self.residual_with_budget(weights, &mut Budget::new(self.limits))
    }

    /// Classify every stationary distribution, using the convex-hull extrema.
    /// Inclusive thresholds match the existing backend. If thresholds coincide
    /// and all classes equal that threshold, Accept takes precedence.
    pub fn analyze(
        &self,
        accepting: &[bool],
        accept: ExactRational,
        reject: ExactRational,
    ) -> Result<SparseStationaryAnalysis, SparseMarkovError> {
        if accepting.len() != self.states() {
            return Err(SparseMarkovError::InvalidReadoutLength {
                expected: self.states(),
                actual: accepting.len(),
            });
        }
        let mut budget = Budget::new(self.limits);
        let accept = budget.normalize(accept, "accept threshold", 0)?;
        let reject = budget.normalize(reject, "reject threshold", 0)?;
        if reject.is_negative()
            || budget.cmp(&reject, &accept)? == Ordering::Greater
            || budget.cmp(&accept, &ExactRational::one())? == Ordering::Greater
        {
            return Err(SparseMarkovError::InvalidThresholds);
        }
        let family = self.solve(&mut budget)?;
        let mut probabilities = Vec::with_capacity(family.extremal.len());
        let mut minimum = None::<ClassReadoutWitness>;
        let mut maximum = None::<ClassReadoutWitness>;
        let mut gap_witness = None;
        for (index, distribution) in family.extremal.iter().enumerate() {
            let mut probability = ExactRational::zero();
            for (state, weight) in &distribution.weights {
                budget.charge(1)?;
                if accepting[*state] {
                    probability = budget.add(&probability, weight)?;
                }
            }
            let witness = ClassReadoutWitness {
                class_index: index,
                probability: probability.clone(),
            };
            if minimum.is_none() {
                minimum = Some(witness.clone());
                maximum = Some(witness.clone());
            } else {
                if budget.cmp(
                    &probability,
                    &minimum.as_ref().expect("first class").probability,
                )? == Ordering::Less
                {
                    minimum = Some(witness.clone());
                }
                if budget.cmp(
                    &probability,
                    &maximum.as_ref().expect("first class").probability,
                )? == Ordering::Greater
                {
                    maximum = Some(witness.clone());
                }
            }
            if gap_witness.is_none()
                && budget.cmp(&probability, &reject)? == Ordering::Greater
                && budget.cmp(&probability, &accept)? == Ordering::Less
            {
                gap_witness = Some(witness);
            }
            probabilities.push(probability);
        }
        let minimum = minimum.ok_or(SparseMarkovError::InternalInvariant {
            reason: "finite stochastic chain has no closed recurrent class",
        })?;
        let maximum = maximum.expect("minimum and maximum initialized together");
        let decision = if budget.cmp(&minimum.probability, &accept)? != Ordering::Less {
            SparseStationaryDecision::Accept
        } else if budget.cmp(&maximum.probability, &reject)? != Ordering::Greater {
            SparseStationaryDecision::Reject
        } else {
            SparseStationaryDecision::Ambiguous
        };
        Ok(SparseStationaryAnalysis {
            family,
            acceptance_probabilities: probabilities,
            minimum,
            maximum,
            gap_witness,
            decision,
            stats: budget.stats,
        })
    }

    fn solve(&self, budget: &mut Budget) -> Result<SparseStationaryFamily, SparseMarkovError> {
        let classes = self.closed_classes(budget)?;
        let mut extremal = Vec::with_capacity(classes.len());
        for class in classes {
            let local = self.solve_class(&class, budget)?;
            let weights: Vec<_> = class.iter().copied().zip(local).collect();
            if weights
                .iter()
                .any(|(_, value)| value.is_negative() || value.is_zero())
            {
                return Err(SparseMarkovError::InternalInvariant {
                    reason: "closed irreducible class did not produce positive invariant weights",
                });
            }
            let certificate = self.residual_with_budget(&weights, budget)?;
            if !certificate.is_stationary() {
                return Err(SparseMarkovError::InternalInvariant {
                    reason: "exact stationary solve failed normalization or residual validation",
                });
            }
            extremal.push(SparseStationaryDistribution {
                recurrent_class: class,
                weights,
                certificate,
            });
        }
        Ok(SparseStationaryFamily {
            states: self.states(),
            extremal,
            stats: budget.stats,
        })
    }

    /// Iterative Kosaraju: no guest-sized Rust recursion or dense reachability.
    fn closed_classes(&self, budget: &mut Budget) -> Result<Vec<Vec<usize>>, SparseMarkovError> {
        let states = self.states();
        let mut reverse = vec![Vec::new(); states];
        for (source, row) in self.rows.iter().enumerate() {
            budget.charge(1)?;
            for (target, _) in row {
                budget.charge(1)?;
                reverse[*target].push(source);
            }
        }
        let mut seen = vec![false; states];
        let mut finish = Vec::with_capacity(states);
        let mut stack = Vec::<(usize, usize)>::new();
        for start in 0..states {
            budget.charge(1)?;
            if seen[start] {
                continue;
            }
            seen[start] = true;
            stack.push((start, 0));
            while let Some((state, edge)) = stack.last_mut() {
                budget.charge(1)?;
                if let Some((target, _)) = self.rows[*state].get(*edge) {
                    *edge += 1;
                    if !seen[*target] {
                        seen[*target] = true;
                        stack.push((*target, 0));
                    }
                } else {
                    finish.push(*state);
                    stack.pop();
                }
            }
        }
        let mut component = vec![usize::MAX; states];
        let mut classes = Vec::<Vec<usize>>::new();
        let mut pending = Vec::new();
        for start in finish.into_iter().rev() {
            budget.charge(1)?;
            if component[start] != usize::MAX {
                continue;
            }
            let id = classes.len();
            component[start] = id;
            pending.push(start);
            let mut class = Vec::new();
            while let Some(state) = pending.pop() {
                budget.charge(1)?;
                class.push(state);
                for &source in &reverse[state] {
                    budget.charge(1)?;
                    if component[source] == usize::MAX {
                        component[source] = id;
                        pending.push(source);
                    }
                }
            }
            class.sort_unstable();
            classes.push(class);
        }
        let mut closed = vec![true; classes.len()];
        for (source, row) in self.rows.iter().enumerate() {
            budget.charge(1)?;
            for (target, _) in row {
                budget.charge(1)?;
                if component[source] != component[*target] {
                    closed[component[source]] = false;
                }
            }
        }
        let mut result: Vec<_> = classes
            .into_iter()
            .zip(closed)
            .filter_map(|(class, closed)| closed.then_some(class))
            .collect();
        result.sort();
        Ok(result)
    }

    fn solve_class(
        &self,
        class: &[usize],
        budget: &mut Budget,
    ) -> Result<Vec<ExactRational>, SparseMarkovError> {
        let n = class.len();
        if n == 1 {
            budget.charge(1)?;
            return Ok(vec![ExactRational::one()]);
        }
        let mut matrix = vec![BTreeMap::<usize, ExactRational>::new(); n];
        let mut entries = 0;
        // First n-1 stationarity equations; normalization replaces the
        // dependent last equation. Zero entries never occupy the matrix.
        for (source_index, &source) in class.iter().enumerate() {
            for (target, probability) in &self.rows[source] {
                budget.charge(1)?;
                let equation = class.binary_search(target).map_err(|_| {
                    SparseMarkovError::InternalInvariant {
                        reason: "closed SCC has an outgoing edge",
                    }
                })?;
                if equation < n - 1 {
                    set_entry(
                        &mut matrix[equation],
                        source_index,
                        probability.clone(),
                        &mut entries,
                        budget,
                    )?;
                }
            }
            if source_index < n - 1 {
                let old = matrix[source_index]
                    .get(&source_index)
                    .cloned()
                    .unwrap_or_else(ExactRational::zero);
                let diagonal = budget.sub(&old, &ExactRational::one())?;
                set_entry(
                    &mut matrix[source_index],
                    source_index,
                    diagonal,
                    &mut entries,
                    budget,
                )?;
            }
            set_entry(
                &mut matrix[n - 1],
                source_index,
                ExactRational::one(),
                &mut entries,
                budget,
            )?;
        }
        set_entry(
            &mut matrix[n - 1],
            n,
            ExactRational::one(),
            &mut entries,
            budget,
        )?;
        for column in 0..n {
            let mut found = None;
            for (row, coefficients) in matrix.iter().enumerate().skip(column) {
                budget.charge(1)?;
                if coefficients.contains_key(&column) {
                    found = Some(row);
                    break;
                }
            }
            let pivot_row = found.ok_or(SparseMarkovError::InternalInvariant {
                reason: "closed-class stationary system is rank deficient",
            })?;
            matrix.swap(column, pivot_row);
            let pivot = matrix[column].get(&column).expect("nonzero pivot").clone();
            // Temporary row copies have at most n+1 entries. The charged work
            // and stored matrix cap bound growth; no dense n*n copy is made.
            let pivot_entries: Vec<_> = matrix[column]
                .iter()
                .map(|(&index, value)| (index, value.clone()))
                .collect();
            for (index, value) in pivot_entries {
                let normalized = budget.div(&value, &pivot)?;
                set_entry(&mut matrix[column], index, normalized, &mut entries, budget)?;
            }
            let pivot_entries: Vec<_> = matrix[column]
                .iter()
                .map(|(&index, value)| (index, value.clone()))
                .collect();
            for (row_index, row) in matrix.iter_mut().enumerate() {
                budget.charge(1)?;
                if row_index == column {
                    continue;
                }
                let Some(factor) = row.get(&column).cloned() else {
                    continue;
                };
                for (index, value) in &pivot_entries {
                    let product = budget.mul(&factor, value)?;
                    let old = row.get(index).cloned().unwrap_or_else(ExactRational::zero);
                    let difference = budget.sub(&old, &product)?;
                    set_entry(row, *index, difference, &mut entries, budget)?;
                }
            }
        }
        Ok(matrix
            .into_iter()
            .map(|mut row| row.remove(&n).unwrap_or_else(ExactRational::zero))
            .collect())
    }

    fn residual_with_budget(
        &self,
        weights: &[(usize, ExactRational)],
        budget: &mut Budget,
    ) -> Result<ExactStationaryResidual, SparseMarkovError> {
        check_limit(
            SparseMarkovResource::States,
            weights.len() as u64,
            self.states() as u64,
        )?;
        let mut prior = None;
        let mut normalized = Vec::with_capacity(weights.len());
        for (state, weight) in weights {
            budget.charge(1)?;
            if *state >= self.states() || prior.is_some_and(|prior| prior >= *state) {
                return Err(SparseMarkovError::InvalidDistribution {
                    reason: "distribution state IDs must be in range and strictly increasing",
                });
            }
            let weight = budget.normalize(weight.clone(), "distribution weight", *state)?;
            if weight.is_negative() {
                return Err(SparseMarkovError::InvalidDistribution {
                    reason: "distribution has negative weight",
                });
            }
            prior = Some(*state);
            normalized.push((*state, weight));
        }
        let mut sum = ExactRational::zero();
        let mut residual = BTreeMap::<usize, ExactRational>::new();
        for (source, weight) in &normalized {
            sum = budget.add(&sum, weight)?;
            for (target, probability) in &self.rows[*source] {
                budget.charge(1)?;
                let product = budget.mul(weight, probability)?;
                let old = residual.remove(target).unwrap_or_else(ExactRational::zero);
                let value = budget.add(&old, &product)?;
                if !value.is_zero() {
                    residual.insert(*target, value);
                }
            }
        }
        for (state, weight) in normalized {
            let old = residual.remove(&state).unwrap_or_else(ExactRational::zero);
            let value = budget.sub(&old, &weight)?;
            if !value.is_zero() {
                residual.insert(state, value);
            }
        }
        Ok(ExactStationaryResidual {
            normalization: sum,
            residual: residual.into_iter().collect(),
        })
    }
}

impl SparseMarkovLimits {
    fn validate(self) -> Result<(), SparseMarkovError> {
        for (resource, requested, maximum) in [
            (
                SparseMarkovResource::States,
                self.max_states as u64,
                MAX_SPARSE_MARKOV_STATES as u64,
            ),
            (
                SparseMarkovResource::RawEdges,
                self.max_raw_edges as u64,
                MAX_SPARSE_MARKOV_RAW_EDGES as u64,
            ),
            (
                SparseMarkovResource::IntegerBits,
                self.max_integer_bits,
                MAX_SPARSE_MARKOV_INTEGER_BITS,
            ),
            (
                SparseMarkovResource::MatrixEntries,
                self.max_matrix_entries as u64,
                MAX_SPARSE_MARKOV_MATRIX_ENTRIES as u64,
            ),
            (
                SparseMarkovResource::Operations,
                self.max_operations,
                MAX_SPARSE_MARKOV_OPERATIONS,
            ),
        ] {
            if requested == 0 || requested > maximum {
                return Err(SparseMarkovError::InvalidLimit {
                    resource,
                    requested,
                    maximum,
                });
            }
        }
        Ok(())
    }
}

fn check_limit(
    resource: SparseMarkovResource,
    required: u64,
    limit: u64,
) -> Result<(), SparseMarkovError> {
    if required > limit {
        Err(SparseMarkovError::ResourceLimit {
            resource,
            limit,
            required,
        })
    } else {
        Ok(())
    }
}

fn set_entry(
    row: &mut BTreeMap<usize, ExactRational>,
    column: usize,
    value: ExactRational,
    entries: &mut usize,
    budget: &mut Budget,
) -> Result<(), SparseMarkovError> {
    budget.charge(1)?;
    if value.is_zero() {
        if row.remove(&column).is_some() {
            *entries -= 1;
        }
    } else {
        if !row.contains_key(&column) {
            check_limit(
                SparseMarkovResource::MatrixEntries,
                *entries as u64 + 1,
                budget.limits.max_matrix_entries as u64,
            )?;
            *entries += 1;
            budget.stats.peak_matrix_entries = budget.stats.peak_matrix_entries.max(*entries);
        }
        row.insert(column, value);
    }
    Ok(())
}

struct Budget {
    limits: SparseMarkovLimits,
    stats: SparseMarkovStats,
}
impl Budget {
    fn new(limits: SparseMarkovLimits) -> Self {
        Self {
            limits,
            stats: SparseMarkovStats::default(),
        }
    }
    fn charge(&mut self, amount: u64) -> Result<(), SparseMarkovError> {
        let required = self.stats.charged_operations.saturating_add(amount);
        check_limit(
            SparseMarkovResource::Operations,
            required,
            self.limits.max_operations,
        )?;
        self.stats.charged_operations = required;
        Ok(())
    }
    fn bits(&mut self, required: u64) -> Result<(), SparseMarkovError> {
        check_limit(
            SparseMarkovResource::IntegerBits,
            required,
            self.limits.max_integer_bits,
        )?;
        self.stats.peak_integer_bits = self.stats.peak_integer_bits.max(required);
        Ok(())
    }
    fn check_rational(&mut self, value: &ExactRational) -> Result<(), SparseMarkovError> {
        self.bits(value.numer().bits().max(value.denom().bits()))
    }
    fn normalize(
        &mut self,
        value: ExactRational,
        context: &'static str,
        index: usize,
    ) -> Result<ExactRational, SparseMarkovError> {
        self.charge(1)?;
        self.check_rational(&value)?;
        if value.denom().is_zero() {
            return Err(SparseMarkovError::ZeroDenominator { context, index });
        }
        let value = ExactRational::new(value.numer().clone(), value.denom().clone());
        self.check_rational(&value)?;
        Ok(value)
    }
    fn add(
        &mut self,
        a: &ExactRational,
        b: &ExactRational,
    ) -> Result<ExactRational, SparseMarkovError> {
        self.add_sub(a, b, false)
    }
    fn sub(
        &mut self,
        a: &ExactRational,
        b: &ExactRational,
    ) -> Result<ExactRational, SparseMarkovError> {
        self.add_sub(a, b, true)
    }
    fn add_sub(
        &mut self,
        a: &ExactRational,
        b: &ExactRational,
        subtract: bool,
    ) -> Result<ExactRational, SparseMarkovError> {
        self.charge(1)?;
        if b.is_zero() {
            return Ok(a.clone());
        }
        if a.is_zero() {
            return Ok(if subtract { -b.clone() } else { b.clone() });
        }
        if subtract && same_rational(a, b) {
            return Ok(ExactRational::zero());
        }
        self.bits(
            (a.numer().bits() + b.denom().bits()).max(b.numer().bits() + a.denom().bits()) + 1,
        )?;
        self.bits(a.denom().bits() + b.denom().bits())?;
        let result = if subtract { a - b } else { a + b };
        self.check_rational(&result)?;
        Ok(result)
    }
    fn mul(
        &mut self,
        a: &ExactRational,
        b: &ExactRational,
    ) -> Result<ExactRational, SparseMarkovError> {
        self.charge(1)?;
        if a.is_zero() || b.is_zero() {
            return Ok(ExactRational::zero());
        }
        if a.is_one() {
            return Ok(b.clone());
        }
        if b.is_one() {
            return Ok(a.clone());
        }
        self.bits(a.numer().bits() + b.numer().bits())?;
        self.bits(a.denom().bits() + b.denom().bits())?;
        let result = a * b;
        self.check_rational(&result)?;
        Ok(result)
    }
    fn div(
        &mut self,
        a: &ExactRational,
        b: &ExactRational,
    ) -> Result<ExactRational, SparseMarkovError> {
        self.charge(1)?;
        if b.is_zero() {
            return Err(SparseMarkovError::InternalInvariant {
                reason: "zero stationary elimination divisor",
            });
        }
        if a.is_zero() {
            return Ok(ExactRational::zero());
        }
        if b.is_one() {
            return Ok(a.clone());
        }
        if same_rational(a, b) {
            return Ok(ExactRational::one());
        }
        self.bits(a.numer().bits() + b.denom().bits())?;
        self.bits(a.denom().bits() + b.numer().bits())?;
        let result = a / b;
        self.check_rational(&result)?;
        Ok(result)
    }
    fn cmp(&mut self, a: &ExactRational, b: &ExactRational) -> Result<Ordering, SparseMarkovError> {
        self.charge(1)?;
        if a.denom() == b.denom() {
            return Ok(a.numer().cmp(b.numer()));
        }
        if a.is_zero() || b.is_zero() {
            return Ok(a.numer().cmp(b.numer()));
        }
        self.bits(a.numer().bits() + b.denom().bits())?;
        self.bits(b.numer().bits() + a.denom().bits())?;
        // Avoid Ratio's recursive continued-fraction comparison. Both cross
        // products are admitted before allocation and use integer comparison.
        Ok((a.numer() * b.denom()).cmp(&(b.numer() * a.denom())))
    }
}

// Internal operands are canonical. Direct integer equality avoids recursive
// Ratio comparison and needs no cross products or extra bit growth.
fn same_rational(a: &ExactRational, b: &ExactRational) -> bool {
    a.numer() == b.numer() && a.denom() == b.denom()
}
