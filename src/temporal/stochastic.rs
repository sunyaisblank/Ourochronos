//! Exact finite classical Deutsch semantics for rational Markov chains.
//!
//! The ordinary VM induces a deterministic transition. This module covers the
//! more general classical Aaronson--Watrous model: a row-stochastic rational
//! matrix. It returns one exact stationary distribution for every closed
//! recurrent class; all stationary distributions are their convex hull.

#![allow(clippy::needless_range_loop)] // Matrix algorithms are clearest in indexed form.

use std::cmp::Ordering;
use std::fmt;

/// Reduced exact rational number backed by signed 128-bit integers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rational {
    numerator: i128,
    denominator: i128,
}

impl Rational {
    pub const ZERO: Self = Self {
        numerator: 0,
        denominator: 1,
    };
    pub const ONE: Self = Self {
        numerator: 1,
        denominator: 1,
    };

    pub fn new(numerator: i128, denominator: i128) -> Result<Self, String> {
        if denominator == 0 {
            return Err("rational denominator must be nonzero".into());
        }
        if numerator == 0 {
            return Ok(Self::ZERO);
        }
        let (numerator, denominator) = if denominator < 0 {
            (
                numerator.checked_neg().ok_or("rational overflow")?,
                denominator.checked_neg().ok_or("rational overflow")?,
            )
        } else {
            (numerator, denominator)
        };
        let divisor = gcd(numerator.unsigned_abs(), denominator as u128) as i128;
        Ok(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    pub fn numerator(self) -> i128 {
        self.numerator
    }
    pub fn denominator(self) -> i128 {
        self.denominator
    }
    pub fn is_zero(self) -> bool {
        self.numerator == 0
    }
    pub fn is_negative(self) -> bool {
        self.numerator < 0
    }

    /// Exact checked comparison without floating point.
    pub fn cmp_checked(self, rhs: Self) -> Result<Ordering, String> {
        let left = self
            .numerator
            .checked_mul(rhs.denominator)
            .ok_or("rational overflow")?;
        let right = rhs
            .numerator
            .checked_mul(self.denominator)
            .ok_or("rational overflow")?;
        Ok(left.cmp(&right))
    }

    fn add(self, rhs: Self) -> Result<Self, String> {
        let left = self
            .numerator
            .checked_mul(rhs.denominator)
            .ok_or("rational overflow")?;
        let right = rhs
            .numerator
            .checked_mul(self.denominator)
            .ok_or("rational overflow")?;
        let numerator = left.checked_add(right).ok_or("rational overflow")?;
        let denominator = self
            .denominator
            .checked_mul(rhs.denominator)
            .ok_or("rational overflow")?;
        Self::new(numerator, denominator)
    }

    fn sub(self, rhs: Self) -> Result<Self, String> {
        self.add(Self::new(
            rhs.numerator.checked_neg().ok_or("rational overflow")?,
            rhs.denominator,
        )?)
    }

    fn mul(self, rhs: Self) -> Result<Self, String> {
        let numerator = self
            .numerator
            .checked_mul(rhs.numerator)
            .ok_or("rational overflow")?;
        let denominator = self
            .denominator
            .checked_mul(rhs.denominator)
            .ok_or("rational overflow")?;
        Self::new(numerator, denominator)
    }

    fn div(self, rhs: Self) -> Result<Self, String> {
        if rhs.is_zero() {
            return Err("division by zero during stationary solve".into());
        }
        let numerator = self
            .numerator
            .checked_mul(rhs.denominator)
            .ok_or("rational overflow")?;
        let denominator = self
            .denominator
            .checked_mul(rhs.numerator)
            .ok_or("rational overflow")?;
        Self::new(numerator, denominator)
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator == 1 {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "{}/{}", self.numerator, self.denominator)
        }
    }
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a.max(1)
}

/// A finite row-stochastic transition matrix with exact rational entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalMarkovChain {
    transition: Vec<Vec<Rational>>,
}

/// Extremal stationary distributions, one per closed recurrent class.
/// Every stationary distribution is a convex combination of these vectors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StationaryFamily {
    pub extremal: Vec<Vec<Rational>>,
    pub recurrent_classes: Vec<Vec<usize>>,
}

/// Aaronson--Watrous decision classification across every extremal fixed point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StationaryDecision {
    Accept,
    Reject,
    /// Some fixed points fall in the promise gap or disagree on the decision.
    Ambiguous,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StationaryAnalysis {
    pub family: StationaryFamily,
    pub acceptance_probabilities: Vec<Rational>,
    pub decision: StationaryDecision,
}

impl RationalMarkovChain {
    pub fn new(transition: Vec<Vec<Rational>>) -> Result<Self, String> {
        let states = transition.len();
        if states == 0 {
            return Err("a Markov chain must have at least one state".into());
        }
        for (row_index, row) in transition.iter().enumerate() {
            if row.len() != states {
                return Err(format!(
                    "row {} has length {}, expected {}",
                    row_index,
                    row.len(),
                    states
                ));
            }
            if row.iter().any(|probability| probability.is_negative()) {
                return Err(format!("row {} contains a negative probability", row_index));
            }
            let sum = row
                .iter()
                .try_fold(Rational::ZERO, |sum, value| sum.add(*value))?;
            if sum != Rational::ONE {
                return Err(format!("row {} sums to {}, expected 1", row_index, sum));
            }
        }
        Ok(Self { transition })
    }

    pub fn states(&self) -> usize {
        self.transition.len()
    }

    /// Compute all extremal stationary distributions exactly.
    pub fn stationary_family(&self) -> Result<StationaryFamily, String> {
        let recurrent_classes = self.closed_recurrent_classes();
        let mut extremal = Vec::with_capacity(recurrent_classes.len());
        for class in &recurrent_classes {
            let local = self.solve_closed_class(class)?;
            let mut global = vec![Rational::ZERO; self.states()];
            for (index, state) in class.iter().enumerate() {
                global[*state] = local[index];
            }
            extremal.push(global);
        }
        Ok(StationaryFamily {
            extremal,
            recurrent_classes,
        })
    }

    fn closed_recurrent_classes(&self) -> Vec<Vec<usize>> {
        let n = self.states();
        let mut reachable = vec![vec![false; n]; n];
        for i in 0..n {
            reachable[i][i] = true;
            for j in 0..n {
                reachable[i][j] |= !self.transition[i][j].is_zero();
            }
        }
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    reachable[i][j] |= reachable[i][k] && reachable[k][j];
                }
            }
        }

        let mut visited = vec![false; n];
        let mut closed = Vec::new();
        for state in 0..n {
            if visited[state] {
                continue;
            }
            let class: Vec<usize> = (0..n)
                .filter(|other| reachable[state][*other] && reachable[*other][state])
                .collect();
            for member in &class {
                visited[*member] = true;
            }
            let is_closed = class.iter().all(|member| {
                (0..n).all(|target| {
                    class.contains(&target) || self.transition[*member][target].is_zero()
                })
            });
            if is_closed {
                closed.push(class);
            }
        }
        closed
    }

    fn solve_closed_class(&self, class: &[usize]) -> Result<Vec<Rational>, String> {
        let n = class.len();
        let mut augmented = vec![vec![Rational::ZERO; n + 1]; n];

        // n-1 independent equations from pi P = pi.
        for equation in 0..n.saturating_sub(1) {
            let target = class[equation];
            for (variable, source) in class.iter().enumerate() {
                augmented[equation][variable] = self.transition[*source][target];
                if variable == equation {
                    augmented[equation][variable] =
                        augmented[equation][variable].sub(Rational::ONE)?;
                }
            }
        }
        // Normalization replaces the dependent final stationarity equation.
        for variable in 0..n {
            augmented[n - 1][variable] = Rational::ONE;
        }
        augmented[n - 1][n] = Rational::ONE;

        gaussian_solve(augmented)
    }
}

impl TryFrom<&crate::ast::MarkovDeclaration> for RationalMarkovChain {
    type Error = String;

    fn try_from(declaration: &crate::ast::MarkovDeclaration) -> Result<Self, Self::Error> {
        let transition = declaration
            .transition
            .iter()
            .map(|row| {
                row.iter()
                    .map(|literal| {
                        Rational::new(literal.numerator as i128, literal.denominator as i128)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(transition)
    }
}

impl StationaryFamily {
    /// Exact acceptance probability for each extremal fixed point.
    pub fn acceptance_probabilities(
        &self,
        accepting_states: &[bool],
    ) -> Result<Vec<Rational>, String> {
        if self
            .extremal
            .first()
            .is_some_and(|distribution| distribution.len() != accepting_states.len())
        {
            return Err("accepting-state vector has the wrong length".into());
        }
        self.extremal
            .iter()
            .map(|distribution| {
                distribution.iter().zip(accepting_states).try_fold(
                    Rational::ZERO,
                    |sum, (weight, accepts)| {
                        if *accepts {
                            sum.add(*weight)
                        } else {
                            Ok(sum)
                        }
                    },
                )
            })
            .collect()
    }

    /// Apply bounded-error decision thresholds to every extremal fixed point.
    pub fn classify(
        &self,
        accepting_states: &[bool],
        accept_at_least: Rational,
        reject_at_most: Rational,
    ) -> Result<StationaryAnalysis, String> {
        if reject_at_most.cmp_checked(accept_at_least)? == Ordering::Greater {
            return Err("reject threshold exceeds accept threshold".into());
        }
        if reject_at_most.is_negative()
            || accept_at_least.cmp_checked(Rational::ONE)? == Ordering::Greater
        {
            return Err("decision thresholds must lie in [0,1]".into());
        }
        let probabilities = self.acceptance_probabilities(accepting_states)?;
        let all_accept = probabilities.iter().try_fold(true, |all, probability| {
            Ok::<bool, String>(all && probability.cmp_checked(accept_at_least)? != Ordering::Less)
        })?;
        let all_reject = probabilities.iter().try_fold(true, |all, probability| {
            Ok::<bool, String>(all && probability.cmp_checked(reject_at_most)? != Ordering::Greater)
        })?;
        let decision = if all_accept {
            StationaryDecision::Accept
        } else if all_reject {
            StationaryDecision::Reject
        } else {
            StationaryDecision::Ambiguous
        };
        Ok(StationaryAnalysis {
            family: self.clone(),
            acceptance_probabilities: probabilities,
            decision,
        })
    }
}

/// Parse, solve, and classify a source-level stochastic Deutsch declaration.
pub fn analyze_declaration(
    declaration: &crate::ast::MarkovDeclaration,
) -> Result<StationaryAnalysis, String> {
    if declaration.states > crate::parser::MAX_MARKOV_STATES {
        return Err(format!(
            "MARKOV state count {} exceeds maximum {}",
            declaration.states,
            crate::parser::MAX_MARKOV_STATES
        ));
    }
    if declaration.transition.len() != declaration.states {
        return Err(format!(
            "MARKOV declares {} states but its transition chain has {}",
            declaration.states,
            declaration.transition.len()
        ));
    }
    if let Some(state) = declaration
        .accepting_states
        .iter()
        .find(|state| **state >= declaration.states)
    {
        return Err(format!(
            "MARKOV accepting state {} is outside 0..{}",
            state, declaration.states
        ));
    }
    let chain = RationalMarkovChain::try_from(declaration)?;
    if declaration.states != chain.states() {
        return Err(format!(
            "MARKOV declares {} states but its transition chain has {}",
            declaration.states,
            chain.states()
        ));
    }
    let family = chain.stationary_family()?;
    let mut accepting = vec![false; declaration.states];
    for state in &declaration.accepting_states {
        accepting[*state] = true;
    }
    let accept = Rational::new(
        declaration.accept_at_least.numerator as i128,
        declaration.accept_at_least.denominator as i128,
    )?;
    let reject = Rational::new(
        declaration.reject_at_most.numerator as i128,
        declaration.reject_at_most.denominator as i128,
    )?;
    family.classify(&accepting, accept, reject)
}

fn gaussian_solve(mut matrix: Vec<Vec<Rational>>) -> Result<Vec<Rational>, String> {
    let rows = matrix.len();
    let columns = rows;
    let mut pivot_row = 0;
    let mut pivots = Vec::new();
    for column in 0..columns {
        let Some(found) = (pivot_row..rows).find(|row| !matrix[*row][column].is_zero()) else {
            continue;
        };
        matrix.swap(pivot_row, found);
        let pivot = matrix[pivot_row][column];
        for entry in column..=columns {
            matrix[pivot_row][entry] = matrix[pivot_row][entry].div(pivot)?;
        }
        for row in 0..rows {
            if row == pivot_row || matrix[row][column].is_zero() {
                continue;
            }
            let factor = matrix[row][column];
            for entry in column..=columns {
                matrix[row][entry] =
                    matrix[row][entry].sub(factor.mul(matrix[pivot_row][entry])?)?;
            }
        }
        pivots.push((pivot_row, column));
        pivot_row += 1;
    }
    if pivot_row != columns {
        return Err("stationary system was unexpectedly rank-deficient".into());
    }
    let mut solution = vec![Rational::ZERO; columns];
    for (row, column) in pivots {
        solution[column] = matrix[row][columns];
    }
    if solution.iter().any(|value| value.is_negative()) {
        return Err("stationary solve produced a negative probability".into());
    }
    Ok(solution)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: i128, d: i128) -> Rational {
        Rational::new(n, d).unwrap()
    }

    #[test]
    fn deterministic_cycle_has_uniform_stationary_distribution() {
        let z = Rational::ZERO;
        let o = Rational::ONE;
        let chain =
            RationalMarkovChain::new(vec![vec![z, o, z], vec![z, z, o], vec![o, z, z]]).unwrap();
        let family = chain.stationary_family().unwrap();
        assert_eq!(family.recurrent_classes, vec![vec![0, 1, 2]]);
        assert_eq!(family.extremal[0], vec![r(1, 3), r(1, 3), r(1, 3)]);
    }

    #[test]
    fn stochastic_chain_is_solved_exactly() {
        let chain =
            RationalMarkovChain::new(vec![vec![r(1, 2), r(1, 2)], vec![r(1, 4), r(3, 4)]]).unwrap();
        let family = chain.stationary_family().unwrap();
        assert_eq!(family.extremal[0], vec![r(1, 3), r(2, 3)]);
        assert_eq!(
            family.acceptance_probabilities(&[false, true]).unwrap(),
            vec![r(2, 3)]
        );
    }

    #[test]
    fn multiple_recurrent_classes_expose_all_fixed_points() {
        let z = Rational::ZERO;
        let o = Rational::ONE;
        let h = r(1, 2);
        let chain =
            RationalMarkovChain::new(vec![vec![o, z, z], vec![z, o, z], vec![h, h, z]]).unwrap();
        let family = chain.stationary_family().unwrap();
        assert_eq!(family.recurrent_classes, vec![vec![0], vec![1]]);
        assert_eq!(
            family
                .acceptance_probabilities(&[true, false, false])
                .unwrap(),
            vec![o, z]
        );
    }

    #[test]
    fn invalid_probability_rows_are_rejected() {
        assert!(RationalMarkovChain::new(vec![vec![r(1, 2)]]).is_err());
        assert!(RationalMarkovChain::new(vec![vec![r(-1, 1)]]).is_err());
    }

    #[test]
    fn source_markov_declaration_is_solved_and_accepted() {
        let program = crate::parser::parse(
            "MARKOV biased {\n\
             STATES 2;\n\
             ROW 0 { 1/2 1/2 };\n\
             ROW 1 { 1/4 3/4 };\n\
             ACCEPTING { 1 };\n\
             ACCEPT_AT_LEAST 2/3;\n\
             REJECT_AT_MOST 1/3;\n\
             }",
        )
        .unwrap();
        let analysis = analyze_declaration(program.markov_declaration.as_ref().unwrap()).unwrap();
        assert_eq!(analysis.acceptance_probabilities, vec![r(2, 3)]);
        assert_eq!(analysis.decision, StationaryDecision::Accept);
    }

    #[test]
    fn disagreement_between_recurrent_classes_is_ambiguous() {
        let program = crate::parser::parse(
            "MARKOV split {\n\
             STATES 2;\n\
             ROW 0 { 1/1 0/1 };\n\
             ROW 1 { 0/1 1/1 };\n\
             ACCEPTING { 0 };\n\
             }",
        )
        .unwrap();
        let analysis = analyze_declaration(program.markov_declaration.as_ref().unwrap()).unwrap();
        assert_eq!(
            analysis.acceptance_probabilities,
            vec![Rational::ONE, Rational::ZERO]
        );
        assert_eq!(analysis.decision, StationaryDecision::Ambiguous);
    }

    #[test]
    fn malformed_source_markov_models_fail_loudly() {
        let missing =
            crate::parser::parse("MARKOV bad { STATES 2; ROW 0 { 1/1 0/1 }; ACCEPTING { 0 }; }")
                .unwrap_err();
        assert!(missing.contains("missing row 1"));

        let non_stochastic =
            crate::parser::parse("MARKOV bad { STATES 1; ROW 0 { 1/2 }; ACCEPTING { 0 }; }")
                .unwrap();
        let error =
            analyze_declaration(non_stochastic.markov_declaration.as_ref().unwrap()).unwrap_err();
        assert!(error.contains("sums to"));
    }

    #[test]
    fn manual_markov_declarations_validate_dimensions_before_allocation() {
        let one = crate::ast::RationalLiteral {
            numerator: 1,
            denominator: 1,
        };
        let mut declaration = crate::ast::MarkovDeclaration {
            name: "manual".to_string(),
            states: usize::MAX,
            transition: vec![vec![one]],
            accepting_states: vec![0],
            accept_at_least: one,
            reject_at_most: one,
        };
        assert_eq!(
            analyze_declaration(&declaration).unwrap_err(),
            format!(
                "MARKOV state count {} exceeds maximum {}",
                usize::MAX,
                crate::parser::MAX_MARKOV_STATES
            )
        );

        declaration.states = 2;
        assert_eq!(
            analyze_declaration(&declaration).unwrap_err(),
            "MARKOV declares 2 states but its transition chain has 1"
        );

        declaration.states = 1;
        declaration.accepting_states = vec![1];
        assert_eq!(
            analyze_declaration(&declaration).unwrap_err(),
            "MARKOV accepting state 1 is outside 0..1"
        );
    }
}
