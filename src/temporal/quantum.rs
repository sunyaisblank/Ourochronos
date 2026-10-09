//! Finite-dimensional numerical Deutsch CTC semantics for quantum channels.
//!
//! Channels are supplied in Kraus form, which guarantees complete positivity;
//! the constructor numerically verifies `sum K_i^dagger K_i = I` (trace
//! preservation). Fixed density operators are approximated by Cesaro averages
//! of channel iterates, a method that also handles periodic channels.

#![allow(clippy::needless_range_loop)] // Dense elimination uses mathematical indices.

use std::ops::{Add, Mul, Sub};

use super::stochastic::StationaryDecision;

/// Largest dense matrix dimension accepted by the numerical research API.
/// Source-level QCHANNEL declarations are qubits (dimension two).
pub const MAX_QUANTUM_DIMENSION: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };

    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

impl Add for Complex64 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Sub for Complex64 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for Complex64 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

/// Dense square complex matrix in row-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexMatrix {
    dimension: usize,
    data: Vec<Complex64>,
}

impl ComplexMatrix {
    pub fn new(dimension: usize, data: Vec<Complex64>) -> Result<Self, String> {
        if dimension == 0 {
            return Err("matrix dimension must be positive".into());
        }
        if dimension > MAX_QUANTUM_DIMENSION {
            return Err(format!(
                "matrix dimension {dimension} exceeds maximum {MAX_QUANTUM_DIMENSION}"
            ));
        }
        let expected = dimension
            .checked_mul(dimension)
            .ok_or("matrix size overflow")?;
        if data.len() != expected {
            return Err(format!(
                "matrix has {} entries, expected {} for dimension {}",
                data.len(),
                expected,
                dimension
            ));
        }
        if data
            .iter()
            .any(|entry| !entry.re.is_finite() || !entry.im.is_finite())
        {
            return Err("matrix entries must be finite".into());
        }
        Ok(Self { dimension, data })
    }

    pub fn from_real(dimension: usize, data: Vec<f64>) -> Result<Self, String> {
        Self::new(
            dimension,
            data.into_iter().map(|re| Complex64::new(re, 0.0)).collect(),
        )
    }

    pub fn identity(dimension: usize) -> Result<Self, String> {
        let mut matrix = Self::zero(dimension)?;
        for index in 0..dimension {
            matrix.set(index, index, Complex64::ONE);
        }
        Ok(matrix)
    }

    pub fn zero(dimension: usize) -> Result<Self, String> {
        if dimension == 0 || dimension > MAX_QUANTUM_DIMENSION {
            return Err(format!(
                "matrix dimension {dimension} must be in 1..={MAX_QUANTUM_DIMENSION}"
            ));
        }
        let entries = dimension
            .checked_mul(dimension)
            .ok_or("matrix size overflow")?;
        Self::new(dimension, vec![Complex64::ZERO; entries])
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }
    pub fn get(&self, row: usize, column: usize) -> Complex64 {
        self.data[row * self.dimension + column]
    }
    pub fn set(&mut self, row: usize, column: usize, value: Complex64) {
        self.data[row * self.dimension + column] = value;
    }

    // Fallible matrix operations revalidate operands so mutation cannot bypass
    // the channel and analysis entry points' finite-value checks.
    fn validate_finite(&self) -> Result<(), String> {
        if self.data.iter().any(|entry| !entry.is_finite()) {
            return Err("matrix entries must be finite".into());
        }
        Ok(())
    }

    pub fn dagger(&self) -> Result<Self, String> {
        self.validate_finite()?;
        let mut result = Self::zero(self.dimension)?;
        for row in 0..self.dimension {
            for column in 0..self.dimension {
                result.set(column, row, self.get(row, column).conj());
            }
        }
        Ok(result)
    }

    pub fn multiply(&self, rhs: &Self) -> Result<Self, String> {
        if self.dimension != rhs.dimension {
            return Err("matrix dimensions differ".into());
        }
        self.validate_finite()?;
        rhs.validate_finite()?;
        let mut result = Self::zero(self.dimension)?;
        for row in 0..self.dimension {
            for column in 0..self.dimension {
                let mut sum = Complex64::ZERO;
                for inner in 0..self.dimension {
                    let product = self.get(row, inner) * rhs.get(inner, column);
                    if !product.is_finite() {
                        return Err("nonfinite intermediate in matrix multiplication".into());
                    }
                    sum = sum + product;
                    if !sum.is_finite() {
                        return Err("nonfinite intermediate in matrix multiplication".into());
                    }
                }
                result.set(row, column, sum);
            }
        }
        Ok(result)
    }

    fn add_assign(&mut self, rhs: &Self) -> Result<(), String> {
        if self.dimension != rhs.dimension {
            return Err("matrix dimensions differ".into());
        }
        self.validate_finite()?;
        rhs.validate_finite()?;
        for (left, right) in self.data.iter_mut().zip(&rhs.data) {
            *left = *left + *right;
            if !left.is_finite() {
                return Err("nonfinite intermediate in matrix addition".into());
            }
        }
        Ok(())
    }

    fn scaled(&self, factor: f64) -> Result<Self, String> {
        self.validate_finite()?;
        finite(factor, "matrix scale factor")?;
        Self::new(
            self.dimension,
            self.data
                .iter()
                .map(|entry| Complex64::new(entry.re * factor, entry.im * factor))
                .collect(),
        )
    }

    pub fn trace(&self) -> Complex64 {
        (0..self.dimension).fold(Complex64::ZERO, |sum, index| sum + self.get(index, index))
    }

    pub fn frobenius_distance(&self, rhs: &Self) -> Result<f64, String> {
        if self.dimension != rhs.dimension {
            return Err("matrix dimensions differ".into());
        }
        self.validate_finite()?;
        rhs.validate_finite()?;
        finite(
            self.data
                .iter()
                .zip(&rhs.data)
                .map(|(left, right)| (*left - *right).norm_sqr())
                .sum::<f64>()
                .sqrt(),
            "Frobenius residual",
        )
    }
}

#[derive(Debug, Clone)]
pub struct QuantumChannel {
    dimension: usize,
    kraus: Vec<ComplexMatrix>,
    validation_tolerance: f64,
}

#[derive(Debug, Clone)]
pub struct QuantumFixedPoint {
    pub density: ComplexMatrix,
    /// Frobenius residual `||Phi(rho)-rho||_F`.
    pub residual: f64,
    pub iterations: usize,
}

/// Numerical basis-readout range over the fixed affine set inside the Bloch
/// ball. Rank and extrema depend on the analysis tolerance; this is not an
/// independently certified error bound.
#[derive(Debug, Clone, PartialEq)]
pub struct QubitFixedSpaceAnalysis {
    pub affine_dimension: usize,
    pub minimum_acceptance: f64,
    pub maximum_acceptance: f64,
    pub decision: StationaryDecision,
    /// Whether an undecided readout lies within the tolerance guard of a
    /// decision threshold. This differs from a range spanning both decisions.
    /// False still does not certify the numerical rank or error bound.
    pub numerical_uncertainty: bool,
    pub analysis_tolerance: f64,
    /// Minimum-norm Bloch vector in the fixed affine space.
    pub center_bloch: [f64; 3],
}

impl QuantumChannel {
    pub fn from_kraus(
        kraus: Vec<ComplexMatrix>,
        validation_tolerance: f64,
    ) -> Result<Self, String> {
        if kraus.is_empty() {
            return Err("a quantum channel needs at least one Kraus operator".into());
        }
        if !validation_tolerance.is_finite() || validation_tolerance <= 0.0 {
            return Err("validation tolerance must be finite and positive".into());
        }
        let dimension = kraus[0].dimension();
        if kraus
            .iter()
            .any(|operator| operator.dimension() != dimension)
        {
            return Err("all Kraus operators must have the same square dimension".into());
        }
        let mut completeness = ComplexMatrix::zero(dimension)?;
        for operator in &kraus {
            completeness.add_assign(&operator.dagger()?.multiply(operator)?)?;
        }
        let error = completeness.frobenius_distance(&ComplexMatrix::identity(dimension)?)?;
        if error > validation_tolerance {
            return Err(format!(
                "Kraus operators are not trace preserving: completeness residual {} exceeds {}",
                error, validation_tolerance
            ));
        }
        Ok(Self {
            dimension,
            kraus,
            validation_tolerance,
        })
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn apply(&self, density: &ComplexMatrix) -> Result<ComplexMatrix, String> {
        if density.dimension() != self.dimension {
            return Err("density matrix has the wrong dimension".into());
        }
        density.validate_finite()?;
        let mut result = ComplexMatrix::zero(self.dimension)?;
        for operator in &self.kraus {
            let term = operator.multiply(density)?.multiply(&operator.dagger()?)?;
            result.add_assign(&term)?;
        }
        Ok(result)
    }

    /// Approximate a Deutsch fixed density operator by Cesaro averaging.
    pub fn fixed_point(
        &self,
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<QuantumFixedPoint, String> {
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err("fixed-point tolerance must be finite and positive".into());
        }
        if max_iterations == 0 {
            return Err("max_iterations must be positive".into());
        }
        let mut current =
            ComplexMatrix::identity(self.dimension)?.scaled(1.0 / self.dimension as f64)?;
        let mut average = ComplexMatrix::zero(self.dimension)?;
        for iteration in 1..=max_iterations {
            average = average.scaled((iteration - 1) as f64 / iteration as f64)?;
            average.add_assign(&current.scaled(1.0 / iteration as f64)?)?;
            let mapped = self.apply(&average)?;
            let residual = mapped.frobenius_distance(&average)?;
            if residual <= tolerance {
                let trace = average.trace();
                if !trace.is_finite()
                    || (trace.re - 1.0).abs() > self.validation_tolerance * 10.0
                    || trace.im.abs() > self.validation_tolerance * 10.0
                {
                    return Err("fixed-point approximation lost unit trace".into());
                }
                return Ok(QuantumFixedPoint {
                    density: average,
                    residual,
                    iterations: iteration,
                });
            }
            current = self.apply(&current)?;
        }
        let residual = self.apply(&average)?.frobenius_distance(&average)?;
        Err(format!(
            "quantum fixed-point approximation did not reach tolerance {} after {} iterations (residual {})",
            tolerance, max_iterations, residual
        ))
    }

    /// Estimate the qubit fixed space and computational-basis readout range.
    /// Numerical rank and extrema remain tolerance-dependent.
    pub fn analyze_qubit_basis_readout(
        &self,
        accepting_basis: usize,
        accept_at_least: f64,
        reject_at_most: f64,
        tolerance: f64,
    ) -> Result<QubitFixedSpaceAnalysis, String> {
        if self.dimension != 2 {
            return Err("numerical fixed-space analysis currently supports qubits only".into());
        }
        if accepting_basis > 1 {
            return Err("accepting_basis must be 0 or 1".into());
        }
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err("analysis tolerance must be finite and positive".into());
        }
        if !(0.0..=1.0).contains(&reject_at_most)
            || !(0.0..=1.0).contains(&accept_at_least)
            || reject_at_most > accept_at_least
        {
            return Err("decision thresholds must satisfy 0 <= reject <= accept <= 1".into());
        }

        // Phi maps Bloch vectors as r -> A r + c. Evaluate the origin and
        // three coordinate pure states to recover A and c.
        let origin = density_from_bloch([0.0, 0.0, 0.0]);
        let c = bloch_from_density(&self.apply(&origin)?)?;
        let mut a = [[0.0; 3]; 3];
        for column in 0..3 {
            let mut basis = [0.0; 3];
            basis[column] = 1.0;
            let mapped = bloch_from_density(&self.apply(&density_from_bloch(basis))?)?;
            for row in 0..3 {
                a[row][column] = finite(mapped[row] - c[row], "affine channel coefficient")?;
            }
        }

        // Fixed points satisfy (I-A)r=c.
        let mut augmented = [[0.0; 4]; 3];
        for row in 0..3 {
            for column in 0..3 {
                augmented[row][column] = finite(
                    if row == column { 1.0 } else { 0.0 } - a[row][column],
                    "fixed-space coefficient",
                )?;
            }
            augmented[row][3] = c[row];
        }
        let (particular, null_basis) = affine_solve_3(augmented, tolerance)?;
        let orthonormal = orthonormalize(&null_basis, tolerance)?;

        // Remove null-space components to get the minimum-norm point. The
        // remaining intersection is a Euclidean ball within the affine plane.
        let mut center = particular;
        for direction in &orthonormal {
            let projection = finite(dot(center, *direction), "fixed-space projection")?;
            for index in 0..3 {
                center[index] = finite(
                    center[index] - projection * direction[index],
                    "fixed-space center",
                )?;
            }
        }
        let center_norm_sqr = finite(dot(center, center), "fixed-space center norm")?;
        if center_norm_sqr > 1.0 + tolerance {
            return Err(
                "the channel's affine fixed space does not intersect the Bloch ball".into(),
            );
        }
        let radius = (1.0 - center_norm_sqr).max(0.0).sqrt();

        // Computational basis probabilities are (1 +/- r_z)/2.
        let sign = if accepting_basis == 0 { 0.5 } else { -0.5 };
        let observable = [0.0, 0.0, sign];
        let midpoint = finite(0.5 + dot(observable, center), "readout midpoint")?;
        let projected_norm = finite(
            orthonormal
                .iter()
                .map(|direction| dot(observable, *direction).powi(2))
                .sum::<f64>()
                .sqrt(),
            "readout projection norm",
        )?;
        let spread = finite(radius * projected_norm, "readout spread")?;
        let minimum = finite(midpoint - spread, "minimum acceptance")?.clamp(0.0, 1.0);
        let maximum = finite(midpoint + spread, "maximum acceptance")?.clamp(0.0, 1.0);
        // Tolerance withholds a near-threshold decision instead of relaxing
        // the requested promise. It is a numerical guard, not a derived bound
        // on the conditioning or rank error of the fixed-space solve.
        // Outward rounding also withholds decisions for sub-ulp tolerances.
        // MSRV 1.85 predates f64::next_up/next_down.
        let guarded_minimum = adjacent_down(minimum - tolerance);
        let guarded_maximum = adjacent_up(maximum + tolerance);
        let decision = if guarded_minimum >= accept_at_least {
            StationaryDecision::Accept
        } else if guarded_maximum <= reject_at_most {
            StationaryDecision::Reject
        } else {
            StationaryDecision::Ambiguous
        };
        let numerical_uncertainty = decision == StationaryDecision::Ambiguous
            && ((guarded_minimum < accept_at_least
                && adjacent_up(minimum + tolerance) >= accept_at_least)
                || (adjacent_down(maximum - tolerance) <= reject_at_most
                    && guarded_maximum > reject_at_most));
        Ok(QubitFixedSpaceAnalysis {
            affine_dimension: orthonormal.len(),
            minimum_acceptance: minimum,
            maximum_acceptance: maximum,
            decision,
            numerical_uncertainty,
            analysis_tolerance: tolerance,
            center_bloch: center,
        })
    }
}

/// Build and numerically analyze a source-level qubit channel declaration.
pub fn analyze_quantum_declaration(
    declaration: &crate::ast::QuantumChannelDeclaration,
) -> Result<QubitFixedSpaceAnalysis, String> {
    let kraus = declaration
        .kraus
        .iter()
        .map(|operator| {
            let entries = operator
                .iter()
                .map(|entry| {
                    Complex64::new(
                        entry.real.numerator as f64 / entry.real.denominator as f64,
                        entry.imaginary.numerator as f64 / entry.imaginary.denominator as f64,
                    )
                })
                .collect();
            ComplexMatrix::new(2, entries)
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Integer conversion and division each round. Enclose exact rational
    // thresholds instead of silently replacing them with nearby floats.
    let (validation_tolerance, _) = rational_enclosure(declaration.validation_tolerance)?;
    let (_, analysis_tolerance) = rational_enclosure(declaration.analysis_tolerance)?;
    let (accept_lower, accept_at_least) = rational_enclosure(declaration.accept_at_least)?;
    let (reject_at_most, reject_upper) = rational_enclosure(declaration.reject_at_most)?;
    for threshold in [declaration.accept_at_least, declaration.reject_at_most] {
        if threshold.numerator > threshold.denominator {
            return Err("decision thresholds must lie in [0,1]".into());
        }
    }
    if u128::from(declaration.reject_at_most.numerator)
        * u128::from(declaration.accept_at_least.denominator)
        > u128::from(declaration.accept_at_least.numerator)
            * u128::from(declaration.reject_at_most.denominator)
    {
        return Err("reject threshold exceeds accept threshold".into());
    }
    let channel = QuantumChannel::from_kraus(kraus, validation_tolerance)?;
    let mut analysis = channel.analyze_qubit_basis_readout(
        declaration.accepting_basis,
        accept_at_least.min(1.0),
        reject_at_most.max(0.0),
        analysis_tolerance,
    )?;
    if analysis.decision == StationaryDecision::Ambiguous
        && ((adjacent_up(analysis.minimum_acceptance + analysis_tolerance) >= accept_lower
            && adjacent_down(analysis.minimum_acceptance - analysis_tolerance) <= accept_at_least)
            || (adjacent_up(analysis.maximum_acceptance + analysis_tolerance) >= reject_at_most
                && adjacent_down(analysis.maximum_acceptance - analysis_tolerance) <= reject_upper))
    {
        analysis.numerical_uncertainty = true;
    }
    Ok(analysis)
}

// Adjacent IEEE-754 values implement outward rounding of finite arithmetic.
// Infinities at outward endpoints remain outside every valid threshold.
fn adjacent_up(value: f64) -> f64 {
    if value == f64::INFINITY {
        value
    } else if value == 0.0 {
        f64::from_bits(1)
    } else if value > 0.0 {
        f64::from_bits(value.to_bits() + 1)
    } else {
        f64::from_bits(value.to_bits() - 1)
    }
}

fn adjacent_down(value: f64) -> f64 {
    -adjacent_up(-value)
}

fn rational_enclosure(value: crate::ast::RationalLiteral) -> Result<(f64, f64), String> {
    if value.denominator == 0 {
        return Err("rational denominator must be positive".into());
    }
    if value.numerator == 0 {
        return Ok((0.0, 0.0));
    }
    let numerator = value.numerator as f64;
    let denominator = value.denominator as f64;
    Ok((
        adjacent_down(adjacent_down(numerator) / adjacent_up(denominator)),
        adjacent_up(adjacent_up(numerator) / adjacent_down(denominator)),
    ))
}

fn density_from_bloch(vector: [f64; 3]) -> ComplexMatrix {
    ComplexMatrix::new(
        2,
        vec![
            Complex64::new((1.0 + vector[2]) / 2.0, 0.0),
            Complex64::new(vector[0] / 2.0, -vector[1] / 2.0),
            Complex64::new(vector[0] / 2.0, vector[1] / 2.0),
            Complex64::new((1.0 - vector[2]) / 2.0, 0.0),
        ],
    )
    .expect("fixed 2x2 density shape")
}

fn bloch_from_density(density: &ComplexMatrix) -> Result<[f64; 3], String> {
    if density.dimension() != 2 {
        return Err("Bloch conversion requires a qubit density".into());
    }
    density.validate_finite()?;
    Ok([
        finite(2.0 * density.get(0, 1).re, "Bloch coordinate")?,
        finite(-2.0 * density.get(0, 1).im, "Bloch coordinate")?,
        finite(
            density.get(0, 0).re - density.get(1, 1).re,
            "Bloch coordinate",
        )?,
    ])
}

fn finite(value: f64, context: &str) -> Result<f64, String> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("nonfinite {context}"))
    }
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

/// RREF solve of a 3x3 affine system, returning one solution and a null basis.
fn affine_solve_3(
    mut matrix: [[f64; 4]; 3],
    tolerance: f64,
) -> Result<([f64; 3], Vec<[f64; 3]>), String> {
    for value in matrix.iter().flatten() {
        finite(*value, "fixed-space equation")?;
    }
    let mut pivot_row = 0;
    let mut pivot_columns = Vec::new();
    for column in 0..3 {
        let Some(found) = (pivot_row..3)
            .filter(|row| matrix[*row][column].abs() > tolerance)
            .max_by(|left, right| {
                matrix[*left][column]
                    .abs()
                    .total_cmp(&matrix[*right][column].abs())
            })
        else {
            continue;
        };
        matrix.swap(pivot_row, found);
        let pivot = matrix[pivot_row][column];
        for entry in column..4 {
            matrix[pivot_row][entry] = finite(
                matrix[pivot_row][entry] / pivot,
                "fixed-space pivot normalization",
            )?;
        }
        for row in 0..3 {
            if row == pivot_row {
                continue;
            }
            let factor = matrix[row][column];
            for entry in column..4 {
                matrix[row][entry] = finite(
                    matrix[row][entry] - factor * matrix[pivot_row][entry],
                    "fixed-space elimination",
                )?;
            }
        }
        pivot_columns.push(column);
        pivot_row += 1;
    }
    for row in pivot_row..3 {
        if matrix[row][0..3]
            .iter()
            .all(|value| value.abs() <= tolerance)
            && matrix[row][3].abs() > tolerance
        {
            return Err("quantum fixed-space equations are inconsistent".into());
        }
    }
    let mut particular = [0.0; 3];
    for (row, column) in pivot_columns.iter().enumerate() {
        particular[*column] = matrix[row][3];
    }
    let free_columns: Vec<usize> = (0..3)
        .filter(|column| !pivot_columns.contains(column))
        .collect();
    let mut null_basis = Vec::new();
    for free in free_columns {
        let mut direction = [0.0; 3];
        direction[free] = 1.0;
        for (row, pivot) in pivot_columns.iter().enumerate() {
            direction[*pivot] = -matrix[row][free];
        }
        null_basis.push(direction);
    }
    Ok((particular, null_basis))
}

fn orthonormalize(vectors: &[[f64; 3]], tolerance: f64) -> Result<Vec<[f64; 3]>, String> {
    let mut result: Vec<[f64; 3]> = Vec::new();
    for vector in vectors {
        let mut candidate = *vector;
        for existing in &result {
            let projection = finite(dot(candidate, *existing), "null-space projection")?;
            for index in 0..3 {
                candidate[index] = finite(
                    candidate[index] - projection * existing[index],
                    "null-space component",
                )?;
            }
        }
        let norm = finite(dot(candidate, candidate).sqrt(), "null-space norm")?;
        if norm > tolerance {
            for value in &mut candidate {
                *value = finite(*value / norm, "normalized null-space component")?;
            }
            result.push(candidate);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_channel_fixes_maximally_mixed_state() {
        let channel =
            QuantumChannel::from_kraus(vec![ComplexMatrix::identity(2).unwrap()], 1e-12).unwrap();
        let fixed = channel.fixed_point(1e-12, 10).unwrap();
        assert_eq!(fixed.iterations, 1);
        assert!((fixed.density.get(0, 0).re - 0.5).abs() < 1e-12);
        assert!((fixed.density.get(1, 1).re - 0.5).abs() < 1e-12);
    }

    #[test]
    fn amplitude_damping_converges_to_ground_state() {
        let gamma: f64 = 0.25;
        let k0 = ComplexMatrix::from_real(2, vec![1.0, 0.0, 0.0, (1.0 - gamma).sqrt()]).unwrap();
        let k1 = ComplexMatrix::from_real(2, vec![0.0, gamma.sqrt(), 0.0, 0.0]).unwrap();
        let channel = QuantumChannel::from_kraus(vec![k0, k1], 1e-12).unwrap();
        let fixed = channel.fixed_point(1e-4, 20_000).unwrap();
        assert!(fixed.density.get(0, 0).re > 0.999);
        assert!(fixed.density.get(1, 1).re < 0.001);
        assert!(fixed.residual <= 1e-4);
    }

    #[test]
    fn non_trace_preserving_kraus_set_is_rejected() {
        let bad = ComplexMatrix::from_real(2, vec![0.5, 0.0, 0.0, 0.5]).unwrap();
        assert!(QuantumChannel::from_kraus(vec![bad], 1e-12).is_err());
    }

    #[test]
    fn identity_channel_exposes_the_entire_bloch_ball_and_ambiguous_readout() {
        let channel =
            QuantumChannel::from_kraus(vec![ComplexMatrix::identity(2).unwrap()], 1e-12).unwrap();
        let analysis = channel
            .analyze_qubit_basis_readout(1, 2.0 / 3.0, 1.0 / 3.0, 1e-10)
            .unwrap();
        assert_eq!(analysis.affine_dimension, 3);
        assert!(analysis.minimum_acceptance < 1e-10);
        assert!((analysis.maximum_acceptance - 1.0).abs() < 1e-10);
        assert_eq!(analysis.decision, StationaryDecision::Ambiguous);
    }

    #[test]
    fn amplitude_damping_has_one_fixed_density_and_unanimous_readout() {
        let gamma: f64 = 0.25;
        let k0 = ComplexMatrix::from_real(2, vec![1.0, 0.0, 0.0, (1.0 - gamma).sqrt()]).unwrap();
        let k1 = ComplexMatrix::from_real(2, vec![0.0, gamma.sqrt(), 0.0, 0.0]).unwrap();
        let channel = QuantumChannel::from_kraus(vec![k0, k1], 1e-12).unwrap();
        let analysis = channel
            .analyze_qubit_basis_readout(0, 2.0 / 3.0, 1.0 / 3.0, 1e-10)
            .unwrap();
        assert_eq!(analysis.affine_dimension, 0);
        assert!((analysis.minimum_acceptance - 1.0).abs() < 1e-9);
        assert!((analysis.maximum_acceptance - 1.0).abs() < 1e-9);
        assert_eq!(analysis.decision, StationaryDecision::Accept);
    }

    #[test]
    fn dense_matrix_constructors_reject_hostile_dimensions_without_overflow() {
        assert!(ComplexMatrix::zero(usize::MAX).is_err());
        assert!(ComplexMatrix::identity(usize::MAX).is_err());
        assert!(ComplexMatrix::new(usize::MAX, Vec::new()).is_err());
        // This square fits usize; the dimension limit must precede allocation.
        assert!(ComplexMatrix::zero(1_000_000).is_err());
    }

    #[test]
    fn source_reset_channel_accepts_on_every_fixed_density() {
        let program = crate::parser::parse(
            "QCHANNEL reset {\n\
             QUBIT;\n\
             KRAUS { C 1/1 0/1 C 0/1 0/1 C 0/1 0/1 C 0/1 0/1 };\n\
             KRAUS { C 0/1 0/1 C 1/1 0/1 C 0/1 0/1 C 0/1 0/1 };\n\
             ACCEPT_BASIS 0;\n\
             }",
        )
        .unwrap();
        let analysis =
            analyze_quantum_declaration(program.quantum_declaration.as_ref().unwrap()).unwrap();
        assert_eq!(analysis.affine_dimension, 0);
        assert!((analysis.minimum_acceptance - 1.0).abs() < 1e-8);
        assert_eq!(analysis.decision, StationaryDecision::Accept);
    }

    #[test]
    fn source_identity_channel_reports_all_fixed_density_ambiguity() {
        let program = crate::parser::parse(
            "QCHANNEL identity {\n\
             QUBIT;\n\
             KRAUS { C 1/1 0/1 C 0/1 0/1 C 0/1 0/1 C 1/1 0/1 };\n\
             ACCEPT_BASIS 1;\n\
             }",
        )
        .unwrap();
        let analysis =
            analyze_quantum_declaration(program.quantum_declaration.as_ref().unwrap()).unwrap();
        assert_eq!(analysis.affine_dimension, 3);
        assert_eq!(analysis.decision, StationaryDecision::Ambiguous);
        assert!(!analysis.numerical_uncertainty);
    }

    #[test]
    fn source_complex_amplitudes_parse_and_validate() {
        let program = crate::parser::parse(
            "QCHANNEL pauli_y {\n\
             QUBIT;\n\
             KRAUS { C 0/1 0/1 C 0/1 -1/1 C 0/1 1/1 C 0/1 0/1 };\n\
             ACCEPT_BASIS 0;\n\
             }",
        )
        .unwrap();
        let declaration = program.quantum_declaration.as_ref().unwrap();
        let entries = declaration.kraus[0].as_slice();
        assert_eq!(entries[1].imaginary.numerator, -1);
        assert!(analyze_quantum_declaration(declaration).is_ok());
    }

    #[test]
    fn default_tolerance_does_not_accept_a_zero_probability_readout() {
        let program = crate::parser::parse(
            "QCHANNEL exact_reset_threshold {\n\
             QUBIT;\n\
             KRAUS { C 1/1 0/1 C 0/1 0/1 C 0/1 0/1 C 0/1 0/1 };\n\
             KRAUS { C 0/1 0/1 C 1/1 0/1 C 0/1 0/1 C 0/1 0/1 };\n\
             ACCEPT_BASIS 1;\n\
             ACCEPT_AT_LEAST 1/2000000000;\n\
             REJECT_AT_MOST 0/1;\n\
             }",
        )
        .unwrap();
        let analysis =
            analyze_quantum_declaration(program.quantum_declaration.as_ref().unwrap()).unwrap();
        assert_eq!(analysis.minimum_acceptance, 0.0);
        assert_eq!(analysis.maximum_acceptance, 0.0);
        assert_eq!(analysis.decision, StationaryDecision::Ambiguous);
        assert!(analysis.numerical_uncertainty);
    }

    #[test]
    fn near_threshold_readouts_withhold_both_decisions() {
        // The replacement channel rho -> I/2 has exactly one fixed density
        // with basis-readout probability 1/2.
        let amplitude = 0.5_f64.sqrt();
        let mut kraus = Vec::new();
        for index in 0..4 {
            let mut entries = vec![0.0; 4];
            entries[index] = amplitude;
            kraus.push(ComplexMatrix::from_real(2, entries).unwrap());
        }
        let channel = QuantumChannel::from_kraus(kraus, 1e-12).unwrap();
        for (accept, reject) in [(0.5 + 5e-11, 0.3), (0.9, 0.5 - 5e-11)] {
            let analysis = channel
                .analyze_qubit_basis_readout(0, accept, reject, 1e-10)
                .unwrap();
            assert_eq!(analysis.affine_dimension, 0);
            assert!((analysis.minimum_acceptance - 0.5).abs() < 1e-12);
            assert!((analysis.maximum_acceptance - 0.5).abs() < 1e-12);
            assert_eq!(analysis.decision, StationaryDecision::Ambiguous);
        }
    }

    #[test]
    fn sub_ulp_tolerance_and_rational_thresholds_cannot_round_into_acceptance() {
        let mut program =
            crate::parser::parse(include_str!("../../examples/quantum_reset.ouro")).unwrap();
        let declaration = program.quantum_declaration.as_mut().unwrap();
        declaration.accept_at_least = crate::ast::RationalLiteral {
            numerator: 1,
            denominator: 1,
        };
        declaration.analysis_tolerance = crate::ast::RationalLiteral {
            numerator: 1,
            denominator: 1_000_000_000_000_000_000,
        };
        let analysis = analyze_quantum_declaration(declaration).unwrap();
        assert_eq!(analysis.decision, StationaryDecision::Ambiguous);
        assert!(analysis.numerical_uncertainty);

        // Exact basis-1 probability is 16/25, below the threshold, although
        // both round to the same float. The channel replaces every density.
        let program = crate::parser::parse(
            "QCHANNEL rational_boundary { QUBIT;\n\
             KRAUS { C 3/5 0/1 C 0/1 0/1 C 0/1 0/1 C 0/1 0/1 };\n\
             KRAUS { C 0/1 0/1 C 3/5 0/1 C 0/1 0/1 C 0/1 0/1 };\n\
             KRAUS { C 0/1 0/1 C 0/1 0/1 C 4/5 0/1 C 0/1 0/1 };\n\
             KRAUS { C 0/1 0/1 C 0/1 0/1 C 0/1 0/1 C 4/5 0/1 };\n\
             ACCEPT_BASIS 1;\n\
             ACCEPT_AT_LEAST 640000000000000001/1000000000000000000;\n\
             REJECT_AT_MOST 0/1;\n\
             ANALYSIS_TOLERANCE 1/1000000000000000000; }",
        )
        .unwrap();
        let analysis =
            analyze_quantum_declaration(program.quantum_declaration.as_ref().unwrap()).unwrap();
        assert_eq!(analysis.decision, StationaryDecision::Ambiguous);
        assert!(analysis.numerical_uncertainty);
    }

    #[test]
    fn mutated_nonfinite_matrices_are_rejected_at_numeric_boundaries() {
        let identity = ComplexMatrix::identity(2).unwrap();
        let channel = QuantumChannel::from_kraus(vec![identity.clone()], 1e-12).unwrap();
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for invalid in [Complex64::new(value, 0.0), Complex64::new(0.0, value)] {
                let mut mutated = identity.clone();
                mutated.set(0, 0, invalid);
                assert!(QuantumChannel::from_kraus(vec![mutated.clone()], 1e-12).is_err());
                assert!(channel.apply(&mutated).is_err());
                assert!(mutated.multiply(&identity).is_err());
                assert!(identity.frobenius_distance(&mutated).is_err());
            }
        }
    }

    #[test]
    fn finite_overflow_cannot_become_valid_channel_or_residual() {
        let huge = Complex64::new(1e308, 1e308);
        let operator =
            ComplexMatrix::new(2, vec![huge, Complex64::ZERO, Complex64::ZERO, huge]).unwrap();
        assert!(QuantumChannel::from_kraus(vec![operator.clone()], 1e-10).is_err());
        assert!(operator.multiply(&operator).is_err());
        assert!(operator
            .frobenius_distance(&ComplexMatrix::zero(2).unwrap())
            .is_err());
        let amplitude = 0.5_f64.sqrt();
        let hadamard =
            ComplexMatrix::from_real(2, vec![amplitude, amplitude, amplitude, -amplitude]).unwrap();
        let channel = QuantumChannel::from_kraus(vec![hadamard], 1e-12).unwrap();
        assert!(channel
            .apply(&ComplexMatrix::new(2, vec![huge; 4]).unwrap())
            .is_err());

        // Fixed-space elimination has its own arithmetic boundary even when
        // every supplied coefficient is finite.
        assert!(affine_solve_3([[1e-308, 1e308, 0.0, 1.0], [0.0; 4], [0.0; 4]], 1e-310).is_err());
        assert!(orthonormalize(&[[1e308, 1e308, 0.0]], 1e-10).is_err());
    }
}
