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

    pub fn dagger(&self) -> Result<Self, String> {
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
        let mut result = Self::zero(self.dimension)?;
        for row in 0..self.dimension {
            for column in 0..self.dimension {
                let mut sum = Complex64::ZERO;
                for inner in 0..self.dimension {
                    sum = sum + self.get(row, inner) * rhs.get(inner, column);
                }
                result.set(row, column, sum);
            }
        }
        Ok(result)
    }

    fn add_assign(&mut self, rhs: &Self) {
        for (left, right) in self.data.iter_mut().zip(&rhs.data) {
            *left = *left + *right;
        }
    }

    fn scaled(&self, factor: f64) -> Self {
        Self {
            dimension: self.dimension,
            data: self
                .data
                .iter()
                .map(|entry| Complex64::new(entry.re * factor, entry.im * factor))
                .collect(),
        }
    }

    pub fn trace(&self) -> Complex64 {
        (0..self.dimension).fold(Complex64::ZERO, |sum, index| sum + self.get(index, index))
    }

    pub fn frobenius_distance(&self, rhs: &Self) -> Result<f64, String> {
        if self.dimension != rhs.dimension {
            return Err("matrix dimensions differ".into());
        }
        Ok(self
            .data
            .iter()
            .zip(&rhs.data)
            .map(|(left, right)| (*left - *right).norm_sqr())
            .sum::<f64>()
            .sqrt())
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

/// Complete basis-readout range over every fixed density operator of a qubit
/// channel, obtained from the affine fixed set inside the Bloch ball.
#[derive(Debug, Clone, PartialEq)]
pub struct QubitFixedSpaceAnalysis {
    pub affine_dimension: usize,
    pub minimum_acceptance: f64,
    pub maximum_acceptance: f64,
    pub decision: StationaryDecision,
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
            completeness.add_assign(&operator.dagger()?.multiply(operator)?);
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
        let mut result = ComplexMatrix::zero(self.dimension)?;
        for operator in &self.kraus {
            let term = operator.multiply(density)?.multiply(&operator.dagger()?)?;
            result.add_assign(&term);
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
            ComplexMatrix::identity(self.dimension)?.scaled(1.0 / self.dimension as f64);
        let mut average = ComplexMatrix::zero(self.dimension)?;
        for iteration in 1..=max_iterations {
            average = average.scaled((iteration - 1) as f64 / iteration as f64);
            average.add_assign(&current.scaled(1.0 / iteration as f64));
            let mapped = self.apply(&average)?;
            let residual = mapped.frobenius_distance(&average)?;
            if residual <= tolerance {
                let trace = average.trace();
                if (trace.re - 1.0).abs() > self.validation_tolerance * 10.0
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

    /// Characterize every fixed qubit density and verify a computational-basis
    /// readout over the complete fixed set.
    pub fn analyze_qubit_basis_readout(
        &self,
        accepting_basis: usize,
        accept_at_least: f64,
        reject_at_most: f64,
        tolerance: f64,
    ) -> Result<QubitFixedSpaceAnalysis, String> {
        if self.dimension != 2 {
            return Err("complete fixed-space analysis currently supports qubits only".into());
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
                a[row][column] = mapped[row] - c[row];
            }
        }

        // Fixed points satisfy (I-A)r=c.
        let mut augmented = [[0.0; 4]; 3];
        for row in 0..3 {
            for column in 0..3 {
                augmented[row][column] = if row == column { 1.0 } else { 0.0 } - a[row][column];
            }
            augmented[row][3] = c[row];
        }
        let (particular, null_basis) = affine_solve_3(augmented, tolerance)?;
        let orthonormal = orthonormalize(&null_basis, tolerance);

        // Remove null-space components to get the minimum-norm point. The
        // remaining intersection is a Euclidean ball within the affine plane.
        let mut center = particular;
        for direction in &orthonormal {
            let projection = dot(center, *direction);
            for index in 0..3 {
                center[index] -= projection * direction[index];
            }
        }
        let center_norm_sqr = dot(center, center);
        if center_norm_sqr > 1.0 + tolerance {
            return Err(
                "the channel's affine fixed space does not intersect the Bloch ball".into(),
            );
        }
        let radius = (1.0 - center_norm_sqr).max(0.0).sqrt();

        // Computational basis probabilities are (1 +/- r_z)/2.
        let sign = if accepting_basis == 0 { 0.5 } else { -0.5 };
        let observable = [0.0, 0.0, sign];
        let midpoint = 0.5 + dot(observable, center);
        let projected_norm = orthonormal
            .iter()
            .map(|direction| dot(observable, *direction).powi(2))
            .sum::<f64>()
            .sqrt();
        let spread = radius * projected_norm;
        let minimum = (midpoint - spread).clamp(0.0, 1.0);
        let maximum = (midpoint + spread).clamp(0.0, 1.0);
        let decision = if minimum + tolerance >= accept_at_least {
            StationaryDecision::Accept
        } else if maximum - tolerance <= reject_at_most {
            StationaryDecision::Reject
        } else {
            StationaryDecision::Ambiguous
        };
        Ok(QubitFixedSpaceAnalysis {
            affine_dimension: orthonormal.len(),
            minimum_acceptance: minimum,
            maximum_acceptance: maximum,
            decision,
            center_bloch: center,
        })
    }
}

/// Build and completely analyze a source-level qubit channel declaration.
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
    let validation_tolerance = declaration.validation_tolerance.numerator as f64
        / declaration.validation_tolerance.denominator as f64;
    let analysis_tolerance = declaration.analysis_tolerance.numerator as f64
        / declaration.analysis_tolerance.denominator as f64;
    let accept_at_least = declaration.accept_at_least.numerator as f64
        / declaration.accept_at_least.denominator as f64;
    let reject_at_most =
        declaration.reject_at_most.numerator as f64 / declaration.reject_at_most.denominator as f64;
    let channel = QuantumChannel::from_kraus(kraus, validation_tolerance)?;
    channel.analyze_qubit_basis_readout(
        declaration.accepting_basis,
        accept_at_least,
        reject_at_most,
        analysis_tolerance,
    )
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
    Ok([
        2.0 * density.get(0, 1).re,
        -2.0 * density.get(0, 1).im,
        density.get(0, 0).re - density.get(1, 1).re,
    ])
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

/// RREF solve of a 3x3 affine system, returning one solution and a null basis.
fn affine_solve_3(
    mut matrix: [[f64; 4]; 3],
    tolerance: f64,
) -> Result<([f64; 3], Vec<[f64; 3]>), String> {
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
            matrix[pivot_row][entry] /= pivot;
        }
        for row in 0..3 {
            if row == pivot_row {
                continue;
            }
            let factor = matrix[row][column];
            for entry in column..4 {
                matrix[row][entry] -= factor * matrix[pivot_row][entry];
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

fn orthonormalize(vectors: &[[f64; 3]], tolerance: f64) -> Vec<[f64; 3]> {
    let mut result: Vec<[f64; 3]> = Vec::new();
    for vector in vectors {
        let mut candidate = *vector;
        for existing in &result {
            let projection = dot(candidate, *existing);
            for index in 0..3 {
                candidate[index] -= projection * existing[index];
            }
        }
        let norm = dot(candidate, candidate).sqrt();
        if norm > tolerance {
            for value in &mut candidate {
                *value /= norm;
            }
            result.push(candidate);
        }
    }
    result
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
}
