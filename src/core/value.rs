//! Value type for OUROCHRONOS: 64-bit unsigned integer with provenance tracking.
//!
//! All arithmetic is performed modulo 2^64 (wrapping semantics).
//! Provenance tracks which anamnesis cells influenced this value.

use std::ops::{Add, Sub, Mul, Div, Rem, Not, BitAnd, BitOr, BitXor};
use std::fmt;

use super::provenance::Provenance;
use super::error::{OuroError, OuroResult, DivisionByZeroPolicy, SourceLocation};

/// A value in OUROCHRONOS: 64-bit unsigned integer with provenance tracking.
///
/// All arithmetic is performed modulo 2^64 (wrapping semantics).
/// Provenance tracks which anamnesis cells influenced this value.
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct Value {
    /// The numeric value (64-bit unsigned).
    pub val: u64,
    /// Causal provenance (which oracle cells influenced this value).
    pub prov: Provenance,
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{:?}]", self.val, self.prov)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.val)
    }
}

/// An item in the program output buffer.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OutputItem {
    /// A numeric value output (e.g., from OUTPUT opcode).
    Val(Value),
    /// A single character output (e.g., from EMIT opcode).
    Char(u8),
}

impl Default for OutputItem {
    fn default() -> Self {
        OutputItem::Val(Value::ZERO)
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Value {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.val.cmp(&other.val)
    }
}

impl Value {
    /// The zero value with no provenance.
    pub const ZERO: Value = Value { val: 0, prov: Provenance::none() };

    /// The one value with no provenance.
    pub const ONE: Value = Value { val: 1, prov: Provenance::none() };

    /// Minus one (-1) as two's complement.
    pub const MINUS_ONE: Value = Value { val: u64::MAX, prov: Provenance::none() };

    /// Create a new value with no provenance.
    pub fn new(v: u64) -> Self {
        Value { val: v, prov: Provenance::none() }
    }

    /// Create a new value from a signed integer.
    ///
    /// The value is stored as two's complement representation.
    pub fn from_signed(v: i64) -> Self {
        Value { val: v as u64, prov: Provenance::none() }
    }

    /// Create a value with explicit provenance.
    pub fn with_provenance(v: u64, prov: Provenance) -> Self {
        Value { val: v, prov }
    }

    /// Create a signed value with explicit provenance.
    pub fn signed_with_provenance(v: i64, prov: Provenance) -> Self {
        Value { val: v as u64, prov }
    }

    /// Check if this value is temporally pure.
    pub fn is_pure(&self) -> bool {
        self.prov.is_pure()
    }

    /// Convert to boolean (0 = false, nonzero = true).
    pub fn to_bool(&self) -> bool {
        self.val != 0
    }

    /// Create boolean value (1 or 0) with merged provenance.
    pub fn from_bool_with_prov(b: bool, prov: Provenance) -> Self {
        Value { val: if b { 1 } else { 0 }, prov }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Signed Integer Operations
    // ═══════════════════════════════════════════════════════════════════

    /// Interpret this value as a signed 64-bit integer.
    ///
    /// Uses two's complement representation.
    #[inline]
    pub fn as_signed(&self) -> i64 {
        self.val as i64
    }

    /// Check if this value, interpreted as signed, is negative.
    #[inline]
    pub fn is_negative(&self) -> bool {
        (self.val as i64) < 0
    }

    /// Check if this value, interpreted as signed, is positive.
    #[inline]
    pub fn is_positive(&self) -> bool {
        let signed = self.val as i64;
        signed > 0
    }

    /// Signed addition with wrapping semantics.
    pub fn signed_add(self, rhs: Self) -> Self {
        Value {
            val: (self.as_signed().wrapping_add(rhs.as_signed())) as u64,
            prov: self.prov.merge(&rhs.prov),
        }
    }

    /// Signed subtraction with wrapping semantics.
    pub fn signed_sub(self, rhs: Self) -> Self {
        Value {
            val: (self.as_signed().wrapping_sub(rhs.as_signed())) as u64,
            prov: self.prov.merge(&rhs.prov),
        }
    }

    /// Signed multiplication with wrapping semantics.
    pub fn signed_mul(self, rhs: Self) -> Self {
        Value {
            val: (self.as_signed().wrapping_mul(rhs.as_signed())) as u64,
            prov: self.prov.merge(&rhs.prov),
        }
    }

    /// Signed division with wrapping semantics.
    ///
    /// Returns zero if divisor is zero (matches unsigned behavior).
    pub fn signed_div(self, rhs: Self) -> Self {
        if rhs.val == 0 {
            Value {
                val: 0,
                prov: self.prov.merge(&rhs.prov),
            }
        } else {
            Value {
                val: (self.as_signed().wrapping_div(rhs.as_signed())) as u64,
                prov: self.prov.merge(&rhs.prov),
            }
        }
    }

    /// Signed modulo with wrapping semantics.
    ///
    /// Returns zero if divisor is zero (matches unsigned behavior).
    pub fn signed_mod(self, rhs: Self) -> Self {
        if rhs.val == 0 {
            Value {
                val: 0,
                prov: self.prov.merge(&rhs.prov),
            }
        } else {
            Value {
                val: (self.as_signed().wrapping_rem(rhs.as_signed())) as u64,
                prov: self.prov.merge(&rhs.prov),
            }
        }
    }

    /// Signed less-than comparison.
    pub fn signed_lt(self, rhs: Self) -> Self {
        let result = self.as_signed() < rhs.as_signed();
        Value::from_bool_with_prov(result, self.prov.merge(&rhs.prov))
    }

    /// Signed greater-than comparison.
    pub fn signed_gt(self, rhs: Self) -> Self {
        let result = self.as_signed() > rhs.as_signed();
        Value::from_bool_with_prov(result, self.prov.merge(&rhs.prov))
    }

    /// Signed less-than-or-equal comparison.
    pub fn signed_lte(self, rhs: Self) -> Self {
        let result = self.as_signed() <= rhs.as_signed();
        Value::from_bool_with_prov(result, self.prov.merge(&rhs.prov))
    }

    /// Signed greater-than-or-equal comparison.
    pub fn signed_gte(self, rhs: Self) -> Self {
        let result = self.as_signed() >= rhs.as_signed();
        Value::from_bool_with_prov(result, self.prov.merge(&rhs.prov))
    }

    /// Absolute value (signed interpretation).
    pub fn abs(self) -> Self {
        let signed = self.as_signed();
        Value {
            val: signed.wrapping_abs() as u64,
            prov: self.prov,
        }
    }

    /// Sign of the value: -1, 0, or 1.
    pub fn signum(self) -> Self {
        let signed = self.as_signed();
        Value {
            val: signed.signum() as u64,
            prov: self.prov,
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Configurable Arithmetic Operations
    // ═══════════════════════════════════════════════════════════════════

    /// Division with configurable zero-divisor policy.
    ///
    /// # Arguments
    /// * `rhs` - The divisor
    /// * `policy` - How to handle division by zero
    ///
    /// # Returns
    /// `Ok(result)` for successful division, or `Err` if policy is Error and divisor is zero.
    #[inline]
    pub fn checked_div(self, rhs: Self, policy: DivisionByZeroPolicy) -> OuroResult<Self> {
        self.checked_div_at(rhs, policy, SourceLocation::default())
    }

    /// Division with configurable policy and location tracking.
    pub fn checked_div_at(
        self,
        rhs: Self,
        policy: DivisionByZeroPolicy,
        location: SourceLocation,
    ) -> OuroResult<Self> {
        if rhs.val == 0 {
            match policy {
                DivisionByZeroPolicy::ReturnZero => Ok(Value {
                    val: 0,
                    prov: self.prov.merge(&rhs.prov),
                }),
                DivisionByZeroPolicy::Sentinel(v) => Ok(Value {
                    val: v,
                    prov: self.prov.merge(&rhs.prov),
                }),
                DivisionByZeroPolicy::Error => Err(OuroError::DivisionByZero {
                    dividend: self.val,
                    location,
                }),
            }
        } else {
            Ok(Value {
                val: self.val.wrapping_div(rhs.val),
                prov: self.prov.merge(&rhs.prov),
            })
        }
    }

    /// Modulo with configurable zero-divisor policy.
    #[inline]
    pub fn checked_mod(self, rhs: Self, policy: DivisionByZeroPolicy) -> OuroResult<Self> {
        self.checked_mod_at(rhs, policy, SourceLocation::default())
    }

    /// Modulo with configurable policy and location tracking.
    pub fn checked_mod_at(
        self,
        rhs: Self,
        policy: DivisionByZeroPolicy,
        location: SourceLocation,
    ) -> OuroResult<Self> {
        if rhs.val == 0 {
            match policy {
                DivisionByZeroPolicy::ReturnZero => Ok(Value {
                    val: 0,
                    prov: self.prov.merge(&rhs.prov),
                }),
                DivisionByZeroPolicy::Sentinel(v) => Ok(Value {
                    val: v,
                    prov: self.prov.merge(&rhs.prov),
                }),
                DivisionByZeroPolicy::Error => Err(OuroError::ModuloByZero {
                    dividend: self.val,
                    location,
                }),
            }
        } else {
            Ok(Value {
                val: self.val.wrapping_rem(rhs.val),
                prov: self.prov.merge(&rhs.prov),
            })
        }
    }
}

// Arithmetic operations with provenance merging

impl Add for Value {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Value {
            val: self.val.wrapping_add(rhs.val),
            prov: self.prov.merge(&rhs.prov),
        }
    }
}

impl Sub for Value {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Value {
            val: self.val.wrapping_sub(rhs.val),
            prov: self.prov.merge(&rhs.prov),
        }
    }
}

impl Mul for Value {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Value {
            val: self.val.wrapping_mul(rhs.val),
            prov: self.prov.merge(&rhs.prov),
        }
    }
}

impl Div for Value {
    type Output = Self;
    /// Division with zero-divisor handling per specification §5.3:
    /// v₁ ⟨Div⟩ v₂ = v₁ ÷ v₂ if v₂ ≠ 0, else 0
    fn div(self, rhs: Self) -> Self::Output {
        Value {
            val: if rhs.val == 0 { 0 } else { self.val.wrapping_div(rhs.val) },
            prov: self.prov.merge(&rhs.prov),
        }
    }
}

impl Rem for Value {
    type Output = Self;
    /// Modulo with zero-divisor handling per specification §5.3:
    /// v₁ ⟨Mod⟩ v₂ = v₁ mod v₂ if v₂ ≠ 0, else 0
    fn rem(self, rhs: Self) -> Self::Output {
        Value {
            val: if rhs.val == 0 { 0 } else { self.val.wrapping_rem(rhs.val) },
            prov: self.prov.merge(&rhs.prov),
        }
    }
}

// Bitwise operations

impl Not for Value {
    type Output = Self;
    fn not(self) -> Self::Output {
        Value {
            val: !self.val,
            prov: self.prov,
        }
    }
}

impl BitAnd for Value {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        Value {
            val: self.val & rhs.val,
            prov: self.prov.merge(&rhs.prov),
        }
    }
}

impl BitOr for Value {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Value {
            val: self.val | rhs.val,
            prov: self.prov.merge(&rhs.prov),
        }
    }
}

impl BitXor for Value {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        Value {
            val: self.val ^ rhs.val,
            prov: self.prov.merge(&rhs.prov),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_arithmetic() {
        let a = Value::new(10);
        let b = Value::new(3);

        assert_eq!((a.clone() + b.clone()).val, 13);
        assert_eq!((a.clone() - b.clone()).val, 7);
        assert_eq!((a.clone() * b.clone()).val, 30);
        assert_eq!((a.clone() / b.clone()).val, 3);
        assert_eq!((a.clone() % b.clone()).val, 1);
    }

    #[test]
    fn test_division_by_zero_returns_zero() {
        let a = Value::new(42);
        let zero = Value::new(0);

        assert_eq!((a.clone() / zero.clone()).val, 0);
        assert_eq!((a.clone() % zero.clone()).val, 0);
    }

    #[test]
    fn test_wrapping_arithmetic() {
        let max = Value::new(u64::MAX);
        let one = Value::new(1);

        assert_eq!((max + one).val, 0); // Wraps around
    }

    #[test]
    fn test_signed_from_signed() {
        let pos = Value::from_signed(42);
        assert_eq!(pos.as_signed(), 42);

        let neg = Value::from_signed(-42);
        assert_eq!(neg.as_signed(), -42);

        let zero = Value::from_signed(0);
        assert_eq!(zero.as_signed(), 0);
    }

    #[test]
    fn test_signed_is_negative() {
        let neg = Value::from_signed(-1);
        assert!(neg.is_negative());
        assert!(!neg.is_positive());

        let pos = Value::from_signed(1);
        assert!(!pos.is_negative());
        assert!(pos.is_positive());

        let zero = Value::ZERO;
        assert!(!zero.is_negative());
        assert!(!zero.is_positive());
    }

    #[test]
    fn test_signed_arithmetic() {
        let a = Value::from_signed(-10);
        let b = Value::from_signed(3);

        // Signed addition
        let sum = a.clone().signed_add(b.clone());
        assert_eq!(sum.as_signed(), -7);

        // Signed subtraction
        let diff = a.clone().signed_sub(b.clone());
        assert_eq!(diff.as_signed(), -13);

        // Signed multiplication
        let prod = a.clone().signed_mul(b.clone());
        assert_eq!(prod.as_signed(), -30);

        // Signed division
        let quot = a.clone().signed_div(b.clone());
        assert_eq!(quot.as_signed(), -3);

        // Signed modulo
        let rem = a.clone().signed_mod(b);
        assert_eq!(rem.as_signed(), -1);
    }

    #[test]
    fn test_signed_comparisons() {
        let neg = Value::from_signed(-5);
        let pos = Value::from_signed(5);

        // Signed less-than
        assert_eq!(neg.clone().signed_lt(pos.clone()).val, 1);
        assert_eq!(pos.clone().signed_lt(neg.clone()).val, 0);

        // Signed greater-than
        assert_eq!(pos.clone().signed_gt(neg.clone()).val, 1);
        assert_eq!(neg.clone().signed_gt(pos.clone()).val, 0);
    }

    #[test]
    fn test_signed_abs_signum() {
        let neg = Value::from_signed(-42);
        let pos = Value::from_signed(42);

        assert_eq!(neg.clone().abs().as_signed(), 42);
        assert_eq!(pos.clone().abs().as_signed(), 42);

        assert_eq!(neg.signum().as_signed(), -1);
        assert_eq!(pos.signum().as_signed(), 1);
        assert_eq!(Value::ZERO.signum().as_signed(), 0);
    }

    #[test]
    fn test_minus_one_constant() {
        assert_eq!(Value::MINUS_ONE.as_signed(), -1);
        assert_eq!(Value::MINUS_ONE.val, u64::MAX);
    }

    #[test]
    fn test_checked_div_return_zero_policy() {
        let a = Value::new(42);
        let zero = Value::new(0);

        let result = a.checked_div(zero, DivisionByZeroPolicy::ReturnZero);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().val, 0);
    }

    #[test]
    fn test_checked_div_error_policy() {
        let a = Value::new(42);
        let zero = Value::new(0);

        let result = a.checked_div(zero, DivisionByZeroPolicy::Error);
        assert!(result.is_err());
    }

    #[test]
    fn test_checked_div_sentinel_policy() {
        let a = Value::new(42);
        let zero = Value::new(0);

        let result = a.checked_div(zero, DivisionByZeroPolicy::Sentinel(u64::MAX));
        assert!(result.is_ok());
        assert_eq!(result.unwrap().val, u64::MAX);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Property Tests for Arithmetic Invariants
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_signed_additive_inverse() {
        // Property: a + (-a) == 0 for all a (except i64::MIN which cannot be negated)
        let test_values: Vec<i64> = vec![
            0, 1, -1, 42, -42,
            i64::MAX,
            i64::MAX / 2, i64::MIN / 2,
            1000000, -1000000,
        ];

        for val in test_values {
            let a = Value::from_signed(val);
            let neg_a = Value::from_signed(val.wrapping_neg());
            let sum = a.signed_add(neg_a);
            assert_eq!(sum.val, 0, "a + (-a) should equal 0 for a = {}", val);
        }

        // Special case: i64::MIN
        // i64::MIN.wrapping_neg() == i64::MIN due to two's complement
        // So i64::MIN + i64::MIN = 0 in wrapping arithmetic
        let min = Value::from_signed(i64::MIN);
        let neg_min = Value::from_signed(i64::MIN.wrapping_neg());
        let sum = min.signed_add(neg_min);
        assert_eq!(sum.val, 0, "i64::MIN + wrapping_neg(i64::MIN) should equal 0");
    }

    #[test]
    fn test_addition_commutativity() {
        // Property: a + b == b + a
        let test_pairs: Vec<(u64, u64)> = vec![
            (0, 0), (1, 0), (0, 1), (1, 1),
            (42, 100), (u64::MAX, 0), (u64::MAX, 1),
            (u64::MAX / 2, u64::MAX / 2),
        ];

        for (av, bv) in test_pairs {
            let a = Value::new(av);
            let b = Value::new(bv);
            let sum1 = a.clone() + b.clone();
            let sum2 = b + a;
            assert_eq!(sum1.val, sum2.val, "a + b should equal b + a for {} + {}", av, bv);
        }
    }

    #[test]
    fn test_multiplication_commutativity() {
        // Property: a * b == b * a
        let test_pairs: Vec<(u64, u64)> = vec![
            (0, 0), (1, 0), (0, 1), (1, 1),
            (7, 13), (42, 100), (u64::MAX, 0), (u64::MAX, 1),
        ];

        for (av, bv) in test_pairs {
            let a = Value::new(av);
            let b = Value::new(bv);
            let prod1 = a.clone() * b.clone();
            let prod2 = b * a;
            assert_eq!(prod1.val, prod2.val, "a * b should equal b * a for {} * {}", av, bv);
        }
    }

    #[test]
    fn test_division_policy_consistency() {
        // Property: checked_div behaves same as checked_div_at with default location
        let dividend = Value::new(100);
        let divisors = vec![
            Value::new(0), Value::new(1), Value::new(7), Value::new(100),
        ];
        let policies = vec![
            DivisionByZeroPolicy::ReturnZero,
            DivisionByZeroPolicy::Error,
            DivisionByZeroPolicy::Sentinel(999),
        ];

        for divisor in &divisors {
            for policy in &policies {
                let r1 = dividend.clone().checked_div(divisor.clone(), *policy);
                let r2 = dividend.clone().checked_div_at(
                    divisor.clone(),
                    *policy,
                    SourceLocation::default()
                );

                match (r1, r2) {
                    (Ok(v1), Ok(v2)) => assert_eq!(v1.val, v2.val),
                    (Err(_), Err(_)) => {} // Both errors is OK
                    _ => panic!("checked_div and checked_div_at should produce same result type"),
                }
            }
        }
    }

    #[test]
    fn test_modulo_policy_consistency() {
        // Property: checked_mod behaves same as checked_mod_at with default location
        let dividend = Value::new(100);
        let divisors = vec![
            Value::new(0), Value::new(1), Value::new(7), Value::new(100),
        ];
        let policies = vec![
            DivisionByZeroPolicy::ReturnZero,
            DivisionByZeroPolicy::Error,
            DivisionByZeroPolicy::Sentinel(999),
        ];

        for divisor in &divisors {
            for policy in &policies {
                let r1 = dividend.clone().checked_mod(divisor.clone(), *policy);
                let r2 = dividend.clone().checked_mod_at(
                    divisor.clone(),
                    *policy,
                    SourceLocation::default()
                );

                match (r1, r2) {
                    (Ok(v1), Ok(v2)) => assert_eq!(v1.val, v2.val),
                    (Err(_), Err(_)) => {} // Both errors is OK
                    _ => panic!("checked_mod and checked_mod_at should produce same result type"),
                }
            }
        }
    }

    #[test]
    fn test_bitwise_identity() {
        // Property: a & a == a, a | a == a, a ^ 0 == a
        let test_values: Vec<u64> = vec![
            0, 1, 42, u64::MAX, u64::MAX / 2, 0xDEADBEEF,
        ];

        for val in test_values {
            let a = Value::new(val);
            let zero = Value::ZERO;

            // a & a == a
            assert_eq!((a.clone() & a.clone()).val, val, "a & a should equal a");

            // a | a == a
            assert_eq!((a.clone() | a.clone()).val, val, "a | a should equal a");

            // a ^ 0 == a
            assert_eq!((a.clone() ^ zero).val, val, "a ^ 0 should equal a");
        }
    }

    #[test]
    fn test_wrapping_behavior() {
        // Property: MAX + 1 wraps to 0
        let max = Value::new(u64::MAX);
        let one = Value::ONE;
        assert_eq!((max + one).val, 0, "MAX + 1 should wrap to 0");

        // Property: 0 - 1 wraps to MAX
        let zero = Value::ZERO;
        let one = Value::ONE;
        assert_eq!((zero - one).val, u64::MAX, "0 - 1 should wrap to MAX");
    }
}
