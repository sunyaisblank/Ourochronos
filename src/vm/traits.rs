//! VM trait extraction for eliminating code duplication.
//!
//! This module defines traits for VM operations that can be specialized
//! for different execution contexts:
//! - Full VM (Executor): Tracks provenance for causal dependency analysis
//! - Fast VM (FastExecutor): Operates on raw u64 values for pure code
//!
//! # Design
//!
//! The trait hierarchy separates:
//! - Primitive operations (add, sub, mul, etc.) that each VM specializes
//! - Compound operations (default implementations using primitives)
//! - Stack operations (specialized per VM due to different value types)

use crate::core::Value;
#[cfg(test)]
use crate::core::provenance::Provenance;

/// Trait for binary arithmetic operations.
///
/// Each VM implementation provides its own versions that handle
/// provenance tracking (or lack thereof) appropriately.
pub trait ArithmeticOps {
    type Value: Clone;
    type Error;

    /// Add two values.
    fn value_add(a: Self::Value, b: Self::Value) -> Self::Value;

    /// Subtract two values.
    fn value_sub(a: Self::Value, b: Self::Value) -> Self::Value;

    /// Multiply two values.
    fn value_mul(a: Self::Value, b: Self::Value) -> Self::Value;

    /// Divide two values (returns 0 for division by zero).
    fn value_div(a: Self::Value, b: Self::Value) -> Self::Value;

    /// Modulo operation (returns 0 for modulo by zero).
    fn value_mod(a: Self::Value, b: Self::Value) -> Self::Value;

    /// Negate a value.
    fn value_neg(a: Self::Value) -> Self::Value;

    /// Absolute value.
    fn value_abs(a: Self::Value) -> Self::Value;

    /// Minimum of two values.
    fn value_min(a: Self::Value, b: Self::Value) -> Self::Value;

    /// Maximum of two values.
    fn value_max(a: Self::Value, b: Self::Value) -> Self::Value;

    /// Sign of a value (-1, 0, or 1).
    fn value_sign(a: Self::Value) -> Self::Value;
}

/// Trait for bitwise operations.
pub trait BitwiseOps: ArithmeticOps {
    /// Bitwise NOT.
    fn value_not(a: Self::Value) -> Self::Value;

    /// Bitwise AND.
    fn value_and(a: Self::Value, b: Self::Value) -> Self::Value;

    /// Bitwise OR.
    fn value_or(a: Self::Value, b: Self::Value) -> Self::Value;

    /// Bitwise XOR.
    fn value_xor(a: Self::Value, b: Self::Value) -> Self::Value;

    /// Left shift.
    fn value_shl(a: Self::Value, n: Self::Value) -> Self::Value;

    /// Right shift.
    fn value_shr(a: Self::Value, n: Self::Value) -> Self::Value;
}

/// Trait for comparison operations.
pub trait ComparisonOps: ArithmeticOps {
    /// Equality comparison.
    fn value_eq(a: Self::Value, b: Self::Value) -> Self::Value;

    /// Inequality comparison.
    fn value_neq(a: Self::Value, b: Self::Value) -> Self::Value;

    /// Less than (unsigned).
    fn value_lt(a: Self::Value, b: Self::Value) -> Self::Value;

    /// Greater than (unsigned).
    fn value_gt(a: Self::Value, b: Self::Value) -> Self::Value;

    /// Less than or equal (unsigned).
    fn value_lte(a: Self::Value, b: Self::Value) -> Self::Value;

    /// Greater than or equal (unsigned).
    fn value_gte(a: Self::Value, b: Self::Value) -> Self::Value;

    /// Less than (signed).
    fn value_slt(a: Self::Value, b: Self::Value) -> Self::Value;

    /// Greater than (signed).
    fn value_sgt(a: Self::Value, b: Self::Value) -> Self::Value;

    /// Less than or equal (signed).
    fn value_slte(a: Self::Value, b: Self::Value) -> Self::Value;

    /// Greater than or equal (signed).
    fn value_sgte(a: Self::Value, b: Self::Value) -> Self::Value;
}

// ═══════════════════════════════════════════════════════════════════════════
// Implementation for Value (with provenance tracking)
// ═══════════════════════════════════════════════════════════════════════════

/// Marker type for provenance-tracked operations.
pub struct ProvenanceOps;

impl ArithmeticOps for ProvenanceOps {
    type Value = Value;
    type Error = String;

    #[inline]
    fn value_add(a: Value, b: Value) -> Value {
        a + b
    }

    #[inline]
    fn value_sub(a: Value, b: Value) -> Value {
        a - b
    }

    #[inline]
    fn value_mul(a: Value, b: Value) -> Value {
        a * b
    }

    #[inline]
    fn value_div(a: Value, b: Value) -> Value {
        a / b
    }

    #[inline]
    fn value_mod(a: Value, b: Value) -> Value {
        a % b
    }

    #[inline]
    fn value_neg(a: Value) -> Value {
        Value {
            val: a.val.wrapping_neg(),
            prov: a.prov,
        }
    }

    #[inline]
    fn value_abs(a: Value) -> Value {
        a.abs()
    }

    #[inline]
    fn value_min(a: Value, b: Value) -> Value {
        let prov = a.prov.merge(&b.prov);
        let val = if a.val < b.val { a.val } else { b.val };
        Value { val, prov }
    }

    #[inline]
    fn value_max(a: Value, b: Value) -> Value {
        let prov = a.prov.merge(&b.prov);
        let val = if a.val > b.val { a.val } else { b.val };
        Value { val, prov }
    }

    #[inline]
    fn value_sign(a: Value) -> Value {
        a.signum()
    }
}

impl BitwiseOps for ProvenanceOps {
    #[inline]
    fn value_not(a: Value) -> Value {
        !a
    }

    #[inline]
    fn value_and(a: Value, b: Value) -> Value {
        a & b
    }

    #[inline]
    fn value_or(a: Value, b: Value) -> Value {
        a | b
    }

    #[inline]
    fn value_xor(a: Value, b: Value) -> Value {
        a ^ b
    }

    #[inline]
    fn value_shl(a: Value, n: Value) -> Value {
        let shift = (n.val % 64) as u32;
        Value {
            val: a.val.wrapping_shl(shift),
            prov: a.prov.merge(&n.prov),
        }
    }

    #[inline]
    fn value_shr(a: Value, n: Value) -> Value {
        let shift = (n.val % 64) as u32;
        Value {
            val: a.val.wrapping_shr(shift),
            prov: a.prov.merge(&n.prov),
        }
    }
}

impl ComparisonOps for ProvenanceOps {
    #[inline]
    fn value_eq(a: Value, b: Value) -> Value {
        let prov = a.prov.merge(&b.prov);
        Value { val: if a.val == b.val { 1 } else { 0 }, prov }
    }

    #[inline]
    fn value_neq(a: Value, b: Value) -> Value {
        let prov = a.prov.merge(&b.prov);
        Value { val: if a.val != b.val { 1 } else { 0 }, prov }
    }

    #[inline]
    fn value_lt(a: Value, b: Value) -> Value {
        let prov = a.prov.merge(&b.prov);
        Value { val: if a.val < b.val { 1 } else { 0 }, prov }
    }

    #[inline]
    fn value_gt(a: Value, b: Value) -> Value {
        let prov = a.prov.merge(&b.prov);
        Value { val: if a.val > b.val { 1 } else { 0 }, prov }
    }

    #[inline]
    fn value_lte(a: Value, b: Value) -> Value {
        let prov = a.prov.merge(&b.prov);
        Value { val: if a.val <= b.val { 1 } else { 0 }, prov }
    }

    #[inline]
    fn value_gte(a: Value, b: Value) -> Value {
        let prov = a.prov.merge(&b.prov);
        Value { val: if a.val >= b.val { 1 } else { 0 }, prov }
    }

    #[inline]
    fn value_slt(a: Value, b: Value) -> Value {
        let prov = a.prov.merge(&b.prov);
        let a_signed = a.val as i64;
        let b_signed = b.val as i64;
        Value { val: if a_signed < b_signed { 1 } else { 0 }, prov }
    }

    #[inline]
    fn value_sgt(a: Value, b: Value) -> Value {
        let prov = a.prov.merge(&b.prov);
        let a_signed = a.val as i64;
        let b_signed = b.val as i64;
        Value { val: if a_signed > b_signed { 1 } else { 0 }, prov }
    }

    #[inline]
    fn value_slte(a: Value, b: Value) -> Value {
        let prov = a.prov.merge(&b.prov);
        let a_signed = a.val as i64;
        let b_signed = b.val as i64;
        Value { val: if a_signed <= b_signed { 1 } else { 0 }, prov }
    }

    #[inline]
    fn value_sgte(a: Value, b: Value) -> Value {
        let prov = a.prov.merge(&b.prov);
        let a_signed = a.val as i64;
        let b_signed = b.val as i64;
        Value { val: if a_signed >= b_signed { 1 } else { 0 }, prov }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Implementation for u64 (no provenance tracking)
// ═══════════════════════════════════════════════════════════════════════════

/// Marker type for pure (non-provenance) operations.
pub struct PureOps;

impl ArithmeticOps for PureOps {
    type Value = u64;
    type Error = String;

    #[inline]
    fn value_add(a: u64, b: u64) -> u64 {
        a.wrapping_add(b)
    }

    #[inline]
    fn value_sub(a: u64, b: u64) -> u64 {
        a.wrapping_sub(b)
    }

    #[inline]
    fn value_mul(a: u64, b: u64) -> u64 {
        a.wrapping_mul(b)
    }

    #[inline]
    fn value_div(a: u64, b: u64) -> u64 {
        if b == 0 { 0 } else { a / b }
    }

    #[inline]
    fn value_mod(a: u64, b: u64) -> u64 {
        if b == 0 { 0 } else { a % b }
    }

    #[inline]
    fn value_neg(a: u64) -> u64 {
        a.wrapping_neg()
    }

    #[inline]
    fn value_abs(a: u64) -> u64 {
        let signed = a as i64;
        signed.abs() as u64
    }

    #[inline]
    fn value_min(a: u64, b: u64) -> u64 {
        if a < b { a } else { b }
    }

    #[inline]
    fn value_max(a: u64, b: u64) -> u64 {
        if a > b { a } else { b }
    }

    #[inline]
    fn value_sign(a: u64) -> u64 {
        let signed = a as i64;
        match signed.cmp(&0) {
            std::cmp::Ordering::Less => (-1i64) as u64,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        }
    }
}

impl BitwiseOps for PureOps {
    #[inline]
    fn value_not(a: u64) -> u64 {
        !a
    }

    #[inline]
    fn value_and(a: u64, b: u64) -> u64 {
        a & b
    }

    #[inline]
    fn value_or(a: u64, b: u64) -> u64 {
        a | b
    }

    #[inline]
    fn value_xor(a: u64, b: u64) -> u64 {
        a ^ b
    }

    #[inline]
    fn value_shl(a: u64, n: u64) -> u64 {
        a.wrapping_shl((n % 64) as u32)
    }

    #[inline]
    fn value_shr(a: u64, n: u64) -> u64 {
        a.wrapping_shr((n % 64) as u32)
    }
}

impl ComparisonOps for PureOps {
    #[inline]
    fn value_eq(a: u64, b: u64) -> u64 {
        if a == b { 1 } else { 0 }
    }

    #[inline]
    fn value_neq(a: u64, b: u64) -> u64 {
        if a != b { 1 } else { 0 }
    }

    #[inline]
    fn value_lt(a: u64, b: u64) -> u64 {
        if a < b { 1 } else { 0 }
    }

    #[inline]
    fn value_gt(a: u64, b: u64) -> u64 {
        if a > b { 1 } else { 0 }
    }

    #[inline]
    fn value_lte(a: u64, b: u64) -> u64 {
        if a <= b { 1 } else { 0 }
    }

    #[inline]
    fn value_gte(a: u64, b: u64) -> u64 {
        if a >= b { 1 } else { 0 }
    }

    #[inline]
    fn value_slt(a: u64, b: u64) -> u64 {
        let a_signed = a as i64;
        let b_signed = b as i64;
        if a_signed < b_signed { 1 } else { 0 }
    }

    #[inline]
    fn value_sgt(a: u64, b: u64) -> u64 {
        let a_signed = a as i64;
        let b_signed = b as i64;
        if a_signed > b_signed { 1 } else { 0 }
    }

    #[inline]
    fn value_slte(a: u64, b: u64) -> u64 {
        let a_signed = a as i64;
        let b_signed = b as i64;
        if a_signed <= b_signed { 1 } else { 0 }
    }

    #[inline]
    fn value_sgte(a: u64, b: u64) -> u64 {
        let a_signed = a as i64;
        let b_signed = b as i64;
        if a_signed >= b_signed { 1 } else { 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provenance_add() {
        let a = Value::new(5);
        let b = Value::new(3);
        let result = ProvenanceOps::value_add(a, b);
        assert_eq!(result.val, 8);
        assert!(result.prov.is_pure());
    }

    #[test]
    fn test_provenance_add_with_temporal() {
        let a = Value { val: 5, prov: Provenance::single(0) };
        let b = Value::new(3);
        let result = ProvenanceOps::value_add(a, b);
        assert_eq!(result.val, 8);
        assert!(result.prov.is_temporal());
    }

    #[test]
    fn test_pure_add() {
        let result = PureOps::value_add(5, 3);
        assert_eq!(result, 8);
    }

    #[test]
    fn test_pure_div_by_zero() {
        let result = PureOps::value_div(10, 0);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_provenance_comparison() {
        let a = Value::new(5);
        let b = Value::new(3);
        let result = ProvenanceOps::value_lt(a, b);
        assert_eq!(result.val, 0); // 5 < 3 is false

        let a = Value::new(3);
        let b = Value::new(5);
        let result = ProvenanceOps::value_lt(a, b);
        assert_eq!(result.val, 1); // 3 < 5 is true
    }

    #[test]
    fn test_signed_comparison() {
        // Test with negative numbers (as u64)
        let neg_one = (-1i64) as u64;
        let one = 1u64;

        // Unsigned: -1 (as u64::MAX) > 1
        assert_eq!(PureOps::value_gt(neg_one, one), 1);

        // Signed: -1 < 1
        assert_eq!(PureOps::value_slt(neg_one, one), 1);
        assert_eq!(PureOps::value_sgt(neg_one, one), 0);
    }
}
