//! Core types for the OUROCHRONOS virtual machine.
//!
//! This module defines the fundamental data types:
//! - Value: 64-bit unsigned integers with provenance tracking
//! - Address: 16-bit memory indices
//! - Memory: The memory state (65536 cells)
//!
//! # Safety Properties
//!
//! Memory operations provide both unchecked (fast) and checked (safe) variants.
//! The bounds-checked operations return `OuroResult` and respect the configured
//! `BoundsPolicy` for handling out-of-bounds access.

use crate::provenance::Provenance;
use crate::error::{OuroError, OuroResult, BoundsPolicy, DivisionByZeroPolicy, MemoryOperation, SourceLocation};
use std::ops::{Add, Sub, Mul, Div, Rem, Not, BitAnd, BitOr, BitXor};
use std::fmt;

/// Memory address (16-bit index).
pub type Address = u16;

/// The size of the memory space (2^16 = 65536 cells).
pub const MEMORY_SIZE: usize = 65536;

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
    pub fn checked_div(self, rhs: Self, policy: DivisionByZeroPolicy) -> OuroResult<Self> {
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
                    location: SourceLocation::default(),
                }),
            }
        } else {
            Ok(Value {
                val: self.val.wrapping_div(rhs.val),
                prov: self.prov.merge(&rhs.prov),
            })
        }
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
    pub fn checked_mod(self, rhs: Self, policy: DivisionByZeroPolicy) -> OuroResult<Self> {
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
                    location: SourceLocation::default(),
                }),
            }
        } else {
            Ok(Value {
                val: self.val.wrapping_rem(rhs.val),
                prov: self.prov.merge(&rhs.prov),
            })
        }
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

// ═══════════════════════════════════════════════════════════════════════════
// Stack Type
// ═══════════════════════════════════════════════════════════════════════════

/// A bounds-checked stack for the OUROCHRONOS virtual machine.
///
/// Provides safe stack operations with configurable limits and error handling.
#[derive(Clone, Default)]
pub struct Stack {
    elements: Vec<Value>,
    max_depth: usize,
}

impl Stack {
    /// Create a new empty stack with unlimited depth.
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            max_depth: 0, // 0 means unlimited
        }
    }

    /// Create a stack with a maximum depth limit.
    pub fn with_max_depth(max_depth: usize) -> Self {
        Self {
            elements: Vec::new(),
            max_depth,
        }
    }

    /// Get the current depth of the stack.
    #[inline]
    pub fn depth(&self) -> usize {
        self.elements.len()
    }

    /// Check if the stack is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Push a value onto the stack with overflow checking.
    pub fn push_checked(&mut self, value: Value, location: SourceLocation) -> OuroResult<()> {
        if self.max_depth > 0 && self.elements.len() >= self.max_depth {
            return Err(OuroError::StackOverflow {
                max_depth: self.max_depth,
                location,
            });
        }
        self.elements.push(value);
        Ok(())
    }

    /// Push a value without checking (for performance).
    #[inline]
    pub fn push(&mut self, value: Value) {
        self.elements.push(value);
    }

    /// Pop a value with underflow checking.
    pub fn pop_checked(&mut self, operation: &str, location: SourceLocation) -> OuroResult<Value> {
        self.elements.pop().ok_or_else(|| OuroError::StackUnderflow {
            operation: operation.to_string(),
            required: 1,
            available: 0,
            location,
        })
    }

    /// Pop a value, returning zero if empty (permissive mode).
    #[inline]
    pub fn pop_or_zero(&mut self) -> Value {
        self.elements.pop().unwrap_or(Value::ZERO)
    }

    /// Pop a value, returning None if empty.
    #[inline]
    pub fn pop(&mut self) -> Option<Value> {
        self.elements.pop()
    }

    /// Peek at the top value with underflow checking.
    pub fn peek_checked(&self, operation: &str, location: SourceLocation) -> OuroResult<Value> {
        self.elements.last().cloned().ok_or_else(|| OuroError::StackUnderflow {
            operation: operation.to_string(),
            required: 1,
            available: 0,
            location,
        })
    }

    /// Peek at the top value, returning None if empty.
    #[inline]
    pub fn peek(&self) -> Option<&Value> {
        self.elements.last()
    }

    /// Ensure at least n elements are on the stack.
    pub fn require(&self, n: usize, operation: &str, location: SourceLocation) -> OuroResult<()> {
        if self.elements.len() < n {
            Err(OuroError::StackUnderflow {
                operation: operation.to_string(),
                required: n,
                available: self.elements.len(),
                location,
            })
        } else {
            Ok(())
        }
    }

    /// DUP with bounds checking.
    pub fn dup_checked(&mut self, location: SourceLocation) -> OuroResult<()> {
        self.require(1, "DUP", location.clone())?;
        let val = self.elements.last().unwrap().clone();
        self.push_checked(val, location)
    }

    /// SWAP with bounds checking.
    pub fn swap_checked(&mut self, location: SourceLocation) -> OuroResult<()> {
        self.require(2, "SWAP", location)?;
        let len = self.elements.len();
        self.elements.swap(len - 1, len - 2);
        Ok(())
    }

    /// OVER with bounds checking.
    pub fn over_checked(&mut self, location: SourceLocation) -> OuroResult<()> {
        self.require(2, "OVER", location.clone())?;
        let val = self.elements[self.elements.len() - 2].clone();
        self.push_checked(val, location)
    }

    /// ROT with bounds checking: ( a b c -- b c a )
    pub fn rot_checked(&mut self, location: SourceLocation) -> OuroResult<()> {
        self.require(3, "ROT", location)?;
        let len = self.elements.len();
        let a = self.elements.remove(len - 3);
        self.elements.push(a);
        Ok(())
    }

    /// PICK with bounds checking: copy nth element to top.
    pub fn pick_checked(&mut self, n: usize, location: SourceLocation) -> OuroResult<()> {
        let len = self.elements.len();
        if n >= len {
            return Err(OuroError::StackUnderflow {
                operation: format!("PICK {}", n),
                required: n + 1,
                available: len,
                location,
            });
        }
        let val = self.elements[len - 1 - n].clone();
        self.push_checked(val, location)
    }

    /// ROLL with bounds checking: move nth element to top.
    pub fn roll_checked(&mut self, n: usize, location: SourceLocation) -> OuroResult<()> {
        let len = self.elements.len();
        if n >= len {
            return Err(OuroError::StackUnderflow {
                operation: format!("ROLL {}", n),
                required: n + 1,
                available: len,
                location,
            });
        }
        let val = self.elements.remove(len - 1 - n);
        self.elements.push(val);
        Ok(())
    }

    /// REVERSE with bounds checking: reverse top n elements.
    pub fn reverse_checked(&mut self, n: usize, location: SourceLocation) -> OuroResult<()> {
        if n > self.elements.len() {
            return Err(OuroError::StackUnderflow {
                operation: format!("REVERSE {}", n),
                required: n,
                available: self.elements.len(),
                location,
            });
        }
        if n > 1 {
            let start = self.elements.len() - n;
            self.elements[start..].reverse();
        }
        Ok(())
    }

    /// Pop n values and return them as a vector.
    pub fn pop_n(&mut self, n: usize, operation: &str, location: SourceLocation) -> OuroResult<Vec<Value>> {
        if n > self.elements.len() {
            return Err(OuroError::StackUnderflow {
                operation: operation.to_string(),
                required: n,
                available: self.elements.len(),
                location,
            });
        }
        let start = self.elements.len() - n;
        Ok(self.elements.split_off(start))
    }

    /// Get a reference to the underlying elements.
    pub fn as_slice(&self) -> &[Value] {
        &self.elements
    }

    /// Get a mutable reference to the underlying elements.
    pub fn as_mut_slice(&mut self) -> &mut [Value] {
        &mut self.elements
    }

    /// Clear the stack.
    pub fn clear(&mut self) {
        self.elements.clear();
    }
}

impl std::fmt::Debug for Stack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Stack{:?}", self.elements)
    }
}

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
    #[inline]
    fn hash_mix(addr: Address, val: u64) -> u64 {
        if val == 0 {
            return 0; // Zero values contribute nothing to hash
        }
        // Use a variant of FxHash mixing
        let mut h = val.wrapping_mul(0x517cc1b727220a95);
        h = h.wrapping_add(addr as u64);
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
    }
    
    /// Check if two memory states are equal (by value, ignoring provenance).
    /// This is the fixed-point check: P_final = A_initial.
    pub fn values_equal(&self, other: &Memory) -> bool {
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
    pub fn is_empty(&self) -> bool {
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

// ═══════════════════════════════════════════════════════════════════════════
// Data Structure Storage
// ═══════════════════════════════════════════════════════════════════════════

use std::collections::{HashMap, HashSet};

/// Handle type for referencing data structures.
///
/// Data structures are stored externally and referenced by handles.
/// This keeps the core Value type as a simple u64 while allowing
/// complex data structures to be manipulated.
pub type Handle = u64;

/// Storage for dynamically allocated vectors.
///
/// Vectors are identified by handles (indices into this storage).
/// Each vector can grow dynamically and supports random access.
#[derive(Clone, Default, Debug)]
pub struct VecStore {
    /// Storage for vectors, indexed by handle.
    vectors: Vec<Vec<Value>>,
}

impl VecStore {
    /// Create a new empty vector store.
    pub fn new() -> Self {
        Self { vectors: Vec::new() }
    }

    /// Allocate a new empty vector and return its handle.
    pub fn alloc(&mut self) -> Handle {
        let handle = self.vectors.len() as Handle;
        self.vectors.push(Vec::new());
        handle
    }

    /// Check if a handle is valid.
    #[inline]
    pub fn is_valid(&self, handle: Handle) -> bool {
        (handle as usize) < self.vectors.len()
    }

    /// Get a reference to a vector by handle.
    pub fn get(&self, handle: Handle) -> Option<&Vec<Value>> {
        self.vectors.get(handle as usize)
    }

    /// Get a mutable reference to a vector by handle.
    pub fn get_mut(&mut self, handle: Handle) -> Option<&mut Vec<Value>> {
        self.vectors.get_mut(handle as usize)
    }

    /// Push a value onto a vector.
    pub fn push(&mut self, handle: Handle, value: Value) -> OuroResult<()> {
        if let Some(vec) = self.vectors.get_mut(handle as usize) {
            vec.push(value);
            Ok(())
        } else {
            Err(OuroError::InvalidHandle {
                handle_type: "vector".to_string(),
                handle,
                max_handle: self.vectors.len() as u64,
                location: SourceLocation::default(),
            })
        }
    }

    /// Pop a value from a vector.
    pub fn pop(&mut self, handle: Handle) -> OuroResult<Value> {
        if let Some(vec) = self.vectors.get_mut(handle as usize) {
            vec.pop().ok_or_else(|| OuroError::EmptyStructure {
                structure_type: "vector".to_string(),
                operation: "pop from".to_string(),
                location: SourceLocation::default(),
            })
        } else {
            Err(OuroError::InvalidHandle {
                handle_type: "vector".to_string(),
                handle,
                max_handle: self.vectors.len() as u64,
                location: SourceLocation::default(),
            })
        }
    }

    /// Get a value at index from a vector.
    pub fn get_at(&self, handle: Handle, index: u64) -> OuroResult<Value> {
        if let Some(vec) = self.vectors.get(handle as usize) {
            if (index as usize) < vec.len() {
                Ok(vec[index as usize].clone())
            } else {
                Err(OuroError::IndexOutOfBounds {
                    structure_type: "vector".to_string(),
                    index,
                    length: vec.len() as u64,
                    location: SourceLocation::default(),
                })
            }
        } else {
            Err(OuroError::InvalidHandle {
                handle_type: "vector".to_string(),
                handle,
                max_handle: self.vectors.len() as u64,
                location: SourceLocation::default(),
            })
        }
    }

    /// Set a value at index in a vector.
    pub fn set_at(&mut self, handle: Handle, index: u64, value: Value) -> OuroResult<()> {
        if let Some(vec) = self.vectors.get_mut(handle as usize) {
            if (index as usize) < vec.len() {
                vec[index as usize] = value;
                Ok(())
            } else {
                Err(OuroError::IndexOutOfBounds {
                    structure_type: "vector".to_string(),
                    index,
                    length: vec.len() as u64,
                    location: SourceLocation::default(),
                })
            }
        } else {
            Err(OuroError::InvalidHandle {
                handle_type: "vector".to_string(),
                handle,
                max_handle: self.vectors.len() as u64,
                location: SourceLocation::default(),
            })
        }
    }

    /// Get the length of a vector.
    pub fn len(&self, handle: Handle) -> OuroResult<u64> {
        if let Some(vec) = self.vectors.get(handle as usize) {
            Ok(vec.len() as u64)
        } else {
            Err(OuroError::InvalidHandle {
                handle_type: "vector".to_string(),
                handle,
                max_handle: self.vectors.len() as u64,
                location: SourceLocation::default(),
            })
        }
    }

    /// Get the total number of vectors allocated.
    pub fn count(&self) -> usize {
        self.vectors.len()
    }
}

/// Storage for dynamically allocated hash tables.
///
/// Hash tables map u64 keys to Values.
/// Operations are O(1) average case.
#[derive(Clone, Default, Debug)]
pub struct HashStore {
    /// Storage for hash tables, indexed by handle.
    tables: Vec<HashMap<u64, Value>>,
}

impl HashStore {
    /// Create a new empty hash store.
    pub fn new() -> Self {
        Self { tables: Vec::new() }
    }

    /// Allocate a new empty hash table and return its handle.
    pub fn alloc(&mut self) -> Handle {
        let handle = self.tables.len() as Handle;
        self.tables.push(HashMap::new());
        handle
    }

    /// Check if a handle is valid.
    #[inline]
    pub fn is_valid(&self, handle: Handle) -> bool {
        (handle as usize) < self.tables.len()
    }

    /// Get a reference to a hash table by handle.
    pub fn get(&self, handle: Handle) -> Option<&HashMap<u64, Value>> {
        self.tables.get(handle as usize)
    }

    /// Get a mutable reference to a hash table by handle.
    pub fn get_mut(&mut self, handle: Handle) -> Option<&mut HashMap<u64, Value>> {
        self.tables.get_mut(handle as usize)
    }

    /// Insert or update a key-value pair in a hash table.
    pub fn put(&mut self, handle: Handle, key: u64, value: Value) -> OuroResult<()> {
        if let Some(table) = self.tables.get_mut(handle as usize) {
            table.insert(key, value);
            Ok(())
        } else {
            Err(OuroError::InvalidHandle {
                handle_type: "hash table".to_string(),
                handle,
                max_handle: self.tables.len() as u64,
                location: SourceLocation::default(),
            })
        }
    }

    /// Get a value by key from a hash table.
    /// Returns (value, found) where found is true if the key exists.
    pub fn get_key(&self, handle: Handle, key: u64) -> OuroResult<(Value, bool)> {
        if let Some(table) = self.tables.get(handle as usize) {
            match table.get(&key) {
                Some(value) => Ok((value.clone(), true)),
                None => Ok((Value::ZERO, false)),
            }
        } else {
            Err(OuroError::InvalidHandle {
                handle_type: "hash table".to_string(),
                handle,
                max_handle: self.tables.len() as u64,
                location: SourceLocation::default(),
            })
        }
    }

    /// Delete a key from a hash table.
    pub fn delete(&mut self, handle: Handle, key: u64) -> OuroResult<()> {
        if let Some(table) = self.tables.get_mut(handle as usize) {
            table.remove(&key);
            Ok(())
        } else {
            Err(OuroError::InvalidHandle {
                handle_type: "hash table".to_string(),
                handle,
                max_handle: self.tables.len() as u64,
                location: SourceLocation::default(),
            })
        }
    }

    /// Check if a key exists in a hash table.
    pub fn has(&self, handle: Handle, key: u64) -> OuroResult<bool> {
        if let Some(table) = self.tables.get(handle as usize) {
            Ok(table.contains_key(&key))
        } else {
            Err(OuroError::InvalidHandle {
                handle_type: "hash table".to_string(),
                handle,
                max_handle: self.tables.len() as u64,
                location: SourceLocation::default(),
            })
        }
    }

    /// Get the number of entries in a hash table.
    pub fn len(&self, handle: Handle) -> OuroResult<u64> {
        if let Some(table) = self.tables.get(handle as usize) {
            Ok(table.len() as u64)
        } else {
            Err(OuroError::InvalidHandle {
                handle_type: "hash table".to_string(),
                handle,
                max_handle: self.tables.len() as u64,
                location: SourceLocation::default(),
            })
        }
    }

    /// Get the total number of hash tables allocated.
    pub fn count(&self) -> usize {
        self.tables.len()
    }
}

/// Storage for dynamically allocated sets.
///
/// Sets store unique u64 values.
/// Operations are O(1) average case.
#[derive(Clone, Default, Debug)]
pub struct SetStore {
    /// Storage for sets, indexed by handle.
    sets: Vec<HashSet<u64>>,
}

impl SetStore {
    /// Create a new empty set store.
    pub fn new() -> Self {
        Self { sets: Vec::new() }
    }

    /// Allocate a new empty set and return its handle.
    pub fn alloc(&mut self) -> Handle {
        let handle = self.sets.len() as Handle;
        self.sets.push(HashSet::new());
        handle
    }

    /// Check if a handle is valid.
    #[inline]
    pub fn is_valid(&self, handle: Handle) -> bool {
        (handle as usize) < self.sets.len()
    }

    /// Get a reference to a set by handle.
    pub fn get(&self, handle: Handle) -> Option<&HashSet<u64>> {
        self.sets.get(handle as usize)
    }

    /// Get a mutable reference to a set by handle.
    pub fn get_mut(&mut self, handle: Handle) -> Option<&mut HashSet<u64>> {
        self.sets.get_mut(handle as usize)
    }

    /// Add a value to a set.
    pub fn add(&mut self, handle: Handle, value: u64) -> OuroResult<()> {
        if let Some(set) = self.sets.get_mut(handle as usize) {
            set.insert(value);
            Ok(())
        } else {
            Err(OuroError::InvalidHandle {
                handle_type: "set".to_string(),
                handle,
                max_handle: self.sets.len() as u64,
                location: SourceLocation::default(),
            })
        }
    }

    /// Check if a value exists in a set.
    pub fn has(&self, handle: Handle, value: u64) -> OuroResult<bool> {
        if let Some(set) = self.sets.get(handle as usize) {
            Ok(set.contains(&value))
        } else {
            Err(OuroError::InvalidHandle {
                handle_type: "set".to_string(),
                handle,
                max_handle: self.sets.len() as u64,
                location: SourceLocation::default(),
            })
        }
    }

    /// Remove a value from a set.
    pub fn delete(&mut self, handle: Handle, value: u64) -> OuroResult<()> {
        if let Some(set) = self.sets.get_mut(handle as usize) {
            set.remove(&value);
            Ok(())
        } else {
            Err(OuroError::InvalidHandle {
                handle_type: "set".to_string(),
                handle,
                max_handle: self.sets.len() as u64,
                location: SourceLocation::default(),
            })
        }
    }

    /// Get the number of elements in a set.
    pub fn len(&self, handle: Handle) -> OuroResult<u64> {
        if let Some(set) = self.sets.get(handle as usize) {
            Ok(set.len() as u64)
        } else {
            Err(OuroError::InvalidHandle {
                handle_type: "set".to_string(),
                handle,
                max_handle: self.sets.len() as u64,
                location: SourceLocation::default(),
            })
        }
    }

    /// Get the total number of sets allocated.
    pub fn count(&self) -> usize {
        self.sets.len()
    }
}

/// Combined data structure storage for the VM.
///
/// This groups all dynamically allocated data structures together
/// for easy management and cloning during temporal operations.
#[derive(Clone, Default, Debug)]
pub struct DataStructures {
    /// Vector storage.
    pub vectors: VecStore,
    /// Hash table storage.
    pub hashes: HashStore,
    /// Set storage.
    pub sets: SetStore,
}

impl DataStructures {
    /// Create new empty data structure storage.
    pub fn new() -> Self {
        Self {
            vectors: VecStore::new(),
            hashes: HashStore::new(),
            sets: SetStore::new(),
        }
    }

    /// Clear all data structures (for epoch reset).
    pub fn clear(&mut self) {
        self.vectors = VecStore::new();
        self.hashes = HashStore::new();
        self.sets = SetStore::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{BoundsPolicy, DivisionByZeroPolicy, MemoryOperation, SourceLocation, OuroError};

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

    // ═══════════════════════════════════════════════════════════════════
    // Bounds-Checked Operation Tests
    // ═══════════════════════════════════════════════════════════════════

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
        match result.unwrap_err() {
            OuroError::MemoryBoundsViolation { address, .. } => {
                assert_eq!(address, 65536);
            }
            _ => panic!("Expected MemoryBoundsViolation"),
        }

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

    // ═══════════════════════════════════════════════════════════════════
    // Configurable Division Tests
    // ═══════════════════════════════════════════════════════════════════

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
        match result.unwrap_err() {
            OuroError::DivisionByZero { dividend, .. } => {
                assert_eq!(dividend, 42);
            }
            _ => panic!("Expected DivisionByZero error"),
        }
    }

    #[test]
    fn test_checked_div_sentinel_policy() {
        let a = Value::new(42);
        let zero = Value::new(0);

        let result = a.checked_div(zero, DivisionByZeroPolicy::Sentinel(u64::MAX));
        assert!(result.is_ok());
        assert_eq!(result.unwrap().val, u64::MAX);
    }

    #[test]
    fn test_checked_div_normal_case() {
        let a = Value::new(42);
        let b = Value::new(7);

        let result = a.checked_div(b, DivisionByZeroPolicy::Error);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().val, 6);
    }

    #[test]
    fn test_checked_mod_return_zero_policy() {
        let a = Value::new(42);
        let zero = Value::new(0);

        let result = a.checked_mod(zero, DivisionByZeroPolicy::ReturnZero);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().val, 0);
    }

    #[test]
    fn test_checked_mod_error_policy() {
        let a = Value::new(42);
        let zero = Value::new(0);

        let result = a.checked_mod(zero, DivisionByZeroPolicy::Error);
        assert!(result.is_err());
        match result.unwrap_err() {
            OuroError::ModuloByZero { dividend, .. } => {
                assert_eq!(dividend, 42);
            }
            _ => panic!("Expected ModuloByZero error"),
        }
    }

    #[test]
    fn test_checked_mod_normal_case() {
        let a = Value::new(42);
        let b = Value::new(5);

        let result = a.checked_mod(b, DivisionByZeroPolicy::Error);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().val, 2); // 42 % 5 = 2
    }

    // ═══════════════════════════════════════════════════════════════════
    // Signed Integer Tests
    // ═══════════════════════════════════════════════════════════════════

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

    // ═══════════════════════════════════════════════════════════════════
    // Stack Tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_stack_basic_operations() {
        let mut stack = Stack::new();
        let loc = SourceLocation::default();

        // Empty stack
        assert!(stack.is_empty());
        assert_eq!(stack.depth(), 0);

        // Push and pop
        stack.push(Value::new(42));
        assert_eq!(stack.depth(), 1);
        assert!(!stack.is_empty());

        let val = stack.pop_checked("test", loc.clone()).unwrap();
        assert_eq!(val.val, 42);
        assert!(stack.is_empty());

        // Pop from empty should error
        let result = stack.pop_checked("test", loc);
        assert!(result.is_err());
    }

    #[test]
    fn test_stack_with_max_depth() {
        let mut stack = Stack::with_max_depth(3);
        let loc = SourceLocation::default();

        // Push up to limit
        stack.push_checked(Value::new(1), loc.clone()).unwrap();
        stack.push_checked(Value::new(2), loc.clone()).unwrap();
        stack.push_checked(Value::new(3), loc.clone()).unwrap();

        // Next push should fail
        let result = stack.push_checked(Value::new(4), loc);
        assert!(result.is_err());
        match result.unwrap_err() {
            OuroError::StackOverflow { max_depth, .. } => {
                assert_eq!(max_depth, 3);
            }
            _ => panic!("Expected StackOverflow"),
        }
    }

    #[test]
    fn test_stack_dup_swap_over() {
        let mut stack = Stack::new();
        let loc = SourceLocation::default();

        stack.push(Value::new(10));
        stack.push(Value::new(20));

        // DUP
        stack.dup_checked(loc.clone()).unwrap();
        assert_eq!(stack.depth(), 3);
        assert_eq!(stack.peek().unwrap().val, 20);

        // SWAP
        stack.pop().unwrap();
        stack.swap_checked(loc.clone()).unwrap();
        assert_eq!(stack.pop().unwrap().val, 10);
        assert_eq!(stack.pop().unwrap().val, 20);

        // OVER
        stack.push(Value::new(1));
        stack.push(Value::new(2));
        stack.over_checked(loc).unwrap();
        assert_eq!(stack.pop().unwrap().val, 1);
    }

    #[test]
    fn test_stack_rot() {
        let mut stack = Stack::new();
        let loc = SourceLocation::default();

        stack.push(Value::new(1)); // bottom
        stack.push(Value::new(2));
        stack.push(Value::new(3)); // top

        // ROT: (1 2 3 -- 2 3 1)
        stack.rot_checked(loc).unwrap();

        assert_eq!(stack.pop().unwrap().val, 1);
        assert_eq!(stack.pop().unwrap().val, 3);
        assert_eq!(stack.pop().unwrap().val, 2);
    }

    #[test]
    fn test_stack_pick() {
        let mut stack = Stack::new();
        let loc = SourceLocation::default();

        stack.push(Value::new(100));
        stack.push(Value::new(200));
        stack.push(Value::new(300));

        // 0 PICK = DUP
        stack.pick_checked(0, loc.clone()).unwrap();
        assert_eq!(stack.pop().unwrap().val, 300);

        // 2 PICK = copy third from top
        stack.pick_checked(2, loc.clone()).unwrap();
        assert_eq!(stack.pop().unwrap().val, 100);

        // Out of bounds
        let result = stack.pick_checked(10, loc);
        assert!(result.is_err());
    }

    #[test]
    fn test_stack_roll() {
        let mut stack = Stack::new();
        let loc = SourceLocation::default();

        stack.push(Value::new(1));
        stack.push(Value::new(2));
        stack.push(Value::new(3));
        stack.push(Value::new(4));

        // 2 ROLL: move third from top to top
        // (1 2 3 4) -> (1 3 4 2)
        stack.roll_checked(2, loc).unwrap();

        assert_eq!(stack.pop().unwrap().val, 2);
        assert_eq!(stack.pop().unwrap().val, 4);
        assert_eq!(stack.pop().unwrap().val, 3);
        assert_eq!(stack.pop().unwrap().val, 1);
    }

    #[test]
    fn test_stack_reverse() {
        let mut stack = Stack::new();
        let loc = SourceLocation::default();

        stack.push(Value::new(1));
        stack.push(Value::new(2));
        stack.push(Value::new(3));
        stack.push(Value::new(4));

        // Reverse top 3
        stack.reverse_checked(3, loc).unwrap();

        assert_eq!(stack.pop().unwrap().val, 2);
        assert_eq!(stack.pop().unwrap().val, 3);
        assert_eq!(stack.pop().unwrap().val, 4);
        assert_eq!(stack.pop().unwrap().val, 1);
    }

    #[test]
    fn test_stack_pop_n() {
        let mut stack = Stack::new();
        let loc = SourceLocation::default();

        stack.push(Value::new(1));
        stack.push(Value::new(2));
        stack.push(Value::new(3));

        let values = stack.pop_n(2, "test", loc.clone()).unwrap();
        assert_eq!(values.len(), 2);
        assert_eq!(values[0].val, 2);
        assert_eq!(values[1].val, 3);
        assert_eq!(stack.depth(), 1);

        // Pop more than available should fail
        let result = stack.pop_n(5, "test", loc);
        assert!(result.is_err());
    }
}
