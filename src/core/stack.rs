//! Stack type for the OUROCHRONOS virtual machine.
//!
//! Provides safe stack operations with configurable limits and error handling.

use std::fmt;
use super::value::Value;
use super::error::{OuroError, OuroResult, SourceLocation};

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

impl fmt::Debug for Stack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Stack{:?}", self.elements)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
