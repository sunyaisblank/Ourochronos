//! Data structure storage for the OUROCHRONOS virtual machine.
//!
//! Provides dynamically allocated vectors, hash tables, and sets
//! that are referenced by handles from the stack.

use std::collections::{HashMap, HashSet};
use super::address::Handle;
use super::value::Value;
use super::error::{OuroError, OuroResult};

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

    /// Validate a handle and return an error if invalid.
    #[inline]
    fn validate(&self, handle: Handle) -> OuroResult<usize> {
        let idx = handle as usize;
        if idx < self.vectors.len() {
            Ok(idx)
        } else {
            Err(OuroError::invalid_handle("vector", handle, self.vectors.len() as u64))
        }
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
        let idx = self.validate(handle)?;
        self.vectors[idx].push(value);
        Ok(())
    }

    /// Pop a value from a vector.
    pub fn pop(&mut self, handle: Handle) -> OuroResult<Value> {
        let idx = self.validate(handle)?;
        self.vectors[idx].pop().ok_or_else(|| OuroError::empty_structure("vector", "pop from"))
    }

    /// Get a value at index from a vector.
    pub fn get_at(&self, handle: Handle, index: u64) -> OuroResult<Value> {
        let idx = self.validate(handle)?;
        let vec = &self.vectors[idx];
        if (index as usize) < vec.len() {
            Ok(vec[index as usize].clone())
        } else {
            Err(OuroError::index_out_of_bounds("vector", index, vec.len() as u64))
        }
    }

    /// Set a value at index in a vector.
    pub fn set_at(&mut self, handle: Handle, index: u64, value: Value) -> OuroResult<()> {
        let idx = self.validate(handle)?;
        let vec = &mut self.vectors[idx];
        if (index as usize) < vec.len() {
            vec[index as usize] = value;
            Ok(())
        } else {
            Err(OuroError::index_out_of_bounds("vector", index, vec.len() as u64))
        }
    }

    /// Get the length of a vector.
    pub fn len(&self, handle: Handle) -> OuroResult<u64> {
        let idx = self.validate(handle)?;
        Ok(self.vectors[idx].len() as u64)
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

    /// Validate a handle and return an error if invalid.
    #[inline]
    fn validate(&self, handle: Handle) -> OuroResult<usize> {
        let idx = handle as usize;
        if idx < self.tables.len() {
            Ok(idx)
        } else {
            Err(OuroError::invalid_handle("hash table", handle, self.tables.len() as u64))
        }
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
        let idx = self.validate(handle)?;
        self.tables[idx].insert(key, value);
        Ok(())
    }

    /// Get a value by key from a hash table.
    /// Returns (value, found) where found is true if the key exists.
    pub fn get_key(&self, handle: Handle, key: u64) -> OuroResult<(Value, bool)> {
        let idx = self.validate(handle)?;
        match self.tables[idx].get(&key) {
            Some(value) => Ok((value.clone(), true)),
            None => Ok((Value::ZERO, false)),
        }
    }

    /// Delete a key from a hash table.
    pub fn delete(&mut self, handle: Handle, key: u64) -> OuroResult<()> {
        let idx = self.validate(handle)?;
        self.tables[idx].remove(&key);
        Ok(())
    }

    /// Check if a key exists in a hash table.
    pub fn has(&self, handle: Handle, key: u64) -> OuroResult<bool> {
        let idx = self.validate(handle)?;
        Ok(self.tables[idx].contains_key(&key))
    }

    /// Get the number of entries in a hash table.
    pub fn len(&self, handle: Handle) -> OuroResult<u64> {
        let idx = self.validate(handle)?;
        Ok(self.tables[idx].len() as u64)
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

    /// Validate a handle and return an error if invalid.
    #[inline]
    fn validate(&self, handle: Handle) -> OuroResult<usize> {
        let idx = handle as usize;
        if idx < self.sets.len() {
            Ok(idx)
        } else {
            Err(OuroError::invalid_handle("set", handle, self.sets.len() as u64))
        }
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
        let idx = self.validate(handle)?;
        self.sets[idx].insert(value);
        Ok(())
    }

    /// Check if a value exists in a set.
    pub fn has(&self, handle: Handle, value: u64) -> OuroResult<bool> {
        let idx = self.validate(handle)?;
        Ok(self.sets[idx].contains(&value))
    }

    /// Remove a value from a set.
    pub fn delete(&mut self, handle: Handle, value: u64) -> OuroResult<()> {
        let idx = self.validate(handle)?;
        self.sets[idx].remove(&value);
        Ok(())
    }

    /// Get the number of elements in a set.
    pub fn len(&self, handle: Handle) -> OuroResult<u64> {
        let idx = self.validate(handle)?;
        Ok(self.sets[idx].len() as u64)
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

    // ═══════════════════════════════════════════════════════════════════
    // VecStore Tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_vecstore_basic_operations() {
        let mut store = VecStore::new();

        let h = store.alloc();
        assert!(store.is_valid(h));
        assert_eq!(store.len(h).unwrap(), 0);

        store.push(h, Value::new(42)).unwrap();
        assert_eq!(store.len(h).unwrap(), 1);
        assert_eq!(store.get_at(h, 0).unwrap().val, 42);

        store.set_at(h, 0, Value::new(99)).unwrap();
        assert_eq!(store.get_at(h, 0).unwrap().val, 99);

        let popped = store.pop(h).unwrap();
        assert_eq!(popped.val, 99);
        assert_eq!(store.len(h).unwrap(), 0);
    }

    #[test]
    fn test_vecstore_invalid_handle_error() {
        let store = VecStore::new();

        // Access with invalid handle should fail
        let result = store.len(999);
        assert!(result.is_err());

        if let Err(OuroError::InvalidHandle { handle_type, handle, max_handle, .. }) = result {
            assert_eq!(handle_type, "vector");
            assert_eq!(handle, 999);
            assert_eq!(max_handle, 0);
        } else {
            panic!("Expected InvalidHandle error");
        }
    }

    #[test]
    fn test_vecstore_index_out_of_bounds_error() {
        let mut store = VecStore::new();
        let h = store.alloc();
        store.push(h, Value::new(1)).unwrap();
        store.push(h, Value::new(2)).unwrap();

        // Access out of bounds
        let result = store.get_at(h, 10);
        assert!(result.is_err());

        if let Err(OuroError::IndexOutOfBounds { structure_type, index, length, .. }) = result {
            assert_eq!(structure_type, "vector");
            assert_eq!(index, 10);
            assert_eq!(length, 2);
        } else {
            panic!("Expected IndexOutOfBounds error");
        }
    }

    #[test]
    fn test_vecstore_empty_structure_error() {
        let mut store = VecStore::new();
        let h = store.alloc();

        // Pop from empty vector
        let result = store.pop(h);
        assert!(result.is_err());

        if let Err(OuroError::EmptyStructure { structure_type, operation, .. }) = result {
            assert_eq!(structure_type, "vector");
            assert_eq!(operation, "pop from");
        } else {
            panic!("Expected EmptyStructure error");
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // HashStore Tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_hashstore_basic_operations() {
        let mut store = HashStore::new();

        let h = store.alloc();
        assert!(store.is_valid(h));
        assert_eq!(store.len(h).unwrap(), 0);

        store.put(h, 42, Value::new(100)).unwrap();
        assert_eq!(store.len(h).unwrap(), 1);
        assert!(store.has(h, 42).unwrap());

        let (val, found) = store.get_key(h, 42).unwrap();
        assert!(found);
        assert_eq!(val.val, 100);

        let (val, found) = store.get_key(h, 999).unwrap();
        assert!(!found);
        assert_eq!(val.val, 0);

        store.delete(h, 42).unwrap();
        assert!(!store.has(h, 42).unwrap());
    }

    #[test]
    fn test_hashstore_invalid_handle_error() {
        let store = HashStore::new();

        let result = store.len(999);
        assert!(result.is_err());

        if let Err(OuroError::InvalidHandle { handle_type, handle, max_handle, .. }) = result {
            assert_eq!(handle_type, "hash table");
            assert_eq!(handle, 999);
            assert_eq!(max_handle, 0);
        } else {
            panic!("Expected InvalidHandle error");
        }
    }

    #[test]
    fn test_hashstore_put_invalid_handle() {
        let mut store = HashStore::new();

        let result = store.put(999, 1, Value::new(42));
        assert!(result.is_err());

        if let Err(OuroError::InvalidHandle { handle_type, handle, .. }) = result {
            assert_eq!(handle_type, "hash table");
            assert_eq!(handle, 999);
        } else {
            panic!("Expected InvalidHandle error");
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // SetStore Tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_setstore_basic_operations() {
        let mut store = SetStore::new();

        let h = store.alloc();
        assert!(store.is_valid(h));
        assert_eq!(store.len(h).unwrap(), 0);

        store.add(h, 42).unwrap();
        assert_eq!(store.len(h).unwrap(), 1);
        assert!(store.has(h, 42).unwrap());
        assert!(!store.has(h, 999).unwrap());

        // Adding same value again shouldn't increase length
        store.add(h, 42).unwrap();
        assert_eq!(store.len(h).unwrap(), 1);

        store.delete(h, 42).unwrap();
        assert!(!store.has(h, 42).unwrap());
        assert_eq!(store.len(h).unwrap(), 0);
    }

    #[test]
    fn test_setstore_invalid_handle_error() {
        let store = SetStore::new();

        let result = store.len(999);
        assert!(result.is_err());

        if let Err(OuroError::InvalidHandle { handle_type, handle, max_handle, .. }) = result {
            assert_eq!(handle_type, "set");
            assert_eq!(handle, 999);
            assert_eq!(max_handle, 0);
        } else {
            panic!("Expected InvalidHandle error");
        }
    }

    #[test]
    fn test_setstore_add_invalid_handle() {
        let mut store = SetStore::new();

        let result = store.add(999, 42);
        assert!(result.is_err());

        if let Err(OuroError::InvalidHandle { handle_type, handle, .. }) = result {
            assert_eq!(handle_type, "set");
            assert_eq!(handle, 999);
        } else {
            panic!("Expected InvalidHandle error");
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // DataStructures Combined Tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_data_structures_clear() {
        let mut ds = DataStructures::new();

        // Allocate some structures
        let vh = ds.vectors.alloc();
        let hh = ds.hashes.alloc();
        let sh = ds.sets.alloc();

        ds.vectors.push(vh, Value::new(42)).unwrap();
        ds.hashes.put(hh, 1, Value::new(100)).unwrap();
        ds.sets.add(sh, 999).unwrap();

        // Clear everything
        ds.clear();

        // Old handles should now be invalid
        assert!(!ds.vectors.is_valid(vh));
        assert!(!ds.hashes.is_valid(hh));
        assert!(!ds.sets.is_valid(sh));

        assert_eq!(ds.vectors.count(), 0);
        assert_eq!(ds.hashes.count(), 0);
        assert_eq!(ds.sets.count(), 0);
    }

    #[test]
    fn test_multiple_allocations() {
        let mut store = VecStore::new();

        let h1 = store.alloc();
        let h2 = store.alloc();
        let h3 = store.alloc();

        assert_eq!(h1, 0);
        assert_eq!(h2, 1);
        assert_eq!(h3, 2);

        store.push(h1, Value::new(1)).unwrap();
        store.push(h2, Value::new(2)).unwrap();
        store.push(h3, Value::new(3)).unwrap();

        assert_eq!(store.get_at(h1, 0).unwrap().val, 1);
        assert_eq!(store.get_at(h2, 0).unwrap().val, 2);
        assert_eq!(store.get_at(h3, 0).unwrap().val, 3);
    }

    #[test]
    fn test_set_at_out_of_bounds() {
        let mut store = VecStore::new();
        let h = store.alloc();
        store.push(h, Value::new(1)).unwrap();

        let result = store.set_at(h, 5, Value::new(99));
        assert!(result.is_err());

        if let Err(OuroError::IndexOutOfBounds { structure_type, index, length, .. }) = result {
            assert_eq!(structure_type, "vector");
            assert_eq!(index, 5);
            assert_eq!(length, 1);
        } else {
            panic!("Expected IndexOutOfBounds error");
        }
    }
}
