//! Data structure storage for the OUROCHRONOS virtual machine.
//!
//! Provides dynamically allocated vectors, hash tables, and sets
//! that are referenced by handles from the stack.

use std::collections::{HashMap, HashSet};
use super::address::Handle;
use super::value::Value;
use super::error::{OuroError, OuroResult, SourceLocation};

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
