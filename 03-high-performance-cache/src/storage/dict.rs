use seahash::hash;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const INITIAL_SIZE: usize = 4;
const RESIZE_RATIO: usize = 5;

/// Entry in the hash table
#[derive(Debug)]
struct Entry<K, V> {
    key: K,
    value: V,
    next: Option<Box<Entry<K, V>>>,
}

/// Hash table
#[derive(Debug)]
struct HashTable<K, V> {
    buckets: Vec<Option<Box<Entry<K, V>>>>,
    size: usize,
    mask: usize,
    used: usize,
}

impl<K, V> HashTable<K, V> {
    fn new() -> Self {
        Self {
            buckets: Vec::new(),
            size: 0,
            mask: 0,
            used: 0,
        }
    }

    fn with_size(size: usize) -> Self {
        let size = size.next_power_of_two();
        let buckets = (0..size).map(|_| None).collect();
        Self {
            buckets,
            size,
            mask: size - 1,
            used: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.used == 0
    }
}

/// Dictionary with incremental rehashing
#[derive(Debug)]
pub struct Dict<K, V> {
    tables: [HashTable<K, V>; 2],
    rehash_idx: Option<usize>,
}

impl<K: Hash + Eq + Clone, V> Dict<K, V> {
    /// Create a new dictionary
    pub fn new() -> Self {
        Self {
            tables: [HashTable::new(), HashTable::new()],
            rehash_idx: None,
        }
    }

    /// Check if rehashing is in progress
    pub fn is_rehashing(&self) -> bool {
        self.rehash_idx.is_some()
    }

    /// Get a value by key
    pub fn get(&self, key: &K) -> Option<&V> {
        if self.tables[0].size == 0 && self.tables[1].size == 0 {
            return None;
        }

        // Search both tables during rehashing
        for table_idx in 0..2 {
            if table_idx == 1 && !self.is_rehashing() {
                break;
            }

            let table = &self.tables[table_idx];
            if table.size == 0 {
                continue;
            }

            let hash = self.hash_key(key);
            let idx = hash as usize & table.mask;

            let mut entry = table.buckets[idx].as_ref();
            while let Some(e) = entry {
                if e.key == *key {
                    return Some(&e.value);
                }
                entry = e.next.as_ref();
            }
        }

        None
    }

    /// Get a mutable reference to a value
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        if self.is_rehashing() {
            self.rehash_step();
        }

        if self.tables[0].size == 0 && self.tables[1].size == 0 {
            return None;
        }

        // Compute hash before borrowing tables mutably
        let hash = self.hash_key(key);
        let is_rehashing = self.is_rehashing();

        // Determine which table has the key
        let mut found_table: Option<usize> = None;

        // Check table 0
        if self.tables[0].size > 0 {
            let idx = hash as usize & self.tables[0].mask;
            let mut entry = self.tables[0].buckets[idx].as_ref();
            while let Some(e) = entry {
                if e.key == *key {
                    found_table = Some(0);
                    break;
                }
                entry = e.next.as_ref();
            }
        }

        // Check table 1 if rehashing and not found in table 0
        if found_table.is_none() && is_rehashing && self.tables[1].size > 0 {
            let idx = hash as usize & self.tables[1].mask;
            let mut entry = self.tables[1].buckets[idx].as_ref();
            while let Some(e) = entry {
                if e.key == *key {
                    found_table = Some(1);
                    break;
                }
                entry = e.next.as_ref();
            }
        }

        // Now get mutable reference from the correct table
        match found_table {
            Some(0) => {
                let idx = hash as usize & self.tables[0].mask;
                let mut entry = self.tables[0].buckets[idx].as_mut();
                while let Some(e) = entry {
                    if e.key == *key {
                        return Some(&mut e.value);
                    }
                    entry = e.next.as_mut();
                }
                None
            }
            Some(1) => {
                let idx = hash as usize & self.tables[1].mask;
                let mut entry = self.tables[1].buckets[idx].as_mut();
                while let Some(e) = entry {
                    if e.key == *key {
                        return Some(&mut e.value);
                    }
                    entry = e.next.as_mut();
                }
                None
            }
            _ => None,
        }
    }

    /// Insert a key-value pair
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.expand_if_needed();

        if self.is_rehashing() {
            self.rehash_step();
        }

        // Compute hash before borrowing tables mutably
        let hash = self.hash_key(&key);
        let is_rehashing = self.is_rehashing();

        // Check if key exists
        for table_idx in 0..2 {
            if table_idx == 1 && !is_rehashing {
                break;
            }

            let table = &mut self.tables[table_idx];
            if table.size == 0 {
                continue;
            }

            let idx = hash as usize & table.mask;

            let mut entry = table.buckets[idx].as_mut();
            while let Some(e) = entry {
                if e.key == key {
                    return Some(std::mem::replace(&mut e.value, value));
                }
                entry = e.next.as_mut();
            }
        }

        // Insert into appropriate table
        let table_idx = if is_rehashing { 1 } else { 0 };
        let table = &mut self.tables[table_idx];

        if table.size == 0 {
            *table = HashTable::with_size(INITIAL_SIZE);
        }

        let idx = hash as usize & table.mask;

        let new_entry = Box::new(Entry {
            key,
            value,
            next: table.buckets[idx].take(),
        });
        table.buckets[idx] = Some(new_entry);
        table.used += 1;

        None
    }

    /// Remove a key
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if self.tables[0].size == 0 && self.tables[1].size == 0 {
            return None;
        }

        if self.is_rehashing() {
            self.rehash_step();
        }

        // Compute hash before borrowing tables mutably
        let hash = self.hash_key(key);
        let is_rehashing = self.is_rehashing();

        for table_idx in 0..2 {
            if table_idx == 1 && !is_rehashing {
                break;
            }

            let table = &mut self.tables[table_idx];
            if table.size == 0 {
                continue;
            }

            let idx = hash as usize & table.mask;

            // Check if first entry matches
            if let Some(ref entry) = table.buckets[idx] {
                if entry.key == *key {
                    let mut removed = table.buckets[idx].take().unwrap();
                    table.buckets[idx] = removed.next.take();
                    table.used -= 1;
                    return Some(removed.value);
                }
            }

            // Search in the rest of the list
            let mut current = &mut table.buckets[idx];
            while let Some(ref mut entry) = current {
                if let Some(ref next) = entry.next {
                    if next.key == *key {
                        let mut removed = entry.next.take().unwrap();
                        entry.next = removed.next.take();
                        table.used -= 1;
                        return Some(removed.value);
                    }
                }
                current = &mut entry.next;
            }
        }

        None
    }

    /// Check if key exists
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.tables[0].used + self.tables[1].used
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Hash a key
    fn hash_key(&self, key: &K) -> u64 {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Expand the hash table if needed
    fn expand_if_needed(&mut self) {
        if self.is_rehashing() {
            return;
        }

        let table = &self.tables[0];

        // First insert
        if table.size == 0 {
            return;
        }

        // Check if we need to expand
        if table.used >= table.size * RESIZE_RATIO {
            let new_size = table.used * 2;
            self.tables[1] = HashTable::with_size(new_size);
            self.rehash_idx = Some(0);
        }
    }

    /// Perform one step of incremental rehashing
    fn rehash_step(&mut self) {
        if let Some(idx) = self.rehash_idx {
            let mut entries_moved = 0;
            let mut current_idx = idx;

            // Move up to 10 entries
            while entries_moved < 10 && current_idx < self.tables[0].size {
                if self.tables[0].buckets[current_idx].is_some() {
                    // Move all entries in this bucket
                    while let Some(mut entry) = self.tables[0].buckets[current_idx].take() {
                        let next = entry.next.take();

                        // Insert into new table
                        let hash = self.hash_key(&entry.key);
                        let new_idx = hash as usize & self.tables[1].mask;
                        entry.next = self.tables[1].buckets[new_idx].take();
                        self.tables[1].buckets[new_idx] = Some(entry);
                        self.tables[1].used += 1;
                        self.tables[0].used -= 1;

                        self.tables[0].buckets[current_idx] = next;
                        entries_moved += 1;
                    }
                }
                current_idx += 1;
            }

            // Check if rehashing is complete
            if current_idx >= self.tables[0].size {
                // Use split_at_mut to avoid double mutable borrow
                let (first, second) = self.tables.split_at_mut(1);
                std::mem::swap(&mut first[0], &mut second[0]);
                self.tables[1] = HashTable::new();
                self.rehash_idx = None;
            } else {
                self.rehash_idx = Some(current_idx);
            }
        }
    }

    /// Get random keys for sampling
    pub fn random_keys(&self, count: usize) -> Vec<K> {
        let mut keys = Vec::with_capacity(count);
        let total = self.len();

        if total == 0 || count == 0 {
            return keys;
        }

        for _ in 0..count {
            // Simple random sampling (not perfectly uniform)
            let table_idx = if self.is_rehashing() && rand::random::<bool>() {
                1
            } else {
                0
            };

            let table = &self.tables[table_idx];
            if table.size == 0 {
                continue;
            }

            let start_idx = rand::random::<usize>() % table.size;

            // Find first non-empty bucket
            for i in 0..table.size {
                let idx = (start_idx + i) % table.size;
                if let Some(entry) = &table.buckets[idx] {
                    keys.push(entry.key.clone());
                    break;
                }
            }
        }

        keys
    }

    /// Iterate over all keys deterministically
    pub fn iter_keys(&self) -> Vec<K> {
        let mut keys = Vec::with_capacity(self.len());

        // Iterate over both tables (needed during rehashing)
        for table in &self.tables {
            for bucket in &table.buckets {
                let mut entry = bucket.as_ref();
                while let Some(e) = entry {
                    keys.push(e.key.clone());
                    entry = e.next.as_ref();
                }
            }
        }

        keys
    }
}

impl<K: Hash + Eq + Clone, V> Default for Dict<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut dict: Dict<String, i32> = Dict::new();

        assert!(dict.is_empty());

        dict.insert("foo".to_string(), 1);
        dict.insert("bar".to_string(), 2);
        dict.insert("baz".to_string(), 3);

        assert_eq!(dict.len(), 3);
        assert_eq!(dict.get(&"foo".to_string()), Some(&1));
        assert_eq!(dict.get(&"bar".to_string()), Some(&2));
        assert_eq!(dict.get(&"qux".to_string()), None);

        // Update
        dict.insert("foo".to_string(), 10);
        assert_eq!(dict.get(&"foo".to_string()), Some(&10));

        // Remove
        assert_eq!(dict.remove(&"bar".to_string()), Some(2));
        assert_eq!(dict.len(), 2);
        assert_eq!(dict.get(&"bar".to_string()), None);
    }

    #[test]
    fn test_rehashing() {
        let mut dict: Dict<i32, i32> = Dict::new();

        // Insert many entries to trigger rehashing
        for i in 0..1000 {
            dict.insert(i, i * 2);
        }

        assert_eq!(dict.len(), 1000);

        // Verify all entries
        for i in 0..1000 {
            assert_eq!(dict.get(&i), Some(&(i * 2)));
        }
    }
}
