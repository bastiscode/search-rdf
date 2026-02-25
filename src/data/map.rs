use anyhow::{Result, anyhow};
use byte_trie::{AdaptiveRadixTrie, PrefixSearch};
use fst::Map;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::{fs::File, ops::Range, path::Path};

use crate::utils::{load_bincode, write_bincode};

#[derive(Debug)]
pub struct TrieMap {
    trie: AdaptiveRadixTrie<u32>,
}

impl Default for TrieMap {
    fn default() -> Self {
        Self::new()
    }
}

impl TrieMap {
    pub fn new() -> Self {
        Self {
            trie: AdaptiveRadixTrie::default(),
        }
    }

    pub fn add<I>(&mut self, identifier: I, id: u32) -> Result<()>
    where
        I: AsRef<str>,
    {
        if self
            .trie
            .insert(identifier.as_ref().as_bytes(), id)
            .is_some()
        {
            return Err(anyhow!(
                "duplicate identifier found: {}",
                identifier.as_ref()
            ));
        }
        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        write_bincode(path, &self.trie)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let trie = load_bincode(path)?;
        Ok(Self { trie })
    }

    pub fn get<I>(&self, identifier: I) -> Option<u32>
    where
        I: AsRef<str>,
    {
        self.trie.get(identifier.as_ref().as_bytes()).copied()
    }
}

#[derive(Debug)]
enum FstMapState {
    Building(Vec<(String, u32)>),
    Loaded(Map<Mmap>),
}

/// A memory-mapped FST-based map for string identifiers to u32 values.
/// This is more memory-efficient than TrieMap but slightly slower for lookups.
///
/// Usage:
/// - Create with `new()` and use `add()` to build the map
/// - Call `save()` to persist to disk
/// - Use `load()` to create a memory-mapped instance from disk
/// - Once loaded, the map is immutable (cannot add more items)
#[derive(Debug)]
pub struct FstMap {
    state: FstMapState,
}

impl Default for FstMap {
    fn default() -> Self {
        Self::new()
    }
}

impl FstMap {
    /// Create a new FST map in building mode.
    pub fn new() -> Self {
        Self {
            state: FstMapState::Building(Vec::new()),
        }
    }

    /// Add an identifier with its ID.
    /// Only works when in building mode (not after loading from disk).
    /// Duplicates are checked when save() is called.
    pub fn add<I>(&mut self, identifier: I, id: u32) -> Result<()>
    where
        I: AsRef<str>,
    {
        match &mut self.state {
            FstMapState::Building(items) => {
                items.push((identifier.as_ref().to_string(), id));
                Ok(())
            }
            FstMapState::Loaded(_) => Err(anyhow!(
                "Cannot add to a loaded FstMap. The FST is immutable once loaded."
            )),
        }
    }

    /// Save the FST to a file.
    /// Only works when in building mode.
    /// Items are automatically sorted and checked for duplicates before writing.
    pub fn save(&self, path: &Path) -> Result<()> {
        match &self.state {
            FstMapState::Building(items) => {
                // Sort items by key
                let mut sorted_items = items.clone();
                sorted_items.sort_by(|a, b| a.0.cmp(&b.0));

                // Check for duplicates
                for i in 1..sorted_items.len() {
                    if sorted_items[i].0 == sorted_items[i - 1].0 {
                        return Err(anyhow!("duplicate identifier found: {}", sorted_items[i].0));
                    }
                }

                let file = File::create(path)?;
                let mut builder = fst::MapBuilder::new(file)?;

                for (key, value) in &sorted_items {
                    builder.insert(key.as_bytes(), *value as u64)?;
                }

                builder.finish()?;
                Ok(())
            }
            FstMapState::Loaded(_) => Err(anyhow!(
                "Cannot save a loaded FstMap. It's already persisted to disk."
            )),
        }
    }

    /// Load a memory-mapped FST from a file.
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let map = Map::new(mmap)?;
        Ok(Self {
            state: FstMapState::Loaded(map),
        })
    }

    /// Get the ID for a given identifier.
    /// Works in both building and loaded modes.
    /// In building mode, performs linear search (slower).
    /// In loaded mode, uses FST lookup (faster, memory-efficient).
    pub fn get<I>(&self, identifier: I) -> Option<u32>
    where
        I: AsRef<str>,
    {
        match &self.state {
            FstMapState::Building(items) => items
                .iter()
                .find(|(k, _)| k == identifier.as_ref())
                .map(|(_, v)| *v),
            FstMapState::Loaded(map) => map.get(identifier.as_ref().as_bytes()).map(|v| v as u32),
        }
    }

    /// Get the number of entries in the map.
    pub fn len(&self) -> usize {
        match &self.state {
            FstMapState::Building(items) => items.len(),
            FstMapState::Loaded(map) => map.len(),
        }
    }

    /// Check if the map is empty.
    pub fn is_empty(&self) -> bool {
        match &self.state {
            FstMapState::Building(items) => items.is_empty(),
            FstMapState::Loaded(map) => map.is_empty(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct OrderedDataMap {
    offsets: Vec<usize>,
    counts: Vec<u16>,
    pub total_count: u32,
    pub max_count: u16,
}

impl OrderedDataMap {
    pub fn new() -> Self {
        Self {
            offsets: Vec::new(),
            counts: Vec::new(),
            total_count: 0,
            max_count: 0,
        }
    }

    pub fn add(&mut self, length: usize, count: u16) -> Result<()> {
        if self.total_count as u64 + count as u64 > u32::MAX as u64 {
            return Err(anyhow!(
                "total number of fields exceeds maximum of {}",
                u32::MAX
            ));
        }
        let last_offset = self.offsets.last().copied().unwrap_or(0);
        self.offsets.push(last_offset + length);
        self.counts.push(count);
        self.total_count += count as u32;
        self.max_count = self.max_count.max(count);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    pub fn count(&self, index: usize) -> Option<u16> {
        self.counts.get(index).copied()
    }

    pub fn range(&self, index: usize) -> Option<Range<usize>> {
        let offset = *self.offsets.get(index)?;
        let last_offset = if index > 0 {
            self.offsets[index - 1]
        } else {
            0
        };
        Some(last_offset..offset)
    }

    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    pub fn counts(&self) -> &[u16] {
        &self.counts
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        write_bincode(path, &self)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        load_bincode(path)
    }
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct OrderedIdMap {
    ids: Vec<u32>,
    offsets: Vec<usize>,
    pub total_count: u32,
    pub max_count: u16,
}

impl OrderedIdMap {
    pub fn new() -> Self {
        Self {
            ids: Vec::new(),
            offsets: Vec::new(),
            total_count: 0,
            max_count: 0,
        }
    }

    pub fn from_ids(ids: &[u32]) -> Result<Self> {
        let mut map = Self::new();
        let mut last_id: Option<u32> = None;
        let mut count = 0;
        for &id in ids {
            if last_id.is_none_or(|last| last == id) {
                last_id = Some(id);
                count += 1;
                continue;
            }
            map.add(last_id.expect("should not happen"), count)?;
            last_id = Some(id);
            count = 1;
        }
        if let Some(id) = last_id {
            map.add(id, count)?;
        }
        Ok(map)
    }

    pub fn add(&mut self, id: u32, length: usize) -> Result<()> {
        if length > u16::MAX as usize {
            return Err(anyhow!(
                "too many fields for a single data point, max is {}",
                u16::MAX
            ));
        }
        let last_offset = self.offsets.last().copied().unwrap_or(0);
        let offset = last_offset + length;

        if offset > u32::MAX as usize {
            return Err(anyhow!(
                "total number of fields exceeds maximum of {}",
                u32::MAX
            ));
        }

        if Some(id) <= self.ids.last().copied() {
            return Err(anyhow!(
                "IDs are expected to be unique and in ascending order for OrderedIdMap"
            ));
        }

        self.ids.push(id);
        self.offsets.push(offset);
        self.total_count += length as u32;
        self.max_count = self.max_count.max(length as u16);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn count(&self, id: u32) -> Option<u16> {
        let index = self.ids.binary_search(&id).ok()?;
        let offset = self.offsets[index];
        let last_offset = if index > 0 {
            self.offsets[index - 1]
        } else {
            0
        };
        Some((offset - last_offset) as u16)
    }

    pub fn data_id_for_field(&self, field_id: usize) -> Option<u32> {
        let i = self.offsets.partition_point(|&off| off <= field_id);
        self.ids.get(i).copied()
    }

    pub fn range(&self, id: u32) -> Option<Range<usize>> {
        let index = self.ids.binary_search(&id).ok()?;
        let offset = self.offsets[index];
        let last_offset = if index > 0 {
            self.offsets[index - 1]
        } else {
            0
        };
        Some(last_offset..offset)
    }

    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        write_bincode(path, &self)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let map = load_bincode(path)?;
        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_trie_map_basic() {
        let mut map = TrieMap::new();

        map.add("Q1", 0).expect("Failed to add");
        map.add("Q2", 1).expect("Failed to add");
        map.add("Q100", 100).expect("Failed to add");

        assert_eq!(map.get("Q1"), Some(0));
        assert_eq!(map.get("Q2"), Some(1));
        assert_eq!(map.get("Q100"), Some(100));
        assert_eq!(map.get("Q999"), None);
    }

    #[test]
    fn test_trie_map_duplicate_identifier() {
        let mut map = TrieMap::new();

        map.add("Q1", 0).expect("Failed to add");
        let result = map.add("Q1", 1);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("duplicate identifier")
        );
    }

    #[test]
    fn test_trie_map_save_load() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let path = temp_dir.path().join("trie-map.bin");

        let mut map = TrieMap::new();
        map.add("Alice", 0).expect("Failed to add");
        map.add("Bob", 1).expect("Failed to add");
        map.add("Charlie", 2).expect("Failed to add");

        map.save(&path).expect("Failed to save");
        let loaded = TrieMap::load(&path).expect("Failed to load");

        assert_eq!(loaded.get("Alice"), Some(0));
        assert_eq!(loaded.get("Bob"), Some(1));
        assert_eq!(loaded.get("Charlie"), Some(2));
        assert_eq!(loaded.get("David"), None);
    }

    #[test]
    fn test_ordered_data_map_basic() {
        let mut map = OrderedDataMap::new();

        assert_eq!(map.len(), 0);
        assert!(map.is_empty());

        map.add(10, 2).expect("Failed to add");
        map.add(15, 3).expect("Failed to add");
        map.add(8, 1).expect("Failed to add");

        assert_eq!(map.len(), 3);
        assert!(!map.is_empty());

        assert_eq!(map.count(0), Some(2));
        assert_eq!(map.count(1), Some(3));
        assert_eq!(map.count(2), Some(1));
        assert_eq!(map.count(3), None);
        assert_eq!(map.total_count, 6);
    }

    #[test]
    fn test_ordered_data_map_range() {
        let mut map = OrderedDataMap::new();

        map.add(10, 2).expect("Failed to add"); // offset 0..10
        map.add(15, 3).expect("Failed to add"); // offset 10..25
        map.add(8, 1).expect("Failed to add"); // offset 25..33

        assert_eq!(map.range(0), Some(0..10));
        assert_eq!(map.range(1), Some(10..25));
        assert_eq!(map.range(2), Some(25..33));
        assert_eq!(map.range(3), None);
    }

    #[test]
    fn test_ordered_data_map_accessors() {
        let mut map = OrderedDataMap::new();

        map.add(10, 2).expect("Failed to add");
        map.add(15, 3).expect("Failed to add");

        assert_eq!(map.offsets(), &[10, 25]);
        assert_eq!(map.counts(), &[2, 3]);
        assert_eq!(map.total_count, 5);
        assert_eq!(map.max_count, 3);
    }

    #[test]
    fn test_ordered_data_map_save_load() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let path = temp_dir.path().join("data-map.bin");

        let mut map = OrderedDataMap::new();
        map.add(10, 2).expect("Failed to add");
        map.add(15, 3).expect("Failed to add");
        map.add(8, 1).expect("Failed to add");

        map.save(&path).expect("Failed to save");
        let loaded = OrderedDataMap::load(&path).expect("Failed to load");

        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.count(0), Some(2));
        assert_eq!(loaded.count(1), Some(3));
        assert_eq!(loaded.count(2), Some(1));
        assert_eq!(loaded.range(0), Some(0..10));
        assert_eq!(loaded.range(1), Some(10..25));
        assert_eq!(loaded.range(2), Some(25..33));
        assert_eq!(loaded.total_count, 6);
        assert_eq!(loaded.max_count, 3);
    }

    #[test]
    fn test_ordered_id_map_from_ids_sorted() {
        let ids = vec![10, 20, 30, 40];
        let map = OrderedIdMap::from_ids(&ids).expect("Failed to create map");

        assert_eq!(map.len(), 4);
        assert_eq!(map.count(10), Some(1));
        assert_eq!(map.count(20), Some(1));
        assert_eq!(map.count(30), Some(1));
        assert_eq!(map.count(40), Some(1));
        assert_eq!(map.count(50), None);

        assert_eq!(map.range(10), Some(0..1));
        assert_eq!(map.range(20), Some(1..2));
        assert_eq!(map.range(30), Some(2..3));
        assert_eq!(map.range(40), Some(3..4));
    }

    #[test]
    fn test_ordered_id_map_data_id_for_field() {
        // IDs 10 (3 fields), 20 (2 fields), 30 (1 field)
        // field layout: [0,1,2]=10, [3,4]=20, [5]=30
        let ids = vec![10, 10, 10, 20, 20, 30];
        let map = OrderedIdMap::from_ids(&ids).expect("Failed to create map");

        assert_eq!(map.data_id_for_field(0), Some(10));
        assert_eq!(map.data_id_for_field(1), Some(10));
        assert_eq!(map.data_id_for_field(2), Some(10));
        assert_eq!(map.data_id_for_field(3), Some(20));
        assert_eq!(map.data_id_for_field(4), Some(20));
        assert_eq!(map.data_id_for_field(5), Some(30));
        assert_eq!(map.data_id_for_field(6), None);
    }

    #[test]
    fn test_ordered_id_map_from_ids_with_duplicates() {
        let ids = vec![10, 10, 10, 20, 20, 30];
        let map = OrderedIdMap::from_ids(&ids).expect("Failed to create map");

        assert_eq!(map.len(), 3);
        assert_eq!(map.count(10), Some(3));
        assert_eq!(map.count(20), Some(2));
        assert_eq!(map.count(30), Some(1));

        assert_eq!(map.range(10), Some(0..3));
        assert_eq!(map.range(20), Some(3..5));
        assert_eq!(map.range(30), Some(5..6));

        assert_eq!(map.total_count, 6);
        assert_eq!(map.max_count, 3);
    }

    #[test]
    fn test_ordered_id_map_from_ids_unsorted() {
        let ids = vec![30, 10, 20];
        let result = OrderedIdMap::from_ids(&ids);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("ascending order"));
    }

    #[test]
    fn test_ordered_id_map_add_ascending() {
        let mut map = OrderedIdMap::new();

        map.add(10, 2).expect("Failed to add");
        map.add(20, 3).expect("Failed to add");
        map.add(30, 1).expect("Failed to add");

        assert_eq!(map.len(), 3);
        assert_eq!(map.count(10), Some(2));
        assert_eq!(map.count(20), Some(3));
        assert_eq!(map.count(30), Some(1));

        assert_eq!(map.range(10), Some(0..2));
        assert_eq!(map.range(20), Some(2..5));
        assert_eq!(map.range(30), Some(5..6));
    }

    #[test]
    fn test_ordered_id_map_add_descending() {
        let mut map = OrderedIdMap::new();

        map.add(30, 1).expect("Failed to add");
        let result = map.add(20, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("ascending order"));
    }

    #[test]
    fn test_ordered_id_map_add_duplicate() {
        let mut map = OrderedIdMap::new();

        map.add(10, 1).expect("Failed to add");
        let result = map.add(10, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("ascending order"));
    }

    #[test]
    fn test_ordered_id_map_accessors() {
        let mut map = OrderedIdMap::new();

        map.add(10, 2).expect("Failed to add");
        map.add(20, 3).expect("Failed to add");

        assert_eq!(map.ids(), &[10, 20]);
        assert_eq!(map.offsets(), &[2, 5]);
        assert_eq!(map.total_count, 5);
        assert_eq!(map.max_count, 3);
    }

    #[test]
    fn test_ordered_id_map_save_load() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let path = temp_dir.path().join("id-map.bin");

        let mut map = OrderedIdMap::new();
        map.add(100, 2).expect("Failed to add");
        map.add(200, 3).expect("Failed to add");
        map.add(300, 1).expect("Failed to add");

        map.save(&path).expect("Failed to save");
        let loaded = OrderedIdMap::load(&path).expect("Failed to load");

        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.count(100), Some(2));
        assert_eq!(loaded.count(200), Some(3));
        assert_eq!(loaded.count(300), Some(1));
        assert_eq!(loaded.range(100), Some(0..2));
        assert_eq!(loaded.range(200), Some(2..5));
        assert_eq!(loaded.range(300), Some(5..6));
        assert_eq!(loaded.total_count, 6);
        assert_eq!(loaded.max_count, 3);
    }

    #[test]
    fn test_ordered_id_map_binary_search() {
        let ids = vec![10, 50, 100, 500, 1000];
        let map = OrderedIdMap::from_ids(&ids).expect("Failed to create map");

        // Test that binary search works correctly
        assert_eq!(map.count(10), Some(1));
        assert_eq!(map.count(50), Some(1));
        assert_eq!(map.count(100), Some(1));
        assert_eq!(map.count(500), Some(1));
        assert_eq!(map.count(1000), Some(1));

        // Test non-existent IDs
        assert_eq!(map.count(5), None);
        assert_eq!(map.count(25), None);
        assert_eq!(map.count(2000), None);
    }

    #[test]
    fn test_ordered_data_map_empty() {
        let map = OrderedDataMap::new();

        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert_eq!(map.count(0), None);
        assert_eq!(map.range(0), None);
        assert_eq!(map.total_count, 0);
        assert_eq!(map.max_count, 0);
    }

    #[test]
    fn test_ordered_id_map_empty() {
        let map = OrderedIdMap::new();

        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert_eq!(map.count(0), None);
        assert_eq!(map.range(0), None);
        assert_eq!(map.total_count, 0);
        assert_eq!(map.max_count, 0);
    }

    #[test]
    fn test_fst_map_basic() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let path = temp_dir.path().join("fst-map.bin");

        let mut map = super::FstMap::new();
        map.add("Q1", 0).expect("Failed to add");
        map.add("Q2", 1).expect("Failed to add");
        map.add("Q100", 100).expect("Failed to add");

        assert_eq!(map.len(), 3);
        assert!(!map.is_empty());

        // Test get in building mode
        assert_eq!(map.get("Q1"), Some(0));
        assert_eq!(map.get("Q2"), Some(1));

        map.save(&path).expect("Failed to save");

        let loaded_map = super::FstMap::load(&path).expect("Failed to load");

        assert_eq!(loaded_map.len(), 3);
        assert!(!loaded_map.is_empty());
        assert_eq!(loaded_map.get("Q1"), Some(0));
        assert_eq!(loaded_map.get("Q2"), Some(1));
        assert_eq!(loaded_map.get("Q100"), Some(100));
        assert_eq!(loaded_map.get("Q999"), None);
    }

    #[test]
    fn test_fst_map_duplicate_identifier() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let path = temp_dir.path().join("fst-map-dup.bin");

        let mut map = super::FstMap::new();
        map.add("Q1", 0).expect("Failed to add");
        map.add("Q1", 1).expect("Failed to add"); // No error yet

        // Error should occur during save
        let result = map.save(&path);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("duplicate identifier")
        );
    }

    #[test]
    fn test_fst_map_auto_sort() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let path = temp_dir.path().join("fst-map-sort.bin");

        let mut map = super::FstMap::new();
        // Add in non-sorted order
        map.add("Charlie", 2).expect("Failed to add");
        map.add("Alice", 0).expect("Failed to add");
        map.add("Bob", 1).expect("Failed to add");

        map.save(&path).expect("Failed to save");

        let loaded_map = super::FstMap::load(&path).expect("Failed to load");

        assert_eq!(loaded_map.get("Alice"), Some(0));
        assert_eq!(loaded_map.get("Bob"), Some(1));
        assert_eq!(loaded_map.get("Charlie"), Some(2));
    }

    #[test]
    fn test_fst_map_empty() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let path = temp_dir.path().join("fst-map-empty.bin");

        let map = super::FstMap::new();
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());

        map.save(&path).expect("Failed to save");

        let loaded_map = super::FstMap::load(&path).expect("Failed to load");
        assert_eq!(loaded_map.len(), 0);
        assert!(loaded_map.is_empty());
        assert_eq!(loaded_map.get("anything"), None);
    }

    #[test]
    fn test_fst_map_large_ids() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let path = temp_dir.path().join("fst-map-large.bin");

        let mut map = super::FstMap::new();
        map.add("max", u32::MAX).expect("Failed to add");
        map.add("zero", 0).expect("Failed to add");

        map.save(&path).expect("Failed to save");

        let loaded_map = super::FstMap::load(&path).expect("Failed to load");
        assert_eq!(loaded_map.get("max"), Some(u32::MAX));
        assert_eq!(loaded_map.get("zero"), Some(0));
    }

    #[test]
    fn test_fst_map_cannot_add_after_load() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let path = temp_dir.path().join("fst-map-immutable.bin");

        let mut map = super::FstMap::new();
        map.add("Q1", 0).expect("Failed to add");
        map.save(&path).expect("Failed to save");

        let mut loaded_map = super::FstMap::load(&path).expect("Failed to load");
        let result = loaded_map.add("Q2", 1);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("immutable"));
    }

    #[test]
    fn test_fst_map_cannot_save_after_load() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let path = temp_dir.path().join("fst-map-no-save.bin");
        let path2 = temp_dir.path().join("fst-map-no-save-2.bin");

        let mut map = super::FstMap::new();
        map.add("Q1", 0).expect("Failed to add");
        map.save(&path).expect("Failed to save");

        let loaded_map = super::FstMap::load(&path).expect("Failed to load");
        let result = loaded_map.save(&path2);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("persisted"));
    }
}
