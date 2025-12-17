pub mod embedding;
pub mod python;
pub use embedding::TextEmbeddings;

use crate::data::map::{OrderedDataMap, TrieMap};

use super::DataSource;
use anyhow::{Result, anyhow};
use memmap2::Mmap;
use std::fs::File;
use std::mem::size_of;
use std::path::Path;
use std::sync::Arc;

const U16_SIZE: usize = size_of::<u16>();
const U32_SIZE: usize = size_of::<u32>();

struct Inner {
    data_map: OrderedDataMap,
    identifier_map: TrieMap,
    data: Mmap,
}

/// Text data source backed by a TSV file
#[derive(Clone)]
pub struct TextData {
    inner: Arc<Inner>,
}

impl TextData {
    fn key(data: &[u8]) -> (&str, usize) {
        let key_length_bytes = data[..U16_SIZE]
            .try_into()
            .expect("key length should be a u16");
        let key_length = u16::from_le_bytes(key_length_bytes) as usize;

        let key_start = U16_SIZE;
        let key_end = key_start + key_length;
        let key = unsafe { std::str::from_utf8_unchecked(&data[key_start..key_end]) };

        (key, key_end)
    }

    fn value(data: &[u8]) -> (&str, usize) {
        let value_length_bytes = data[..U32_SIZE]
            .try_into()
            .expect("value length should be a u32");
        let value_length = u32::from_le_bytes(value_length_bytes) as usize;

        let value_start = U32_SIZE;
        let value_end = value_start + value_length;
        let value = unsafe { std::str::from_utf8_unchecked(&data[value_start..value_end]) };

        (value, value_end)
    }

    fn values(data: &[u8]) -> impl Iterator<Item = (&str, usize)> + '_ {
        // read number of values first
        let num_values_bytes = data[..U16_SIZE]
            .try_into()
            .expect("num values should be a u16");
        let num_values = u16::from_le_bytes(num_values_bytes) as usize;

        // now read each value, again with length and string
        let mut offset = U16_SIZE;
        (0..num_values).map(move |_| {
            let (value, n) = Self::value(&data[offset..]);
            offset += n;
            (value, n)
        })
    }

    pub fn identifier(&self, id: u32) -> Option<&str> {
        let range = self.inner.data_map.range(id as usize)?;
        let (identifier, _) = Self::key(&self.inner.data[range]);
        Some(identifier)
    }

    pub fn build(data_dir: &Path) -> Result<()> {
        let data_file = data_dir.join("data");

        let mmap = unsafe { Mmap::map(&File::open(data_file)?)? };

        let mut identifier_map = TrieMap::new();
        let mut data_map = OrderedDataMap::new();

        let mut offset = 0;
        let mut id = 0;
        while offset < mmap.len() {
            if id == u32::MAX {
                return Err(anyhow!("too many unique identifiers, max is {}", u32::MAX));
            }

            let (identifier, n) = Self::key(&mmap[offset..]);

            let mut num_fields = 0;
            let mut length = n; // Just the key size
            for (_value, n) in Self::values(&mmap[offset + length..]) {
                num_fields += 1;
                length += n;
            }
            length += U16_SIZE; // Add the num_values header that values() consumed

            identifier_map.add(identifier, id);
            data_map.add(length, num_fields)?;
            offset += length;
            id += 1;
        }

        let data_map_file = data_dir.join("data-map.bin");
        data_map.save(&data_map_file)?;

        let identifier_map_file = data_dir.join("identifier-map.bin");
        identifier_map.save(&identifier_map_file)?;

        Ok(())
    }

    pub fn load(data_dir: &Path) -> Result<Self> {
        let data_file = data_dir.join("data");
        let data_map_file = data_dir.join("data-map.bin");
        let identifier_map_file = data_dir.join("identifier-map.bin");

        let data = unsafe { Mmap::map(&File::open(&data_file)?)? };
        let data_map = OrderedDataMap::load(&data_map_file)?;
        let identifier_map = TrieMap::load(&identifier_map_file)?;

        Ok(TextData {
            inner: Arc::new(Inner {
                data_map,
                identifier_map,
                data,
            }),
        })
    }

    pub fn max_fields_per_id(&self) -> usize {
        self.inner.data_map.max_count as usize
    }

    pub fn id_from_identifier(&self, identifier: &str) -> Option<u32> {
        self.inner.identifier_map.get(identifier)
    }

    pub fn data_map(&self) -> &OrderedDataMap {
        &self.inner.data_map
    }
}

impl DataSource for TextData {
    type Field<'a> = &'a str;

    fn len(&self) -> usize {
        self.inner.data_map.len()
    }

    fn num_fields(&self, id: u32) -> Option<usize> {
        self.inner.data_map.count(id as usize).map(|c| c as usize)
    }

    fn field(&self, id: u32, field: usize) -> Option<&str> {
        if let Some(count) = self.inner.data_map.count(id as usize)
            && field >= count as usize
        {
            return None;
        }
        let mut range = self.inner.data_map.range(id as usize)?;
        let (_, n) = Self::key(&self.inner.data[range.clone()]);
        range.start += n;
        Self::values(&self.inner.data[range])
            .nth(field)
            .map(|(value, _)| value)
    }

    fn fields(&self, id: u32) -> Option<impl Iterator<Item = Self::Field<'_>>> {
        let mut range = self.inner.data_map.range(id as usize)?;
        let (_, n) = Self::key(&self.inner.data[range.clone()]);
        range.start += n;
        Some(Self::values(&self.inner.data[range]).map(|(value, _)| value))
    }

    fn data_type(&self) -> &'static str {
        "TextData"
    }

    fn total_fields(&self) -> usize {
        self.inner.data_map.total_count as usize
    }

    fn items(&self) -> impl Iterator<Item = (u32, Vec<&str>)> + '_ {
        (0..self.len()).filter_map(|id| {
            let id = id as u32;
            let fields = self.fields(id)?.collect();
            Some((id, fields))
        })
    }

    fn max_fields(&self) -> usize {
        self.inner.data_map.max_count as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufWriter, Write};
    use tempfile::tempdir;

    // Helper to create a data file in the expected format
    fn create_test_data_file(data_dir: &Path, rows: &[(&str, &str)]) -> Result<()> {
        std::fs::create_dir_all(data_dir)?;
        let data_file = data_dir.join("data");
        let mut file = BufWriter::new(File::create(&data_file)?);

        for (identifier, tsv_row) in rows {
            // Split tab-separated fields
            let fields: Vec<_> = tsv_row.split('\t').collect();

            // Write identifier length (u16)
            let key_bytes = identifier.as_bytes();
            file.write_all(&(key_bytes.len() as u16).to_le_bytes())?;

            // Write identifier
            file.write_all(key_bytes)?;

            // Write number of fields (u16)
            file.write_all(&(fields.len() as u16).to_le_bytes())?;

            // Write each field
            for field in fields {
                // Write field length (u32)
                let value_bytes = field.as_bytes();
                file.write_all(&(value_bytes.len() as u32).to_le_bytes())?;

                // Write field value
                file.write_all(value_bytes)?;
            }
        }

        Ok(())
    }

    #[test]
    fn test_text_data() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let data_dir = temp_dir.path().join("data");

        // Create test data: identifier -> TSV row (tab-separated fields)
        let rows = vec![
            ("Q1", "Universe\tCosmos"),
            ("Q2", "Earth\tWorld"),
            ("Q3", "Human\tPerson"),
        ];

        create_test_data_file(&data_dir, &rows).expect("Failed to create test data");

        // Build and load
        TextData::build(&data_dir).expect("Failed to build");
        let data = TextData::load(&data_dir).expect("Failed to load");

        // Test basic operations
        assert_eq!(data.len(), 3);
        assert!(!data.is_empty());

        // Test field access by ID (0-indexed: 0 = first searchable field)
        assert_eq!(data.field(0, 0), Some("Universe"));
        assert_eq!(data.field(0, 1), Some("Cosmos"));

        assert_eq!(data.field(1, 0), Some("Earth"));
        assert_eq!(data.field(1, 1), Some("World"));

        // Test fields access (excludes identifier)
        let fields: Vec<_> = data.fields(0).expect("Failed to get fields").collect();
        assert_eq!(fields, vec!["Universe", "Cosmos"]);

        // Test num_fields (excludes identifier)
        assert_eq!(data.num_fields(0), Some(2));
        assert_eq!(data.num_fields(3), None); // Invalid ID

        // Test identifier lookup
        assert_eq!(data.id_from_identifier("Q1"), Some(0));
        assert_eq!(data.id_from_identifier("Q2"), Some(1));
        assert_eq!(data.id_from_identifier("Q3"), Some(2));
        assert_eq!(data.id_from_identifier("Q999"), None);
    }
}
