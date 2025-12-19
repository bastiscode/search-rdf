pub mod embedding;
pub mod item;
pub mod python;
pub use embedding::TextEmbeddings;

use crate::data::map::{OrderedDataMap, TrieMap};
use crate::data::text::item::TextItem;

use super::DataSource;
use anyhow::{Result, anyhow};
use memmap2::Mmap;
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

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
    pub fn identifier(&self, id: u32) -> Option<&str> {
        let range = self.inner.data_map.range(id as usize)?;
        let (identifier, _) = TextItem::decode_key(&self.inner.data[range]);
        Some(identifier)
    }

    pub fn build(items: impl IntoIterator<Item = Result<TextItem>>, data_dir: &Path) -> Result<()> {
        create_dir_all(data_dir)?;

        let data_file = data_dir.join("data.bin");
        let mut data = BufWriter::new(File::create(&data_file)?);

        let mut identifier_map = TrieMap::new();
        let mut data_map = OrderedDataMap::new();

        for (id, item) in items.into_iter().enumerate() {
            let item = item?;
            let id = id as u32;
            if id == u32::MAX {
                return Err(anyhow!("too many unique identifiers, max is {}", u32::MAX));
            }

            let encoded = item.encode();
            data.write_all(&encoded)?;

            identifier_map.add(&item.identifier, id)?;
            data_map.add(encoded.len(), item.num_fields())?;
        }

        let data_map_file = data_dir.join("data-map.bin");
        data_map.save(&data_map_file)?;

        let identifier_map_file = data_dir.join("identifier-map.bin");
        identifier_map.save(&identifier_map_file)?;

        Ok(())
    }

    pub fn load(data_dir: &Path) -> Result<Self> {
        let data_file = data_dir.join("data.bin");
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

    fn num_fields(&self, id: u32) -> Option<u16> {
        self.inner.data_map.count(id as usize)
    }

    fn field(&self, id: u32, field: usize) -> Option<&str> {
        if let Some(count) = self.inner.data_map.count(id as usize)
            && field >= count as usize
        {
            return None;
        }
        let mut range = self.inner.data_map.range(id as usize)?;
        let (_, n) = TextItem::decode_key(&self.inner.data[range.clone()]);
        range.start += n;
        TextItem::decode_values(&self.inner.data[range])
            .nth(field)
            .map(|(value, _)| value)
    }

    fn fields(&self, id: u32) -> Option<impl Iterator<Item = Self::Field<'_>>> {
        let mut range = self.inner.data_map.range(id as usize)?;
        let (_, n) = TextItem::decode_key(&self.inner.data[range.clone()]);
        range.start += n;
        Some(TextItem::decode_values(&self.inner.data[range]).map(|(value, _)| value))
    }

    fn data_type(&self) -> &'static str {
        "TextData"
    }

    fn total_fields(&self) -> u32 {
        self.inner.data_map.total_count
    }

    fn items(&self) -> impl Iterator<Item = (u32, Vec<&str>)> + '_ {
        (0..self.len()).filter_map(|id| {
            let id = id as u32;
            let fields = self.fields(id)?.collect();
            Some((id, fields))
        })
    }

    fn max_fields(&self) -> u16 {
        self.inner.data_map.max_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_text_data() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let data_dir = temp_dir.path().join("data");

        // Create test data: identifier -> fields
        let items = vec![
            Ok(TextItem::new(
                "Q1".to_string(),
                vec!["Universe".to_string(), "Cosmos".to_string()],
            )
            .expect("Failed to create TextItem")),
            Ok(TextItem::new(
                "Q2".to_string(),
                vec!["Earth".to_string(), "World".to_string()],
            )
            .expect("Failed to create TextItem")),
            Ok(TextItem::new(
                "Q3".to_string(),
                vec!["Human".to_string(), "Person".to_string()],
            )
            .expect("Failed to create TextItem")),
        ];

        // Build and load
        TextData::build(items, &data_dir).expect("Failed to build");
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
