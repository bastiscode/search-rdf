pub mod embedding;
pub mod item;
pub mod map;
pub mod python;

use std::{
    fs::{File, create_dir_all},
    io::{BufWriter, Write},
    path::Path,
    sync::Arc,
};

use anyhow::{Result, anyhow};
pub use embedding::{EmbeddingRef, Embeddings, EmbeddingsWithData, Precision};
use log::info;
use memmap2::Mmap;

use crate::data::{
    item::{FieldRef, Item, ItemRef},
    map::{OrderedDataMap, TrieMap},
};

/// Core trait for data sources that can be indexed
pub trait DataSource: Send + Sync + Clone {
    /// The type of data in a single field
    type Field<'a>
    where
        Self: 'a;

    /// Number of data points
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of fields across all data points
    /// u32::MAX is max supported
    fn total_fields(&self) -> u32;

    /// Avg. number of fields per data point
    fn avg_fields(&self) -> f32 {
        self.total_fields() as f32 / self.len().max(1) as f32
    }

    /// Max number of fields for any single data point
    /// u16::MAX is max supported per data point
    fn max_fields(&self) -> u16;

    /// Get number of searchable fields for a data point
    /// Returns None if the ID is invalid
    fn num_fields(&self, id: u32) -> Option<u16>;

    /// Get a specific field value for a data point
    fn field(&self, id: u32, field: usize) -> Option<Self::Field<'_>>;

    /// Get all searchable fields as a collection for a data point
    fn fields(&self, id: u32) -> Option<impl Iterator<Item = Self::Field<'_>>>;

    fn items(&self) -> impl Iterator<Item = (u32, Vec<Self::Field<'_>>)> + '_;
}

// Generic mmapped data implementation from items
#[derive(Debug)]
struct Inner {
    data_map: OrderedDataMap,
    identifier_map: TrieMap,
    data: Mmap,
}

#[derive(Debug, Clone)]
pub struct Data {
    inner: Arc<Inner>,
}

impl Data {
    pub fn identifier(&self, id: u32) -> Option<&str> {
        let range = self.inner.data_map.range(id as usize)?;
        let item = ItemRef::decode(&self.inner.data[range]).ok()?;
        Some(item.identifier())
    }

    pub fn build(items: impl IntoIterator<Item = Result<Item>>, data_dir: &Path) -> Result<()> {
        create_dir_all(data_dir)?;

        let data_file = data_dir.join("data.bin");
        let mut data = BufWriter::new(File::create(&data_file)?);

        let mut identifier_map = TrieMap::new();
        let mut data_map = OrderedDataMap::new();

        let log_every = 1_000_000;

        let mut total_fields = 0u64;
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

            total_fields += item.num_fields() as u64;
            if (id + 1).is_multiple_of(log_every) {
                info!("  Processed {} items and {} fields", id + 1, total_fields)
            }
        }

        let data_map_file = data_dir.join("data-map.bin");
        data_map.save(&data_map_file)?;

        let identifier_map_file = data_dir.join("identifier-map.bin");
        identifier_map.save(&identifier_map_file)?;

        info!(
            "Built data with {} items and {} total fields",
            data_map.len(),
            total_fields
        );

        Ok(())
    }

    pub fn load(data_dir: &Path) -> Result<Self> {
        let data_file = data_dir.join("data.bin");
        let data_map_file = data_dir.join("data-map.bin");
        let identifier_map_file = data_dir.join("identifier-map.bin");

        let data = unsafe { Mmap::map(&File::open(&data_file)?)? };
        let data_map = OrderedDataMap::load(&data_map_file)?;
        let identifier_map = TrieMap::load(&identifier_map_file)?;

        Ok(Self {
            inner: Arc::new(Inner {
                data_map,
                identifier_map,
                data,
            }),
        })
    }

    pub fn id_from_identifier(&self, identifier: &str) -> Option<u32> {
        self.inner.identifier_map.get(identifier)
    }

    pub fn data_map(&self) -> &OrderedDataMap {
        &self.inner.data_map
    }
}

impl DataSource for Data {
    type Field<'a> = FieldRef<'a>;

    fn len(&self) -> usize {
        self.inner.data_map.len()
    }

    fn num_fields(&self, id: u32) -> Option<u16> {
        self.inner.data_map.count(id as usize)
    }

    fn field(&self, id: u32, field: usize) -> Option<Self::Field<'_>> {
        if let Some(count) = self.inner.data_map.count(id as usize)
            && field >= count as usize
        {
            return None;
        }
        let range = self.inner.data_map.range(id as usize)?;
        let item = ItemRef::decode(&self.inner.data[range]).ok()?;
        item.field(field)
    }

    fn fields(&self, id: u32) -> Option<impl Iterator<Item = Self::Field<'_>>> {
        let range = self.inner.data_map.range(id as usize)?;
        let item = ItemRef::decode(&self.inner.data[range]).ok()?;
        Some(item.fields())
    }

    fn total_fields(&self) -> u32 {
        self.inner.data_map.total_count
    }

    fn items(&self) -> impl Iterator<Item = (u32, Vec<Self::Field<'_>>)> + '_ {
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
