use crate::data::Data;
use crate::data::item::FieldRef;
use crate::data::map::OrderedIdMap;

use super::DataSource;
use anyhow::{Result, anyhow};
use memmap2::Mmap;
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::mem::size_of;
use std::path::Path;
use std::sync::Arc;
use usearch::ScalarKind;

const F32_SIZE: usize = size_of::<f32>();

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    #[serde(alias = "fp32")]
    Float32,
    #[serde(alias = "fp16")]
    Float16,
    #[serde(alias = "bfp16", alias = "bf16")]
    BFloat16,
    #[serde(alias = "i8", alias = "uint8")]
    Int8,
    #[serde(alias = "bit", alias = "ubinary")]
    Binary,
}

impl Precision {
    pub fn to_usearch_scalar_kind(&self) -> ScalarKind {
        match self {
            Precision::Float32 => ScalarKind::F32,
            Precision::Float16 => ScalarKind::F16,
            Precision::BFloat16 => ScalarKind::BF16,
            Precision::Int8 => ScalarKind::I8,
            Precision::Binary => ScalarKind::B1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingModality {
    Text,
    Image,
}

// Only support Float32 for now, to always have the possibility of
// reranking
pub type Embedding = Vec<f32>;

pub type EmbeddingRef<'a> = &'a [f32];

#[derive(Debug)]
pub struct Tensors {
    // need to keep mmap alive as long as safetensors is used
    #[allow(dead_code)]
    mmap: Mmap,
    #[allow(dead_code)]
    safetensors: SafeTensors<'static>,
    embedding: TensorView<'static>,
    id: Option<TensorView<'static>>,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub modality: Vec<EmbeddingModality>,
}

impl Tensors {
    pub fn load(path: &Path) -> Result<Self> {
        let mmap = unsafe { Mmap::map(&File::open(path)?)? };
        let safetensors = unsafe {
            std::mem::transmute::<SafeTensors<'_>, SafeTensors<'static>>(SafeTensors::deserialize(
                &mmap,
            )?)
        };

        let embedding = safetensors
            .tensor("embedding")
            .map_err(|_| anyhow!("'embedding' tensor not found in safetensors file"))?;

        let embedding_shape = embedding.shape();
        if embedding_shape.len() != 2 {
            return Err(anyhow!(
                "Expected 2D embedding tensor, got shape {:?}",
                embedding_shape
            ));
        }

        if embedding.dtype() != Dtype::F32 {
            return Err(anyhow!(
                "Expected F32 embedding tensor, got dtype {:?}",
                embedding.dtype()
            ));
        }

        let (_, raw_metadata) = SafeTensors::read_metadata(&mmap)?;
        let metadata = raw_metadata.metadata();

        let model = metadata.as_ref().and_then(|m| m.get("model").cloned());
        let provider = metadata.as_ref().and_then(|m| m.get("provider").cloned());
        let modality = metadata
            .as_ref()
            .and_then(|m| m.get("modality").cloned())
            .map(|s| {
                s.split(',')
                    .map(|m| serde_plain::from_str(m.trim()))
                    .collect::<std::result::Result<Vec<_>, _>>()
            })
            .transpose()?
            .unwrap_or_default();

        let id = safetensors.tensor("id").ok();
        if let Some(id) = id.as_ref() {
            let id_shape = id.shape();
            if id_shape.len() != 1 {
                return Err(anyhow!("Expected 1D id tensor, got shape {:?}", id_shape));
            }

            if id_shape[0] != embedding_shape[0] {
                return Err(anyhow!(
                    "ID tensor length ({}) does not match embedding length ({})",
                    id_shape[0],
                    embedding_shape[0]
                ));
            }
        };

        Ok(Self {
            mmap,
            safetensors,
            embedding,
            id,
            model,
            provider,
            modality,
        })
    }

    pub fn len(&self) -> usize {
        self.embedding.shape()[0]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn num_dimensions(&self) -> usize {
        self.embedding.shape()[1]
    }

    pub fn get(&self, idx: usize) -> Option<EmbeddingRef<'_>> {
        let data = self.embedding.data();

        let start = idx * self.num_dimensions() * F32_SIZE;
        let end = start + self.num_dimensions() * F32_SIZE;

        if end > data.len() {
            return None;
        }

        let slice = &data[start..end];
        let (head, floats, tail) = unsafe { slice.align_to::<f32>() };

        if !head.is_empty() || !tail.is_empty() || floats.len() != self.num_dimensions() {
            return None;
        }
        Some(floats)
    }

    pub fn ids(&self) -> Option<&[u32]> {
        let id_tensor = self.id.as_ref()?;
        let id_data = id_tensor.data();
        let (head, ids_slice, tail) = unsafe { id_data.align_to::<u32>() };
        if !head.is_empty() || !tail.is_empty() || ids_slice.len() != self.len() {
            return None;
        }
        Some(ids_slice)
    }
}

#[derive(Debug)]
struct EmbeddingsInner {
    tensors: Tensors,
    id_map: OrderedIdMap,
}

#[derive(Clone, Debug)]
pub struct Embeddings {
    inner: Arc<EmbeddingsInner>,
}

impl Embeddings {
    pub fn build(data_dir: &Path) -> Result<()> {
        let embedding_path = data_dir.join("embedding.safetensors");
        let tensors = Tensors::load(&embedding_path)?;

        // Build ID map from 'id' tensor
        let Some(ids) = tensors.ids() else {
            return Err(anyhow!("embedding file missing 'id' tensor"));
        };

        let id_map_file = data_dir.join("id-map.bin");
        let id_map = OrderedIdMap::from_ids(ids)?;
        id_map.save(&id_map_file)?;

        Ok(())
    }

    /// Load Embeddings from directory
    pub fn load(data_dir: &Path) -> Result<Self> {
        let embedding_path = data_dir.join("embedding.safetensors");
        let tensors = Tensors::load(&embedding_path)?;

        if tensors.ids().is_none() {
            return Err(anyhow!("embedding file missing 'id' tensor"));
        }

        let id_map_file = data_dir.join("id-map.bin");
        let id_map = OrderedIdMap::load(&id_map_file)?;

        Ok(Self {
            inner: Arc::new(EmbeddingsInner { tensors, id_map }),
        })
    }

    pub fn data_id_for_field(&self, field_id: usize) -> Option<u32> {
        self.inner.id_map.data_id_for_field(field_id)
    }

    pub fn field_embedding(&self, field_id: usize) -> Option<EmbeddingRef<'_>> {
        self.inner.tensors.get(field_id)
    }

    /// Get the number of dimensions
    pub fn num_dimensions(&self) -> usize {
        self.inner.tensors.num_dimensions()
    }

    /// Get the model name from metadata
    pub fn model(&self) -> Option<&str> {
        self.inner.tensors.model.as_deref()
    }

    /// Get the provider from metadata
    pub fn provider(&self) -> Option<&str> {
        self.inner.tensors.provider.as_deref()
    }

    /// Get the modalities from metadata
    pub fn modality(&self) -> &[EmbeddingModality] {
        &self.inner.tensors.modality
    }
}

impl DataSource for Embeddings {
    type Field<'a> = EmbeddingRef<'a>;

    fn len(&self) -> usize {
        self.inner.id_map.len()
    }

    fn num_fields(&self, id: u32) -> Option<u16> {
        self.inner.id_map.count(id)
    }

    fn field(&self, id: u32, field: usize) -> Option<Self::Field<'_>> {
        let range = self.inner.id_map.range(id)?;
        if field >= range.end {
            return None;
        }
        let tensor_idx = range.start + field;
        self.inner.tensors.get(tensor_idx)
    }

    fn fields(&self, id: u32) -> Option<impl Iterator<Item = Self::Field<'_>>> {
        let range = self.inner.id_map.range(id)?;
        Some(
            range
                .into_iter()
                .filter_map(|idx| self.inner.tensors.get(idx)),
        )
    }

    fn total_fields(&self) -> u32 {
        self.inner.id_map.total_count
    }

    fn items(&self) -> impl Iterator<Item = (u32, Vec<Self::Field<'_>>)> + '_ {
        self.inner.id_map.ids().iter().filter_map(|&id| {
            let embeddings = self.fields(id)?.collect();
            Some((id, embeddings))
        })
    }

    fn max_fields(&self) -> u16 {
        self.inner.id_map.max_count
    }
}

#[derive(Debug)]
struct EmbeddingsWithDataInner {
    data: Data,
    tensors: Tensors,
}

#[derive(Debug, Clone)]
pub struct EmbeddingsWithData {
    inner: Arc<EmbeddingsWithDataInner>,
}

impl EmbeddingsWithData {
    pub fn load(data: Data, embedding_path: &Path) -> Result<Self> {
        let tensors = Tensors::load(embedding_path)?;

        if data.total_fields() as usize != tensors.len() {
            return Err(anyhow!(
                "Number of embeddings ({}) does not match number of data fields ({})",
                tensors.len(),
                data.total_fields()
            ));
        }

        Ok(Self {
            inner: Arc::new(EmbeddingsWithDataInner { data, tensors }),
        })
    }

    /// Get the number of dimensions
    pub fn num_dimensions(&self) -> usize {
        self.inner.tensors.num_dimensions()
    }

    /// Get the model name from metadata
    pub fn model(&self) -> Option<&str> {
        self.inner.tensors.model.as_deref()
    }

    /// Get the provider from metadata
    pub fn provider(&self) -> Option<&str> {
        self.inner.tensors.provider.as_deref()
    }

    /// Get the modalities from metadata
    pub fn modality(&self) -> &[EmbeddingModality] {
        &self.inner.tensors.modality
    }

    pub fn embedding_items(&self) -> impl Iterator<Item = (u32, Vec<EmbeddingRef<'_>>)> + '_ {
        let data_map = self.inner.data.data_map();
        let mut start = 0;
        (0..data_map.len()).filter_map(move |index| {
            let count = data_map.count(index)? as usize;
            let end = start + count;

            let embeddings = (start..end)
                .map(|tensor_idx| self.inner.tensors.get(tensor_idx))
                .collect::<Option<_>>()?;

            start = end;

            Some((index as u32, embeddings))
        })
    }

    pub fn field_embedding(&self, field_id: usize) -> Option<EmbeddingRef<'_>> {
        self.inner.tensors.get(field_id)
    }

    pub fn id_from_identifier(&self, identifier: &str) -> Option<u32> {
        self.inner.data.id_from_identifier(identifier)
    }

    pub fn data(&self) -> &Data {
        &self.inner.data
    }
}

impl DataSource for EmbeddingsWithData {
    type Field<'a> = FieldRef<'a>;

    fn len(&self) -> usize {
        self.inner.data.len()
    }

    fn num_fields(&self, id: u32) -> Option<u16> {
        self.inner.data.num_fields(id)
    }

    fn field(&self, id: u32, field: usize) -> Option<Self::Field<'_>> {
        self.inner.data.field(id, field)
    }

    fn fields(&self, id: u32) -> Option<impl Iterator<Item = Self::Field<'_>>> {
        self.inner.data.fields(id)
    }

    fn total_fields(&self) -> u32 {
        self.inner.data.total_fields()
    }

    fn items(&self) -> impl Iterator<Item = (u32, Vec<Self::Field<'_>>)> + '_ {
        self.inner.data.items()
    }

    fn max_fields(&self) -> u16 {
        self.inner.data.max_fields()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::tempdir;

    fn create_test_safetensors(
        path: &Path,
        embeddings: Vec<Vec<f32>>,
        ids: Vec<u32>,
    ) -> Result<()> {
        use safetensors::serialize;
        use safetensors::tensor::{Dtype, TensorView};

        assert_eq!(embeddings.len(), ids.len());

        let num_embeddings = embeddings.len();
        let num_dimensions = embeddings[0].len();

        // Flatten embeddings into single vector
        let embedding_data: Vec<f32> = embeddings.into_iter().flatten().collect();
        let embedding_bytes: Vec<u8> = embedding_data
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        // Convert IDs to bytes
        let id_bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();

        // Create tensors
        let embedding_tensor = TensorView::new(
            Dtype::F32,
            vec![num_embeddings, num_dimensions],
            &embedding_bytes,
        )?;
        let id_tensor = TensorView::new(Dtype::U32, vec![num_embeddings], &id_bytes)?;

        // Serialize with model metadata
        let tensors = vec![("embedding", embedding_tensor), ("id", id_tensor)];
        let bytes = serialize(
            tensors,
            Some(HashMap::from([
                (String::from("model"), String::from("test-model")),
                (String::from("provider"), String::from("test-provider")),
                (String::from("modality"), String::from("text")),
            ])),
        )?;

        // Write to file
        std::fs::write(path, bytes)?;

        Ok(())
    }

    #[test]
    fn test_embeddings_load() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).expect("Failed to create data dir");

        let embedding_path = data_dir.join("embedding.safetensors");

        // Create test embeddings with sorted IDs (required for OrderedIdMap)
        let embeddings = vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.5, 0.6, 0.7, 0.8],
            vec![0.9, 1.0, 1.1, 1.2],
        ];
        let ids = vec![100, 200, 300]; // Must be sorted for OrderedIdMap

        create_test_safetensors(&embedding_path, embeddings.clone(), ids.clone())
            .expect("Failed to create safetensors");

        // Build and load
        Embeddings::build(&data_dir).expect("Failed to build");
        let data = Embeddings::load(&data_dir).expect("Failed to load");

        // Test basic properties
        assert_eq!(data.len(), 3);
        assert_eq!(data.num_dimensions(), 4);
        // Precision is now handled at index level, data is always Float32
        assert_eq!(data.model(), Some("test-model"));
        assert_eq!(data.provider(), Some("test-provider"));
        assert_eq!(data.modality(), &[EmbeddingModality::Text]);

        // Test DataSource trait - field(id, field_idx) returns embeddings for that ID
        assert_eq!(data.num_fields(100), Some(1));
        assert_eq!(data.num_fields(200), Some(1));
        assert_eq!(data.num_fields(300), Some(1));
        assert_eq!(data.num_fields(999), None);

        // Test field access - returns the embedding at field_idx for the given id
        // EmbeddingRef is now just &[f32]
        if let Some(embedding) = data.field(100, 0) {
            assert_eq!(embedding.len(), 4);
            assert!((embedding[0] - 0.1).abs() < 1e-6);
            assert!((embedding[1] - 0.2).abs() < 1e-6);
        } else {
            panic!("Expected embedding for id 100");
        }

        if let Some(embedding) = data.field(200, 0) {
            assert_eq!(embedding.len(), 4);
            assert!((embedding[0] - 0.5).abs() < 1e-6);
            assert!((embedding[1] - 0.6).abs() < 1e-6);
        } else {
            panic!("Expected embedding for id 200");
        }

        if let Some(embedding) = data.field(300, 0) {
            assert_eq!(embedding.len(), 4);
            assert!((embedding[0] - 0.9).abs() < 1e-6);
            assert!((embedding[1] - 1.0).abs() < 1e-6);
        } else {
            panic!("Expected embedding for id 300");
        }
    }

    #[test]
    fn test_embeddings_optional_metadata() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).expect("Failed to create data dir");

        let embedding_path = data_dir.join("embedding.safetensors");

        use safetensors::serialize;
        use safetensors::tensor::{Dtype, TensorView};

        let embeddings = vec![vec![1.0, 2.0]];
        let ids = [1u32];

        let embedding_data: Vec<f32> = embeddings.into_iter().flatten().collect();
        let embedding_bytes: Vec<u8> = embedding_data
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let id_bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();

        let embedding_tensor = TensorView::new(Dtype::F32, vec![1, 2], &embedding_bytes).unwrap();
        let id_tensor = TensorView::new(Dtype::U32, vec![1], &id_bytes).unwrap();

        // No metadata at all
        let tensors = vec![("embedding", embedding_tensor), ("id", id_tensor)];
        let bytes = serialize(tensors.clone(), None).unwrap();
        std::fs::write(&embedding_path, bytes).unwrap();

        Embeddings::build(&data_dir).expect("Should succeed without metadata");
        let data = Embeddings::load(&data_dir).expect("Should load without metadata");
        assert_eq!(data.model(), None);
        assert_eq!(data.provider(), None);
        assert!(data.modality().is_empty());

        // Empty metadata
        let bytes = serialize(tensors, Some(HashMap::new())).unwrap();
        std::fs::write(&embedding_path, bytes).unwrap();

        Embeddings::build(&data_dir).expect("Should succeed with empty metadata");
        let data = Embeddings::load(&data_dir).expect("Should load with empty metadata");
        assert_eq!(data.model(), None);
        assert_eq!(data.provider(), None);
        assert!(data.modality().is_empty());
    }

    #[test]
    fn test_embeddings_validation_mismatched_lengths() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).expect("Failed to create data dir");

        let embedding_path = data_dir.join("embedding.safetensors");

        use safetensors::serialize;
        use safetensors::tensor::{Dtype, TensorView};

        let embeddings = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let ids = [1u32, 2, 3]; // Wrong length!

        let embedding_data: Vec<f32> = embeddings.into_iter().flatten().collect();
        let embedding_bytes: Vec<u8> = embedding_data
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let id_bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();

        let embedding_tensor = TensorView::new(Dtype::F32, vec![2, 2], &embedding_bytes).unwrap();
        let id_tensor = TensorView::new(Dtype::U32, vec![3], &id_bytes).unwrap();

        let tensors = vec![("embedding", embedding_tensor), ("id", id_tensor)];
        let bytes = serialize(
            tensors,
            Some(HashMap::from([(
                String::from("model"),
                String::from("test"),
            )])),
        )
        .unwrap();
        std::fs::write(&embedding_path, bytes).unwrap();

        let result = Embeddings::build(&data_dir);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("ID tensor length (3) does not match embedding length (2)")
        );
    }
}
