use crate::data::embedding::{EmbeddingRef, Tensors};

use super::{DataSource, TextData};
use anyhow::{Result, anyhow};
use std::path::Path;
use std::sync::Arc;

#[derive(Debug)]
struct Inner {
    data: TextData,
    tensors: Tensors,
}

#[derive(Debug, Clone)]
pub struct TextEmbeddings {
    inner: Arc<Inner>,
}

impl TextEmbeddings {
    pub fn load(data: TextData, embeddings_file: &Path) -> Result<Self> {
        let tensors = Tensors::load(embeddings_file)?;

        if data.total_fields() as usize != tensors.len() {
            return Err(anyhow!(
                "Number of embeddings ({}) does not match number of text fields ({})",
                tensors.len(),
                data.total_fields()
            ));
        }

        Ok(Self {
            inner: Arc::new(Inner { data, tensors }),
        })
    }

    /// Get the number of dimensions
    pub fn num_dimensions(&self) -> usize {
        self.inner.tensors.num_dimensions()
    }

    /// Get the model name from metadata
    pub fn model(&self) -> &str {
        &self.inner.tensors.model
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

    pub fn text_data(&self) -> &TextData {
        &self.inner.data
    }
}

impl DataSource for TextEmbeddings {
    type Field<'a> = &'a str;

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

    fn data_type(&self) -> &'static str {
        "text-embedding"
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
    use crate::data::item::Item;
    use std::collections::HashMap;
    use tempfile::tempdir;

    fn create_test_safetensors(path: &Path, embeddings: Vec<Vec<f32>>) -> Result<()> {
        use safetensors::serialize;
        use safetensors::tensor::{Dtype, TensorView};

        let num_embeddings = embeddings.len();
        let num_dimensions = embeddings[0].len();

        // Flatten embeddings into single vector
        let data: Vec<f32> = embeddings.into_iter().flatten().collect();

        // Convert to bytes
        let data_bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Create tensor
        let shape = vec![num_embeddings, num_dimensions];
        let tensor = TensorView::new(Dtype::F32, shape, &data_bytes)?;

        // Serialize
        let tensors = vec![("embedding", tensor)];
        let bytes = serialize(
            tensors,
            Some(HashMap::from([(
                String::from("model"),
                String::from("test"),
            )])),
        )?;

        // Write to file
        std::fs::write(path, bytes)?;

        Ok(())
    }

    #[test]
    fn test_text_embeddings_f32() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let data_dir = temp_dir.path().join("data");

        // Create test data (3 rows, 2 fields each = 6 total fields)
        let items = vec![
            Ok(Item::new(
                "Q1".to_string(),
                vec!["Universe".to_string(), "Cosmos".to_string()],
            )
            .expect("Failed to create Item")),
            Ok(Item::new(
                "Q2".to_string(),
                vec!["Earth".to_string(), "World".to_string()],
            )
            .expect("Failed to create Item")),
            Ok(Item::new(
                "Q3".to_string(),
                vec!["Human".to_string(), "Person".to_string()],
            )
            .expect("Failed to create Item")),
        ];

        // Build TextData
        TextData::build(items, &data_dir).expect("Failed to build TextData");
        let data = TextData::load(&data_dir).expect("Failed to load TextData");

        // Create test embeddings (6 embeddings, 4 dimensions each)
        let embeddings = vec![
            vec![0.1, 0.2, 0.3, 0.4], // Q1 name
            vec![0.5, 0.6, 0.7, 0.8], // Q1 alias
            vec![0.2, 0.3, 0.4, 0.5], // Q2 name
            vec![0.6, 0.7, 0.8, 0.9], // Q2 alias
            vec![0.3, 0.4, 0.5, 0.6], // Q3 name
            vec![0.7, 0.8, 0.9, 1.0], // Q3 alias
        ];

        let embeddings_file = data_dir.join("embedding.safetensors");
        create_test_safetensors(&embeddings_file, embeddings.clone())
            .expect("Failed to create safetensors");

        // Load TextEmbeddings
        let data = TextEmbeddings::load(data, &embeddings_file).expect("Failed to load");

        // Test basic properties
        assert_eq!(data.len(), 3);
        assert_eq!(data.num_dimensions(), 4);
        // Precision is now handled at index level, data is always Float32
        assert_eq!(data.model(), "test");

        // Test DataSource delegation (text data)
        assert_eq!(data.field(0, 0), Some("Universe"));
        assert_eq!(data.field(0, 1), Some("Cosmos"));
        assert_eq!(data.text_data().id_from_identifier("Q2"), Some(1));

        // Test embedding retrieval through embedding_items
        let items: Vec<_> = data.embedding_items().collect();
        assert_eq!(items.len(), 3); // 3 documents

        // Check Q1 embeddings (id=0)
        let (id, embs) = &items[0];
        assert_eq!(*id, 0);
        assert_eq!(embs.len(), 2); // 2 fields
        // EmbeddingRef is now just &[f32]
        let emb = embs[0];
        assert!((emb[0] - 0.1).abs() < 1e-6);
        assert!((emb[1] - 0.2).abs() < 1e-6);

        let emb = embs[1];
        assert!((emb[0] - 0.5).abs() < 1e-6);
        assert!((emb[1] - 0.6).abs() < 1e-6);
    }
}
