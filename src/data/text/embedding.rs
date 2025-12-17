use crate::data::embedding::{Embedding, Precision, Tensors};

use super::{DataSource, TextData};
use anyhow::{Result, anyhow};
use std::path::Path;
use std::sync::Arc;

struct Inner {
    data: TextData,
    tensors: Tensors,
}

#[derive(Clone)]
pub struct TextEmbeddings {
    inner: Arc<Inner>,
}

impl TextEmbeddings {
    pub fn load(data_dir: &Path) -> Result<Self> {
        let data = TextData::load(data_dir)?;

        let embeddings_file = data_dir.join("embedding.safetensors");
        let tensors = Tensors::load(&embeddings_file)?;

        if data.total_fields() != tensors.len() {
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
    /// Get the precision of the embeddings
    pub fn precision(&self) -> Precision {
        self.inner.tensors.precision
    }

    /// Get the number of dimensions
    pub fn num_dimensions(&self) -> usize {
        self.inner.tensors.num_dimensions()
    }

    /// Get the model name from metadata
    pub fn model(&self) -> &str {
        &self.inner.tensors.model
    }

    pub fn embedding_items(&self) -> impl Iterator<Item = (u32, Vec<Embedding<'_>>)> + '_ {
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

    fn num_fields(&self, id: u32) -> Option<usize> {
        self.inner.data.num_fields(id)
    }

    fn field(&self, id: u32, field: usize) -> Option<Self::Field<'_>> {
        self.inner.data.field(id, field)
    }

    fn fields(&self, id: u32) -> Option<impl Iterator<Item = Self::Field<'_>>> {
        self.inner.data.fields(id)
    }

    fn data_type(&self) -> &'static str {
        "TextEmbeddings"
    }

    fn total_fields(&self) -> usize {
        self.inner.data.total_fields()
    }

    fn items(&self) -> impl Iterator<Item = (u32, Vec<Self::Field<'_>>)> + '_ {
        self.inner.data.items()
    }

    fn max_fields(&self) -> usize {
        self.inner.data.max_fields()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::fs::File;
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
        let rows = vec![
            ("Q1", "Universe\tCosmos"),
            ("Q2", "Earth\tWorld"),
            ("Q3", "Human\tPerson"),
        ];

        create_test_data_file(&data_dir, &rows).expect("Failed to create test data");

        // Build TextData
        TextData::build(&data_dir).expect("Failed to build TextData");

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
        let data = TextEmbeddings::load(&data_dir).expect("Failed to load");

        // Test basic properties
        assert_eq!(data.len(), 3);
        assert_eq!(data.num_dimensions(), 4);
        assert_eq!(data.precision(), Precision::Float32);
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
        if let Embedding::F32(emb) = embs[0] {
            assert!((emb[0] - 0.1).abs() < 1e-6);
            assert!((emb[1] - 0.2).abs() < 1e-6);
        }
        if let Embedding::F32(emb) = embs[1] {
            assert!((emb[0] - 0.5).abs() < 1e-6);
            assert!((emb[1] - 0.6).abs() < 1e-6);
        }
    }
}
