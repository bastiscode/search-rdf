use crate::data::DataSource;
use crate::data::embedding::{EmbeddingRef, Precision};
use crate::data::text::embedding::TextEmbeddings;
use crate::index::embedding::Metadata;
use crate::index::{EmbeddingIndexParams, Match, SearchIndex, SearchParams};
use crate::utils::load_u32_vec;
use crate::utils::{load_json, write_json};
use anyhow::{Result, anyhow};
use ordered_float::OrderedFloat;
use std::cmp::Reverse;
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use usearch::ffi::IndexOptions;
use usearch::{Index, b1x8};

struct Inner {
    data: TextEmbeddings,
    index: Index,
    field_to_data: Vec<u32>,
    metadata: Metadata,
}

pub struct TextEmbeddingIndex {
    inner: Arc<Inner>,
}

impl TextEmbeddingIndex {
    #[inline]
    fn filter_internal(&self, filter: impl Fn(u32) -> bool) -> impl Fn(u64) -> bool {
        move |field_id: u64| {
            let Some(data_id) = self.inner.field_to_data.get(field_id as usize).copied() else {
                return false;
            };
            filter(data_id)
        }
    }

    fn field_to_column(&self, field_id: u32) -> usize {
        let field_to_data = &self.inner.field_to_data;

        let mut field_id = field_id as usize;
        let data_id = field_to_data[field_id] as usize;

        let mut offset = 0;
        while field_id > 0 && field_to_data[field_id - 1] == data_id as u32 {
            field_id -= 1;
            offset += 1;
        }
        offset
    }

    fn search_internal<F>(
        &self,
        embedding: &EmbeddingRef<'_>,
        params: SearchParams,
        filter: Option<F>,
    ) -> Result<Vec<Match>>
    where
        F: Fn(u32) -> bool,
    {
        let data = &self.inner.data;
        let index = &self.inner.index;

        let num_dimensions = data.num_dimensions();
        let search_k = params.search_k(data);

        let predicate = filter.map(|f| self.filter_internal(f));

        // Validate embedding matches index precision and dimensions
        let results = match (data.precision(), embedding) {
            (Precision::Float32, EmbeddingRef::F32(vec)) => {
                if vec.len() != data.num_dimensions() {
                    return Err(anyhow!(
                        "Query embedding has {} dimensions, expected {}",
                        vec.len(),
                        data.num_dimensions()
                    ));
                }

                if let Some(pred) = predicate {
                    if params.exact {
                        return Err(anyhow!("Exact search with filter is not supported yet"));
                    }
                    index.filtered_search(vec, search_k, pred)?
                } else if params.exact {
                    index.exact_search(vec, search_k)?
                } else {
                    index.search(vec, search_k)?
                }
            }
            (Precision::UBinary, EmbeddingRef::Binary(bytes)) => {
                let expected_bytes = data.num_dimensions().div_ceil(8);
                if bytes.len() != expected_bytes {
                    return Err(anyhow!(
                        "Query embedding has {} bytes, expected {} ({} bits)",
                        bytes.len(),
                        expected_bytes,
                        data.num_dimensions()
                    ));
                }

                let query = b1x8::from_u8s(bytes);
                if let Some(pred) = predicate {
                    if params.exact {
                        return Err(anyhow!("Exact search with filter is not supported yet"));
                    }
                    index.filtered_search(query, search_k, pred)?
                } else if params.exact {
                    index.exact_search(query, search_k)?
                } else {
                    index.search(query, search_k)?
                }
            }
            _ => {
                return Err(anyhow!(
                    "Query embedding type does not match index precision"
                ));
            }
        };

        let mut matches: Vec<_> = results
            .keys
            .iter()
            .zip(results.distances.iter())
            .filter_map(|(&field_id, &distance)| {
                let data_id = self.inner.field_to_data[field_id as usize];

                // usearch returns distances (lower is better) for all metrics.
                // Convert to a score where higher is better.
                let score = self
                    .inner
                    .metadata
                    .params
                    .metric
                    .to_score(distance, num_dimensions);

                // Apply min_score filter
                if let Some(min_score) = params.min_score
                    && score < min_score
                {
                    return None;
                }

                Some((data_id, field_id, score))
            })
            .collect();

        // Sort by data_id, then by score descending, then by field_id
        matches.sort_by_key(|&(data_id, field_id, score)| {
            (data_id, Reverse(OrderedFloat(score)), field_id)
        });

        // Deduplicate by data_id (keeping the best score)
        matches.dedup_by(|a, b| a.0 == b.0);

        // Sort by score descending, then by data_id
        matches.sort_by_key(|&(data_id, _field_id, score)| (Reverse(OrderedFloat(score)), data_id));

        // Take top k
        let matches: Vec<Match> = matches
            .into_iter()
            .map(|(data_id, field_id, score)| {
                Match::WithField(data_id, self.field_to_column(field_id as u32), score)
            })
            .take(params.k)
            .collect();

        Ok(matches)
    }
}

pub enum Query<'e> {
    String(&'e str),
    Embedding(EmbeddingRef<'e>),
}

impl SearchIndex for TextEmbeddingIndex {
    type Data = TextEmbeddings;
    type Query<'q> = Query<'q>;
    type BuildParams = EmbeddingIndexParams;

    fn build(data: &Self::Data, index_dir: &Path, params: Self::BuildParams) -> Result<()> {
        // Validate metric is compatible with precision
        params.metric.validate_precision(data.precision())?;

        create_dir_all(index_dir)?;

        // Create usearch index
        let num_dimensions = data.num_dimensions();
        let precision = data.precision();

        let options = IndexOptions {
            dimensions: num_dimensions,
            metric: params.metric.to_usearch_metric(),
            quantization: precision.to_usearch_scalar_kind(),
            connectivity: params.connectivity,
            expansion_add: params.expansion_add,
            expansion_search: params.expansion_search,
            multi: false,
        };

        let index = Index::new(&options)?;

        index.reserve(data.total_fields())?;

        let mut field_to_data_file =
            BufWriter::new(File::create(index_dir.join("index.field-to-data"))?);
        let mut field_id: u32 = 0;
        for (id, embeddings) in data.embedding_items() {
            for emb in embeddings {
                match emb {
                    EmbeddingRef::F32(embedding) => {
                        index.add(field_id as u64, embedding)?;
                    }
                    EmbeddingRef::Binary(embedding) => {
                        index.add(field_id as u64, b1x8::from_u8s(embedding))?;
                    }
                }
                if field_id == u32::MAX {
                    return Err(anyhow!("too many fields, max {} supported", u32::MAX));
                }
                field_id += 1;
                field_to_data_file.write_all(&id.to_le_bytes())?;
            }
        }

        // Save the index
        let index_file = index_dir.join("index.usearch");
        index.save(index_file.to_str().ok_or_else(|| anyhow!("Invalid path"))?)?;

        // Save metadata as JSON
        let metadata = Metadata {
            params,
            precision,
            dimensions: num_dimensions,
        };
        write_json(&index_dir.join("index.metadata"), &metadata)?;

        Ok(())
    }

    fn load(data: Self::Data, index_dir: &Path) -> Result<Self> {
        // Load metadata from JSON
        let metadata: Metadata = load_json(&index_dir.join("index.metadata"))?;

        // Validate data precision matches index precision
        if data.precision() != metadata.precision {
            return Err(anyhow!(
                "Data precision {:?} does not match index precision {:?}",
                data.precision(),
                metadata.precision
            ));
        }

        // Load the index
        let index_file = index_dir.join("index.usearch");

        let index = Index::new(&IndexOptions {
            dimensions: metadata.dimensions,
            metric: metadata.params.metric.to_usearch_metric(),
            quantization: metadata.precision.to_usearch_scalar_kind(),
            connectivity: metadata.params.connectivity,
            expansion_add: metadata.params.expansion_add,
            expansion_search: metadata.params.expansion_search,
            multi: false,
        })?;

        index.view(index_file.to_str().ok_or_else(|| anyhow!("Invalid path"))?)?;

        let field_to_data = load_u32_vec(&index_dir.join("index.field-to-data"))?;

        Ok(Self {
            inner: Arc::new(Inner {
                data,
                index,
                field_to_data,
                metadata,
            }),
        })
    }

    fn data(&self) -> &Self::Data {
        &self.inner.data
    }

    fn index_type(&self) -> &'static str {
        "TextEmbeddingIndex"
    }

    fn search(&self, query: &Self::Query<'_>, params: SearchParams) -> Result<Vec<Match>> {
        match query {
            Query::Embedding(embedding) => {
                self.search_internal(embedding, params, None::<fn(u32) -> bool>)
            }
            Query::String(_) => Err(anyhow!("String queries are not supported yet")),
        }
    }

    fn search_with_filter<F>(
        &self,
        query: &Self::Query<'_>,
        params: SearchParams,
        filter: F,
    ) -> Result<Vec<Match>>
    where
        F: Fn(u32) -> bool,
    {
        match query {
            Query::Embedding(embedding) => self.search_internal(embedding, params, Some(filter)),
            Query::String(_) => Err(anyhow!("String queries are not supported yet")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::text::item::TextItem;
    use crate::data::text::{TextData, embedding::TextEmbeddings};
    use crate::index::EmbeddingIndexParams;
    use crate::index::embedding::Metric;
    use std::collections::HashMap;
    use std::fs::create_dir_all;
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

    fn normalize(vec: &mut [f32]) {
        let norm: f32 = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }

    #[test]
    fn test_embedding_index_cosine() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let data_dir = temp_dir.path().join("data");
        let index_dir = temp_dir.path().join("index");

        create_dir_all(&index_dir).expect("Failed to create index dir");

        // Create test data (3 rows, 2 fields each = 6 total fields)
        let items = vec![
            Ok(TextItem::new(
                "Q1".to_string(),
                vec!["Cat".to_string(), "Feline".to_string()],
            )
            .expect("Failed to create TextItem")),
            Ok(TextItem::new(
                "Q2".to_string(),
                vec!["Dog".to_string(), "Canine".to_string()],
            )
            .expect("Failed to create TextItem")),
            Ok(TextItem::new(
                "Q3".to_string(),
                vec!["Bird".to_string(), "Avian".to_string()],
            )
            .expect("Failed to create TextItem")),
        ];

        // Build TextData
        TextData::build(items, &data_dir).expect("Failed to build TextData");
        let data = TextData::load(&data_dir).expect("Failed to load TextData");

        // Create normalized test embeddings (6 embeddings, 4 dimensions each)
        let mut embeddings = vec![
            vec![1.0, 0.0, 0.0, 0.0], // Q1 name (Cat)
            vec![0.9, 0.1, 0.0, 0.0], // Q1 alias (Feline) - similar to Cat
            vec![0.0, 1.0, 0.0, 0.0], // Q2 name (Dog)
            vec![0.0, 0.9, 0.1, 0.0], // Q2 alias (Canine) - similar to Dog
            vec![0.0, 0.0, 1.0, 0.0], // Q3 name (Bird)
            vec![0.0, 0.0, 0.9, 0.1], // Q3 alias (Avian) - similar to Bird
        ];

        // Normalize all embeddings
        for emb in &mut embeddings {
            normalize(emb);
        }

        let embeddings_file = data_dir.join("embedding.safetensors");
        create_test_safetensors(&embeddings_file, embeddings.clone())
            .expect("Failed to create safetensors");

        // Load data
        let data = TextEmbeddings::load(data, &embeddings_file).expect("Failed to load data");

        let params = EmbeddingIndexParams::from_precision(data.precision());

        TextEmbeddingIndex::build(&data, &index_dir, params).expect("Failed to build index");
        let index = TextEmbeddingIndex::load(data, &index_dir).expect("Failed to load index");

        // Test search - query with Cat-like embedding
        let mut query = vec![1.0, 0.0, 0.0, 0.0];
        normalize(&mut query);

        let results = index
            .search(
                &Query::Embedding(EmbeddingRef::F32(&query)),
                SearchParams::default(),
            )
            .expect("Failed to search");

        // Should find Q1 (Cat) as top result
        assert!(!results.is_empty());

        // The query [1.0, 0.0, 0.0, 0.0] should match Q1's embeddings best
        // Check that Q1 (id=0) is in the top results
        let top_ids: Vec<u32> = results
            .iter()
            .take(3)
            .map(|m| match m {
                Match::WithField(id, _, _) => *id,
                _ => panic!("Expected Match::WithField"),
            })
            .collect();
        assert!(
            top_ids.contains(&0),
            "Expected Q1 (id=0) in top 3 results, got {:?}",
            results.iter().take(3).collect::<Vec<_>>()
        );

        // Test search with filter - exclude Q1
        let results_filtered = index
            .search_with_filter(
                &Query::Embedding(EmbeddingRef::F32(&query)),
                SearchParams::default(),
                |id| id != 0,
            )
            .expect("Failed to search with filter");

        // Should find Q2 or Q3, but not Q1
        assert!(!results_filtered.is_empty());
        if let Match::WithField(id, _, _) = results_filtered[0] {
            assert_ne!(id, 0);
        } else {
            panic!("Expected Match::WithField");
        }
    }

    #[test]
    fn test_embedding_index_inner_product() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let data_dir = temp_dir.path().join("data");
        let index_dir = temp_dir.path().join("index");

        create_dir_all(&index_dir).expect("Failed to create index dir");

        // Create test data
        let items = vec![
            Ok(TextItem::new("Q1".to_string(), vec!["Item1".to_string()])
                .expect("Failed to create TextItem")),
            Ok(TextItem::new("Q2".to_string(), vec!["Item2".to_string()])
                .expect("Failed to create TextItem")),
        ];

        // Build TextData
        TextData::build(items, &data_dir).expect("Failed to build TextData");
        let data = TextData::load(&data_dir).expect("Failed to load TextData");

        // Create test embeddings (not normalized)
        let embeddings = vec![
            vec![1.0, 2.0, 3.0, 4.0], // Q1
            vec![4.0, 3.0, 2.0, 1.0], // Q2
        ];

        let embeddings_file = data_dir.join("embedding.safetensors");
        create_test_safetensors(&embeddings_file, embeddings.clone())
            .expect("Failed to create safetensors");

        // Load data and build index
        let data = TextEmbeddings::load(data, &embeddings_file).expect("Failed to load data");

        let params = EmbeddingIndexParams::default().with_metric(Metric::InnerProduct);

        TextEmbeddingIndex::build(&data, &index_dir, params).expect("Failed to build index");
        let index = TextEmbeddingIndex::load(data, &index_dir).expect("Failed to load index");

        // Test search
        let query = vec![1.0, 1.0, 1.0, 1.0];

        let results = index
            .search(
                &Query::Embedding(EmbeddingRef::F32(&query)),
                SearchParams::default(),
            )
            .expect("Failed to search");

        assert!(!results.is_empty());
        // Q1: 1*1 + 2*1 + 3*1 + 4*1 = 10
        // Q2: 4*1 + 3*1 + 2*1 + 1*1 = 10
        // Should get both with same score
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_max_fields_per_data_point() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let data_dir = temp_dir.path().join("data");
        let index_dir = temp_dir.path().join("index");

        create_dir_all(&index_dir).expect("Failed to create index dir");

        // Create test data with varying number of fields
        // Q1: 1 field, Q2: 3 fields, Q3: 2 fields
        // Max should be 3
        let items = vec![
            Ok(TextItem::new("Q1".to_string(), vec!["A".to_string()])
                .expect("Failed to create TextItem")),
            Ok(TextItem::new(
                "Q2".to_string(),
                vec!["B".to_string(), "C".to_string(), "D".to_string()],
            )
            .expect("Failed to create TextItem")),
            Ok(
                TextItem::new("Q3".to_string(), vec!["E".to_string(), "F".to_string()])
                    .expect("Failed to create TextItem"),
            ),
        ];

        // Build TextData
        TextData::build(items, &data_dir).expect("Failed to build TextData");
        let data = TextData::load(&data_dir).expect("Failed to load TextData");

        // Create normalized embeddings (6 total: 1 + 3 + 2)
        let mut embeddings = vec![
            vec![1.0, 0.0, 0.0, 0.0], // Q1 field1
            vec![0.0, 1.0, 0.0, 0.0], // Q2 field1
            vec![0.0, 0.0, 1.0, 0.0], // Q2 field2
            vec![0.0, 0.0, 0.0, 1.0], // Q2 field3
            vec![1.0, 1.0, 0.0, 0.0], // Q3 field1
            vec![0.0, 1.0, 1.0, 0.0], // Q3 field2
        ];

        for emb in &mut embeddings {
            normalize(emb);
        }

        let embeddings_file = data_dir.join("embedding.safetensors");
        create_test_safetensors(&embeddings_file, embeddings.clone())
            .expect("Failed to create safetensors");

        // Load data and build index
        let data = TextEmbeddings::load(data, &embeddings_file).expect("Failed to load data");

        let params = EmbeddingIndexParams::from_precision(data.precision());

        TextEmbeddingIndex::build(&data, &index_dir, params).expect("Failed to build index");
        let index = TextEmbeddingIndex::load(data, &index_dir).expect("Failed to load index");

        // Verify max_fields_per_id is 3 (from Q2)
        assert_eq!(index.data().text_data().max_fields_per_id(), 3);
    }
}
