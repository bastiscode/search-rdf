use crate::data::Embeddings;
use crate::index::SearchParams;
use crate::{
    data::{
        DataSource,
        embedding::{Embedding, Precision},
    },
    index::{Match, SearchIndex},
};
use anyhow::{Result, anyhow};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::fs::{File, create_dir_all};
use std::io::BufWriter;
use std::path::Path;
use std::sync::Arc;
use usearch::ffi::MetricKind;
use usearch::ffi::{IndexOptions, ScalarKind};
use usearch::{Index, b1x8};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    CosineNormalized,
    Cosine,
    InnerProduct,
    L2,
    Hamming,
}

impl Metric {
    pub fn to_score(&self, distance: f32, num_dimensions: usize) -> f32 {
        match self {
            // [0, 2] where 0 is identical, 2 is opposite
            // to [-1, 1] where 1 is identical, -1 is opposite
            Metric::Cosine | Metric::CosineNormalized => 1.0 - distance,
            // [0, inf) where 0 is identical
            // to (0, 1] where 1 is identical
            Metric::L2 => 1.0 / (1.0 + distance),
            // (-inf, inf) where higher is better
            Metric::InnerProduct => -distance,
            // [0, num_dimensions] where 0 is identical
            // to [0, 1] where 1 is identical
            Metric::Hamming => 1.0 - (distance / num_dimensions as f32),
        }
    }

    pub fn to_usearch_metric(self) -> MetricKind {
        match self {
            Metric::Cosine => MetricKind::Cos,
            Metric::CosineNormalized | Metric::InnerProduct => MetricKind::IP,
            Metric::L2 => MetricKind::L2sq,
            Metric::Hamming => MetricKind::Hamming,
        }
    }

    pub fn validate_precision(&self, precision: Precision) -> Result<()> {
        match (self, precision) {
            (Metric::Hamming, Precision::UBinary) => Ok(()),
            (Metric::Hamming, Precision::Float32) => {
                Err(anyhow!("Hamming metric only works with binary embeddings"))
            }
            (
                Metric::Cosine | Metric::CosineNormalized | Metric::InnerProduct | Metric::L2,
                Precision::Float32,
            ) => Ok(()),
            (
                Metric::Cosine | Metric::CosineNormalized | Metric::InnerProduct | Metric::L2,
                Precision::UBinary,
            ) => Err(anyhow!(
                "Cosine/CosineNormalized/InnerProduct/L2 metrics only work with F32 embeddings"
            )),
        }
    }

    pub fn default_for_precision(precision: Precision) -> Self {
        match precision {
            Precision::Float32 => Metric::CosineNormalized,
            Precision::UBinary => Metric::Hamming,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Metadata {
    pub metric: Metric,
    pub precision: Precision,
    pub dimensions: usize,
}

#[derive(Debug, Clone)]
pub struct EmbeddingParams {
    /// Metric to use for similarity search
    pub metric: Metric,
    /// Usearch index options
    pub connectivity: usize,
    pub expansion_add: usize,
    pub expansion_search: usize,
}

impl EmbeddingParams {
    pub fn from_precision(precision: Precision) -> Self {
        Self {
            metric: Metric::default_for_precision(precision),
            connectivity: 16,
            expansion_add: 128,
            expansion_search: 64,
        }
    }

    pub fn with_metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    pub fn with_connectivity(mut self, connectivity: usize) -> Self {
        self.connectivity = connectivity;
        self
    }

    pub fn with_expansion_add(mut self, expansion_add: usize) -> Self {
        self.expansion_add = expansion_add;
        self
    }

    pub fn with_expansion_search(mut self, expansion_search: usize) -> Self {
        self.expansion_search = expansion_search;
        self
    }
}

impl Default for EmbeddingParams {
    fn default() -> Self {
        Self {
            metric: Metric::CosineNormalized,
            connectivity: 16,
            expansion_add: 128,
            expansion_search: 64,
        }
    }
}

struct Inner {
    data: Embeddings,
    index: Index,
}

pub struct EmbeddingIndex {
    inner: Arc<Inner>,
}

impl EmbeddingIndex {
    fn search_internal<'e, F>(
        &self,
        embedding: Embedding<'e>,
        params: SearchParams<F>,
    ) -> Result<Vec<Match>>
    where
        F: Fn(u32) -> bool,
    {
        let data = &self.inner.data;
        let index = &self.inner.index;

        let search_k = params.search_k(data);

        let predicate = params.filter.map(|f| move |id| f(id as u32));

        // Validate embedding matches index precision and dimensions
        let results = match (self.inner.data.precision(), embedding) {
            (Precision::Float32, Embedding::F32(vec)) => {
                if vec.len() != self.inner.data.num_dimensions() {
                    return Err(anyhow!(
                        "Query embedding has {} dimensions, expected {}",
                        vec.len(),
                        self.inner.data.num_dimensions()
                    ));
                }

                if let Some(ref pred) = predicate {
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
            (Precision::UBinary, Embedding::Binary(bytes)) => {
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
                if let Some(ref pred) = predicate {
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
            .filter_map(|(&id, &distance)| {
                let id = id as u32;

                // usearch returns distances (lower is better) for all metrics.
                // Convert to a similarity score where higher is better.
                let score = -distance;

                // Apply min_score filter
                if let Some(min_score) = params.min_score
                    && score < min_score
                {
                    return None;
                }

                Some((id, score))
            })
            .collect();

        // Deduplicate by ID (keeping the best score for each ID)
        matches.sort_by_key(|&(id, score)| (id, Reverse(OrderedFloat(score))));
        matches.dedup_by(|a, b| a.0 == b.0);

        // Sort by score descending, then by ID
        matches.sort_by_key(|&(id, score)| (Reverse(OrderedFloat(score)), id));

        // Take top k and create Match objects
        let matches: Vec<Match> = matches
            .into_iter()
            .map(|(id, score)| Match::Regular(id, score))
            .take(params.k)
            .collect();

        Ok(matches)
    }
}

impl SearchIndex for EmbeddingIndex {
    type Data = Embeddings;
    type Query<'q> = Embedding<'q>;
    type BuildParams = EmbeddingParams;

    fn build(data: &Self::Data, index_dir: &Path, params: Self::BuildParams) -> Result<()> {
        // Validate metric is compatible with precision
        params.metric.validate_precision(data.precision())?;

        create_dir_all(index_dir)?;

        // Create usearch index
        let dimensions = data.num_dimensions();
        let precision = data.precision();

        let scalar_kind = match precision {
            Precision::Float32 => ScalarKind::F32,
            Precision::UBinary => ScalarKind::B1,
        };

        let options = IndexOptions {
            dimensions,
            metric: params.metric.to_usearch_metric(),
            quantization: scalar_kind,
            connectivity: params.connectivity,
            expansion_add: params.expansion_add,
            expansion_search: params.expansion_search,
            multi: true, // Enable multi-index to support duplicate IDs
        };

        let index = Index::new(&options)?;

        index.reserve(data.total_fields())?;

        // Add all embeddings to the index using their IDs as keys (not indices)
        // Multiple embeddings can have the same ID
        for (id, embeddings) in data.items() {
            for emb in embeddings {
                match emb {
                    Embedding::F32(embedding) => {
                        index.add(id as u64, embedding)?;
                    }
                    Embedding::Binary(embedding) => {
                        index.add(id as u64, b1x8::from_u8s(embedding))?;
                    }
                }
            }
        }

        // Save the index
        let index_file = index_dir.join("index.usearch");
        index.save(index_file.to_str().ok_or_else(|| anyhow!("Invalid path"))?)?;

        // Save metadata as JSON
        let metadata = Metadata {
            metric: params.metric,
            precision,
            dimensions,
        };
        let metadata_file = index_dir.join("index.metadata");
        let mut writer = BufWriter::new(File::create(&metadata_file)?);
        serde_json::to_writer_pretty(&mut writer, &metadata)?;

        Ok(())
    }

    fn load(data: Self::Data, index_dir: &Path) -> Result<Self> {
        // Load metadata from JSON
        let metadata_file = index_dir.join("index.metadata");
        let metadata: Metadata = serde_json::from_reader(File::open(&metadata_file)?)?;

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
            dimensions: data.num_dimensions(),
            metric: metadata.metric.to_usearch_metric(),
            quantization: match metadata.precision {
                Precision::Float32 => ScalarKind::F32,
                Precision::UBinary => ScalarKind::B1,
            },
            connectivity: 16,
            expansion_add: 128,
            expansion_search: 64,
            multi: true, // Enable multi-index to support duplicate IDs
        })?;

        index.view(index_file.to_str().ok_or_else(|| anyhow!("Invalid path"))?)?;

        Ok(Self {
            inner: Arc::new(Inner { data, index }),
        })
    }

    fn data(&self) -> &Self::Data {
        &self.inner.data
    }

    fn index_type(&self) -> &'static str {
        "EmbeddingIndex"
    }

    fn search<'q, F>(&self, query: &Self::Query<'q>, params: SearchParams<F>) -> Result<Vec<Match>>
    where
        F: Fn(u32) -> bool,
    {
        self.search_internal(*query, params)
    }
}

#[cfg(test)]
mod embedding_index_tests {
    use super::*;
    use crate::data::Embeddings;
    use std::collections::HashMap;
    use std::fs::create_dir_all;
    use tempfile::tempdir;

    fn create_test_safetensors(
        path: &Path,
        embeddings: Vec<Vec<f32>>,
        ids: Vec<u32>,
    ) -> Result<()> {
        use safetensors::serialize;
        use safetensors::tensor::{Dtype, TensorView};

        let num_embeddings = embeddings.len();
        let num_dimensions = embeddings[0].len();

        // Flatten embeddings into single vector
        let data: Vec<f32> = embeddings.into_iter().flatten().collect();

        // Convert to bytes
        let data_bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let id_bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();

        // Create tensors
        let embedding_tensor = TensorView::new(
            Dtype::F32,
            vec![num_embeddings, num_dimensions],
            &data_bytes,
        )?;
        let id_tensor = TensorView::new(Dtype::U32, vec![num_embeddings], &id_bytes)?;

        // Serialize
        let tensors = vec![("embedding", embedding_tensor), ("id", id_tensor)];
        let bytes = serialize(
            tensors,
            Some(HashMap::from([(
                String::from("model"),
                String::from("test-model"),
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

        create_dir_all(&data_dir).expect("Failed to create data dir");
        create_dir_all(&index_dir).expect("Failed to create index dir");

        let embeddings_file = data_dir.join("embedding.safetensors");

        // Create normalized test embeddings with duplicate IDs
        let mut embeddings = vec![
            vec![1.0, 0.0, 0.0, 0.0], // ID 100
            vec![0.9, 0.1, 0.0, 0.0], // ID 100 (duplicate - similar to first)
            vec![0.0, 1.0, 0.0, 0.0], // ID 200
            vec![0.0, 0.0, 1.0, 0.0], // ID 300
        ];

        for emb in &mut embeddings {
            normalize(emb);
        }

        let ids = vec![100, 100, 200, 300]; // ID 100 appears twice

        create_test_safetensors(&embeddings_file, embeddings.clone(), ids)
            .expect("Failed to create safetensors");

        // Build data and index
        Embeddings::build(&data_dir).expect("Failed to build data");
        let data = Embeddings::load(&data_dir).expect("Failed to load data");

        let params = EmbeddingParams::from_precision(data.precision());

        EmbeddingIndex::build(&data, &index_dir, params).expect("Failed to build index");
        let index = EmbeddingIndex::load(data, &index_dir).expect("Failed to load index");

        // Test search - query with first embedding
        let mut query = vec![1.0, 0.0, 0.0, 0.0];
        normalize(&mut query);

        let results = index
            .search(&Embedding::F32(&query), SearchParams::default())
            .expect("Failed to search");

        // Should find ID 100 as top result (deduped from two embeddings)
        assert!(!results.is_empty());
        if let Match::Regular(id, _score) = results[0] {
            assert_eq!(id, 100);
        } else {
            panic!("Expected Match::Regular");
        }

        // Verify deduplication: ID 100 should appear only once despite having 2 embeddings
        let id_100_count = results
            .iter()
            .filter(|m| matches!(m, Match::Regular(100, _)))
            .count();
        assert_eq!(id_100_count, 1);

        // Test search with filter - exclude ID 100
        let results_filtered = index
            .search(
                &Embedding::F32(&query),
                SearchParams::default().filter(|id| id != 100),
            )
            .expect("Failed to search with filter");

        // Should find ID 200 or 300, but not 100
        assert!(!results_filtered.is_empty());
        if let Match::Regular(id, _score) = results_filtered[0] {
            assert_ne!(id, 100);
        } else {
            panic!("Expected Match::Regular");
        }
    }

    #[test]
    fn test_embedding_index_inner_product() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let data_dir = temp_dir.path().join("data");
        let index_dir = temp_dir.path().join("index");

        create_dir_all(&data_dir).expect("Failed to create data dir");
        create_dir_all(&index_dir).expect("Failed to create index dir");

        let embeddings_file = data_dir.join("embedding.safetensors");

        // Create test embeddings (not normalized)
        let embeddings = vec![vec![1.0, 2.0, 3.0, 4.0], vec![4.0, 3.0, 2.0, 1.0]];

        let ids = vec![10, 20];

        create_test_safetensors(&embeddings_file, embeddings, ids)
            .expect("Failed to create safetensors");

        // Build data and index
        Embeddings::build(&data_dir).expect("Failed to build data");
        let data = Embeddings::load(&data_dir).expect("Failed to load data");

        let params = EmbeddingParams::default().with_metric(Metric::InnerProduct);

        EmbeddingIndex::build(&data, &index_dir, params).expect("Failed to build index");
        let index = EmbeddingIndex::load(data, &index_dir).expect("Failed to load index");

        // Test search
        let query = vec![1.0, 1.0, 1.0, 1.0];

        let results = index
            .search(&Embedding::F32(&query), SearchParams::default())
            .expect("Failed to search");

        assert!(!results.is_empty());
        // Both should have same score: 1*1 + 2*1 + 3*1 + 4*1 = 10
        assert_eq!(results.len(), 2);
    }
}
