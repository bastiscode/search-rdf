use crate::data::{Embeddings, EmbeddingsWithData};
use crate::index::SearchParamsExt;
use crate::utils::{load_json, load_u32_vec, progress_bar, write_json};
use crate::{
    data::{
        DataSource,
        embedding::{EmbeddingRef, Precision},
    },
    index::{Match, Search},
};
use anyhow::{Result, anyhow};
use itertools::Itertools;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use serde_aux::prelude::*;
use std::cmp::Reverse;
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use usearch::ffi::IndexOptions;
use usearch::ffi::MetricKind;
use usearch::{Index, b1x8};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Metric {
    #[serde(rename = "cosine-normalized")]
    CosineNormalized,
    Cosine,
    #[serde(rename = "inner-product")]
    InnerProduct,
    L2,
    Hamming,
}

impl From<Metric> for MetricKind {
    fn from(metric: Metric) -> Self {
        match metric {
            Metric::CosineNormalized => MetricKind::IP,
            Metric::Cosine => MetricKind::Cos,
            Metric::InnerProduct => MetricKind::IP,
            Metric::L2 => MetricKind::L2sq,
            Metric::Hamming => MetricKind::Hamming,
        }
    }
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
        self.into()
    }

    pub fn validate_precision(&self, precision: Precision) -> Result<()> {
        match (self, precision) {
            (Metric::Hamming, Precision::Binary) => Ok(()),
            (Metric::Hamming, _) => Err(anyhow!("Hamming metric only works with binary precision")),
            (metric, Precision::Binary) => Err(anyhow!(
                "Metric {:?} does not work with binary precision",
                metric
            )),
            _ => Ok(()),
        }
    }

    pub fn default_for_precision(precision: Precision) -> Self {
        match precision {
            Precision::Binary => Metric::Hamming,
            _ => Metric::CosineNormalized,
        }
    }
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct EmbeddingIndexParams {
    /// Metric to use for similarity search
    #[serde(default = "default_metric")]
    pub metric: Metric,
    /// Precision to use for index
    #[serde(default = "default_precision")]
    pub precision: Precision,
    /// Usearch index options
    #[serde(default = "default_connectivity")]
    pub connectivity: usize,
    #[serde(default = "default_expansion_add")]
    pub expansion_add: usize,
    #[serde(default = "default_expansion_search")]
    pub expansion_search: usize,
}

fn default_metric() -> Metric {
    Metric::CosineNormalized
}

fn default_precision() -> Precision {
    Precision::BFloat16
}

fn default_connectivity() -> usize {
    16
}

fn default_expansion_add() -> usize {
    128
}

fn default_expansion_search() -> usize {
    64
}

impl EmbeddingIndexParams {
    pub fn from_precision(precision: Precision) -> Self {
        Self {
            metric: Metric::default_for_precision(precision),
            precision,
            ..Default::default()
        }
    }

    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.precision = precision;
        self
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

impl Default for EmbeddingIndexParams {
    fn default() -> Self {
        Self {
            metric: Metric::CosineNormalized,
            precision: Precision::BFloat16,
            connectivity: 16,
            expansion_add: 128,
            expansion_search: 64,
        }
    }
}

pub fn binary_quantization(embedding: &[f32]) -> Result<Vec<u8>> {
    if !embedding.len().is_multiple_of(8) {
        return Err(anyhow!(
            "Embedding length must be a multiple of 8 for binary quantization"
        ));
    }
    let num_bytes = embedding.len() / 8;
    let mut binary_emb = vec![0u8; num_bytes];
    for (i, v) in embedding.iter().enumerate() {
        if *v <= 0.0 {
            continue;
        }
        let byte_index = i / 8;
        let bit_index = i % 8;
        binary_emb[byte_index] |= 1 << bit_index;
    }
    Ok(binary_emb)
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingSearchParams {
    #[serde(
        default = "crate::index::default_k",
        deserialize_with = "deserialize_number_from_string"
    )]
    pub k: usize,
    #[serde(
        default,
        rename = "min-score",
        deserialize_with = "deserialize_option_number_from_string"
    )]
    pub min_score: Option<f32>,
    #[serde(default, deserialize_with = "deserialize_bool_from_anything")]
    pub exact: bool,
    #[serde(default, deserialize_with = "deserialize_option_number_from_string")]
    pub rerank: Option<f32>,
}

impl Default for EmbeddingSearchParams {
    fn default() -> Self {
        Self {
            k: 10,
            min_score: None,
            exact: false,
            rerank: None,
        }
    }
}

impl EmbeddingSearchParams {
    pub fn do_rerank(&self) -> bool {
        self.rerank.is_some()
    }
}

impl SearchParamsExt for EmbeddingSearchParams {
    fn search_k(&self, data: &impl DataSource) -> usize {
        let mut k = self.k();
        if let Some(factor) = self.rerank {
            k = (k as f32 * factor).ceil() as usize;
        }
        if self.exact() {
            k * data.max_fields().max(1) as usize
        } else {
            (k as f32 * data.avg_fields()).ceil() as usize
        }
    }

    fn k(&self) -> usize {
        self.k
    }

    fn exact(&self) -> bool {
        self.exact
    }
}

struct EmbeddingIndexInner {
    data: Embeddings,
    index: Index,
    params: EmbeddingIndexParams,
}

impl std::fmt::Debug for EmbeddingIndexInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inner")
            .field("data", &self.data)
            .field("index", &"Index { ... }")
            .field("params", &self.params)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingIndex {
    inner: Arc<EmbeddingIndexInner>,
}

impl EmbeddingIndex {
    fn search(
        embedding: EmbeddingRef<'_>,
        index: &Index,
        metric: Metric,
        precision: Precision,
        data: &impl DataSource,
        params: &EmbeddingSearchParams,
        filter: Option<impl Fn(u64) -> bool>,
    ) -> Result<Vec<(u64, f32)>> {
        // TODO: only needed because of usearch bug
        // with not being able to auto-convert to binary embeddings
        // remove when fixed in usearch togehter with if is_binary block
        let is_binary = precision == Precision::Binary;

        let num_dimensions = index.dimensions();
        let search_k = params.search_k(data);

        let results = if is_binary {
            let binary_emb = binary_quantization(embedding)?;
            let embedding = b1x8::from_u8s(&binary_emb);

            if let Some(ref pred) = filter {
                index.filtered_search(embedding, search_k, pred)?
            } else {
                index.search(embedding, search_k)?
            }
        } else if let Some(ref pred) = filter {
            index.filtered_search(embedding, search_k, pred)?
        } else {
            index.search(embedding, search_k)?
        };

        Ok(results
            .keys
            .iter()
            .zip(results.distances.iter())
            .filter_map(|(&id, &distance)| {
                // usearch returns distances (lower is better) for all metrics.
                // Convert to a score where higher is better.
                let score = metric.to_score(distance, num_dimensions);

                // Apply min_score filter
                if let Some(min_score) = params.min_score
                    && score < min_score
                {
                    return None;
                }

                Some((id, score))
            })
            .collect())
    }

    fn search_internal(
        &self,
        embedding: EmbeddingRef<'_>,
        params: &EmbeddingSearchParams,
        filter: Option<impl Fn(u32) -> bool>,
    ) -> Result<Vec<Match>> {
        let data = &self.inner.data;
        let index = &self.inner.index;

        let predicate = filter.map(|f| move |id| f(id as u32));

        let mut matches = Self::search(
            embedding,
            index,
            self.inner.params.metric,
            self.inner.params.precision,
            data,
            params,
            predicate,
        )?;

        // Deduplicate by ID (keeping the best score for each ID)
        matches.sort_by_key(|&(id, score)| (id, Reverse(OrderedFloat(score))));
        matches.dedup_by(|a, b| a.0 == b.0);

        // Sort by score descending, then by ID
        matches.sort_by_key(|&(id, score)| (Reverse(OrderedFloat(score)), id));

        // Take top k and create Match objects
        let matches: Vec<Match> = matches
            .into_iter()
            .map(|(id, score)| Match::Regular(id as u32, score))
            .take(params.k)
            .collect();

        Ok(matches)
    }
}

impl Search for EmbeddingIndex {
    type Data = Embeddings;
    type Query<'q> = EmbeddingRef<'q>;
    type BuildParams = EmbeddingIndexParams;
    type SearchParams = EmbeddingSearchParams;

    fn build(data: &Self::Data, index_dir: &Path, params: &Self::BuildParams) -> Result<()> {
        // Validate metric is compatible with precision
        params.metric.validate_precision(params.precision)?;

        create_dir_all(index_dir)?;

        // Create usearch index
        let num_dimensions = data.num_dimensions();

        let options = IndexOptions {
            dimensions: num_dimensions,
            metric: params.metric.to_usearch_metric(),
            quantization: params.precision.to_usearch_scalar_kind(),
            connectivity: params.connectivity,
            expansion_add: params.expansion_add,
            expansion_search: params.expansion_search,
            multi: true, // Enable multi-index to support duplicate IDs
        };

        let index = Index::new(&options)?;

        let total_fields = data.total_fields();
        index.reserve_capacity_and_threads(total_fields as usize, num_cpus::get_physical())?;

        let pb = progress_bar("Building embedding index", Some(total_fields as u64))?;

        // Add all embeddings to the index using their IDs as keys (not indices)
        // Multiple embeddings can have the same ID
        for (id, embeddings) in data.items() {
            for emb in embeddings {
                if params.precision == Precision::Binary {
                    let binary_emb = binary_quantization(emb)?;
                    index.add(id as u64, b1x8::from_u8s(&binary_emb))?;
                } else {
                    index.add(id as u64, emb)?;
                }

                pb.inc(1);
            }
        }

        pb.finish_with_message("Embedding index built");

        // Save the index
        let index_file = index_dir.join("index.usearch");
        index.save(index_file.to_str().ok_or_else(|| anyhow!("Invalid path"))?)?;

        // Save params as JSON
        write_json(&index_dir.join("index.params"), params)?;

        Ok(())
    }

    fn load(data: Self::Data, index_dir: &Path) -> Result<Self> {
        // Load params from JSON
        let params: EmbeddingIndexParams = load_json(&index_dir.join("index.params"))?;

        // Load the index
        let index_file = index_dir.join("index.usearch");

        let options = IndexOptions {
            dimensions: data.num_dimensions(),
            metric: params.metric.to_usearch_metric(),
            quantization: params.precision.to_usearch_scalar_kind(),
            connectivity: params.connectivity,
            expansion_add: params.expansion_add,
            expansion_search: params.expansion_search,
            multi: true, // Enable multi-index to support duplicate IDs
        };
        let index = Index::new(&options)?;

        index.view(index_file.to_str().ok_or_else(|| anyhow!("Invalid path"))?)?;

        Ok(Self {
            inner: Arc::new(EmbeddingIndexInner {
                data,
                index,
                params,
            }),
        })
    }

    fn data(&self) -> &Self::Data {
        &self.inner.data
    }

    fn index_type(&self) -> &'static str {
        "embedding"
    }

    fn search(&self, query: Self::Query<'_>, params: &Self::SearchParams) -> Result<Vec<Match>> {
        self.search_internal(query, params, None::<fn(u32) -> bool>)
    }

    fn search_with_filter<F>(
        &self,
        query: Self::Query<'_>,
        params: &Self::SearchParams,
        filter: F,
    ) -> Result<Vec<Match>>
    where
        F: Fn(u32) -> bool,
    {
        self.search_internal(query, params, Some(filter))
    }
}
struct EmbeddingIndexWithDataInner {
    data: EmbeddingsWithData,
    index: Index,
    field_to_data: Vec<u32>,
    params: EmbeddingIndexParams,
}

impl std::fmt::Debug for EmbeddingIndexWithDataInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inner")
            .field("data", &self.data)
            .field("index", &"Index { ... }")
            .field("field_to_data", &self.field_to_data)
            .field("params", &self.params)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingIndexWithData {
    inner: Arc<EmbeddingIndexWithDataInner>,
}

impl EmbeddingIndexWithData {
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

    fn search_internal(
        &self,
        embedding: EmbeddingRef<'_>,
        params: &EmbeddingSearchParams,
        filter: Option<impl Fn(u32) -> bool>,
    ) -> Result<Vec<Match>> {
        let data = &self.inner.data;
        let index = &self.inner.index;
        let field_to_data = &self.inner.field_to_data;

        let predicate = filter.map(|f| self.filter_internal(f));

        let matches = EmbeddingIndex::search(
            embedding,
            index,
            self.inner.params.metric,
            self.inner.params.precision,
            data,
            params,
            predicate,
        )?;

        // Sort by data_id, then by score descending, then by field_id
        let mut matches: Vec<_> = matches
            .into_iter()
            .map(|(field_id, score)| {
                let data_id = field_to_data[field_id as usize];
                (data_id, field_id, score)
            })
            .sorted_by_key(|&(data_id, field_id, score)| {
                (data_id, Reverse(OrderedFloat(score)), field_id)
            })
            .collect();

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

impl Search for EmbeddingIndexWithData {
    type Data = EmbeddingsWithData;
    type Query<'q> = EmbeddingRef<'q>;
    type BuildParams = EmbeddingIndexParams;
    type SearchParams = EmbeddingSearchParams;

    fn build(data: &Self::Data, index_dir: &Path, params: &Self::BuildParams) -> Result<()> {
        // Validate metric is compatible with precision
        params.metric.validate_precision(params.precision)?;

        create_dir_all(index_dir)?;

        // Create usearch index
        let num_dimensions = data.num_dimensions();

        let options = IndexOptions {
            dimensions: num_dimensions,
            metric: params.metric.to_usearch_metric(),
            quantization: params.precision.to_usearch_scalar_kind(),
            connectivity: params.connectivity,
            expansion_add: params.expansion_add,
            expansion_search: params.expansion_search,
            multi: false,
        };

        let index = Index::new(&options)?;

        let total_fields = data.total_fields();
        index.reserve_capacity_and_threads(total_fields as usize, num_cpus::get_physical())?;

        let pb = progress_bar("Building text embedding index", Some(total_fields as u64))?;

        let mut field_to_data_file =
            BufWriter::new(File::create(index_dir.join("index.field-to-data"))?);
        let mut field_id: u32 = 0;
        for (id, embeddings) in data.embedding_items() {
            for emb in embeddings {
                if field_id == u32::MAX {
                    return Err(anyhow!("too many fields, max {} supported", u32::MAX));
                }

                if params.precision == Precision::Binary {
                    let binary_emb = binary_quantization(emb)?;
                    index.add(field_id as u64, b1x8::from_u8s(&binary_emb))?;
                } else {
                    index.add(field_id as u64, emb)?;
                }

                field_id += 1;
                field_to_data_file.write_all(&id.to_le_bytes())?;

                pb.inc(1);
            }
        }

        pb.finish_with_message("Text embedding index built");

        // Save the index
        let index_file = index_dir.join("index.usearch");
        index.save(index_file.to_str().ok_or_else(|| anyhow!("Invalid path"))?)?;

        // Save params as JSON
        write_json(&index_dir.join("index.params"), params)?;

        Ok(())
    }

    fn load(data: Self::Data, index_dir: &Path) -> Result<Self> {
        // Load params from JSON
        let params: EmbeddingIndexParams = load_json(&index_dir.join("index.params"))?;

        // Load the index
        let index_file = index_dir.join("index.usearch");

        let index = Index::new(&IndexOptions {
            dimensions: data.num_dimensions(),
            metric: params.metric.to_usearch_metric(),
            quantization: params.precision.to_usearch_scalar_kind(),
            connectivity: params.connectivity,
            expansion_add: params.expansion_add,
            expansion_search: params.expansion_search,
            multi: false,
        })?;

        index.view(index_file.to_str().ok_or_else(|| anyhow!("Invalid path"))?)?;

        let field_to_data = load_u32_vec(&index_dir.join("index.field-to-data"))?;

        Ok(Self {
            inner: Arc::new(EmbeddingIndexWithDataInner {
                data,
                index,
                field_to_data,
                params,
            }),
        })
    }

    fn data(&self) -> &Self::Data {
        &self.inner.data
    }

    fn index_type(&self) -> &'static str {
        "text-embedding"
    }

    fn search(&self, query: Self::Query<'_>, params: &Self::SearchParams) -> Result<Vec<Match>> {
        self.search_internal(query, params, None::<fn(u32) -> bool>)
    }

    fn search_with_filter<F>(
        &self,
        query: Self::Query<'_>,
        params: &Self::SearchParams,
        filter: F,
    ) -> Result<Vec<Match>>
    where
        F: Fn(u32) -> bool,
    {
        self.search_internal(query, params, Some(filter))
    }
}

#[cfg(test)]
mod embedding_index_tests {
    use super::*;
    use crate::data::{Embeddings, Precision};
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
    fn test_embedding_index() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let data_dir = temp_dir.path().join("data");

        create_dir_all(&data_dir).expect("Failed to create data dir");

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

        // Build data
        Embeddings::build(&data_dir).expect("Failed to build data");
        let data = Embeddings::load(&data_dir).expect("Failed to load data");

        // Test metrics that work well with normalized embeddings
        let metrics = vec![Metric::CosineNormalized, Metric::Cosine, Metric::L2];

        // Test all precisions (except Binary which only works with Hamming)
        let precisions = vec![
            Precision::Float32,
            Precision::Float16,
            Precision::BFloat16,
            Precision::Int8,
        ];

        for metric in &metrics {
            for precision in &precisions {
                let index_dir = temp_dir
                    .path()
                    .join(format!("index_{:?}_{:?}", metric, precision));
                create_dir_all(&index_dir).expect("Failed to create index dir");

                let params = EmbeddingIndexParams::default()
                    .with_metric(*metric)
                    .with_precision(*precision);

                EmbeddingIndex::build(&data, &index_dir, &params).expect("Failed to build index");
                let index =
                    EmbeddingIndex::load(data.clone(), &index_dir).expect("Failed to load index");

                let mut query = vec![1.0, 0.0, 0.0, 0.0];
                normalize(&mut query);

                let results = index
                    .search(&query, &EmbeddingSearchParams::default())
                    .expect("Failed to search");

                // Should find ID 100 as top result (deduped from two embeddings)
                assert!(
                    !results.is_empty(),
                    "No results for {:?} {:?}",
                    metric,
                    precision
                );

                if let Match::Regular(id, score) = results[0] {
                    assert_eq!(id, 100, "Expected ID 100 for {:?} {:?}", metric, precision);

                    // Verify score is in expected range
                    match metric {
                        Metric::CosineNormalized | Metric::Cosine => {
                            // Score should be close to 1.0 for perfect match (allowing for quantization error)
                            assert!(
                                score > 0.9,
                                "Cosine score too low: {} for {:?} {:?}",
                                score,
                                metric,
                                precision
                            );
                        }
                        Metric::L2 => {
                            // L2 score = 1.0 / (1.0 + distance), perfect match = 1.0
                            assert!(
                                score > 0.5,
                                "L2 score too low: {} for {:?} {:?}",
                                score,
                                metric,
                                precision
                            );
                        }
                        _ => {}
                    }
                } else {
                    panic!("Expected Match::Regular for {:?} {:?}", metric, precision);
                }

                // Verify deduplication: ID 100 should appear only once despite having 2 embeddings
                let id_100_count = results
                    .iter()
                    .filter(|m| matches!(m, Match::Regular(100, _)))
                    .count();
                assert_eq!(
                    id_100_count, 1,
                    "Deduplication failed for {:?} {:?}",
                    metric, precision
                );

                // Test search with filter - exclude ID 100
                let results_filtered = index
                    .search_with_filter(&query, &EmbeddingSearchParams::default(), |id| id != 100)
                    .expect("Failed to search with filter");

                // Should find ID 200 or 300, but not 100
                assert!(
                    !results_filtered.is_empty(),
                    "No filtered results for {:?} {:?}",
                    metric,
                    precision
                );
                if let Match::Regular(id, _) = results_filtered[0] {
                    assert_ne!(id, 100, "Filter failed for {:?} {:?}", metric, precision);
                } else {
                    panic!("Expected Match::Regular for {:?} {:?}", metric, precision);
                }
            }
        }

        // Test InnerProduct metric with unnormalized embeddings
        let data_dir_ip = temp_dir.path().join("data_ip");
        create_dir_all(&data_dir_ip).expect("Failed to create data dir");
        let embeddings_file_ip = data_dir_ip.join("embedding.safetensors");

        // Create unnormalized embeddings
        let embeddings_ip = vec![
            vec![2.0, 0.0, 0.0, 0.0], // ID 100
            vec![1.8, 0.2, 0.0, 0.0], // ID 100 (duplicate - similar)
            vec![0.0, 2.0, 0.0, 0.0], // ID 200
            vec![0.0, 0.0, 2.0, 0.0], // ID 300
        ];

        create_test_safetensors(&embeddings_file_ip, embeddings_ip, vec![100, 100, 200, 300])
            .expect("Failed to create safetensors");

        Embeddings::build(&data_dir_ip).expect("Failed to build data");
        let data_ip = Embeddings::load(&data_dir_ip).expect("Failed to load data");

        for precision in &precisions {
            let index_dir = temp_dir
                .path()
                .join(format!("index_InnerProduct_{:?}", precision));
            create_dir_all(&index_dir).expect("Failed to create index dir");

            let params = EmbeddingIndexParams::default()
                .with_metric(Metric::InnerProduct)
                .with_precision(*precision);

            EmbeddingIndex::build(&data_ip, &index_dir, &params).expect("Failed to build index");
            let index =
                EmbeddingIndex::load(data_ip.clone(), &index_dir).expect("Failed to load index");

            let query_ip = vec![1.0, 0.0, 0.0, 0.0]; // Unnormalized query

            let results = index
                .search(&query_ip, &EmbeddingSearchParams::default())
                .expect("Failed to search");

            assert!(
                !results.is_empty(),
                "No results for InnerProduct {:?}",
                precision
            );

            if let Match::Regular(id, score) = results[0] {
                assert_eq!(id, 100, "Expected ID 100 for InnerProduct {:?}", precision);
                // Inner product of [1,0,0,0] with [2,0,0,0] = 2.0
                // With IP metric, usearch distance = 1.0 - IP = 1.0 - 2.0 = -1.0
                // Score = -distance = -(-1.0) = 1.0, but need to account for quantization
                assert!(
                    score.abs() > 0.1,
                    "IP score too small: {} for {:?}",
                    score,
                    precision
                );
            } else {
                panic!("Expected Match::Regular for InnerProduct {:?}", precision);
            }
        }

        // Test Hamming metric with Binary precision
        // Binary precision in usearch expects dimensions to be multiples of 8 bits (1 byte)
        // Since we're storing float32 vectors and converting to binary, use 32D (4 bytes packed)
        let data_dir_hamming = temp_dir.path().join("data_hamming");
        create_dir_all(&data_dir_hamming).expect("Failed to create data dir");
        let embeddings_file_hamming = data_dir_hamming.join("embedding.safetensors");

        // Create 32-dimensional binary-like embeddings (4 bytes when packed)
        let mut emb1 = vec![1.0; 16];
        emb1.extend(vec![0.0; 16]);
        let mut emb2 = vec![1.0; 14];
        emb2.extend(vec![0.0; 18]);
        let mut emb3 = vec![0.0; 16];
        emb3.extend(vec![1.0; 16]);
        let mut emb4 = vec![0.0; 14];
        emb4.extend(vec![1.0; 18]);

        let embeddings_hamming = vec![emb1, emb2, emb3, emb4];

        create_test_safetensors(
            &embeddings_file_hamming,
            embeddings_hamming.clone(),
            vec![100, 100, 200, 300],
        )
        .expect("Failed to create safetensors");

        Embeddings::build(&data_dir_hamming).expect("Failed to build data");
        let data_hamming = Embeddings::load(&data_dir_hamming).expect("Failed to load data");

        let index_dir_hamming = temp_dir.path().join("index_hamming");
        create_dir_all(&index_dir_hamming).expect("Failed to create index dir");

        let params = EmbeddingIndexParams::default()
            .with_metric(Metric::Hamming)
            .with_precision(Precision::Binary);

        EmbeddingIndex::build(&data_hamming, &index_dir_hamming, &params)
            .expect("Failed to build index");
        let index_hamming =
            EmbeddingIndex::load(data_hamming, &index_dir_hamming).expect("Failed to load index");

        let mut query_hamming = vec![1.0; 16];
        query_hamming.extend(vec![0.0; 16]);

        let results = index_hamming
            .search(&query_hamming, &EmbeddingSearchParams::default())
            .expect("Failed to search");

        assert!(!results.is_empty(), "No results for Hamming Binary");
        if let Match::Regular(id, score) = results[0] {
            assert_eq!(id, 100, "Expected ID 100 for Hamming Binary");
            // Hamming score should be in [0, 1] where 1 is identical
            assert!(
                score >= 0.0 && score <= 1.0,
                "Hamming score out of range: {}",
                score
            );
        } else {
            panic!("Expected Match::Regular for Hamming Binary");
        }
    }
}
