use crate::data::embedding::EmbeddingRef;
use crate::data::text::embedding::TextEmbeddings;
use crate::data::{DataSource, Precision};
use crate::index::embedding::{Metadata, binary_quantization};
use crate::index::{EmbeddingIndexParams, Match, Search, SearchParamsExt};
use crate::utils::{load_json, write_json};
use crate::utils::{load_u32_vec, progress_bar};
use anyhow::{Result, anyhow};
use ordered_float::OrderedFloat;
use serde::Deserialize;
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

impl std::fmt::Debug for Inner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inner")
            .field("data", &self.data)
            .field("index", &"Index { ... }")
            .field("field_to_data", &self.field_to_data)
            .field("metadata", &self.metadata)
            .finish()
    }
}

#[derive(Debug, Clone)]
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
        embedding: EmbeddingRef<'_>,
        params: &TextEmbeddingSearchParams,
        filter: Option<F>,
    ) -> Result<Vec<Match>>
    where
        F: Fn(u32) -> bool,
    {
        let data = &self.inner.data;
        let index = &self.inner.index;
        // TODO: support binary embeddings, should not be necessary, remove once usearch fixes this
        let is_binary = self.inner.metadata.index.precision == Precision::Binary;

        let num_dimensions = data.num_dimensions();
        let search_k = params.search_k(data);

        let predicate = filter.map(|f| self.filter_internal(f));

        // Validate embedding matches index precision and dimensions
        if embedding.len() != num_dimensions {
            return Err(anyhow!(
                "Query embedding has {} dimensions, expected {}",
                embedding.len(),
                num_dimensions
            ));
        }

        let results = if is_binary {
            let binary_emb = binary_quantization(embedding)?;
            let embedding = b1x8::from_u8s(&binary_emb);
            if let Some(ref pred) = predicate {
                if params.exact {
                    return Err(anyhow!("Exact search with filter is not supported yet"));
                }
                index.filtered_search(embedding, search_k, pred)?
            } else if params.exact {
                index.exact_search(embedding, search_k)?
            } else {
                index.search(embedding, search_k)?
            }
        } else if let Some(ref pred) = predicate {
            if params.exact {
                return Err(anyhow!("Exact search with filter is not supported yet"));
            }
            index.filtered_search(embedding, search_k, pred)?
        } else if params.exact {
            index.exact_search(embedding, search_k)?
        } else {
            index.search(embedding, search_k)?
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
                    .index
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

#[derive(Debug, Clone, Deserialize)]
pub struct TextEmbeddingSearchParams {
    #[serde(default = "crate::index::default_k")]
    pub k: usize,
    #[serde(default)]
    pub min_score: Option<f32>,
    #[serde(default)]
    pub exact: bool,
}

impl SearchParamsExt for TextEmbeddingSearchParams {
    fn k(&self) -> usize {
        self.k
    }

    fn exact(&self) -> bool {
        self.exact
    }
}

impl Search for TextEmbeddingIndex {
    type Data = TextEmbeddings;
    type Query<'q> = EmbeddingRef<'q>;
    type BuildParams = EmbeddingIndexParams;
    type SearchParams = TextEmbeddingSearchParams;

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

        // Save metadata as JSON
        let metadata = Metadata {
            index: *params,
            num_dimensions,
        };
        write_json(&index_dir.join("index.metadata"), &metadata)?;

        Ok(())
    }

    fn load(data: Self::Data, index_dir: &Path) -> Result<Self> {
        // Load metadata from JSON
        let metadata: Metadata = load_json(&index_dir.join("index.metadata"))?;

        // Load the index
        let index_file = index_dir.join("index.usearch");

        let index = Index::new(&IndexOptions {
            dimensions: metadata.num_dimensions,
            metric: metadata.index.metric.to_usearch_metric(),
            quantization: metadata.index.precision.to_usearch_scalar_kind(),
            connectivity: metadata.index.connectivity,
            expansion_add: metadata.index.expansion_add,
            expansion_search: metadata.index.expansion_search,
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
mod tests {
    use super::*;
    use crate::data::Precision;
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
    fn test_text_embedding_index() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let data_dir = temp_dir.path().join("data");

        // Create test data with varying field counts: 1, 3, 2 (max=3)
        let items = vec![
            Ok(TextItem::new("Q1".to_string(), vec!["Cat".to_string()])
                .expect("Failed to create TextItem")),
            Ok(TextItem::new(
                "Q2".to_string(),
                vec!["Dog".to_string(), "Canine".to_string(), "Hound".to_string()],
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

        // Create normalized test embeddings (6 embeddings total: 1 + 3 + 2)
        let mut embeddings = vec![
            vec![1.0, 0.0, 0.0, 0.0], // Q1 field (Cat)
            vec![0.0, 1.0, 0.0, 0.0], // Q2 field1 (Dog)
            vec![0.0, 0.9, 0.1, 0.0], // Q2 field2 (Canine)
            vec![0.0, 0.8, 0.2, 0.0], // Q2 field3 (Hound)
            vec![0.0, 0.0, 1.0, 0.0], // Q3 field1 (Bird)
            vec![0.0, 0.0, 0.9, 0.1], // Q3 field2 (Avian)
        ];

        for emb in &mut embeddings {
            normalize(emb);
        }

        let embeddings_file = data_dir.join("embedding.safetensors");
        create_test_safetensors(&embeddings_file, embeddings.clone())
            .expect("Failed to create safetensors");

        let data = TextEmbeddings::load(data, &embeddings_file).expect("Failed to load data");

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

                TextEmbeddingIndex::build(&data, &index_dir, &params)
                    .expect("Failed to build index");
                let index = TextEmbeddingIndex::load(data.clone(), &index_dir)
                    .expect("Failed to load index");

                let mut query = vec![1.0, 0.0, 0.0, 0.0];
                normalize(&mut query);

                let results = index
                    .search(&query, &SearchParams::default())
                    .expect("Failed to search");

                // Should find Q1 (Cat) as top result
                assert!(
                    !results.is_empty(),
                    "No results for {:?} {:?}",
                    metric,
                    precision
                );

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
                    "Expected Q1 (id=0) in top 3 results for {:?} {:?}, got {:?}",
                    metric,
                    precision,
                    results.iter().take(3).collect::<Vec<_>>()
                );

                // Verify score is in expected range
                if let Match::WithField(id, _, score) = results[0] {
                    assert_eq!(id, 0, "Expected ID 0 for {:?} {:?}", metric, precision);

                    match metric {
                        Metric::CosineNormalized | Metric::Cosine => {
                            assert!(
                                score > 0.9,
                                "Cosine score too low: {} for {:?} {:?}",
                                score,
                                metric,
                                precision
                            );
                        }
                        Metric::L2 => {
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
                }

                // Test search with filter - exclude Q1
                let results_filtered = index
                    .search_with_filter(&query, &SearchParams::default(), |id| id != 0)
                    .expect("Failed to search with filter");

                assert!(
                    !results_filtered.is_empty(),
                    "No filtered results for {:?} {:?}",
                    metric,
                    precision
                );
                if let Match::WithField(id, _, _) = results_filtered[0] {
                    assert_ne!(id, 0, "Filter failed for {:?} {:?}", metric, precision);
                } else {
                    panic!("Expected Match::WithField for {:?} {:?}", metric, precision);
                }

                // Verify max_fields is 3 (from Q2 with 3 fields) - only check once
                if metric == &Metric::CosineNormalized && precision == &Precision::Float32 {
                    assert_eq!(index.data().text_data().max_fields_per_id(), 3);
                }
            }
        }

        // Test InnerProduct metric with unnormalized embeddings
        let data_dir_ip = temp_dir.path().join("data_ip");

        let items_ip = vec![
            Ok(TextItem::new("Q1".to_string(), vec!["A".to_string()])
                .expect("Failed to create TextItem")),
            Ok(TextItem::new("Q2".to_string(), vec!["B".to_string()])
                .expect("Failed to create TextItem")),
        ];

        TextData::build(items_ip, &data_dir_ip).expect("Failed to build TextData");
        let data_text = TextData::load(&data_dir_ip).expect("Failed to load TextData");

        // Create unnormalized embeddings
        let embeddings_ip = vec![
            vec![2.0, 0.0, 0.0, 0.0], // Q1
            vec![0.0, 2.0, 0.0, 0.0], // Q2
        ];

        let embeddings_file_ip = data_dir_ip.join("embedding.safetensors");
        create_test_safetensors(&embeddings_file_ip, embeddings_ip)
            .expect("Failed to create safetensors");

        let data_ip =
            TextEmbeddings::load(data_text, &embeddings_file_ip).expect("Failed to load data");

        for precision in &precisions {
            let index_dir = temp_dir
                .path()
                .join(format!("index_InnerProduct_{:?}", precision));
            create_dir_all(&index_dir).expect("Failed to create index dir");

            let params = EmbeddingIndexParams::default()
                .with_metric(Metric::InnerProduct)
                .with_precision(*precision);

            TextEmbeddingIndex::build(&data_ip, &index_dir, &params)
                .expect("Failed to build index");
            let index = TextEmbeddingIndex::load(data_ip.clone(), &index_dir)
                .expect("Failed to load index");

            let query_ip = vec![1.0, 0.0, 0.0, 0.0]; // Unnormalized query

            let results = index
                .search(&query_ip, &SearchParams::default())
                .expect("Failed to search");

            assert!(
                !results.is_empty(),
                "No results for InnerProduct {:?}",
                precision
            );

            if let Match::WithField(id, _, score) = results[0] {
                assert_eq!(id, 0, "Expected ID 0 for InnerProduct {:?}", precision);
                assert!(
                    score.abs() > 0.1,
                    "IP score too small: {} for {:?}",
                    score,
                    precision
                );
            } else {
                panic!("Expected Match::WithField for InnerProduct {:?}", precision);
            }
        }

        // Test Hamming metric with Binary precision
        let data_dir_hamming = temp_dir.path().join("data_hamming");

        let items_hamming = vec![
            Ok(TextItem::new("Q1".to_string(), vec!["A".to_string()])
                .expect("Failed to create TextItem")),
            Ok(TextItem::new("Q2".to_string(), vec!["B".to_string()])
                .expect("Failed to create TextItem")),
        ];

        TextData::build(items_hamming, &data_dir_hamming).expect("Failed to build TextData");
        let data_text_hamming = TextData::load(&data_dir_hamming).expect("Failed to load TextData");

        // Create 32-dimensional binary-like embeddings
        let mut emb1 = vec![1.0; 16];
        emb1.extend(vec![0.0; 16]);
        let mut emb2 = vec![0.0; 16];
        emb2.extend(vec![1.0; 16]);

        let embeddings_hamming = vec![emb1, emb2];

        let embeddings_file_hamming = data_dir_hamming.join("embedding.safetensors");
        create_test_safetensors(&embeddings_file_hamming, embeddings_hamming.clone())
            .expect("Failed to create safetensors");

        let data_hamming = TextEmbeddings::load(data_text_hamming, &embeddings_file_hamming)
            .expect("Failed to load data");

        let index_dir_hamming = temp_dir.path().join("index_hamming");
        create_dir_all(&index_dir_hamming).expect("Failed to create index dir");

        let params = EmbeddingIndexParams::default()
            .with_metric(Metric::Hamming)
            .with_precision(Precision::Binary);

        TextEmbeddingIndex::build(&data_hamming, &index_dir_hamming, &params)
            .expect("Failed to build index");
        let index_hamming = TextEmbeddingIndex::load(data_hamming, &index_dir_hamming)
            .expect("Failed to load index");

        let mut query_hamming = vec![1.0; 16];
        query_hamming.extend(vec![0.0; 16]);

        let results = index_hamming
            .search(&query_hamming, &SearchParams::default())
            .expect("Failed to search");

        assert!(!results.is_empty(), "No results for Hamming Binary");
        if let Match::WithField(id, _, score) = results[0] {
            assert_eq!(id, 0, "Expected ID 0 for Hamming Binary");
            assert!(
                score >= 0.0 && score <= 1.0,
                "Hamming score out of range: {}",
                score
            );
        } else {
            panic!("Expected Match::WithField for Hamming Binary");
        }
    }

    #[test]
    fn test_field_matching() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let data_dir = temp_dir.path().join("data");

        // Create test data where each item has 3 fields with distinct embeddings
        let items = vec![
            Ok(TextItem::new(
                "Entity1".to_string(),
                vec![
                    "FieldA".to_string(),
                    "FieldB".to_string(),
                    "FieldC".to_string(),
                ],
            )
            .expect("Failed to create TextItem")),
            Ok(TextItem::new(
                "Entity2".to_string(),
                vec![
                    "FieldX".to_string(),
                    "FieldY".to_string(),
                    "FieldZ".to_string(),
                ],
            )
            .expect("Failed to create TextItem")),
        ];

        TextData::build(items, &data_dir).expect("Failed to build TextData");
        let data = TextData::load(&data_dir).expect("Failed to load TextData");

        // Create embeddings where each field has a unique, easily identifiable embedding
        // Entity1: field 0=[1,0,0,0], field 1=[0,1,0,0], field 2=[0,0,1,0]
        // Entity2: field 0=[0,0,0,1], field 1=[1,1,0,0], field 2=[0,1,1,0]
        let mut embeddings = vec![
            vec![1.0, 0.0, 0.0, 0.0], // Entity1, field 0
            vec![0.0, 1.0, 0.0, 0.0], // Entity1, field 1
            vec![0.0, 0.0, 1.0, 0.0], // Entity1, field 2
            vec![0.0, 0.0, 0.0, 1.0], // Entity2, field 0
            vec![1.0, 1.0, 0.0, 0.0], // Entity2, field 1
            vec![0.0, 1.0, 1.0, 0.0], // Entity2, field 2
        ];

        for emb in &mut embeddings {
            normalize(emb);
        }

        let embeddings_file = data_dir.join("embedding.safetensors");
        create_test_safetensors(&embeddings_file, embeddings.clone())
            .expect("Failed to create safetensors");

        let text_embeddings =
            TextEmbeddings::load(data, &embeddings_file).expect("Failed to load data");

        let index_dir = temp_dir.path().join("index");
        create_dir_all(&index_dir).expect("Failed to create index dir");

        let params = EmbeddingIndexParams::from_precision(Precision::Float32);
        TextEmbeddingIndex::build(&text_embeddings, &index_dir, &params)
            .expect("Failed to build index");
        let index = TextEmbeddingIndex::load(text_embeddings.clone(), &index_dir)
            .expect("Failed to load index");

        // Test 1: Query matching Entity1, field 1 (the middle field)
        let mut query1 = vec![0.0, 1.0, 0.0, 0.0];
        normalize(&mut query1);

        let results1 = index
            .search(&query1, &SearchParams::default())
            .expect("Failed to search");

        assert!(!results1.is_empty(), "No results for query1");
        if let Match::WithField(id, field_idx, score) = results1[0] {
            assert_eq!(id, 0, "Expected Entity1 (id=0)");
            assert_eq!(field_idx, 1, "Expected field 1 for Entity1");
            assert!(score > 0.9, "Score too low: {}", score);

            // Retrieve actual field text
            let field_text = text_embeddings
                .field(id, field_idx)
                .expect("Failed to get field text");
            assert_eq!(
                field_text, "FieldB",
                "Expected 'FieldB' as the matched field"
            );
        } else {
            panic!("Expected Match::WithField");
        }

        // Test 2: Query matching Entity1, field 2 (the last field)
        let mut query2 = vec![0.0, 0.0, 1.0, 0.0];
        normalize(&mut query2);

        let results2 = index
            .search(&query2, &SearchParams::default())
            .expect("Failed to search");

        assert!(!results2.is_empty(), "No results for query2");
        if let Match::WithField(id, field_idx, score) = results2[0] {
            assert_eq!(id, 0, "Expected Entity1 (id=0)");
            assert_eq!(field_idx, 2, "Expected field 2 for Entity1");
            assert!(score > 0.9, "Score too low: {}", score);

            let field_text = text_embeddings
                .field(id, field_idx)
                .expect("Failed to get field text");
            assert_eq!(
                field_text, "FieldC",
                "Expected 'FieldC' as the matched field"
            );
        } else {
            panic!("Expected Match::WithField");
        }

        // Test 3: Query matching Entity2, field 0 (the first field)
        let mut query3 = vec![0.0, 0.0, 0.0, 1.0];
        normalize(&mut query3);

        let results3 = index
            .search(&query3, &SearchParams::default())
            .expect("Failed to search");

        assert!(!results3.is_empty(), "No results for query3");
        if let Match::WithField(id, field_idx, score) = results3[0] {
            assert_eq!(id, 1, "Expected Entity2 (id=1)");
            assert_eq!(field_idx, 0, "Expected field 0 for Entity2");
            assert!(score > 0.9, "Score too low: {}", score);

            let field_text = text_embeddings
                .field(id, field_idx)
                .expect("Failed to get field text");
            assert_eq!(
                field_text, "FieldX",
                "Expected 'FieldX' as the matched field"
            );
        } else {
            panic!("Expected Match::WithField");
        }

        // Test 4: Query matching Entity2, field 2 (mix of dimensions)
        let mut query4 = vec![0.0, 1.0, 1.0, 0.0];
        normalize(&mut query4);

        let results4 = index
            .search(&query4, &SearchParams::default())
            .expect("Failed to search");

        assert!(!results4.is_empty(), "No results for query4");
        if let Match::WithField(id, field_idx, score) = results4[0] {
            assert_eq!(id, 1, "Expected Entity2 (id=1)");
            assert_eq!(field_idx, 2, "Expected field 2 for Entity2");
            assert!(score > 0.9, "Score too low: {}", score);

            let field_text = text_embeddings
                .field(id, field_idx)
                .expect("Failed to get field text");
            assert_eq!(
                field_text, "FieldZ",
                "Expected 'FieldZ' as the matched field"
            );
        } else {
            panic!("Expected Match::WithField");
        }
    }
}
