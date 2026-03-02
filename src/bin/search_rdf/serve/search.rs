use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use axum::extract::{Json, Path as AxumPath, State};
use axum::http::StatusCode;
use futures::future::try_join_all;
use search_rdf::data::embedding::Embedding;
use search_rdf::data::item::load_image_ndarray_from_url;
use search_rdf::model::EmbeddingParams;
use serde::{Deserialize, Serialize};

use search_rdf::{
    data::{Data, DataSource},
    index::{Match, Scored, Search, merge_neighbor_matches},
    model::{Embed, EmbeddingModel},
};
use serde_json::{Value, json};

use crate::search_rdf::index::{SearchIndex, SearchParams};

use super::types::{AppError, AppState};

// Search request/response types
#[derive(Deserialize)]
pub struct SearchRequest {
    queries: Vec<Query>,
    #[serde(flatten)]
    params: HashMap<String, Value>,
}

#[derive(Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "lowercase")]
pub enum Query {
    Text(String),
    Identifier(String),
    Embedding(Embedding),
}

impl Query {
    pub fn query_type(&self) -> &str {
        match self {
            Query::Text(_) => "text",
            Query::Identifier(_) => "identifier",
            Query::Embedding(_) => "embedding",
        }
    }
}

#[derive(Serialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum MatchInfo {
    None,
    Field { identifier: String, field: String },
}

#[derive(Serialize)]
pub struct SearchMatch {
    pub id: u32,
    pub score: f32,
    #[serde(skip_serializing_if = "matches_none")]
    pub info: MatchInfo,
}

fn matches_none(info: &MatchInfo) -> bool {
    *info == MatchInfo::None
}

impl SearchMatch {
    fn new(id: u32, score: f32, info: MatchInfo) -> Self {
        Self { id, score, info }
    }
}

impl From<Match> for SearchMatch {
    fn from(m: Match) -> Self {
        SearchMatch::new(m.id(), m.score(), MatchInfo::None)
    }
}

impl Scored for SearchMatch {
    fn id(&self) -> u32 {
        self.id
    }
    fn score(&self) -> f32 {
        self.score
    }
}

#[derive(Serialize)]
pub struct SearchResponse {
    matches: Vec<Vec<SearchMatch>>,
}

fn convert_to_search_matches(matches: Vec<Match>, data: &Data) -> Result<Vec<SearchMatch>> {
    let mut result = Vec::with_capacity(matches.len());

    for m in matches {
        let info = if let Match::WithField(id, field, ..) = m {
            let identifier = data
                .identifier(id)
                .ok_or_else(|| anyhow!("Failed to get identifier for id {}", id))?
                .to_string();

            let field = data
                .field(id, field)
                .map(|f| f.as_str().to_string())
                .ok_or_else(|| anyhow!("Failed to get field {} for id {}", field, id))?;

            MatchInfo::Field { identifier, field }
        } else {
            MatchInfo::None
        };
        result.push(SearchMatch::new(m.id(), m.score(), info));
    }
    Ok(result)
}

async fn search_parallel<I: Send + Sync + 'static>(
    inputs: impl IntoIterator<Item = (usize, I)>,
    search_fn: impl Fn(I) -> Result<Vec<SearchMatch>> + Send + Clone + 'static,
) -> Result<Vec<(usize, Vec<SearchMatch>)>> {
    let handles: Vec<_> = inputs
        .into_iter()
        .map(|(idx, input)| {
            let f = search_fn.clone();
            tokio::task::spawn_blocking(move || f(input).map(|r| (idx, r)))
        })
        .collect();

    let results = try_join_all(handles).await?;
    results.into_iter().collect()
}

pub fn embed_query(
    query: Query,
    model: &EmbeddingModel,
    params: &EmbeddingParams,
) -> Result<Embedding> {
    match (query, model) {
        (Query::Embedding(emb), _) => Ok(emb),
        (Query::Text(text), EmbeddingModel::SentenceTransformer(m)) => m
            .embed(&[text], params)
            .into_iter()
            .flatten()
            .next()
            .ok_or_else(|| anyhow!("Failed to embed query using sentence transformer model")),
        (Query::Text(text), EmbeddingModel::Vllm(m)) => m
            .embed(&[text], params)
            .into_iter()
            .flatten()
            .next()
            .ok_or_else(|| anyhow!("Failed to embed query using VLLM model")),
        (Query::Text(url), EmbeddingModel::HuggingFaceImage(m)) => {
            let image =
                load_image_ndarray_from_url(&url).context("Failed to load image from URL")?;
            m.embed(&[&image], params)
                .into_iter()
                .flatten()
                .next()
                .ok_or_else(|| anyhow!("Failed to embed image query"))
        }
        (query, model) => Err(anyhow!(
            "Cannot embed query of type {} with model {} of type {}",
            query.query_type(),
            model.model_name(),
            model.model_type()
        )),
    }
}

/// Perform text search on an index with given query, parameters, and filter
/// Returns a vector of search results for each query in the input
pub async fn perform_search_with_filter(
    index: SearchIndex,
    queries: Vec<Query>,
    params: SearchParams,
    model: Option<&(EmbeddingModel, EmbeddingParams)>,
    filter: impl Fn(u32) -> bool + Clone + Send + 'static,
) -> Result<Vec<Vec<SearchMatch>>> {
    match index {
        SearchIndex::Keyword(index) => {
            let SearchParams::Keyword(search_params) = params else {
                return Err(anyhow!("Invalid search parameters for keyword index"));
            };

            search_parallel(queries.into_iter().enumerate(), move |query| {
                let Query::Text(query) = query else {
                    return Err(anyhow!("Non-text query provided to keyword index"));
                };
                let matches =
                    index.search_with_filter(query.as_str(), &search_params, filter.clone())?;
                convert_to_search_matches(matches, index.data())
            })
            .await
            .map(|results| results.into_iter().map(|(_, r)| r).collect())
        }
        SearchIndex::Fuzzy(index) => {
            let SearchParams::Fuzzy(search_params) = params else {
                return Err(anyhow!("Invalid search parameters for fuzzy index"));
            };

            search_parallel(queries.into_iter().enumerate(), move |query| {
                let Query::Text(query) = query else {
                    return Err(anyhow!("Non-text query provided to fuzzy index"));
                };
                let matches =
                    index.search_with_filter(query.as_str(), &search_params, filter.clone())?;
                convert_to_search_matches(matches, index.data())
            })
            .await
            .map(|results| results.into_iter().map(|(_, r)| r).collect())
        }
        SearchIndex::FullText(index) => {
            let SearchParams::FullText(search_params) = params else {
                return Err(anyhow!("Invalid search parameters for full-text index"));
            };

            search_parallel(queries.into_iter().enumerate(), move |query| {
                let Query::Text(query) = query else {
                    return Err(anyhow!("Non-text query provided to full-text index"));
                };
                let matches =
                    index.search_with_filter(query.as_str(), &search_params, filter.clone())?;
                convert_to_search_matches(matches, index.data())
            })
            .await
            .map(|results| results.into_iter().map(|(_, r)| r).collect())
        }
        SearchIndex::EmbeddingWithData(index) => {
            let SearchParams::Embedding(search_params) = params else {
                return Err(anyhow!("Invalid search parameters for embedding index"));
            };

            // For text embedding index with text query, we'd need to embed the text first
            let Some((model, model_params)) = model.cloned() else {
                return Err(anyhow!("No embedding model specified for embedding search"));
            };

            search_parallel(queries.into_iter().enumerate(), move |query| {
                let emb = embed_query(query, &model, &model_params)?;
                let matches = index.search_with_filter(&emb, &search_params, filter.clone())?;
                convert_to_search_matches(matches, index.data().data())
            })
            .await
            .map(|results| results.into_iter().map(|(_, r)| r).collect())
        }
        _ => Err(anyhow!(
            "Filtered search is not supported for this index type"
        )),
    }
}

/// Perform search on an index with given query and parameters
/// Returns a vector of search results for each query in the input
pub async fn perform_search(
    index: SearchIndex,
    queries: Vec<Query>,
    params: SearchParams,
    model: Option<&(EmbeddingModel, EmbeddingParams)>,
) -> Result<Vec<Vec<SearchMatch>>> {
    // same as above but without filter
    match index {
        SearchIndex::Keyword(index) => {
            let SearchParams::Keyword(search_params) = params else {
                return Err(anyhow!("Invalid search parameters for keyword index"));
            };

            search_parallel(queries.into_iter().enumerate(), move |query| {
                let Query::Text(query) = query else {
                    return Err(anyhow!("Non-text query provided to keyword index"));
                };
                let matches = index.search(query.as_str(), &search_params)?;
                convert_to_search_matches(matches, index.data())
            })
            .await
            .map(|results| results.into_iter().map(|(_, r)| r).collect())
        }
        SearchIndex::Fuzzy(index) => {
            let SearchParams::Fuzzy(search_params) = params else {
                return Err(anyhow!("Invalid search parameters for fuzzy index"));
            };

            search_parallel(queries.into_iter().enumerate(), move |query| {
                let Query::Text(query) = query else {
                    return Err(anyhow!("Non-text query provided to fuzzy index"));
                };
                let matches = index.search(query.as_str(), &search_params)?;
                convert_to_search_matches(matches, index.data())
            })
            .await
            .map(|results| results.into_iter().map(|(_, r)| r).collect())
        }
        SearchIndex::FullText(index) => {
            let SearchParams::FullText(search_params) = params else {
                return Err(anyhow!("Invalid search parameters for full-text index"));
            };

            search_parallel(queries.into_iter().enumerate(), move |query| {
                let Query::Text(query) = query else {
                    return Err(anyhow!("Non-text query provided to full-text index"));
                };
                let matches = index.search(query.as_str(), &search_params)?;
                convert_to_search_matches(matches, index.data())
            })
            .await
            .map(|results| results.into_iter().map(|(_, r)| r).collect())
        }
        SearchIndex::EmbeddingWithData(index) => {
            let SearchParams::Embedding(search_params) = params else {
                return Err(anyhow!("Invalid search parameters for embedding index"));
            };

            // For text embedding index with text query, we'd need to embed the text first
            let Some((model, model_params)) = model.cloned() else {
                return Err(anyhow!("No embedding model specified for embedding search"));
            };

            search_parallel(queries.into_iter().enumerate(), move |query| {
                let emb = embed_query(query, &model, &model_params)?;
                let matches = index.search(&emb, &search_params)?;
                convert_to_search_matches(matches, index.data().data())
            })
            .await
            .map(|results| results.into_iter().map(|(_, r)| r).collect())
        }
        _ => Err(anyhow!("Search is not supported for this index type")),
    }
}

/// Extract embedding queries for neighbor search from a known item.
/// Only supported for embedding indices; all other index types return an error.
fn get_neighbor_queries(index: &SearchIndex, data_id: u32) -> Result<Vec<Query>> {
    match index {
        SearchIndex::EmbeddingWithData(idx) => idx
            .field_embeddings(data_id)
            .ok_or_else(|| anyhow!("Item {} not found in embedding index", data_id))
            .map(|embs| embs.map(|e| Query::Embedding(e.to_vec())).collect()),
        SearchIndex::Embedding(idx) => idx
            .data()
            .fields(data_id)
            .ok_or_else(|| anyhow!("Item {} not found in embedding index", data_id))
            .map(|embs| embs.map(|e| Query::Embedding(e.to_vec())).collect()),
        _ => Err(anyhow!(
            "Neighbor search is only supported for embedding indices"
        )),
    }
}

/// Find the top-k neighbors of a known item, excluding the item itself.
///
/// Uses the k+1 trick: searches for k+1 results via approximate search, then post-filters
/// self from each field result set before merging. This avoids the expensive exact/linear
/// scan that `search_with_filter` would otherwise impose on HNSW indices.
pub async fn perform_neighbor_search(
    index: SearchIndex,
    data_id: u32,
    params: SearchParams,
    model: Option<&(EmbeddingModel, EmbeddingParams)>,
) -> Result<Vec<SearchMatch>> {
    let k = params.k();
    let queries = get_neighbor_queries(&index, data_id)?;
    let all_results = perform_search(index, queries, params.bump_k(), model).await?;
    let filtered: Vec<Vec<SearchMatch>> = all_results
        .into_iter()
        .map(|field_results| {
            field_results
                .into_iter()
                .filter(|m| m.id != data_id)
                .collect()
        })
        .collect();
    Ok(merge_neighbor_matches(filtered, k))
}

/// Find the top-k neighbors of a known item with an additional filter, excluding the item itself.
///
/// Combines self-exclusion with the provided filter and delegates to `search_with_filter`.
pub async fn perform_neighbor_search_with_filter(
    index: SearchIndex,
    data_id: u32,
    params: SearchParams,
    model: Option<&(EmbeddingModel, EmbeddingParams)>,
    filter: impl Fn(u32) -> bool + Clone + Send + 'static,
) -> Result<Vec<SearchMatch>> {
    let k = params.k();
    let queries = get_neighbor_queries(&index, data_id)?;
    let combined_filter = move |id: u32| id != data_id && filter(id);
    let all_results =
        perform_search_with_filter(index, queries, params, model, combined_filter).await?;
    Ok(merge_neighbor_matches(all_results, k))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_search_parallel_preserves_index() {
        // Each input yields one match; the returned usize must match the input index.
        let results = search_parallel((0u32..3).map(|i| (i as usize, i)), |v| {
            Ok(vec![SearchMatch::new(v, v as f32, MatchInfo::None)])
        })
        .await
        .unwrap();

        assert_eq!(results.len(), 3);
        for (idx, matches) in &results {
            assert_eq!(matches.len(), 1);
            assert_eq!(matches[0].id, *idx as u32);
            assert_eq!(matches[0].score, *idx as f32);
        }
    }

    #[tokio::test]
    async fn test_search_parallel_groups_by_index() {
        // Simulate two query items (index 0 and 1), each with 2 field embeddings.
        // The caller is responsible for grouping and merging; here we just verify
        // that the usize tag is threaded through correctly.
        let inputs = vec![(0usize, 10u32), (0, 11), (1, 20), (1, 21)];
        let results = search_parallel(inputs, |v| {
            Ok(vec![SearchMatch::new(v, v as f32, MatchInfo::None)])
        })
        .await
        .unwrap();

        assert_eq!(results.len(), 4);

        let mut by_index: std::collections::HashMap<usize, Vec<u32>> =
            std::collections::HashMap::new();
        for (idx, matches) in results {
            for m in matches {
                by_index.entry(idx).or_default().push(m.id);
            }
        }

        assert_eq!(by_index.len(), 2, "should have exactly 2 groups");

        let mut ids_0 = by_index[&0].clone();
        ids_0.sort_unstable();
        assert_eq!(ids_0, vec![10, 11]);

        let mut ids_1 = by_index[&1].clone();
        ids_1.sort_unstable();
        assert_eq!(ids_1, vec![20, 21]);
    }
}

pub async fn search(
    AxumPath(index_name): AxumPath<String>,
    State(state): State<AppState>,
    Json(mut req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, AppError> {
    let index = state
        .inner
        .indices
        .get(&index_name)
        .cloned()
        .ok_or_else(|| {
            AppError(
                StatusCode::BAD_REQUEST,
                anyhow!("Index not found: {}", index_name),
            )
        })?;

    let model = state
        .inner
        .index_to_model
        .get(&index_name)
        .and_then(|model_name| state.inner.models.get(model_name));

    // parse params
    req.params.insert(
        "type".to_string(),
        Value::String(index.index_type().to_string()),
    );

    let params = SearchParams::deserialize(json!(req.params)).map_err(|e| {
        AppError(
            StatusCode::BAD_REQUEST,
            anyhow!("Failed to parse search parameters: {}", e),
        )
    })?;

    let matches = perform_search(index, req.queries, params, model).await?;

    Ok(Json(SearchResponse { matches }))
}
