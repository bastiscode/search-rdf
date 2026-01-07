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
    index::{Match, Search},
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
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Query {
    Text(String),
    Url(String),
    Embedding(Embedding),
}

impl Query {
    pub fn query_type(&self) -> &str {
        match self {
            Query::Text(_) => "text",
            Query::Url(_) => "url",
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
    inputs: Vec<I>,
    search_fn: impl Fn(I) -> Result<Vec<SearchMatch>> + Send + Clone + 'static,
) -> Result<Vec<Vec<SearchMatch>>> {
    let handles: Vec<_> = inputs
        .into_iter()
        .map(|input| {
            let f = search_fn.clone();
            tokio::task::spawn_blocking(move || f(input))
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
        (Query::Url(url), EmbeddingModel::HuggingFaceImage(m)) => {
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

            search_parallel(queries, move |query| {
                let Query::Text(query) = query else {
                    return Err(anyhow!("Non-text query provided to keyword index"));
                };
                let matches =
                    index.search_with_filter(query.as_str(), &search_params, filter.clone())?;
                convert_to_search_matches(matches, index.data())
            })
            .await
        }
        SearchIndex::FullText(index) => {
            let SearchParams::FullText(search_params) = params else {
                return Err(anyhow!("Invalid search parameters for full-text index"));
            };

            search_parallel(queries, move |query| {
                let Query::Text(query) = query else {
                    return Err(anyhow!("Non-text query provided to full-text index"));
                };
                let matches =
                    index.search_with_filter(query.as_str(), &search_params, filter.clone())?;
                convert_to_search_matches(matches, index.data())
            })
            .await
        }
        SearchIndex::EmbeddingWithData(index) => {
            let SearchParams::Embedding(search_params) = params else {
                return Err(anyhow!("Invalid search parameters for embedding index"));
            };

            // For text embedding index with text query, we'd need to embed the text first
            let Some((model, model_params)) = model.cloned() else {
                return Err(anyhow!("No embedding model specified for embedding search"));
            };

            search_parallel(queries, move |query| {
                let emb = embed_query(query, &model, &model_params)?;
                let matches = index.search_with_filter(&emb, &search_params, filter.clone())?;
                convert_to_search_matches(matches, index.data().data())
            })
            .await
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

            search_parallel(queries, move |query| {
                let Query::Text(query) = query else {
                    return Err(anyhow!("Non-text query provided to keyword index"));
                };
                let matches = index.search(query.as_str(), &search_params)?;
                convert_to_search_matches(matches, index.data())
            })
            .await
        }
        SearchIndex::FullText(index) => {
            let SearchParams::FullText(search_params) = params else {
                return Err(anyhow!("Invalid search parameters for full-text index"));
            };

            search_parallel(queries, move |query| {
                let Query::Text(query) = query else {
                    return Err(anyhow!("Non-text query provided to full-text index"));
                };
                let matches = index.search(query.as_str(), &search_params)?;
                convert_to_search_matches(matches, index.data())
            })
            .await
        }
        SearchIndex::EmbeddingWithData(index) => {
            let SearchParams::Embedding(search_params) = params else {
                return Err(anyhow!("Invalid search parameters for embedding index"));
            };

            // For text embedding index with text query, we'd need to embed the text first
            let Some((model, model_params)) = model.cloned() else {
                return Err(anyhow!("No embedding model specified for embedding search"));
            };

            search_parallel(queries, move |query| {
                let emb = embed_query(query, &model, &model_params)?;
                let matches = index.search(&emb, &search_params)?;
                convert_to_search_matches(matches, index.data().data())
            })
            .await
        }
        _ => Err(anyhow!("Search is not supported for this index type")),
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
