use std::collections::HashMap;

use anyhow::{Result, anyhow};
use axum::extract::{Json, Path as AxumPath, State};
use axum::http::StatusCode;
use futures::future::try_join_all;
use search_rdf::model::EmbeddingParams;
use serde::{Deserialize, Serialize};

use search_rdf::{
    data::{DataSource, TextData},
    index::{Match, Search},
    model::{Embed, EmbeddingModel},
};
use serde_json::{Value, json};

use crate::search_rdf::index::{SearchIndex, SearchParams};

use super::types::{AppError, AppState};

// Search request/response types
#[derive(Deserialize)]
pub struct SearchRequest {
    query: QueryType,
    #[serde(flatten)]
    params: HashMap<String, Value>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum QueryType {
    Text(Vec<String>),
    Embedding(Vec<Vec<f32>>),
}

impl QueryType {
    pub fn type_name(&self) -> &'static str {
        match self {
            QueryType::Text(..) => "text",
            QueryType::Embedding(..) => "embedding",
        }
    }
}

#[derive(Serialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MatchInfo {
    None,
    Text { identifier: String, field: String },
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

fn convert_matches(matches: Vec<Match>) -> Vec<SearchMatch> {
    matches.into_iter().map(|m| m.into()).collect()
}

fn convert_to_text_search_matches(
    matches: Vec<Match>,
    data: &TextData,
) -> Result<Vec<SearchMatch>> {
    let mut result = Vec::with_capacity(matches.len());

    for m in matches {
        let info = if let Match::WithField(id, field, ..) = m {
            let identifier = data
                .identifier(id)
                .ok_or_else(|| anyhow!("Failed to get identifier for id {}", id))?
                .to_string();
            let field = data
                .field(id, field)
                .ok_or_else(|| anyhow!("Failed to get field {} for id {}", field, id))?
                .to_string();
            MatchInfo::Text { identifier, field }
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

/// Perform text search on an index with given query and parameters
/// Returns a vector of search results for each query in the input
pub async fn perform_text_search(
    index: SearchIndex,
    queries: Vec<String>,
    params: SearchParams,
    model: Option<&(EmbeddingModel, EmbeddingParams)>,
) -> Result<Vec<Vec<SearchMatch>>> {
    match index {
        SearchIndex::Keyword(index) => {
            let SearchParams::Keyword(params) = params else {
                return Err(anyhow!("Invalid search parameters for keyword index"));
            };

            search_parallel(queries, move |query| {
                let matches = index.search(query.as_str(), &params)?;
                convert_to_text_search_matches(matches, index.data())
            })
            .await
        }
        SearchIndex::FullText(index) => {
            let SearchParams::FullText(params) = params else {
                return Err(anyhow!("Invalid search parameters for full-text index"));
            };

            search_parallel(queries, move |query| {
                let matches = index.search(query.as_str(), &params)?;
                convert_to_text_search_matches(matches, index.data())
            })
            .await
        }
        SearchIndex::TextEmbedding(index) => {
            let SearchParams::TextEmbedding(params) = params else {
                return Err(anyhow!(
                    "Invalid search parameters for text embedding index"
                ));
            };

            // For text embedding index with text query, we'd need to embed the text first
            let Some((model, model_params)) = model else {
                return Err(anyhow!("No embedding model specified for embedding search"));
            };
            let embeddings = match model {
                EmbeddingModel::SentenceTransformer(m) => m.embed(&queries, model_params)?,
                EmbeddingModel::Vllm(m) => m.embed(&queries, model_params)?,
            };
            search_parallel(embeddings, move |emb| {
                let matches = index.search(&emb, &params)?;
                convert_to_text_search_matches(matches, index.data().text_data())
            })
            .await
        }
        _ => Err(anyhow!(
            "Text query not supported by index type {}",
            index.index_type()
        )),
    }
}

/// Perform text search on an index with given query, parameters, and filter
/// Returns a vector of search results for each query in the input
pub async fn perform_text_search_with_filter(
    index: SearchIndex,
    queries: Vec<String>,
    params: SearchParams,
    model: Option<&(EmbeddingModel, EmbeddingParams)>,
    filter: impl Fn(u32) -> bool + Clone + Send + 'static,
) -> Result<Vec<Vec<SearchMatch>>> {
    match index {
        SearchIndex::Keyword(index) => {
            let SearchParams::Keyword(params) = params else {
                return Err(anyhow!("Invalid search parameters for keyword index"));
            };

            search_parallel(queries, move |query| {
                let matches = index.search_with_filter(query.as_str(), &params, filter.clone())?;
                convert_to_text_search_matches(matches, index.data())
            })
            .await
        }
        SearchIndex::FullText(index) => {
            let SearchParams::FullText(params) = params else {
                return Err(anyhow!("Invalid search parameters for full-text index"));
            };

            search_parallel(queries, move |query| {
                let matches = index.search_with_filter(query.as_str(), &params, filter.clone())?;
                convert_to_text_search_matches(matches, index.data())
            })
            .await
        }
        SearchIndex::TextEmbedding(index) => {
            let SearchParams::TextEmbedding(params) = params else {
                return Err(anyhow!(
                    "Invalid search parameters for text embedding index"
                ));
            };

            // For text embedding index with text query, we'd need to embed the text first
            let Some((model, model_params)) = model else {
                return Err(anyhow!("No embedding model specified for embedding search"));
            };
            let embeddings = match model {
                EmbeddingModel::SentenceTransformer(m) => m.embed(&queries, model_params)?,
                EmbeddingModel::Vllm(m) => m.embed(&queries, model_params)?,
            };
            search_parallel(embeddings, move |emb| {
                let matches = index.search_with_filter(&emb, &params, filter.clone())?;
                convert_to_text_search_matches(matches, index.data().text_data())
            })
            .await
        }
        _ => Err(anyhow!(
            "Text query not supported by index type {}",
            index.index_type()
        )),
    }
}

/// Perform search on an index with given query and parameters
/// Returns a vector of search results for each query in the input
pub async fn perform_search(
    index: SearchIndex,
    query: QueryType,
    params: SearchParams,
    model: Option<&(EmbeddingModel, EmbeddingParams)>,
) -> Result<Vec<Vec<SearchMatch>>> {
    match (index, query) {
        (index, QueryType::Text(text)) => perform_text_search(index, text, params, model).await,
        (SearchIndex::TextEmbedding(index), QueryType::Embedding(embedding)) => {
            let SearchParams::TextEmbedding(params) = params else {
                return Err(anyhow!(
                    "Invalid search parameters for text embedding index"
                ));
            };

            search_parallel(embedding, move |emb| {
                let matches = index.search(&emb, &params)?;
                convert_to_text_search_matches(matches, index.data().text_data())
            })
            .await
        }
        (SearchIndex::Embedding(index), QueryType::Embedding(embedding)) => {
            let SearchParams::Embedding(params) = params else {
                return Err(anyhow!("Invalid search parameters for embedding index"));
            };

            search_parallel(embedding, move |emb| {
                index.search(&emb, &params).map(convert_matches)
            })
            .await
        }
        (index, query) => Err(anyhow!(
            "Query type {} not supported by index type {}",
            query.type_name(),
            index.index_type()
        )),
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

    let matches = perform_search(index, req.query, params, model).await?;

    Ok(Json(SearchResponse { matches }))
}
