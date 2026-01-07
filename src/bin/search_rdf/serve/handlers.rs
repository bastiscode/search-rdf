use axum::extract::{Json, State};
use serde::Serialize;

use crate::search_rdf::index::SearchIndex;

use super::types::AppState;

// Health check endpoint
pub async fn health() -> &'static str {
    "OK"
}

// List available indices
#[derive(Serialize)]
pub struct IndexInfo {
    name: String,
    #[serde(rename = "type")]
    index_type: String,
    #[serde(rename = "supported-query-types")]
    query_types: Vec<&'static str>,
}

pub async fn list_indices(State(state): State<AppState>) -> Json<Vec<IndexInfo>> {
    let indices = state
        .inner
        .indices
        .iter()
        .map(|(name, index)| IndexInfo {
            name: name.clone(),
            index_type: index.index_type().to_string(),
            query_types: match index {
                SearchIndex::Keyword(..) => vec!["text"],
                SearchIndex::FullText(..) => vec!["text"],
                SearchIndex::Embedding(..) => vec!["embedding"],
                SearchIndex::EmbeddingWithData(..) => vec!["text", "embedding"],
            },
        })
        .collect();

    Json(indices)
}
