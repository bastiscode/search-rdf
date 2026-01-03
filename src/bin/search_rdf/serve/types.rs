use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use search_rdf::model::{EmbeddingModel, EmbeddingParams};
use std::collections::HashMap;
use std::sync::Arc;

use crate::search_rdf::config::SparqlConfig;
use crate::search_rdf::index::SearchIndex;

pub struct Inner {
    pub indices: HashMap<String, SearchIndex>,
    pub models: HashMap<String, (EmbeddingModel, EmbeddingParams)>,
    pub index_to_model: HashMap<String, String>,
    pub sparql: Option<SparqlConfig>,
}

#[derive(Clone)]
pub struct AppState {
    pub inner: Arc<Inner>,
}

impl AppState {
    pub fn new(
        indices: HashMap<String, SearchIndex>,
        models: HashMap<String, (EmbeddingModel, EmbeddingParams)>,
        index_to_model: HashMap<String, String>,
        sparql: Option<SparqlConfig>,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                indices,
                models,
                index_to_model,
                sparql,
            }),
        }
    }
}

// Error handling
pub struct AppError(pub StatusCode, pub anyhow::Error);

impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        AppError(StatusCode::INTERNAL_SERVER_ERROR, err)
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (self.0, format!("Error: {}", self.1)).into_response()
    }
}
