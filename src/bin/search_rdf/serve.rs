use anyhow::{Context, Result, anyhow};
use axum::{
    Router,
    extract::{Json, Path as AxumPath, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};

use search_rdf::data::{
    DataSource,
    text::{TextData, TextEmbeddings},
};
use search_rdf::index::text::{KeywordIndex, TextEmbeddingIndex};
use search_rdf::index::{EmbeddingIndex, Match, SearchIndex, SearchParams};
use search_rdf::{data::embedding::Embeddings, index::text::embedding::Query};

use crate::search_rdf::config::Config;

#[derive(Clone)]
struct AppState {
    indices: Arc<HashMap<String, LoadedIndex>>,
}

enum LoadedIndex {
    Keyword { index: KeywordIndex },
    TextEmbedding { index: TextEmbeddingIndex },
    Embedding { index: EmbeddingIndex },
}

impl LoadedIndex {
    fn index_type(&self) -> &'static str {
        match self {
            LoadedIndex::Keyword { index } => index.index_type(),
            LoadedIndex::TextEmbedding { index } => index.index_type(),
            LoadedIndex::Embedding { index } => index.index_type(),
        }
    }
}

pub async fn run(config_path: &str) -> Result<()> {
    let config = Config::load(config_path)?;

    let Some(server) = config.server else {
        println!("No server configuration found.");
        return Ok(());
    };

    println!("Loading indices...");
    let mut indices = HashMap::new();

    for server_index in &server.indices {
        println!(
            "  Loading {} ({})...",
            server_index.name, server_index.index_type
        );

        let index = load_index(&server_index.index_type, &server_index.path)?;
        indices.insert(server_index.name.clone(), index);

        println!("  [OK] {}", server_index.name);
    }

    let state = AppState {
        indices: Arc::new(indices),
    };

    // Build router
    let mut app = Router::new()
        .route("/health", get(health))
        .route("/indices", get(list_indices))
        .route("/search/:index", post(search))
        .with_state(state);

    // Add CORS if enabled
    if server.cors {
        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any);
        app = app.layer(cors);
    }

    let addr = format!("{}:{}", server.host, server.port);
    println!("\nServing on http://{}", addr);
    println!("Available endpoints:");
    println!("  GET  /health");
    println!("  GET  /indices");
    println!("  POST /search/:index");

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

fn load_index(index_type: &str, path: &Path) -> Result<LoadedIndex> {
    match index_type.to_lowercase().as_str() {
        "keyword" => {
            let text_data = TextData::load(
                path.parent()
                    .and_then(|p| p.parent())
                    .ok_or_else(|| anyhow!("Invalid index path structure"))?,
            )?;
            let index = KeywordIndex::load(text_data, path)?;
            Ok(LoadedIndex::Keyword { index })
        }
        "text_embedding" | "text-embedding" => {
            let base_path = path
                .parent()
                .and_then(|p| p.parent())
                .ok_or_else(|| anyhow!("Invalid index path structure"))?;

            let text_embeddings = TextEmbeddings::load(base_path, base_path)?;

            let index = TextEmbeddingIndex::load(text_embeddings, index_path)?;
            Ok(LoadedIndex::TextEmbedding { index })
        }
        "embedding" => {
            let base_path = path
                .parent()
                .and_then(|p| p.parent())
                .ok_or_else(|| anyhow!("Invalid index path structure"))?;

            let embeddings = Embeddings::load(base_path)?;
            let index = EmbeddingIndex::load(embeddings, index_path)?;
            Ok(LoadedIndex::Embedding { index })
        }
        _ => Err(anyhow!("Unknown index type: {}", index_type)),
    }
}

// Health check endpoint
async fn health() -> &'static str {
    "OK"
}

// List available indices
#[derive(Serialize)]
struct IndexInfo {
    name: String,
    index_type: String,
}

async fn list_indices(State(state): State<AppState>) -> Json<Vec<IndexInfo>> {
    let indices = state
        .indices
        .iter()
        .map(|(name, index)| IndexInfo {
            name: name.clone(),
            index_type: match index {
                LoadedIndex::Keyword { index } => index.index_type(),
                LoadedIndex::TextEmbedding { index } => index.index_type(),
                LoadedIndex::Embedding { index } => index.index_type(),
            },
        })
        .collect();

    Json(indices)
}

// Search request/response types
#[derive(Deserialize)]
struct SearchRequest {
    query: QueryType,
    #[serde(default = "default_k")]
    k: usize,
    min_score: Option<f32>,
    #[serde(default)]
    exact: bool,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum QueryType {
    Text { text: String },
    Embedding { embedding: Vec<f32> },
}

fn default_k() -> usize {
    10
}

#[derive(Serialize)]
struct SearchResponse {
    matches: Vec<SearchMatch>,
}

#[derive(Serialize)]
struct SearchMatch {
    id: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    field: Option<usize>,
    score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
}

async fn search(
    AxumPath(index_name): AxumPath<String>,
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, AppError> {
    let index = state
        .indices
        .get(&index_name)
        .ok_or_else(|| anyhow!("Index not found: {}", index_name))?;

    let mut params = SearchParams::default().with_k(req.k).with_exact(req.exact);

    if let Some(min_score) = req.min_score {
        params = params.with_min_score(min_score);
    }

    let matches = match (index, &req.query) {
        (LoadedIndex::Keyword { index }, QueryType::Text { text }) => {
            index.search(text.as_str(), &params)?
        }
        (LoadedIndex::TextEmbedding { index }, QueryType::Text { text }) => {
            // For text embedding index with text query, we'd need to embed the text first
            // This requires having an embedding model available
            return Err(anyhow!(
                "Text queries for text_embedding indices require embedding model (not yet implemented)"
            )
            .into());
        }
        (LoadedIndex::TextEmbedding { index }, QueryType::Embedding { embedding }) => {
            index.search(Query::Embedding(embedding), &params)?
        }
        (LoadedIndex::Embedding { index }, QueryType::Embedding { embedding }) => {
            index.search(embedding.as_slice(), &params)?
        }
        _ => {
            return Err(anyhow!("Query type doesn't match index type").into());
        }
    };

    // Convert matches to response format and optionally fetch text
    let response_matches = matches
        .into_iter()
        .map(|m| match m {
            Match::Regular(id, score) => SearchMatch {
                id,
                field: None,
                score,
                text: None,
            },
            Match::WithField(id, field, score) => {
                // Try to fetch text if it's a text-based index
                let text = match index {
                    LoadedIndex::Keyword { index } => {
                        index.data().field(id, field).map(|s| s.to_string())
                    }
                    LoadedIndex::TextEmbedding { index } => index
                        .data()
                        .text_data()
                        .field(id, field)
                        .map(|s| s.to_string()),
                    LoadedIndex::Embedding { .. } => None,
                };

                SearchMatch {
                    id,
                    field: Some(field),
                    score,
                    text,
                }
            }
        })
        .collect();

    Ok(Json(SearchResponse {
        matches: response_matches,
    }))
}

// Error handling
struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Error: {}", self.0),
        )
            .into_response()
    }
}

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}
