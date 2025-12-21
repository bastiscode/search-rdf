use anyhow::{Result, anyhow};
use axum::{
    Router,
    extract::{Json, Path as AxumPath, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use futures::future::try_join_all;
use log::info;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};

use crate::search_rdf::index::load_index;
use search_rdf::index::text::embedding::Query;
use search_rdf::index::{Match, Search, SearchParams};
use search_rdf::{
    data::{DataSource, TextData},
    index::SearchIndex,
};

use crate::search_rdf::config::Config;

#[derive(Clone)]
struct AppState {
    indices: Arc<HashMap<String, SearchIndex>>,
}

impl AppState {
    fn new(indices: HashMap<String, SearchIndex>) -> Self {
        Self {
            indices: Arc::new(indices),
        }
    }
}

pub async fn run(config_path: &str) -> Result<()> {
    let config = Config::load(config_path)?;

    let Some(server) = config.server else {
        info!("No server configuration found.");
        return Ok(());
    };

    let Some(indices) = config.indices else {
        info!("No index configuration found.");
        return Ok(());
    };

    info!("Loading indices...");
    let mut search_indices = HashMap::new();

    for name in server.indices {
        info!("Loading {}...", name);

        let index_config = indices
            .iter()
            .find(|&index| index.name == name)
            .ok_or_else(|| anyhow!("Index configuration not found for {}", name))?;

        let search_index = load_index(&index_config.index_type, &index_config.output)?;
        info!("[OK] {}", name);

        search_indices.insert(name, search_index);
    }

    let state = AppState::new(search_indices);

    // Build router
    let mut app = Router::new()
        .route("/health", get(health))
        .route("/indices", get(list_indices))
        .route("/search/{index}", post(search))
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
    info!("Serving on http://{}", addr);
    info!("Available endpoints:");
    info!("GET  /health");
    info!("GET  /indices");
    info!("POST /search/{{index}}");

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
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
            index_type: index.index_type().to_string(),
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
    Text(Vec<String>),
    Embedding(Vec<Vec<f32>>),
}

impl QueryType {
    fn type_name(&self) -> &'static str {
        match self {
            QueryType::Text(..) => "text",
            QueryType::Embedding(..) => "embedding",
        }
    }
}

fn default_k() -> usize {
    10
}

#[derive(Serialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
enum MatchInfo {
    None,
    Text { identifier: String, field: String },
}

#[derive(Serialize)]
struct SearchMatch {
    id: u32,
    score: f32,
    #[serde(skip_serializing_if = "matches_none")]
    info: MatchInfo,
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
struct SearchResponse {
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
            MatchInfo::Text {
                identifier,
                field: field,
            }
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

async fn search(
    AxumPath(index_name): AxumPath<String>,
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, AppError> {
    let index = state.indices.get(&index_name).cloned().ok_or_else(|| {
        AppError(
            StatusCode::BAD_REQUEST,
            anyhow!("Index not found: {}", index_name),
        )
    })?;

    let mut params = SearchParams::default().with_k(req.k).with_exact(req.exact);

    if let Some(min_score) = req.min_score {
        params = params.with_min_score(min_score);
    }

    let matches = match (index, req.query) {
        (SearchIndex::Keyword(index), QueryType::Text(text)) => {
            search_parallel(text, move |text| {
                let matches = index.search(text.as_str(), &params)?;
                convert_to_text_search_matches(matches, index.data())
            })
            .await?
        }
        (SearchIndex::TextEmbedding(..), QueryType::Text(..)) => {
            // For text embedding index with text query, we'd need to embed the text first
            // This requires having an embedding model available
            return Err(AppError(
                StatusCode::BAD_REQUEST,
                anyhow!(
                    "Text queries for text_embedding indices require embedding model (not yet implemented)"
                ),
            ));
        }
        (SearchIndex::TextEmbedding(index), QueryType::Embedding(embedding)) => {
            search_parallel(embedding, move |emb| {
                let matches = index.search(Query::Embedding(&emb), &params)?;
                convert_to_text_search_matches(matches, index.data().text_data())
            })
            .await?
        }
        (SearchIndex::Embedding(index), QueryType::Embedding(embedding)) => {
            search_parallel(embedding, move |emb| {
                index.search(&emb, &params).map(convert_matches)
            })
            .await?
        }
        (index, query) => {
            return Err(AppError(
                StatusCode::BAD_REQUEST,
                anyhow!(
                    "Query type {} doesn't match index type {}",
                    query.type_name(),
                    index.index_type()
                ),
            ));
        }
    };

    Ok(Json(SearchResponse { matches }))
}

// Error handling
struct AppError(StatusCode, anyhow::Error);

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
