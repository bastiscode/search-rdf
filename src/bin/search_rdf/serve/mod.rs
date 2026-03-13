mod handlers;
mod search;
mod sparql;
mod types;

use anyhow::{Result, anyhow};
use axum::{
    Router,
    extract::DefaultBodyLimit,
    routing::{get, post},
};
use log::info;
use parse_size::parse_size;
use std::{collections::HashMap, path::Path};
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};

use crate::search_rdf::index::load_index;
use crate::search_rdf::{config::Config, model::load_model_and_params};

use self::handlers::{health, list_indices};
use self::search::search;
use self::sparql::{qlproxy, service};
use self::types::AppState;

pub async fn run(config_path: &Path) -> Result<()> {
    let config = Config::load(config_path)?;
    let config_dir = config_path
        .parent()
        .ok_or_else(|| anyhow!("Failed to get config directory"))?;

    info!("Starting server from {}...", config_path.display());

    let Some(server) = &config.server else {
        info!("No server configuration found.");
        return Ok(());
    };

    let mut search_indices = HashMap::new();
    let mut models = HashMap::new();
    let mut index_to_model: HashMap<String, String> = HashMap::new();
    let mut descriptions: HashMap<String, String> = HashMap::new();

    for name in &server.indices {
        info!("Loading index {}...", name);

        let index_config = config
            .indices
            .as_ref()
            .ok_or_else(|| anyhow!("No index configurations found"))?
            .iter()
            .find(|&index| &index.name == name)
            .ok_or_else(|| anyhow!("Index configuration not found for {}", name))?;

        if let Some(model) = index_config.index_type.get_model() {
            index_to_model.insert(name.to_string(), model.to_string());

            if !models.contains_key(model) {
                info!("Index requires model: {}", model);
                let (emb_model, emb_params) = load_model_and_params(model, &config)?;
                models.insert(model.to_string(), (emb_model, emb_params));
            }
        }

        if let Some(desc) = &index_config.description {
            descriptions.insert(name.to_string(), desc.clone());
        }

        let search_index = load_index(config_dir, &index_config.index_type, &index_config.output)?;
        info!("[OK] {}", name);

        search_indices.insert(name.to_string(), search_index);
    }

    let addr = format!("{}:{}", server.host, server.port);
    info!("Serving on http://{}", addr);
    info!("Available endpoints:");
    info!("GET  /health");
    info!("GET  /indices");
    info!("POST /search/{{index}}");

    let body_limit: usize = parse_size(&server.max_input_size)?.try_into()?;

    // Build router
    let mut app = Router::new()
        .layer(DefaultBodyLimit::max(body_limit))
        .route("/health", get(health))
        .route("/indices", get(list_indices))
        .route("/search/{index}", post(search));

    if server.sparql.is_some() {
        // increase body limit if SPARQL service is enabled, because we
        // might receive large results in the qlproxy endpoint
        app = app
            .route("/sparql/{index}", post(service))
            .route("/sparql/qlproxy/{index}", post(qlproxy));

        info!("POST /sparql/{{index}}");
        info!("POST /sparql/qlproxy/{{index}}");
    };

    // Add CORS if enabled
    if server.cors {
        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any);
        app = app.layer(cors);
    }

    let state = AppState::new(
        search_indices,
        models,
        index_to_model,
        descriptions,
        server.sparql.clone(),
    );
    let app = app.with_state(state);

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
