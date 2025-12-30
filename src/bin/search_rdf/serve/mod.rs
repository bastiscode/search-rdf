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
use std::{collections::HashMap, path::Path};
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};

use crate::search_rdf::config::Config;
use crate::search_rdf::{index::load_index, model::load_model};

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

    info!("Loading indices and models...");
    let mut search_indices = HashMap::new();
    let mut models = HashMap::new();
    let mut index_to_model: HashMap<String, String> = HashMap::new();

    for name in &server.indices {
        info!("Loading index {}...", name);

        let index_config = config
            .indices
            .as_ref()
            .ok_or_else(|| anyhow!("No index configurations found"))?
            .iter()
            .find(|&index| &index.name == name)
            .ok_or_else(|| anyhow!("Index configuration not found for {}", name))?;

        if let Some(model) = index_config.index_type.get_model()
            && !models.contains_key(model)
        {
            info!("  - Requires model: {}", model);

            let model_config = config
                .models
                .as_ref()
                .ok_or_else(|| anyhow!("No model configurations found"))?
                .iter()
                .find(|&m| m.name == model)
                .ok_or_else(|| anyhow!("Model configuration not found for {}", model))?;

            let model = load_model(&model_config.model_type)?;
            info!(
                "  [OK] {} (type: {}, dimensions: {}, max_input_len: {})",
                name,
                model.model_type(),
                model.num_dimensions(),
                model
                    .max_input_len()
                    .map(|len| len.to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            );

            models.insert(model_config.name.clone(), (model, model_config.params));
            index_to_model.insert(name.to_string(), model_config.name.clone());
        }

        let search_index = load_index(config_dir, &index_config.index_type, &index_config.output)?;
        info!("[OK] {}", name);

        search_indices.insert(name.to_string(), search_index);
    }

    let state = AppState::new(search_indices, models, index_to_model);

    // Build router
    let mut app = Router::new()
        .route("/health", get(health))
        .route("/indices", get(list_indices))
        .route("/search/{index}", post(search))
        .route("/service/{index}", post(service))
        .route("/qlproxy/{index}", post(qlproxy))
        .layer(DefaultBodyLimit::max(1024 * 1024 * 1024)) // 1 GB
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
    info!("POST /service/{{index}}");
    info!("POST /qlproxy/{{index}}");

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
