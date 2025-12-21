use anyhow::{Context, Result, anyhow};
use log::info;
use std::collections::HashMap;
use std::path::Path;

use search_rdf::data::DataSource;
use search_rdf::data::text::TextData;
use search_rdf::model::{EmbeddingModel, EmbeddingParams};

use crate::search_rdf::config::{Config, EmbeddingDatasetConfig};
use crate::search_rdf::model::load_model;

pub fn run(config_path: &str, force: bool) -> Result<()> {
    let config = Config::load(config_path)?;

    let Some(embed) = config.embeddings else {
        info!("No embedding configuration found.");
        return Ok(());
    };

    let Some(mut model_configs) = config.models else {
        info!("No models defined in configuration.");
        return Ok(());
    };

    model_configs.retain(|model| {
        embed
            .iter()
            .any(|embed_config| embed_config.model == model.name)
    });

    // Load models
    let mut models: HashMap<String, EmbeddingModel> = HashMap::new();
    info!("Loading {} models...", model_configs.len());

    for model_config in model_configs {
        let model = load_model(&model_config.model_type)?;
        info!("[OK] Model loaded: {}", model_config.name);
        models.insert(model_config.name, model);
    }

    info!("Generating embeddings for {} datasets...", embed.len());

    for embed_config in &embed {
        if embed_config.output.exists() && !force {
            info!(
                "[SKIP] {} (output exists, use --force to rebuild)",
                embed_config.name
            );
            continue;
        }

        let model = models
            .get(&embed_config.model)
            .ok_or_else(|| anyhow!("Model not found: {}", embed_config.model))?;

        info!("[BUILD] {}...", embed_config.name);

        match &embed_config.dataset {
            EmbeddingDatasetConfig::Text { dataset, params } => {
                build_text_embeddings(dataset, params, model)?;
            }
        }

        info!(
            "[OK] {} -> {}",
            embed_config.name,
            embed_config.output.display()
        );
    }

    Ok(())
}

fn build_text_embeddings(
    dataset: &Path,
    _params: &EmbeddingParams,
    _model: &EmbeddingModel,
) -> Result<()> {
    // Load text data
    let text_data = TextData::load(dataset).context(format!(
        "Failed to load text data from: {}",
        dataset.display()
    ))?;

    info!("Loaded {} text items", text_data.len());

    info!("Embedding {} text fields...", text_data.total_fields());

    // Generate embeddings as batches and write them to a temporary file
    // Then convert that file to a safetensors file

    todo!();
}
