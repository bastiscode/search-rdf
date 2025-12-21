use anyhow::{Context, Result, anyhow};
use itertools::Itertools;
use log::info;
use memmap2::Mmap;
use safetensors::serialize_to_file;
use safetensors::tensor::{Dtype, TensorView};
use search_rdf::model::Embed;
use std::collections::HashMap;
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::mem::size_of;
use std::path::Path;
use tempfile::NamedTempFile;

use search_rdf::data::DataSource;
use search_rdf::data::text::TextData;
use search_rdf::model::{EmbeddingModel, EmbeddingParams};

use crate::search_rdf::config::{Config, EmbeddingDatasetConfig};
use crate::search_rdf::model::load_model;

const F32_SIZE: usize = size_of::<f32>();

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
        info!(
            "[OK] {} (type: {}, dimensions: {})",
            model_config.name,
            model.model_type(),
            model.num_dimensions()
        );
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
            EmbeddingDatasetConfig::Text {
                dataset,
                batch_size,
                params,
            } => {
                build_text_embeddings(dataset, &embed_config.output, *batch_size, params, model)?;
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
    output: &Path,
    batch_size: usize,
    params: &EmbeddingParams,
    model: &EmbeddingModel,
) -> Result<()> {
    // Load text data
    let text_data = TextData::load(dataset).context(format!(
        "Failed to load text data from: {}",
        dataset.display()
    ))?;

    info!("Loaded {} text items", text_data.len());
    info!(
        "Embedding {} text fields with batch size {}...",
        text_data.total_fields(),
        batch_size
    );

    // Log every 5% or every 100,000 embeddings, whichever is smaller
    let log_every = (text_data.total_fields() / 20).min(100_000).max(1);

    let temp_file = NamedTempFile::new().context("Failed to create temporary file")?;
    let temp_file_path = temp_file.path().to_path_buf();
    let mut temp_file = BufWriter::new(temp_file);

    let mut num_embeddings: usize = 0;
    for chunk in &text_data
        .items()
        .flat_map(|(_, fields)| fields)
        .chunks(batch_size)
    {
        let chunk: Vec<&str> = chunk.collect();
        let embeddings = match model {
            EmbeddingModel::SentenceTransformer(m) => m.embed(&chunk, params)?,
            EmbeddingModel::Vllm(m) => m.embed(&chunk, params)?,
        };

        let mut buffer = Vec::with_capacity(embeddings.len() * model.num_dimensions() * F32_SIZE);
        for embedding in &embeddings {
            for dim in embedding {
                buffer.extend(dim.to_le_bytes());
            }
            num_embeddings += 1;

            if num_embeddings.is_multiple_of(log_every) {
                let percentage = (num_embeddings as f64 / text_data.total_fields() as f64) * 100.0;
                info!(
                    "Generated {} / {} embeddings ({:.1}%)",
                    num_embeddings,
                    text_data.total_fields(),
                    percentage,
                );
            }
        }
        temp_file
            .write_all(&buffer)
            .context("Failed to write embeddings to temporary file")?;
    }

    let temp_bytes = unsafe { Mmap::map(&File::open(&temp_file_path)?)? };

    // Create tensors
    let embedding_tensor = TensorView::new(
        Dtype::F32,
        vec![text_data.total_fields() as usize, model.num_dimensions()],
        &temp_bytes,
    )?;

    // Create output directory
    if let Some(parent) = output.parent() {
        create_dir_all(parent)?;
    }

    // Serialize with model metadata
    serialize_to_file(
        [("embedding", embedding_tensor)],
        Some(HashMap::from([(
            String::from("model"),
            model.model_name().to_string(),
        )])),
        output,
    )?;

    info!("Saved embeddings to {}", output.display());

    Ok(())
}
