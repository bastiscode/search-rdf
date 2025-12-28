use anyhow::{Context, Result, anyhow};
use itertools::Itertools;
use log::info;
use memmap2::Mmap;
use safetensors::serialize_to_file;
use safetensors::tensor::{Dtype, TensorView};
use search_rdf::model::Embed;
use std::collections::HashMap;
use std::fs::{File, create_dir_all, remove_file};
use std::io::{BufWriter, Write};
use std::path::Path;

use search_rdf::data::DataSource;
use search_rdf::data::text::TextData;
use search_rdf::model::{EmbeddingModel, EmbeddingParams};
use search_rdf::utils::progress_bar;

use crate::search_rdf::config::{Config, EmbeddingDatasetConfig};
use crate::search_rdf::model::load_model;

pub fn run(config_path: &Path, force: bool) -> Result<()> {
    let config = Config::load(config_path)?;
    let config_dir = config_path
        .parent()
        .expect("Failed to get config directory");

    info!("Building embeddings in {}", config_dir.display());

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
    let mut models = HashMap::new();
    info!("Loading {} models...", model_configs.len());

    for model_config in &model_configs {
        let model = load_model(&model_config.model_type)?;
        info!(
            "[OK] {} (type: {}, dimensions: {}, max_input_len: {})",
            model_config.name,
            model.model_type(),
            model.num_dimensions(),
            model
                .max_input_len()
                .map(|len| len.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        );
        models.insert(model_config.name.clone(), (model, model_config.params));
    }

    info!("Generating embeddings for {} datasets...", embed.len());

    for embed_config in &embed {
        if config_dir.join(&embed_config.output).exists() && !force {
            info!(
                "[SKIP] {} (output exists, use --force to rebuild)",
                embed_config.name
            );
            continue;
        }

        let (model, params) = models
            .get(&embed_config.model)
            .ok_or_else(|| anyhow!("Model not found: {}", embed_config.model))?;

        info!("[BUILD] {}...", embed_config.name);

        match &embed_config.dataset {
            EmbeddingDatasetConfig::Text {
                dataset,
                batch_size,
            } => {
                build_text_embeddings(
                    config_dir,
                    dataset,
                    &embed_config.output,
                    *batch_size,
                    params,
                    model,
                )?;
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
    base_dir: &Path,
    dataset: &Path,
    output: &Path,
    batch_size: usize,
    params: &EmbeddingParams,
    model: &EmbeddingModel,
) -> Result<()> {
    // Load text data
    let text_data = TextData::load(&base_dir.join(dataset)).context(format!(
        "Failed to load text data from: {}",
        dataset.display()
    ))?;

    info!("Loaded {} text items", text_data.len());
    info!(
        "Embedding {} text fields with batch size {}...",
        text_data.total_fields(),
        batch_size
    );

    // Create output directory
    if let Some(parent) = base_dir.join(output).parent() {
        create_dir_all(parent)?;
    }

    let temp_file_path = base_dir.join(output).with_added_extension("tmp");
    let (temp_file, skip) = if temp_file_path.exists() {
        let temp_bytes = unsafe { Mmap::map(&File::open(&temp_file_path)?)? };
        if !temp_bytes
            .len()
            .is_multiple_of(size_of::<f32>() * model.num_dimensions())
        {
            return Err(anyhow!(
                "Temporary file at {} has invalid size, please remove it and try again",
                temp_file_path.display()
            ));
        }
        info!(
            "Resuming from existing temporary file: {}",
            temp_file_path.display()
        );
        let skip = temp_bytes.len() / (size_of::<f32>() * model.num_dimensions());
        info!("Skipping {} already embedded fields", skip);
        let file = File::options().append(true).open(&temp_file_path)?;
        (file, skip)
    } else {
        info!(
            "Using temporary file for embeddings: {}",
            temp_file_path.display()
        );
        (File::create(&temp_file_path)?, 0)
    };
    let mut temp_file = BufWriter::new(temp_file);

    let pb = progress_bar(
        "Generating embeddings",
        Some(text_data.total_fields() as u64 - skip as u64),
    )?;
    for chunk in &text_data
        .items()
        .flat_map(|(_, fields)| fields)
        .skip(skip)
        .chunks(batch_size)
    {
        let chunk: Vec<&str> = chunk.collect();
        let embeddings = match model {
            EmbeddingModel::SentenceTransformer(m) => m.embed(&chunk, params)?,
            EmbeddingModel::Vllm(m) => m.embed(&chunk, params)?,
        };
        pb.inc(embeddings.len() as u64);
        embeddings
            .into_iter()
            .flatten()
            .map(|f| f.to_le_bytes())
            .try_for_each(|b| temp_file.write_all(&b))
            .context("Failed to write embeddings to temporary file")?;
    }

    temp_file.flush()?;
    pb.finish_with_message("Embeddings generated");

    let temp_bytes = unsafe { Mmap::map(&File::open(&temp_file_path)?)? };

    // Create tensors
    let embedding_tensor = TensorView::new(
        Dtype::F32,
        vec![text_data.total_fields() as usize, model.num_dimensions()],
        &temp_bytes,
    )?;

    // Serialize with model metadata
    serialize_to_file(
        [("embedding", embedding_tensor)],
        Some(HashMap::from([(
            String::from("model"),
            model.model_name().to_string(),
        )])),
        &base_dir.join(output),
    )?;

    info!("Saved embeddings to {}", output.display());

    // Remove temporary file
    remove_file(&temp_file_path)?;
    info!(
        "Removed temporary embedding file at {}",
        temp_file_path.display()
    );

    Ok(())
}
