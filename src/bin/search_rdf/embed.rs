use anyhow::{Context, Result, anyhow};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use search_rdf::data::DataSource;
use search_rdf::model::EmbeddingModel;
use search_rdf::model::text::vllm::VLLM;
use search_rdf::{data::text::TextData, model::text::sentence_transformer::SentenceTransformer};

use crate::search_rdf::config::{Config, ModelConfig, ModelType};

pub fn run(config_path: &str, force: bool, only: Option<Vec<String>>) -> Result<()> {
    let config = Config::load(config_path)?;

    let Some(embed) = config.embeddings else {
        println!("No embedding configuration found.");
        return Ok(());
    };

    // Load models
    let mut models: HashMap<String, Box<dyn EmbeddingModel<Input = str, Params = _>>> =
        HashMap::new();

    for model_config in &config.embeddings.models {
        println!("Loading model: {}...", model_config.name);
        let model = load_model(model_config)?;
        models.insert(model_config.name.clone(), model);
        println!(
            "  [OK] Model loaded: {} ({} dimensions)",
            model_config.name,
            models[&model_config.name].num_dimensions()
        );
    }

    // Filter datasets if --only is specified
    let datasets: Vec<_> = if let Some(only_names) = only {
        config
            .embeddings
            .datasets
            .iter()
            .filter(|ds| only_names.contains(&ds.name))
            .collect()
    } else {
        config.embeddings.datasets.iter().collect()
    };

    println!("Generating embeddings for {} datasets...", datasets.len());

    for dataset in datasets {
        let output_path = Path::new(&dataset.output);

        if output_path.exists() && !force {
            println!(
                "  [SKIP] {} (output exists, use --force to rebuild)",
                dataset.name
            );
            continue;
        }

        println!("  [BUILD] {}...", dataset.name);

        let model = models
            .get(&dataset.model)
            .ok_or_else(|| anyhow!("Model not found: {}", dataset.model))?;

        build_embeddings(dataset, model.as_ref())?;
        println!("  [OK] {} -> {}", dataset.name, dataset.output);
    }

    println!("Done!");
    Ok(())
}

fn load_model(config: &ModelConfig) -> Result<Box<dyn EmbeddingModel<Input = str, Params = _>>> {
    match &config.model_type {
        ModelType::Vllm {
            endpoint,
            model_name,
        } => {
            let model = VLLM::new(endpoint.clone(), model_name.clone())?;
            Ok(Box::new(model))
        }
        ModelType::SentenceTransformer { model_name, device } => {
            let model = SentenceTransformer::load(model_name, device)?;
            Ok(Box::new(model))
        }
    }
}

fn build_embeddings(
    dataset: &crate::search_rdf::config::TextEmbeddingDataset,
    model: &dyn EmbeddingModel<Input = str, Params = _>,
) -> Result<()> {
    // Load text data
    let text_data_path = Path::new(&dataset.text_data);
    let text_data = TextData::load(text_data_path).context(format!(
        "Failed to load text data from: {}",
        dataset.text_data
    ))?;

    println!("    Loaded {} text items", text_data.len());

    // Create output directory
    let output_path = Path::new(&dataset.output);
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Collect all (id, fields) pairs
    let mut ids = Vec::new();
    let mut all_texts = Vec::new();

    for (id, fields) in text_data.items() {
        for field in fields {
            ids.push(id);
            all_texts.push(field.to_string());
        }
    }

    println!("    Embedding {} text fields...", all_texts.len());

    // Generate embeddings in batches
    let batch_size = dataset.params.batch_size;
    let mut all_embeddings = Vec::new();

    for (i, chunk) in all_texts.chunks(batch_size).enumerate() {
        if dataset.params.show_progress {
            println!(
                "    Batch {}/{} ({} texts)",
                i + 1,
                (all_texts.len() + batch_size - 1) / batch_size,
                chunk.len()
            );
        }

        // Determine params based on model type
        let embeddings = if model.model_type() == "vLLM" {
            // vLLM uses () as params
            model.embed(chunk, ())?
        } else {
            // SentenceTransformer uses EmbeddingParams
            use search_rdf::model::text::sentence_transformer::EmbeddingParams;
            let params = EmbeddingParams {
                batch_size: dataset.params.batch_size,
                num_dimensions: None,
                normalize: dataset.params.normalize,
                show_progress: false, // We handle progress ourselves
            };
            model.embed(chunk, params)?
        };

        all_embeddings.extend(embeddings);
    }

    println!("    Generated {} embeddings", all_embeddings.len());

    // Save as safetensors
    save_embeddings(&ids, &all_embeddings, model.model_name(), output_path)?;

    println!("    Saved to {}", output_path.display());

    Ok(())
}

fn save_embeddings(
    ids: &[u32],
    embeddings: &[Vec<f32>],
    model_name: &str,
    output_path: &Path,
) -> Result<()> {
    use safetensors::serialize;
    use safetensors::tensor::{Dtype, TensorView};
    use std::collections::HashMap;

    if ids.len() != embeddings.len() {
        return Err(anyhow!(
            "ID count ({}) doesn't match embedding count ({})",
            ids.len(),
            embeddings.len()
        ));
    }

    let num_embeddings = embeddings.len();
    let num_dimensions = embeddings[0].len();

    // Flatten embeddings
    let embedding_data: Vec<f32> = embeddings.iter().flatten().copied().collect();
    let embedding_bytes: Vec<u8> = embedding_data
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    // Convert IDs to bytes
    let id_bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();

    // Create tensors
    let embedding_tensor = TensorView::new(
        Dtype::F32,
        vec![num_embeddings, num_dimensions],
        &embedding_bytes,
    )?;
    let id_tensor = TensorView::new(Dtype::U32, vec![num_embeddings], &id_bytes)?;

    // Serialize with model metadata
    let tensors = vec![("embedding", embedding_tensor), ("id", id_tensor)];
    let metadata = HashMap::from([("model".to_string(), model_name.to_string())]);
    let bytes = serialize(tensors, Some(metadata))?;

    // Write to file
    let safetensors_path = output_path.join("embedding.safetensors");
    fs::write(&safetensors_path, bytes)?;

    // Build the id-map
    search_rdf::data::embedding::Embeddings::build(output_path)?;

    Ok(())
}
