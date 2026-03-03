use std::collections::HashMap;
use std::collections::hash_map::Entry::{Occupied, Vacant};

use anyhow::{Result, anyhow};
use log::info;
use search_rdf::model::EmbeddingParams;
use search_rdf::model::{
    EmbeddingModel,
    image::huggingface::HuggingFaceImageModel,
    multimodal::open_clip::OpenClipModel,
    text::{sentence_transformer::SentenceTransformer, vllm::VLLM},
};

use crate::search_rdf::config::{Config, ModelType};

pub fn load_model(model_type: &ModelType) -> Result<EmbeddingModel> {
    match model_type {
        ModelType::Vllm {
            endpoint,
            model_name,
        } => {
            let vllm = VLLM::new(endpoint, model_name)?;
            Ok(EmbeddingModel::Vllm(vllm))
        }

        ModelType::SentenceTransformer {
            model_name,
            device,
            batch_size,
        } => {
            let st = SentenceTransformer::load(model_name, device, *batch_size)?;
            Ok(EmbeddingModel::SentenceTransformer(st))
        }

        ModelType::HuggingFaceImage {
            model_name,
            device,
            batch_size,
        } => {
            let img_model = HuggingFaceImageModel::load(model_name, device, *batch_size)?;
            Ok(EmbeddingModel::HuggingFaceImage(img_model))
        }

        ModelType::OpenClip {
            model,
            device,
            batch_size,
        } => {
            let clip_model = OpenClipModel::load(model, device, *batch_size)?;
            Ok(EmbeddingModel::OpenClip(clip_model))
        }
    }
}

pub fn load_model_and_params(
    model: &str,
    config: &Config,
) -> Result<(EmbeddingModel, EmbeddingParams)> {
    let model_config = config
        .models
        .as_ref()
        .ok_or_else(|| anyhow!("No model configurations found"))?
        .iter()
        .find(|m| m.name.as_str() == model)
        .ok_or_else(|| anyhow!("Model configuration not found for {}", model))?;

    info!("Loading model: {}", model_config.name);
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
    Ok((model, model_config.params))
}

pub fn get_or_load_model_and_params<'m>(
    model: &str,
    models: &'m mut HashMap<String, (EmbeddingModel, EmbeddingParams)>,
    config: &Config,
) -> Result<&'m (EmbeddingModel, EmbeddingParams)> {
    match models.entry(model.to_string()) {
        Occupied(entry) => Ok(entry.into_mut()),
        Vacant(entry) => {
            let (model, params) = load_model_and_params(model, config)?;
            Ok(entry.insert((model, params)))
        }
    }
}
