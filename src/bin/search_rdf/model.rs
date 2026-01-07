use anyhow::Result;
use search_rdf::model::{
    EmbeddingModel,
    image::huggingface::HuggingFaceImageModel,
    text::{sentence_transformer::SentenceTransformer, vllm::VLLM},
};

use crate::search_rdf::config::ModelType;

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
    }
}
