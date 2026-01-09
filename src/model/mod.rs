use crate::{
    data::embedding::Embedding,
    model::image::huggingface::HuggingFaceImageModel,
    model::text::{sentence_transformer::SentenceTransformer, vllm::VLLM},
};
use anyhow::Result;
use serde::{Deserialize, Serialize};

pub mod image;
pub mod text;
pub(crate) mod utils;

#[derive(Debug, Clone)]
pub enum EmbeddingModel {
    SentenceTransformer(SentenceTransformer),
    Vllm(VLLM),
    HuggingFaceImage(HuggingFaceImageModel),
}

impl EmbeddingModel {
    pub fn model_name(&self) -> &str {
        match self {
            EmbeddingModel::SentenceTransformer(m) => m.model_name(),
            EmbeddingModel::Vllm(m) => m.model_name(),
            EmbeddingModel::HuggingFaceImage(m) => m.model_name(),
        }
    }

    pub fn model_type(&self) -> &str {
        match self {
            EmbeddingModel::SentenceTransformer(m) => m.model_type(),
            EmbeddingModel::Vllm(m) => m.model_type(),
            EmbeddingModel::HuggingFaceImage(m) => m.model_type(),
        }
    }

    pub fn max_input_len(&self) -> Option<usize> {
        match self {
            EmbeddingModel::SentenceTransformer(..) => None,
            EmbeddingModel::Vllm(m) => Some(m.max_model_len()),
            EmbeddingModel::HuggingFaceImage(..) => None,
        }
    }

    pub fn num_dimensions(&self) -> usize {
        match self {
            EmbeddingModel::SentenceTransformer(m) => m.num_dimensions(),
            EmbeddingModel::Vllm(m) => m.num_dimensions(),
            EmbeddingModel::HuggingFaceImage(m) => m.num_dimensions(),
        }
    }
}

pub trait AsInput<T: ?Sized> {
    fn as_input(&self) -> &T;
}

impl<T: ?Sized, U: ?Sized + AsInput<T>> AsInput<T> for &U {
    fn as_input(&self) -> &T {
        (*self).as_input()
    }
}

pub trait Embed {
    type Input: ?Sized;
    type Params;

    fn embed<I>(&self, inputs: &[I], params: &Self::Params) -> Result<Vec<Embedding>>
    where
        I: AsInput<Self::Input>;

    fn num_dimensions(&self) -> usize;

    fn model_name(&self) -> &str;

    fn model_type(&self) -> &str;
}

pub fn normalize_embedding(mut embedding: Embedding) -> Embedding {
    let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return embedding;
    }
    for val in embedding.iter_mut() {
        *val /= norm;
    }
    embedding
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct EmbeddingParams {
    num_dimensions: Option<usize>,
    #[serde(default = "default_normalize")]
    normalize: bool,
}

impl EmbeddingParams {
    pub fn apply(&self, mut embedding: Embedding) -> Embedding {
        if let Some(num_dimensions) = self.num_dimensions {
            embedding.truncate(num_dimensions);
        }
        if self.normalize {
            embedding = normalize_embedding(embedding);
        }
        embedding
    }
}

fn default_normalize() -> bool {
    true
}

impl Default for EmbeddingParams {
    fn default() -> Self {
        Self {
            num_dimensions: None,
            normalize: true,
        }
    }
}
