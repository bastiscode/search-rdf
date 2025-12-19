use crate::data::embedding::Embedding;
use anyhow::Result;
use serde::{Deserialize, Serialize};

pub mod text;

pub trait EmbeddingModel {
    type Input: ?Sized;
    type Params;

    fn embed<I>(&self, inputs: &[I], params: &Self::Params) -> Result<Vec<Embedding>>
    where
        I: AsRef<Self::Input>;

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

#[derive(Debug, Serialize, Deserialize)]
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
