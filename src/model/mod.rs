use crate::data::embedding::Embedding;
use anyhow::Result;

pub mod text;

pub trait EmbeddingModel<I> {
    type Params;

    fn embed(&self, inputs: &[I], params: Self::Params) -> Result<Vec<Embedding>>;

    fn num_dimensions(&self) -> usize;

    fn model_name(&self) -> &str;

    fn model_type(&self) -> &str;
}
