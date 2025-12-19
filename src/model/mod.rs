use crate::data::embedding::Embedding;
use anyhow::Result;

pub mod text;

pub trait EmbeddingModel {
    type Input: ?Sized;
    type Params;

    fn embed<I>(&self, inputs: &[I], params: Self::Params) -> Result<Vec<Embedding>>
    where
        I: AsRef<Self::Input>;

    fn num_dimensions(&self) -> usize;

    fn model_name(&self) -> &str;

    fn model_type(&self) -> &str;
}
