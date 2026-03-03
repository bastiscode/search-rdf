use numpy::ndarray::Array3;

use crate::model::AsInput;

pub mod open_clip;

/// Input for multi-modal embedding models that accept both text and images.
#[derive(Debug, Clone)]
pub enum MultiModalInput {
    Text(String),
    Image(Array3<u8>),
}

impl AsInput<MultiModalInput> for MultiModalInput {
    fn as_input(&self) -> &MultiModalInput {
        self
    }
}
