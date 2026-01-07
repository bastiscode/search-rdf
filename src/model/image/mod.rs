use numpy::ndarray::{ArrayBase, Data, Dimension};

use crate::model::AsInput;

pub mod huggingface;

impl<S, D> AsInput<ArrayBase<S, D>> for ArrayBase<S, D>
where
    S: Data,
    D: Dimension,
{
    fn as_input(&self) -> &ArrayBase<S, D> {
        self
    }
}
