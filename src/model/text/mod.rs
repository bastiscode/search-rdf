use crate::model::AsInput;

pub mod sentence_transformer;
pub mod vllm;

impl AsInput<str> for String {
    fn as_input(&self) -> &str {
        self.as_str()
    }
}

impl AsInput<str> for str {
    fn as_input(&self) -> &str {
        self
    }
}
