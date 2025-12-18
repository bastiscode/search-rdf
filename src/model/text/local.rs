use anyhow::Result;
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};

use crate::{
    data::{Precision, embedding::Embedding},
    model::EmbeddingModel,
};

pub struct EmbeddingParams {
    precision: Precision,
    batch_size: usize,
    num_dimensions: Option<usize>,
    normalize: bool,
    show_progress: bool,
}

pub struct TextEmbeddingModel {
    model: Py<PyAny>,
    name: String,
}

impl TextEmbeddingModel {
    fn load_python_model<'py>(
        py: Python<'py>,
        name: &str,
        device: &str,
    ) -> Result<Bound<'py, PyAny>> {
        let module = py.import("search_rdf.model")?;
        let model_class = module.getattr("TextEmbeddingModel")?;
        let model_instance = model_class.call1((name, device))?;
        Ok(model_instance)
    }

    pub fn load(name: &str, device: &str) -> Result<Self> {
        let model = Python::attach(|py| -> Result<Py<PyAny>> {
            let model_instance = Self::load_python_model(py, name, device)?;
            Ok(model_instance.into())
        })?;
        Ok(Self {
            model,
            name: name.to_string(),
        })
    }
}

impl<I> EmbeddingModel<I> for TextEmbeddingModel
where
    I: AsRef<str>,
{
    type Params = EmbeddingParams;

    fn embed(&self, inputs: &[I], params: Self::Params) -> Result<Vec<Embedding>> {
        Python::attach(|py| {
            let model = self.model.bind(py);
            let py_inputs = PyList::new(py, inputs.iter().map(|s| s.as_ref()))?;
            let py_params = PyDict::new(py);
            py_params.set_item("precision", params.precision.into_pyobject(py)?)?;
            py_params.set_item("batch_size", params.batch_size)?;
            py_params.set_item("num_dimensions", params.num_dimensions)?;
            py_params.set_item("normalize", params.normalize)?;
            py_params.set_item("show_progress", params.show_progress)?;

            let py_embeddings = model.getattr("embed")?.call1((py_inputs, py_params))?;
            Ok(py_embeddings.extract()?)
        })
    }

    fn num_dimensions(&self) -> usize {
        Python::attach(|py| {
            let model = self.model.bind(py);
            model
                .getattr("dim")
                .expect("Failed to get model dim attribute")
                .extract()
                .expect("Failed to extract model dim as usize")
        })
    }

    fn model_name(&self) -> &str {
        &self.name
    }

    fn model_type(&self) -> &str {
        "TextEmbeddingModel"
    }
}
