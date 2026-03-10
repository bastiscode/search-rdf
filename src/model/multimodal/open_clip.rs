use std::sync::Arc;

use anyhow::Result;
use numpy::ToPyArray;
use pyo3::{prelude::*, types::PyDict};

use crate::model::AsInput;
use crate::model::utils::ignore_system_signals;
use crate::{
    data::embedding::Embedding,
    model::{Embed, EmbeddingParams},
};

use super::MultiModalInput;

#[derive(Debug)]
struct Inner {
    model: Py<PyAny>,
    name: String,
    batch_size: usize,
}

#[derive(Debug, Clone)]
pub struct OpenClipModel {
    inner: Arc<Inner>,
}

impl OpenClipModel {
    fn load_python_model<'py>(
        py: Python<'py>,
        name: &str,
        device: &str,
    ) -> Result<Bound<'py, PyAny>> {
        let module = py.import("search_rdf.model")?;
        let model_class = module.getattr("OpenClipModel")?;
        let model_instance = model_class.call1((name, device))?;
        Ok(model_instance)
    }

    pub fn load(name: &str, device: &str, batch_size: usize) -> Result<Self> {
        let model = Python::attach(|py| -> Result<Py<PyAny>> {
            ignore_system_signals(py)?;
            let model_instance = Self::load_python_model(py, name, device)?;
            Ok(model_instance.into())
        })?;
        Ok(Self {
            inner: Arc::new(Inner {
                model,
                name: name.to_string(),
                batch_size,
            }),
        })
    }
}

impl Embed for OpenClipModel {
    type Input = MultiModalInput;
    type Params = EmbeddingParams;

    fn embed<I>(&self, inputs: &[I], params: &Self::Params) -> Result<Vec<Embedding>>
    where
        I: AsInput<Self::Input>,
    {
        // Partition inputs into text and image groups, tracking original indices
        let mut text_indices = Vec::new();
        let mut text_values = Vec::new();
        let mut image_indices = Vec::new();
        let mut image_values = Vec::new();

        for (i, input) in inputs.iter().enumerate() {
            match input.as_input() {
                MultiModalInput::Text(s) => {
                    text_indices.push(i);
                    text_values.push(s.clone());
                }
                MultiModalInput::Image(arr) => {
                    image_indices.push(i);
                    image_values.push(arr.clone());
                }
            }
        }

        Python::attach(|py| {
            let model = self.inner.model.bind(py);
            let mut result: Vec<Option<Embedding>> = vec![None; inputs.len()];

            // Embed texts
            if !text_values.is_empty() {
                let kwargs = PyDict::new(py);
                kwargs.set_item("batch_size", self.inner.batch_size)?;
                kwargs.set_item("normalize", params.normalize)?;

                let embeddings: Vec<Embedding> = model
                    .call_method("embed_text", (text_values,), Some(&kwargs))?
                    .extract()?;

                for (idx, emb) in text_indices.into_iter().zip(embeddings) {
                    result[idx] = Some(emb);
                }
            }

            // Embed images
            if !image_values.is_empty() {
                let numpy_images: Vec<_> =
                    image_values.iter().map(|img| img.to_pyarray(py)).collect();

                let kwargs = PyDict::new(py);
                kwargs.set_item("batch_size", self.inner.batch_size)?;
                kwargs.set_item("normalize", params.normalize)?;

                let embeddings: Vec<Embedding> = model
                    .call_method("embed_image", (numpy_images,), Some(&kwargs))?
                    .extract()?;

                for (idx, emb) in image_indices.into_iter().zip(embeddings) {
                    result[idx] = Some(emb);
                }
            }

            Ok(result.into_iter().map(|e| e.unwrap()).collect())
        })
    }

    fn num_dimensions(&self) -> usize {
        Python::attach(|py| {
            let model = self.inner.model.bind(py);
            model
                .getattr("dim")
                .expect("Failed to get model dim attribute")
                .extract()
                .expect("Failed to extract model dim as usize")
        })
    }

    fn model_name(&self) -> &str {
        &self.inner.name
    }

    fn provider(&self) -> &str {
        "open-clip"
    }
}

#[cfg(test)]
mod tests {
    use numpy::ndarray::Array3;

    use super::*;

    const TEST_MODEL: &str = "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K";
    const TEST_DIM: usize = 512;

    fn create_dummy_image(height: usize, width: usize, channels: usize) -> Array3<u8> {
        Array3::from_shape_fn((height, width, channels), |(h, w, c)| {
            ((h * width * channels + w * channels + c) % 256) as u8
        })
    }

    #[test]
    #[ignore] // Requires model download and Python environment
    fn test_load_model() {
        let result = OpenClipModel::load(TEST_MODEL, "cpu", 16);
        assert!(result.is_ok(), "Failed to load model: {:?}", result.err());
    }

    #[test]
    #[ignore]
    fn test_num_dimensions() {
        let model = OpenClipModel::load(TEST_MODEL, "cpu", 16).expect("Failed to load model");
        assert_eq!(model.num_dimensions(), TEST_DIM);
    }

    #[test]
    #[ignore]
    fn test_embed_text() {
        let model = OpenClipModel::load(TEST_MODEL, "cpu", 16).expect("Failed to load model");
        let inputs = vec![
            MultiModalInput::Text("a photo of a cat".to_string()),
            MultiModalInput::Text("a photo of a dog".to_string()),
        ];
        let params = EmbeddingParams::default();

        let embeddings = model.embed(&inputs, &params).unwrap();
        assert_eq!(embeddings.len(), 2);
        for emb in &embeddings {
            assert_eq!(emb.len(), TEST_DIM);
            let sum: f32 = emb.iter().sum();
            assert!(sum.abs() > 0.0, "Embedding should not be all zeros");
        }
    }

    #[test]
    #[ignore]
    fn test_embed_image() {
        let model = OpenClipModel::load(TEST_MODEL, "cpu", 16).expect("Failed to load model");
        let img = create_dummy_image(224, 224, 3);
        let inputs = vec![MultiModalInput::Image(img)];
        let params = EmbeddingParams::default();

        let embeddings = model.embed(&inputs, &params).unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), TEST_DIM);
    }

    #[test]
    #[ignore]
    fn test_embed_mixed() {
        let model = OpenClipModel::load(TEST_MODEL, "cpu", 16).expect("Failed to load model");
        let img = create_dummy_image(224, 224, 3);
        let inputs = vec![
            MultiModalInput::Text("a photo of a cat".to_string()),
            MultiModalInput::Image(img),
            MultiModalInput::Text("a photo of a dog".to_string()),
        ];
        let params = EmbeddingParams::default();

        let embeddings = model.embed(&inputs, &params).unwrap();
        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.len(), TEST_DIM);
        }
    }

    #[test]
    #[ignore]
    fn test_embed_empty() {
        let model = OpenClipModel::load(TEST_MODEL, "cpu", 16).expect("Failed to load model");
        let inputs: Vec<MultiModalInput> = vec![];
        let params = EmbeddingParams::default();

        let embeddings = model.embed(&inputs, &params).unwrap();
        assert_eq!(embeddings.len(), 0);
    }

    #[test]
    #[ignore]
    fn test_embed_normalized() {
        let model = OpenClipModel::load(TEST_MODEL, "cpu", 16).expect("Failed to load model");
        let inputs = vec![MultiModalInput::Text("hello world".to_string())];
        let params = EmbeddingParams {
            num_dimensions: None,
            normalize: true,
        };

        let embeddings = model.embed(&inputs, &params).unwrap();
        let magnitude: f32 = embeddings[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (magnitude - 1.0).abs() < 0.01,
            "Normalized embedding should have magnitude ~1.0, got {}",
            magnitude
        );
    }

    #[test]
    #[ignore]
    fn test_text_image_shared_space() {
        let model = OpenClipModel::load(TEST_MODEL, "cpu", 16).expect("Failed to load model");
        let img = create_dummy_image(224, 224, 3);
        let params = EmbeddingParams::default();

        let text_emb = model
            .embed(&[MultiModalInput::Text("a photo".to_string())], &params)
            .unwrap();
        let img_emb = model
            .embed(&[MultiModalInput::Image(img)], &params)
            .unwrap();

        // Both should have the same dimensionality (shared space)
        assert_eq!(text_emb[0].len(), img_emb[0].len());
        assert_eq!(text_emb[0].len(), TEST_DIM);
    }
}
