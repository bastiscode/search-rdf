use std::sync::Arc;

use anyhow::Result;
use numpy::ToPyArray;
use numpy::ndarray::Array3;
use pyo3::{prelude::*, types::PyDict};

use crate::model::AsInput;
use crate::{
    data::embedding::Embedding,
    model::{Embed, EmbeddingParams},
};

#[derive(Debug)]
struct Inner {
    model: Py<PyAny>,
    name: String,
    batch_size: usize,
}

#[derive(Debug, Clone)]
pub struct HuggingFaceImageModel {
    inner: Arc<Inner>,
}

impl HuggingFaceImageModel {
    fn load_python_model<'py>(
        py: Python<'py>,
        name: &str,
        device: &str,
    ) -> Result<Bound<'py, PyAny>> {
        let module = py.import("search_rdf.model")?;
        let model_class = module.getattr("ImageEmbeddingModel")?;
        let model_instance = model_class.call1((name, device))?;
        Ok(model_instance)
    }

    pub fn load(name: &str, device: &str, batch_size: usize) -> Result<Self> {
        let model = Python::attach(|py| -> Result<Py<PyAny>> {
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

impl Embed for HuggingFaceImageModel {
    type Input = Array3<u8>;
    type Params = EmbeddingParams;

    fn embed<I>(&self, inputs: &[I], params: &Self::Params) -> Result<Vec<Embedding>>
    where
        I: AsInput<Self::Input>,
    {
        Python::attach(|py| {
            let model = self.inner.model.bind(py);

            // Convert ndarray images to numpy arrays
            let numpy_images: Vec<_> = inputs
                .iter()
                .map(|input| input.as_input().to_pyarray(py))
                .collect();

            let kwargs = PyDict::new(py);
            kwargs.set_item("batch_size", self.inner.batch_size)?;
            kwargs.set_item("normalize", params.normalize)?;

            let embeddings = model.call_method("embed", (numpy_images,), Some(&kwargs))?;
            Ok(embeddings.extract()?)
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

    fn model_type(&self) -> &str {
        "huggingface-image"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_MODEL: &str = "facebook/dinov2-small";
    const TEST_DIM: usize = 384;

    fn create_dummy_image(height: usize, width: usize, channels: usize) -> Array3<u8> {
        Array3::from_shape_fn((height, width, channels), |(h, w, c)| {
            ((h * width * channels + w * channels + c) % 256) as u8
        })
    }

    #[test]
    #[ignore] // Requires model download and Python environment
    fn test_load_model() {
        let result = HuggingFaceImageModel::load(TEST_MODEL, "cpu", 16);
        assert!(result.is_ok(), "Failed to load model: {:?}", result.err());

        let model = result.unwrap();
        assert_eq!(model.inner.name, TEST_MODEL);
    }

    #[test]
    #[ignore] // Requires model download and Python environment
    fn test_num_dimensions() {
        let model =
            HuggingFaceImageModel::load(TEST_MODEL, "cpu", 16).expect("Failed to load model");

        let dim = model.num_dimensions();
        assert_eq!(
            dim, TEST_DIM,
            "{TEST_MODEL} should have {TEST_DIM} dimensions"
        );
    }

    #[test]
    #[ignore] // Requires model download and Python environment
    fn test_embed_single_image() {
        let model =
            HuggingFaceImageModel::load(TEST_MODEL, "cpu", 16).expect("Failed to load model");

        let image = create_dummy_image(224, 224, 3);
        let inputs = [&image];
        let params = EmbeddingParams::default();

        let result = model.embed(&inputs, &params);
        assert!(result.is_ok(), "Failed to embed: {:?}", result.err());

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), TEST_DIM);

        // Check that embeddings are not all zeros
        let sum: f32 = embeddings[0].iter().sum();
        assert!(sum.abs() > 0.0, "Embeddings should not be all zeros");
    }

    #[test]
    #[ignore] // Requires model download and Python environment
    fn test_embed_multiple_images() {
        let model =
            HuggingFaceImageModel::load(TEST_MODEL, "cpu", 16).expect("Failed to load model");

        let img1 = create_dummy_image(224, 224, 3);
        let img2 = create_dummy_image(224, 224, 3);
        let img3 = create_dummy_image(224, 224, 3);
        let inputs = [&img1, &img2, &img3];
        let params = EmbeddingParams::default();

        let result = model.embed(&inputs, &params);
        assert!(result.is_ok(), "Failed to embed: {:?}", result.err());

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 3);

        for (i, embedding) in embeddings.iter().enumerate() {
            assert_eq!(
                embedding.len(),
                TEST_DIM,
                "Embedding {} has wrong dimension",
                i
            );
            // Check that embeddings are not all zeros
            let sum: f32 = embedding.iter().sum();
            assert!(sum.abs() > 0.0, "Embedding {} should not be all zeros", i);
        }
    }

    #[test]
    #[ignore] // Requires model download and Python environment
    fn test_embed_empty_input() {
        let model =
            HuggingFaceImageModel::load(TEST_MODEL, "cpu", 16).expect("Failed to load model");

        let inputs: [&Array3<u8>; 0] = [];
        let params = EmbeddingParams::default();

        let result = model.embed(&inputs, &params);
        assert!(result.is_ok());

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 0);
    }

    #[test]
    #[ignore] // Requires model download and Python environment
    fn test_embed_with_custom_batch_size() {
        let model =
            HuggingFaceImageModel::load(TEST_MODEL, "cpu", 2).expect("Failed to load model");

        let img1 = create_dummy_image(224, 224, 3);
        let img2 = create_dummy_image(224, 224, 3);
        let img3 = create_dummy_image(224, 224, 3);
        let img4 = create_dummy_image(224, 224, 3);
        let inputs = [&img1, &img2, &img3, &img4];

        let params = EmbeddingParams {
            num_dimensions: None,
            normalize: true,
        };

        let result = model.embed(&inputs, &params);
        assert!(
            result.is_ok(),
            "Failed to embed with batch_size=2: {:?}",
            result.err()
        );

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 4);
    }

    #[test]
    #[ignore] // Requires model download and Python environment
    fn test_embed_normalized_vs_unnormalized() {
        let model =
            HuggingFaceImageModel::load(TEST_MODEL, "cpu", 16).expect("Failed to load model");

        let image = create_dummy_image(224, 224, 3);
        let inputs = [&image];

        // Get normalized embeddings
        let params_normalized = EmbeddingParams {
            num_dimensions: None,
            normalize: true,
        };
        let embeddings_normalized = model.embed(&inputs, &params_normalized).unwrap();

        // Get unnormalized embeddings
        let params_unnormalized = EmbeddingParams {
            num_dimensions: None,
            normalize: false,
        };
        let embeddings_unnormalized = model.embed(&inputs, &params_unnormalized).unwrap();

        assert_eq!(embeddings_normalized.len(), 1);
        assert_eq!(embeddings_unnormalized.len(), 1);

        let vec_normalized = &embeddings_normalized[0];

        // Normalized vectors should have magnitude close to 1
        let magnitude_normalized: f32 = vec_normalized.iter().map(|&x| x * x).sum::<f32>().sqrt();

        assert!(
            (magnitude_normalized - 1.0).abs() < 0.01,
            "Normalized embedding should have magnitude ~1.0, got {}",
            magnitude_normalized
        );
    }

    #[test]
    #[ignore] // Requires model download and Python environment
    fn test_embed_different_sizes() {
        let model =
            HuggingFaceImageModel::load(TEST_MODEL, "cpu", 16).expect("Failed to load model");

        // Test with different image sizes (model should handle resizing)
        let img1 = create_dummy_image(224, 224, 3);
        let img2 = create_dummy_image(256, 256, 3);
        let img3 = create_dummy_image(128, 128, 3);
        let inputs = [&img1, &img2, &img3];
        let params = EmbeddingParams::default();

        let result = model.embed(&inputs, &params);
        assert!(result.is_ok(), "Failed to embed: {:?}", result.err());

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 3);

        for embedding in embeddings.iter() {
            assert_eq!(embedding.len(), TEST_DIM);
        }
    }

    #[test]
    #[ignore] // Requires model download, Python environment, and CUDA GPU
    fn test_load_model_cuda() {
        let result = HuggingFaceImageModel::load(TEST_MODEL, "cuda", 16);
        assert!(
            result.is_ok(),
            "Failed to load model on CUDA: {:?}",
            result.err()
        );

        let model = result.unwrap();
        assert_eq!(model.inner.name, TEST_MODEL);
    }

    #[test]
    #[ignore] // Requires model download, Python environment, and CUDA GPU
    fn test_embed_cuda() {
        let model = HuggingFaceImageModel::load(TEST_MODEL, "cuda", 16)
            .expect("Failed to load model on CUDA");

        let img1 = create_dummy_image(224, 224, 3);
        let img2 = create_dummy_image(224, 224, 3);
        let img3 = create_dummy_image(224, 224, 3);
        let inputs = [&img1, &img2, &img3];
        let params = EmbeddingParams::default();

        let result = model.embed(&inputs, &params);
        assert!(
            result.is_ok(),
            "Failed to embed on CUDA: {:?}",
            result.err()
        );

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 3);

        for (i, embedding) in embeddings.iter().enumerate() {
            assert_eq!(
                embedding.len(),
                TEST_DIM,
                "CUDA embedding {} has wrong dimension",
                i
            );
            let sum: f32 = embedding.iter().sum();
            assert!(
                sum.abs() > 0.0,
                "CUDA embedding {} should not be all zeros",
                i
            );
        }
    }

    #[test]
    #[ignore] // Requires model download, Python environment, and CUDA GPU
    fn test_cuda_cpu_consistency() {
        let model_cpu = HuggingFaceImageModel::load(TEST_MODEL, "cpu", 16)
            .expect("Failed to load model on CPU");
        let model_cuda = HuggingFaceImageModel::load(TEST_MODEL, "cuda", 16)
            .expect("Failed to load model on CUDA");

        let image = create_dummy_image(224, 224, 3);
        let inputs = [&image];
        let params = EmbeddingParams::default();

        let embeddings_cpu = model_cpu.embed(&inputs, &params).unwrap();
        let embeddings_cuda = model_cuda
            .embed(&inputs, &EmbeddingParams::default())
            .unwrap();

        assert_eq!(embeddings_cpu.len(), embeddings_cuda.len());

        let cpu_vec = &embeddings_cpu[0];
        let cuda_vec = &embeddings_cuda[0];

        assert_eq!(cpu_vec.len(), cuda_vec.len());
        assert_eq!(cpu_vec.len(), TEST_DIM);

        // Check that vectors are similar (within tolerance for FP differences)
        let max_diff = cpu_vec
            .iter()
            .zip(cuda_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |max, diff| max.max(diff));

        assert!(
            max_diff < 1e-4,
            "CPU and CUDA embeddings differ too much. Max difference: {}",
            max_diff
        );
    }

    #[test]
    #[ignore] // Requires model download, Python environment, and CUDA GPU
    fn test_cuda_large_batch() {
        let model = HuggingFaceImageModel::load(TEST_MODEL, "cuda", 32)
            .expect("Failed to load model on CUDA");

        // Create a larger batch to test GPU efficiency
        let images: Vec<Array3<u8>> = (0..100).map(|_| create_dummy_image(224, 224, 3)).collect();
        let inputs: Vec<&Array3<u8>> = images.iter().collect();

        let params = EmbeddingParams {
            num_dimensions: None,
            normalize: true,
        };

        let result = model.embed(&inputs, &params);
        assert!(
            result.is_ok(),
            "Failed to embed large batch on CUDA: {:?}",
            result.err()
        );

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 100);

        for embedding in embeddings.iter() {
            assert_eq!(embedding.len(), TEST_DIM);
        }
    }
}
