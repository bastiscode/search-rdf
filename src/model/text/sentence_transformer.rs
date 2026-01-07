use std::sync::Arc;

use anyhow::Result;
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};

use crate::{
    data::embedding::Embedding,
    model::{AsInput, Embed, EmbeddingParams},
};

#[derive(Debug)]
struct Inner {
    model: Py<PyAny>,
    name: String,
    batch_size: usize,
}

#[derive(Debug, Clone)]
pub struct SentenceTransformer {
    inner: Arc<Inner>,
}

impl SentenceTransformer {
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

impl Embed for SentenceTransformer {
    type Input = str;
    type Params = EmbeddingParams;

    fn embed<I>(&self, inputs: &[I], params: &Self::Params) -> Result<Vec<Embedding>>
    where
        I: AsInput<Self::Input>,
    {
        Python::attach(|py| {
            let model = self.inner.model.bind(py);
            let inputs = PyList::new(py, inputs.iter().map(|s| s.as_input()))?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("batch_size", self.inner.batch_size)?;
            kwargs.set_item("embedding_dim", params.num_dimensions)?;
            kwargs.set_item("normalize", params.normalize)?;

            let embeddings = model.call_method("embed", (inputs,), Some(&kwargs))?;
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
        "sentence-transformer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_MODEL: &str = "mixedbread-ai/mxbai-embed-xsmall-v1";
    const TEST_DIM: usize = 384;

    #[test]
    fn test_embedding_params_default() {
        let params = EmbeddingParams::default();
        assert_eq!(params.num_dimensions, None);
        assert!(params.normalize);
    }

    #[test]
    #[ignore] // Requires model download and Python environment
    fn test_load_model() {
        let result = SentenceTransformer::load(TEST_MODEL, "cpu", 16);
        assert!(result.is_ok(), "Failed to load model: {:?}", result.err());

        let model = result.unwrap();
        // Call methods directly on the struct
        assert_eq!(model.inner.name, TEST_MODEL);
    }

    #[test]
    #[ignore] // Requires model download and Python environment
    fn test_num_dimensions() {
        let model = SentenceTransformer::load(TEST_MODEL, "cpu", 16).expect("Failed to load model");

        // Need to specify the type parameter for EmbeddingModel trait
        let dim = model.num_dimensions();
        assert_eq!(dim, TEST_DIM, "{TEST_MODEL} should have 384 dimensions");
    }

    #[test]
    #[ignore] // Requires model download and Python environment
    fn test_embed_single_text() {
        let model = SentenceTransformer::load(TEST_MODEL, "cpu", 16).expect("Failed to load model");

        let inputs = vec!["This is a test sentence."];
        let params = EmbeddingParams::default();

        let result = model.embed(&inputs, &params);
        assert!(result.is_ok(), "Failed to embed: {:?}", result.err());

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 1);

        // Embedding is now just Vec<f32>
        assert_eq!(embeddings[0].len(), TEST_DIM);
        // Check that embeddings are not all zeros
        let sum: f32 = embeddings[0].iter().sum();
        assert!(sum.abs() > 0.0, "Embeddings should not be all zeros");
    }

    #[test]
    #[ignore] // Requires model download and Python environment
    fn test_embed_multiple_texts() {
        let model = SentenceTransformer::load(TEST_MODEL, "cpu", 16).expect("Failed to load model");

        let inputs = vec![
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Rust is a systems programming language.",
        ];
        let params = EmbeddingParams::default();

        let result = model.embed(&inputs, &params);
        assert!(result.is_ok(), "Failed to embed: {:?}", result.err());

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 3);

        // Embedding is now just Vec<f32>
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

        // Verify that different texts produce different embeddings
        assert_ne!(embeddings[0], embeddings[1]);
        assert_ne!(embeddings[1], embeddings[2]);
        assert_ne!(embeddings[0], embeddings[2]);
    }

    #[test]
    #[ignore] // Requires model download and Python environment
    fn test_embed_with_string_types() {
        let model = SentenceTransformer::load(TEST_MODEL, "cpu", 16).expect("Failed to load model");

        let params = EmbeddingParams::default();

        // Test with &str
        let str_inputs = vec!["hello world", "rust programming"];
        let result1 = model.embed(&str_inputs, &params);
        assert!(result1.is_ok());

        // Test with String
        let string_inputs = vec!["hello world".to_string(), "rust programming".to_string()];
        let result2 = model.embed(&string_inputs, &EmbeddingParams::default());
        assert!(result2.is_ok());

        // Extract and compare the embeddings (now just Vec<f32>)
        let embeddings1 = result1.unwrap();
        let embeddings2 = result2.unwrap();

        assert_eq!(embeddings1.len(), embeddings2.len());
        for i in 0..embeddings1.len() {
            assert_eq!(
                embeddings1[i], embeddings2[i],
                "Embeddings {} differ between &str and String",
                i
            );
        }
    }

    #[test]
    #[ignore] // Requires model download and Python environment
    fn test_embed_empty_input() {
        let model = SentenceTransformer::load(TEST_MODEL, "cpu", 16).expect("Failed to load model");

        let inputs: Vec<&str> = vec![];
        let params = EmbeddingParams::default();

        let result = model.embed(&inputs, &params);
        assert!(result.is_ok());

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 0);
    }

    #[test]
    #[ignore] // Requires model download and Python environment
    fn test_embed_with_custom_batch_size() {
        let model = SentenceTransformer::load(TEST_MODEL, "cpu", 2).expect("Failed to load model");

        let inputs = vec![
            "First sentence.",
            "Second sentence.",
            "Third sentence.",
            "Fourth sentence.",
        ];

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
        let model = SentenceTransformer::load(TEST_MODEL, "cpu", 16).expect("Failed to load model");

        let inputs = vec!["Normalization test."];

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

        // Embeddings are now just Vec<f32>, verify normalization
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
    #[ignore] // Requires model download, Python environment, and CUDA GPU
    fn test_load_model_cuda() {
        let result = SentenceTransformer::load(TEST_MODEL, "cuda", 16);
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
        let model = SentenceTransformer::load(TEST_MODEL, "cuda", 16)
            .expect("Failed to load model on CUDA");

        let inputs = vec![
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Rust is a systems programming language.",
        ];
        let params = EmbeddingParams::default();

        let result = model.embed(&inputs, &params);
        assert!(
            result.is_ok(),
            "Failed to embed on CUDA: {:?}",
            result.err()
        );

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 3);

        // Verify embeddings are valid (Embedding is now just Vec<f32>)
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
        // Load same model on both CPU and CUDA
        let model_cpu =
            SentenceTransformer::load(TEST_MODEL, "cpu", 16).expect("Failed to load model on CPU");
        let model_cuda = SentenceTransformer::load(TEST_MODEL, "cuda", 16)
            .expect("Failed to load model on CUDA");

        let inputs = vec!["Test sentence for CPU-CUDA consistency check."];
        let params = EmbeddingParams::default();

        let embeddings_cpu = model_cpu.embed(&inputs, &params).unwrap();
        let embeddings_cuda = model_cuda
            .embed(&inputs, &EmbeddingParams::default())
            .unwrap();

        assert_eq!(embeddings_cpu.len(), embeddings_cuda.len());

        // Compare embeddings - they should be very close but not necessarily identical
        // due to floating point precision differences between CPU and GPU
        // Embeddings are now just Vec<f32>
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
        let model = SentenceTransformer::load(TEST_MODEL, "cuda", 32)
            .expect("Failed to load model on CUDA");

        // Create a larger batch to test GPU efficiency
        let inputs: Vec<String> = (0..100)
            .map(|i| format!("This is test sentence number {}.", i))
            .collect();

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

        // Verify all embeddings are valid (Embedding is now just Vec<f32>)
        for embedding in embeddings.iter() {
            assert_eq!(embedding.len(), TEST_DIM);
        }
    }
}
