use std::sync::Arc;

use crate::model::{Embed, EmbeddingParams};
use anyhow::{Result, anyhow};
use log::info;
use ureq::Agent;

#[derive(Debug)]
struct Inner {
    endpoint: String,
    model_name: String,
    num_dimensions: usize,
    max_model_len: usize,
    agent: Agent,
}

#[derive(Debug, Clone)]
pub struct VLLM {
    inner: Arc<Inner>,
}

fn embedding_from_value(value: &serde_json::Value, params: &EmbeddingParams) -> Result<Vec<f32>> {
    let embedding_array = value
        .get("embedding")
        .ok_or_else(|| anyhow!("Missing 'embedding' field in response"))?
        .as_array()
        .ok_or_else(|| anyhow!("'embedding' field is not an array"))?;

    let embedding = embedding_array
        .iter()
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| anyhow!("Embedding value is not a number"))
                .map(|num| num as f32)
        })
        .collect::<Result<_>>()?;

    Ok(params.apply(embedding))
}

fn fetch_max_model_len(agent: &Agent, model_name: &str, endpoint: &str) -> Result<usize> {
    let response = agent
        .get(format!("{endpoint}/v1/models"))
        .header("Accept", "application/json")
        .call()?;

    let value: serde_json::Value = response.into_body().read_json()?;
    let max_len = value
        .get("data")
        .ok_or_else(|| anyhow!("Missing 'data' field in response"))?
        .as_array()
        .ok_or_else(|| anyhow!("'data' field is not an array"))?
        .iter()
        .find_map(|value| {
            let id = value.get("id")?.as_str()?;
            if id == model_name { Some(value) } else { None }
        })
        .ok_or_else(|| anyhow!("Model '{}' not found in server", model_name))?
        .get("max_model_len")
        .ok_or_else(|| anyhow!("Missing 'max_model_len' field in response"))?
        .as_u64()
        .ok_or_else(|| anyhow!("'max_model_len' field cannot be converted to u64"))?
        as usize;

    Ok(max_len)
}

fn embed(
    agent: &Agent,
    endpoint: &str,
    inputs: &[impl AsRef<str>],
    max_model_len: usize,
    params: &EmbeddingParams,
) -> Result<Vec<Vec<f32>>> {
    if inputs.is_empty() {
        return Ok(vec![]);
    }
    // Implement the logic to call the VLLM endpoint and retrieve embeddings
    // This is a placeholder implementation
    let response = agent
        .post(format!("{endpoint}/v1/embeddings"))
        .header("Content-Type", "application/json")
        .send_json(serde_json::json!({
            "input": inputs.iter().map(|s| s.as_ref()).collect::<Vec<_>>(),
            "truncate_prompt_tokens": max_model_len,
        }))?;

    let value: serde_json::Value = response.into_body().read_json()?;
    value
        .get("data")
        .ok_or_else(|| anyhow!("Missing 'data' field in response"))?
        .as_array()
        .ok_or_else(|| anyhow!("'data' field is not an array"))?
        .iter()
        .map(|value| embedding_from_value(value, params))
        .collect()
}

impl VLLM {
    pub fn new(endpoint: &str, model_name: &str) -> Result<Self> {
        let agent = Agent::new_with_defaults();
        info!(
            "[vLLM] Initializing model '{}' with endpoint '{}'",
            model_name, endpoint
        );

        let max_model_len = fetch_max_model_len(&agent, model_name, endpoint)?;
        info!(
            "[vLLM] Got max_model_len={} for model '{}'",
            max_model_len, model_name
        );

        let test_inputs = vec!["test"];
        let embeddings = embed(
            &agent,
            endpoint,
            &test_inputs,
            max_model_len,
            &EmbeddingParams::default(),
        )?;
        if embeddings.len() != test_inputs.len() {
            return Err(anyhow!(
                "Failed to validate VLLM embedding model: unexpected number of embeddings returned"
            ));
        }
        let num_dimensions = embeddings[0].len();
        info!(
            "[vLLM] Got num_dimensions={} for model '{}'",
            num_dimensions, model_name
        );

        Ok(Self {
            inner: Arc::new(Inner {
                endpoint: endpoint.to_string(),
                model_name: model_name.to_string(),
                agent,
                num_dimensions,
                max_model_len,
            }),
        })
    }

    pub fn max_model_len(&self) -> usize {
        self.inner.max_model_len
    }
}

impl Embed for VLLM {
    type Input = str;
    type Params = EmbeddingParams;

    fn embed<I>(&self, inputs: &[I], params: &Self::Params) -> Result<Vec<Vec<f32>>>
    where
        I: AsRef<Self::Input>,
    {
        embed(
            &self.inner.agent,
            &self.inner.endpoint,
            inputs,
            self.inner.max_model_len,
            params,
        )
    }

    fn num_dimensions(&self) -> usize {
        self.inner.num_dimensions
    }

    fn model_name(&self) -> &str {
        &self.inner.model_name
    }

    fn model_type(&self) -> &str {
        "vllm"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_ENDPOINT: &str = "http://localhost:8000";
    const TEST_MODEL: &str = "test-embedding-model";

    #[test]
    #[ignore] // Requires vLLM server running at localhost:8000
    fn test_load_model() {
        let model = VLLM::new(TEST_ENDPOINT, TEST_MODEL).expect("Failed to load model");

        assert_eq!(model.model_name(), TEST_MODEL);
        assert_eq!(model.model_type(), "vllm");
        assert!(
            model.num_dimensions() > 0,
            "Model should have positive dimensions"
        );
    }

    #[test]
    #[ignore] // Requires vLLM server running at localhost:8000
    fn test_num_dimensions() {
        let model = VLLM::new(TEST_ENDPOINT, TEST_MODEL).expect("Failed to load model");

        let dim = model.num_dimensions();
        assert!(dim > 0, "Model dimensions should be positive");

        // Verify consistency - num_dimensions should match actual embedding dimension
        let inputs = vec!["test"];
        let embeddings = model
            .embed(&inputs, &EmbeddingParams::default())
            .expect("Failed to embed");
        assert_eq!(
            embeddings[0].len(),
            dim,
            "Embedding dimension should match num_dimensions"
        );
    }

    #[test]
    #[ignore] // Requires vLLM server running at localhost:8000
    fn test_embed_single_text() {
        let model = VLLM::new(TEST_ENDPOINT, TEST_MODEL).expect("Failed to load model");

        let inputs = vec!["This is a test sentence."];
        let result = model.embed(&inputs, &EmbeddingParams::default());
        assert!(result.is_ok(), "Failed to embed: {:?}", result.err());

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), model.num_dimensions());

        // Check that embeddings are not all zeros
        let sum: f32 = embeddings[0].iter().sum();
        assert!(sum.abs() > 0.0, "Embeddings should not be all zeros");
    }

    #[test]
    #[ignore] // Requires vLLM server running at localhost:8000
    fn test_embed_multiple_texts() {
        let model = VLLM::new(TEST_ENDPOINT, TEST_MODEL).expect("Failed to load model");

        let inputs = vec![
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Rust is a systems programming language.",
        ];
        let result = model.embed(&inputs, &EmbeddingParams::default());
        assert!(result.is_ok(), "Failed to embed: {:?}", result.err());

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 3);

        for (i, embedding) in embeddings.iter().enumerate() {
            assert_eq!(
                embedding.len(),
                model.num_dimensions(),
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
    #[ignore] // Requires vLLM server running at localhost:8000
    fn test_embed_with_string_types() {
        let model = VLLM::new(TEST_ENDPOINT, TEST_MODEL).expect("Failed to load model");

        // Test with &str
        let str_inputs = vec!["hello world", "rust programming"];
        let result1 = model.embed(&str_inputs, &EmbeddingParams::default());
        assert!(result1.is_ok());

        // Test with String
        let string_inputs = vec!["hello world".to_string(), "rust programming".to_string()];
        let result2 = model.embed(&string_inputs, &EmbeddingParams::default());
        assert!(result2.is_ok());

        // Extract and compare the embeddings
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
    #[ignore] // Requires vLLM server running at localhost:8000
    fn test_embed_empty_input() {
        let model = VLLM::new(TEST_ENDPOINT, TEST_MODEL).expect("Failed to load model");

        let inputs: Vec<&str> = vec![];
        let result = model.embed(&inputs, &EmbeddingParams::default());
        assert!(result.is_ok());

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 0);
    }

    #[test]
    #[ignore] // Requires vLLM server running at localhost:8000
    fn test_embed_large_batch() {
        let model = VLLM::new(TEST_ENDPOINT, TEST_MODEL).expect("Failed to load model");

        // Create a larger batch to test API efficiency
        let inputs: Vec<String> = (0..100)
            .map(|i| format!("This is test sentence number {}.", i))
            .collect();

        let result = model.embed(&inputs, &EmbeddingParams::default());
        assert!(
            result.is_ok(),
            "Failed to embed large batch: {:?}",
            result.err()
        );

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 100);

        // Verify all embeddings are valid
        for embedding in embeddings.iter() {
            assert_eq!(embedding.len(), model.num_dimensions());
        }
    }

    #[test]
    #[ignore] // Requires vLLM server running at localhost:8000
    fn test_model_name_and_type() {
        let model = VLLM::new(TEST_ENDPOINT, TEST_MODEL).expect("Failed to load model");

        assert_eq!(model.model_name(), TEST_MODEL);
        assert_eq!(model.model_type(), "vllm");
    }

    #[test]
    #[ignore] // Requires vLLM server running at localhost:8000
    fn test_endpoint_formatting() {
        // Test that endpoint is properly formatted with /v1/embeddings
        let model = VLLM::new("http://localhost:8000", TEST_MODEL).expect("Failed to load model");

        // The endpoint should be formatted with /v1/embeddings suffix
        assert_eq!(model.inner.endpoint, "http://localhost:8000/v1/embeddings");
    }

    #[test]
    #[ignore] // Requires vLLM server running at localhost:8000
    fn test_invalid_endpoint() {
        // Test with an invalid endpoint that doesn't exist
        let result = VLLM::new("http://localhost:9999", TEST_MODEL);

        // Should fail because the endpoint doesn't exist
        assert!(result.is_err(), "Should fail with invalid endpoint");
    }

    #[test]
    #[ignore] // Requires vLLM server running at localhost:8000
    fn test_consistency() {
        // Test that the same input produces the same output
        let model = VLLM::new(TEST_ENDPOINT, TEST_MODEL).expect("Failed to load model");

        let inputs = vec!["Consistency test sentence."];

        let embeddings1 = model
            .embed(&inputs, &EmbeddingParams::default())
            .expect("First embed failed");
        let embeddings2 = model
            .embed(&inputs, &EmbeddingParams::default())
            .expect("Second embed failed");

        assert_eq!(embeddings1.len(), embeddings2.len());
        assert_eq!(embeddings1[0].len(), embeddings2[0].len());

        // Embeddings should be identical (or very close due to floating point)
        let max_diff = embeddings1[0]
            .iter()
            .zip(embeddings2[0].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |max, diff| max.max(diff));

        assert!(
            max_diff < 1e-6,
            "Embeddings should be consistent. Max difference: {}",
            max_diff
        );
    }
}
