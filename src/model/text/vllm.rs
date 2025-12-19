use crate::model::EmbeddingModel;
use anyhow::{Result, anyhow};
use ureq::Agent;

pub struct VLLMEmbeddingModel {
    pub endpoint: String,
    pub model_name: String,
    num_dimensions: usize,
    agent: Agent,
}

fn embedding_from_value(value: &serde_json::Value) -> Result<Vec<f32>> {
    let embedding_array = value
        .get("embedding")
        .ok_or_else(|| anyhow!("Missing 'embedding' field in response"))?
        .as_array()
        .ok_or_else(|| anyhow!("'embedding' field is not an array"))?;

    embedding_array
        .iter()
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| anyhow!("Embedding value is not a number"))
                .map(|num| num as f32)
        })
        .collect()
}

fn embed(agent: &Agent, endpoint: &str, inputs: &[impl AsRef<str>]) -> Result<Vec<Vec<f32>>> {
    // Implement the logic to call the VLLM endpoint and retrieve embeddings
    // This is a placeholder implementation
    let response = agent
        .post(endpoint)
        .header("Content-Type", "application/json")
        .send_json(serde_json::json!({
            "input": inputs.iter().map(|s| s.as_ref()).collect::<Vec<_>>(),
        }))?;

    let value: serde_json::Value = response.into_body().read_json()?;
    value
        .get("data")
        .ok_or_else(|| anyhow!("Missing 'data' field in response"))?
        .as_array()
        .ok_or_else(|| anyhow!("'data' field is not an array"))?
        .iter()
        .map(embedding_from_value)
        .collect()
}

impl VLLMEmbeddingModel {
    pub fn new(endpoint: String, model_name: String) -> Result<Self> {
        let agent = Agent::new_with_defaults();
        let endpoint = format!("{}/v1/embeddings", endpoint);

        let test_inputs = vec!["test"];
        let embeddings = embed(&agent, &endpoint, &test_inputs)?;
        if embeddings.len() != test_inputs.len() {
            return Err(anyhow!(
                "Failed to validate VLLM embedding model: unexpected number of embeddings returned"
            ));
        }
        let num_dimensions = embeddings[0].len();

        Ok(Self {
            endpoint,
            model_name,
            agent,
            num_dimensions,
        })
    }
}

impl EmbeddingModel for VLLMEmbeddingModel {
    type Input = str;
    type Params = ();

    fn embed<I>(&self, inputs: &[I], _params: Self::Params) -> Result<Vec<Vec<f32>>>
    where
        I: AsRef<Self::Input>,
    {
        embed(&self.agent, &self.endpoint, inputs)
    }

    fn num_dimensions(&self) -> usize {
        self.num_dimensions
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn model_type(&self) -> &str {
        "vLLM"
    }
}
