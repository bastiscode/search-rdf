use anyhow::{Context, Result};
use search_rdf::data::text::item::sparql::SPARQLResultFormat;
use search_rdf::index::EmbeddingIndexParams;
use search_rdf::model::EmbeddingParams;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize, Serialize)]
pub struct Config {
    #[serde(default)]
    pub datasets: Option<Vec<DataConfig>>,
    #[serde(default)]
    pub models: Option<Vec<ModelConfig>>,
    #[serde(default)]
    pub embeddings: Option<Vec<EmbeddingConfig>>,
    #[serde(default)]
    pub indices: Option<Vec<IndexConfig>>,
    #[serde(default)]
    pub server: Option<ServerConfig>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct DataConfig {
    pub name: String,
    pub output: PathBuf,
    #[serde(flatten)]
    pub data_type: DataType,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DataType {
    Text(TextSource),
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TextSource {
    #[serde(rename = "sparql-query")]
    SparqlQuery {
        endpoint: String,
        query: String,
        format: SPARQLResultFormat,
        headers: Option<HashMap<String, String>>,
    },
    Sparql {
        path: PathBuf,
        format: SPARQLResultFormat,
    },
    Jsonl {
        path: PathBuf,
    },
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ModelConfig {
    pub name: String,
    #[serde(flatten)]
    pub model_type: ModelType,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelType {
    Vllm {
        endpoint: String,
        model_name: String,
    },
    #[serde(rename = "sentence-transformer")]
    SentenceTransformer {
        model_name: String,
        #[serde(default = "default_device")]
        device: String,
        #[serde(default = "default_batch_size")]
        batch_size: usize,
        #[serde(default = "default_show_progress")]
        show_progress: bool,
    },
}

fn default_device() -> String {
    "cpu".to_string()
}

fn default_show_progress() -> bool {
    false
}

fn default_batch_size() -> usize {
    16
}

#[derive(Debug, Deserialize, Serialize)]
pub struct EmbeddingConfig {
    pub name: String,
    pub output: PathBuf,
    pub model: String,
    #[serde(flatten)]
    pub dataset: EmbeddingDatasetConfig,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EmbeddingDatasetConfig {
    Text {
        dataset: PathBuf,
        params: EmbeddingParams,
    },
}

#[derive(Debug, Deserialize, Serialize)]
pub struct IndexConfig {
    pub name: String,
    pub output: PathBuf,
    #[serde(flatten)]
    pub index_type: IndexType,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IndexType {
    Keyword {
        text_data: PathBuf,
    },
    #[serde(rename = "text-embedding")]
    TextEmbedding {
        text_data: PathBuf,
        embedding_data: PathBuf,
        params: EmbeddingIndexParams,
    },
    Embedding {
        embedding_data: PathBuf,
        params: EmbeddingIndexParams,
    },
}

#[derive(Debug, Default, Deserialize, Serialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    // References the names of the indices specified in the indexes section
    pub indices: Vec<String>,
    #[serde(default)]
    pub models: Vec<ModelConfig>,
    #[serde(default)]
    pub cors: bool,
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}

fn default_port() -> u16 {
    8080
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref())
            .context(format!("Failed to read config file: {:?}", path.as_ref()))?;

        serde_yaml::from_str(&content).context("Failed to parse config file")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_config() {
        let yaml = r#"
data:
  text: []
embeddings:
  models: []
  datasets: []
indices: []
server:
  host: "0.0.0.0"
  port: 8080
"#;
        let config: Config = serde_yaml::from_str(yaml).expect("Failed to parse config");
        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port, 8080);
    }

    #[test]
    fn test_parse_full_config() {
        let yaml = r#"
data:
  text:
    - name: wikidata_items
      source:
        type: sparql
        endpoint: https://query.wikidata.org/sparql
        query_file: queries/get_items.sparql
        format: json
      output: data/text/wikidata

embeddings:
  models:
    - name: primary_model
      type: vllm
      endpoint: http://localhost:8000
      model_name: mixedbread-ai/mxbai-embed-large-v1

  datasets:
    - name: wikidata_embeddings
      text_data: data/text/wikidata
      model: primary_model
      output: data/embeddings/wikidata
      params:
        batch_size: 32
        normalize: true
        show_progress: true

indices:
  - name: wikidata_keyword
    type: keyword
    text_data: data/text/wikidata
    output: indices/wikidata/keyword

  - name: wikidata_semantic
    type: text_embedding
    text_data: data/text/wikidata
    embedding_data: data/embeddings/wikidata
    output: indices/wikidata/semantic
    metric: cosine_normalized
    precision: float32

server:
  host: 0.0.0.0
  port: 8080
  cors: true
  indices:
    - name: wikidata_keyword
      path: indices/wikidata/keyword
      type: keyword
"#;
        let config: Config = serde_yaml::from_str(yaml).expect("Failed to parse config");
        assert_eq!(config.data.text.len(), 1);
        assert_eq!(config.embeddings.models.len(), 1);
        assert_eq!(config.embeddings.datasets.len(), 1);
        assert_eq!(config.indices.len(), 2);
        assert_eq!(config.server.indices.len(), 1);
    }
}
