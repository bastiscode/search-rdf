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
    Text { source: TextSource },
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
    #[serde(default)]
    pub params: EmbeddingParams,
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
        #[serde(default = "default_st_batch_size")]
        batch_size: usize,
    },
}

fn default_device() -> String {
    "cpu".to_string()
}

fn default_st_batch_size() -> usize {
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
        #[serde(default = "default_emb_batch_size")]
        batch_size: usize,
    },
}

fn default_emb_batch_size() -> usize {
    64
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
    #[serde(rename = "full-text")]
    FullText {
        text_data: PathBuf,
    },
    #[serde(rename = "text-embedding")]
    TextEmbedding {
        text_data: PathBuf,
        embedding_data: PathBuf,
        model: String,
        #[serde(default)]
        params: EmbeddingIndexParams,
    },
    Embedding {
        embedding_data: PathBuf,
        #[serde(default)]
        params: EmbeddingIndexParams,
    },
}

impl IndexType {
    pub fn get_model(&self) -> Option<&str> {
        match self {
            IndexType::TextEmbedding { model, .. } => Some(model),
            _ => None,
        }
    }
}

#[derive(Debug, Default, Deserialize, Serialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    // References the names of the indices specified in the indexes section
    #[serde(default)]
    pub indices: Vec<String>,
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
datasets: []
models: []
embeddings: []
indices: []
server:
  host: "0.0.0.0"
  port: 8080
"#;
        let config: Config = serde_yaml::from_str(yaml).expect("Failed to parse config");
        let server = config.server.as_ref().unwrap();
        assert_eq!(server.host, "0.0.0.0");
        assert_eq!(server.port, 8080);
    }

    #[test]
    fn test_parse_full_config() {
        let yaml = r#"
datasets:
  - name: test-dataset
    output: data/text/test
    type: text
    source:
      type: sparql-query
      endpoint: https://query.wikidata.org/sparql
      query: |
        SELECT ?item ?label
        WHERE {
          ?item wdt:P31 wd:Q5 .
          ?item rdfs:label ?label .
          FILTER(LANG(?label) = "en")
        }
        LIMIT 1000
      format: json

models:
  - name: primary_model
    type: vllm
    endpoint: http://localhost:8000
    model_name: mixedbread-ai/mxbai-embed-large-v1

embeddings:
  - name: wikidata_embeddings
    model: primary_model
    output: data/embeddings/wikidata
    type: text
    dataset: data/text/wikidata
    params:
      normalize: true

indices:
  - name: wikidata_keyword
    type: keyword
    text_data: data/text/wikidata
    output: indices/wikidata/keyword

  - name: wikidata_semantic
    type: text-embedding
    text_data: data/text/wikidata
    embedding_data: data/embeddings/wikidata
    output: indices/wikidata/semantic
    params:
      metric: cosine-normalized
      precision: float32
    model: primary_model

server:
  host: 0.0.0.0
  port: 8080
  cors: true
  indices:
    - wikidata_keyword
"#;
        let config: Config = serde_yaml::from_str(yaml).expect("Failed to parse config");

        // Check datasets
        assert_eq!(config.datasets.as_ref().unwrap().len(), 1);
        let dataset = &config.datasets.as_ref().unwrap()[0];
        assert_eq!(dataset.name, "test-dataset");
        assert_eq!(dataset.output, PathBuf::from("data/text/test"));
        match &dataset.data_type {
            DataType::Text { source } => match source {
                TextSource::SparqlQuery {
                    endpoint,
                    query,
                    format,
                    headers,
                } => {
                    assert_eq!(endpoint, "https://query.wikidata.org/sparql");
                    assert!(query.contains("SELECT ?item ?label"));
                    assert_eq!(format, &SPARQLResultFormat::JSON);
                    assert!(headers.is_none());
                }
                _ => panic!("Expected SparqlQuery source"),
            },
        }

        assert_eq!(config.models.as_ref().unwrap().len(), 1);
        assert_eq!(config.embeddings.as_ref().unwrap().len(), 1);
        assert_eq!(config.indices.as_ref().unwrap().len(), 2);
        let server = config.server.as_ref().unwrap();
        assert_eq!(server.indices.len(), 1);
    }
}
