use anyhow::{Context, Result};
use search_rdf::data::item::FieldType;
use search_rdf::data::item::sparql::SPARQLResultFormat;
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
#[serde(rename_all = "kebab-case")]
pub struct DataConfig {
    pub name: String,
    pub output: PathBuf,
    pub source: DataSource,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum DataSource {
    SparqlQuery {
        endpoint: String,
        query: Option<String>,
        path: Option<PathBuf>,
        format: SPARQLResultFormat,
        headers: Option<HashMap<String, String>>,
        default_field_type: FieldType,
    },
    Sparql {
        path: PathBuf,
        format: SPARQLResultFormat,
        default_field_type: FieldType,
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
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ModelType {
    Vllm {
        endpoint: String,
        model_name: String,
    },
    SentenceTransformer {
        model_name: String,
        #[serde(default = "default_device")]
        device: String,
        #[serde(default = "default_model_batch_size")]
        batch_size: usize,
    },
    #[serde(alias = "huggingface-image")]
    HuggingFaceImage {
        model_name: String,
        #[serde(default = "default_device")]
        device: String,
        #[serde(default = "default_model_batch_size")]
        batch_size: usize,
    },
    OpenClip {
        model: String,
        #[serde(default = "default_device")]
        device: String,
        #[serde(default = "default_model_batch_size")]
        batch_size: usize,
    },
}

fn default_device() -> String {
    "cpu".to_string()
}

fn default_model_batch_size() -> usize {
    16
}

#[derive(Debug, Deserialize, Serialize)]
pub struct EmbeddingConfig {
    pub name: String,
    pub output: PathBuf,
    pub model: String,
    pub data: PathBuf,
    #[serde(default = "default_emb_batch_size")]
    pub batch_size: usize,
}

fn default_emb_batch_size() -> usize {
    64
}

#[derive(Debug, Deserialize, Serialize)]
pub struct IndexConfig {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub output: PathBuf,
    #[serde(flatten)]
    pub index_type: IndexType,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum IndexType {
    Keyword {
        data: PathBuf,
    },
    Fuzzy {
        data: PathBuf,
    },
    FullText {
        data: PathBuf,
    },
    EmbeddingWithData {
        data: PathBuf,
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
            IndexType::EmbeddingWithData { model, .. } => Some(model),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct SparqlConfig {
    pub prefix: String,
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
    #[serde(default = "default_max_input_size")]
    pub max_input_size: String,
    #[serde(default)]
    pub sparql: Option<SparqlConfig>,
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}

fn default_port() -> u16 {
    8080
}

fn default_max_input_size() -> String {
    "100MB".to_string()
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
    fn test_parse_open_clip_model() {
        let yaml = r#"
models:
  - name: clip
    type: open-clip
    model: "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    device: cuda
    batch_size: 32
"#;
        let config: Config = serde_yaml::from_str(yaml).expect("Failed to parse config");
        let models = config.models.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "clip");
        match &models[0].model_type {
            ModelType::OpenClip {
                model,
                device,
                batch_size,
            } => {
                assert_eq!(model, "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K");
                assert_eq!(device, "cuda");
                assert_eq!(*batch_size, 32);
            }
            _ => panic!("Expected OpenClip model type"),
        }
    }

    #[test]
    fn test_parse_open_clip_defaults() {
        let yaml = r#"
models:
  - name: clip
    type: open-clip
    model: "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
"#;
        let config: Config = serde_yaml::from_str(yaml).expect("Failed to parse config");
        let models = config.models.unwrap();
        match &models[0].model_type {
            ModelType::OpenClip {
                device, batch_size, ..
            } => {
                assert_eq!(device, "cpu");
                assert_eq!(*batch_size, 16);
            }
            _ => panic!("Expected OpenClip model type"),
        }
    }

    #[test]
    fn test_parse_full_config() {
        let yaml = r#"
datasets:
  - name: test-dataset
    output: data/text/test
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
      default_field_type: text

models:
  - name: primary_model
    type: vllm
    endpoint: http://localhost:8000
    model_name: mixedbread-ai/mxbai-embed-large-v1

embeddings:
  - name: wikidata_embeddings
    model: primary_model
    output: data/embeddings/wikidata
    data: data/text/wikidata
    batch_size: 64

indices:
  - name: wikidata_keyword
    type: keyword
    data: data/text/wikidata
    output: indices/wikidata/keyword
    description: Keyword search over Wikidata labels

  - name: wikidata_semantic
    type: embedding-with-data
    data: data/text/wikidata
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
        match &dataset.source {
            DataSource::SparqlQuery {
                endpoint,
                query,
                path,
                format,
                default_field_type,
                headers,
            } => {
                assert_eq!(endpoint, "https://query.wikidata.org/sparql");
                assert!(query.as_ref().unwrap().contains("SELECT ?item ?label"));
                assert!(path.is_none());
                assert_eq!(format, &SPARQLResultFormat::Json);
                assert!(headers.is_none());
                assert_eq!(default_field_type, &FieldType::Text);
            }
            _ => panic!("Expected SparqlQuery source"),
        }

        assert_eq!(config.models.as_ref().unwrap().len(), 1);
        assert_eq!(config.embeddings.as_ref().unwrap().len(), 1);
        let indices = config.indices.as_ref().unwrap();
        assert_eq!(indices.len(), 2);
        assert_eq!(
            indices[0].description.as_deref(),
            Some("Keyword search over Wikidata labels")
        );
        assert_eq!(indices[1].description, None);
        let server = config.server.as_ref().unwrap();
        assert_eq!(server.indices.len(), 1);
    }
}
