use anyhow::{Context, Result};
use log::info;
use search_rdf::index::embedding::EmbeddingSearchParams;
use search_rdf::index::keyword::KeywordSearchParams;
use search_rdf::index::text::embedding::TextEmbeddingSearchParams;
use search_rdf::index::text::full_text::FullTextSearchParams;
use serde::Deserialize;
use serde::de::{IntoDeserializer, value};
use std::collections::HashMap;
use std::path::Path;

use search_rdf::data::DataSource;
use search_rdf::data::embedding::Embeddings;
use search_rdf::data::text::{TextData, TextEmbeddings};
use search_rdf::index::text::{FullTextIndex, KeywordIndex, TextEmbeddingIndex};
use search_rdf::index::{EmbeddingIndex, EmbeddingIndexParams, Search};

use crate::search_rdf::config::{Config, IndexType};

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SearchParams {
    Keyword(KeywordSearchParams),
    #[serde(rename = "full-text")]
    FullText(FullTextSearchParams),
    #[serde(rename = "text-embedding")]
    TextEmbedding(TextEmbeddingSearchParams),
    Embedding(EmbeddingSearchParams),
}

impl TryFrom<HashMap<String, String>> for SearchParams {
    type Error = anyhow::Error;

    fn try_from(map: HashMap<String, String>) -> Result<Self, Self::Error> {
        // use serde to deserialize the hashmap into the appropriate SearchParams variant
        Self::deserialize(map.into_deserializer()).map_err(|e: value::Error| {
            anyhow::anyhow!("Failed to deserialize SearchParams from map: {}", e)
        })
    }
}

#[derive(Debug, Clone)]
pub enum SearchIndex {
    Keyword(KeywordIndex),
    FullText(FullTextIndex),
    TextEmbedding(TextEmbeddingIndex),
    Embedding(EmbeddingIndex),
}

impl SearchIndex {
    pub fn index_type(&self) -> &'static str {
        match self {
            SearchIndex::Keyword(index) => index.index_type(),
            SearchIndex::FullText(index) => index.index_type(),
            SearchIndex::TextEmbedding(index) => index.index_type(),
            SearchIndex::Embedding(index) => index.index_type(),
        }
    }
}

pub fn run(config_path: &Path, force: bool) -> Result<()> {
    let config = Config::load(config_path)?;
    let config_dir = config_path
        .parent()
        .context("Failed to get config directory")?;

    info!("Building indices in {}", config_dir.display());

    let Some(indices) = config.indices else {
        info!("No index configuration found.");
        return Ok(());
    };

    info!("Building {} indices...", indices.len());

    for index_config in indices {
        if config_dir.join(&index_config.output).exists() && !force {
            info!(
                "[SKIP] {} (output exists, use --force to rebuild)",
                index_config.name
            );
            continue;
        }

        info!("[BUILD] {}...", index_config.name);
        build_index(config_dir, &index_config.index_type, &index_config.output)?;
        info!(
            "[OK] {} -> {}",
            index_config.name,
            index_config.output.display()
        );
    }

    Ok(())
}

fn build_index(base_dir: &Path, index_type: &IndexType, index_dir: &Path) -> Result<()> {
    match index_type {
        IndexType::Keyword { text_data } => build_keyword_index(base_dir, text_data, index_dir),
        IndexType::FullText { text_data } => build_full_text_index(base_dir, text_data, index_dir),
        IndexType::TextEmbedding {
            text_data,
            embedding_data,
            params,
            ..
        } => build_text_embedding_index(base_dir, text_data, embedding_data, params, index_dir),
        IndexType::Embedding {
            embedding_data,
            params,
        } => build_embedding_index(base_dir, embedding_data, params, index_dir),
    }
}

pub fn load_index(
    base_dir: &Path,
    index_type: &IndexType,
    index_dir: &Path,
) -> Result<SearchIndex> {
    let index_dir = base_dir.join(index_dir);

    match index_type {
        IndexType::Keyword { text_data } => {
            let text_data = TextData::load(&base_dir.join(text_data))?;
            let index = KeywordIndex::load(text_data, &index_dir)?;
            Ok(SearchIndex::Keyword(index))
        }
        IndexType::FullText { text_data } => {
            let text_data = TextData::load(&base_dir.join(text_data))?;
            let index = FullTextIndex::load(text_data, &index_dir)?;
            Ok(SearchIndex::FullText(index))
        }
        IndexType::TextEmbedding {
            text_data,
            embedding_data,
            ..
        } => {
            let text_data = TextData::load(&base_dir.join(text_data))?;
            let text_embeddings = TextEmbeddings::load(text_data, &base_dir.join(embedding_data))?;
            let index = TextEmbeddingIndex::load(text_embeddings, &index_dir)?;
            Ok(SearchIndex::TextEmbedding(index))
        }
        IndexType::Embedding { embedding_data, .. } => {
            let embeddings = Embeddings::load(&base_dir.join(embedding_data))?;
            let index = EmbeddingIndex::load(embeddings, &index_dir)?;
            Ok(SearchIndex::Embedding(index))
        }
    }
}

fn build_keyword_index(base_dir: &Path, text_data_path: &Path, output_path: &Path) -> Result<()> {
    let text_data_path = base_dir.join(text_data_path);
    let output_path = base_dir.join(output_path);

    let text_data = TextData::load(&text_data_path).context(format!(
        "Failed to load text data from: {}",
        text_data_path.display()
    ))?;

    info!(
        "Loaded {} text items with {} fields",
        text_data.len(),
        text_data.total_fields()
    );

    KeywordIndex::build(&text_data, &output_path, &())?;

    Ok(())
}

fn build_full_text_index(base_dir: &Path, text_data_path: &Path, output_path: &Path) -> Result<()> {
    let text_data_path = base_dir.join(text_data_path);
    let output_path = base_dir.join(output_path);

    let text_data = TextData::load(&text_data_path).context(format!(
        "Failed to load text data from: {}",
        text_data_path.display()
    ))?;

    info!(
        "Loaded {} text items with {} fields",
        text_data.len(),
        text_data.total_fields()
    );

    FullTextIndex::build(&text_data, &output_path, &())?;

    Ok(())
}

fn build_text_embedding_index(
    base_dir: &Path,
    text_data_path: &Path,
    embedding_data_path: &Path,
    params: &EmbeddingIndexParams,
    output_path: &Path,
) -> Result<()> {
    let text_data_path = base_dir.join(text_data_path);
    let embedding_data_path = base_dir.join(embedding_data_path);
    let output_path = base_dir.join(output_path);

    let text_data = TextData::load(&text_data_path).context("Failed to load text data")?;
    let text_embeddings = TextEmbeddings::load(text_data, &embedding_data_path)
        .context("Failed to load text embeddings")?;

    info!(
        "Loaded {} text items with {} fields",
        text_embeddings.text_data().len(),
        text_embeddings.text_data().total_fields()
    );
    info!(
        "Loaded {} embeddings for {} items ({} dimensions)",
        text_embeddings.total_fields(),
        text_embeddings.len(),
        text_embeddings.num_dimensions()
    );

    TextEmbeddingIndex::build(&text_embeddings, &output_path, params)?;

    Ok(())
}

fn build_embedding_index(
    base_dir: &Path,
    embedding_data_path: &Path,
    params: &EmbeddingIndexParams,
    output_path: &Path,
) -> Result<()> {
    let embedding_data_path = base_dir.join(embedding_data_path);
    let output_path = base_dir.join(output_path);

    let embeddings = Embeddings::load(&embedding_data_path).context(format!(
        "Failed to load embeddings from: {}",
        embedding_data_path.display()
    ))?;

    info!(
        "Loaded {} embeddings for {} items ({} dimensions)",
        embeddings.total_fields(),
        embeddings.len(),
        embeddings.num_dimensions()
    );

    EmbeddingIndex::build(&embeddings, &output_path, params)?;

    Ok(())
}
