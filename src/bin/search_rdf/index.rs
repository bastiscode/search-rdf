use anyhow::{Context, Result};
use log::info;
use search_rdf::index::EmbeddingSearchParams;
use search_rdf::index::FullTextSearchParams;
use search_rdf::index::FuzzySearchParams;
use search_rdf::index::KeywordSearchParams;
use serde::Deserialize;
use serde::de::{IntoDeserializer, value};
use std::collections::HashMap;
use std::path::Path;

use search_rdf::data::DataSource;
use search_rdf::data::embedding::Embeddings;
use search_rdf::data::{Data, EmbeddingsWithData};
use search_rdf::index::{EmbeddingIndex, EmbeddingIndexParams, EmbeddingIndexWithData, Search};
use search_rdf::index::{FullTextIndex, FuzzyIndex, KeywordIndex};

use crate::search_rdf::config::{Config, IndexType};

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SearchParams {
    Keyword(KeywordSearchParams),
    Fuzzy(FuzzySearchParams),
    #[serde(rename = "full-text")]
    FullText(FullTextSearchParams),
    #[serde(rename = "embedding", alias = "embedding-with-data")]
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

impl SearchParams {
    pub fn k(&self) -> usize {
        match self {
            SearchParams::Keyword(p) => p.k,
            SearchParams::Fuzzy(p) => p.k,
            SearchParams::FullText(p) => p.k,
            SearchParams::Embedding(p) => p.k,
        }
    }

    pub fn bump_k(self) -> Self {
        match self {
            SearchParams::Keyword(mut p) => {
                p.k += 1;
                SearchParams::Keyword(p)
            }
            SearchParams::Fuzzy(mut p) => {
                p.k += 1;
                SearchParams::Fuzzy(p)
            }
            SearchParams::FullText(mut p) => {
                p.k += 1;
                SearchParams::FullText(p)
            }
            SearchParams::Embedding(mut p) => {
                p.k += 1;
                SearchParams::Embedding(p)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum SearchIndex {
    Keyword(KeywordIndex),
    Fuzzy(FuzzyIndex),
    FullText(FullTextIndex),
    EmbeddingWithData(EmbeddingIndexWithData),
    Embedding(EmbeddingIndex),
}

impl SearchIndex {
    pub fn index_type(&self) -> &'static str {
        match self {
            SearchIndex::Keyword(index) => index.index_type(),
            SearchIndex::Fuzzy(index) => index.index_type(),
            SearchIndex::FullText(index) => index.index_type(),
            SearchIndex::EmbeddingWithData(index) => index.index_type(),
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
        IndexType::Keyword { data } => build_keyword_index(base_dir, data, index_dir),
        IndexType::Fuzzy { data } => build_fuzzy_index(base_dir, data, index_dir),
        IndexType::FullText { data } => build_full_text_index(base_dir, data, index_dir),
        IndexType::EmbeddingWithData {
            data,
            embedding_data,
            params,
            ..
        } => build_embedding_index_with_data(base_dir, data, embedding_data, params, index_dir),
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
        IndexType::Keyword { data } => {
            let text_data = Data::load(&base_dir.join(data))?;
            let index = KeywordIndex::load(text_data, &index_dir)?;
            Ok(SearchIndex::Keyword(index))
        }
        IndexType::Fuzzy { data } => {
            let text_data = Data::load(&base_dir.join(data))?;
            let index = FuzzyIndex::load(text_data, &index_dir)?;
            Ok(SearchIndex::Fuzzy(index))
        }
        IndexType::FullText { data } => {
            let data = Data::load(&base_dir.join(data))?;
            let index = FullTextIndex::load(data, &index_dir)?;
            Ok(SearchIndex::FullText(index))
        }
        IndexType::EmbeddingWithData {
            data,
            embedding_data,
            ..
        } => {
            let data = Data::load(&base_dir.join(data))?;
            let embeddings = EmbeddingsWithData::load(data, &base_dir.join(embedding_data))?;
            let index = EmbeddingIndexWithData::load(embeddings, &index_dir)?;
            Ok(SearchIndex::EmbeddingWithData(index))
        }
        IndexType::Embedding { embedding_data, .. } => {
            let embeddings = Embeddings::load(&base_dir.join(embedding_data))?;
            let index = EmbeddingIndex::load(embeddings, &index_dir)?;
            Ok(SearchIndex::Embedding(index))
        }
    }
}

fn build_keyword_index(base_dir: &Path, data_path: &Path, output_path: &Path) -> Result<()> {
    let data_path = base_dir.join(data_path);
    let output_path = base_dir.join(output_path);

    let data = Data::load(&data_path)
        .context(format!("Failed to load data from: {}", data_path.display()))?;

    info!(
        "Loaded {} items with {} fields",
        data.len(),
        data.total_fields()
    );

    KeywordIndex::build(&data, &output_path, &())?;

    Ok(())
}

fn build_fuzzy_index(base_dir: &Path, data_path: &Path, output_path: &Path) -> Result<()> {
    let data_path = base_dir.join(data_path);
    let output_path = base_dir.join(output_path);

    let data = Data::load(&data_path)
        .context(format!("Failed to load data from: {}", data_path.display()))?;

    info!(
        "Loaded {} items with {} fields",
        data.len(),
        data.total_fields()
    );

    FuzzyIndex::build(&data, &output_path, &())?;

    Ok(())
}

fn build_full_text_index(base_dir: &Path, data_path: &Path, output_path: &Path) -> Result<()> {
    let data_path = base_dir.join(data_path);
    let output_path = base_dir.join(output_path);

    let data =
        Data::load(&data_path).context(format!("Failed to load from: {}", data_path.display()))?;

    info!(
        "Loaded {} items with {} fields",
        data.len(),
        data.total_fields()
    );

    FullTextIndex::build(&data, &output_path, &())?;

    Ok(())
}

fn build_embedding_index_with_data(
    base_dir: &Path,
    data_path: &Path,
    embedding_data_path: &Path,
    params: &EmbeddingIndexParams,
    output_path: &Path,
) -> Result<()> {
    let data_path = base_dir.join(data_path);
    let embedding_data_path = base_dir.join(embedding_data_path);
    let output_path = base_dir.join(output_path);

    let data = Data::load(&data_path).context("Failed to load data")?;
    let embeddings = EmbeddingsWithData::load(data, &embedding_data_path)
        .context("Failed to load embeddings")?;

    info!(
        "Loaded {} items with {} fields",
        embeddings.data().len(),
        embeddings.data().total_fields()
    );
    info!(
        "Loaded {} embeddings for {} items ({} dimensions)",
        embeddings.total_fields(),
        embeddings.len(),
        embeddings.num_dimensions()
    );

    EmbeddingIndexWithData::build(&embeddings, &output_path, params)?;

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
