use anyhow::{Context, Result};
use log::info;
use std::path::Path;

use search_rdf::data::DataSource;
use search_rdf::data::embedding::Embeddings;
use search_rdf::data::text::{TextData, TextEmbeddings};
use search_rdf::index::text::{KeywordIndex, TextEmbeddingIndex};
use search_rdf::index::{EmbeddingIndex, EmbeddingIndexParams, Search, SearchIndex};

use crate::search_rdf::config::{Config, IndexType};

pub fn run(config_path: &str, force: bool) -> Result<()> {
    let config = Config::load(config_path)?;

    let Some(indices) = config.indices else {
        info!("No index configuration found.");
        return Ok(());
    };

    info!("Building {} indices...", indices.len());

    for index_config in indices {
        if index_config.output.exists() && !force {
            info!(
                "[SKIP] {} (output exists, use --force to rebuild)",
                index_config.name
            );
            continue;
        }

        info!("[BUILD] {}...", index_config.name);
        build_index(&index_config.index_type, &index_config.output)?;
        info!(
            "[OK] {} -> {}",
            index_config.name,
            index_config.output.display()
        );
    }

    Ok(())
}

fn build_index(index_type: &IndexType, output: &Path) -> Result<()> {
    match index_type {
        IndexType::Keyword { text_data } => build_keyword_index(text_data, output),
        IndexType::TextEmbedding {
            text_data,
            embedding_data,
            params,
        } => build_text_embedding_index(text_data, embedding_data, params, output),
        IndexType::Embedding {
            embedding_data,
            params,
        } => build_embedding_index(embedding_data, params, output),
    }
}

pub fn load_index(index_type: &IndexType, output: &Path) -> Result<SearchIndex> {
    match index_type {
        IndexType::Keyword { text_data } => {
            let text_data = TextData::load(text_data)?;
            let index = KeywordIndex::load(text_data, output)?;
            Ok(SearchIndex::Keyword(index))
        }
        IndexType::TextEmbedding {
            text_data,
            embedding_data,
            ..
        } => {
            let text_data = TextData::load(text_data)?;
            let text_embeddings = TextEmbeddings::load(text_data, embedding_data)?;
            let index = TextEmbeddingIndex::load(text_embeddings, output)?;
            Ok(SearchIndex::TextEmbedding(index))
        }
        IndexType::Embedding { embedding_data, .. } => {
            let embeddings = Embeddings::load(embedding_data)?;
            let index = EmbeddingIndex::load(embeddings, output)?;
            Ok(SearchIndex::Embedding(index))
        }
    }
}

fn build_keyword_index(text_data_path: &Path, output_path: &Path) -> Result<()> {
    let text_data = TextData::load(text_data_path).context(format!(
        "Failed to load text data from: {}",
        text_data_path.display()
    ))?;

    info!(
        "Loaded {} text items with {} fields",
        text_data.len(),
        text_data.total_fields()
    );

    KeywordIndex::build(&text_data, output_path, &())?;

    Ok(())
}

fn build_text_embedding_index(
    text_data_path: &Path,
    embedding_data_path: &Path,
    params: &EmbeddingIndexParams,
    output_path: &Path,
) -> Result<()> {
    let text_data = TextData::load(text_data_path).context("Failed to load text data")?;
    let text_embeddings = TextEmbeddings::load(text_data, embedding_data_path)
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

    TextEmbeddingIndex::build(&text_embeddings, output_path, params)?;

    Ok(())
}

fn build_embedding_index(
    embedding_data_path: &Path,
    params: &EmbeddingIndexParams,
    output_path: &Path,
) -> Result<()> {
    let embeddings = Embeddings::load(embedding_data_path).context(format!(
        "Failed to load embeddings from: {}",
        embedding_data_path.display()
    ))?;

    info!(
        "Loaded {} embeddings for {} items ({} dimensions)",
        embeddings.total_fields(),
        embeddings.len(),
        embeddings.num_dimensions()
    );

    EmbeddingIndex::build(&embeddings, output_path, params)?;

    Ok(())
}
