use anyhow::{Context, Result};
use std::path::Path;

use search_rdf::data::DataSource;
use search_rdf::data::embedding::Embeddings;
use search_rdf::data::text::{TextData, TextEmbeddings};
use search_rdf::index::text::{KeywordIndex, TextEmbeddingIndex};
use search_rdf::index::{EmbeddingIndex, EmbeddingIndexParams, SearchIndex};

use crate::search_rdf::config::{Config, IndexConfig, IndexType};

pub fn run(config_path: &str, force: bool, only: Option<Vec<String>>) -> Result<()> {
    let config = Config::load(config_path)?;

    let Some(indices) = config.indices else {
        println!("No index configuration found.");
        return Ok(());
    };

    // Filter indices if --only is specified
    let indices: Vec<_> = if let Some(only_names) = only {
        indices
            .into_iter()
            .filter(|idx| only_names.contains(&idx.name))
            .collect()
    } else {
        indices
    };

    println!("Building {} indices...", indices.len());

    for index_config in indices {
        if index_config.output.exists() && !force {
            println!(
                "  [SKIP] {} (output exists, use --force to rebuild)",
                index_config.name
            );
            continue;
        }

        println!("  [BUILD] {}...", index_config.name);
        build_index(&index_config)?;
        println!(
            "  [OK] {} -> {}",
            index_config.name,
            index_config.output.display()
        );
    }

    println!("Done!");
    Ok(())
}

fn build_index(index_config: &IndexConfig) -> Result<()> {
    let output = &index_config.output;
    match &index_config.index_type {
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

fn build_keyword_index(text_data_path: &Path, output_path: &Path) -> Result<()> {
    let text_data = TextData::load(text_data_path).context(format!(
        "Failed to load text data from: {}",
        text_data_path.display()
    ))?;

    println!("    Loaded {} text items", text_data.len());

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

    println!(
        "    Loaded {} text items",
        text_embeddings.text_data().len()
    );
    println!(
        "    Loaded {} embeddings ({} dimensions)",
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

    println!(
        "    Loaded {} embeddings ({} dimensions)",
        embeddings.len(),
        embeddings.num_dimensions()
    );

    EmbeddingIndex::build(&embeddings, output_path, &params)?;

    Ok(())
}
