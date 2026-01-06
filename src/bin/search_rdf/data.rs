use anyhow::Result;
use log::info;
use search_rdf::data::TextData;
use search_rdf::data::item::Item;
use search_rdf::data::item::jsonl::stream_items_from_jsonl_file;
use search_rdf::data::item::sparql::{
    SPARQLResultFormat, stream_items_from_sparql_result, stream_items_from_sparql_result_file,
};
use std::collections::HashMap;
use std::fs::read_to_string;
use std::io::Read;
use std::path::Path;

use crate::search_rdf::config::{Config, DataType, TextSource};

pub fn run(config_path: &Path, force: bool) -> Result<()> {
    let config = Config::load(config_path)?;
    let config_dir = config_path
        .parent()
        .expect("Failed to get config directory");

    info!("Building datasets in {}", config_dir.display());

    let Some(datasets) = config.datasets else {
        info!("No datasets defined in configuration.");
        return Ok(());
    };

    info!("Building {} datasets...", datasets.len());

    for dataset in &datasets {
        if config_dir.join(&dataset.output).exists() && !force {
            info!(
                "[SKIP] {} (output exists, use --force to rebuild)",
                dataset.name
            );
            continue;
        }

        info!("[BUILD] {}...", dataset.name);
        match &dataset.data_type {
            DataType::Text { source } => {
                build_text_data(config_dir, source, &dataset.output)?;
            }
        }

        info!("[OK] {} -> {}", dataset.name, dataset.output.display());
    }

    Ok(())
}

fn build_text_data(base_dir: &Path, source: &TextSource, output: &Path) -> Result<()> {
    // Get iterator of Items based on source type
    let items: Box<dyn Iterator<Item = Result<Item>>> = match source {
        TextSource::Jsonl { path } => {
            info!("Streaming text data from JSONL file: {}", path.display());
            Box::new(stream_items_from_jsonl_file(path)?)
        }
        TextSource::Sparql { path, format } => {
            info!(
                "Streaming text data from SPARQL result file: {}",
                path.display()
            );
            Box::new(stream_items_from_sparql_result_file(path, *format)?)
        }
        TextSource::SparqlQuery {
            endpoint,
            query,
            path,
            format,
            headers,
        } => {
            let query = match (query, path) {
                (Some(q), None) => q.clone(),
                (None, Some(p)) => read_to_string(p)?,
                _ => {
                    return Err(anyhow::anyhow!(
                        "Either 'query' or 'path' must be provided, but not both."
                    ));
                }
            };

            // Execute SPARQL query
            let response = execute_sparql_query(endpoint, &query, *format, headers.as_ref())?;

            // Create iterator from response
            let items = stream_items_from_sparql_result(response, *format)?;

            Box::new(items)
        }
    };

    info!("Building text dataset...");

    // Build TextData
    TextData::build(items, &base_dir.join(output))?;

    Ok(())
}

pub fn execute_sparql_query(
    endpoint: &str,
    query: &str,
    format: SPARQLResultFormat,
    headers: Option<&HashMap<String, String>>,
) -> Result<impl Read + use<>> {
    info!(
        "Executing SPARQL query against {}:\n{}",
        endpoint,
        query.trim()
    );
    let url = format!("{}?query={}", endpoint, urlencoding::encode(query));

    let mut request = ureq::get(&url)
        .header("User-Agent", "search-rdf-bot")
        .header("Accept", format.mime_type());

    if let Some(headers) = headers {
        for (key, value) in headers {
            info!("With header \"{}: {}\"", key, value);
            request = request.header(key, value);
        }
    }

    let response = request.call()?;
    Ok(response.into_body().into_reader())
}
