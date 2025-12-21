use anyhow::Result;
use search_rdf::data::TextData;
use search_rdf::data::text::item::TextItem;
use search_rdf::data::text::item::jsonl::stream_text_items_from_jsonl_file;
use search_rdf::data::text::item::sparql::{
    SPARQLResultFormat, stream_text_items_from_sparql_result,
    stream_text_items_from_sparql_result_file,
};
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use crate::search_rdf::config::{Config, DataType, TextSource};

pub fn run(config_path: &str, force: bool) -> Result<()> {
    let config = Config::load(config_path)?;

    let Some(datasets) = config.datasets else {
        println!("No datasets defined in configuration.");
        return Ok(());
    };

    println!("Building {} datasets...", datasets.len());

    for dataset in &datasets {
        if dataset.output.exists() && !force {
            println!(
                "  [SKIP] {} (output exists, use --force to rebuild)",
                dataset.name
            );
            continue;
        }

        println!("  [BUILD] {}...", dataset.name);
        match &dataset.data_type {
            DataType::Text { source } => {
                build_text_data(source, &dataset.output)?;
            }
        }

        println!("  [OK] {} -> {}", dataset.name, dataset.output.display());
    }

    Ok(())
}

fn build_text_data(source: &TextSource, output: &Path) -> Result<()> {
    // Get iterator of TextItems based on source type
    let items: Box<dyn Iterator<Item = Result<TextItem>>> = match source {
        TextSource::Jsonl { path } => Box::new(stream_text_items_from_jsonl_file(path)?),
        TextSource::Sparql { path, format } => {
            Box::new(stream_text_items_from_sparql_result_file(path, *format)?)
        }
        TextSource::SparqlQuery {
            endpoint,
            query,
            format,
            headers,
        } => {
            // Execute SPARQL query
            let response = execute_sparql_query(endpoint, query, *format, headers.as_ref())?;

            // Create iterator from response
            let items = stream_text_items_from_sparql_result(response, *format)?;

            Box::new(items)
        }
    };

    // Build TextData
    TextData::build(items, output)?;

    Ok(())
}

pub fn execute_sparql_query(
    endpoint: &str,
    query: &str,
    format: SPARQLResultFormat,
    headers: Option<&HashMap<String, String>>,
) -> Result<impl Read> {
    let mut request = ureq::post(endpoint)
        .header("User-Agent", "search-rdf-bot")
        .header("Accept", format.mime_type());

    if let Some(headers) = headers {
        for (key, value) in headers {
            request = request.header(key, value);
        }
    }

    let response = request.send_json(serde_json::json!({ "query": query }))?;
    Ok(response.into_body().into_reader())
}
