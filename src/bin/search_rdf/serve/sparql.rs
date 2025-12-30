use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, anyhow};
use axum::body::Bytes;
use axum::extract::{Path as AxumPath, Query, State};
use axum::http::StatusCode;
use axum::response::Response;
use log::info;
use oxrdf::vocab::xsd;
use oxrdf::{Literal, Term, Variable};
use search_rdf::index::{Search, SearchIndex, SearchParams};
use serde::Deserialize;
use sparesults::{
    QueryResultsFormat, QueryResultsParser, QueryResultsSerializer, QuerySolution,
    ReaderQueryResultsParserOutput,
};

use crate::search_rdf::serve::search::{MatchInfo, perform_text_search_with_filter};

use super::search::SearchMatch;
use super::types::{AppError, AppState};

#[derive(Debug, Deserialize)]
pub struct QlProxyParams {
    // Search parameters (query is required)
    query: String,
    k: Option<usize>,
    #[serde(rename = "min-score")]
    min_score: Option<f32>,
    exact: Option<bool>,

    // Row variable name (default: "row")
    #[serde(default = "default_row_var")]
    rowvar: String,
}

fn default_row_var() -> String {
    "row".to_string()
}

/// Convert a SearchMatch to SPARQL bindings based on output variable configuration
/// Returns a tuple of (variables, values) that can be used to create a QuerySolution
fn search_match_to_solution(
    search_match: SearchMatch,
    variables: &[Variable],
    row: Term,
    rank: usize,
) -> Result<QuerySolution> {
    let mut values = vec![
        // row
        Some(row),
        // rank
        Some(Term::Literal(Literal::new_typed_literal(
            rank.to_string(),
            xsd::INTEGER,
        ))),
        // id
        Some(Term::Literal(Literal::new_typed_literal(
            search_match.id.to_string(),
            xsd::INTEGER,
        ))),
        // score
        Some(Term::Literal(Literal::new_typed_literal(
            search_match.score.to_string(),
            xsd::FLOAT,
        ))),
    ];

    // Extract identifier and field from MatchInfo if available
    if let MatchInfo::Text { identifier, field } = search_match.info {
        values.push(Some(Term::Literal(Literal::new_simple_literal(identifier))));
        values.push(Some(Term::Literal(Literal::new_simple_literal(field))));
    }

    if variables.len() != values.len() {
        return Err(anyhow!(
            "Mismatch between number of variables ({}) and values ({})",
            variables.len(),
            values.len()
        ));
    }

    Ok(QuerySolution::from((variables, values)))
}

fn get_id_from_identifier(index: &SearchIndex, identifier: &str) -> Option<u32> {
    match index {
        SearchIndex::Keyword(idx) => idx.data().id_from_identifier(identifier),
        SearchIndex::TextEmbedding(idx) => idx.data().text_data().id_from_identifier(identifier),
        _ => None,
    }
}

/// SPARQL SERVICE endpoint - simple search with parameters only
/// Receives a group graph pattern in the body (not SPARQL Results JSON)
pub async fn service(
    AxumPath(index_name): AxumPath<String>,
    State(_state): State<AppState>,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    // Log the incoming request
    info!("SERVICE endpoint called for index: {}", index_name);
    info!("Headers:\n{:?}", headers);
    info!("Body:\n{}", String::from_utf8_lossy(&body));

    Err(AppError(
        StatusCode::NOT_IMPLEMENTED,
        anyhow!("SERVICE endpoint not yet implemented"),
    ))
}

pub async fn qlproxy(
    AxumPath(index_name): AxumPath<String>,
    State(state): State<AppState>,
    Query(params): Query<QlProxyParams>,
    body: Bytes,
) -> Result<Response, AppError> {
    info!(
        "QL Proxy endpoint called for index: {}, params: {:?}",
        index_name, params
    );
    // Get index and model from state
    let index = state
        .inner
        .indices
        .get(&index_name)
        .cloned()
        .ok_or_else(|| {
            AppError(
                StatusCode::BAD_REQUEST,
                anyhow!("Index not found: {}", index_name),
            )
        })?;

    let model = state
        .inner
        .index_to_model
        .get(&index_name)
        .and_then(|model_name| state.inner.models.get(model_name));

    // Parse SPARQL Results JSON from request body
    let parser = QueryResultsParser::from_format(QueryResultsFormat::Json);
    let parsed = parser.for_reader(body.as_ref()).map_err(|e| {
        AppError(
            StatusCode::BAD_REQUEST,
            anyhow!("Failed to parse request: {}", e),
        )
    })?;

    let ReaderQueryResultsParserOutput::Solutions(solutions) = parsed else {
        return Err(AppError(
            StatusCode::BAD_REQUEST,
            anyhow!("Expected SPARQL results in JSON format"),
        ));
    };

    let start = Instant::now();
    // Parse input bindings - build id_to_row mapping
    let mut id_to_row: HashMap<u32, Term> = HashMap::new();
    for solution_result in solutions.into_iter() {
        let solution = solution_result.map_err(|e| {
            AppError(
                StatusCode::BAD_REQUEST,
                anyhow!("Failed to parse solution: {}", e),
            )
        })?;

        // Get row identifier
        let row_binding = solution.get(params.rowvar.as_str()).ok_or_else(|| {
            AppError(
                StatusCode::BAD_REQUEST,
                anyhow!("Missing required variable: {}", params.rowvar),
            )
        })?;

        // Get identifier for filter (required)
        let term = solution.get("identifier").ok_or_else(|| {
            AppError(
                StatusCode::BAD_REQUEST,
                anyhow!("Missing input variable: identifier"),
            )
        })?;

        let id = match term {
            Term::Literal(lit) => {
                let id = lit.value().parse::<u32>().map_err(|e| {
                    anyhow!("Expected integer literal representing an Id-based identifier: {e}")
                })?;
                Some(id)
            }
            Term::NamedNode(node) => get_id_from_identifier(&index, node.as_str()),
            _ => {
                return Err(AppError(
                    StatusCode::BAD_REQUEST,
                    anyhow!("Expected integer literal or IRI value as identifier"),
                ));
            }
        };

        if let Some(id) = id {
            id_to_row.insert(id, row_binding.clone());
        }
    }

    info!(
        "Filtered to {} identifiers for search in {}ms",
        id_to_row.len(),
        start.elapsed().as_millis()
    );

    // Build search params
    let mut search_params = SearchParams::default();

    if let Some(k) = params.k {
        search_params = search_params.with_k(k);
    }

    if let Some(exact) = params.exact {
        search_params = search_params.with_exact(exact);
    }

    if let Some(min_score) = params.min_score {
        search_params = search_params.with_min_score(min_score);
    }

    // Make thread-safe id_to_row map for filtering
    let id_to_row = Arc::new(id_to_row);
    let id_to_row_clone = id_to_row.clone();
    let filter = move |id| id_to_row_clone.contains_key(&id);

    // Always perform filtered search
    let start = Instant::now();
    let matches =
        perform_text_search_with_filter(index, vec![params.query], search_params, model, filter)
            .await?;

    info!("Search completed in {}ms", start.elapsed().as_millis());

    // Determine output variable names
    let variables = vec![
        Variable::new_unchecked(params.rowvar),
        Variable::new_unchecked("rank"),
        Variable::new_unchecked("id"),
        Variable::new_unchecked("score"),
        Variable::new_unchecked("identifier"),
        Variable::new_unchecked("field"),
    ];

    let mut response_buffer = Vec::new();
    let mut serializer = QueryResultsSerializer::from_format(QueryResultsFormat::Json)
        .serialize_solutions_to_writer(&mut response_buffer, variables.clone())
        .map_err(|e| anyhow!("Failed to create serializer: {}", e))?;

    for (rank, search_match) in matches.into_iter().flatten().enumerate() {
        // Get the row binding from id_to_row map
        let row = id_to_row.get(&search_match.id).cloned().ok_or_else(|| {
            anyhow!(
                "No row found for identifier ID {} in search match",
                search_match.id
            )
        })?;

        let solution = search_match_to_solution(search_match, &variables, row, rank + 1)?;

        serializer
            .serialize(&solution)
            .map_err(|e| anyhow!("Failed to serialize solution: {}", e))?;
    }

    serializer
        .finish()
        .map_err(|e| anyhow!("Failed to finish serialization: {}", e))?;

    // Return as response with correct content type
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/sparql-results+json")
        .body(response_buffer.into())
        .map_err(|e| anyhow!("Failed to build response: {}", e))?)
}
