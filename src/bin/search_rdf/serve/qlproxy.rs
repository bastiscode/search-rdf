use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use axum::body::Bytes;
use axum::extract::{Path as AxumPath, Query, State};
use axum::http::StatusCode;
use axum::response::Response;
use oxrdf::{Literal, Term, Variable};
use search_rdf::index::{Search, SearchIndex, SearchParams};
use serde::Deserialize;
use sparesults::{
    QueryResultsFormat, QueryResultsParser, QueryResultsSerializer, QuerySolution,
    ReaderQueryResultsParserOutput,
};

use crate::search_rdf::serve::search::{
    MatchInfo, perform_text_search, perform_text_search_with_filter,
};

use super::search::SearchMatch;
use super::types::{AppError, AppState};

#[derive(Deserialize)]
pub struct QlProxyParams {
    // Input variable names
    #[serde(rename = "input-query")]
    input_query: Option<String>,
    #[serde(rename = "input-identifier")]
    input_identifier: Option<String>,

    // Output variable names
    #[serde(rename = "output-id")]
    output_id: Option<String>,
    #[serde(rename = "output-score")]
    output_score: Option<String>,

    // For text match info
    #[serde(rename = "output-identifier")]
    output_identifier: Option<String>,
    #[serde(rename = "output-field")]
    output_field: Option<String>,

    // Search parameters
    #[serde(rename = "param-query")]
    param_query: Option<String>,
    #[serde(rename = "param-k")]
    param_k: Option<usize>,
    #[serde(rename = "param-min-score")]
    param_min_score: Option<f32>,
    #[serde(rename = "param-exact")]
    param_exact: Option<bool>,

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
    row: Term,
    row_var: &str,
    params: &QlProxyParams,
) -> QuerySolution {
    let mut variables = Vec::new();
    let mut values = Vec::new();

    // Always include row variable
    variables.push(Variable::new_unchecked(row_var));
    values.push(Some(row));

    // Add output variables based on params
    if let Some(var) = &params.output_id {
        variables.push(Variable::new_unchecked(var));
        values.push(Some(Term::Literal(Literal::new_simple_literal(
            search_match.id.to_string(),
        ))));
    }

    if let Some(var) = &params.output_score {
        variables.push(Variable::new_unchecked(var));
        values.push(Some(Term::Literal(Literal::new_simple_literal(
            search_match.score.to_string(),
        ))));
    }

    // Extract identifier and field from MatchInfo if available
    if let MatchInfo::Text { identifier, field } = search_match.info {
        if let Some(var) = &params.output_identifier {
            variables.push(Variable::new_unchecked(var));
            values.push(Some(Term::Literal(Literal::new_simple_literal(identifier))));
        }

        if let Some(var) = &params.output_field {
            variables.push(Variable::new_unchecked(var));
            values.push(Some(Term::Literal(Literal::new_simple_literal(field))));
        }
    }

    QuerySolution::from((variables, values))
}

fn determine_output_vars(params: &QlProxyParams, row_var: &str) -> Vec<Variable> {
    let mut vars = vec![Variable::new_unchecked(row_var)];

    if let Some(var) = &params.output_id {
        vars.push(Variable::new_unchecked(var));
    }
    if let Some(var) = &params.output_score {
        vars.push(Variable::new_unchecked(var));
    }
    if let Some(var) = &params.output_identifier {
        vars.push(Variable::new_unchecked(var));
    }
    if let Some(var) = &params.output_field {
        vars.push(Variable::new_unchecked(var));
    }

    vars
}

fn get_id_from_identifier(index: &SearchIndex, identifier: &str) -> Option<u32> {
    match index {
        SearchIndex::Keyword(idx) => idx.data().id_from_identifier(identifier),
        SearchIndex::TextEmbedding(idx) => idx.data().text_data().id_from_identifier(identifier),
        _ => None,
    }
}

pub async fn qlproxy(
    AxumPath(index_name): AxumPath<String>,
    State(state): State<AppState>,
    Query(params): Query<QlProxyParams>,
    body: Bytes,
) -> Result<Response, AppError> {
    // Validate that we have either input-query or param-query (but not both)
    let has_input_query = params.input_query.is_some();
    let has_param_query = params.param_query.is_some();

    if has_input_query && has_param_query {
        return Err(AppError(
            StatusCode::BAD_REQUEST,
            anyhow!("Cannot specify both input-query and param-query"),
        ));
    }

    if !has_input_query && !has_param_query {
        return Err(AppError(
            StatusCode::BAD_REQUEST,
            anyhow!("Must specify either input-query or param-query"),
        ));
    }

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

    // Extract input variable names
    let row_var = params.rowvar.as_str();
    let input_query_var = params.input_query.as_deref();
    let input_identifier_var = params.input_identifier.as_deref();

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

    // Parse input bindings
    let mut rows = vec![];
    let mut queries: Option<Vec<String>> = None;
    let mut id_to_row: Option<HashMap<u32, Term>> = None;
    for solution_result in solutions.into_iter() {
        let solution = solution_result.map_err(|e| {
            AppError(
                StatusCode::BAD_REQUEST,
                anyhow!("Failed to parse solution: {}", e),
            )
        })?;

        // Get row identifier
        let row_binding = solution.get(row_var).ok_or_else(|| {
            AppError(
                StatusCode::BAD_REQUEST,
                anyhow!("Missing required variable: {}", row_var),
            )
        })?;
        rows.push(row_binding.clone());

        // Get query
        if let Some(var) = input_query_var {
            let term = solution.get(var).ok_or_else(|| {
                AppError(
                    StatusCode::BAD_REQUEST,
                    anyhow!("Missing input variable: {}", var),
                )
            })?;

            let Term::Literal(lit) = term else {
                return Err(AppError(
                    StatusCode::BAD_REQUEST,
                    anyhow!("Expected literal value for query"),
                ));
            };

            let query = lit.value().to_string();
            queries.get_or_insert_default().push(query);
        };

        // Get identifier for filter
        if let Some(var) = input_identifier_var {
            let term = solution.get(var).ok_or_else(|| {
                AppError(
                    StatusCode::BAD_REQUEST,
                    anyhow!("Missing input variable: {}", var),
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
                id_to_row
                    .get_or_insert_default()
                    .insert(id, row_binding.clone());
            }
        };
    }

    // Build search params
    let mut search_params = SearchParams::default();

    if let Some(k) = params.param_k {
        search_params = search_params.with_k(k);
    }

    if let Some(exact) = params.param_exact {
        search_params = search_params.with_exact(exact);
    }

    if let Some(min_score) = params.param_min_score {
        search_params = search_params.with_min_score(min_score);
    }

    let queries = queries
        .or_else(|| params.param_query.as_ref().map(|q| vec![q.to_string()]))
        .expect("Either input-query or param-query must be provided");

    // make thread safe id_to_row map for filtering
    let id_to_row = id_to_row.map(Arc::new);

    let matches = if let Some(id_to_row) = id_to_row.as_ref() {
        let id_to_row = id_to_row.clone();
        let filter = move |id| id_to_row.contains_key(&id);
        perform_text_search_with_filter(index, queries, search_params, model, filter).await?
    } else {
        perform_text_search(index, queries, search_params, model).await?
    };

    // Determine output variable names
    let output_vars = determine_output_vars(&params, row_var);

    let mut response_buffer = Vec::new();
    let mut serializer = QueryResultsSerializer::from_format(QueryResultsFormat::Json)
        .serialize_solutions_to_writer(&mut response_buffer, output_vars)
        .map_err(|e| anyhow!("Failed to create serializer: {}", e))?;

    let mut num_rows = 0;
    for search_matches in matches {
        for search_match in search_matches {
            let row = if let Some(id_to_row) = &id_to_row {
                id_to_row.get(&search_match.id).cloned().ok_or_else(|| {
                    anyhow!(
                        "No row found for identifier ID {} in search match",
                        search_match.id
                    )
                })?
            } else {
                Term::Literal(Literal::new_simple_literal(num_rows.to_string()))
            };
            let solution = search_match_to_solution(search_match, row, row_var, &params);

            serializer
                .serialize(&solution)
                .map_err(|e| anyhow!("Failed to serialize solution: {}", e))?;

            num_rows += 1;
        }
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
