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
use oxrdf::{Literal, NamedNode, Term, Variable};
use search_rdf::index::Search;
use serde::Deserialize;
use sparesults::{
    QueryResultsFormat, QueryResultsParser, QueryResultsSerializer, QuerySolution,
    ReaderQueryResultsParserOutput,
};
use spargebra::algebra::GraphPattern;
use spargebra::term::{NamedNodePattern, TermPattern};
use spargebra::{Query as SparqlQuery, SparqlParser};

use crate::search_rdf::index::{SearchIndex, SearchParams};
use crate::search_rdf::serve::search::{
    MatchInfo, perform_text_search, perform_text_search_with_filter,
};

use super::search::SearchMatch;
use super::types::{AppError, AppState};

#[derive(Debug, Deserialize)]
pub struct QlProxyParams {
    // Search parameters (query is required)
    query: String,
    #[serde(default = "default_rowvar")]
    rowvar: String,
    #[serde(flatten)]
    params: SearchParams,
}

fn default_rowvar() -> String {
    "row".to_string()
}

/// Convert a SearchMatch to SPARQL bindings based on output variable configuration
/// Returns a tuple of (variables, values) that can be used to create a QuerySolution
fn search_match_to_solution(
    search_match: SearchMatch,
    variables: &[Variable],
    values: &[Option<Term>],
    rank: usize,
) -> Result<QuerySolution> {
    let mut values = values.to_vec();
    values.extend([
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
    ]);

    // Extract identifier and field from MatchInfo if available
    if let MatchInfo::Text { identifier, field } = search_match.info {
        values.push(Some(Term::NamedNode(NamedNode::new(identifier).map_err(
            |e| anyhow!("Failed to create IRI from identifier: {e}"),
        )?)));
        values.push(Some(Term::Literal(Literal::new_typed_literal(
            field,
            xsd::STRING,
        ))));
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
    State(state): State<AppState>,
    body: Bytes,
) -> Result<Response, AppError> {
    let sparql_config = state
        .inner
        .sparql
        .as_ref()
        .ok_or_else(|| anyhow!("No SPARQL configuration found in server"))?;

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

    // Log the incoming request
    let body = String::from_utf8_lossy(&body);
    info!(
        "SERVICE endpoint called for index {}:\n{}",
        index_name, body
    );

    let SparqlQuery::Select { pattern, .. } =
        SparqlParser::new().parse_query(&body).map_err(|e| {
            AppError(
                StatusCode::BAD_REQUEST,
                anyhow!("Failed to parse SERVICE clause body: {e}"),
            )
        })?
    else {
        return Err(AppError(
            StatusCode::BAD_REQUEST,
            anyhow!("Only SELECT queries are supported in SERVICE endpoint"),
        ));
    };

    let GraphPattern::Project { inner: pattern, .. } = pattern else {
        return Err(AppError(
            StatusCode::BAD_REQUEST,
            anyhow!("Invalid GRAPH PATTERN in SERVICE endpoint"),
        ));
    };

    let GraphPattern::Bgp { patterns } = *pattern else {
        return Err(AppError(
            StatusCode::BAD_REQUEST,
            anyhow!("Only basic triple patterns are supported in this SERVICE endpoint"),
        ));
    };

    let mut query = None;
    let mut variables = HashMap::new();
    let mut params = HashMap::new();

    for tp in patterns {
        match &tp.subject {
            TermPattern::NamedNode(iri) => {
                let Some("config") = iri.as_str().strip_prefix(&sparql_config.prefix) else {
                    return Err(AppError(
                        StatusCode::BAD_REQUEST,
                        anyhow!("Subject IRI must always be {}config", sparql_config.prefix),
                    ));
                };
            }
            _ => {
                return Err(AppError(
                    StatusCode::BAD_REQUEST,
                    anyhow!("Subject in pattern {tp} is not an IRI"),
                ));
            }
        }

        match &tp.predicate {
            NamedNodePattern::NamedNode(iri) => {
                let pred = iri
                    .as_str()
                    .strip_prefix(&sparql_config.prefix)
                    .ok_or_else(|| {
                        AppError(
                            StatusCode::BAD_REQUEST,
                            anyhow!(
                                "Predicate IRI must use prefix {}: {}",
                                sparql_config.prefix,
                                iri.as_str()
                            ),
                        )
                    })?;

                match pred {
                    "query" => {
                        if let TermPattern::Literal(lit) = tp.object {
                            query = Some(lit.value().to_string());
                        } else {
                            return Err(AppError(
                                StatusCode::BAD_REQUEST,
                                anyhow!("Expected literal for 'query' predicate"),
                            ));
                        }
                    }
                    name @ ("id" | "field" | "identifier" | "score" | "rank") => {
                        if let TermPattern::Variable(var) = tp.object {
                            variables.insert(name.to_string(), var);
                        } else {
                            return Err(AppError(
                                StatusCode::BAD_REQUEST,
                                anyhow!("Expected literal for '{name}' predicate"),
                            ));
                        }
                    }
                    param => {
                        if let TermPattern::Literal(lit) = tp.object {
                            params.insert(param.to_string(), lit.value().to_string());
                        } else {
                            return Err(AppError(
                                StatusCode::BAD_REQUEST,
                                anyhow!("Expected literal for '{param}' predicate"),
                            ));
                        }
                    }
                }
            }
            _ => {
                return Err(AppError(
                    StatusCode::BAD_REQUEST,
                    anyhow!("Predicate in pattern {tp} is not an IRI"),
                ));
            }
        }
    }

    let Some(query) = query else {
        return Err(AppError(
            StatusCode::BAD_REQUEST,
            anyhow!("Missing required 'query' config parameter"),
        ));
    };

    if !variables.contains_key("identifier") {
        return Err(AppError(
            StatusCode::BAD_REQUEST,
            anyhow!("Missing required 'identifier' variable binding"),
        ));
    }

    let vars: Vec<_> = ["rank", "id", "score", "identifier", "field"]
        .into_iter()
        .filter_map(|name| variables.get(name).cloned())
        .collect();

    // convert params
    let params = params.try_into().map_err(|e| {
        AppError(
            StatusCode::BAD_REQUEST,
            anyhow!("Failed to parse search parameters: {}", e),
        )
    })?;

    // perform search
    let matches = perform_text_search(index, vec![query], params, model).await?;

    // write results
    let mut buffer = Vec::new();
    let mut serializer = QueryResultsSerializer::from_format(QueryResultsFormat::Json)
        .serialize_solutions_to_writer(&mut buffer, variables.values().cloned().collect())
        .map_err(|e| anyhow!("Failed to create serializer: {}", e))?;

    for (rank, search_match) in matches.into_iter().flatten().enumerate() {
        let mut values = vec![];
        if variables.contains_key("rank") {
            values.push(Some(Term::Literal(Literal::new_typed_literal(
                (rank + 1).to_string(),
                xsd::INTEGER,
            ))));
        }

        if variables.contains_key("id") {
            values.push(Some(Term::Literal(Literal::new_typed_literal(
                search_match.id.to_string(),
                xsd::INTEGER,
            ))));
        }

        if variables.contains_key("score") {
            values.push(Some(Term::Literal(Literal::new_typed_literal(
                search_match.score.to_string(),
                xsd::FLOAT,
            ))));
        }

        // Extract identifier and field from MatchInfo if available
        let mut identifier_term = None;
        let mut field_term = None;
        if let MatchInfo::Text { identifier, field } = search_match.info {
            identifier_term = Some(Term::NamedNode(
                NamedNode::new(identifier)
                    .map_err(|e| anyhow!("Failed to create IRI from identifier: {e}"))?,
            ));

            if variables.contains_key("field") {
                field_term = Some(Term::Literal(Literal::new_simple_literal(field)));
            }
        }

        if identifier_term.is_none() {
            return Err(AppError(
                StatusCode::BAD_REQUEST,
                anyhow!(
                    "Missing identifier in search match, the index may not support identifier retrieval"
                ),
            ));
        }

        values.push(identifier_term);

        if field_term.is_some() {
            values.push(field_term);
        }

        let solution = QuerySolution::from((vars.as_ref(), values));
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
        .body(buffer.into())
        .map_err(|e| anyhow!("Failed to build response: {}", e))?)
}

fn serialize_search_matches(
    matches: Vec<Vec<SearchMatch>>,
    variables: &[Variable],
    row_fn: impl Fn(u32) -> Result<Term>,
) -> Result<Vec<u8>> {
    let mut buffer = Vec::new();
    let mut serializer = QueryResultsSerializer::from_format(QueryResultsFormat::Json)
        .serialize_solutions_to_writer(&mut buffer, variables.to_vec())
        .map_err(|e| anyhow!("Failed to create serializer: {}", e))?;

    for (rank, search_match) in matches.into_iter().flatten().enumerate() {
        let row = row_fn(search_match.id)?;
        let solution = search_match_to_solution(search_match, variables, &[Some(row)], rank + 1)?;

        serializer
            .serialize(&solution)
            .map_err(|e| anyhow!("Failed to serialize solution: {}", e))?;
    }

    serializer
        .finish()
        .map_err(|e| anyhow!("Failed to finish serialization: {}", e))?;

    Ok(buffer)
}

pub async fn qlproxy(
    AxumPath(index_name): AxumPath<String>,
    State(state): State<AppState>,
    Query(params): Query<QlProxyParams>,
    body: Bytes,
) -> Result<Response, AppError> {
    info!(
        "QL Proxy endpoint called for index {}:\n{:?}",
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

    // Make thread-safe id_to_row map for filtering
    let id_to_row = Arc::new(id_to_row);
    let id_to_row_clone = id_to_row.clone();
    let filter = move |id| id_to_row_clone.contains_key(&id);

    // Always perform filtered search
    let start = Instant::now();
    let matches =
        perform_text_search_with_filter(index, vec![params.query], params.params, model, filter)
            .await?;

    info!("Search completed in {}ms", start.elapsed().as_millis());

    // Determine output variables
    let variables = vec![
        Variable::new_unchecked(params.rowvar),
        Variable::new_unchecked("rank"),
        Variable::new_unchecked("id"),
        Variable::new_unchecked("score"),
        Variable::new_unchecked("identifier"),
        Variable::new_unchecked("field"),
    ];

    // Custom values
    let row_fn = |id: u32| {
        // Get the row binding from id_to_row map
        id_to_row
            .get(&id)
            .cloned()
            .ok_or_else(|| anyhow!("No row found for identifier ID {} in search match", id))
    };

    let response = serialize_search_matches(matches, &variables, row_fn)?;

    // Return as response with correct content type
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/sparql-results+json")
        .body(response.into())
        .map_err(|e| anyhow!("Failed to build response: {}", e))?)
}
