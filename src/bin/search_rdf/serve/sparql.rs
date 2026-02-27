use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, anyhow};
use axum::body::Bytes;
use axum::extract::{Path as AxumPath, Query as AxumQuery, State};
use axum::http::StatusCode;
use axum::response::Response;
use log::info;
use oxrdf::vocab::xsd;
use oxrdf::{Literal, NamedNode, Term, Variable};
use search_rdf::index::Search;
use serde::Deserialize;
use serde_json::{Value, json};
use sparesults::{
    QueryResultsFormat, QueryResultsParser, QueryResultsSerializer, QuerySolution,
    ReaderQueryResultsParserOutput,
};
use spargebra::algebra::GraphPattern;
use spargebra::term::{NamedNodePattern, TermPattern};
use spargebra::{Query as SparqlQuery, SparqlParser};

use crate::search_rdf::index::{SearchIndex, SearchParams};
use crate::search_rdf::serve::search::{
    MatchInfo, Query, perform_neighbor_search, perform_neighbor_search_with_filter,
    perform_search, perform_search_with_filter,
};

use super::search::SearchMatch;
use super::types::{AppError, AppState};

fn trim_identifier(iri: &str) -> &str {
    iri.trim_start_matches('<').trim_end_matches('>')
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
    if let MatchInfo::Field { identifier, field } = search_match.info {
        values.push(Some(Term::NamedNode(
            NamedNode::new(trim_identifier(&identifier))
                .map_err(|e| anyhow!("Failed to create IRI from identifier: {e}"))?,
        )));
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

fn resolve_to_id(index: &SearchIndex, term: &Term) -> Result<Option<u32>, AppError> {
    match term {
        Term::Literal(lit) => {
            let id = lit.value().parse::<u32>().map_err(|e| {
                AppError(StatusCode::BAD_REQUEST, anyhow!("Expected integer id: {e}"))
            })?;
            Ok(Some(id))
        }
        Term::NamedNode(node) => Ok(get_id_from_identifier(index, node.as_str())),
        _ => Err(AppError(
            StatusCode::BAD_REQUEST,
            anyhow!("Expected IRI or integer literal in filter/query column"),
        )),
    }
}

fn get_id_from_identifier(index: &SearchIndex, identifier: &str) -> Option<u32> {
    match index {
        SearchIndex::Keyword(idx) => idx.data().id_from_identifier(identifier),
        SearchIndex::Fuzzy(idx) => idx.data().id_from_identifier(identifier),
        SearchIndex::FullText(idx) => idx.data().id_from_identifier(identifier),
        SearchIndex::EmbeddingWithData(idx) => idx.data().data().id_from_identifier(identifier),
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
                anyhow!("Index not found: {index_name}"),
            )
        })?;

    let model = state
        .inner
        .index_to_model
        .get(&index_name)
        .and_then(|model_name| state.inner.models.get(model_name));

    // Log the incoming request
    let body = String::from_utf8_lossy(&body);
    info!("SERVICE endpoint called for index {index_name}:\n{body}");

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
    params.insert("type".to_string(), index.index_type().to_string());

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
                    "query" => match tp.object {
                        TermPattern::Literal(lit) => {
                            query = Some(Query::Text(lit.value().to_string()));
                        }
                        TermPattern::NamedNode(iri) => {
                            query = Some(Query::Identifier(iri.as_str().to_string()));
                        }
                        _ => {
                            return Err(AppError(
                                StatusCode::BAD_REQUEST,
                                anyhow!("Expected literal or IRI for 'query' predicate"),
                            ));
                        }
                    },
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
    let params = SearchParams::deserialize(json!(params)).map_err(|e| {
        AppError(
            StatusCode::BAD_REQUEST,
            anyhow!("Failed to parse search parameters: {}", e),
        )
    })?;

    // perform search — if the query is a known identifier, do neighbor search instead
    let maybe_neighbor_id = match &query {
        Query::Identifier(iri) => get_id_from_identifier(&index, iri),
        _ => None,
    };

    let matches = if let Some(data_id) = maybe_neighbor_id {
        vec![perform_neighbor_search(index, data_id, params, model).await?]
    } else {
        match query {
            Query::Identifier(iri) => {
                return Err(AppError(
                    StatusCode::BAD_REQUEST,
                    anyhow!("IRI '{iri}' is not a known identifier in index '{index_name}'"),
                ))
            }
            query => perform_search(index, vec![query], params, model).await?,
        }
    };

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
        if let MatchInfo::Field { identifier, field } = search_match.info {
            identifier_term = Some(Term::NamedNode(
                NamedNode::new(trim_identifier(&identifier))
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

fn serialize_identifier_mode_matches(
    pairs: Vec<(Term, Vec<SearchMatch>)>,
    variables: &[Variable],
) -> Result<Vec<u8>> {
    let mut buffer = Vec::new();
    let mut serializer = QueryResultsSerializer::from_format(QueryResultsFormat::Json)
        .serialize_solutions_to_writer(&mut buffer, variables.to_vec())
        .map_err(|e| anyhow!("Failed to create serializer: {}", e))?;
    let mut rank = 1usize;
    for (row_term, matches) in pairs {
        for search_match in matches {
            let solution =
                search_match_to_solution(search_match, variables, &[Some(row_term.clone())], rank)?;
            serializer.serialize(&solution).map_err(|e| anyhow!("{e}"))?;
            rank += 1;
        }
    }
    serializer.finish().map_err(|e| anyhow!("{e}"))?;
    Ok(buffer)
}

#[derive(Debug, Deserialize)]
pub struct QlProxyParams {
    // Search parameters (query is optional; if absent, query body column or text is used)
    query: Option<String>,
    #[serde(default = "default_rowvar")]
    rowvar: String,
    #[serde(flatten)]
    params: HashMap<String, Value>,
}

fn default_rowvar() -> String {
    "row".to_string()
}

pub async fn qlproxy(
    AxumPath(index_name): AxumPath<String>,
    State(state): State<AppState>,
    AxumQuery(mut params): AxumQuery<QlProxyParams>,
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
    // Parse input bindings - build id_to_row (filter column) and query_items (query column)
    let mut id_to_row: HashMap<u32, Term> = HashMap::new();
    let mut query_items: Vec<(Term, u32)> = Vec::new();
    let mut has_filter_column = false;
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

        // Optional `filter` column — constrains the search space
        if let Some(term) = solution.get("filter") {
            has_filter_column = true;
            if let Some(id) = resolve_to_id(&index, term)? {
                id_to_row.insert(id, row_binding.clone());
            }
        }

        // Optional `query` column — identifier mode: one neighbor search per item
        if let Some(term) = solution.get("query") {
            match term {
                Term::NamedNode(node) => {
                    if let Some(data_id) = get_id_from_identifier(&index, node.as_str()) {
                        query_items.push((row_binding.clone(), data_id));
                    }
                }
                _ => {
                    return Err(AppError(
                        StatusCode::BAD_REQUEST,
                        anyhow!("Expected IRI in 'query' body column"),
                    ));
                }
            }
        }
    }

    info!(
        "Parsed {} filter entries and {} query items in {}ms",
        id_to_row.len(),
        query_items.len(),
        start.elapsed().as_millis()
    );

    // Conflict: cannot specify both HTTP query param and query body column
    if params.query.is_some() && !query_items.is_empty() {
        return Err(AppError(
            StatusCode::BAD_REQUEST,
            anyhow!("Cannot specify both 'query' HTTP parameter and 'query' body column"),
        ));
    }

    // parse params
    params.params.insert(
        "type".to_string(),
        Value::String(index.index_type().to_string()),
    );

    let search_params = SearchParams::deserialize(json!(params.params)).map_err(|e| {
        AppError(
            StatusCode::BAD_REQUEST,
            anyhow!("Failed to parse search parameters: {}", e),
        )
    })?;

    // Determine output variables
    let variables = vec![
        Variable::new_unchecked(params.rowvar),
        Variable::new_unchecked("rank"),
        Variable::new_unchecked("id"),
        Variable::new_unchecked("score"),
        Variable::new_unchecked("identifier"),
        Variable::new_unchecked("field"),
    ];

    let start = Instant::now();
    let response = if !query_items.is_empty() {
        // Identifier mode: loop over query items, one neighbor search per item
        // If no filter column was present at all, skip filtering entirely (pass-through).
        // An empty filter (column present but all IRIs unknown) still restricts to zero results.
        let filter_ids: Arc<HashSet<u32>> = Arc::new(id_to_row.keys().copied().collect());
        let mut identifier_results: Vec<(Term, Vec<SearchMatch>)> = Vec::new();
        for (row_binding, data_id) in query_items {
            let matches = if has_filter_column {
                let fids = filter_ids.clone();
                perform_neighbor_search_with_filter(
                    index.clone(),
                    data_id,
                    search_params.clone(),
                    model,
                    move |id| fids.contains(&id),
                )
                .await?
            } else {
                perform_neighbor_search(index.clone(), data_id, search_params.clone(), model)
                    .await?
            };
            identifier_results.push((row_binding, matches));
        }
        serialize_identifier_mode_matches(identifier_results, &variables)?
    } else {
        // Text mode: single search with filter
        let query_str = params.query.ok_or_else(|| {
            AppError(
                StatusCode::BAD_REQUEST,
                anyhow!("Missing required 'query' HTTP parameter"),
            )
        })?;

        let is_iri = NamedNode::new(&query_str).is_ok();
        let maybe_neighbor_id = if is_iri {
            get_id_from_identifier(&index, &query_str)
        } else {
            None
        };

        let id_to_row = Arc::new(id_to_row);

        let matches = if let Some(data_id) = maybe_neighbor_id {
            vec![if has_filter_column {
                let id_to_row_filter = id_to_row.clone();
                perform_neighbor_search_with_filter(
                    index,
                    data_id,
                    search_params,
                    model,
                    move |id| id_to_row_filter.contains_key(&id),
                )
                .await?
            } else {
                perform_neighbor_search(index, data_id, search_params, model).await?
            }]
        } else if is_iri {
            return Err(AppError(
                StatusCode::BAD_REQUEST,
                anyhow!("IRI '{query_str}' not known in index"),
            ));
        } else if has_filter_column {
            let id_to_row_filter = id_to_row.clone();
            perform_search_with_filter(
                index,
                vec![Query::Text(query_str)],
                search_params,
                model,
                move |id| id_to_row_filter.contains_key(&id),
            )
            .await?
        } else {
            perform_search(index, vec![Query::Text(query_str)], search_params, model).await?
        };

        let row_fn = move |id: u32| {
            id_to_row
                .get(&id)
                .cloned()
                .ok_or_else(|| anyhow!("No row for id {id}"))
        };
        serialize_search_matches(matches, &variables, row_fn)?
    };

    info!("Search completed in {}ms", start.elapsed().as_millis());

    // Return as response with correct content type
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/sparql-results+json")
        .body(response.into())
        .map_err(|e| anyhow!("Failed to build response: {}", e))?)
}
