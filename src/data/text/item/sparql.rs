use anyhow::{Result, anyhow};
use oxrdf::{
    Term,
    vocab::{rdf, xsd},
};
use std::{
    fs::File,
    io::{BufReader, Read},
    mem::take,
    path::Path,
};

use sparesults::{
    QueryResultsFormat, QueryResultsParseError, QueryResultsParser, QuerySolution,
    ReaderQueryResultsParserOutput,
};

use crate::data::text::item::TextItem;

pub enum SPARQLResultFormat {
    JSON,
    XML,
    CSV,
    TSV,
}

impl From<SPARQLResultFormat> for QueryResultsFormat {
    fn from(format: SPARQLResultFormat) -> Self {
        match format {
            SPARQLResultFormat::JSON => QueryResultsFormat::Json,
            SPARQLResultFormat::XML => QueryResultsFormat::Xml,
            SPARQLResultFormat::CSV => QueryResultsFormat::Csv,
            SPARQLResultFormat::TSV => QueryResultsFormat::Tsv,
        }
    }
}

pub struct SPARQLResultIterator<I>
where
    I: IntoIterator<Item = Result<QuerySolution, QueryResultsParseError>>,
{
    identifier: Option<String>,
    fields: Vec<String>,
    inner: I,
}

impl<I> SPARQLResultIterator<I>
where
    I: IntoIterator<Item = Result<QuerySolution, QueryResultsParseError>>,
{
    pub fn new(inner: I) -> Self {
        Self {
            identifier: None,
            fields: Vec::new(),
            inner,
        }
    }
}

fn parse_solution(solution: &QuerySolution) -> Result<(String, String)> {
    if solution.len() != 2 {
        return Err(anyhow!(
            "Expected exactly 2 variables in solution, found {}",
            solution.len()
        ));
    }

    let values = solution.values();
    let Some(first) = &values[0] else {
        return Err(anyhow!("Expected first variable to be bound, found None"));
    };
    let Some(second) = &values[1] else {
        return Err(anyhow!("Expected second variable to be bound, found None"));
    };

    let Term::NamedNode(node) = first else {
        return Err(anyhow!(
            "Expected first variable to be an IRI, found {:?}",
            first
        ));
    };

    let identifier = node.as_str().to_string();

    let Term::Literal(literal) = second else {
        return Err(anyhow!(
            "Expected second variable to be a literal, found {:?}",
            second
        ));
    };

    let field = match literal.datatype() {
        xsd::STRING | rdf::LANG_STRING => literal.value().to_string(),
        _ => {
            return Err(anyhow!(
                "Expected second variable to be a string literal, found datatype {:?}",
                literal.datatype()
            ));
        }
    };

    Ok((identifier, field))
}

impl<I> Iterator for SPARQLResultIterator<I>
where
    I: Iterator<Item = Result<QuerySolution, QueryResultsParseError>>,
{
    type Item = Result<TextItem>;

    fn next(&mut self) -> Option<Self::Item> {
        for result in self.inner.by_ref() {
            let Ok(solution) = result else {
                return Some(Err(anyhow!(
                    "Failed to parse SPARQL solution: {}",
                    result.unwrap_err()
                )));
            };

            let solution = parse_solution(&solution);
            let Ok((identifier, field)) = solution else {
                return Some(Err(anyhow!(
                    "Failed to parse SPARQL solution: {}",
                    solution.unwrap_err()
                )));
            };

            if self.identifier.as_ref().is_none_or(|id| id == &identifier) {
                self.fields.push(field);
                self.identifier.get_or_insert(identifier);
                continue;
            }

            let identifier = self.identifier.replace(identifier)?;
            let fields = take(&mut self.fields);
            self.fields.push(field);
            return Some(TextItem::new(identifier, fields));
        }

        let identifier = self.identifier.take()?;
        let fields = take(&mut self.fields);

        Some(TextItem::new(identifier, fields))
    }
}

pub fn stream_text_items_from_sparql_result<R: Read>(
    reader: R,
    format: SPARQLResultFormat,
) -> Result<impl Iterator<Item = Result<TextItem>>> {
    let json_parser = QueryResultsParser::from_format(format.into());

    let parser = json_parser
        .for_reader(reader)
        .map_err(|e| anyhow!("Failed to create SPARQL JSON parser: {}", e))?;

    let ReaderQueryResultsParserOutput::Solutions(solutions) = parser else {
        return Err(anyhow!("Expected SPARQL JSON select result"));
    };

    Ok(SPARQLResultIterator::new(solutions))
}

pub fn stream_text_items_from_sparql_result_file(
    file_path: &Path,
    format: SPARQLResultFormat,
) -> Result<impl Iterator<Item = Result<TextItem>>> {
    let reader = BufReader::new(File::open(file_path)?);
    stream_text_items_from_sparql_result(reader, format)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_stream_single_identifier_multiple_fields() {
        let sparql_json = r#"{
            "head": {"vars": ["s", "label"]},
            "results": {
                "bindings": [
                    {
                        "s": {"type": "uri", "value": "http://example.org/Q1"},
                        "label": {"type": "literal", "value": "Universe"}
                    },
                    {
                        "s": {"type": "uri", "value": "http://example.org/Q1"},
                        "label": {"type": "literal", "value": "Cosmos"}
                    }
                ]
            }
        }"#;

        let cursor = Cursor::new(sparql_json);
        let items: Vec<_> = stream_text_items_from_sparql_result(cursor, SPARQLResultFormat::JSON)
            .expect("Failed to create iterator")
            .collect::<Result<Vec<_>>>()
            .expect("Failed to parse SPARQL JSON");

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].identifier, "http://example.org/Q1");
        assert_eq!(items[0].fields, vec!["Universe", "Cosmos"]);
    }

    #[test]
    fn test_stream_multiple_identifiers() {
        let sparql_json = r#"{
            "head": {"vars": ["s", "label"]},
            "results": {
                "bindings": [
                    {
                        "s": {"type": "uri", "value": "http://example.org/Q1"},
                        "label": {"type": "literal", "value": "First"}
                    },
                    {
                        "s": {"type": "uri", "value": "http://example.org/Q2"},
                        "label": {"type": "literal", "value": "Second"}
                    },
                    {
                        "s": {"type": "uri", "value": "http://example.org/Q2"},
                        "label": {"type": "literal", "value": "Another"}
                    },
                    {
                        "s": {"type": "uri", "value": "http://example.org/Q3"},
                        "label": {"type": "literal", "value": "Third"}
                    }
                ]
            }
        }"#;

        let cursor = Cursor::new(sparql_json);
        let items: Vec<_> = stream_text_items_from_sparql_result(cursor, SPARQLResultFormat::JSON)
            .expect("Failed to create iterator")
            .collect::<Result<Vec<_>>>()
            .expect("Failed to parse SPARQL JSON");

        assert_eq!(items.len(), 3);
        assert_eq!(items[0].identifier, "http://example.org/Q1");
        assert_eq!(items[0].fields, vec!["First"]);
        assert_eq!(items[1].identifier, "http://example.org/Q2");
        assert_eq!(items[1].fields, vec!["Second", "Another"]);
        assert_eq!(items[2].identifier, "http://example.org/Q3");
        assert_eq!(items[2].fields, vec!["Third"]);
    }

    #[test]
    fn test_stream_empty_bindings() {
        let sparql_json = r#"{
            "head": {"vars": ["s", "label"]},
            "results": {
                "bindings": []
            }
        }"#;

        let cursor = Cursor::new(sparql_json);
        let items: Vec<_> = stream_text_items_from_sparql_result(cursor, SPARQLResultFormat::JSON)
            .expect("Failed to create iterator")
            .collect::<Result<Vec<_>>>()
            .expect("Failed to parse SPARQL JSON");

        assert_eq!(items.len(), 0);
    }

    #[test]
    fn test_invalid_binding_wrong_number_of_vars() {
        let sparql_json = r#"{
            "head": {"vars": ["s", "label", "extra"]},
            "results": {
                "bindings": [
                    {
                        "s": {"type": "uri", "value": "http://example.org/Q1"},
                        "label": {"type": "literal", "value": "First"},
                        "extra": {"type": "literal", "value": "Extra"}
                    }
                ]
            }
        }"#;

        let cursor = Cursor::new(sparql_json);
        let result: Result<Vec<_>> =
            stream_text_items_from_sparql_result(cursor, SPARQLResultFormat::JSON)
                .expect("Failed to create iterator")
                .collect();

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Expected exactly 2 variables")
        );
    }

    #[test]
    fn test_invalid_binding_wrong_type() {
        let sparql_json = r#"{
            "head": {"vars": ["s", "label"]},
            "results": {
                "bindings": [
                    {
                        "s": {"type": "literal", "value": "NotAURI"},
                        "label": {"type": "literal", "value": "Field"}
                    }
                ]
            }
        }"#;

        let cursor = Cursor::new(sparql_json);
        let result: Result<Vec<_>> =
            stream_text_items_from_sparql_result(cursor, SPARQLResultFormat::JSON)
                .expect("Failed to create iterator")
                .collect();

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Expected first variable to be an IRI")
        );
    }
}
