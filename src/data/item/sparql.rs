use anyhow::{Result, anyhow};
use log::warn;
use oxrdf::{
    Term,
    vocab::{rdf, xsd},
};
use serde::{Deserialize, Serialize};
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

use crate::data::item::{FieldType, Item, StringField};

#[derive(Default, Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SPARQLResultFormat {
    #[default]
    JSON,
    XML,
    TSV,
}

impl SPARQLResultFormat {
    pub fn mime_type(&self) -> &str {
        match self {
            SPARQLResultFormat::JSON => "application/sparql-results+json",
            SPARQLResultFormat::XML => "application/sparql-results+xml",
            SPARQLResultFormat::TSV => "text/tab-separated-values",
        }
    }
}

impl From<SPARQLResultFormat> for QueryResultsFormat {
    fn from(format: SPARQLResultFormat) -> Self {
        match format {
            SPARQLResultFormat::JSON => QueryResultsFormat::Json,
            SPARQLResultFormat::XML => QueryResultsFormat::Xml,
            SPARQLResultFormat::TSV => QueryResultsFormat::Tsv,
        }
    }
}

pub struct SPARQLResultIterator<I>
where
    I: IntoIterator<Item = Result<QuerySolution, QueryResultsParseError>>,
{
    identifier: Option<String>,
    fields: Vec<StringField>,
    default_field_type: FieldType,
    inner: I,
}

impl<I> SPARQLResultIterator<I>
where
    I: IntoIterator<Item = Result<QuerySolution, QueryResultsParseError>>,
{
    pub fn new(inner: I, default_field_type: FieldType) -> Self {
        Self {
            identifier: None,
            fields: Vec::new(),
            default_field_type,
            inner,
        }
    }
}

fn parse_solution(
    solution: &QuerySolution,
    default_field_type: FieldType,
) -> Result<(String, StringField)> {
    if solution.len() != 2 && solution.len() != 3 {
        return Err(anyhow!(
            "Expected 2 or 3 variables in solution, found {}",
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

    let value = match second {
        Term::Literal(literal) if [xsd::STRING, rdf::LANG_STRING].contains(&literal.datatype()) => {
            literal.value().to_string()
        }
        Term::NamedNode(iri) => iri.as_str().to_string(),
        _ => {
            return Err(anyhow!(
                "Expected second variable to be a string literal or IRI, found {:?}",
                second
            ));
        }
    };

    let field_type = if let Some(Some(third)) = values.get(2) {
        let Term::Literal(type_literal) = third else {
            return Err(anyhow!(
                "Expected third variable (type) to be a string literal, found {:?}",
                third
            ));
        };
        serde_plain::from_str(type_literal.value())?
    } else {
        default_field_type
    };

    Ok((identifier, StringField { value, field_type }))
}

impl<I> Iterator for SPARQLResultIterator<I>
where
    I: Iterator<Item = Result<QuerySolution, QueryResultsParseError>>,
{
    type Item = Result<Item>;

    fn next(&mut self) -> Option<Self::Item> {
        for result in self.inner.by_ref() {
            let Ok(solution) = result else {
                warn!(
                    "Unexpected error from SPARQL result iterator: {}",
                    result.unwrap_err()
                );
                continue;
            };

            let (identifier, field) = match parse_solution(&solution, self.default_field_type) {
                Ok(sol) => sol,
                Err(e) => return Some(Err(anyhow!("Failed to parse SPARQL solution: {}", e))),
            };

            if self.identifier.as_ref().is_none_or(|id| id == &identifier) {
                self.fields.push(field);
                self.identifier.get_or_insert(identifier);
                continue;
            }

            let identifier = self.identifier.replace(identifier)?;
            let fields = take(&mut self.fields);
            self.fields.push(field);
            return Some(Item::from_string_fields(identifier, fields));
        }

        let identifier = self.identifier.take()?;
        let fields = take(&mut self.fields);

        Some(Item::from_string_fields(identifier, fields))
    }
}

pub fn stream_items_from_sparql_result<R: Read>(
    reader: R,
    format: SPARQLResultFormat,
    default_field_type: FieldType,
) -> Result<impl Iterator<Item = Result<Item>>> {
    let json_parser = QueryResultsParser::from_format(format.into());

    let parser = json_parser
        .for_reader(reader)
        .map_err(|e| anyhow!("Failed to create SPARQL result parser: {}", e))?;

    let ReaderQueryResultsParserOutput::Solutions(solutions) = parser else {
        return Err(anyhow!("Expected SPARQL result in {:?} format", format));
    };

    Ok(SPARQLResultIterator::new(solutions, default_field_type))
}

pub fn stream_items_from_sparql_result_file(
    file_path: &Path,
    format: SPARQLResultFormat,
    default_field_type: FieldType,
) -> Result<impl Iterator<Item = Result<Item>>> {
    let reader = BufReader::new(File::open(file_path)?);
    stream_items_from_sparql_result(reader, format, default_field_type)
}

pub fn guess_sparql_result_format_from_extension(file_path: &Path) -> Result<SPARQLResultFormat> {
    let ext = file_path
        .extension()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow!("File has no extension"))?
        .to_lowercase();

    let format = match ext.as_str() {
        "json" => SPARQLResultFormat::JSON,
        "xml" => SPARQLResultFormat::XML,
        "tsv" => SPARQLResultFormat::TSV,
        _ => {
            return Err(anyhow!(
                "Could not guess SPARQL result format from file extension: {}",
                ext
            ));
        }
    };

    Ok(format)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::item::Field;
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
        let items: Vec<_> =
            stream_items_from_sparql_result(cursor, SPARQLResultFormat::JSON, FieldType::Text)
                .expect("Failed to create iterator")
                .collect::<Result<Vec<_>>>()
                .expect("Failed to parse SPARQL JSON");

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].identifier, "http://example.org/Q1");
        assert_eq!(items[0].num_fields(), 2);
        assert_eq!(items[0].fields[0], Field::Text("Universe".to_string()));
        assert_eq!(items[0].fields[1], Field::Text("Cosmos".to_string()));
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
        let items: Vec<_> =
            stream_items_from_sparql_result(cursor, SPARQLResultFormat::JSON, FieldType::Text)
                .expect("Failed to create iterator")
                .collect::<Result<Vec<_>>>()
                .expect("Failed to parse SPARQL JSON");

        assert_eq!(items.len(), 3);
        assert_eq!(items[0].identifier, "http://example.org/Q1");
        assert_eq!(items[0].num_fields(), 1);
        assert_eq!(items[0].fields[0], Field::Text("First".to_string()));
        assert_eq!(items[1].identifier, "http://example.org/Q2");
        assert_eq!(items[1].num_fields(), 2);
        assert_eq!(items[1].fields[0], Field::Text("Second".to_string()));
        assert_eq!(items[1].fields[1], Field::Text("Another".to_string()));
        assert_eq!(items[2].identifier, "http://example.org/Q3");
        assert_eq!(items[2].num_fields(), 1);
        assert_eq!(items[2].fields[0], Field::Text("Third".to_string()));
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
        let items: Vec<_> =
            stream_items_from_sparql_result(cursor, SPARQLResultFormat::JSON, FieldType::Text)
                .expect("Failed to create iterator")
                .collect::<Result<Vec<_>>>()
                .expect("Failed to parse SPARQL JSON");

        assert_eq!(items.len(), 0);
    }

    #[test]
    fn test_invalid_binding_wrong_number_of_vars() {
        let sparql_json = r#"{
            "head": {"vars": ["s", "label", "extra", "extra2"]},
            "results": {
                "bindings": [
                    {
                        "s": {"type": "uri", "value": "http://example.org/Q1"},
                        "label": {"type": "literal", "value": "First"},
                        "extra": {"type": "literal", "value": "Extra"},
                        "extra2": {"type": "literal", "value": "Extra2"}
                    }
                ]
            }
        }"#;

        let cursor = Cursor::new(sparql_json);
        let result: Result<Vec<_>> =
            stream_items_from_sparql_result(cursor, SPARQLResultFormat::JSON, FieldType::Text)
                .expect("Failed to create iterator")
                .collect();

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Expected 2 or 3 variables")
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
            stream_items_from_sparql_result(cursor, SPARQLResultFormat::JSON, FieldType::Text)
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

    #[test]
    fn test_stream_xml_format_single_identifier() {
        let sparql_xml = r#"<?xml version="1.0"?>
<sparql xmlns="http://www.w3.org/2005/sparql-results#">
  <head>
    <variable name="s"/>
    <variable name="label"/>
  </head>
  <results>
    <result>
      <binding name="s"><uri>http://example.org/Q1</uri></binding>
      <binding name="label"><literal>Universe</literal></binding>
    </result>
    <result>
      <binding name="s"><uri>http://example.org/Q1</uri></binding>
      <binding name="label"><literal>Cosmos</literal></binding>
    </result>
  </results>
</sparql>"#;

        let cursor = Cursor::new(sparql_xml);
        let items: Vec<_> =
            stream_items_from_sparql_result(cursor, SPARQLResultFormat::XML, FieldType::Text)
                .expect("Failed to create iterator")
                .collect::<Result<Vec<_>>>()
                .expect("Failed to parse SPARQL XML");

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].identifier, "http://example.org/Q1");
        assert_eq!(items[0].num_fields(), 2);
        assert_eq!(items[0].fields[0], Field::Text("Universe".to_string()));
        assert_eq!(items[0].fields[1], Field::Text("Cosmos".to_string()));
    }

    #[test]
    fn test_stream_xml_format_multiple_identifiers() {
        let sparql_xml = r#"<?xml version="1.0"?>
<sparql xmlns="http://www.w3.org/2005/sparql-results#">
  <head>
    <variable name="s"/>
    <variable name="label"/>
  </head>
  <results>
    <result>
      <binding name="s"><uri>http://example.org/Q1</uri></binding>
      <binding name="label"><literal>First</literal></binding>
    </result>
    <result>
      <binding name="s"><uri>http://example.org/Q2</uri></binding>
      <binding name="label"><literal>Second</literal></binding>
    </result>
    <result>
      <binding name="s"><uri>http://example.org/Q2</uri></binding>
      <binding name="label"><literal>Another</literal></binding>
    </result>
    <result>
      <binding name="s"><uri>http://example.org/Q3</uri></binding>
      <binding name="label"><literal>Third</literal></binding>
    </result>
  </results>
</sparql>"#;

        let cursor = Cursor::new(sparql_xml);
        let items: Vec<_> =
            stream_items_from_sparql_result(cursor, SPARQLResultFormat::XML, FieldType::Text)
                .expect("Failed to create iterator")
                .collect::<Result<Vec<_>>>()
                .expect("Failed to parse SPARQL XML");

        assert_eq!(items.len(), 3);
        assert_eq!(items[0].identifier, "http://example.org/Q1");
        assert_eq!(items[0].num_fields(), 1);
        assert_eq!(items[0].fields[0], Field::Text("First".to_string()));
        assert_eq!(items[1].identifier, "http://example.org/Q2");
        assert_eq!(items[1].num_fields(), 2);
        assert_eq!(items[1].fields[0], Field::Text("Second".to_string()));
        assert_eq!(items[1].fields[1], Field::Text("Another".to_string()));
        assert_eq!(items[2].identifier, "http://example.org/Q3");
        assert_eq!(items[2].num_fields(), 1);
        assert_eq!(items[2].fields[0], Field::Text("Third".to_string()));
    }

    #[test]
    fn test_stream_tsv_format() {
        let sparql_tsv = "?s\t?label\n\
            <http://example.org/Q1>\t\"Universe\"\n\
            <http://example.org/Q1>\t\"Cosmos\"\n\
            <http://example.org/Q2>\t\"Earth\"";

        let cursor = Cursor::new(sparql_tsv);
        let items: Vec<_> =
            stream_items_from_sparql_result(cursor, SPARQLResultFormat::TSV, FieldType::Text)
                .expect("Failed to create iterator")
                .collect::<Result<Vec<_>>>()
                .expect("Failed to parse SPARQL TSV");

        assert_eq!(items.len(), 2);
        assert_eq!(items[0].identifier, "http://example.org/Q1");
        assert_eq!(items[0].num_fields(), 2);
        assert_eq!(items[0].fields[0], Field::Text("Universe".to_string()));
        assert_eq!(items[0].fields[1], Field::Text("Cosmos".to_string()));
        assert_eq!(items[1].identifier, "http://example.org/Q2");
        assert_eq!(items[1].num_fields(), 1);
        assert_eq!(items[1].fields[0], Field::Text("Earth".to_string()));
    }
}
