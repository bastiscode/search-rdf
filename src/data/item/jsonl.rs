use anyhow::Result;
use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
};

use crate::data::item::Item;

pub fn stream_items_from_jsonl<R: Read>(reader: R) -> Result<impl Iterator<Item = Result<Item>>> {
    let buffered = BufReader::new(reader);
    let iter = buffered.lines().map(|line| {
        let line = line?;
        let item: Item = serde_json::from_str(&line)?;
        item.validate()
    });
    Ok(iter)
}

pub fn stream_items_from_jsonl_file(
    file_path: &Path,
) -> Result<impl Iterator<Item = Result<Item>>> {
    let reader = BufReader::new(File::open(file_path)?);
    stream_items_from_jsonl(reader)
}

#[test]
fn test_stream_items_from_jsonl() {
    use std::io::Cursor;

    let jsonl_data = r#"{"identifier":"Q1","fields":["Universe","Cosmos"]}
{"identifier":"Q2","fields":["Earth","World"]}
{"identifier":"Q3","fields":["Human"]}"#;

    let cursor = Cursor::new(jsonl_data);
    let items: Vec<_> = stream_items_from_jsonl(cursor)
        .expect("Failed to create iterator")
        .collect::<Result<Vec<_>>>()
        .expect("Failed to parse JSONL");

    assert_eq!(items.len(), 3);
    assert_eq!(items[0].identifier, "Q1");
    assert_eq!(items[0].fields, vec!["Universe", "Cosmos"]);
    assert_eq!(items[1].identifier, "Q2");
    assert_eq!(items[1].fields, vec!["Earth", "World"]);
    assert_eq!(items[2].identifier, "Q3");
    assert_eq!(items[2].fields, vec!["Human"]);
}

#[test]
fn test_stream_items_from_jsonl_file() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    writeln!(
        temp_file,
        r#"{{"identifier":"Q42","fields":["Douglas Adams","Author"]}}"#
    )
    .expect("Failed to write");
    writeln!(
        temp_file,
        r#"{{"identifier":"Q100","fields":["Test","Example","Demo"]}}"#
    )
    .expect("Failed to write");

    let items: Vec<_> = stream_items_from_jsonl_file(temp_file.path())
        .expect("Failed to create iterator")
        .collect::<Result<Vec<_>>>()
        .expect("Failed to read JSONL file");

    assert_eq!(items.len(), 2);
    assert_eq!(items[0].identifier, "Q42");
    assert_eq!(items[0].fields, vec!["Douglas Adams", "Author"]);
    assert_eq!(items[1].identifier, "Q100");
    assert_eq!(items[1].fields, vec!["Test", "Example", "Demo"]);
}
