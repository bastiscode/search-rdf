use anyhow::Result;
use serde::Deserialize;
use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
};

use crate::data::item::{Item, StringField};

#[derive(Deserialize)]
struct ItemJson {
    identifier: String,
    fields: Vec<StringField>,
}

impl TryFrom<ItemJson> for Item {
    type Error = anyhow::Error;

    fn try_from(item_json: ItemJson) -> Result<Self> {
        Item::from_string_fields(item_json.identifier, item_json.fields)
    }
}

pub fn stream_items_from_jsonl<R: Read>(reader: R) -> Result<impl Iterator<Item = Result<Item>>> {
    let buffered = BufReader::new(reader);
    let iter = buffered.lines().map(|line| {
        let line = line?;
        let item_json: ItemJson = serde_json::from_str(&line)?;
        item_json.try_into()
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
    use crate::data::item::Field;
    use std::io::Cursor;

    let jsonl_data = r#"{"identifier":"Q1","fields":[{"type":"text","value":"Universe"},{"type":"text","value":"Cosmos"}]}
{"identifier":"Q2","fields":[{"type":"text","value":"Earth"},{"type":"text","value":"World"}]}
{"identifier":"Q3","fields":[{"type":"text","value":"Human"}]}"#;

    let cursor = Cursor::new(jsonl_data);
    let items: Vec<_> = stream_items_from_jsonl(cursor)
        .expect("Failed to create iterator")
        .collect::<Result<Vec<_>>>()
        .expect("Failed to parse JSONL");

    assert_eq!(items.len(), 3);
    assert_eq!(items[0].identifier, "Q1");
    assert_eq!(items[0].num_fields(), 2);
    assert_eq!(items[0].fields[0], Field::text("Universe"));
    assert_eq!(items[0].fields[1], Field::text("Cosmos"));

    assert_eq!(items[1].identifier, "Q2");
    assert_eq!(items[1].num_fields(), 2);
    assert_eq!(items[1].fields[0], Field::text("Earth"));
    assert_eq!(items[1].fields[1], Field::text("World"));

    assert_eq!(items[2].identifier, "Q3");
    assert_eq!(items[2].num_fields(), 1);
    assert_eq!(items[2].fields[0], Field::text("Human"));
}

#[test]
fn test_stream_items_from_jsonl_file() {
    use crate::data::item::Field;
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    writeln!(
        temp_file,
        r#"{{"identifier":"Q42","fields":[{{"type":"text","value":"Douglas Adams"}},{{"type":"text","value":"Author"}}]}}"#
    )
    .expect("Failed to write");
    writeln!(
        temp_file,
        r#"{{"identifier":"Q100","fields":[{{"type":"text","value":"Test"}},{{"type":"text","value":"Example"}},{{"type":"text","value":"Demo"}}]}}"#
    )
    .expect("Failed to write");

    let items: Vec<_> = stream_items_from_jsonl_file(temp_file.path())
        .expect("Failed to create iterator")
        .collect::<Result<Vec<_>>>()
        .expect("Failed to read JSONL file");

    assert_eq!(items.len(), 2);
    assert_eq!(items[0].identifier, "Q42");
    assert_eq!(items[0].num_fields(), 2);
    assert_eq!(items[0].fields[0], Field::text("Douglas Adams"));
    assert_eq!(items[0].fields[1], Field::text("Author"));

    assert_eq!(items[1].identifier, "Q100");
    assert_eq!(items[1].num_fields(), 3);
    assert_eq!(items[1].fields[0], Field::text("Test"));
    assert_eq!(items[1].fields[1], Field::text("Example"));
    assert_eq!(items[1].fields[2], Field::text("Demo"));
}
