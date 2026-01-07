use anyhow::{Result, anyhow};
use ndarray::Array3;
use serde::{Deserialize, Serialize};
use std::mem::size_of;

pub mod jsonl;
pub mod sparql;

const U16_SIZE: usize = size_of::<u16>();
const U64_SIZE: usize = size_of::<u64>();

#[derive(Clone, Debug)]
pub struct Item {
    pub identifier: String,
    pub fields: Vec<Field>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct StringField {
    #[serde(default, rename = "type")]
    pub field_type: FieldType,
    pub value: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Field {
    Text(String),
    ImageInline { url: String, data: Vec<u8> },
    Image(String),
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FieldType {
    #[default]
    Text,
    Image,
    #[serde(rename = "image-inline")]
    ImageInline,
}

impl Item {
    pub fn from_string_fields(identifier: String, fields: Vec<StringField>) -> Result<Self> {
        if identifier.len() > u16::MAX as usize {
            return Err(anyhow!("Identifier too long, max length is {}", u16::MAX));
        }
        if fields.len() > u16::MAX as usize {
            return Err(anyhow!("Too many fields, max is {}", u16::MAX));
        }

        let fields = fields
            .into_iter()
            .map(|field| field.field_type.create_field(field.value))
            .collect::<Result<_>>()?;

        Ok(Item { identifier, fields })
    }

    pub fn num_fields(&self) -> u16 {
        self.fields.len() as u16
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        let id_bytes = self.identifier.as_bytes();
        bytes.extend_from_slice(&(id_bytes.len() as u16).to_le_bytes());
        bytes.extend_from_slice(id_bytes);

        bytes.extend_from_slice(&(self.fields.len() as u16).to_le_bytes());
        for field in &self.fields {
            field.encode_into(&mut bytes);
        }

        bytes
    }
}

impl Field {
    fn encode_into(&self, bytes: &mut Vec<u8>) {
        match self {
            Field::Text(text) => {
                bytes.push(0u8);
                encode_string(bytes, text);
            }
            Field::ImageInline { url, data } => {
                bytes.push(1u8);
                encode_string(bytes, url);
                bytes.extend_from_slice(&(data.len() as u64).to_le_bytes());
                bytes.extend_from_slice(data);
            }
            Field::Image(url) => {
                bytes.push(2u8);
                encode_string(bytes, url);
            }
        }
    }
}

impl FieldType {
    pub fn create_field(&self, value: String) -> Result<Field> {
        match self {
            FieldType::Text => Ok(Field::Text(value)),
            FieldType::Image => Ok(Field::Image(value)),
            FieldType::ImageInline => {
                let data = load_bytes_from_url(&value)?;
                Ok(Field::ImageInline { url: value, data })
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ItemRef<'a> {
    data: &'a [u8],
}

impl<'a> ItemRef<'a> {
    pub fn decode(data: &'a [u8]) -> Result<Self> {
        if data.len() < 2 * U16_SIZE {
            return Err(anyhow!("Data too short"));
        }
        Ok(ItemRef { data })
    }

    pub fn identifier(&self) -> &'a str {
        let id_len = u16::from_le_bytes(
            self.data[..U16_SIZE]
                .try_into()
                .expect("Failed to read identifier length"),
        ) as usize;
        unsafe { std::str::from_utf8_unchecked(&self.data[U16_SIZE..U16_SIZE + id_len]) }
    }

    pub fn num_fields(&self) -> u16 {
        let id_len = u16::from_le_bytes(
            self.data[..U16_SIZE]
                .try_into()
                .expect("Failed to read identifier length"),
        ) as usize;
        let offset = U16_SIZE + id_len;
        u16::from_le_bytes(
            self.data[offset..offset + U16_SIZE]
                .try_into()
                .expect("Failed to read number of fields"),
        )
    }

    pub fn fields(&self) -> FieldIter<'a> {
        let mut id_len = u16::from_le_bytes(
            self.data[..U16_SIZE]
                .try_into()
                .expect("Failed to read identifier length"),
        ) as usize;

        id_len += U16_SIZE;
        let num_fields = u16::from_le_bytes(
            self.data[id_len..id_len + U16_SIZE]
                .try_into()
                .expect("Failed to read number of fields"),
        );
        id_len += U16_SIZE;

        FieldIter {
            data: self.data,
            offset: id_len,
            remaining: num_fields,
        }
    }

    pub fn field(&self, index: usize) -> Option<FieldRef<'a>> {
        self.fields().nth(index)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FieldRef<'a> {
    Text(&'a str),
    ImageInline { url: &'a str, data: &'a [u8] },
    Image(&'a str),
}

impl<'a> FieldRef<'a> {
    pub fn as_str(&self) -> &'a str {
        match self {
            FieldRef::Text(s) => s,
            FieldRef::ImageInline { url, .. } => url,
            FieldRef::Image(url) => url,
        }
    }

    pub fn get_data(&self) -> Option<&'a [u8]> {
        match self {
            FieldRef::ImageInline { data, .. } => Some(data),
            _ => None,
        }
    }

    pub fn field_type(&self) -> FieldType {
        match self {
            FieldRef::Text(_) => FieldType::Text,
            FieldRef::ImageInline { .. } => FieldType::ImageInline,
            FieldRef::Image(_) => FieldType::Image,
        }
    }

    pub fn is_text(&self) -> bool {
        matches!(self, FieldRef::Text(_))
    }

    pub fn is_image(&self) -> bool {
        matches!(self, FieldRef::ImageInline { .. } | FieldRef::Image(_))
    }

    pub fn is_image_inline(&self) -> bool {
        matches!(self, FieldRef::ImageInline { .. })
    }

    pub fn load_data(&self) -> Result<Vec<u8>> {
        match self {
            FieldRef::ImageInline { data, .. } => Ok(data.to_vec()),
            FieldRef::Image(url) => load_bytes_from_url(url),
            FieldRef::Text(_) => Err(anyhow!("No data to load for text field")),
        }
    }
}

pub struct FieldIter<'a> {
    data: &'a [u8],
    offset: usize,
    remaining: u16,
}

impl<'a> Iterator for FieldIter<'a> {
    type Item = FieldRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let field_type = self.data[self.offset];
        self.offset += 1;

        let field = match field_type {
            0 => {
                let (text, len) = decode_str(self.data, self.offset);
                self.offset += len;
                FieldRef::Text(text)
            }
            1 => {
                let (url, len) = decode_str(self.data, self.offset);
                self.offset += len;

                let data_len = u64::from_le_bytes(
                    self.data[self.offset..self.offset + U64_SIZE]
                        .try_into()
                        .expect("Failed to read image data length"),
                ) as usize;
                self.offset += U64_SIZE;

                let data = &self.data[self.offset..self.offset + data_len];
                self.offset += data_len;

                FieldRef::ImageInline { url, data }
            }
            2 => {
                let (url, len) = decode_str(self.data, self.offset);
                self.offset += len;
                FieldRef::Image(url)
            }
            _ => panic!("Unknown field type: {}", field_type),
        };

        self.remaining -= 1;
        Some(field)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let count = self.remaining as usize;
        (count, Some(count))
    }
}

impl<'a> ExactSizeIterator for FieldIter<'a> {}

fn load_bytes_from_url(url: &str) -> Result<Vec<u8>> {
    use url::Url;

    let parsed = Url::parse(url);

    match parsed {
        Ok(url_parsed) => match url_parsed.scheme() {
            "http" | "https" => {
                let response = reqwest::blocking::get(url)?;
                Ok(response.bytes()?.to_vec())
            }
            "file" => {
                let path = url_parsed
                    .to_file_path()
                    .map_err(|_| anyhow!("Invalid file:// URL"))?;
                Ok(std::fs::read(path)?)
            }
            scheme => Err(anyhow!("Unsupported URL scheme: {}", scheme)),
        },
        Err(_) => Ok(std::fs::read(url)?),
    }
}

pub fn load_image_ndarray_from_bytes(bytes: &[u8]) -> Result<Array3<u8>> {
    let img = image::load_from_memory(bytes)?.to_rgb8();
    let (width, height) = img.dimensions();
    let array = Array3::from_shape_vec((height as usize, width as usize, 3), img.into_raw())?;
    Ok(array)
}

pub fn load_image_ndarray_from_url(url: &str) -> Result<Array3<u8>> {
    let bytes = load_bytes_from_url(url)?;
    load_image_ndarray_from_bytes(&bytes)
}

fn encode_string(bytes: &mut Vec<u8>, s: &str) {
    let s_bytes = s.as_bytes();
    bytes.extend_from_slice(&(s_bytes.len() as u64).to_le_bytes());
    bytes.extend_from_slice(s_bytes);
}

fn decode_str(data: &[u8], mut offset: usize) -> (&str, usize) {
    let len = u64::from_le_bytes(
        data[offset..offset + U64_SIZE]
            .try_into()
            .expect("Failed to read string length"),
    ) as usize;
    offset += U64_SIZE;
    let text = unsafe { std::str::from_utf8_unchecked(&data[offset..offset + len]) };
    (text, U64_SIZE + len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_field_creation() {
        let item = Item::from_string_fields(
            "Q64".to_string(),
            vec![
                StringField {
                    field_type: FieldType::Text,
                    value: "Berlin".to_string(),
                },
                StringField {
                    field_type: FieldType::Text,
                    value: "Capital of Germany".to_string(),
                },
            ],
        )
        .expect("Failed to create item");

        assert_eq!(item.identifier, "Q64");
        assert_eq!(item.num_fields(), 2);
        assert_eq!(item.fields[0], Field::Text("Berlin".to_string()));
    }

    #[test]
    fn test_image_field_creation() {
        let item = Item::from_string_fields(
            "Q64".to_string(),
            vec![
                StringField {
                    field_type: FieldType::Image,
                    value: "https://example.com/berlin1.jpg".to_string(),
                },
                StringField {
                    field_type: FieldType::Image,
                    value: "file:///path/to/berlin2.jpg".to_string(),
                },
            ],
        )
        .expect("Failed to create item");

        assert_eq!(item.num_fields(), 2);
        assert_eq!(
            item.fields[0],
            Field::Image("https://example.com/berlin1.jpg".to_string())
        );
    }


    #[test]
    fn test_encode_decode_text() {
        let item = Item::from_string_fields(
            "Q42".to_string(),
            vec![
                StringField {
                    field_type: FieldType::Text,
                    value: "Douglas Adams".to_string(),
                },
                StringField {
                    field_type: FieldType::Text,
                    value: "Hitchhiker's Guide".to_string(),
                },
            ],
        )
        .expect("Failed to create item");

        let encoded = item.encode();
        let item_ref = ItemRef::decode(&encoded).expect("Failed to decode");

        assert_eq!(item_ref.identifier(), "Q42");
        assert_eq!(item_ref.num_fields(), 2);

        let fields: Vec<_> = item_ref.fields().collect();
        assert_eq!(fields[0].as_str(), "Douglas Adams");
        assert_eq!(fields[1].as_str(), "Hitchhiker's Guide");
        assert!(fields[0].is_text());
    }


    #[test]
    fn test_item_ref_field_access() {
        let item = Item::from_string_fields(
            "Q1".to_string(),
            vec![
                StringField {
                    field_type: FieldType::Text,
                    value: "first".to_string(),
                },
                StringField {
                    field_type: FieldType::Text,
                    value: "second".to_string(),
                },
                StringField {
                    field_type: FieldType::Text,
                    value: "third".to_string(),
                },
            ],
        )
        .expect("Failed to create item");

        let encoded = item.encode();
        let item_ref = ItemRef::decode(&encoded).expect("Failed to decode");

        assert_eq!(item_ref.field(0).unwrap().as_str(), "first");
        assert_eq!(item_ref.field(1).unwrap().as_str(), "second");
        assert_eq!(item_ref.field(2).unwrap().as_str(), "third");
        assert!(item_ref.field(3).is_none());
    }

    #[test]
    fn test_field_iter_exact_size() {
        let item = Item::from_string_fields(
            "Q1".to_string(),
            vec![
                StringField {
                    field_type: FieldType::Text,
                    value: "a".to_string(),
                },
                StringField {
                    field_type: FieldType::Text,
                    value: "b".to_string(),
                },
                StringField {
                    field_type: FieldType::Text,
                    value: "c".to_string(),
                },
            ],
        )
        .expect("Failed to create item");

        let encoded = item.encode();
        let item_ref = ItemRef::decode(&encoded).expect("Failed to decode");

        let iter = item_ref.fields();
        assert_eq!(iter.len(), 3);
        assert_eq!(iter.size_hint(), (3, Some(3)));
    }

    #[test]
    fn test_identifier_too_long() {
        let long_id = "a".repeat((u16::MAX as usize) + 1);
        let result = Item::from_string_fields(
            long_id,
            vec![StringField {
                field_type: FieldType::Text,
                value: "field".to_string(),
            }],
        );

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Identifier too long")
        );
    }

    #[test]
    fn test_too_many_fields() {
        let fields = vec![
            StringField {
                field_type: FieldType::Text,
                value: "field".to_string(),
            };
            (u16::MAX as usize) + 1
        ];
        let result = Item::from_string_fields("Q1".to_string(), fields);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Too many fields"));
    }

    #[test]
    fn test_empty_fields() {
        let item = Item::from_string_fields("Q1".to_string(), vec![])
            .expect("Failed to create item");

        let encoded = item.encode();
        let item_ref = ItemRef::decode(&encoded).expect("Failed to decode");

        assert_eq!(item_ref.num_fields(), 0);
        assert_eq!(item_ref.fields().count(), 0);
    }

    #[test]
    fn test_special_characters() {
        let item = Item::from_string_fields(
            "Q1".to_string(),
            vec![
                StringField {
                    field_type: FieldType::Text,
                    value: "UTF-8: 你好世界".to_string(),
                },
                StringField {
                    field_type: FieldType::Text,
                    value: "Emoji: 🦀".to_string(),
                },
                StringField {
                    field_type: FieldType::Text,
                    value: "Special: @#$%".to_string(),
                },
            ],
        )
        .expect("Failed to create item");

        let encoded = item.encode();
        let item_ref = ItemRef::decode(&encoded).expect("Failed to decode");

        let fields: Vec<_> = item_ref.fields().collect();
        assert_eq!(fields[0].as_str(), "UTF-8: 你好世界");
        assert_eq!(fields[1].as_str(), "Emoji: 🦀");
        assert_eq!(fields[2].as_str(), "Special: @#$%");
    }
}
