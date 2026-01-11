use anyhow::{Result, anyhow};
use http::{
    Response, StatusCode,
    header::{RETRY_AFTER, USER_AGENT},
};
use log::warn;
use numpy::ndarray::Array3;
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::mem::size_of;
use std::thread::sleep;
use std::time::Duration;
use ureq::{Agent, Body, config::Config};
use url::Url;

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
    #[serde(default)]
    pub tags: Vec<FieldTag>,
}

#[derive(Clone, Debug)]
pub struct Field {
    data: FieldData,
    tags: Vec<FieldTag>,
}

impl Field {
    pub fn type_and_tags(&self) -> u8 {
        let mut type_and_tag = self.data.to_type().to_bit();
        for tag in &self.tags {
            // set tag bits
            type_and_tag |= tag.to_bit();
        }
        type_and_tag
    }
}

#[derive(Clone, Debug)]
pub enum FieldData {
    Text(String),
    ImageInline { url: String, data: Vec<u8> },
    Image(String),
}

impl FieldData {
    pub fn to_type(&self) -> FieldType {
        match self {
            FieldData::Text(_) => FieldType::Text,
            FieldData::Image(_) => FieldType::Image,
            FieldData::ImageInline { .. } => FieldType::ImageInline,
        }
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FieldType {
    #[default]
    Text,
    Image,
    ImageInline,
}

impl FieldType {
    pub fn to_bit(&self) -> u8 {
        let num = match self {
            FieldType::Text => 0,
            FieldType::Image => 1,
            FieldType::ImageInline => 2,
        };
        num << 4
    }

    pub fn from_type_and_tags(byte: u8) -> Option<FieldType> {
        match byte >> 4 {
            0 => Some(FieldType::Text),
            1 => Some(FieldType::Image),
            2 => Some(FieldType::ImageInline),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FieldTag {
    // use this to tag fields when indexing
    // one bit per tag, currently we use a u8
    // to store field type and tags. reserve
    // 4 bits for tags and 4 bits for field types.
    // so we can have up to 4 tags and 16 field types.
    Main,
}

impl FieldTag {
    pub fn to_bit(&self) -> u8 {
        match self {
            FieldTag::Main => 0b0000_0001,
        }
    }

    pub fn is_set(&self, byte: u8) -> bool {
        let bit = self.to_bit();
        (byte & bit) != 0
    }
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
            .map(|field| field.field_type.create_field(field.value, field.tags))
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
        bytes.push(self.type_and_tags());

        match &self.data {
            FieldData::Text(text) => {
                encode_string(bytes, text);
            }
            FieldData::ImageInline { url, data } => {
                encode_string(bytes, url);
                bytes.extend_from_slice(&(data.len() as u64).to_le_bytes());
                bytes.extend_from_slice(data);
            }
            FieldData::Image(url) => {
                encode_string(bytes, url);
            }
        }
    }
}

impl FieldType {
    pub fn create_field(&self, value: String, tags: Vec<FieldTag>) -> Result<Field> {
        let data = match self {
            FieldType::Text => FieldData::Text(value),
            FieldType::Image => FieldData::Image(value),
            FieldType::ImageInline => {
                let data = load_bytes_from_url(&value)?;
                FieldData::ImageInline { url: value, data }
            }
        };

        Ok(Field { data, tags })
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
pub struct FieldRef<'a> {
    data: FieldRefData<'a>,
    type_and_tags: u8,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FieldRefData<'a> {
    Text(&'a str),
    ImageInline { url: &'a str, data: &'a [u8] },
    Image(&'a str),
}

impl FieldRefData<'_> {
    pub fn to_type(&self) -> FieldType {
        match self {
            FieldRefData::Text(_) => FieldType::Text,
            FieldRefData::ImageInline { .. } => FieldType::ImageInline,
            FieldRefData::Image(_) => FieldType::Image,
        }
    }
}

impl<'a> FieldRef<'a> {
    pub fn as_str(&self) -> &'a str {
        match self.data {
            FieldRefData::Text(s) => s,
            FieldRefData::ImageInline { url, .. } => url,
            FieldRefData::Image(url) => url,
        }
    }

    pub fn get_data(&self) -> Option<&'a [u8]> {
        match self.data {
            FieldRefData::ImageInline { data, .. } => Some(data),
            _ => None,
        }
    }

    pub fn field_type(&self) -> FieldType {
        self.data.to_type()
    }

    pub fn is_text(&self) -> bool {
        matches!(self.field_type(), FieldType::Text)
    }

    pub fn is_image(&self) -> bool {
        matches!(self.field_type(), FieldType::Image | FieldType::ImageInline)
    }

    pub fn is_image_inline(&self) -> bool {
        matches!(self.field_type(), FieldType::ImageInline)
    }

    pub fn has_tag(&self, tag: FieldTag) -> bool {
        tag.is_set(self.type_and_tags)
    }

    pub fn load_data(&self) -> Result<Vec<u8>> {
        match self.data {
            FieldRefData::ImageInline { data, .. } => Ok(data.to_vec()),
            FieldRefData::Image(url) => load_bytes_from_url(url),
            FieldRefData::Text(_) => Err(anyhow!("No data to load for text field")),
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

        let type_and_tags = self.data[self.offset];
        self.offset += 1;
        let field_type = FieldType::from_type_and_tags(type_and_tags)?;

        let data = match field_type {
            FieldType::Text => {
                let (text, len) = decode_str(self.data, self.offset);
                self.offset += len;
                FieldRefData::Text(text)
            }
            FieldType::ImageInline => {
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

                FieldRefData::ImageInline { url, data }
            }
            FieldType::Image => {
                let (url, len) = decode_str(self.data, self.offset);
                self.offset += len;
                FieldRefData::Image(url)
            }
        };

        self.remaining -= 1;
        Some(FieldRef {
            data,
            type_and_tags,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let count = self.remaining as usize;
        (count, Some(count))
    }
}

impl<'a> ExactSizeIterator for FieldIter<'a> {}

fn load_bytes_from_url(url: &str) -> Result<Vec<u8>> {
    let parsed = Url::parse(url);

    match parsed {
        Ok(url_parsed) => match url_parsed.scheme() {
            "http" | "https" => fetch_http_with_retry(url),
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

fn error_from_response(mut response: Response<Body>) -> anyhow::Error {
    anyhow!(
        "HTTP {}: {}",
        response.status(),
        response
            .body_mut()
            .read_to_string()
            .unwrap_or_else(|e| format!("Failed to read response body: {}", e))
    )
}

fn fetch_http_with_retry(url: &str) -> Result<Vec<u8>> {
    const MAX_TRIES: u32 = 5;
    const INITIAL_BACKOFF_MS: u64 = 1000;

    let config = Config::builder().http_status_as_error(false).build();
    let agent = Agent::new_with_config(config);

    let mut attempt = 0;
    loop {
        match agent
            .get(url)
            .header(USER_AGENT, "search-rdf-http-client")
            .call()
        {
            Ok(response) if should_retry(response.status()) => {
                if attempt >= MAX_TRIES {
                    warn!(
                        "Request to {} failed with status {}. Max retries reached.",
                        url,
                        response.status(),
                    );
                    return Err(error_from_response(response));
                }
                let backoff_ms = parse_retry_after(&response)
                    .unwrap_or_else(|| INITIAL_BACKOFF_MS * 2u64.pow(attempt));

                warn!(
                    "Request to {} failed with status {} during attempt {}/{}. Retrying after {:.2}s...",
                    url,
                    response.status(),
                    attempt + 1,
                    MAX_TRIES,
                    Duration::from_millis(backoff_ms).as_secs_f64(),
                );
                sleep(Duration::from_millis(backoff_ms));
                attempt += 1;
            }
            Ok(response) if response.status().is_success() => {
                let mut bytes = Vec::new();
                response.into_body().into_reader().read_to_end(&mut bytes)?;
                return Ok(bytes);
            }
            Ok(response) => {
                warn!(
                    "Request to {} failed with unexpected status {}. Not retrying.",
                    url,
                    response.status()
                );
                return Err(error_from_response(response));
            }
            Err(e) => {
                return Err(anyhow!("Unexpected error fetching {}: {}", url, e));
            }
        }
    }
}

fn should_retry(status: StatusCode) -> bool {
    status.is_server_error() || status == 429
}

fn parse_retry_after(response: &Response<Body>) -> Option<u64> {
    response.headers().get(RETRY_AFTER).and_then(|value| {
        // The Retry-After header can be either a number of seconds or an HTTP date
        // We'll only handle the seconds format for simplicity
        value
            .to_str()
            .ok()
            .and_then(|str| str.parse::<u64>().ok())
            .map(|seconds| seconds * 1000)
    })
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
        let item =
            Item::from_string_fields("Q1".to_string(), vec![]).expect("Failed to create item");

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
