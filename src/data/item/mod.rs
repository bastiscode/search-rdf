use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::mem::size_of;
pub mod jsonl;
pub mod sparql;

const U16_SIZE: usize = size_of::<u16>();
const U32_SIZE: usize = size_of::<u32>();

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Item {
    pub identifier: String,
    pub fields: Vec<String>,
}

impl Item {
    pub fn validate(self) -> Result<Self> {
        if self.identifier.len() > u32::MAX as usize {
            return Err(anyhow!("identifier length exceeds maximum of {}", u32::MAX));
        } else if self.fields.len() > u16::MAX as usize {
            return Err(anyhow!("number of fields exceeds maximum of {}", u16::MAX));
        } else if self
            .fields
            .iter()
            .any(|field| field.len() > u32::MAX as usize)
        {
            return Err(anyhow!(
                "one or more field lengths exceed maximum of {}",
                u32::MAX
            ));
        }
        Ok(self)
    }

    pub fn new(identifier: String, fields: Vec<String>) -> Result<Self> {
        let item = Item { identifier, fields };
        item.validate()
    }

    pub fn num_fields(&self) -> u16 {
        self.fields.len() as u16
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        let id_bytes = self.identifier.as_bytes();
        let id_len = id_bytes.len() as u32;
        bytes.extend_from_slice(&id_len.to_le_bytes());
        bytes.extend_from_slice(id_bytes);

        let num_fields = self.fields.len() as u16;
        bytes.extend_from_slice(&num_fields.to_le_bytes());
        for field in &self.fields {
            let field_bytes = field.as_bytes();
            let field_len = field_bytes.len() as u32;
            bytes.extend_from_slice(&field_len.to_le_bytes());
            bytes.extend_from_slice(field_bytes);
        }

        bytes
    }

    pub fn decode_key(data: &[u8]) -> (&str, usize) {
        let key_length_bytes = data[..U32_SIZE]
            .try_into()
            .expect("Failed to read key length");
        let key_length = u32::from_le_bytes(key_length_bytes) as usize;

        let key_start = U32_SIZE;
        let key_end = key_start + key_length;
        let key = unsafe { std::str::from_utf8_unchecked(&data[key_start..key_end]) };

        (key, key_end)
    }

    pub fn decode_values(data: &[u8]) -> impl Iterator<Item = (&str, usize)> + '_ {
        let num_values_bytes = data[..U16_SIZE]
            .try_into()
            .expect("Failed to read number of values");
        let num_values = u16::from_le_bytes(num_values_bytes) as usize;

        let mut offset = U16_SIZE;
        (0..num_values).map(move |_| {
            let value_length_bytes = data[offset..offset + U32_SIZE]
                .try_into()
                .expect("Failed to read value length");
            let value_length = u32::from_le_bytes(value_length_bytes) as usize;

            let value_start = offset + U32_SIZE;
            let value_end = value_start + value_length;
            let value = unsafe { std::str::from_utf8_unchecked(&data[value_start..value_end]) };

            let consumed = U32_SIZE + value_length;
            offset = value_end;
            (value, consumed)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_item_creation_valid() {
        let item = Item::new(
            "Q1".to_string(),
            vec!["field1".to_string(), "field2".to_string()],
        )
        .expect("Failed to create Item");

        assert_eq!(item.identifier, "Q1");
        assert_eq!(item.fields.len(), 2);
        assert_eq!(item.num_fields(), 2);
    }

    #[test]
    fn test_item_creation_empty_fields() {
        let item =
            Item::new("Q1".to_string(), vec![]).expect("Failed to create Item with empty fields");

        assert_eq!(item.identifier, "Q1");
        assert_eq!(item.fields.len(), 0);
        assert_eq!(item.num_fields(), 0);
    }

    #[test]
    fn test_item_creation_single_field() {
        let item = Item::new("Q1".to_string(), vec!["single".to_string()])
            .expect("Failed to create Item with single field");

        assert_eq!(item.fields.len(), 1);
        assert_eq!(item.num_fields(), 1);
    }

    #[test]
    fn test_item_identifier_too_long() {
        let long_id = "a".repeat((u32::MAX as usize) + 1);
        let result = Item::new(long_id, vec!["field".to_string()]);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("identifier length exceeds maximum")
        );
    }

    #[test]
    fn test_item_too_many_fields() {
        let fields = vec!["field".to_string(); (u16::MAX as usize) + 1];
        let result = Item::new("Q1".to_string(), fields);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("number of fields exceeds maximum")
        );
    }

    #[test]
    fn test_item_field_too_long() {
        let long_field = "a".repeat((u32::MAX as usize) + 1);
        let result = Item::new("Q1".to_string(), vec![long_field]);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("field lengths exceed maximum")
        );
    }

    #[test]
    fn test_item_encode_decode_roundtrip() {
        let item = Item::new(
            "Q42".to_string(),
            vec![
                "Douglas Adams".to_string(),
                "Hitchhiker's Guide".to_string(),
                "42".to_string(),
            ],
        )
        .expect("Failed to create Item");

        let encoded = item.encode();

        // Decode identifier
        let (decoded_id, id_end) = Item::decode_key(&encoded);
        assert_eq!(decoded_id, "Q42");

        // Decode fields
        let decoded_fields: Vec<&str> = Item::decode_values(&encoded[id_end..])
            .map(|(s, _)| s)
            .collect();

        assert_eq!(decoded_fields.len(), 3);
        assert_eq!(decoded_fields[0], "Douglas Adams");
        assert_eq!(decoded_fields[1], "Hitchhiker's Guide");
        assert_eq!(decoded_fields[2], "42");
    }

    #[test]
    fn test_item_encode_decode_empty_fields() {
        let item = Item::new("Q1".to_string(), vec![]).expect("Failed to create Item");

        let encoded = item.encode();

        // Decode identifier
        let (decoded_id, id_end) = Item::decode_key(&encoded);
        assert_eq!(decoded_id, "Q1");

        // Decode fields
        let decoded_fields: Vec<&str> = Item::decode_values(&encoded[id_end..])
            .map(|(s, _)| s)
            .collect();

        assert_eq!(decoded_fields.len(), 0);
    }

    #[test]
    fn test_item_encode_decode_special_characters() {
        let item = Item::new(
            "Q1".to_string(),
            vec![
                "Hello, World!".to_string(),
                "UTF-8: 你好世界".to_string(),
                "Emoji: 🦀".to_string(),
                "Symbols: @#$%^&*()".to_string(),
            ],
        )
        .expect("Failed to create Item");

        let encoded = item.encode();

        // Decode identifier
        let (decoded_id, id_end) = Item::decode_key(&encoded);
        assert_eq!(decoded_id, "Q1");

        // Decode fields
        let decoded_fields: Vec<&str> = Item::decode_values(&encoded[id_end..])
            .map(|(s, _)| s)
            .collect();

        assert_eq!(decoded_fields.len(), 4);
        assert_eq!(decoded_fields[0], "Hello, World!");
        assert_eq!(decoded_fields[1], "UTF-8: 你好世界");
        assert_eq!(decoded_fields[2], "Emoji: 🦀");
        assert_eq!(decoded_fields[3], "Symbols: @#$%^&*()");
    }

    #[test]
    fn test_item_encode_decode_empty_strings() {
        let item = Item::new(
            "Q1".to_string(),
            vec!["".to_string(), "non-empty".to_string(), "".to_string()],
        )
        .expect("Failed to create Item");

        let encoded = item.encode();

        // Decode identifier
        let (decoded_id, id_end) = Item::decode_key(&encoded);
        assert_eq!(decoded_id, "Q1");

        // Decode fields
        let decoded_fields: Vec<&str> = Item::decode_values(&encoded[id_end..])
            .map(|(s, _)| s)
            .collect();

        assert_eq!(decoded_fields.len(), 3);
        assert_eq!(decoded_fields[0], "");
        assert_eq!(decoded_fields[1], "non-empty");
        assert_eq!(decoded_fields[2], "");
    }

    #[test]
    fn test_decode_values_size_tracking() {
        let item = Item::new(
            "Q1".to_string(),
            vec!["short".to_string(), "medium-length".to_string()],
        )
        .expect("Failed to create Item");

        let encoded = item.encode();
        let (_, id_end) = Item::decode_key(&encoded);

        // Verify that sizes are correctly tracked
        let values_with_sizes: Vec<(&str, usize)> =
            Item::decode_values(&encoded[id_end..]).collect();

        assert_eq!(values_with_sizes.len(), 2);

        // First value: "short" (5 bytes) + u32 length header (4 bytes) = 9 bytes
        assert_eq!(values_with_sizes[0].0, "short");
        assert_eq!(values_with_sizes[0].1, 9);

        // Second value: "medium-length" (13 bytes) + u32 length header (4 bytes) = 17 bytes
        assert_eq!(values_with_sizes[1].0, "medium-length");
        assert_eq!(values_with_sizes[1].1, 17);
    }

    #[test]
    fn test_item_encode_format() {
        let item =
            Item::new("Q1".to_string(), vec!["field1".to_string()]).expect("Failed to create Item");

        let encoded = item.encode();

        // Expected format:
        // [id_len: u32][id_bytes][num_fields: u16][field1_len: u32][field1_bytes]

        let mut offset = 0;

        // Check identifier length
        let id_len = u32::from_le_bytes(encoded[offset..offset + 4].try_into().unwrap());
        assert_eq!(id_len, 2); // "Q1" is 2 bytes
        offset += 4;

        // Check identifier
        let id_bytes = &encoded[offset..offset + id_len as usize];
        assert_eq!(std::str::from_utf8(id_bytes).unwrap(), "Q1");
        offset += id_len as usize;

        // Check number of fields
        let num_fields = u16::from_le_bytes(encoded[offset..offset + 2].try_into().unwrap());
        assert_eq!(num_fields, 1);
        offset += 2;

        // Check field length
        let field_len = u32::from_le_bytes(encoded[offset..offset + 4].try_into().unwrap());
        assert_eq!(field_len, 6); // "field1" is 6 bytes
        offset += 4;

        // Check field value
        let field_bytes = &encoded[offset..offset + field_len as usize];
        assert_eq!(std::str::from_utf8(field_bytes).unwrap(), "field1");
    }

    #[test]
    fn test_item_max_valid_sizes() {
        // Test with maximum valid number of fields (u16::MAX)
        let max_fields = vec!["field".to_string(); u16::MAX as usize];
        let result = Item::new("Q1".to_string(), max_fields);
        assert!(result.is_ok());
    }
}
