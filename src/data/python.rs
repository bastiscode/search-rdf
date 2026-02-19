use std::path::Path;

use crate::data::Data as RustData;
use crate::data::DataSource;
use crate::data::item::jsonl::{ItemJson, stream_items_from_jsonl_file};
use crate::data::item::sparql::{
    SPARQLResultFormat, guess_sparql_result_format_from_extension,
    stream_items_from_sparql_result_file,
};
use crate::data::item::{FieldTag, FieldType, Item, StringField};
use anyhow::{Result, anyhow};
use pyo3::prelude::*;
use pythonize::depythonize;

#[pyclass]
pub struct Data {
    pub inner: RustData,
}

impl<'a, 'py> FromPyObject<'a, 'py> for SPARQLResultFormat {
    type Error = anyhow::Error;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let s: &str = obj.extract()?;
        serde_plain::from_str(s).map_err(|_| {
            anyhow!(
                "Invalid SPARQL result format: {}. Expected one of: json, xml, csv, tsv",
                s
            )
        })
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for FieldType {
    type Error = anyhow::Error;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let s: &str = obj.extract()?;
        serde_plain::from_str(s).map_err(|_| {
            anyhow!(
                "Invalid field type: {}. Expected one of: text, image, image-inline",
                s
            )
        })
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for FieldTag {
    type Error = anyhow::Error;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let s: &str = obj.extract()?;
        serde_plain::from_str(s).map_err(|_| anyhow!("Invalid tag: {}. Expected one of: main", s))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for StringField {
    type Error = anyhow::Error;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        depythonize(&obj).map_err(|e| anyhow!("Failed to convert to string field: {}", e))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Item {
    type Error = anyhow::Error;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let item_json: ItemJson =
            depythonize(&obj).map_err(|e| anyhow!("Failed to convert to item: {}", e))?;
        item_json.try_into()
    }
}

#[pymethods]
impl Data {
    #[staticmethod]
    pub fn build_from_jsonl(file_path: &str, data_dir: &str) -> Result<()> {
        RustData::build(
            stream_items_from_jsonl_file(Path::new(file_path))?,
            data_dir.as_ref(),
        )
    }

    #[staticmethod]
    #[pyo3(signature = (file_path, data_dir, format = None, default_field_type = FieldType::Text))]
    pub fn build_from_sparql_result(
        file_path: &str,
        data_dir: &str,
        format: Option<SPARQLResultFormat>,
        default_field_type: FieldType,
    ) -> Result<()> {
        let format = match format {
            Some(f) => f,
            None => guess_sparql_result_format_from_extension(Path::new(file_path))?,
        };
        RustData::build(
            stream_items_from_sparql_result_file(Path::new(file_path), format, default_field_type)?,
            data_dir.as_ref(),
        )
    }

    #[staticmethod]
    pub fn build_from_items(items: Vec<Item>, data_dir: &str) -> Result<()> {
        RustData::build(items.into_iter().map(Ok), data_dir.as_ref())
    }

    #[staticmethod]
    pub fn load(data_dir: &str) -> Result<Self> {
        let inner = RustData::load(data_dir.as_ref())?;
        Ok(Data { inner })
    }

    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn __iter__(slf: PyRef<'_, Self>) -> DataIterator {
        DataIterator {
            data: slf.inner.clone(),
            index: 0,
        }
    }

    pub fn num_fields(&self, id: u32) -> Option<u16> {
        self.inner.num_fields(id)
    }

    pub fn field(&self, id: u32, field: usize) -> Option<&str> {
        self.inner.field(id, field).map(|s| s.as_str())
    }

    pub fn main_field(&self, id: u32) -> Option<&str> {
        self.inner.main_field(id).map(|s| s.as_str())
    }

    pub fn fields(&self, id: u32) -> Option<Vec<&str>> {
        self.inner
            .fields(id)
            .map(|vec| vec.into_iter().map(|s| s.as_str()).collect())
    }

    pub fn identifier(&self, id: u32) -> Option<&str> {
        self.inner.identifier(id)
    }

    pub fn id_from_identifier(&self, identifier: &str) -> Option<u32> {
        self.inner.id_from_identifier(identifier)
    }
}

#[pyclass]
pub struct DataIterator {
    data: RustData,
    index: u32,
}

#[pymethods]
impl DataIterator {
    fn __iter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<(&str, Vec<&str>)> {
        let identifier = self.data.identifier(self.index)?;
        let fields = self
            .data
            .fields(self.index)?
            .map(|field| field.as_str())
            .collect();
        self.index += 1;
        Some((identifier, fields))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::{PyDict, PyList};

    #[test]
    fn test_from_pyobject_sparql_result_format_json() {
        Python::attach(|py| {
            let s = "json".into_pyobject(py).unwrap();
            let format: SPARQLResultFormat = s.extract().unwrap();
            assert!(matches!(format, SPARQLResultFormat::Json));
        });
    }

    #[test]
    fn test_from_pyobject_sparql_result_format_xml() {
        Python::attach(|py| {
            let s = "xml".into_pyobject(py).unwrap();
            let format: SPARQLResultFormat = s.extract().unwrap();
            assert!(matches!(format, SPARQLResultFormat::Xml));
        });
    }

    #[test]
    fn test_from_pyobject_sparql_result_format_tsv() {
        Python::attach(|py| {
            let s = "tsv".into_pyobject(py).unwrap();
            let format: SPARQLResultFormat = s.extract().unwrap();
            assert!(matches!(format, SPARQLResultFormat::Tsv));
        });
    }

    #[test]
    fn test_from_pyobject_sparql_result_format_invalid() {
        Python::attach(|py| {
            let s = "invalid".into_pyobject(py).unwrap();
            let result: Result<SPARQLResultFormat> = s.extract();
            assert!(result.is_err());
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("Invalid SPARQL result format")
            );
        });
    }

    #[test]
    fn test_from_pyobject_field_type_text() {
        Python::attach(|py| {
            let s = "text".into_pyobject(py).unwrap();
            let field_type: FieldType = s.extract().unwrap();
            assert_eq!(field_type, FieldType::Text);
        });
    }

    #[test]
    fn test_from_pyobject_field_type_image() {
        Python::attach(|py| {
            let s = "image".into_pyobject(py).unwrap();
            let field_type: FieldType = s.extract().unwrap();
            assert_eq!(field_type, FieldType::Image);
        });
    }

    #[test]
    fn test_from_pyobject_field_type_image_inline() {
        Python::attach(|py| {
            let s = "image-inline".into_pyobject(py).unwrap();
            let field_type: FieldType = s.extract().unwrap();
            assert_eq!(field_type, FieldType::ImageInline);
        });
    }

    #[test]
    fn test_from_pyobject_field_type_invalid() {
        Python::attach(|py| {
            let s = "invalid".into_pyobject(py).unwrap();
            let result: Result<FieldType> = s.extract();
            assert!(result.is_err());
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("Invalid field type")
            );
        });
    }

    #[test]
    fn test_from_pyobject_field_tag_main() {
        Python::attach(|py| {
            let s = "main".into_pyobject(py).unwrap();
            let tag: FieldTag = s.extract().unwrap();
            assert_eq!(tag, FieldTag::Main);
        });
    }

    #[test]
    fn test_from_pyobject_field_tag_invalid() {
        Python::attach(|py| {
            let s = "invalid".into_pyobject(py).unwrap();
            let result: Result<FieldTag> = s.extract();
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("Invalid tag"));
        });
    }

    #[test]
    fn test_from_pyobject_string_field_text() {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("type", "text").unwrap();
            dict.set_item("value", "Hello World").unwrap();

            let field: StringField = dict.extract().unwrap();
            assert_eq!(field.field_type, FieldType::Text);
            assert_eq!(field.value, "Hello World");
            assert!(field.tags.is_empty());
        });
    }

    #[test]
    fn test_from_pyobject_string_field_with_tags() {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("type", "text").unwrap();
            dict.set_item("value", "Main Label").unwrap();
            let tags = PyList::new(py, ["main"]).unwrap();
            dict.set_item("tags", tags).unwrap();

            let field: StringField = dict.extract().unwrap();
            assert_eq!(field.field_type, FieldType::Text);
            assert_eq!(field.value, "Main Label");
            assert_eq!(field.tags.len(), 1);
            assert_eq!(field.tags[0], FieldTag::Main);
        });
    }

    #[test]
    fn test_from_pyobject_string_field_image() {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("type", "image").unwrap();
            dict.set_item("value", "https://example.com/image.jpg")
                .unwrap();

            let field: StringField = dict.extract().unwrap();
            assert_eq!(field.field_type, FieldType::Image);
            assert_eq!(field.value, "https://example.com/image.jpg");
        });
    }

    #[test]
    fn test_from_pyobject_string_field_default_type() {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("value", "No type specified").unwrap();

            let field: StringField = dict.extract().unwrap();
            assert_eq!(field.field_type, FieldType::Text); // Default should be Text
            assert_eq!(field.value, "No type specified");
        });
    }

    #[test]
    fn test_from_pyobject_item_simple() {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("identifier", "Q1").unwrap();

            let fields = PyList::empty(py);
            let field1 = PyDict::new(py);
            field1.set_item("type", "text").unwrap();
            field1.set_item("value", "Universe").unwrap();
            fields.append(field1).unwrap();

            let field2 = PyDict::new(py);
            field2.set_item("type", "text").unwrap();
            field2.set_item("value", "Cosmos").unwrap();
            fields.append(field2).unwrap();

            dict.set_item("fields", fields).unwrap();

            let item: Item = dict.extract().unwrap();
            assert_eq!(item.identifier, "Q1");
            assert_eq!(item.num_fields(), 2);
        });
    }

    #[test]
    fn test_from_pyobject_item_with_tags() {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("identifier", "Q42").unwrap();

            let fields = PyList::empty(py);
            let field1 = PyDict::new(py);
            field1.set_item("type", "text").unwrap();
            field1.set_item("value", "Douglas Adams").unwrap();
            let tags = PyList::new(py, ["main"]).unwrap();
            field1.set_item("tags", tags).unwrap();
            fields.append(field1).unwrap();

            dict.set_item("fields", fields).unwrap();

            let item: Item = dict.extract().unwrap();
            assert_eq!(item.identifier, "Q42");
            assert_eq!(item.num_fields(), 1);
            assert!(item.fields[0].has_tag(FieldTag::Main));
        });
    }

    #[test]
    fn test_from_pyobject_item_empty_fields() {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("identifier", "Q100").unwrap();
            let fields = PyList::empty(py);
            dict.set_item("fields", fields).unwrap();

            let item: Item = dict.extract().unwrap();
            assert_eq!(item.identifier, "Q100");
            assert_eq!(item.num_fields(), 0);
        });
    }

    #[test]
    fn test_from_pyobject_item_missing_identifier() {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            let fields = PyList::empty(py);
            dict.set_item("fields", fields).unwrap();

            let result: Result<Item> = dict.extract();
            assert!(result.is_err());
        });
    }

    #[test]
    fn test_from_pyobject_item_missing_fields() {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("identifier", "Q1").unwrap();

            let result: Result<Item> = dict.extract();
            assert!(result.is_err());
        });
    }
}
