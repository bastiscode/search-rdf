use std::convert::Infallible;

use crate::data::Data as RustData;
use crate::data::item::FieldType;
use crate::data::item::jsonl::stream_items_from_jsonl_file;
use crate::data::item::sparql::{
    SPARQLResultFormat, guess_sparql_result_format_from_extension,
    stream_items_from_sparql_result_file,
};
use crate::data::{DataSource, Precision};
use anyhow::{Result, anyhow};
use pyo3::prelude::*;
use pyo3::types::PyString;

impl<'py> IntoPyObject<'py> for Precision {
    type Target = PyString;

    type Output = Bound<'py, Self::Target>;

    type Error = Infallible;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
        let s = match self {
            Precision::Float32 => "float32",
            Precision::Binary => "ubinary",
            Precision::Float16 => "float16",
            Precision::BFloat16 => "bfloat16",
            Precision::Int8 => "int8",
        };
        s.into_pyobject(py)
    }
}

#[derive(Clone)]
#[pyclass]
pub struct Data {
    pub inner: RustData,
}

impl<'a, 'py> FromPyObject<'a, 'py> for SPARQLResultFormat {
    type Error = anyhow::Error;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let s: &str = obj.extract()?;
        let format = match s.to_lowercase().as_str() {
            "json" => SPARQLResultFormat::JSON,
            "xml" => SPARQLResultFormat::XML,
            "tsv" => SPARQLResultFormat::TSV,
            _ => {
                return Err(anyhow!(
                    "Invalid QueryResultsFormat: {}. Expected one of: json, xml, csv, tsv",
                    s
                ));
            }
        };
        Ok(format)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for FieldType {
    type Error = anyhow::Error;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let s: &str = obj.extract()?;
        let field_type = match s.to_lowercase().as_str() {
            "text" => FieldType::Text,
            "image" => FieldType::Image,
            "image-inline" => FieldType::ImageInline,
            _ => {
                return Err(anyhow!(
                    "Invalid FieldType: {}. Expected one of: text, image, image-inline",
                    s
                ));
            }
        };
        Ok(field_type)
    }
}

#[pymethods]
impl Data {
    #[staticmethod]
    pub fn build_from_jsonl(file_path: &str, data_dir: &str) -> Result<()> {
        RustData::build(
            stream_items_from_jsonl_file(file_path.as_ref())?,
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
            None => guess_sparql_result_format_from_extension(file_path.as_ref())?,
        };
        RustData::build(
            stream_items_from_sparql_result_file(file_path.as_ref(), format, default_field_type)?,
            data_dir.as_ref(),
        )
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
