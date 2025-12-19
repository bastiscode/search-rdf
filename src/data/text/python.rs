use std::convert::Infallible;

use crate::data::text::item::jsonl::stream_text_items_from_jsonl_file;
use crate::data::text::item::sparql::{
    SPARQLResultFormat, guess_sparql_result_format_from_extension,
    stream_text_items_from_sparql_result_file,
};
use crate::data::text::{TextData as RustTextData, TextEmbeddings as RustTextEmbeddings};
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
pub struct TextData {
    pub inner: RustTextData,
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

#[pymethods]
impl TextData {
    #[staticmethod]
    pub fn from_jsonl(data_file: &str, data_dir: &str) -> Result<()> {
        let items = stream_text_items_from_jsonl_file(data_file.as_ref())?;
        RustTextData::build(items, data_dir.as_ref())
    }

    #[staticmethod]
    #[pyo3(signature = (data_file, data_dir, format = None))]
    pub fn from_sparql_result(
        data_file: &str,
        data_dir: &str,
        format: Option<SPARQLResultFormat>,
    ) -> Result<()> {
        let format = if let Some(format) = format {
            format
        } else {
            guess_sparql_result_format_from_extension(data_file.as_ref())?
        };
        let items = stream_text_items_from_sparql_result_file(data_file.as_ref(), format)?;
        RustTextData::build(items, data_dir.as_ref())
    }

    #[staticmethod]
    pub fn load(data_dir: &str) -> Result<Self> {
        let inner = RustTextData::load(data_dir.as_ref())?;
        Ok(TextData { inner })
    }

    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn __iter__(slf: PyRef<'_, Self>) -> TextDataIterator {
        TextDataIterator {
            data: slf.inner.clone(),
            index: 0,
        }
    }

    pub fn num_fields(&self, id: u32) -> Option<u16> {
        self.inner.num_fields(id)
    }

    pub fn field(&self, id: u32, field: usize) -> Option<&str> {
        self.inner.field(id, field)
    }

    pub fn fields(&self, id: u32) -> Option<Vec<String>> {
        self.inner
            .fields(id)
            .map(|vec| vec.into_iter().map(|s| s.to_string()).collect())
    }

    pub fn identifier(&self, id: u32) -> Option<String> {
        self.inner.identifier(id).map(|s| s.to_string())
    }

    pub fn id_from_identifier(&self, identifier: &str) -> Option<u32> {
        self.inner.id_from_identifier(identifier)
    }
}

#[pyclass]
pub struct TextDataIterator {
    data: RustTextData,
    index: u32,
}

#[pymethods]
impl TextDataIterator {
    fn __iter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<Vec<&str>> {
        let fields = self.data.fields(self.index)?.collect();
        self.index += 1;
        Some(fields)
    }
}

#[derive(Clone)]
#[pyclass]
pub struct TextEmbeddings {
    pub inner: RustTextEmbeddings,
}

#[pymethods]
impl TextEmbeddings {
    #[staticmethod]
    pub fn load(data: TextData, embeddings_file: &str) -> Result<Self> {
        let inner = RustTextEmbeddings::load(data.inner, embeddings_file.as_ref())?;
        Ok(TextEmbeddings { inner })
    }

    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn num_dimensions(&self) -> usize {
        self.inner.num_dimensions()
    }

    pub fn model(&self) -> &str {
        self.inner.model()
    }
}
