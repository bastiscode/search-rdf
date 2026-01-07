use std::convert::Infallible;

use crate::data::item::sparql::SPARQLResultFormat;
use crate::data::{Data as RustData, embedding::EmbeddingsWithData as RustEmbeddingsWithData};
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

#[pymethods]
impl Data {
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

    fn __next__(&mut self) -> Option<Vec<&str>> {
        let fields = self
            .data
            .fields(self.index)?
            .map(|field| field.as_str())
            .collect();
        self.index += 1;
        Some(fields)
    }
}

#[derive(Clone)]
#[pyclass]
pub struct Embeddings {
    pub inner: RustEmbeddingsWithData,
}

#[pymethods]
impl Embeddings {
    #[staticmethod]
    pub fn load(data: Data, embeddings_file: &str) -> Result<Self> {
        let inner = RustEmbeddingsWithData::load(data.inner, embeddings_file.as_ref())?;
        Ok(Self { inner })
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
