use crate::data::text::{TextData as RustTextData, TextEmbeddings as RustTextEmbeddings};
use crate::data::{DataSource, Precision};
use anyhow::Result;
use pyo3::prelude::*;

#[derive(Clone)]
#[pyclass]
pub struct TextData {
    pub inner: RustTextData,
}

#[pymethods]
impl TextData {
    #[staticmethod]
    pub fn build(data_dir: &str) -> Result<()> {
        RustTextData::build(data_dir.as_ref())
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

    pub fn num_fields(&self, id: u32) -> Option<usize> {
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

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<Vec<String>> {
        while (slf.index as usize) < slf.data.len() {
            let current = slf.index;
            slf.index += 1;

            if let Some(fields) = slf.data.fields(current) {
                return Some(fields.into_iter().map(|s| s.to_string()).collect());
            }
        }
        None
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
    pub fn load(data_dir: &str) -> Result<Self> {
        let inner = RustTextEmbeddings::load(data_dir.as_ref())?;
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

    pub fn precision(&self) -> Precision {
        self.inner.precision()
    }

    pub fn model(&self) -> &str {
        self.inner.model()
    }

    pub fn text_data(&self) -> TextData {
        TextData {
            inner: self.inner.text_data().clone(),
        }
    }

    pub fn id_from_identifier(&self, identifier: &str) -> Option<u32> {
        self.inner.text_data().id_from_identifier(identifier)
    }
}
