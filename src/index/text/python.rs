use std::collections::HashSet;

use crate::data::python::Data;
use crate::index::Match;
use crate::index::Search;
use crate::index::text::fuzzy::FuzzyIndex as RustFuzzyIndex;
use crate::index::text::fuzzy::FuzzySearchParams;
use crate::index::text::keyword::KeywordIndex as RustKeywordIndex;
use crate::index::text::keyword::KeywordSearchParams;
use anyhow::Result;
use pyo3::prelude::*;

#[pyclass]
pub struct KeywordIndex {
    inner: RustKeywordIndex,
}

#[pymethods]
impl KeywordIndex {
    #[staticmethod]
    pub fn build(data: &Data, index_dir: &str) -> Result<()> {
        RustKeywordIndex::build(&data.inner, index_dir.as_ref(), &())
    }

    #[staticmethod]
    pub fn load(data: &Data, index_dir: &str) -> Result<Self> {
        let inner = RustKeywordIndex::load(data.inner.clone(), index_dir.as_ref())?;
        Ok(KeywordIndex { inner })
    }

    #[pyo3(signature = (query, k=10, exact=false, min_score=None, allow_ids=None))]
    pub fn search(
        &self,
        query: &str,
        k: usize,
        exact: bool,
        min_score: Option<f32>,
        allow_ids: Option<HashSet<u32>>,
    ) -> Result<Vec<Match>> {
        let params = KeywordSearchParams {
            k,
            exact,
            min_score,
        };

        if let Some(ids) = allow_ids {
            self.inner
                .search_with_filter(query, &params, move |id| ids.contains(&id))
        } else {
            self.inner.search(query, &params)
        }
    }

    #[getter]
    pub fn index_type(&self) -> &str {
        self.inner.index_type()
    }

    pub fn data(&self) -> Data {
        Data {
            inner: self.inner.data().clone(),
        }
    }
}

#[pyclass]
pub struct FuzzyIndex {
    inner: RustFuzzyIndex,
}

#[pymethods]
impl FuzzyIndex {
    #[staticmethod]
    pub fn build(data: &Data, index_dir: &str) -> Result<()> {
        RustFuzzyIndex::build(&data.inner, index_dir.as_ref(), &())
    }

    #[staticmethod]
    pub fn load(data: &Data, index_dir: &str) -> Result<Self> {
        let inner = RustFuzzyIndex::load(data.inner.clone(), index_dir.as_ref())?;
        Ok(FuzzyIndex { inner })
    }

    #[pyo3(signature = (query, k=10, exact=false, min_score=None, allow_ids=None))]
    pub fn search(
        &self,
        query: &str,
        k: usize,
        exact: bool,
        min_score: Option<f32>,
        allow_ids: Option<HashSet<u32>>,
    ) -> Result<Vec<Match>> {
        let params = FuzzySearchParams {
            k,
            exact,
            min_score,
        };

        if let Some(ids) = allow_ids {
            self.inner
                .search_with_filter(query, &params, move |id| ids.contains(&id))
        } else {
            self.inner.search(query, &params)
        }
    }

    #[getter]
    pub fn index_type(&self) -> &str {
        self.inner.index_type()
    }

    pub fn data(&self) -> Data {
        Data {
            inner: self.inner.data().clone(),
        }
    }
}
