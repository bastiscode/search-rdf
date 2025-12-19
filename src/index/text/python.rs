use std::collections::HashSet;

use crate::data::Precision;
use crate::data::embedding::Embedding;
use crate::data::text::python::{TextData, TextEmbeddings};
use crate::index::Match;
use crate::index::SearchIndex;
use crate::index::SearchParams;
use crate::index::embedding::{EmbeddingIndexParams, Metric};
use crate::index::text::embedding::{Query, TextEmbeddingIndex as RustTextEmbeddingIndex};
use crate::index::text::keyword::KeywordIndex as RustKeywordIndex;
use anyhow::{Result, anyhow};
use pyo3::prelude::*;

impl<'a, 'py> FromPyObject<'a, 'py> for Metric {
    type Error = anyhow::Error;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let Ok(value) = obj.extract::<&str>() else {
            return Err(anyhow!("Metric must be a string"));
        };
        match value.to_lowercase().as_str() {
            "cosinenormalized" | "cosine_normalized" => Ok(Metric::CosineNormalized),
            "cosine" => Ok(Metric::Cosine),
            "inner_product" | "innerproduct" | "ip" => Ok(Metric::InnerProduct),
            "euclidean" | "l2" => Ok(Metric::L2),
            "hamming" => Ok(Metric::Hamming),
            other => Err(anyhow!("unsupported metric: {other}")),
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Precision {
    type Error = anyhow::Error;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let s: &str = obj.extract()?;
        let precision = match s.to_lowercase().as_str() {
            "float32" | "fp32" => Precision::Float32,
            "binary" | "bit" => Precision::Binary,
            "float16" | "fp16" => Precision::Float16,
            "bfloat16" | "bfp16" => Precision::BFloat16,
            "int8" | "i8" => Precision::Int8,
            _ => {
                return Err(anyhow!(
                    "Invalid Precision: {}. Expected one of: float32, binary, float16, bfloat16, int8",
                    s
                ));
            }
        };
        Ok(precision)
    }
}

#[pyclass]
pub struct KeywordIndex {
    inner: RustKeywordIndex,
}

#[pymethods]
impl KeywordIndex {
    #[staticmethod]
    pub fn build(data: &TextData, index_dir: &str) -> Result<()> {
        RustKeywordIndex::build(&data.inner, index_dir.as_ref(), &())
    }

    #[staticmethod]
    pub fn load(data: TextData, index_dir: &str) -> Result<Self> {
        let inner = RustKeywordIndex::load(data.inner, index_dir.as_ref())?;
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
        let mut params = SearchParams::default().with_k(k).with_exact(exact);
        if let Some(min_score) = min_score {
            params = params.with_min_score(min_score);
        }

        if let Some(ids) = allow_ids {
            self.inner
                .search_with_filter(query, &params, move |id| ids.contains(&id))
        } else {
            self.inner.search(query, &params)
        }
    }
}

#[pyclass]
pub struct TextEmbeddingIndex {
    inner: RustTextEmbeddingIndex,
}

#[pymethods]
impl TextEmbeddingIndex {
    #[staticmethod]
    #[pyo3(signature = (data, index_dir, metric=None, precision=None))]
    pub fn build(
        data: &TextEmbeddings,
        index_dir: &str,
        metric: Option<Metric>,
        precision: Option<Precision>,
    ) -> Result<()> {
        let mut params = if let Some(precision) = precision {
            EmbeddingIndexParams::from_precision(precision)
        } else {
            EmbeddingIndexParams::default()
        };
        if let Some(metric) = metric {
            params = params.with_metric(metric);
        };
        RustTextEmbeddingIndex::build(&data.inner, index_dir.as_ref(), &params)
    }

    #[staticmethod]
    pub fn load(data: TextEmbeddings, index_dir: &str) -> Result<Self> {
        let inner = RustTextEmbeddingIndex::load(data.inner, index_dir.as_ref())?;
        Ok(TextEmbeddingIndex { inner })
    }

    #[pyo3(signature = (embedding, k=100, exact=false, min_score=None, allow_ids=None))]
    pub fn search(
        &self,
        embedding: Embedding,
        k: usize,
        exact: bool,
        min_score: Option<f32>,
        allow_ids: Option<HashSet<u32>>,
    ) -> Result<Vec<Match>> {
        let mut params = SearchParams::default().with_k(k).with_exact(exact);
        if let Some(min_score) = min_score {
            params = params.with_min_score(min_score);
        }

        let query = Query::Embedding(embedding.as_ref());

        if let Some(ids) = allow_ids {
            self.inner
                .search_with_filter(query, &params, move |id| ids.contains(&id))
        } else {
            self.inner.search(query, &params)
        }
    }
}
