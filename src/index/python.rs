use std::collections::HashSet;

use crate::data::embedding::Embedding;
use crate::data::python::Data;
use crate::data::{EmbeddingsWithData, Precision};
use crate::index::embedding::{EmbeddingIndexWithData, EmbeddingSearchParams};
use crate::index::{EmbeddingIndexParams, Match, Metric, Search};

use anyhow::{Result, anyhow};
use pyo3::prelude::*;

impl<'a, 'py> FromPyObject<'a, 'py> for Metric {
    type Error = anyhow::Error;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let Ok(value) = obj.extract::<&str>() else {
            return Err(anyhow!("Metric must be a string"));
        };
        match value.to_lowercase().as_str() {
            "cosinenormalized" | "cosine-normalized" => Ok(Metric::CosineNormalized),
            "cosine" => Ok(Metric::Cosine),
            "inner-product" | "innerproduct" | "ip" => Ok(Metric::InnerProduct),
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
pub struct EmbeddingIndex {
    inner: EmbeddingIndexWithData,
}

#[pymethods]
impl EmbeddingIndex {
    #[staticmethod]
    #[pyo3(signature = (data, embeddings_path, index_dir, metric=None, precision=None))]
    pub fn build(
        data: Data,
        embeddings_path: &str,
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

        let data = EmbeddingsWithData::load(data.inner, embeddings_path.as_ref())?;
        EmbeddingIndexWithData::build(&data, index_dir.as_ref(), &params)
    }

    #[staticmethod]
    pub fn load(data: Data, embeddings_path: &str, index_dir: &str) -> Result<Self> {
        let data = EmbeddingsWithData::load(data.inner, embeddings_path.as_ref())?;
        let inner = EmbeddingIndexWithData::load(data, index_dir.as_ref())?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (query, k=10, exact=false, min_score=None, rerank=None, allow_ids=None))]
    pub fn search(
        &self,
        query: Embedding,
        k: usize,
        exact: bool,
        min_score: Option<f32>,
        rerank: Option<f32>,
        allow_ids: Option<HashSet<u32>>,
    ) -> Result<Vec<Match>> {
        let params = EmbeddingSearchParams {
            k,
            exact,
            min_score,
            rerank,
        };

        if let Some(ids) = allow_ids {
            self.inner
                .search_with_filter(&query, &params, move |id| ids.contains(&id))
        } else {
            self.inner.search(&query, &params)
        }
    }

    #[getter]
    pub fn index_type(&self) -> &str {
        self.inner.index_type()
    }

    pub fn data(&self) -> Data {
        Data {
            inner: self.inner.data().data().clone(),
        }
    }

    #[getter]
    pub fn num_dimensions(&self) -> usize {
        self.inner.data().num_dimensions()
    }

    #[getter]
    pub fn model(&self) -> &str {
        self.inner.data().model()
    }
}
