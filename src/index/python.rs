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
        let s: &str = obj.extract()?;
        serde_plain::from_str(s).map_err(|_| {
            anyhow!(
                "Invalid metric: {}. Expected one of: cosine, cosine-normalized, l2, inner-product, hamming",
                s
            )
        })
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Precision {
    type Error = anyhow::Error;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let s: &str = obj.extract()?;
        serde_plain::from_str(s).map_err(|_| {
            anyhow!(
                "Invalid precision: {}. Expected one of: float32, float16, bfloat16, int8, binary",
                s
            )
        })
    }
}
#[pyclass]
pub struct EmbeddingIndex {
    inner: EmbeddingIndexWithData,
}

#[pymethods]
impl EmbeddingIndex {
    #[staticmethod]
    #[pyo3(signature = (data, embedding_path, index_dir, metric=None, precision=None))]
    pub fn build(
        data: &Data,
        embedding_path: &str,
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

        let data = EmbeddingsWithData::load(data.inner.clone(), embedding_path.as_ref())?;
        EmbeddingIndexWithData::build(&data, index_dir.as_ref(), &params)
    }

    #[staticmethod]
    pub fn load(data: &Data, embedding_path: &str, index_dir: &str) -> Result<Self> {
        let data = EmbeddingsWithData::load(data.inner.clone(), embedding_path.as_ref())?;
        let inner = EmbeddingIndexWithData::load(data, index_dir.as_ref())?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (embedding, k=10, exact=false, min_score=None, rerank=None, allow_ids=None))]
    pub fn search(
        &self,
        embedding: Embedding,
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
                .search_with_filter(&embedding, &params, move |id| ids.contains(&id))
        } else {
            self.inner.search(&embedding, &params)
        }
    }

    #[getter]
    pub fn index_type(&self) -> &str {
        "embedding"
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

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    #[test]
    fn test_from_pyobject_metric_cosine() {
        Python::attach(|py| {
            let s = "cosine".into_pyobject(py).unwrap();
            let metric: Metric = s.extract().unwrap();
            assert!(matches!(metric, Metric::Cosine));
        });
    }

    #[test]
    fn test_from_pyobject_metric_cosine_normalized() {
        Python::attach(|py| {
            let s = "cosine-normalized".into_pyobject(py).unwrap();
            let metric: Metric = s.extract().unwrap();
            assert!(matches!(metric, Metric::CosineNormalized));
        });
    }

    #[test]
    fn test_from_pyobject_metric_l2() {
        Python::attach(|py| {
            let s = "l2".into_pyobject(py).unwrap();
            let metric: Metric = s.extract().unwrap();
            assert!(matches!(metric, Metric::L2));
        });
    }

    #[test]
    fn test_from_pyobject_metric_euclidean_alias() {
        Python::attach(|py| {
            let s = "euclidean".into_pyobject(py).unwrap();
            let metric: Metric = s.extract().unwrap();
            assert!(matches!(metric, Metric::L2));
        });
    }

    #[test]
    fn test_from_pyobject_metric_inner_product() {
        Python::attach(|py| {
            let s = "inner-product".into_pyobject(py).unwrap();
            let metric: Metric = s.extract().unwrap();
            assert!(matches!(metric, Metric::InnerProduct));
        });
    }

    #[test]
    fn test_from_pyobject_metric_dot_product_alias() {
        Python::attach(|py| {
            let s = "dot-product".into_pyobject(py).unwrap();
            let metric: Metric = s.extract().unwrap();
            assert!(matches!(metric, Metric::InnerProduct));
        });
    }

    #[test]
    fn test_from_pyobject_metric_hamming() {
        Python::attach(|py| {
            let s = "hamming".into_pyobject(py).unwrap();
            let metric: Metric = s.extract().unwrap();
            assert!(matches!(metric, Metric::Hamming));
        });
    }

    #[test]
    fn test_from_pyobject_metric_invalid() {
        Python::attach(|py| {
            let s = "invalid".into_pyobject(py).unwrap();
            let result: Result<Metric> = s.extract();
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("Invalid metric"));
        });
    }

    #[test]
    fn test_from_pyobject_precision_float32() {
        Python::attach(|py| {
            let s = "float32".into_pyobject(py).unwrap();
            let precision: Precision = s.extract().unwrap();
            assert!(matches!(precision, Precision::Float32));
        });
    }

    #[test]
    fn test_from_pyobject_precision_fp32_alias() {
        Python::attach(|py| {
            let s = "fp32".into_pyobject(py).unwrap();
            let precision: Precision = s.extract().unwrap();
            assert!(matches!(precision, Precision::Float32));
        });
    }

    #[test]
    fn test_from_pyobject_precision_float16() {
        Python::attach(|py| {
            let s = "float16".into_pyobject(py).unwrap();
            let precision: Precision = s.extract().unwrap();
            assert!(matches!(precision, Precision::Float16));
        });
    }

    #[test]
    fn test_from_pyobject_precision_fp16_alias() {
        Python::attach(|py| {
            let s = "fp16".into_pyobject(py).unwrap();
            let precision: Precision = s.extract().unwrap();
            assert!(matches!(precision, Precision::Float16));
        });
    }

    #[test]
    fn test_from_pyobject_precision_bfloat16() {
        Python::attach(|py| {
            let s = "bfloat16".into_pyobject(py).unwrap();
            let precision: Precision = s.extract().unwrap();
            assert!(matches!(precision, Precision::BFloat16));
        });
    }

    #[test]
    fn test_from_pyobject_precision_bf16_alias() {
        Python::attach(|py| {
            let s = "bf16".into_pyobject(py).unwrap();
            let precision: Precision = s.extract().unwrap();
            assert!(matches!(precision, Precision::BFloat16));
        });
    }

    #[test]
    fn test_from_pyobject_precision_int8() {
        Python::attach(|py| {
            let s = "int8".into_pyobject(py).unwrap();
            let precision: Precision = s.extract().unwrap();
            assert!(matches!(precision, Precision::Int8));
        });
    }

    #[test]
    fn test_from_pyobject_precision_i8_alias() {
        Python::attach(|py| {
            let s = "i8".into_pyobject(py).unwrap();
            let precision: Precision = s.extract().unwrap();
            assert!(matches!(precision, Precision::Int8));
        });
    }

    #[test]
    fn test_from_pyobject_precision_binary() {
        Python::attach(|py| {
            let s = "binary".into_pyobject(py).unwrap();
            let precision: Precision = s.extract().unwrap();
            assert!(matches!(precision, Precision::Binary));
        });
    }

    #[test]
    fn test_from_pyobject_precision_ubinary_alias() {
        Python::attach(|py| {
            let s = "ubinary".into_pyobject(py).unwrap();
            let precision: Precision = s.extract().unwrap();
            assert!(matches!(precision, Precision::Binary));
        });
    }

    #[test]
    fn test_from_pyobject_precision_bit_alias() {
        Python::attach(|py| {
            let s = "bit".into_pyobject(py).unwrap();
            let precision: Precision = s.extract().unwrap();
            assert!(matches!(precision, Precision::Binary));
        });
    }

    #[test]
    fn test_from_pyobject_precision_invalid() {
        Python::attach(|py| {
            let s = "invalid".into_pyobject(py).unwrap();
            let result: Result<Precision> = s.extract();
            assert!(result.is_err());
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("Invalid precision")
            );
        });
    }
}
