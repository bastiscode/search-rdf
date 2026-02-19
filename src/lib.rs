pub mod data;
pub mod index;
pub mod model;
pub mod utils;

use pyo3::prelude::*;

/// Get SIMD backend without creating a full index
#[pyfunction]
fn embedding_index_hardware_acceleration() -> PyResult<String> {
    // Create a minimal dummy index just to query SIMD capabilities
    let options = usearch::IndexOptions {
        dimensions: 1,
        metric: usearch::MetricKind::Cos,
        quantization: usearch::ScalarKind::F32,
        connectivity: 0,
        expansion_add: 0,
        expansion_search: 0,
        multi: false,
    };

    let index = usearch::new_index(&options)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.what().to_string()))?;
    Ok(index.hardware_acceleration())
}

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add data class
    m.add_class::<data::python::Data>()?;

    // Add index classes
    m.add_class::<index::text::python::KeywordIndex>()?;
    m.add_class::<index::text::python::FuzzyIndex>()?;
    m.add_class::<index::python::EmbeddingIndex>()?;

    // Add SIMD detection function
    m.add_function(wrap_pyfunction!(embedding_index_hardware_acceleration, m)?)?;

    Ok(())
}
