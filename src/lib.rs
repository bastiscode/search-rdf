pub mod data;
pub mod index;
pub mod model;
pub mod utils;

use pyo3::prelude::*;

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add data class
    m.add_class::<data::python::Data>()?;

    // Add index classes
    m.add_class::<index::text::python::KeywordIndex>()?;
    m.add_class::<index::python::EmbeddingIndex>()?;

    Ok(())
}
