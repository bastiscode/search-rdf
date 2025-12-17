pub mod data;
pub mod index;
pub mod utils;

use pyo3::prelude::*;

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add data classes
    m.add_class::<data::text::python::TextData>()?;
    m.add_class::<data::text::python::TextEmbeddings>()?;

    // Add index classes
    m.add_class::<index::text::python::KeywordIndex>()?;
    m.add_class::<index::text::python::TextEmbeddingIndex>()?;

    Ok(())
}
