use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};

mod dataset;
mod vp_tree;
pub use dataset::Dataset;
pub use vp_tree::VpTree;

/// This module is implemented in Rust.
#[pymodule]
fn vecnnpy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    fn get_one() -> f32 {
        1.0
    }
    m.add_class::<Dataset>()?;
    m.add_class::<VpTree>()?;
    Ok(())
}
