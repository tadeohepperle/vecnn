use std::collections::HashSet;

use crate::hnsw::{Hnsw, HnswParams};
use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArray1, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};

mod dataset;
mod hnsw;
mod utils;
mod vp_tree;

use crate::dataset::Dataset;
use crate::utils::{pyarray1_to_slice, static_python, KnnResult};
use crate::vp_tree::VpTree;

// fn foo() {}

/// This module is implemented in Rust.
#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    fn get_one() -> f32 {
        1.0
    }
    m.add_class::<Dataset>()?;
    m.add_class::<VpTree>()?;
    m.add_class::<KnnResult>()?;
    m.add_class::<Hnsw>()?;
    m.add_class::<HnswParams>()?;
    m.add_function(wrap_pyfunction!(linear_knn, m)?)?;
    m.add_function(wrap_pyfunction!(knn_recall, m)?)?;
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn linear_knn<'py>(
    py: Python<'py>,
    data: crate::Dataset,
    query: Py<PyArray1<f32>>,
    k: usize,
) -> PyResult<KnnResult> {
    let q = pyarray1_to_slice(query, Some(data.dims()))?;
    let results = vecnn::utils::linear_knn_search(data.as_dyn_dataset_ref(), q, k);
    let indices: Py<PyArray1<usize>> =
        ndarray::Array::from_iter(results.iter().map(|e| e.i as usize))
            .into_pyarray_bound(py)
            .unbind();
    let distances: Py<PyArray1<f32>> = ndarray::Array::from_iter(results.iter().map(|e| e.dist))
        .into_pyarray_bound(py)
        .unbind();
    Ok(KnnResult {
        indices,
        distances,
        num_distance_calculations: data.len(),
    })
}

#[pyfunction]
fn knn_recall(truth: Py<PyArray1<usize>>, reported: Py<PyArray1<usize>>) -> PyResult<f64> {
    let truth = pyarray1_to_slice(truth, None)?;
    let reported = pyarray1_to_slice(reported, None)?;

    let truth_hs = HashSet::<usize>::from_iter(truth.iter().copied());
    let mut reported_count: usize = 0;
    for e in reported {
        if truth_hs.contains(e) {
            reported_count += 1;
        }
    }
    let recall = reported_count as f64 / truth.len() as f64;
    Ok(recall)
}
