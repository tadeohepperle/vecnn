use numpy::{IntoPyArray, PyArray1};
use pyo3::{exceptions::PyTypeError, pyclass, pyfunction, pymethods, Bound, Py, PyResult, Python};
use vecnn::{
    distance::{cos, dot, l1, l2, DistanceFn},
    hnsw::HnswParams,
    transition::TransitionParams,
};

use crate::utils::{pyarray1_to_slice, KnnResult};

#[pyfunction]
pub fn build_hnsw_by_transition(
    data: crate::Dataset,
    max_chunk_size: usize,
    same_chunk_max_neighbors: usize,
    neg_fraction: f32,
    distance_fn: String,
) -> PyResult<Hnsw> {
    let params = TransitionParams {
        max_chunk_size,
        same_chunk_max_neighbors,
        neg_fraction,
        distance_fn: dist_fn_from_str(&distance_fn)?,
    };
    let hnsw = vecnn::transition::build_hnsw_by_transition(data.as_dyn_dataset(), params);
    Ok(Hnsw(hnsw))
}

pub fn dist_fn_from_str(str: &str) -> PyResult<DistanceFn> {
    let distance_fn: vecnn::distance::DistanceFn = match str {
        "l1" => l1,
        "l2" => l2,
        "cos" => cos,
        "dot" => dot,
        _ => return Err(PyTypeError::new_err("Array is not standard layout")),
    };
    Ok(distance_fn)
}

#[pyclass]
pub struct Hnsw(vecnn::hnsw::Hnsw);

#[pymethods]
impl Hnsw {
    #[new]
    fn new(
        data: crate::Dataset,
        level_norm_param: f32,
        ef_construction: usize,
        m_max: usize,
        m_max_0: usize,
        distance_fn: String,
    ) -> PyResult<Self> {
        let params = HnswParams {
            level_norm_param,
            ef_construction,
            m_max,
            m_max_0,
            distance_fn: dist_fn_from_str(&distance_fn)?,
        };
        let hnsw = vecnn::hnsw::Hnsw::new(data.as_dyn_dataset(), params);
        Ok(Hnsw(hnsw))
    }

    #[getter]
    fn num_distance_calculations_in_build(&self) -> PyResult<i32> {
        Ok(self.0.build_stats.num_distance_calculations as i32)
    }

    fn knn<'py>(&self, py: Python<'py>, query: Py<PyArray1<f32>>, k: usize) -> PyResult<KnnResult> {
        let q = pyarray1_to_slice(query, Some(self.0.data.dims()))?;
        let (res, stats) = self.0.knn_search(q, k);
        let indices = ndarray::Array::from_iter(res.iter().map(|e| e.id as usize))
            .into_pyarray_bound(py)
            .unbind();
        let distances = ndarray::Array::from_iter(res.iter().map(|e| e.d_to_q))
            .into_pyarray_bound(py)
            .unbind();
        Ok(KnnResult {
            indices,
            distances,
            num_distance_calculations: stats.num_distance_calculations,
        })
    }
}
