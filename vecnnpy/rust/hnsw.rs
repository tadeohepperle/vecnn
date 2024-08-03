use numpy::{IntoPyArray, PyArray1};
use pyo3::{exceptions::PyTypeError, pyclass, pyfunction, pymethods, Bound, Py, PyResult, Python};
use vecnn::{
    dataset::DatasetT,
    distance::{cos, dot, l1, l2, Distance},
    hnsw::HnswParams,
    transition::{StitchMode, TransitionParams},
    utils::Stats,
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
        keep_fraction: neg_fraction * neg_fraction, // todo!!
        distance: dist_from_str(&distance_fn)?,
        stitch_mode: StitchMode::RandomNegToPosCenterAndBack,
        stop_after_stitching_n_chunks: None,
        x: 3,
    };
    let hnsw = vecnn::transition::build_hnsw_by_transition(data.as_dyn_dataset(), params);
    Ok(Hnsw(Inner::OldImpl(hnsw)))
}

pub fn dist_from_str(str: &str) -> PyResult<Distance> {
    let dist: Distance = match str {
        "l2" => Distance::L2,
        "cos" => Distance::Cos,
        "dot" => Distance::Dot,
        _ => return Err(PyTypeError::new_err("Array is not standard layout")),
    };
    Ok(dist)
}

#[pyclass]
pub struct Hnsw(Inner);
unsafe impl Send for Hnsw {}

enum Inner {
    OldImpl(vecnn::hnsw::Hnsw),
    NewImpl(vecnn::slice_hnsw::SliceHnsw),
}

impl Inner {
    fn build_stats(&self) -> Stats {
        match self {
            Inner::OldImpl(hnsw) => hnsw.build_stats,
            Inner::NewImpl(hnsw) => hnsw.build_stats,
        }
    }

    fn data(&self) -> &dyn DatasetT {
        match self {
            Inner::OldImpl(hnsw) => &*hnsw.data,
            Inner::NewImpl(hnsw) => &*hnsw.data,
        }
    }
}

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
            distance: dist_from_str(&distance_fn)?,
        };
        let hnsw = vecnn::slice_hnsw::SliceHnsw::new(data.as_dyn_dataset(), params);
        Ok(Hnsw(Inner::NewImpl(hnsw)))
        // let hnsw = vecnn::hnsw::Hnsw::new(data.as_dyn_dataset(), params);
        // Ok(Hnsw(Inner::OldImpl(hnsw)))
    }

    #[getter]
    fn num_distance_calculations_in_build(&self) -> PyResult<i32> {
        Ok(self.0.build_stats().num_distance_calculations as i32)
    }

    fn knn<'py>(
        &self,
        py: Python<'py>,
        query: Py<PyArray1<f32>>,
        k: usize,
        ef: usize,
    ) -> PyResult<KnnResult> {
        let q = pyarray1_to_slice(query, Some(self.0.data().dims()))?;
        // ugly temporary solution
        match &self.0 {
            Inner::OldImpl(hnsw) => {
                let (res, stats) = hnsw.knn_search(q, k, ef);
                let indices = ndarray::Array::from_iter(res.iter().map(|e| e.id))
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
            Inner::NewImpl(hnsw) => {
                let (res, stats) = hnsw.knn_search(q, k, ef);
                let indices = ndarray::Array::from_iter(res.iter().map(|e| e.1))
                    .into_pyarray_bound(py)
                    .unbind();
                let distances = ndarray::Array::from_iter(res.iter().map(|e| e.dist()))
                    .into_pyarray_bound(py)
                    .unbind();
                Ok(KnnResult {
                    indices,
                    distances,
                    num_distance_calculations: stats.num_distance_calculations,
                })
            }
        }
    }
}
