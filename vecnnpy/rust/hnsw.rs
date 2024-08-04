use numpy::{IntoPyArray, PyArray1};
use pyo3::{exceptions::PyTypeError, pyclass, pyfunction, pymethods, Bound, Py, PyResult, Python};
use vecnn::{
    dataset::DatasetT,
    distance::{cos, dot, l1, l2, Distance},
    hnsw::HnswParams,
    transition::{EnsembleParams, StitchMode, TransitionParams},
    utils::Stats,
};

use crate::utils::{pyarray1_to_slice, KnnResult};

#[pyfunction]
pub fn build_hnsw_by_vp_tree_ensemble(
    data: crate::Dataset,
    max_chunk_size: usize,
    same_chunk_m_max: usize,
    m_max: usize,
    m_max_0: usize,
    n_vp_trees: usize,
    level_norm: f32,
    distance: String,
    seed: u64,
) -> PyResult<Hnsw> {
    // todo! the python side of this interface needs to be adjusted!!!
    let params = EnsembleParams {
        max_chunk_size,
        same_chunk_m_max,
        m_max,
        n_vp_trees,
        m_max_0,
        level_norm,
        distance: dist_from_str(&distance)?,
    };

    let hnsw = vecnn::transition::build_hnsw_by_vp_tree_ensemble_multi_layer(
        data.as_dyn_dataset(),
        params,
        seed,
    );
    Ok(Hnsw(Inner::SliceImpl(hnsw)))
}

#[pyfunction]
pub fn build_hnsw_by_transition(
    data: crate::Dataset,
    max_chunk_size: usize,
    same_chunk_m_max: usize,
    m_max: usize,
    neg_fraction: f32,
    keep_fraction: f32,
    distance: String,
    seed: u64,
) -> PyResult<Hnsw> {
    // todo! the python side of this interface needs to be adjusted!!!
    let params = TransitionParams {
        max_chunk_size,
        same_chunk_m_max,
        m_max,
        neg_fraction,
        keep_fraction, // todo!!
        distance: dist_from_str(&distance)?,
        stitch_mode: StitchMode::RandomNegToPosCenterAndBack,
        stop_after_stitching_n_chunks: None,
        x: 3,
    };
    let hnsw = vecnn::transition::build_hnsw_by_transition(data.as_dyn_dataset(), params, seed);
    Ok(Hnsw(Inner::SliceImpl(hnsw)))
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
    ConstImpl(vecnn::hnsw::Hnsw),
    SliceImpl(vecnn::slice_hnsw::SliceHnsw),
}

impl Inner {
    fn build_stats(&self) -> Stats {
        match self {
            Inner::ConstImpl(hnsw) => hnsw.build_stats,
            Inner::SliceImpl(hnsw) => hnsw.build_stats,
        }
    }

    fn data(&self) -> &dyn DatasetT {
        match self {
            Inner::ConstImpl(hnsw) => &*hnsw.data,
            Inner::SliceImpl(hnsw) => &*hnsw.data,
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
        distance: String,
        use_const_impl: bool,
        seed: u64,
    ) -> PyResult<Self> {
        let params = HnswParams {
            level_norm_param,
            ef_construction,
            m_max,
            m_max_0,
            distance: dist_from_str(&distance)?,
        };

        if use_const_impl {
            let hnsw = vecnn::hnsw::Hnsw::new(data.as_dyn_dataset(), params, seed);
            Ok(Hnsw(Inner::ConstImpl(hnsw)))
        } else {
            let hnsw = vecnn::slice_hnsw::SliceHnsw::new(data.as_dyn_dataset(), params, seed);
            Ok(Hnsw(Inner::SliceImpl(hnsw)))
        }

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
            Inner::ConstImpl(hnsw) => {
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
            Inner::SliceImpl(hnsw) => {
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
