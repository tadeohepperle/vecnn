use numpy::{IntoPyArray, PyArray1};
use pyo3::{exceptions::PyTypeError, pyclass, pyfunction, pymethods, Bound, Py, PyResult, Python};
use vecnn::distance::{cos, cos_for_spherical, l1, l2, DistanceFn};

use crate::utils::{pyarray1_to_slice, KnnResult};

#[pyclass]
pub struct HnswParams {
    #[pyo3(get, set)]
    pub level_norm_param: f32,
    #[pyo3(get, set)]
    pub ef_construction: usize,
    #[pyo3(get, set)]
    pub m_max: usize,
    #[pyo3(get, set)]
    pub m_max_0: usize,
    #[pyo3(get, set)]
    pub distance_fn: String,
}
#[pymethods]
impl HnswParams {
    #[new]
    fn new(level_norm_param: f32, ef_construction: usize, m_max: usize, m_max_0: usize) -> Self {
        HnswParams {
            level_norm_param,
            ef_construction,
            m_max,
            m_max_0,
            distance_fn: String::from("l2"),
        }
    }
}

#[pyclass]
pub struct TransitionParams {
    #[pyo3(get, set)]
    pub max_chunk_size: usize,
    #[pyo3(get, set)]
    pub same_chunk_max_neighbors: usize,
    #[pyo3(get, set)]
    pub neg_fraction: f32,
}

#[pymethods]
impl TransitionParams {
    #[new]
    fn new(max_chunk_size: usize, same_chunk_max_neighbors: usize, neg_fraction: f32) -> Self {
        TransitionParams {
            max_chunk_size,
            same_chunk_max_neighbors,
            neg_fraction,
        }
    }
}

impl TransitionParams {
    pub fn to_vecnn_params(&self) -> PyResult<vecnn::transition::TransitionParams> {
        Ok(vecnn::transition::TransitionParams {
            max_chunk_size: self.max_chunk_size,
            same_chunk_max_neighbors: self.same_chunk_max_neighbors,
            neg_fraction: self.neg_fraction,
        })
    }
}

#[pyfunction]
pub fn build_hnsw_by_transition<'py>(
    py: Python<'py>,
    data: crate::Dataset,
    params: &'py TransitionParams,
) -> PyResult<Hnsw> {
    let hnsw = vecnn::transition::build_hnsw_by_transition(
        data.as_dyn_dataset(),
        params.to_vecnn_params()?,
    );
    Ok(Hnsw(hnsw))
}

fn dist_fn_from_str(str: &str) -> PyResult<DistanceFn> {
    let distance_fn: vecnn::distance::DistanceFn = match str {
        "l1" => l1,
        "l2" => l2,
        "cos" => cos,
        "cos_for_spherical" => cos_for_spherical,
        _ => return Err(PyTypeError::new_err("Array is not standard layout")),
    };
    Ok(distance_fn)
}

impl HnswParams {
    pub fn to_vecnn_params(&self) -> PyResult<vecnn::hnsw::HnswParams> {
        Ok(vecnn::hnsw::HnswParams {
            level_norm_param: self.level_norm_param,
            ef_construction: self.ef_construction,
            m_max: self.m_max,
            m_max_0: self.m_max_0,
            distance_fn: dist_fn_from_str(self.distance_fn.as_str())?,
        })
    }
}

#[pyclass]
pub struct Hnsw(vecnn::hnsw::Hnsw);

#[pymethods]
impl Hnsw {
    #[new]
    fn new<'py>(py: Python<'py>, data: crate::Dataset, params: &'py HnswParams) -> PyResult<Self> {
        let hnsw = vecnn::hnsw::Hnsw::new(data.as_dyn_dataset(), params.to_vecnn_params()?);
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

    // fn knn<'py>(
    //     &self,
    //     py: Python<'py>,
    //     query: Py<PyArray1<f32>>,
    //     k: usize,
    // ) -> PyResult<(Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<f32>>, i32)> {
    //     let q = pyarray1_to_slice(query, self.0.data.dims())?;
    //     let (res, stats) = self.0.knn_search(q, k);
    //     let idices = ndarray::Array::from_iter(res.iter().map(|e| e.idx)).into_pyarray_bound(py);
    //     let distances =
    //         ndarray::Array::from_iter(res.iter().map(|e| e.dist)).into_pyarray_bound(py);
    //     Ok((idices, distances, stats.num_distance_calculations as i32))
    // }
}
