use numpy::{IntoPyArray, PyArray1};
use pyo3::{exceptions::PyTypeError, pyclass, pymethods, Bound, Py, PyResult, Python};
use vecnn::distance::{cos, cos_for_spherical, l1, l2};

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

impl HnswParams {
    pub fn to_vecnn_hnsw_params(&self) -> PyResult<vecnn::hnsw::HnswParams> {
        let distance_fn: vecnn::distance::DistanceFn = match self.distance_fn.as_str() {
            "l1" => l1,
            "l2" => l2,
            "cos" => cos,
            "cos_for_spherical" => cos_for_spherical,
            _ => return Err(PyTypeError::new_err("Array is not standard layout")),
        };

        Ok(vecnn::hnsw::HnswParams {
            level_norm_param: self.level_norm_param,
            ef_construction: self.ef_construction,
            m_max: self.m_max,
            m_max_0: self.m_max_0,
            distance_fn,
        })
    }
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
pub struct Hnsw(vecnn::hnsw::Hnsw);

#[pymethods]
impl Hnsw {
    #[new]
    fn new<'py>(py: Python<'py>, data: crate::Dataset, params: &'py HnswParams) -> PyResult<Self> {
        let hnsw = vecnn::hnsw::Hnsw::new(data.as_dyn_dataset(), params.to_vecnn_hnsw_params()?);
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
