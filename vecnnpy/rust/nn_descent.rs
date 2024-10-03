use crate::{
    hnsw::dist_from_str,
    utils::{pyarray1_to_slice, KnnResult},
};
use numpy::{IntoPyArray, PyArray1};
use pyo3::{exceptions::PyTypeError, pyclass, pyfunction, pymethods, Bound, Py, PyResult, Python};
use vecnn::{
    distance::{cos, dot, l1, l2, DistanceFn},
    relative_nn_descent::RNNGraphParams,
};

#[pyclass]
pub struct RNNGraph(vecnn::relative_nn_descent::RNNGraph);

#[pymethods]
impl RNNGraph {
    #[new]
    fn new<'py>(
        py: Python<'py>,
        data: crate::Dataset,
        outer_loops: usize,
        inner_loops: usize,
        max_neighbors_after_reverse_pruning: usize,
        initial_neighbors: usize,
        threaded: bool,
        distance: String,
        seed: u64,
    ) -> PyResult<Self> {
        let params = RNNGraphParams {
            outer_loops,
            inner_loops,
            max_neighbors_after_reverse_pruning,
            initial_neighbors,
            distance: dist_from_str(&distance)?,
        };
        let hnsw = vecnn::relative_nn_descent::RNNGraph::new(
            data.as_dyn_dataset(),
            params,
            seed,
            threaded,
        );
        Ok(RNNGraph(hnsw))
    }

    #[getter]
    fn num_distance_calculations_in_build(&self) -> PyResult<i32> {
        Ok(self.0.build_stats.num_distance_calculations as i32)
    }

    fn knn<'py>(
        &self,
        py: Python<'py>,
        query: Py<PyArray1<f32>>,
        k: usize,
        ef: usize,
        start_candidates: usize,
    ) -> PyResult<KnnResult> {
        let q = pyarray1_to_slice(query, Some(self.0.data.dims()))?;
        let (res, stats) = self.0.knn_search(q, k, ef, start_candidates);
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
