use std::{sync::Arc, time::Instant};

use crate::utils::{extend_lifetime, pyarray1_to_slice, KnnResult};
use jpboth_hnsw::api::AnnT;
use numpy::{IntoPyArray, PyArray1};
use pyo3::{exceptions::PyTypeError, pyclass, pymethods, Bound, Py, PyResult, Python};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rust_cv_hnsw::Searcher;
use space::Neighbor;
use vecnn::{
    dataset::DatasetT,
    distance::{dot, l2},
    utils::Stats,
};

type ParameterizedJpBothHnsw = jpboth_hnsw::hnsw::Hnsw<'static, f32, DistDot>;

struct DistDot;

impl jpboth_hnsw::anndists::dist::Distance<f32> for DistDot {
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        1.0 + dot(a, b)
    }
}

#[pyclass]
pub struct JpBothHnsw {
    inner: ParameterizedJpBothHnsw,
    ds: Arc<dyn DatasetT>,
}

#[pymethods]
impl JpBothHnsw {
    #[new]
    fn new(
        data: crate::Dataset,
        ef_construction: usize,
        m_max: usize,
        multi_threaded: bool,
    ) -> Self {
        let max_layer = 10;
        let hnsw: ParameterizedJpBothHnsw =
            jpboth_hnsw::hnsw::Hnsw::new(m_max, data.len(), max_layer, ef_construction, DistDot);
        let ds = data.as_dyn_dataset();
        if multi_threaded {
            (0..data.len())
                .into_par_iter()
                .for_each(|id| hnsw.insert_slice((ds.get(id), id)));
        } else {
            for id in 0..ds.len() {
                hnsw.insert_slice((ds.get(id), id))
            }
        }

        JpBothHnsw { inner: hnsw, ds }
    }

    fn knn<'py>(
        &self,
        py: Python<'py>,
        query: Py<PyArray1<f32>>,
        k: usize,
        ef: usize,
    ) -> PyResult<KnnResult> {
        let q = pyarray1_to_slice(query, Some(self.ds.dims()))?;

        let start = Instant::now();
        let ef = ef.max(k);

        let neighbors = self.inner.search(&extend_lifetime(q), k, ef);
        let indices = ndarray::Array::from_iter(neighbors.iter().map(|e| e.d_id))
            .into_pyarray_bound(py)
            .unbind();
        let distances = ndarray::Array::from_iter(neighbors.iter().map(|e| e.distance))
            .into_pyarray_bound(py)
            .unbind();

        let stats: Stats = Stats {
            num_distance_calculations: 0,
            duration: start.elapsed(),
        };
        Ok(KnnResult {
            indices,
            distances,
            num_distance_calculations: stats.num_distance_calculations,
        })
    }
}
