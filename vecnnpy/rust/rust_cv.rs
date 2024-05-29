use std::{sync::Arc, time::Instant};

use crate::utils::{extend_lifetime, pyarray1_to_slice, KnnResult};
use numpy::{IntoPyArray, PyArray1};
use pyo3::{exceptions::PyTypeError, pyclass, pymethods, Bound, Py, PyResult, Python};
use rust_cv_hnsw::Searcher;
use space::Neighbor;
use vecnn::{
    dataset::DatasetT,
    distance::{DistanceT, SquaredDiffSum},
    vp_tree::Stats,
};

struct Euclidean;
impl space::Metric<&[f32]> for Euclidean {
    type Unit = u32;
    fn distance(&self, a: &&[f32], b: &&[f32]) -> u32 {
        SquaredDiffSum::distance(a, b).to_bits()
    }
}
type MyHnsw = rust_cv_hnsw::Hnsw<Euclidean, &'static [f32], rand_chacha::ChaCha20Rng, 10, 10>;

#[pyclass]
pub struct RustCvHnsw {
    inner: MyHnsw,
    ds: Arc<dyn DatasetT>,
}

#[pymethods]
impl RustCvHnsw {
    #[new]
    fn new<'py>(py: Python<'py>, data: crate::Dataset, ef_construction: usize) -> Self {
        let mut hnsw = rust_cv_hnsw::Hnsw::new_params(
            Euclidean,
            rust_cv_hnsw::Params::new().ef_construction(ef_construction),
        );

        let mut searcher: Searcher<u32> = Default::default();
        let ds = data.as_dyn_dataset();
        for i in 0..ds.len() {
            let q = extend_lifetime(ds.get(i));
            hnsw.insert(q, &mut searcher);
        }

        RustCvHnsw { inner: hnsw, ds }
    }

    fn knn<'py>(
        &self,
        py: Python<'py>,
        query: Py<PyArray1<f32>>,
        k: usize,
        ef: usize,
    ) -> PyResult<KnnResult> {
        let q = pyarray1_to_slice(query, Some(self.ds.dims()))?;
        let mut searcher: Searcher<u32> = Default::default();
        let mut res: Vec<Neighbor<u32>> = vec![
            Neighbor {
                index: usize::MAX,
                distance: u32::MAX,
            };
            k
        ]; // note: `nearest` method attempts to put up to `M` nearest neighbors into `dest`.
           // But M is const. so maybe using k here as the length of the slice is all wrong!!

        let start = Instant::now();
        self.inner
            .nearest(&extend_lifetime(q), ef, &mut searcher, &mut res);
        let stats: Stats = Stats {
            num_distance_calculations: 0,
            duration: start.elapsed(),
        };
        let indices = ndarray::Array::from_iter(res.iter().map(|e| e.index))
            .into_pyarray_bound(py)
            .unbind();
        let distances = ndarray::Array::from_iter(res.iter().map(|e| f32::from_bits(e.distance)))
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
