use ndarray::{Array1, Array2, ArrayBase, ArrayView1, Dim, OwnedRepr};
use numpy::{
    array, IntoPyArray, PyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyTypeError, prelude::*};
use vecnn::distance::l2;

use crate::utils::{pyarray1_to_slice, KnnResult};

#[pyclass]
pub struct VpTree(vecnn::vp_tree::VpTree);

#[pymethods]
impl VpTree {
    #[new]
    fn new(data: crate::Dataset) -> Self {
        let tree =
            vecnn::vp_tree::VpTree::new(data.as_dyn_dataset(), vecnn::distance::Distance::L2);
        Self(tree)
    }

    #[getter]
    fn num_distance_calculations_in_build(&self) -> PyResult<i32> {
        Ok(self.0.build_stats.num_distance_calculations as i32)
    }

    fn knn<'py>(&self, py: Python<'py>, query: Py<PyArray1<f32>>, k: usize) -> PyResult<KnnResult> {
        let q = pyarray1_to_slice(query, Some(self.0.data.dims()))?;
        let (res, stats) = self.0.knn_search(q, k);
        let indices = ndarray::Array::from_iter(res.iter().map(|e| e.i))
            .into_pyarray_bound(py)
            .unbind();
        let distances = ndarray::Array::from_iter(res.iter().map(|e| e.dist))
            .into_pyarray_bound(py)
            .unbind();
        Ok(KnnResult {
            indices,
            distances,
            num_distance_calculations: stats.num_distance_calculations,
        })
    }

    fn distances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        struct Row {
            idx: usize,
            level: usize,
            dist: f32,
        }

        let mut rows: Vec<Row> = vec![];
        self.0.iter_levels(&mut |level, node| {
            rows.push(Row {
                idx: node.idx,
                level,
                dist: node.dist,
            });
        });
        rows.sort_by(|a, b| a.idx.cmp(&b.idx));

        let mut data_vec: Vec<f32> = vec![]; // level1, dist1, level2, dist2, ...

        for row in rows.iter() {
            data_vec.push(row.level as f32);
            data_vec.push(row.dist as f32);
        }
        let distances: Array2<f32> =
            Array2::from_shape_vec((self.0.nodes.len(), 2), data_vec).unwrap();
        distances.into_pyarray_bound(py)
    }
}
