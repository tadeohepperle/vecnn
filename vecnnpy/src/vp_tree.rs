use ndarray::{Array1, Array2, ArrayBase, ArrayView1, Dim, OwnedRepr};
use numpy::{
    array, IntoPyArray, PyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::dataset::Arr2d;

#[pyclass]
pub struct VpTree(vecnn_vptree::vp_tree::VpTree);

#[pymethods]
impl VpTree {
    #[new]
    fn new(data: crate::Dataset) -> Self {
        let tree = vecnn_vptree::vp_tree::VpTree::new(
            data.as_dyn_dataset(),
            vecnn_vptree::distance::SquaredDiffSum,
        );
        Self(tree)
    }

    fn knn<'py>(
        &self,
        py: Python<'py>,
        query: Py<PyArray1<f32>>,
        k: usize,
    ) -> PyResult<(Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<f32>>)> {
        let arr_ref = query.bind(py);
        if !arr_ref.is_contiguous() {
            return Err(PyTypeError::new_err("Array is not contigous"));
        }
        let view: ArrayView1<'static, f32> = unsafe { std::mem::transmute(arr_ref.as_array()) };
        if !view.is_standard_layout() {
            return Err(PyTypeError::new_err("Array is not standard layout"));
        }

        let q = view.as_slice().unwrap();
        if q.len() != self.0.data.dims() {
            return Err(PyTypeError::new_err(
                "Query has not the right number of elements",
            ));
        }

        let res = self.0.knn_search(q, k);

        let idices = ndarray::Array::from_iter(res.iter().map(|e| e.idx)).into_pyarray_bound(py);
        let distances =
            ndarray::Array::from_iter(res.iter().map(|e| e.dist)).into_pyarray_bound(py);

        Ok((idices, distances))
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
