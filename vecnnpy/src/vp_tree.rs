use ndarray::{Array1, Array2, ArrayBase, Dim, OwnedRepr};
use numpy::{array, IntoPyArray, PyArray, PyArray1, PyArray2};
use pyo3::prelude::*;

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
