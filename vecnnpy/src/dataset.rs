use std::{ops::Deref, sync::Arc};

use ndarray::{ArrayBase, Dim, OwnedRepr};
use numpy::{
    ndarray::{Array2, ArrayView2},
    IntoPyArray, PyArray, PyArray2, PyReadonlyArray2, PyReadonlyArrayDyn, ToPyArray,
};
use pyo3::prelude::*;
use vecnn_vptree::dataset::DatasetT;

pub type Arr2d = ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>;

#[derive(Clone)]
#[pyclass]
pub struct Dataset(Arc<DatasetInner>);

#[derive(Debug, Clone)]
pub struct DatasetInner {
    data: Arr2d,
    len: usize,
    dims: usize,
}

unsafe impl Send for Dataset {}

impl Deref for Dataset {
    type Target = DatasetInner;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for DatasetInner {
    type Target = Arr2d;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

#[pymethods]
impl Dataset {
    #[new]
    fn new<'py>(arr: PyReadonlyArray2<'py, f32>) -> PyResult<Self> {
        let arr_ref = arr.as_array();
        let data = arr_ref.as_standard_layout().to_owned();

        let [len, dims] = *data.shape() else {
            panic!("Expected array with two dimensitons!")
        };

        Ok(Self(Arc::new(DatasetInner { data, len, dims })))
    }

    fn __len__(&self) -> usize {
        self.0.len
    }

    fn desc(&self) -> String {
        format!("DATA: {:?}", self.0.data.shape())
    }

    fn row(&self, idx: usize) -> String {
        format!("{:?}", self.get(idx))
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        self.data.to_pyarray_bound(py)
    }
}

// fn new(arr: ArrayView2<'_, f32>) -> Dataset {
//     Dataset {}
// }

impl Dataset {
    pub fn as_dyn_dataset(&self) -> Arc<dyn DatasetT> {
        self.0.clone()
    }
}

impl DatasetT for DatasetInner {
    fn len(&self) -> usize {
        self.len
    }

    fn dims(&self) -> usize {
        self.dims
    }

    fn get<'a>(&'a self, idx: usize) -> &'a [vecnn_vptree::Float] {
        let row = self.data.row(idx);
        let slice = row
            .as_slice()
            .expect("we converted it into `as_standard_layout` when constructing the Dataset");
        // std::fs::write("./log.txt", format!("{:?}", slice));
        // extend the lifetime of the row:
        unsafe { std::mem::transmute(slice) }
    }
}
