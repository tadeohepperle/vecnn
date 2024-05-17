use std::{ops::Deref, sync::Arc};

use ndarray::{ArrayBase, Dim, OwnedRepr};
use numpy::{
    ndarray::{Array2, ArrayView2},
    IntoPyArray, PyArray, PyArray2, PyArrayMethods, PyReadonlyArray2, PyReadonlyArrayDyn,
    PyUntypedArrayMethods, ToPyArray,
};
use pyo3::{exceptions::PyTypeError, prelude::*};
use vecnn::dataset::DatasetT;

use crate::utils::extend_lifetime;

#[derive(Clone)]
#[pyclass]
pub struct Dataset(Arc<DatasetInner>);

#[derive(Debug, Clone)]
pub struct DatasetInner {
    data: Py<PyArray2<f32>>,
    view: ArrayView2<'static, f32>,
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

#[pymethods]
impl Dataset {
    #[new]
    fn new<'py>(py: Python<'py>, py_obj: Py<PyArray2<f32>>) -> PyResult<Self> {
        let arr_ref = py_obj.bind(py);
        if !arr_ref.is_contiguous() {
            return Err(PyTypeError::new_err("Array is not contigous"));
        }
        let view: ArrayView2<'static, f32> = unsafe { std::mem::transmute(arr_ref.as_array()) };
        if !view.is_standard_layout() {
            return Err(PyTypeError::new_err("Array is not standard layout"));
        }
        let [len, dims] = unsafe { std::mem::transmute::<_, [usize; 2]>(arr_ref.dims()) };
        Ok(Dataset(Arc::new(DatasetInner {
            data: py_obj,
            view,
            len,
            dims,
        })))
    }

    fn __len__(&self) -> usize {
        self.0.len
    }

    pub fn len(&self) -> usize {
        self.0.len
    }

    pub fn dims(&self) -> usize {
        self.0.dims
    }

    fn to_numpy<'py>(&self, python: Python<'py>) -> Py<PyArray2<f32>> {
        self.0.data.clone_ref(python)
    }
}

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

    fn get<'a>(&'a self, idx: usize) -> &'a [vecnn::Float] {
        let row = self.view.row(idx);
        extend_lifetime(row.as_slice().unwrap())
    }
}
