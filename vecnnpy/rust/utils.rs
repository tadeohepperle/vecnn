use ndarray::ArrayView1;
use numpy::{PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::{exceptions::PyTypeError, pyclass, Bound, Py, PyResult, Python};

pub fn extend_lifetime<'a, T: ?Sized>(e: &'a T) -> &'static T {
    unsafe { &*(e as *const T) }
}

pub fn static_python() -> Python<'static> {
    unsafe { std::mem::transmute(()) }
}

pub fn pyarray1_to_slice<T: numpy::Element>(
    query: Py<PyArray1<T>>,
    expected_len: Option<usize>,
) -> PyResult<&'static [T]> {
    let arr_ref = query.bind(static_python());
    if !arr_ref.is_contiguous() {
        return Err(PyTypeError::new_err("Array is not contigous"));
    }
    let view: ArrayView1<'_, T> = unsafe { arr_ref.as_array() }; //unsafe { std::mem::transmute(arr_ref.as_array()) };
    if !view.is_standard_layout() {
        return Err(PyTypeError::new_err("Array is not standard layout"));
    }
    let q = view.as_slice().unwrap();
    if let Some(expected_len) = expected_len {
        if q.len() != expected_len {
            return Err(PyTypeError::new_err(
                "Query has not the right number of elements",
            ));
        }
    }
    Ok(extend_lifetime(q))
}

#[pyclass]
pub struct KnnResult {
    #[pyo3(get)]
    pub indices: Py<PyArray1<usize>>,
    #[pyo3(get)]
    pub distances: Py<PyArray1<f32>>,
    #[pyo3(get)]
    pub num_distance_calculations: usize,
}
