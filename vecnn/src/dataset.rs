use std::{alloc::Layout, fmt::Debug, ops::Deref, sync::Arc};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::Float;

/// A collection of D-dimensional points.
///
/// The points are indexed from `0..len`.
/// Each point is a slice of `dims` floats.
pub trait DatasetT: Send + Sync + 'static + Debug {
    /// Number of points in this dataset
    fn len(&self) -> usize;
    /// Number of dimensions (each a f32) each point has.
    fn dims(&self) -> usize;
    /// Returns the point at index idx.
    /// - The returned slice is expected to have `len == self.dims()`;
    /// - Calling [`DataSetT::get()`] for any index 0..len is valid, higher idx values will panic.
    fn get(&self, id: usize) -> &[Float];
}

impl<T, const D: usize> DatasetT for T
where
    T: Deref<Target = [[Float; D]]> + Send + Sync + 'static + Debug,
{
    fn len(&self) -> usize {
        self.deref().len()
    }

    fn dims(&self) -> usize {
        D
    }

    fn get(&self, idx: usize) -> &[Float] {
        &self[idx]
    }
}

impl<const D: usize> DatasetT for [[Float; D]] {
    fn len(&self) -> usize {
        self.len()
    }

    fn dims(&self) -> usize {
        D
    }

    fn get(&self, idx: usize) -> &[Float] {
        &self[idx]
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct FlatDataSet {
    pub dims: usize,
    pub len: usize,
    pub data: Vec<Float>,
}

impl DatasetT for FlatDataSet {
    fn len(&self) -> usize {
        self.len
    }

    fn dims(&self) -> usize {
        self.dims
    }

    fn get(&self, id: usize) -> &[Float] {
        let start = id * self.dims;
        let end = (id + 1) * self.dims;
        &self.data[start..end]
    }
}

impl FlatDataSet {
    pub fn new_random(len: usize, dims: usize) -> Arc<dyn DatasetT> {
        let floats = len * dims;
        let mut data: Vec<Float> = Vec::with_capacity(floats);
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        data.extend((0..floats).map(|_| rng.gen::<f32>()));
        Arc::new(FlatDataSet { dims, len, data })
    }
}
