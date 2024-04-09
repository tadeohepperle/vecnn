use rand::{thread_rng, Rng};

use crate::Float;

/// A collection of D-dimensional points.
///
/// The points are indexed from `0..len`.
/// Each point is a slice of `dims` floats.
pub trait DatasetT {
    /// Number of points in this dataset
    fn len(&self) -> usize;
    /// Number of dimensions (each a f32) each point has.
    fn dims(&self) -> usize;
    /// Returns the point at index idx.
    /// - The returned slice is expected to have `len == self.dims()`;
    /// - Calling [`DataSetT::get()`] for any index 0..len is valid, higher idx values will panic.
    fn get(&self, idx: usize) -> &[Float];
}

const RANDOM_DATA_SET_DIMS: usize = 768;
pub struct RandomDataset {
    inner: Vec<[Float; RANDOM_DATA_SET_DIMS]>,
}

impl std::fmt::Debug for RandomDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(&format!("RandomDataset ({})", self.len()))
            .finish()
    }
}

impl RandomDataset {
    pub fn new(count: usize) -> Self {
        let mut inner: Vec<[Float; RANDOM_DATA_SET_DIMS]> = Vec::with_capacity(count);
        let mut rng = thread_rng();
        for _ in 0..count {
            let mut p: [Float; RANDOM_DATA_SET_DIMS] = [0.0; RANDOM_DATA_SET_DIMS];
            for f in p.iter_mut() {
                *f = rng.gen();
            }
            inner.push(p);
        }
        Self { inner }
    }
}

impl DatasetT for RandomDataset {
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn get(&self, idx: usize) -> &[Float] {
        &self.inner[idx]
    }

    fn dims(&self) -> usize {
        RANDOM_DATA_SET_DIMS
    }
}
