use std::{fmt::Debug, ops::Deref, sync::Arc};

use rand::{thread_rng, Rng};

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
    fn get(&self, idx: usize) -> &[Float];
}

const RANDOM_DATA_SET_DIMS: usize = 768;
pub fn random_data_set_768(count: usize) -> Arc<dyn DatasetT> {
    let mut data: Vec<[Float; RANDOM_DATA_SET_DIMS]> = Vec::with_capacity(count);
    let mut rng = thread_rng();
    for _ in 0..count {
        let mut p: [Float; RANDOM_DATA_SET_DIMS] = [0.0; RANDOM_DATA_SET_DIMS];
        for f in p.iter_mut() {
            *f = rng.gen();
        }
        data.push(p);
    }
    Arc::new(data)
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
