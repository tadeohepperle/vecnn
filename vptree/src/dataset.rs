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
