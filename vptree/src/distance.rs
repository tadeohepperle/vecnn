use std::sync::atomic::{AtomicUsize, Ordering};

use crate::Float;

/// L1 distance
pub struct AbsoluteDiffSum;

/// L2 distance
pub struct SquaredDiffSum;

pub type DistanceFn = fn(&[Float], &[Float]) -> Float;

pub struct DistanceTracker {
    num_calculations: AtomicUsize,
    f: DistanceFn,
}

impl DistanceTracker {
    pub fn new(f: DistanceFn) -> Self {
        DistanceTracker {
            num_calculations: AtomicUsize::new(0),
            f,
        }
    }

    pub fn reset(&mut self) {}

    pub fn num_calculations(&self) -> usize {
        self.num_calculations.load(Ordering::SeqCst)
    }

    #[inline(always)]
    pub fn distance(&self, a: &[Float], b: &[Float]) -> Float {
        self.num_calculations.fetch_add(1, Ordering::Relaxed);
        (self.f)(a, b)
    }
}

pub trait DistanceT: Sized {
    fn distance(a: &[Float], b: &[Float]) -> Float;
}

impl DistanceT for SquaredDiffSum {
    fn distance(a: &[Float], b: &[Float]) -> Float {
        let dims = a.len();
        debug_assert_eq!(a.len(), b.len());
        let mut sum: Float = 0.0;
        for i in 0..dims {
            let d = a[i] - b[i];
            sum += d * d;
        }
        sum
    }
}

impl DistanceT for AbsoluteDiffSum {
    fn distance(a: &[Float], b: &[Float]) -> Float {
        let dims = a.len();
        debug_assert_eq!(a.len(), b.len());
        let mut sum: Float = 0.0;
        for i in 0..dims {
            let d = a[i] - b[i];
            sum += d.abs();
        }
        sum
    }
}
