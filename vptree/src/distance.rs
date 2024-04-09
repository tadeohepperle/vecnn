use crate::Float;

/// L1 distance
pub struct AbsoluteDiffSum;

/// L2 distance
pub struct SquaredDiffSum;

// pub type DistanceFn = fn(&[Float], &[Float]) -> Float;

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
