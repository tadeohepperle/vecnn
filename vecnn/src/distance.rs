use std::sync::atomic::{AtomicUsize, Ordering};

use crate::Float;

/// L1 distance
pub struct AbsoluteDiffSum;

/// L2 distance
pub struct SquaredDiffSum;

pub struct SquaredDiffSumSIMD;

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
        assert_eq!(dims, b.len());
        let mut sum: Float = 0.0;

        let chunks = dims / 8;
        let rem = dims % 8;
        assert_eq!(dims, chunks * 8 + rem);

        #[inline(always)]
        fn sq(x: Float) -> Float {
            x * x
        }

        for c in 0..chunks {
            let i = c * 8;
            sum += unsafe {
                sq(a.get_unchecked(i) - b.get_unchecked(i))
                    + sq(a.get_unchecked(i + 1) - b.get_unchecked(i + 1))
                    + sq(a.get_unchecked(i + 2) - b.get_unchecked(i + 2))
                    + sq(a.get_unchecked(i + 3) - b.get_unchecked(i + 3))
                    + sq(a.get_unchecked(i + 4) - b.get_unchecked(i + 4))
                    + sq(a.get_unchecked(i + 5) - b.get_unchecked(i + 5))
                    + sq(a.get_unchecked(i + 6) - b.get_unchecked(i + 6))
                    + sq(a.get_unchecked(i + 7) - b.get_unchecked(i + 7))
            }
        }

        for i in (dims - rem)..dims {
            sum += unsafe { sq(a.get_unchecked(i) - b.get_unchecked(i)) }
        }

        sum.sqrt()
    }
}

impl DistanceT for SquaredDiffSumSIMD {
    fn distance(a: &[Float], b: &[Float]) -> Float {
        lance_linalg::distance::l2(a, b)
    }
}

impl DistanceT for AbsoluteDiffSum {
    fn distance(a: &[Float], b: &[Float]) -> Float {
        let dims = a.len();
        assert_eq!(dims, b.len());
        let mut sum: Float = 0.0;
        for i in 0..dims {
            let d = a[i] - b[i];
            sum += d.abs();
        }
        sum
    }
}

// #[inline]
// pub(crate) fn l2_f32(from: &[f32], to: &[f32]) -> f32 {
//     use std::arch::x86_64::*;
//     unsafe {
//         // Get the potion of the vector that is aligned to 32 bytes.
//         let len = from.len() / 8 * 8;
//         let mut sums = _mm256_setzero_ps();
//         for i in (0..len).step_by(8) {
//             let left = _mm256_loadu_ps(from.as_ptr().add(i));
//             let right = _mm256_loadu_ps(to.as_ptr().add(i));
//             let sub = _mm256_sub_ps(left, right);
//             // sum = sub * sub + sum
//             sums = _mm256_fmadd_ps(sub, sub, sums);
//         }
//         // Shift and add vector, until only 1 value left.
//         // sums = [x0-x7], shift = [x4-x7]
//         let mut shift = _mm256_permute2f128_ps(sums, sums, 1);
//         // [x0+x4, x1+x5, ..]
//         sums = _mm256_add_ps(sums, shift);
//         shift = _mm256_permute_ps(sums, 14);
//         sums = _mm256_add_ps(sums, shift);
//         sums = _mm256_hadd_ps(sums, sums);
//         let mut results: [f32; 8] = [0f32; 8];
//         _mm256_storeu_ps(results.as_mut_ptr(), sums);

//         // Remaining unaligned values
//         results[0] += l2_scalar(&from[len..], &to[len..]);
//         results[0]
//     }
// }

// fn l2_scalar(from: &[f32], to: &[f32]) -> f32{

//     for i in from.len(){

//     }
// }
