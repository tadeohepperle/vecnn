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

        sum
    }

    // fn distance(a: &[Float], b: &[Float]) -> Float {
    //     assert_eq!(a.len(), b.len());
    //     let mut sum: Float = 0.0;

    //     #[inline(always)]
    //     fn sq(x: Float) -> Float {
    //         x * x
    //     }

    //     let mut iter = a.iter().zip(b.iter()).array_chunks::<8>();
    //     while let Some(
    //         [(a1, b1), (a2, b2), (a3, b3), (a4, b4), (a5, b5), (a6, b6), (a7, b7), (a8, b8)],
    //     ) = iter.next()
    //     {
    //         let e1 = sq(a1 - b1);
    //         let e2 = sq(a2 - b2);
    //         let e3 = sq(a3 - b3);
    //         let e4 = sq(a4 - b4);
    //         let e5 = sq(a5 - b5);
    //         let e6 = sq(a6 - b6);
    //         let e7 = sq(a7 - b7);
    //         let e8 = sq(a8 - b8);
    //         sum += e1 + e2 + e3 + e4 + e5 + e6 + e7 + e8;
    //     }
    //     if let Some(rem) = iter.into_remainder() {
    //         for (a, b) in rem {
    //             sum += sq(*a - *b);
    //         }
    //     }

    //     sum
    // }
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
