use std::sync::atomic::{AtomicUsize, Ordering};

use nanoserde::{DeJson, SerJson};

use crate::{schubert_distance::KMath, Float};

#[derive(Debug, Clone, Copy, PartialEq, SerJson, DeJson, Eq)]
pub enum Distance {
    L2,
    Dot,
    Cos,
}

impl Distance {
    pub fn to_fn(&self) -> DistanceFn {
        match self {
            Distance::L2 => l2,
            Distance::Dot => dot,
            Distance::Cos => cos,
        }
    }
}

pub type DistanceFn = fn(&[Float], &[Float]) -> Float;

pub struct DistanceTracker {
    #[cfg(not(feature = "no_ndc_tracking"))]
    num_calculations: AtomicUsize,
    f: DistanceFn,
}

impl DistanceTracker {
    pub fn new(f: Distance) -> Self {
        DistanceTracker {
            #[cfg(not(feature = "no_ndc_tracking"))]
            num_calculations: AtomicUsize::new(0),
            f: f.to_fn(),
        }
    }

    pub fn reset(&mut self) {}

    pub fn num_calculations(&self) -> usize {
        #[cfg(not(feature = "no_ndc_tracking"))]
        return self.num_calculations.load(Ordering::SeqCst);
        #[cfg(feature = "no_ndc_tracking")]
        return 0;
    }

    #[inline(always)]
    pub fn distance(&self, a: &[Float], b: &[Float]) -> Float {
        #[cfg(not(feature = "no_ndc_tracking"))]
        {
            self.num_calculations.fetch_add(1, Ordering::Relaxed);
        }
        (self.f)(a, b)
    }
}

pub fn cos(a: &[Float], b: &[Float]) -> Float {
    let dims = a.len();
    assert_eq!(dims, b.len());
    let mut dot: Float = 0.0;
    let mut xx: Float = 0.0;
    let mut yy: Float = 0.0;

    let chunks = dims / 8;
    let rem = dims % 8;
    assert_eq!(dims, chunks * 8 + rem);

    macro_rules! accum {
        ($idx:expr) => {
            let &x = unsafe { a.get_unchecked($idx) };
            let &y = unsafe { b.get_unchecked($idx) };
            dot += x * y;
            xx += x * x;
            yy += y * y;
        };
    }

    for c in 0..chunks {
        let i = c * 8;
        accum!(i);
        accum!(i + 1);
        accum!(i + 2);
        accum!(i + 3);
        accum!(i + 4);
        accum!(i + 5);
        accum!(i + 6);
        accum!(i + 7);
    }

    for i in (dims - rem)..dims {
        accum!(i);
    }
    let dot = dot / (xx * yy).sqrt();
    1.0 - dot // to make it a distance not a similarity
}

pub fn dot(a: &[Float], b: &[Float]) -> Float {
    let dims = a.len();
    assert_eq!(dims, b.len());

    // (~76ms / IDK)
    return 1.0 - crate::schubert_distance::UnrollKMath::<f32, 16>::dot(a, b, dims);
    // 1.0 - x to make it a distance

    // return (0..dims)
    //     .map(|i| {
    //         a[i] * b[i]

    //         // unsafe { *a.get_unchecked(i) * *b.get_unchecked(i) }
    //     })
    //     .sum();

    // return crate::schubert_distance::UnrollKMath::<f32, 4>::dot(a, b, dims); // (~100ms / IDK)
    // return crate::schubert_distance::AVX2KMath::<f32, 32>::dot(a, b, dims); // (~700ms / IDK)

    // return crate::schubert_distance::DefaultKMath::<f32>::dot(a, b, dims); // (~430ms / IDK)

    // // Manual implementation following: (~220ms / IDK)

    // let mut dot: Float = 0.0;

    // let chunks = dims / 8;
    // let rem = dims % 8;
    // assert_eq!(dims, chunks * 8 + rem);

    // for c in 0..chunks {
    //     let i = c * 8;
    //     dot += unsafe {
    //         a.get_unchecked(i) * b.get_unchecked(i)
    //             + a.get_unchecked(i + 1) * b.get_unchecked(i + 1)
    //             + a.get_unchecked(i + 2) * b.get_unchecked(i + 2)
    //             + a.get_unchecked(i + 3) * b.get_unchecked(i + 3)
    //             + a.get_unchecked(i + 4) * b.get_unchecked(i + 4)
    //             + a.get_unchecked(i + 5) * b.get_unchecked(i + 5)
    //             + a.get_unchecked(i + 6) * b.get_unchecked(i + 6)
    //             + a.get_unchecked(i + 7) * b.get_unchecked(i + 7)
    //     }
    // }

    // for i in (dims - rem)..dims {
    //     dot += unsafe { a.get_unchecked(i) * b.get_unchecked(i) }
    // }

    // 1.0 - dot // to make it a distance not a similarity
}

pub fn l2(a: &[Float], b: &[Float]) -> Float {
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

pub fn l1(a: &[Float], b: &[Float]) -> Float {
    let dims = a.len();
    assert_eq!(dims, b.len());
    let mut sum: Float = 0.0;
    for i in 0..dims {
        let d = a[i] - b[i];
        sum += d.abs();
    }
    sum
}
