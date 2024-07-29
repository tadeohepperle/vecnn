//! Credit: Prof. Dr. Erich Schubert (https://www-ai.cs.tu-dortmund.de/PERSONAL/schubert.html)

//! Math operations for k-means
//!
//! This helps with using SSE, AVX, and similar instruction sets.
// #![allow(dead_code)]
// use ndarray::{Array2, ArrayBase, Data, Ix2, RawData};
use num_traits::Float;
use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::*;

/// API to allow different optimization levels of low-level math operations
pub trait KMath<N> {
    /// Squared Euclidean distance of point n to center k
    fn sqdist(v1: &[N], v2: &[N], d: usize) -> N;

    /// Multiplication with scalar v1 = v2 * a
    fn mul(v1: &mut [N], v2: &[N], a: N, d: usize) -> ();

    /// Multiply vector with scalar f inplace, v *= f
    fn mul_assign(v: &mut [N], f: N, d: usize) -> ();

    /// Add vectors inplace: v1 += v2
    fn add_assign(v1: &mut [N], v2: &[N], d: usize) -> ();

    /// Sub vectors inplace: v1 -= v2
    fn sub_assign(v1: &mut [N], v2: &[N], d: usize) -> ();

    /// FMA followed by a multiplication: v1 = (v1 * a + v2) * b
    /// Nonstandard, but helpful here
    fn fmamul(v1: &mut [N], a: N, v2: &[N], b: N, d: usize) -> ();

    /// Dot product
    fn dot(v1: &[N], v2: &[N], d: usize) -> N;
}

/// Basic version
pub(crate) struct DefaultKMath<N> {
    phantom: PhantomData<N>,
}

/// Basic version
impl<N> KMath<N> for DefaultKMath<N>
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy,
{
    #[inline(always)]
    fn sqdist(v1: &[N], v2: &[N], d: usize) -> N {
        assert!(v1.len() == d && v2.len() == d); // bounds check
        (0..d)
            .map(|i| unsafe { *v1.get_unchecked(i) - *v2.get_unchecked(i) })
            .map(|x| x * x)
            .sum()
    }

    #[inline(always)]
    fn mul(v1: &mut [N], v2: &[N], a: N, d: usize) -> () {
        assert!(v1.len() == d && v2.len() == d);
        // for i in 0..d { v1[i] = (v1[i] * a + v2[i]) * b; }
        for i in 0..d {
            unsafe {
                *v1.get_unchecked_mut(i) = *v2.get_unchecked(i) * a;
            }
        }
    }

    #[inline(always)]
    fn mul_assign(v: &mut [N], f: N, d: usize) -> () {
        assert!(v.len() == d);
        for i in 0..d {
            unsafe {
                *v.get_unchecked_mut(i) *= f;
            }
        }
    }

    #[inline(always)]
    fn add_assign(v1: &mut [N], v2: &[N], d: usize) -> () {
        assert!(v1.len() == d && v2.len() == d);
        // for i in 0..d { v1[i] += v2[i]; }
        for i in 0..d {
            unsafe {
                *v1.get_unchecked_mut(i) += *v2.get_unchecked(i);
            }
        }
    }

    #[inline(always)]
    fn sub_assign(v1: &mut [N], v2: &[N], d: usize) -> () {
        assert!(v1.len() == d && v2.len() == d);
        // for i in 0..d { v1[i] -= v2[i]; }
        for i in 0..d {
            unsafe {
                *v1.get_unchecked_mut(i) -= *v2.get_unchecked(i);
            }
        }
    }

    #[inline(always)]
    fn fmamul(v1: &mut [N], a: N, v2: &[N], b: N, d: usize) -> () {
        assert!(v1.len() == d && v2.len() == d);
        // for i in 0..d { v1[i] = (v1[i] * a + v2[i]) * b; }
        for i in 0..d {
            unsafe {
                *v1.get_unchecked_mut(i) = (*v1.get_unchecked(i) * a + *v2.get_unchecked(i)) * b;
            }
        }
    }

    #[inline(always)]
    fn dot(v1: &[N], v2: &[N], d: usize) -> N {
        assert!(v1.len() == d && v2.len() == d); // bounds check
        (0..d)
            .map(|i| unsafe { *v1.get_unchecked(i) * *v2.get_unchecked(i) })
            .sum()
    }
}

/// Unrolled math version
pub(crate) struct UnrollKMath<N, const LANES: usize> {
    phantom: PhantomData<N>,
}

/// Unrolled math version
impl<N, const LANES: usize> KMath<N> for UnrollKMath<N, LANES>
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy,
{
    // Inline always to allow CPU optimization!
    // Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
    #[inline(always)]
    fn sqdist(v1: &[N], v2: &[N], d: usize) -> N {
        assert!(LANES.count_ones() == 1); // must be power of two; compile time assertion
        assert!(v1.len() == d && v2.len() == d); // bounds check
        let sd = d & !(LANES - 1);
        let mut vsum = [N::zero(); LANES];
        for i in (0..sd).step_by(LANES) {
            let (vv, cc) = (&v1[i..(i + LANES)], &v2[i..(i + LANES)]);
            for j in 0..LANES {
                unsafe {
                    let x = *vv.get_unchecked(j) - *cc.get_unchecked(j);
                    // emulated: *vsum.get_unchecked_mut(j) = x.mul_add(x, *vsum.get_unchecked(j)); // FMA
                    *vsum.get_unchecked_mut(j) += x * x;
                }
            }
        }
        let mut sum = vsum.iter().copied().sum::<N>();
        if d > sd {
            sum += (sd..d)
                .map(|i| unsafe { *v1.get_unchecked(i) - *v2.get_unchecked(i) })
                .map(|x| x * x)
                .sum()
        }
        sum
    }

    // Inline always to allow CPU optimization!
    // Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
    #[inline(always)]
    fn mul(v1: &mut [N], v2: &[N], a: N, d: usize) -> () {
        assert!(LANES.count_ones() == 1); // must be power of two; compile time assertion
        assert!(v1.len() == d && v2.len() == d);
        let sd = d & !(LANES - 1);
        for i in (0..sd).step_by(LANES) {
            let (b1, b2) = (&mut v1[i..(i + LANES)], &v2[i..(i + LANES)]);
            for j in 0..LANES {
                unsafe {
                    *b1.get_unchecked_mut(j) = *b2.get_unchecked(j) * a;
                }
            }
        }
        for i in sd..d {
            unsafe {
                *v1.get_unchecked_mut(i) = *v2.get_unchecked(i) * a;
            }
        }
    }

    // Inline always to allow CPU optimization!
    // Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
    #[inline(always)]
    fn mul_assign(v: &mut [N], f: N, d: usize) -> () {
        assert!(LANES.count_ones() == 1); // must be power of two; compile time assertion
        assert!(v.len() == d);
        let sd = d & !(LANES - 1);
        for i in (0..sd).step_by(LANES) {
            let v2 = &mut v[i..(i + LANES)];
            for j in 0..LANES {
                unsafe {
                    *v2.get_unchecked_mut(j) *= f;
                }
            }
        }
        for i in sd..d {
            unsafe {
                *v.get_unchecked_mut(i) *= f;
            }
        }
    }

    // Inline always to allow CPU optimization!
    // Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
    #[inline(always)]
    fn add_assign(v1: &mut [N], v2: &[N], d: usize) -> () {
        assert!(LANES.count_ones() == 1); // must be power of two; compile time assertion
        assert!(v1.len() == d && v2.len() == d);
        let sd = d & !(LANES - 1);
        for i in (0..sd).step_by(LANES) {
            let (b1, b2) = (&mut v1[i..(i + LANES)], &v2[i..(i + LANES)]);
            for j in 0..LANES {
                unsafe {
                    *b1.get_unchecked_mut(j) += *b2.get_unchecked(j);
                }
            }
        }
        for i in sd..d {
            unsafe {
                *v1.get_unchecked_mut(i) += *v2.get_unchecked(i);
            }
        }
    }

    // Inline always to allow CPU optimization!
    // Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
    #[inline(always)]
    fn sub_assign(v1: &mut [N], v2: &[N], d: usize) -> () {
        assert!(LANES.count_ones() == 1); // must be power of two; compile time assertion
        assert!(v1.len() == d && v2.len() == d);
        let sd = d & !(LANES - 1);
        for i in (0..sd).step_by(LANES) {
            let (b1, b2) = (&mut v1[i..(i + LANES)], &v2[i..(i + LANES)]);
            for j in 0..LANES {
                unsafe {
                    *b1.get_unchecked_mut(j) -= *b2.get_unchecked(j);
                }
            }
        }
        for i in sd..d {
            unsafe {
                *v1.get_unchecked_mut(i) -= *v2.get_unchecked(i);
            }
        }
    }

    // Inline always to allow CPU optimization!
    // Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
    #[inline(always)]
    fn fmamul(v1: &mut [N], a: N, v2: &[N], b: N, d: usize) -> () {
        assert!(LANES.count_ones() == 1); // must be power of two; compile time assertion
        assert!(v1.len() == d && v2.len() == d);
        let sd = d & !(LANES - 1);
        for i in (0..sd).step_by(LANES) {
            let (b1, b2) = (&mut v1[i..(i + LANES)], &v2[i..(i + LANES)]);
            for j in 0..LANES {
                unsafe {
                    *b1.get_unchecked_mut(j) =
                        (*b1.get_unchecked_mut(j) * a + *b2.get_unchecked(j)) * b;
                }
            }
        }
        for i in sd..d {
            unsafe {
                *v1.get_unchecked_mut(i) = (*v1.get_unchecked(i) * a + *v2.get_unchecked(i)) * b;
            }
        }
    }

    // Inline always to allow CPU optimization!
    // Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
    #[inline(always)]
    fn dot(v1: &[N], v2: &[N], d: usize) -> N {
        assert!(v1.len() == d && v2.len() == d); // bounds check
        if LANES > 1 {
            assert!(LANES.count_ones() == 1); // must be power of two; compile time assertion
            let sd = d & !(LANES - 1);
            let mut vsum = [N::zero(); LANES];
            for i in (0..sd).step_by(LANES) {
                let (vv, cc) = (&v1[i..(i + LANES)], &v2[i..(i + LANES)]);
                for j in 0..LANES {
                    unsafe {
                        let (a, b) = (*vv.get_unchecked(j), *cc.get_unchecked(j));
                        *vsum.get_unchecked_mut(j) += a * b;
                    }
                }
            }
            let mut sum = vsum.iter().copied().sum::<N>();
            if d > sd {
                sum += (sd..d)
                    .map(|i| unsafe { *v1.get_unchecked(i) * *v2.get_unchecked(i) })
                    .sum()
            }
            sum
        } else {
            (0..d)
                .map(|i| unsafe { *v1.get_unchecked(i) * *v2.get_unchecked(i) })
                .sum()
        }
    }
}

/// AVX2 accelerated math
pub(crate) struct AVX2KMath<N, const LANES: usize> {
    phantom: PhantomData<N>,
}

/// AVX2 accelerated math
impl<N, const LANES: usize> KMath<N> for AVX2KMath<N, LANES>
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy,
{
    // Inline always to allow CPU optimization!
    // Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
    #[inline(always)]
    fn sqdist(v1: &[N], v2: &[N], d: usize) -> N {
        assert!(v1.len() == d && v2.len() == d); // bounds check
        if LANES > 1 {
            assert!(LANES.count_ones() == 1); // must be power of two; compile time assertion
            let sd = d & !(LANES - 1);
            let mut vsum = [N::zero(); LANES];
            for i in (0..sd).step_by(LANES) {
                let (vv, cc) = (&v1[i..(i + LANES)], &v2[i..(i + LANES)]);
                for j in 0..LANES {
                    unsafe {
                        let x = *vv.get_unchecked(j) - *cc.get_unchecked(j);
                        *vsum.get_unchecked_mut(j) = x.mul_add(x, *vsum.get_unchecked(j));
                        // FMA
                        //*vsum.get_unchecked_mut(j) += x * x;
                    }
                }
            }
            let mut sum = vsum.iter().copied().sum::<N>();
            if d > sd {
                sum += (sd..d)
                    .map(|i| unsafe { *v1.get_unchecked(i) - *v2.get_unchecked(i) })
                    .map(|x| x * x)
                    .sum()
            }
            sum
        } else {
            (0..d)
                .map(|i| unsafe { *v1.get_unchecked(i) - *v2.get_unchecked(i) })
                .map(|x| x * x)
                .sum()
        }
    }

    // Inline always to allow CPU optimization!
    // Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
    #[inline(always)]
    fn mul(v1: &mut [N], v2: &[N], a: N, d: usize) -> () {
        assert!(v1.len() == d && v2.len() == d);
        if LANES > 1 {
            assert!(LANES.count_ones() == 1); // must be power of two; compile time assertion
            let sd = d & !(LANES - 1);
            for i in (0..sd).step_by(LANES) {
                let (b1, b2) = (&mut v1[i..(i + LANES)], &v2[i..(i + LANES)]);
                for j in 0..LANES {
                    unsafe {
                        *b1.get_unchecked_mut(j) = *b2.get_unchecked(j) * a;
                    }
                }
            }
            for i in sd..d {
                unsafe {
                    *v1.get_unchecked_mut(i) = *v2.get_unchecked(i) * a;
                }
            }
        } else {
            for i in 0..d {
                unsafe {
                    *v1.get_unchecked_mut(i) = *v2.get_unchecked(i) * a;
                }
            }
        }
    }

    // Inline always to allow CPU optimization!
    // Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
    #[inline(always)]
    fn mul_assign(v: &mut [N], f: N, d: usize) -> () {
        assert!(v.len() == d);
        if LANES > 1 {
            assert!(LANES.count_ones() == 1); // must be power of two; compile time assertion
            let sd = d & !(LANES - 1);
            for i in (0..sd).step_by(LANES) {
                let v2 = &mut v[i..(i + LANES)];
                for j in 0..LANES {
                    unsafe {
                        *v2.get_unchecked_mut(j) *= f;
                    }
                }
            }
            for i in sd..d {
                unsafe {
                    *v.get_unchecked_mut(i) *= f;
                }
            }
        } else {
            for i in 0..d {
                unsafe {
                    *v.get_unchecked_mut(i) *= f;
                }
            }
        }
    }

    // Inline always to allow CPU optimization!
    // Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
    #[inline(always)]
    fn add_assign(v1: &mut [N], v2: &[N], d: usize) -> () {
        assert!(v1.len() == d && v2.len() == d);
        assert!(LANES.count_ones() == 1); // must be power of two; compile time assertion
        if LANES > 1 {
            let sd = d & !(LANES - 1);
            for i in (0..sd).step_by(LANES) {
                let (b1, b2) = (&mut v1[i..(i + LANES)], &v2[i..(i + LANES)]);
                for j in 0..LANES {
                    unsafe {
                        *b1.get_unchecked_mut(j) += *b2.get_unchecked(j);
                    }
                }
            }
            for i in sd..d {
                unsafe {
                    *v1.get_unchecked_mut(i) += *v2.get_unchecked(i);
                }
            }
        } else {
            for i in 0..d {
                unsafe {
                    *v1.get_unchecked_mut(i) += *v2.get_unchecked(i);
                }
            }
        }
    }

    // Inline always to allow CPU optimization!
    // Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
    #[inline(always)]
    fn sub_assign(v1: &mut [N], v2: &[N], d: usize) -> () {
        assert!(v1.len() == d && v2.len() == d);
        if LANES > 1 {
            assert!(LANES.count_ones() == 1); // must be power of two; compile time assertion
            let sd = d & !(LANES - 1);
            for i in (0..sd).step_by(LANES) {
                let (b1, b2) = (&mut v1[i..(i + LANES)], &v2[i..(i + LANES)]);
                for j in 0..LANES {
                    unsafe {
                        *b1.get_unchecked_mut(j) -= *b2.get_unchecked(j);
                    }
                }
            }
            for i in sd..d {
                unsafe {
                    *v1.get_unchecked_mut(i) -= *v2.get_unchecked(i);
                }
            }
        } else {
            for i in 0..d {
                unsafe {
                    *v1.get_unchecked_mut(i) -= *v2.get_unchecked(i);
                }
            }
        }
    }

    // Inline always to allow CPU optimization!
    // Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
    #[inline(always)]
    fn fmamul(v1: &mut [N], a: N, v2: &[N], b: N, d: usize) -> () {
        assert!(v1.len() == d && v2.len() == d);
        if LANES > 1 {
            assert!(LANES.count_ones() == 1); // must be power of two; compile time assertion
            let sd = d & !(LANES - 1);
            for i in (0..sd).step_by(LANES) {
                let (b1, b2) = (&mut v1[i..(i + LANES)], &v2[i..(i + LANES)]);
                for j in 0..LANES {
                    unsafe {
                        *b1.get_unchecked_mut(j) =
                            b1.get_unchecked_mut(j).mul_add(a, *b2.get_unchecked(j)) * b;
                    }
                }
            }
            for i in sd..d {
                unsafe {
                    *v1.get_unchecked_mut(i) =
                        v1.get_unchecked(i).mul_add(a, *v2.get_unchecked(i)) * b;
                }
            }
        } else {
            for i in 0..d {
                unsafe {
                    *v1.get_unchecked_mut(i) =
                        v1.get_unchecked(i).mul_add(a, *v2.get_unchecked(i)) * b;
                }
            }
        }
    }

    // Inline always to allow CPU optimization!
    // Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
    #[inline(always)]
    fn dot(v1: &[N], v2: &[N], d: usize) -> N {
        assert!(v1.len() == d && v2.len() == d); // bounds check
        if LANES > 1 {
            assert!(LANES.count_ones() == 1); // must be power of two; compile time assertion
            let sd = d & !(LANES - 1);
            let mut vsum = [N::zero(); LANES];
            for i in (0..sd).step_by(LANES) {
                let (vv, cc) = (&v1[i..(i + LANES)], &v2[i..(i + LANES)]);
                for j in 0..LANES {
                    unsafe {
                        let (a, b) = (*vv.get_unchecked(j), *cc.get_unchecked(j));
                        *vsum.get_unchecked_mut(j) = a.mul_add(b, *vsum.get_unchecked(j));
                        // FMA
                        //*vsum.get_unchecked_mut(j) += a * b;
                    }
                }
            }
            let mut sum = vsum.iter().copied().sum::<N>();
            if d > sd {
                sum += (sd..d)
                    .map(|i| unsafe { *v1.get_unchecked(i) * *v2.get_unchecked(i) })
                    .sum()
            }
            sum
        } else {
            (0..d)
                .map(|i| unsafe { *v1.get_unchecked(i) * *v2.get_unchecked(i) })
                .sum()
        }
    }
}

/// Centers storage
pub struct Centers<N> {
    k: usize,
    d: usize,
    centers: Vec<N>,
}

impl<N> Centers<N>
where
    N: Float + Copy,
{
    #[inline]
    pub fn new(k: usize, d: usize) -> Self {
        Self {
            k: k,
            d: d,
            centers: vec![N::zero(); k * d],
        }
    }

    #[inline(always)]
    pub fn center(&self, i: usize) -> &[N] {
        //&self.centers[i*self.d..i*self.d+self.d]
        unsafe { std::slice::from_raw_parts(self.centers.as_ptr().add(i * self.d), self.d) }
    }

    #[inline(always)]
    pub fn center_mut(&mut self, i: usize) -> &mut [N] {
        //&mut self.centers[i*self.d..i*self.d+self.d]
        unsafe { std::slice::from_raw_parts_mut(self.centers.as_mut_ptr().add(i * self.d), self.d) }
    }
}
