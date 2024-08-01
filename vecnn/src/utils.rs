use rand::Rng;

use crate::{
    dataset::{DatasetT, FlatDataSet},
    distance::{l2, DistanceFn},
    hnsw::DistAnd,
    Float,
};
use std::{collections::BinaryHeap, mem::ManuallyDrop, ptr, sync::Arc};

pub fn linear_knn_search(
    data: &dyn DatasetT,
    q_data: &[f32],
    k: usize,
    distance: DistanceFn,
) -> Vec<DistAnd<usize>> {
    assert_eq!(q_data.len(), data.dims());
    // this stores the item with the greatest distance in the root (first element)
    let mut knn_heap = KnnHeap::new(k);
    for id in 0..data.len() {
        let i_data = data.get(id);
        let dist = distance(q_data, i_data);
        knn_heap.maybe_add(id, dist)
    }
    knn_heap.as_sorted_vec()
}

/// just draw numbers 0..X where X <=9 into the grid to test stuff:
pub fn simple_test_set() -> Arc<dyn DatasetT> {
    let str = "
      3
    4
                            7 
                   2
        9
    6                      0
              5
                     8  1   
    ";

    let mut pts: Vec<(usize, [f32; 2])> = vec![];
    let mut y = 0.0;
    for line in str.lines() {
        let mut x = 0.0;
        for ch in line.chars() {
            if let Some(idx) = ch.to_digit(10) {
                pts.push((idx as usize, [x, y]))
            }
            x += 1.0;
        }
        y += 2.0;
    }
    pts.sort_by(|a, b| a.0.cmp(&b.0));
    for (i, (e, _)) in pts.iter().enumerate() {
        assert_eq!(i, *e)
    }
    let pts: Vec<[f32; 2]> = pts.into_iter().map(|e| e.1).collect();
    Arc::new(pts)
}

pub fn random_data_set(len: usize, dims: usize) -> Arc<dyn DatasetT> {
    FlatDataSet::new_random(len, dims)
}

pub struct KnnHeap {
    k: usize,
    heap: BinaryHeap<DistAnd<usize>>,
}

impl KnnHeap {
    pub fn new(k: usize) -> Self {
        KnnHeap {
            k,
            heap: BinaryHeap::new(),
        }
    }

    pub fn maybe_add(&mut self, id: usize, dist: Float) {
        if self.heap.len() < self.k {
            self.heap.push(DistAnd(dist, id));
        } else {
            let worst_neighbor = self.heap.peek().unwrap();
            if dist < worst_neighbor.dist() {
                self.heap.pop();
                self.heap.push(DistAnd(dist, id));
            } else {
                // do nothing, not inserted
            }
        }
    }

    pub fn worst_nn_dist(&self) -> f32 {
        self.heap.peek().unwrap().dist()
    }

    pub fn as_sorted_vec(self) -> Vec<DistAnd<usize>> {
        let mut res = Vec::from(self.heap);
        res.sort();
        res
    }

    pub fn is_full(&self) -> bool {
        self.heap.len() >= self.k
    }
}

#[cfg(test)]
mod test {
    use crate::distance::l2;

    use super::{linear_knn_search, simple_test_set};

    #[test]
    fn linear_knn() {
        let data = simple_test_set();

        let res = linear_knn_search(&*data, data.get(0), 4, l2);
        let res: Vec<usize> = res.into_iter().map(|e| e.1).collect();
        assert_eq!(res, vec![0, 1, 7, 8]);
    }
}

#[inline(always)]
pub fn extend_lifetime<T: ?Sized>(e: &T) -> &'static T {
    unsafe { &*(e as *const T) }
}

pub trait BinaryHeapExt {
    type Item: Ord;
    // returns true if inserted
    fn insert_if_better(&mut self, item: Self::Item, max_len: usize) -> bool;
}

impl<T: Ord> BinaryHeapExt for BinaryHeap<T> {
    type Item = T;

    fn insert_if_better(&mut self, item: Self::Item, max_len: usize) -> bool {
        if self.len() < max_len {
            self.push(item);
            return true;
        } else {
            let mut worst = self.peek_mut().unwrap();
            if item < *worst {
                *worst = item;
                return true;
            } else {
                return false;
            }
        }
    }
}

pub use binary_heap::SliceBinaryHeap;

mod binary_heap {
    use std::{fmt::Debug, mem::ManuallyDrop, ptr};

    /// Has the property that the max item is always kept as the first element of the slice
    pub struct SliceBinaryHeap<'a, T: Ord> {
        /// Note: slices len is heaps capacity.
        slice: &'a mut [T],
        len: usize,
    }

    impl<'a, T: Ord + Debug> Debug for SliceBinaryHeap<'a, T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_list()
                .entries(self.slice[..self.len].iter())
                .finish()
        }
    }

    impl<T: Ord> SliceBinaryHeap<'static, T> {
        #[inline(always)]
        pub unsafe fn new_uninitialized() -> Self {
            SliceBinaryHeap {
                slice: unsafe {
                    &mut *std::ptr::slice_from_raw_parts_mut::<T>(std::ptr::null_mut(), 0)
                },
                len: 0,
            }
        }
    }

    impl<'a, T: Ord> SliceBinaryHeap<'a, T> {
        pub fn new(slice: &'a mut [T]) -> Self {
            assert!(slice.len() != 0);
            SliceBinaryHeap { slice, len: 0 }
        }

        pub fn iter(&self) -> std::slice::Iter<T> {
            self.slice[..self.len].iter()
        }

        pub fn as_slice(&self) -> &[T] {
            &self.slice[..self.len]
        }

        pub fn clear(&mut self) {
            self.len = 0
        }

        /// returns true if item was included.
        pub fn push_asserted(&mut self, item: T) {
            assert!(self.len < self.slice.len());
            let old_len = self.len;
            self.slice[self.len] = item;
            self.len += 1;
            unsafe { self.sift_up(0, old_len) };
        }

        /// returns true if item was included.
        pub fn insert_if_better(&mut self, item: T) -> bool {
            if self.len < self.slice.len() {
                // push
                let old_len = self.len;
                self.slice[self.len] = item;
                self.len += 1;
                unsafe { self.sift_up(0, old_len) };
                return true;
            } else {
                if item > self.slice[0] {
                    return false;
                } else {
                    self.slice[0] = item;
                    unsafe { self.sift_down(0) };
                    return true;
                }
            }
        }

        unsafe fn sift_up(&mut self, start: usize, pos: usize) -> usize {
            let mut hole = unsafe { Hole::new(&mut self.slice, pos) };

            while hole.pos() > start {
                let parent = (hole.pos() - 1) / 2;
                if hole.element() <= unsafe { hole.get(parent) } {
                    break;
                }
                unsafe { hole.move_to(parent) };
            }

            hole.pos()
        }

        unsafe fn sift_down(&mut self, pos: usize) {
            unsafe { self.sift_down_range(pos, self.len) };
        }

        unsafe fn sift_down_range(&mut self, pos: usize, end: usize) {
            let mut hole = unsafe { Hole::new(&mut self.slice, pos) };
            let mut child = 2 * hole.pos() + 1;
            while child <= end.saturating_sub(2) {
                child += unsafe { hole.get(child) <= hole.get(child + 1) } as usize;
                if hole.element() >= unsafe { hole.get(child) } {
                    return;
                }
                unsafe { hole.move_to(child) };
                child = 2 * hole.pos() + 1;
            }

            if child == end - 1 && hole.element() < unsafe { hole.get(child) } {
                unsafe { hole.move_to(child) };
            }
        }
    }

    /// copied from std::collections::BinaryHeap
    struct Hole<'a, T: 'a> {
        data: &'a mut [T],
        elt: ManuallyDrop<T>,
        pos: usize,
    }

    impl<'a, T> Hole<'a, T> {
        #[inline]
        unsafe fn new(data: &'a mut [T], pos: usize) -> Self {
            debug_assert!(pos < data.len());
            let elt = unsafe { ptr::read(data.get_unchecked(pos)) };
            Hole {
                data,
                elt: ManuallyDrop::new(elt),
                pos,
            }
        }

        #[inline]
        fn pos(&self) -> usize {
            self.pos
        }

        #[inline]
        fn element(&self) -> &T {
            &self.elt
        }

        #[inline]
        unsafe fn get(&self, index: usize) -> &T {
            debug_assert!(index != self.pos);
            debug_assert!(index < self.data.len());
            unsafe { self.data.get_unchecked(index) }
        }

        #[inline]
        unsafe fn move_to(&mut self, index: usize) {
            debug_assert!(index != self.pos);
            debug_assert!(index < self.data.len());
            unsafe {
                let ptr = self.data.as_mut_ptr();
                let index_ptr: *const _ = ptr.add(index);
                let hole_ptr = ptr.add(self.pos);
                ptr::copy_nonoverlapping(index_ptr, hole_ptr, 1);
            }
            self.pos = index;
        }
    }

    impl<T> Drop for Hole<'_, T> {
        #[inline]
        fn drop(&mut self) {
            // fill the hole again
            unsafe {
                let pos = self.pos;
                ptr::copy_nonoverlapping(&*self.elt, self.data.get_unchecked_mut(pos), 1);
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use rand::{seq::SliceRandom, SeedableRng};

        use super::SliceBinaryHeap;

        #[test]
        fn slice_binary_heap() {
            let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(42);
            let mut elements = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

            let mut memory = vec![0, 0, 0, 0, 0];

            for _ in 0..100 {
                elements.shuffle(&mut rng);
                let mut heap = SliceBinaryHeap::new(&mut memory[..]);
                for i in elements.iter() {
                    heap.insert_if_better(*i);
                }
                let mut heap_slice_sorted: Vec<i32> = heap.as_slice().iter().copied().collect();
                heap_slice_sorted.sort();
                assert_eq!(&heap_slice_sorted, &[1, 2, 3, 4, 5])
            }
        }
    }
}
