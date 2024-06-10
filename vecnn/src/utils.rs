use rand::{thread_rng, Rng};

use crate::{
    dataset::{DatasetT, FlatDataSet},
    distance::l2,
    hnsw::IAndDist,
    Float,
};
use std::{collections::BinaryHeap, sync::Arc};

pub fn linear_knn_search(data: &dyn DatasetT, q_data: &[f32], k: usize) -> Vec<IAndDist<usize>> {
    assert_eq!(q_data.len(), data.dims());
    // this stores the item with the greatest distance in the root (first element)
    let mut knn_heap = KnnHeap::new(k);
    for id in 0..data.len() {
        let i_data = data.get(id);
        let dist = l2(q_data, i_data);
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

pub fn random_data_point<const DIMS: usize>() -> [Float; DIMS] {
    let mut rng = thread_rng();
    let mut p: [Float; DIMS] = [0.0; DIMS];
    for f in p.iter_mut() {
        *f = rng.gen();
    }
    p
}

pub struct KnnHeap {
    k: usize,
    heap: BinaryHeap<IAndDist<usize>>,
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
            self.heap.push(IAndDist { dist, i: id });
        } else {
            let worst_neighbor = self.heap.peek().unwrap();
            if dist < worst_neighbor.dist {
                self.heap.pop();
                self.heap.push(IAndDist { dist, i: id });
            } else {
                // do nothing, not inserted
            }
        }
    }

    pub fn worst_nn_dist(&self) -> f32 {
        self.heap.peek().unwrap().dist
    }

    pub fn as_sorted_vec(self) -> Vec<IAndDist<usize>> {
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
    use super::{linear_knn_search, simple_test_set};

    #[test]
    fn linear_knn() {
        let data = simple_test_set();

        let res = linear_knn_search(&*data, data.get(0), 4);
        let res: Vec<usize> = res.into_iter().map(|e| e.i).collect();
        assert_eq!(res, vec![0, 1, 7, 8]);
    }
}

#[inline(always)]
pub fn extend_lifetime<T: ?Sized>(e: &T) -> &'static T {
    unsafe { &*(e as *const T) }
}
