use std::{fmt::Debug, marker::PhantomData};

use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;

use crate::{dataset::DatasetT, distance::DistanceT, Float};

pub struct VpTree {
    heap: Vec<Node>,
}

struct Node {
    idx: usize,
    // the median distance where the split happened for children of node
    dist: Float,
}

pub struct VpTreeConfig {
    // todo!
    // - how to select random split point,
    // - max depth...
}

impl VpTree {
    pub fn new(data: &dyn DatasetT, distance_fn: impl DistanceT) -> Self {
        let seed: u64 = thread_rng().gen();
        let builder = VpTreeBuilder::new(seed, data, distance_fn);
        builder.build()
    }
}

struct VpTreeBuilder<'a, D: DistanceT> {
    rng: ChaCha12Rng,
    tmp: Vec<Tmp>,
    data: &'a dyn DatasetT,
    distance: PhantomData<D>,
}

#[derive(Clone, Copy, PartialEq)]
struct Tmp {
    idx: usize,
    dist: Float,
}

impl Debug for Tmp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("")
            .field(&self.idx)
            .field(&self.dist)
            .finish()
    }
}

impl<'a, D: DistanceT> VpTreeBuilder<'a, D> {
    pub fn new(seed: u64, data: &'a dyn DatasetT, _distance: D) -> Self {
        let rng = ChaCha12Rng::seed_from_u64(seed);
        let len = data.len();
        let mut tmp: Vec<Tmp> = Vec::with_capacity(len);
        for idx in 0..len {
            tmp.push(Tmp { idx, dist: 0.0 });
        }
        VpTreeBuilder {
            rng,
            tmp,
            data,
            distance: PhantomData,
        }
    }

    pub fn build(mut self) -> VpTree {
        // select a random pt from tmp storage,
        let vp_t_i = self.select_random_point(0..self.tmp.len());
        let vp_idx = self.tmp[vp_t_i].idx;
        let vp_pt = self.data.get(vp_idx);
        // swap the vp in tmp storage with first pt.
        self.tmp.swap(vp_t_i, 0);

        // calc distance for all points in tmp:
        let d_range = 1..self.tmp.len();
        for t_i in d_range {
            let t = &mut self.tmp[t_i];
            let t_pt = self.data.get(t.idx);
            t.dist = D::distance(vp_pt, t_pt);
        }

        // quick select the point with the median distance:

        todo!()
    }

    fn select_random_point(&mut self, range: std::ops::Range<usize>) -> usize {
        assert!(range.start < range.end);
        // right now this is very simple, todo! make configurable later.
        self.rng.gen_range(range)
    }
}

/// returns the index at which the median can be found:
fn quick_select_median_dist(tmp: &mut [Tmp]) -> usize {
    let rank = tmp.len() / 2;
    _quick_select(tmp, rank);
    return rank;

    /// l and r both inclusive
    fn _quick_select(tmp: &mut [Tmp], rank: usize) {
        if tmp.len() <= 1 {
            return;
        }
        let last_idx_of_first_part = _partition(tmp);
        if last_idx_of_first_part >= rank {
            _quick_select(&mut tmp[..=last_idx_of_first_part], rank);
        } else {
            _quick_select(
                &mut tmp[(last_idx_of_first_part + 1)..],
                rank - last_idx_of_first_part - 1,
            );
        }
    }

    /// the i returned here is the last index of the first partition
    fn _partition(tmp: &mut [Tmp]) -> usize {
        let pivot_i = 0;
        let pivot_dist = tmp[pivot_i].dist;
        let mut i: i32 = -1;
        let mut j = tmp.len();
        loop {
            i += 1;
            while tmp[i as usize].dist < pivot_dist {
                i += 1;
            }
            j -= 1;
            while tmp[j].dist > pivot_dist {
                j -= 1;
            }
            if i as usize >= j {
                return j;
            }
            tmp.swap(i as usize, j);
        }
    }
}

#[cfg(test)]
pub mod tests {
    use rand::{thread_rng, Rng, SeedableRng};
    use rand_chacha::ChaCha12Rng;

    use super::{quick_select_median_dist, Tmp};

    fn slow_select_median_dist(tmp: &mut [Tmp]) -> usize {
        tmp.sort_by(|a, b| a.dist.total_cmp(&b.dist));
        tmp.len() / 2
    }

    #[test]
    fn compare_slow_and_quick_select() {
        let mut rng = thread_rng();

        for _ in 0..100 {
            let mut tmp: Vec<Tmp> = vec![];
            let n = rng.gen_range(1..4000);
            for i in 0..n {
                tmp.push(Tmp {
                    idx: i,
                    dist: rng.gen(),
                })
            }

            // check that slow_select_median_dist and quick_select_median_dist give same result for median
            // on a separate copy of the same data.
            let mut tmp_cloned = tmp.clone();
            let quick_i = quick_select_median_dist(&mut tmp);
            let quick_e = &tmp[quick_i];
            let slow_i = quick_select_median_dist(&mut tmp_cloned);
            let slow_e = &tmp_cloned[slow_i];
            assert_eq!(quick_e, slow_e);
        }
    }

    #[test]
    fn quick_select_test_1() {
        let mut tmp = vec![
            Tmp { idx: 1, dist: 1.0 },
            Tmp { idx: 2, dist: 2.0 },
            Tmp { idx: 3, dist: 3.0 },
            Tmp { idx: 4, dist: 4.0 },
            Tmp { idx: 5, dist: 5.0 },
            Tmp { idx: 6, dist: 6.0 },
            Tmp { idx: 7, dist: 7.0 },
            Tmp { idx: 8, dist: 8.0 },
            Tmp { idx: 9, dist: 9.0 },
        ];

        let mut rng = ChaCha12Rng::seed_from_u64(0);

        use rand::seq::SliceRandom;
        for _ in 0..1000 {
            tmp.shuffle(&mut rng);
            // let max = |s: &[Tmp]| -> f32 {
            //     s.iter().map(|e| (e.dist * 100000.0) as i32).max().unwrap() as f32
            // };
            // let i = partition(&mut tmp);
            // assert!(max(&tmp[..=i]) < max(&tmp[(i + 1)..]));
            let i = quick_select_median_dist(&mut tmp);
            assert_eq!(tmp[i], Tmp { idx: 5, dist: 5.0 });
        }
    }
}

/*

// Sorts a (portion of an) array, divides it into partitions, then sorts those
algorithm quicksort(A, lo, hi) is
  if lo >= 0 && hi >= 0 && lo < hi then
    p := partition(A, lo, hi)
    quicksort(A, lo, p) // Note: the pivot is now included
    quicksort(A, p + 1, hi)

// Divides array into two partitions
algorithm partition(A, lo, hi) is
  // Pivot value
  pivot := A[lo] // Choose the first element as the pivot

  // Left index
  i := lo - 1

  // Right index
  j := hi + 1

  loop forever
    // Move the left index to the right at least once and while the element at
    // the left index is less than the pivot
    do i := i + 1 while A[i] < pivot

    // Move the right index to the left at least once and while the element at
    // the right index is greater than the pivot
    do j := j - 1 while A[j] > pivot

    // If the indices crossed, return
    if i >= j then return j

    // Swap the elements at the left and right indices
    swap A[i] with A[j]



*/

/*

http://stevehanov.ca/blog/index.php?id=130
Buildtree(left, right)
http://stevehanov.ca/blog/index.php?id=130
https://johnnysswlab.com/performance-through-memory-layout/#:~:text=Binary%20Tree%20Memory%20Layout,-For%20the%20binary&text=Left%20Child%20Neighbor%20Layout%20%E2%80%93%20for,neighboring%20memory%20chunks%20in%20memory.

for each element one node slot in a big vec

[                                    ]



the index of the node can already determine its position in the tree:

          root
      l          r

 ll     lr    rl    rr


binary heap:

root | l, r | ll, lr, rl, rr







root | l  ll lr lll llr ...                  |  r

here all the subtrees are continous in memory


Node{

}




Node{
    left: Option<usize>,
    right: Option<usize>,
}


select one pt from the data.
determine for each



given some value, find the median: O(n) (quickselect)

lets have a memory region called scratch space at first,
where we compute distances every time.


So first, we can just init this space to zeros.

Now when choosing the first vantage point, we can compute distances from this point to each other




steps to build the vp tree:


memory needed:

a scratchpad for organizing the nodes by their distance.

[ (id: usize,  )]

this is just a vec where each element has idx and


the result of the tree should







*/
