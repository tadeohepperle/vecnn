use std::{fmt::Debug, marker::PhantomData};

use rand::{thread_rng, Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha12Rng;

use crate::{dataset::DatasetT, distance::DistanceT, Float};

pub struct VpTree {
    nodes: Vec<Node>,
    data: Box<dyn DatasetT>,
}

impl VpTree {}

pub struct VpTreeConfig {
    // todo!
    // - how to select random split point,
    // - max depth...
}

impl VpTree {
    pub fn new(data: Box<dyn DatasetT>, distance_fn: impl DistanceT) -> Self {
        let seed: u64 = thread_rng().gen();
        let builder = VpTreeBuilder::new(seed, data, distance_fn);
        builder.build()
    }
}

struct VpTreeBuilder<D: DistanceT> {
    nodes: Vec<Node>,
    data: Box<dyn DatasetT>,
    distance: PhantomData<D>,
    rng: ChaCha12Rng,
}

#[derive(Clone, Copy, PartialEq)]
struct Node {
    /// id into the dataset
    idx: usize,
    /// median distance of children of this node
    dist: Float,
}

impl Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("")
            .field(&self.idx)
            .field(&self.dist)
            .finish()
    }
}

impl<D: DistanceT> VpTreeBuilder<D> {
    pub fn new(seed: u64, data: Box<dyn DatasetT>, _distance: D) -> Self {
        let mut tmp: Vec<Node> = Vec::with_capacity(data.len());
        for idx in 0..data.len() {
            tmp.push(Node { idx, dist: 0.0 });
        }
        VpTreeBuilder {
            nodes: tmp,
            data,
            distance: PhantomData,
            rng: ChaCha12Rng::seed_from_u64(seed),
        }
    }

    pub fn build(mut self) -> VpTree {
        // arrange items in self.tmp into a vp tree
        arrange_into_vp_tree::<D>(&mut self.nodes, &*self.data);
        VpTree {
            nodes: self.nodes,
            data: self.data,
        }
    }
}

fn arrange_into_vp_tree<D: DistanceT>(tmp: &mut [Node], data: &dyn DatasetT) {
    // early return if there are only 0,1 or 2 elements left
    match tmp.len() {
        0 => return,
        1 => {
            tmp[0].dist = 0.0;
            return;
        }
        2 => {
            let pt_0 = data.get(tmp[0].idx);
            let pt_1 = data.get(tmp[1].idx);
            tmp[0].dist = D::distance(pt_0, pt_1);
            tmp[1].dist = 0.0;
            return;
        }
        _ => {}
    }
    // select a random index and swap it with the first element:
    tmp.swap(select_random_point(tmp, data), 0);
    let vp_pt = data.get(tmp[0].idx);
    // calculate distances to each other element:
    for i in 1..tmp.len() {
        let other = &mut tmp[i];
        let other_pt = data.get(other.idx);
        other.dist = D::distance(vp_pt, other_pt);
    }
    // partition into points closer and further to median:
    let median_i = quick_select_median_dist(&mut tmp[1..]) + 1;
    assert!(median_i >= 2);
    // set the median distance on the root node, then build left and right sub-trees
    let median_dist = tmp[median_i].dist;
    tmp[0].dist = median_dist;
    arrange_into_vp_tree::<D>(&mut tmp[1..median_i], data);
    arrange_into_vp_tree::<D>(&mut tmp[median_i..], data);
}

fn select_random_point(tmp: &[Node], _data: &dyn DatasetT) -> usize {
    let mut rng = thread_rng();
    // right now this is very simple, todo! make configurable later, use data to find good points
    rng.gen_range(0..tmp.len())
}

/// Modifies the slice to have two paritions: smaller than median and larger than median.
/// Returns the index at which the median can be found (len/2).
/// This index always points at the first element of the second half of the slice.
fn quick_select_median_dist(tmp: &mut [Node]) -> usize {
    let rank = tmp.len() / 2;
    _quick_select(tmp, rank);
    return rank;

    /// l and r both inclusive
    fn _quick_select(tmp: &mut [Node], rank: usize) {
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
    fn _partition(tmp: &mut [Node]) -> usize {
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

    use super::{quick_select_median_dist, Node};

    fn slow_select_median_dist(tmp: &mut [Node]) -> usize {
        tmp.sort_by(|a, b| a.dist.total_cmp(&b.dist));
        tmp.len() / 2
    }

    #[test]
    fn compare_slow_and_quick_select() {
        let mut rng = thread_rng();

        for _ in 0..100 {
            let mut tmp: Vec<Node> = vec![];
            let n = rng.gen_range(1..4000);
            for i in 0..n {
                tmp.push(Node {
                    idx: i,
                    dist: rng.gen(),
                })
            }

            // check that slow_select_median_dist and quick_select_median_dist give same result for median
            // on a separate copy of the same data.
            let mut tmp_cloned = tmp.clone();
            let quick_i = quick_select_median_dist(&mut tmp);
            let quick_e = &tmp[quick_i];
            let slow_i = slow_select_median_dist(&mut tmp_cloned);
            let slow_e = &tmp_cloned[slow_i];
            assert_eq!(quick_e, slow_e);
        }
    }

    #[test]
    fn quick_select_test_1() {
        let mut tmp = vec![
            Node { idx: 1, dist: 1.0 },
            Node { idx: 2, dist: 2.0 },
            Node { idx: 3, dist: 3.0 },
            Node { idx: 4, dist: 4.0 },
            Node { idx: 5, dist: 5.0 },
            Node { idx: 6, dist: 6.0 },
            Node { idx: 7, dist: 7.0 },
            Node { idx: 8, dist: 8.0 },
        ];
        let mut tmp2 = tmp.clone();
        tmp2.push(Node { idx: 9, dist: 9.0 });

        let mut rng = ChaCha12Rng::seed_from_u64(0);

        use rand::seq::SliceRandom;
        for _ in 0..1000 {
            tmp.shuffle(&mut rng);
            tmp2.shuffle(&mut rng);

            // even number of elements (should return index to first element of second half):
            let i = quick_select_median_dist(&mut tmp);
            assert_eq!(i, 4);
            assert_eq!(tmp[i], Node { idx: 5, dist: 5.0 });

            // uneven number of elements:
            let i = quick_select_median_dist(&mut tmp2);
            assert_eq!(i, 4);
            assert_eq!(tmp[i], Node { idx: 5, dist: 5.0 });
        }
    }

    // #[test]
    // fn trailing_zeros() {
    //     for e in 0u64..100 {
    //         // let z = e.();
    //         println!("{e}: {z}");
    //     }
    // }
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
