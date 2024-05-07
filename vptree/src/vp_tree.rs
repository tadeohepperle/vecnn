use std::{collections::BinaryHeap, fmt::Debug, marker::PhantomData, sync::Arc};

use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::{rand_core::impls, ChaCha12Rng};

use crate::{
    dataset::DatasetT,
    distance::{DistanceFn, DistanceT},
    hnsw::DistAnd,
    Float,
};

/// # VpTree
///
/// # Memory layout:
///
/// All nodes are kept next to each other in a big vec:
///
/// example eight nodes:
/// ```txt
/// X l l l h h h h
///   Y l h Y l h h
///     Z Z   Z Z h
///               T
///    |   |   |
///   
/// ```
///
/// example seven nodes:
/// ```txt
/// X l l h h h h
///   Y l h Y l h
///     Z Z   Z Z
///             
/// ```
///
/// example 14 nodes:
/// ```txt
/// X l l l l l l h h h h h h h
///   Y l l h h h Y l l l h h h
///     Z h Z l h   Z l h Z l h
///       T   T T     T T   T T
///        |     |       |
///
///
/// X h
/// X h h
/// X l h h
/// X l h h h
/// X l l h h h
/// X l l h h h h
/// X l l l h h h h
#[derive(Debug, Clone)]
pub struct VpTree {
    pub nodes: Vec<Node>,
    pub data: Arc<dyn DatasetT>,
    pub distance_fn: DistanceFn,
}

impl VpTree {}

impl VpTree {
    pub fn new(data: Arc<dyn DatasetT>, distance: impl DistanceT) -> Self {
        let seed: u64 = thread_rng().gen();
        let builder = VpTreeBuilder::new(seed, data, distance);
        builder.build()
    }

    pub fn iter_levels(&self, f: &mut impl FnMut(usize, &Node)) {
        fn slice_iter(nodes: &[Node], level: usize, f: &mut impl FnMut(usize, &Node)) {
            if nodes.len() == 0 {
                return;
            }
            f(level, &nodes[0]);
            if nodes.len() > 1 {
                slice_iter(left(nodes), level + 1, f);
                slice_iter(right(nodes), level + 1, f);
            }
        }
        slice_iter(&self.nodes, 0, f);
    }

    pub fn knn_search(&self, q: &[Float], k: usize) -> Vec<Node> {
        assert_eq!(q.len(), self.data.dims());
        struct Ctx<F: Fn(usize) -> Float> {
            heap: BinaryHeap<Node>,
            tau: Float, // only nodes with dist < tau can be still fitted into the heap
            dist_to_q: F,
            k: usize,
        }
        fn search_slice<F: Fn(usize) -> Float>(nodes: &[Node], ctx: &mut Ctx<F>) {
            if nodes.len() == 0 {
                return;
            }
            let Node { idx, dist: radius } = nodes[0];
            let dist = (ctx.dist_to_q)(idx);
            if dist < ctx.tau {
                if ctx.heap.len() == ctx.k {
                    ctx.heap.pop();
                }
                ctx.heap.push(Node { idx, dist }); // store the node and its dist to q
                if ctx.heap.len() == ctx.k {
                    ctx.tau = ctx.heap.peek().unwrap().dist
                }
            }
            if nodes.len() == 1 {
                return;
            }

            if dist < radius {
                // search inside the nodes radius first, then outside
                if dist - ctx.tau <= radius {
                    search_slice(left(nodes), ctx)
                }
                if dist + ctx.tau >= radius {
                    search_slice(right(nodes), ctx)
                }
            } else {
                // search outside first then inside
                if dist + ctx.tau >= radius {
                    search_slice(right(nodes), ctx)
                }
                if dist - ctx.tau <= radius {
                    search_slice(left(nodes), ctx)
                }
            }
        }

        let dist_to_q = |idx| {
            let p = self.data.get(idx);
            (self.distance_fn)(p, q)
        };
        let mut ctx = Ctx {
            heap: BinaryHeap::new(),
            tau: Float::MAX,
            dist_to_q,
            k,
        };
        search_slice(&self.nodes, &mut ctx);

        Vec::from(ctx.heap)
    }
}

#[inline(always)]
fn first_element_idx_of_second_part(len: usize) -> usize {
    // len / 2
    (len - 1) / 2
}

#[inline(always)]
fn left(nodes: &[Node]) -> &[Node] {
    let end_idx_excl = first_element_idx_of_second_part(nodes.len() - 1) + 1;
    &nodes[1..end_idx_excl]
}
#[inline(always)]
fn right(nodes: &[Node]) -> &[Node] {
    let start_idx = first_element_idx_of_second_part(nodes.len() - 1) + 1;
    &nodes[start_idx..]
}

struct VpTreeBuilder<D: DistanceT> {
    nodes: Vec<Node>,
    data: Arc<dyn DatasetT>,
    distance: PhantomData<D>,
    rng: ChaCha12Rng,
}

#[derive(Clone, Copy, PartialEq)]
pub struct Node {
    /// id into the dataset
    pub idx: usize,
    /// median distance of children of this node
    pub dist: Float,
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}
impl Ord for Node {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist.total_cmp(&other.dist)
    }
}
impl Eq for Node {}

impl PartialEq<DistAnd<u32>> for Node {
    fn eq(&self, other: &DistAnd<u32>) -> bool {
        self.dist == other.dist && self.idx as u32 == other.i
    }
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
    pub fn new(seed: u64, data: Arc<dyn DatasetT>, _distance: D) -> Self {
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
            distance_fn: D::distance,
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
    // assert!(median_i >= 2);
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
/// Returns an index into the second partition.
///
/// Second partition's size is always >= first partition.
fn quick_select_median_dist(tmp: &mut [Node]) -> usize {
    let rank = first_element_idx_of_second_part(tmp.len());
    _quick_select(tmp, rank);
    return rank;

    /// Note: idx rank will be part of second partition.
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

    #[test]
    fn nums() {
        println!("{}", 5 / 2);
        let n = 11;
        let p = 4;
        for i in 0..p {
            let start = (i * n) / p;
            let end = ((i + 1) * n) / p;
            println!("{start}..{end} / {n}")
        }
    }

    use std::{collections::BinaryHeap, sync::Arc};

    use rand::{thread_rng, Rng, SeedableRng};
    use rand_chacha::ChaCha12Rng;

    use crate::{
        dataset::DatasetT,
        distance::SquaredDiffSum,
        utils::{linear_knn_search, random_data_point, random_data_set, simple_test_set},
        vp_tree::{left, right},
        Float,
    };

    use super::{first_element_idx_of_second_part, quick_select_median_dist, Node, VpTree};

    fn slow_select_median_dist(tmp: &mut [Node]) -> usize {
        tmp.sort_by(|a, b| a.dist.total_cmp(&b.dist));
        first_element_idx_of_second_part(tmp.len())
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
            assert_eq!(quick_i, slow_i);
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
            assert_eq!(i, 3);
            assert_eq!(tmp[i], Node { idx: 4, dist: 4.0 });

            // uneven number of elements:
            let i = quick_select_median_dist(&mut tmp2);
            assert_eq!(i, 4);
            assert_eq!(tmp2[i], Node { idx: 5, dist: 5.0 });
        }
    }

    /// cargo test knn_1 --release
    #[test]
    fn knn_1() {
        let random_set = random_data_set::<768>(1000);
        let test_set = simple_test_set();

        for data in [test_set, random_set] {
            for _ in 0..20 {
                let query_idx = thread_rng().gen_range(0..data.len());
                let query = data.get(query_idx);
                let vp_tree = VpTree::new(data.clone(), SquaredDiffSum);
                let nn = vp_tree.knn_search(query, 1)[0];
                assert_eq!(nn.idx, query_idx);
                assert_eq!(nn.dist, 0.0);
            }
        }
    }

    #[test]
    fn print_is() {
        for i in 0..100 {
            let s = i / 2;
            let s2 = ((i - 1) / 2) + 1;
            println!("i: {i}   s: {s}   s2:{s2}")
        }

        let nodes = (0..7)
            .map(|e| Node { idx: e, dist: 0.0 })
            .collect::<Vec<_>>();

        println!(
            "nodes: {nodes:?}   left: {:?}    right: {:?}",
            left(&nodes),
            right(&nodes)
        );
    }

    #[test]
    fn vptree_knn_compare() {
        // todo! this test currently fails
        let random_set = random_data_set::<3>(300);
        for i in 0..10 {
            let q = random_data_point::<3>();
            let tree = VpTree::new(random_set.clone(), SquaredDiffSum);
            let mut tree_res = tree.knn_search(&q, 20);
            tree_res.sort();
            let lin_res = linear_knn_search(&*random_set, &q, 20);
            assert_eq!(tree_res, lin_res);
        }
    }
}

/*



pub struct VpTreeConfig {
    // todo!
    // - how to select random split point,
    // - max depth...
}


*/
