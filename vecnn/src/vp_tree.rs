use std::{
    collections::BinaryHeap,
    fmt::Debug,
    marker::PhantomData,
    sync::{atomic::AtomicUsize, Arc},
    time::{Duration, Instant},
};

use rand::{Rng, SeedableRng};
use rand_chacha::{rand_core::impls, ChaCha12Rng, ChaCha20Rng};

use crate::{
    dataset::DatasetT,
    distance::{Distance, DistanceFn, DistanceTracker},
    hnsw::DistAnd,
    utils::{KnnHeap, Stats},
    Float,
};
use rayon::{
    iter::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
    slice::ParallelSliceMut,
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
    pub distance: Distance,
    pub build_stats: Stats,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VpTreeParams {
    pub distance: Distance,
    pub threaded: bool,
}
impl Default for VpTreeParams {
    fn default() -> Self {
        Self {
            distance: Distance::L2,
            threaded: false,
        }
    }
}

impl VpTree {
    pub fn new(data: Arc<dyn DatasetT>, params: VpTreeParams, seed: u64) -> Self {
        construct_vp_tree(data, params.distance, seed, params.threaded)
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

    pub fn knn_search_approximative(
        &self,
        q: &[Float],
        k: usize,
        max_visits: usize,
    ) -> (Vec<DistAnd<usize>>, Stats) {
        let tracker = DistanceTracker::new(self.distance);
        let start = Instant::now();
        let dist_to_q = |idx| {
            let p = self.data.get(idx);
            tracker.distance(p, q)
        };

        struct SearchCtx {
            heap: KnnHeap,
            visited: usize,
            max_visits: usize,
        }

        let mut ctx = SearchCtx {
            heap: KnnHeap::new(k),
            visited: 0,
            max_visits,
        };
        impl SearchCtx {
            #[inline]
            fn out_of_budget(&self) -> bool {
                self.visited >= self.max_visits
            }
        }

        fn search_tree(tree: &[Node], dist_to_q: &impl Fn(usize) -> f32, ctx: &mut SearchCtx) {
            if tree.len() == 0 {
                return;
            }

            let root = &tree[0];
            let d = dist_to_q(root.id);
            let t = root.dist; // threshold
            ctx.visited += 1;

            if tree.len() == 1 {
                ctx.heap.insert_if_better(root.id, d);
            } else if d <= t {
                ctx.heap.insert_if_better(root.id, d);
                // search inner side
                search_tree(left(tree), dist_to_q, ctx);
                if (d - t).abs() < ctx.heap.worst_nn_dist() || !ctx.heap.is_full() {
                    if !ctx.out_of_budget() {
                        search_tree(right(tree), dist_to_q, ctx);
                    }
                }
            } else {
                // search other side
                search_tree(right(tree), dist_to_q, ctx);
                if (d - t).abs() < ctx.heap.worst_nn_dist() || !ctx.heap.is_full() {
                    if !ctx.out_of_budget() {
                        ctx.heap.insert_if_better(root.id, d);
                        search_tree(left(tree), dist_to_q, ctx);
                    }
                }
            }
        }

        search_tree(&self.nodes, &dist_to_q, &mut ctx);
        let stats = Stats {
            num_distance_calculations: tracker.num_calculations(),
            duration: start.elapsed(),
        };

        (ctx.heap.as_sorted_vec(), stats)
    }

    pub fn knn_search(&self, q: &[Float], k: usize) -> (Vec<DistAnd<usize>>, Stats) {
        let tracker = DistanceTracker::new(self.distance);
        let start = Instant::now();
        let dist_to_q = |idx| {
            let p = self.data.get(idx);
            tracker.distance(p, q)
        };
        let mut heap = KnnHeap::new(k);

        fn search_tree(tree: &[Node], dist_to_q: &impl Fn(usize) -> f32, heap: &mut KnnHeap) {
            if tree.len() == 0 {
                return;
            }

            let root = &tree[0];
            let d = dist_to_q(root.id);
            let t = root.dist; // threshold

            if tree.len() == 1 {
                heap.insert_if_better(root.id, d);
            } else if d <= t {
                heap.insert_if_better(root.id, d);
                // search inner side
                search_tree(left(tree), dist_to_q, heap);
                if (d - t).abs() < heap.worst_nn_dist() || !heap.is_full() {
                    search_tree(right(tree), dist_to_q, heap);
                }
            } else {
                // search other side
                search_tree(right(tree), dist_to_q, heap);
                if (d - t).abs() < heap.worst_nn_dist() || !heap.is_full() {
                    heap.insert_if_better(root.id, d);
                    search_tree(left(tree), dist_to_q, heap);
                }
            }
        }

        search_tree(&self.nodes, &dist_to_q, &mut heap);
        let stats = Stats {
            num_distance_calculations: tracker.num_calculations(),
            duration: start.elapsed(),
        };

        (heap.as_sorted_vec(), stats)
    }
}

#[inline(always)]
fn first_element_idx_of_second_part(len: usize) -> usize {
    (len - 1) / 2
}

#[inline(always)]
pub fn left_with_root(nodes: &[Node]) -> &[Node] {
    let end_idx_excl = first_element_idx_of_second_part(nodes.len() - 1) + 1;
    &nodes[0..end_idx_excl]
}

#[inline(always)]
pub fn left(nodes: &[Node]) -> &[Node] {
    let end_idx_excl = first_element_idx_of_second_part(nodes.len() - 1) + 1;
    &nodes[1..end_idx_excl]
}

#[inline(always)]
pub fn right(nodes: &[Node]) -> &[Node] {
    let start_idx = first_element_idx_of_second_part(nodes.len() - 1) + 1;
    &nodes[start_idx..]
}

#[derive(Clone, Copy, PartialEq)]
pub struct Node {
    /// id into the dataset
    pub id: usize,
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
        self.dist == other.dist() && self.id as u32 == other.1
    }
}

impl Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("").field(&self.id).field(&self.dist).finish()
    }
}

pub fn construct_vp_tree(
    data: Arc<dyn DatasetT>,
    distance: Distance,
    seed: u64,
    parallel: bool,
) -> VpTree {
    let mut nodes: Vec<Node> = Vec::with_capacity(data.len());
    for idx in 0..data.len() {
        nodes.push(Node { id: idx, dist: 0.0 });
    }
    let distance_tracker = DistanceTracker::new(distance);
    let start = Instant::now();
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    // arrange items in self.tmp into a vp tree
    let data_get = |e: &Node| data.get(e.id);
    if parallel {
        arrange_into_vp_tree_parallel(&mut nodes, &data_get, &distance_tracker, &mut rng);
    } else {
        arrange_into_vp_tree(&mut nodes, &data_get, &distance_tracker, &mut rng);
    }

    let build_stats = Stats {
        num_distance_calculations: distance_tracker.num_calculations(),
        duration: start.elapsed(),
    };
    VpTree {
        nodes,
        data,
        distance,
        build_stats,
    }
}

pub trait StoresDistT {
    fn dist(&self) -> f32;
    fn set_dist(&mut self, dist: f32);
}

impl StoresDistT for Node {
    #[inline(always)]
    fn dist(&self) -> f32 {
        self.dist
    }
    #[inline(always)]
    fn set_dist(&mut self, dist: f32) {
        self.dist = dist;
    }
}

pub fn arrange_into_vp_tree_parallel<'a, T, F>(
    tmp: &mut [T],
    data_get: &'a F,
    distance: &DistanceTracker,
    rng: &mut rand_chacha::ChaCha20Rng,
) where
    T: StoresDistT + Send + Sync,
    F: Fn(&T) -> &'a [f32] + Send + Sync,
{
    // how big a chunk needs to be to be split in two parallel operations or to calculate distances in parallel.
    const PAR_DISTANCE_CALCS_MIN_SIZE: usize = 256;
    const PAR_DISTANCE_CALCS_CHUNK_SIZE: usize = 64;
    const PAR_SPLIT_MIN_SIZE: usize = 256;

    // early return if there are only 0,1 or 2 elements left
    let tmp_len = tmp.len();
    match tmp_len {
        0 => return,
        1 => {
            tmp[0].set_dist(0.0);
            return;
        }
        2 => {
            let pt_0 = data_get(&tmp[0]);
            let pt_1 = data_get(&tmp[1]);
            tmp[0].set_dist(distance.distance(pt_0, pt_1));
            tmp[1].set_dist(0.0);
            return;
        }
        _ => {}
    }
    // select a random index and swap it with the first element:
    tmp.swap(select_random_point(tmp, rng), 0);
    let vp_pt = data_get(&tmp[0]);

    // calculate distances to each other element:
    if tmp_len >= PAR_DISTANCE_CALCS_MIN_SIZE {
        // parallel:
        tmp[1..]
            .par_chunks_mut(PAR_DISTANCE_CALCS_CHUNK_SIZE)
            .for_each(|chunk| {
                for other in chunk.iter_mut() {
                    let other_pt = data_get(other);
                    other.set_dist(distance.distance(vp_pt, other_pt));
                }
            });
    } else {
        // sequential
        for i in 1..tmp_len {
            let other = &mut tmp[i];
            let other_pt = data_get(other);
            other.set_dist(distance.distance(vp_pt, other_pt));
        }
    }

    // partition into points closer and further to median:
    let median_i = quick_select_median_dist(&mut tmp[1..]) + 1;
    // assert!(median_i >= 2);
    // set the median distance on the root node, then build left and right sub-trees
    let median_dist = tmp[median_i].dist();
    tmp[0].set_dist(median_dist);

    if tmp_len >= PAR_SPLIT_MIN_SIZE {
        let (part_a, part_b) = tmp.split_at_mut(median_i);
        let mut rng_a = rng.clone();
        rayon::join(
            || arrange_into_vp_tree_parallel(&mut part_a[1..], data_get, distance, &mut rng_a),
            || arrange_into_vp_tree_parallel(part_b, data_get, distance, rng),
        );
    } else {
        arrange_into_vp_tree_parallel(&mut tmp[1..median_i], data_get, distance, rng);
        arrange_into_vp_tree_parallel(&mut tmp[median_i..], data_get, distance, rng);
    }
}

pub fn arrange_into_vp_tree<'a, T: StoresDistT>(
    tmp: &mut [T],
    data_get: &'a impl Fn(&T) -> &'a [f32],
    distance: &DistanceTracker,
    rng: &mut rand_chacha::ChaCha20Rng,
) {
    // early return if there are only 0,1 or 2 elements left
    match tmp.len() {
        0 => return,
        1 => {
            tmp[0].set_dist(0.0);
            return;
        }
        2 => {
            let pt_0 = data_get(&tmp[0]);
            let pt_1 = data_get(&tmp[1]);
            tmp[0].set_dist(distance.distance(pt_0, pt_1));
            tmp[1].set_dist(0.0);
            return;
        }
        _ => {}
    }
    // select a random index and swap it with the first element:
    tmp.swap(select_random_point(tmp, rng), 0);
    let vp_pt = data_get(&tmp[0]);
    // calculate distances to each other element:
    for i in 1..tmp.len() {
        let other = &mut tmp[i];
        let other_pt = data_get(other);
        other.set_dist(distance.distance(vp_pt, other_pt));
    }
    // partition into points closer and further to median:
    let median_i = quick_select_median_dist(&mut tmp[1..]) + 1;
    // assert!(median_i >= 2);
    // set the median distance on the root node, then build left and right sub-trees
    let median_dist = tmp[median_i].dist();
    tmp[0].set_dist(median_dist);
    arrange_into_vp_tree(&mut tmp[1..median_i], data_get, distance, rng);
    arrange_into_vp_tree(&mut tmp[median_i..], data_get, distance, rng);
}

#[inline]
fn select_random_point<T>(tmp: &[T], rng: &mut rand_chacha::ChaCha20Rng) -> usize {
    // right now this is very simple, todo! make configurable later, use data to find good points
    rng.gen_range(0..tmp.len())
}

/// Modifies the slice to have two paritions: smaller than median and larger than median.
/// Returns an index into the second partition.
///
/// Second partition's size is always >= first partition.
fn quick_select_median_dist<T: StoresDistT>(tmp: &mut [T]) -> usize {
    let rank = first_element_idx_of_second_part(tmp.len());
    _quick_select(tmp, rank);
    return rank;

    /// Note: idx rank will be part of second partition.
    fn _quick_select<T: StoresDistT>(tmp: &mut [T], rank: usize) {
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
    fn _partition<T: StoresDistT>(tmp: &mut [T]) -> usize {
        let pivot_i = 0;
        let pivot_dist = tmp[pivot_i].dist();
        let mut i: i32 = -1;
        let mut j = tmp.len();
        loop {
            i += 1;
            while tmp[i as usize].dist() < pivot_dist {
                i += 1;
            }
            j -= 1;
            while tmp[j].dist() > pivot_dist {
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
        distance::{l2, Distance},
        utils::{linear_knn_search, random_data_set, simple_test_set},
        vp_tree::{left, left_with_root, right, VpTreeParams},
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
                    id: i,
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
            Node { id: 1, dist: 1.0 },
            Node { id: 2, dist: 2.0 },
            Node { id: 3, dist: 3.0 },
            Node { id: 4, dist: 4.0 },
            Node { id: 5, dist: 5.0 },
            Node { id: 6, dist: 6.0 },
            Node { id: 7, dist: 7.0 },
            Node { id: 8, dist: 8.0 },
        ];
        let mut tmp2 = tmp.clone();
        tmp2.push(Node { id: 9, dist: 9.0 });

        let mut rng = ChaCha12Rng::seed_from_u64(0);

        use rand::seq::SliceRandom;
        for _ in 0..1000 {
            tmp.shuffle(&mut rng);
            tmp2.shuffle(&mut rng);

            // even number of elements (should return index to first element of second half):
            let i = quick_select_median_dist(&mut tmp);
            assert_eq!(i, 3);
            assert_eq!(tmp[i], Node { id: 4, dist: 4.0 });

            // uneven number of elements:
            let i = quick_select_median_dist(&mut tmp2);
            assert_eq!(i, 4);
            assert_eq!(tmp2[i], Node { id: 5, dist: 5.0 });
        }
    }

    /// cargo test knn_1 --release
    #[test]
    fn knn_1() {
        let random_set = random_data_set(1000, 300);
        let test_set = simple_test_set();

        for data in [test_set, random_set] {
            for _ in 0..20 {
                let query_idx = thread_rng().gen_range(0..data.len());
                let query = data.get(query_idx);
                let vp_tree = VpTree::new(data.clone(), VpTreeParams::default(), 42);
                let nn = vp_tree.knn_search(query, 1).0[0];
                assert_eq!(nn.1, query_idx);
                assert_eq!(nn.dist(), 0.0);
            }
        }
    }

    #[test]
    fn left_right() {
        let nodes = (0..100)
            .map(|e| Node { id: e, dist: 0.0 })
            .collect::<Vec<_>>();
        for i in 2..100 {
            let sub = &nodes[0..i];
            assert!(left(sub).len() < right(sub).len());
            assert!(left_with_root(sub).len() <= right(sub).len());
            assert!(right(sub).len() - left_with_root(sub).len() <= 1); // right side max 1 bigger than entire left tree with root.
        }
    }

    #[test]
    fn vptree_knn_compare() {
        // todo! this test currently fails
        // let data = random_data_set(300, 3);
        // for i in 0..100 {
        //     let q = random_data_point::<3>();
        //     let tree = VpTree::new(data.clone(), SquaredDiffSum::distance);
        //     let tree_res = tree.knn_search(&q, 5).0;
        //     let lin_res = linear_knn_search(&*data, &q, 5);
        //     assert_eq!(tree_res, lin_res);
        // }

        let data = simple_test_set();

        for i in 0..10 {
            let q = data.get(i);
            let tree = VpTree::new(data.clone(), VpTreeParams::default(), 42);
            dbg!(i);
            println!("{:?}", &tree.nodes);
            let tree_l = left(&tree.nodes);
            let tree_r = right(&tree.nodes);
            println!("left: {:?}", tree_l);
            println!("left left: {:?}", left(tree_l));
            println!("left right: {:?}", right(tree_l));
            println!("right: {:?}", tree_r);
            println!("right left: {:?}", left(tree_r));
            println!("right right: {:?}", right(tree_r));
            let tree_res = tree.knn_search(&q, 5).0;
            let lin_res = linear_knn_search(&*data, &q, 5, l2);

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
