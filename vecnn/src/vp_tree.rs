use std::{
    collections::BinaryHeap,
    fmt::Debug,
    marker::PhantomData,
    sync::{atomic::AtomicUsize, Arc},
    time::{Duration, Instant},
};

use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::{rand_core::impls, ChaCha12Rng};

use crate::{
    dataset::DatasetT,
    distance::{DistanceFn, DistanceTracker},
    hnsw::IAndDist,
    utils::KnnHeap,
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
    pub build_stats: Stats,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    pub num_distance_calculations: usize,
    pub duration: Duration,
}

impl VpTree {}

impl VpTree {
    pub fn new(data: Arc<dyn DatasetT>, distance_fn: DistanceFn) -> Self {
        let seed: u64 = thread_rng().gen();
        let builder = VpTreeBuilder::new(seed, data, distance_fn);
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

    pub fn knn_search(&self, q: &[Float], k: usize) -> (Vec<IAndDist<usize>>, Stats) {
        let tracker = DistanceTracker::new(self.distance_fn);
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
            let d = dist_to_q(root.idx);
            let t = root.dist; // threshold

            if tree.len() == 1 {
                heap.maybe_add(root.idx, d);
            } else if d <= t {
                heap.maybe_add(root.idx, d);
                // search inner side
                search_tree(left(tree), dist_to_q, heap);
                if (d - t).abs() < heap.worst_nn_dist() || !heap.is_full() {
                    search_tree(right(tree), dist_to_q, heap);
                }
            } else {
                // search other side
                search_tree(right(tree), dist_to_q, heap);
                if (d - t).abs() < heap.worst_nn_dist() || !heap.is_full() {
                    heap.maybe_add(root.idx, d);
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

    // pub fn knn_search(&self, q: &[Float], k: usize) -> (Vec<DistAnd<usize>>, Stats) {
    //     let tracker = DistanceTracker::new(self.distance_fn);
    //     let start = Instant::now();
    //     let dist_to_q = |idx| {
    //         let p = self.data.get(idx);
    //         tracker.distance(p, q)
    //     };

    //     fn search_tree(
    //         tree: &[Node],
    //         k: usize,
    //         t: Float,
    //         heap: &mut BinaryHeap<DistAnd<usize>>,
    //         dist_to_q: &impl Fn(usize) -> Float,
    //     ) {
    //         if tree.len() == 0 {
    //             return;
    //         }
    //         let mut tau = t;
    //         let root = &tree[0];
    //         let dist = dist_to_q(root.idx);

    //         if dist < tau {
    //             heap.push(DistAnd { dist, i: root.idx });
    //             if heap.len() > k {
    //                 heap.pop();
    //             }
    //             if heap.len() == k {
    //                 tau = heap.peek().unwrap().dist;
    //             }

    //             if tree.len() == 1 {
    //                 return;
    //             }

    //             if dist < root.dist {
    //                 search_tree(left(tree), k, tau, heap, dist_to_q);
    //                 if dist + tau >= root.dist {
    //                     search_tree(right(tree), k, t, heap, dist_to_q);
    //                 }
    //             } else {
    //                 search_tree(right(tree), k, t, heap, dist_to_q);
    //                 if (dist - tau) <= root.dist {
    //                     search_tree(left(tree), k, t, heap, dist_to_q);
    //                 }
    //             }
    //         }
    //     }

    //     let mut heap: BinaryHeap<DistAnd<usize>> = BinaryHeap::new();
    //     search_tree(&self.nodes, k, 10000000.0, &mut heap, &dist_to_q);
    //     let mut res = Vec::from(heap);
    //     res.sort();

    //     let stats = Stats {
    //         num_distance_calculations: tracker.num_calculations(),
    //         duration: start.elapsed(),
    //     };

    //     (res, stats)
    // }
    /*



    void VPTreeknnSearch(int query, vptree* node, int k, std::priority_queue<HeapItem>* heap, double t, int threadID, int* numNodesVP) {
        if (node == NULL) return;

        // numNodes contains the number of number of nodes each thread has visited
        numNodesVP[threadID]++;

        int d = node->D;
        int n = node->N;
        double tau = t;

        // Find the distance of query point and vantage point
        double dist = 0.0;
        int vantagePointIdx = node->ivp;
        double* queryCoords = (double*)malloc(d * sizeof(double));

        for (int i = 0; i < d; i++) {
            queryCoords[i] = node->A[query * d + i];

            dist += pow(queryCoords[i] - node->VPCords[i], 2);
        }
        dist = sqrt(dist);

        free(queryCoords);

        if (dist < tau) {
            heap->push(HeapItem(vantagePointIdx, dist));
            if (heap->size() == k + 1) {
                heap->pop();
            }
            if (heap->size() == k) {
                tau = heap->top().dist;
            }
        }

        if (node->inner == NULL && node->outer == NULL) {
            return;
        }

        if (dist < node->median) {
            // Search inner subtree first
            VPTreeknnSearch(query, node->inner, k, heap, tau, threadID, numNodesVP);
            if (dist + tau >= node->median) {
                VPTreeknnSearch(query, node->outer, k, heap, tau, threadID, numNodesVP);
            }

        }
        else {
            // Search outer subtree first
            VPTreeknnSearch(query, node->outer, k, heap, tau, threadID, numNodesVP);
            if (dist - tau <= node->median) {
                VPTreeknnSearch(query, node->inner, k, heap, tau, threadID, numNodesVP);
            }

        }
    }





         */
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

struct VpTreeBuilder {
    nodes: Vec<Node>,
    data: Arc<dyn DatasetT>,
    rng: ChaCha12Rng,
    distance_fn: DistanceFn,
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

impl PartialEq<IAndDist<u32>> for Node {
    fn eq(&self, other: &IAndDist<u32>) -> bool {
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

impl VpTreeBuilder {
    pub fn new(seed: u64, data: Arc<dyn DatasetT>, distance_fn: DistanceFn) -> Self {
        let mut nodes: Vec<Node> = Vec::with_capacity(data.len());
        for idx in 0..data.len() {
            nodes.push(Node { idx, dist: 0.0 });
        }
        VpTreeBuilder {
            nodes,
            data,
            distance_fn,
            rng: ChaCha12Rng::seed_from_u64(seed),
        }
    }

    pub fn build(mut self) -> VpTree {
        let tracker = DistanceTracker::new(self.distance_fn);
        let start = Instant::now();
        // arrange items in self.tmp into a vp tree
        arrange_into_vp_tree(&mut self.nodes, &*self.data, &tracker);

        let build_stats = Stats {
            num_distance_calculations: tracker.num_calculations(),
            duration: start.elapsed(),
        };
        VpTree {
            nodes: self.nodes,
            data: self.data,
            distance_fn: self.distance_fn,
            build_stats,
        }
    }
}

pub fn arrange_into_vp_tree(tmp: &mut [Node], data: &dyn DatasetT, distance: &DistanceTracker) {
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
            tmp[0].dist = distance.distance(pt_0, pt_1);
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
        other.dist = distance.distance(vp_pt, other_pt);
    }
    // partition into points closer and further to median:
    let median_i = quick_select_median_dist(&mut tmp[1..]) + 1;
    // assert!(median_i >= 2);
    // set the median distance on the root node, then build left and right sub-trees
    let median_dist = tmp[median_i].dist;
    tmp[0].dist = median_dist;
    arrange_into_vp_tree(&mut tmp[1..median_i], data, distance);
    arrange_into_vp_tree(&mut tmp[median_i..], data, distance);
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
        distance::l2,
        utils::{linear_knn_search, random_data_point, random_data_set, simple_test_set},
        vp_tree::{left, left_with_root, right},
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
        let random_set = random_data_set(1000, 300);
        let test_set = simple_test_set();

        for data in [test_set, random_set] {
            for _ in 0..20 {
                let query_idx = thread_rng().gen_range(0..data.len());
                let query = data.get(query_idx);
                let vp_tree = VpTree::new(data.clone(), l2);
                let nn = vp_tree.knn_search(query, 1).0[0];
                assert_eq!(nn.i, query_idx);
                assert_eq!(nn.dist, 0.0);
            }
        }
    }

    #[test]
    fn left_right() {
        let nodes = (0..100)
            .map(|e| Node { idx: e, dist: 0.0 })
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
            let tree = VpTree::new(data.clone(), l2);
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
