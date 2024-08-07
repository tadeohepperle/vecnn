use std::{
    cell::UnsafeCell,
    cmp::Reverse,
    collections::BinaryHeap,
    sync::Arc,
    time::{Duration, Instant},
};

use ahash::{HashMap, HashSet};
use nanoserde::{DeJson, SerJson};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::{
    dataset::DatasetT,
    distance::{l2, Distance, DistanceFn, DistanceTracker},
    hnsw::DistAnd,
    if_tracking,
    utils::Stats,
};

#[derive(Debug, Clone, Copy, PartialEq, SerJson, DeJson)]
pub struct RNNGraphParams {
    pub outer_loops: usize,
    pub inner_loops: usize,
    pub max_neighbors_after_reverse_pruning: usize,
    pub initial_neighbors: usize,
    pub distance: Distance,
}

impl Default for RNNGraphParams {
    fn default() -> Self {
        Self {
            initial_neighbors: 30,
            outer_loops: 4,
            inner_loops: 15,
            max_neighbors_after_reverse_pruning: Default::default(),
            distance: Distance::L2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RNNGraph {
    pub data: Arc<dyn DatasetT>,
    pub nodes: Vec<Vec<Neighbor>>,
    pub build_stats: Stats,
    pub params: RNNGraphParams,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Neighbor {
    pub idx: usize,
    pub dist: f32,
    pub is_new: bool,
}

impl std::fmt::Debug for Neighbor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {:.3} ({})", self.idx, self.dist, self.is_new)
    }
}
impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx && self.dist == other.dist
    }
}
impl Eq for Neighbor {}
impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}
impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist.total_cmp(&other.dist)
    }
}

impl RNNGraph {
    pub fn knn_search(
        &self,
        q_data: &[f32],
        k: usize,
        start_candidates: usize,
    ) -> (Vec<DistAnd<usize>>, Stats) {
        assert!(start_candidates > 0);
        let distance = DistanceTracker::new(self.params.distance);
        let dist_to_q = |idx: usize| -> f32 {
            let idx_data = self.data.get(idx);
            distance.distance(q_data, idx_data)
        };

        let start = Instant::now();

        let mut visited: ahash::HashSet<usize> = Default::default();
        let mut candidates: BinaryHeap<Reverse<DistAnd<usize>>> = Default::default(); // has min dist item in root, can be peaked
        let mut search_res: BinaryHeap<DistAnd<usize>> = Default::default(); // has top dist in root, to pop it easily

        let i: usize = 0;
        visited.insert(i);
        let dist = dist_to_q(i);
        candidates.push(Reverse(DistAnd(dist, i)));
        search_res.push(DistAnd(dist, i));

        loop {
            let Some(closest_to_q) = candidates.pop() else {
                break;
            };

            let neighbors = &self.nodes[closest_to_q.0 .1];
            let mut any_neighbor_added: bool = false;
            for n in neighbors.iter() {
                let i = n.idx;
                let already_seen = !visited.insert(i);
                if already_seen {
                    continue;
                }
                let added: bool;
                let dist = dist_to_q(n.idx);

                let space_available = search_res.len() < k;

                if space_available {
                    search_res.push(DistAnd(dist, i));
                    added = true;
                } else {
                    let mut worst = search_res.peek_mut().unwrap();
                    if dist < worst.dist() {
                        *worst = DistAnd(dist, i);
                        added = true;
                    } else {
                        added = false;
                    }
                }
                if added {
                    any_neighbor_added = true;
                    candidates.push(Reverse(DistAnd(dist, i)));
                }
                if_tracking!(Tracking.add_event(Event::EdgeHorizontal {
                    from: closest_to_q.0 .1,
                    to: i,
                    level: 0,
                    comment: if space_available {
                        "space"
                    } else if added {
                        "added"
                    } else {
                        "not_good_enough"
                    }
                }))
            }
        }
        let mut results: Vec<DistAnd<usize>> = search_res.into_vec();
        results.sort();
        let stats = Stats {
            num_distance_calculations: distance.num_calculations(),
            duration: start.elapsed(),
        };
        (results, stats)
    }
}

impl RNNGraph {
    pub fn new_empty(data: Arc<dyn DatasetT>, params: RNNGraphParams) -> Self {
        RNNGraph {
            data,
            nodes: Default::default(),
            build_stats: Default::default(),
            params,
        }
    }

    pub fn new(data: Arc<dyn DatasetT>, params: RNNGraphParams, seed: u64) -> Self {
        construct_relative_nn_graph(data, params, seed)
    }
}

fn construct_relative_nn_graph(
    data: Arc<dyn DatasetT>,
    params: RNNGraphParams,
    seed: u64,
) -> RNNGraph {
    let distance_tracker = DistanceTracker::new(params.distance);
    let distance =
        |i: usize, j: usize| -> f32 { distance_tracker.distance(data.get(i), data.get(j)) };

    let start = Instant::now();

    let mut nodes: Vec<Vec<Neighbor>> =
        random_nn_graph_nodes(data.len(), &distance, params.initial_neighbors, seed);
    let mut two_sided_neighbors: Vec<Vec<Neighbor>> =
        (0..nodes.len()).map(|_| Vec::new()).collect();

    for t1 in 0..params.outer_loops {
        for _ in 0..params.inner_loops {
            update_neighbors(&mut nodes, &distance)
        }
        if t1 != params.outer_loops - 1 {
            add_reverse_edges(
                &mut nodes,
                &mut two_sided_neighbors,
                params.max_neighbors_after_reverse_pruning,
            );
        }
    }

    let build_stats = Stats {
        num_distance_calculations: distance_tracker.num_calculations(),
        duration: start.elapsed(), // todo
    };
    RNNGraph {
        data,
        nodes,
        build_stats,
        params,
    }
}

fn update_neighbors(nodes: &mut [Vec<Neighbor>], distance: &impl Fn(usize, usize) -> f32) {
    let mut new_neighbors: Vec<Neighbor> = vec![];

    for u_idx in 0..nodes.len() {
        assert!(new_neighbors.is_empty());
        let mut old_neighbors = std::mem::take(&mut nodes[u_idx]);

        // sort and remove duplicates:
        old_neighbors.sort();
        remove_duplicates_for_sorted(&mut old_neighbors);

        for v in old_neighbors.drain(..) {
            let mut ok = true;

            for w in new_neighbors.iter() {
                if !v.is_new && !w.is_new {
                    continue;
                }
                if v.idx == w.idx {
                    ok = false;
                    break;
                }
                let dist_v_u = v.dist;
                let dist_v_w = distance(w.idx, v.idx);
                if dist_v_w < dist_v_u {
                    // prune by RNG rule
                    ok = false;
                    // insert connection w -> v instead:
                    nodes[w.idx].push(Neighbor {
                        idx: v.idx,
                        dist: dist_v_w,
                        is_new: true,
                    });
                    break;
                }
            }
            if ok {
                new_neighbors.push(v);
            }
        }

        for v in new_neighbors.iter_mut() {
            v.is_new = false;
        }
        std::mem::swap(&mut nodes[u_idx], &mut new_neighbors);
    }
}

/// Warning! expects a sorted vector
pub fn remove_duplicates(neighbors: &mut Vec<Neighbor>) {
    let mut last_idx = usize::MAX;
    neighbors.retain(|e| {
        let retain = e.idx != last_idx;
        last_idx = e.idx;
        retain
    });
}

/// Credit: Erik Thordsen (https://www-ai.cs.tu-dortmund.de/PERSONAL/thordsen.html)
pub fn remove_duplicates_for_sorted(neighbors: &mut Vec<Neighbor>) {
    if neighbors.len() == 0 {
        return;
    }
    // Last index of items to keep
    let mut target: usize = 0;
    // Identifier of the last item to keep for comparisons
    let mut target_idx: usize = neighbors[0].idx;
    for i in 1..neighbors.len() {
        unsafe {
            let i_element = &*neighbors.get_unchecked(i);
            if i_element.idx != target_idx {
                target += 1;
                target_idx = i_element.idx;
                // Move element at i to target (overwriting duplicated between i and target):
                *neighbors.get_unchecked_mut(target) = *i_element;
            }
        }
    }
    neighbors.truncate(target + 1);
}

fn add_reverse_edges(
    nodes: &mut [Vec<Neighbor>],
    two_sided_neighbors: &mut [Vec<Neighbor>],
    max_neighbors_after_reverse_pruning: usize,
) {
    // create list for each node, with all **INCOMING** connections to this node and all **OUTGOING** connections from this node
    assert_eq!(nodes.len(), two_sided_neighbors.len());
    for r in two_sided_neighbors.iter_mut() {
        r.clear();
    }
    for (idx, neighbors) in nodes.iter_mut().enumerate() {
        for n in neighbors.iter_mut() {
            two_sided_neighbors[n.idx].push(Neighbor {
                idx,
                dist: n.dist,
                is_new: n.is_new,
            });
            n.is_new = true;
        }
        two_sided_neighbors[idx].extend(neighbors.drain(..));
    }
    // all neighbors are now cleared.
    // sort all in_and_out neighbors per node and remove duplicates, then shrink to r:
    for in_and_out in two_sided_neighbors.iter_mut() {
        in_and_out.sort();
        remove_duplicates(in_and_out);
        in_and_out.truncate(max_neighbors_after_reverse_pruning);
    }
    for (idx, in_and_out) in two_sided_neighbors.into_iter().enumerate() {
        for nn in in_and_out.drain(..) {
            nodes[nn.idx].push(Neighbor {
                idx,
                dist: nn.dist,
                is_new: nn.is_new,
            })
        }
    }
    for neighbors in nodes.iter_mut() {
        neighbors.sort();
        neighbors.truncate(max_neighbors_after_reverse_pruning);
    }
}

fn random_nn_graph_nodes(
    n: usize,
    distance: &impl Fn(usize, usize) -> f32,
    initial_neighbors_num: usize,
    seed: u64,
) -> Vec<Vec<Neighbor>> {
    let mut nodes: Vec<Vec<Neighbor>> = Vec::with_capacity(n);

    let mut rng = ChaCha20Rng::seed_from_u64(seed);

    for idx in 0..n {
        let mut neighbors: Vec<Neighbor> = Vec::with_capacity(initial_neighbors_num);
        loop {
            while neighbors.len() < initial_neighbors_num {
                let mut random_idx = rng.gen_range(0..n - 1);
                if random_idx >= idx {
                    random_idx += 1 // ensures that random_idx != idx
                }
                neighbors.push(Neighbor {
                    idx: random_idx,
                    dist: 0.0,
                    is_new: true,
                });
            }
            remove_duplicates(&mut neighbors);
            if neighbors.len() == initial_neighbors_num {
                break;
            }
        }
        for n in neighbors.iter_mut() {
            n.dist = distance(idx, n.idx);
        }
        nodes.push(neighbors)
    }
    nodes
}

#[cfg(test)]
mod tests {
    use super::{RNNGraph, RNNGraphParams};

    #[test]
    fn relative_nn_construction() {
        let data = crate::utils::random_data_set(1000, 20);
        let graph = RNNGraph::new(data, RNNGraphParams::default(), 42);
        std::fs::write("graph.txt", format!("{graph:?}"));
    }
}
