use std::{
    cell::UnsafeCell,
    cmp::Reverse,
    collections::BinaryHeap,
    ops::Deref,
    sync::Arc,
    time::{Duration, Instant},
    usize,
};

use ahash::{HashMap, HashSet};
use nanoserde::{DeJson, SerJson};
use parking_lot::Mutex;
use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::iter::{
    IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::{
    dataset::DatasetT,
    distance::{l2, Distance, DistanceFn, DistanceTracker},
    hnsw::DistAnd,
    if_tracking,
    utils::{make_ghost_locks, Stats, YoloCell},
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
        ef: usize,
        start_candidates: usize,
    ) -> (Vec<DistAnd<usize>>, Stats) {
        let start = Instant::now();
        let start_candidates = start_candidates.max(1);
        let distance = DistanceTracker::new(self.params.distance);
        let mut visited: HashSet<usize> = HashSet::default();
        let mut frontier: BinaryHeap<Reverse<DistAnd<usize>>> = BinaryHeap::new();
        let mut found: BinaryHeap<DistAnd<usize>> = BinaryHeap::new();

        let mut rng = ChaCha20Rng::seed_from_u64(self.nodes.len() as u64);
        for i in 0..start_candidates {
            let ep_id = rng.gen_range(0..self.nodes.len());
            let ep_data = self.data.get(ep_id);
            let ep_dist = distance.distance(ep_data, q_data);
            visited.insert(ep_id);
            found.push(DistAnd(ep_dist, ep_id));
            frontier.push(Reverse(DistAnd(ep_dist, ep_id)));
        }

        while frontier.len() > 0 {
            let DistAnd(c_dist, c_idx) = frontier.pop().unwrap().0;
            let worst_dist_found = found.peek().unwrap().0;
            if c_dist > worst_dist_found {
                break;
            };
            for nei in self.nodes[c_idx].iter() {
                let nei_idx = nei.idx; // is id into data at the same time.
                if visited.insert(nei_idx) {
                    // only jumps here if was not visited before (newly inserted -> true)
                    let nei_data = self.data.get(nei_idx);
                    let nei_dist_to_q = distance.distance(nei_data, q_data);

                    if found.len() < ef {
                        // always insert if found still has space:
                        frontier.push(Reverse(DistAnd(nei_dist_to_q, nei_idx)));
                        found.push(DistAnd(nei_dist_to_q, nei_idx));
                    } else {
                        // otherwise only insert, if it is better than the worst found element:
                        let mut worst_found = found.peek_mut().unwrap();
                        if nei_dist_to_q < worst_found.dist() {
                            frontier.push(Reverse(DistAnd(nei_dist_to_q, nei_idx)));
                            *worst_found = DistAnd(nei_dist_to_q, nei_idx)
                        }
                    }
                }
            }
        }
        let mut results: Vec<DistAnd<usize>> = found.into_vec();
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

    pub fn new(data: Arc<dyn DatasetT>, params: RNNGraphParams, seed: u64, threaded: bool) -> Self {
        let distance_tracker = DistanceTracker::new(params.distance);

        let start = Instant::now();
        let distance_idx_to_idx =
            |i: usize, j: usize| -> f32 { distance_tracker.distance(data.get(i), data.get(j)) };
        let mut buffers = RNNConstructionBuffers::new();

        construct_relative_nn_graph(
            params,
            seed,
            &distance_idx_to_idx,
            &mut buffers,
            data.len(),
            threaded,
        );

        let build_stats = Stats {
            num_distance_calculations: distance_tracker.num_calculations(),
            duration: start.elapsed(), // todo
        };
        RNNGraph {
            data,
            nodes: buffers.neighbors,
            build_stats,
            params,
        }
    }
}

pub struct RNNConstructionBuffers {
    pub neighbors: Vec<Vec<Neighbor>>,
    pub two_sided_neighbors: Vec<Vec<Neighbor>>, // swap buffer for a) new neighbors or b) two-sided-neighbors
}

impl RNNConstructionBuffers {
    pub const fn new() -> Self {
        RNNConstructionBuffers {
            neighbors: vec![],
            two_sided_neighbors: vec![],
        }
    }
    pub fn clear_and_prepare(&mut self, n_entries: usize, n_neighbors: usize) {
        for buf in [&mut self.neighbors, &mut self.two_sided_neighbors] {
            for e in buf.iter_mut() {
                e.clear();
            }
            if buf.len() < n_entries {
                let additional = n_entries - buf.len();
                buf.reserve(additional);
                for _ in 0..additional {
                    buf.push(Vec::with_capacity(n_neighbors))
                }
            }
        }
    }
}

pub fn construct_relative_nn_graph<D>(
    params: RNNGraphParams,
    seed: u64,
    distance_idx_to_idx: &D,
    buffers: &mut RNNConstructionBuffers,
    n_entries: usize,
    threaded: bool,
) where
    D: Fn(usize, usize) -> f32 + Send + Sync,
{
    buffers.clear_and_prepare(n_entries, params.initial_neighbors);
    let neighbors = &mut buffers.neighbors[0..n_entries];
    let two_sided_neighbors = &mut buffers.two_sided_neighbors[0..n_entries];
    fill_with_random_neighbors(
        distance_idx_to_idx,
        params.initial_neighbors,
        neighbors,
        seed,
    );
    let locks = if threaded {
        make_ghost_locks(n_entries)
    } else {
        vec![]
    };
    for t1 in 0..params.outer_loops {
        // dbg_stats(&nodes, "before update_neighbors");
        for _ in 0..params.inner_loops {
            if threaded {
                update_neighbors_parallel(neighbors, &locks, distance_idx_to_idx);
            } else {
                update_neighbors(neighbors, distance_idx_to_idx)
            }
        }
        // dbg_stats(&nodes, "after update_neighbors");
        if t1 != params.outer_loops - 1 {
            add_reverse_edges(
                neighbors,
                two_sided_neighbors,
                params.max_neighbors_after_reverse_pruning,
                threaded,
            );
        }
    }
}

fn dbg_stats(nodes: &[Vec<Neighbor>], tag: &str) {
    let mut mean: f32 = 0.0;
    let mut max: usize = 0;
    let mut min: usize = usize::MAX;
    for n in nodes.iter() {
        let m = n.len();
        if m > max {
            max = m;
        }
        if m < min {
            min = m;
        }
        mean += m as f32;
    }
    mean /= nodes.len() as f32;
    println!(
        "RNN neighbors: {tag} len={} mean={mean} min={min} max={max}",
        nodes.len()
    )
}

fn update_neighbors(
    neighbors: &mut [Vec<Neighbor>],
    distance_idx_to_idx: &impl Fn(usize, usize) -> f32,
) {
    let mut new_neighbors: Vec<Neighbor> = vec![]; // reused buffer
    let n_entries = neighbors.len();

    for u_idx in 0..n_entries {
        let mut old_neighbors = std::mem::take(&mut neighbors[u_idx]);
        // sort and remove duplicates:
        old_neighbors.sort(); // (sort by distance to this point)
        remove_duplicates_for_sorted(&mut old_neighbors);

        assert!(new_neighbors.is_empty());
        for v in old_neighbors.drain(..) {
            let mut keep_neighbor = true;

            for w in new_neighbors.iter() {
                if !v.is_new && !w.is_new {
                    continue;
                }
                if v.idx == w.idx {
                    keep_neighbor = false;
                    break;
                }
                let dist_v_u = v.dist;
                let dist_v_w = distance_idx_to_idx(w.idx, v.idx);
                if dist_v_w <= dist_v_u {
                    // prune by RNG rule
                    keep_neighbor = false;
                    // insert connection w -> v instead:
                    neighbors[w.idx].push(Neighbor {
                        idx: v.idx,
                        dist: dist_v_w,
                        is_new: true,
                    });
                    break;
                }
            }
            if keep_neighbor {
                new_neighbors.push(v);
            }
        }

        for v in new_neighbors.iter_mut() {
            v.is_new = false;
        }
        std::mem::swap(&mut neighbors[u_idx], &mut new_neighbors);
    }
}
fn update_neighbors_parallel<D>(
    neighbors: &mut [Vec<Neighbor>],
    locks: &[Mutex<()>],
    distance_idx_to_idx: &D,
) where
    D: Fn(usize, usize) -> f32 + Send + Sync,
{
    assert!(locks.len() == neighbors.len());
    let neighbors: &[YoloCell<Vec<Neighbor>>] = unsafe { std::mem::transmute(neighbors) };
    (0..neighbors.len()).into_par_iter().for_each(|u_idx| {
        let mut old_neighbors: Vec<Neighbor>;
        let guard = locks[u_idx].lock();
        old_neighbors = std::mem::take(unsafe { neighbors[u_idx].get_mut() }); //  clear out the old neighbors list
        drop(guard);
        old_neighbors.sort(); // (sort by distance to this point)
        remove_duplicates_for_sorted(&mut old_neighbors);

        let mut new_neighbors: Vec<Neighbor> = vec![]; // todo: maybe reuse in threadlocal buffer

        for v in old_neighbors.drain(..) {
            let mut keep_neighbor = true;

            for w in new_neighbors.iter() {
                if !v.is_new && !w.is_new {
                    continue;
                }
                let dist_v_u = v.dist;
                let dist_v_w = distance_idx_to_idx(w.idx, v.idx);
                if dist_v_w <= dist_v_u {
                    // prune by RNG rule
                    keep_neighbor = false;

                    // insert connection w -> v instead:
                    let guard = locks[w.idx].lock();
                    let w_neighbors = unsafe { neighbors[w.idx].get_mut() };
                    w_neighbors.push(Neighbor {
                        idx: v.idx,
                        dist: dist_v_w,
                        is_new: true,
                    });
                    drop(guard);
                    break;
                }
            }
            if keep_neighbor {
                new_neighbors.push(v);
            }
        }
        for v in new_neighbors.iter_mut() {
            v.is_new = false;
        }
        let guard = locks[u_idx].lock();
        let u_neighbors = unsafe { neighbors[u_idx].get_mut() };
        *u_neighbors = new_neighbors;
        drop(guard);
    });
}

/// Warning! expects a sorted vector
fn remove_duplicates(neighbors: &mut Vec<Neighbor>) {
    let mut last_idx = usize::MAX;
    neighbors.retain(|e| {
        let retain = e.idx != last_idx;
        last_idx = e.idx;
        retain
    });
}

/// Credit: Erik Thordsen (https://www-ai.cs.tu-dortmund.de/PERSONAL/thordsen.html)
fn remove_duplicates_for_sorted(neighbors: &mut Vec<Neighbor>) {
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
    neighbors: &mut [Vec<Neighbor>],
    two_sided_neighbors: &mut [Vec<Neighbor>],
    max_neighbors_after_reverse_pruning: usize,
    threaded: bool,
) {
    // create list for each node, with all **INCOMING** connections to this node and all **OUTGOING** connections from this node
    assert_eq!(neighbors.len(), two_sided_neighbors.len());
    for r in two_sided_neighbors.iter_mut() {
        r.clear();
    }
    for (idx, neighbors) in neighbors.iter_mut().enumerate() {
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
    if threaded {
        two_sided_neighbors.par_iter_mut().for_each(|in_and_out| {
            in_and_out.sort();
            remove_duplicates(in_and_out);
            in_and_out.truncate(max_neighbors_after_reverse_pruning);
        });
    } else {
        for in_and_out in two_sided_neighbors.iter_mut() {
            in_and_out.sort();
            remove_duplicates(in_and_out);
            in_and_out.truncate(max_neighbors_after_reverse_pruning);
        }
    }
    for (idx, in_and_out) in two_sided_neighbors.into_iter().enumerate() {
        for nn in in_and_out.drain(..) {
            neighbors[nn.idx].push(Neighbor {
                idx,
                dist: nn.dist,
                is_new: nn.is_new,
            })
        }
    }

    if threaded {
        neighbors.par_iter_mut().for_each(|neighbors| {
            neighbors.sort();
            neighbors.truncate(max_neighbors_after_reverse_pruning);
        });
    } else {
        for neighbors in neighbors.iter_mut() {
            neighbors.sort();
            neighbors.truncate(max_neighbors_after_reverse_pruning);
        }
    }
}

// todo: Note: probably not worth it fixing this, the reverse list construction is probably not gonna benefic from multithreading much
// fn add_reverse_edges_parallel_with_memory_corruption_bug(
//     neighbors: &mut [Vec<Neighbor>],
//     two_sided_neighbors: &mut [Vec<Neighbor>],
//     locks: &[Mutex<()>],
//     max_neighbors_after_reverse_pruning: usize,
// ) {
//     assert_eq!(neighbors.len(), two_sided_neighbors.len());
//     assert_eq!(neighbors.len(), locks.len());
//     for r in two_sided_neighbors.iter_mut() {
//         r.clear();
//     }
//     let neighbors: &[YoloCell<Vec<Neighbor>>] = unsafe { std::mem::transmute(neighbors) };
//     // create list for each node, with all **INCOMING** connections to this node and all **OUTGOING** connections from this node
//     let two_sided_neighbors: &[YoloCell<Vec<Neighbor>>] =
//         unsafe { std::mem::transmute(two_sided_neighbors) };
//     (0..neighbors.len()).into_par_iter().for_each(|idx| {
//         let neighbors = unsafe { neighbors[idx].get_mut() }; // noone else accesses this so no lock needed
//         for n in neighbors.iter_mut() {
//             let guard = locks[n.idx].lock();
//             unsafe { two_sided_neighbors[n.idx].get_mut() }.push(Neighbor {
//                 idx,
//                 dist: n.dist,
//                 is_new: n.is_new,
//             });
//             drop(guard);
//             n.is_new = true;
//         }
//         unsafe { two_sided_neighbors[idx].get_mut() }.extend(neighbors.drain(..));
//     });
//     // all neighbors are now cleared and put into the two_sided_neighbors, two times each.
//     // sort all in_and_out neighbors per node and remove duplicates, then shrink to r:
//     two_sided_neighbors.par_iter().for_each(|in_and_out| {
//         let in_and_out = unsafe { in_and_out.get_mut() }; // safe, because each neighbors list accessed independently
//         in_and_out.sort();
//         remove_duplicates(in_and_out);
//         in_and_out.truncate(max_neighbors_after_reverse_pruning);
//     });
//     // put back into neighbors:
//     (0..two_sided_neighbors.len())
//         .into_par_iter()
//         .for_each(|idx| {
//             let in_and_out = unsafe { two_sided_neighbors[idx].get_mut() };
//             for nn in in_and_out.drain(..) {
//                 // clears out two sided neighbor list:
//                 let guard = locks[nn.idx].lock();
//                 // put neighbors back in, lock needed again:
//                 unsafe { neighbors[nn.idx].get_mut() }.push(Neighbor {
//                     idx,
//                     dist: nn.dist,
//                     is_new: nn.is_new,
//                 });
//                 drop(guard);
//             }
//         });
//     // sort and truncate to max neighbors:
//     neighbors.par_iter().for_each(|neighbors| {
//         let neighbors = unsafe { neighbors.get_mut() };
//         neighbors.sort();
//         neighbors.truncate(max_neighbors_after_reverse_pruning);
//     });
// }

fn fill_with_random_neighbors(
    distance_idx_to_idx: &impl Fn(usize, usize) -> f32, // idx to idx distance (NOT! id to id)
    initial_neighbors_num: usize,
    neighbors: &mut [Vec<Neighbor>],
    seed: u64,
) {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let n = neighbors.len();
    if n == 1 {
        return;
    }
    let initial_neighbors_num = initial_neighbors_num.min(n - 1);
    for (idx, neighbors) in neighbors.iter_mut().enumerate() {
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
            remove_duplicates(neighbors);
            if neighbors.len() == initial_neighbors_num {
                break;
            }
        }
        for n in neighbors.iter_mut() {
            n.dist = distance_idx_to_idx(idx, n.idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{RNNGraph, RNNGraphParams};

    #[test]
    fn relative_nn_construction() {
        let data = crate::utils::random_data_set(1000, 20);
        let graph = RNNGraph::new(data, RNNGraphParams::default(), 42, false);
        std::fs::write("graph.txt", format!("{graph:?}"));
    }
}
