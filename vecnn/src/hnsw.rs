use std::{
    cell::UnsafeCell,
    cmp::Reverse,
    collections::BinaryHeap,
    fmt::Debug,
    io::Write,
    process::id,
    sync::{Arc, Barrier},
    time::{Duration, Instant},
};

use super::track;
use arrayvec::ArrayVec;
use heapless::binary_heap::{Max, Min};
use rand::{Rng, SeedableRng};
use rand_chacha::{ChaCha20Rng, ChaChaRng};

use crate::{
    dataset::DatasetT,
    distance::{l2, DistanceFn, DistanceTracker},
    utils::extend_lifetime,
    vp_tree::Stats,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HnswParams {
    /// normalization factor for level generation
    /// Influences the chance of at which level a point is interted.
    pub level_norm_param: f32,
    pub ef_construction: usize,
    pub m_max: usize,
    pub m_max_0: usize,
    pub distance_fn: DistanceFn,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            level_norm_param: 0.5,
            ef_construction: 20,
            m_max: 10,
            m_max_0: 10,
            distance_fn: l2,
        }
    }
}

impl HnswParams {
    /// Returns max number of connections allowed on layer l
    #[inline(always)]
    fn m_max_on_level(&self, l: usize) -> usize {
        if l == 0 {
            self.m_max_0
        } else {
            self.m_max
        }
    }
}

#[derive(Debug, Clone)]
pub struct Hnsw {
    pub params: HnswParams,
    pub data: Arc<dyn DatasetT>,
    pub layers: Vec<Layer>,
    pub build_stats: Stats,
}

#[derive(Debug, Clone)]
pub struct Layer {
    pub level: usize,
    pub entries: Vec<LayerEntry>,
}

type ID = u32;
pub const NEIGHBORS_LIST_MAX_LEN: usize = 40;

#[derive(Debug, Clone)]
pub struct LayerEntry {
    pub id: ID,
    /// pos where we can find this entry at a lower level.
    /// insignificat on level 0, just set to u32::MAX.
    pub lower_level_idx: u32,
    /// a Max-Heap, such that we can easily pop off the item with the largest distance to make space.
    /// DistAnd<u32> is distances to, and idx's of neighbors
    pub neighbors: Neighbors, // the u32 stores the index in the layer
}

#[derive(Debug, Clone, Default)]
pub struct Neighbors(heapless::BinaryHeap<IAndDist<u32>, Max, NEIGHBORS_LIST_MAX_LEN>);

impl Neighbors {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn iter<'a>(&'a self) -> std::slice::Iter<'a, IAndDist<u32>> {
        self.0.iter()
    }

    /// # Panics
    ///
    /// if no space anymore
    #[inline]
    pub fn insert_asserted(&mut self, idx_in_layer: u32, dist: f32) {
        self.0
            .push(IAndDist {
                i: idx_in_layer,
                dist,
            })
            .expect("no more space in neighbors list, use insert_if_better instead!")
    }

    #[inline]
    pub fn insert_if_better(&mut self, idx_in_layer: u32, dist: f32, max_neighbors: usize) {
        if self.0.len() < max_neighbors {
            self.0
                .push(IAndDist {
                    i: idx_in_layer,
                    dist,
                })
                .expect("should have space too");
        } else {
            // if all neighbors in n_neighbors are closer already, dont add connection from n to q:
            let max_d = self.0.peek().unwrap().dist;
            if max_d > dist {
                // because this is a max heap, pop removes the item with the greatest distance.
                self.0.pop().unwrap();
                self.0
                    .push(IAndDist {
                        i: idx_in_layer,
                        dist,
                    })
                    .unwrap();
            }
        }
    }
}

impl LayerEntry {
    fn new(id: u32, lower_level_idx: u32) -> LayerEntry {
        LayerEntry {
            id,
            lower_level_idx,
            neighbors: Default::default(),
        }
    }
}

impl Hnsw {
    pub fn new(data: Arc<dyn DatasetT>, params: HnswParams) -> Self {
        construct_hnsw(data, params)
    }
    pub fn new_empty(data: Arc<dyn DatasetT>, params: HnswParams) -> Self {
        Hnsw {
            params,
            data,
            layers: vec![],
            build_stats: Stats::default(),
        }
    }

    pub fn knn_search(&self, q_data: &[f32], k: usize) -> (Vec<SearchLayerRes>, Stats) {
        assert_eq!(q_data.len(), self.data.dims());

        let distance = DistanceTracker::new(self.params.distance_fn);
        let start = Instant::now();

        let mut ep_idx_in_layer = 0;
        for i in (1..self.layers.len()).rev() {
            let layer = &self.layers[i];
            let res =
                closest_point_in_layer(layer, &*self.data, q_data, ep_idx_in_layer, &distance);
            ep_idx_in_layer = res.idx_in_lower_layer;
            track!(EdgeDown {
                from: res.id as usize,
                upper_level: i,
            });
        }

        let ef = self.params.ef_construction.max(k);
        let mut ctx = SearchCtx::new(ef);
        closest_points_in_layer(
            &self.layers[0].entries,
            &*self.data,
            q_data,
            &[ep_idx_in_layer],
            ef, // todo! maybe not right?
            &mut ctx,
            &distance,
        );
        let mut results: Vec<SearchLayerRes> = vec![];
        select_neighbors(&self.layers[0], &mut ctx.search_res, k, &mut results);
        let stats = Stats {
            num_distance_calculations: distance.num_calculations(),
            duration: start.elapsed(),
        };
        (results, stats)
    }
}

fn construct_hnsw(data: Arc<dyn DatasetT>, params: HnswParams) -> Hnsw {
    let tracker = DistanceTracker::new(params.distance_fn);
    let start = Instant::now();

    let mut hnsw = Hnsw::new_empty(data, params);
    let len = hnsw.data.len();
    if len == 0 {
        return hnsw;
    }

    // insert a first layer with a first entry
    let mut entries = Vec::with_capacity(len);
    entries.push(LayerEntry {
        id: 0,
        lower_level_idx: u32::MAX,
        neighbors: Default::default(),
    });
    hnsw.layers.push(Layer {
        level: hnsw.layers.len(),
        entries,
    });
    // insert the rest of the points one by one
    let mut insert_ctx = InsertCtx::new(&hnsw.params);
    for id in 1..len as u32 {
        insert(&mut hnsw, id, &tracker, &mut insert_ctx);
    }

    hnsw.build_stats = Stats {
        num_distance_calculations: tracker.num_calculations(),
        duration: start.elapsed(),
    };

    hnsw
}

pub struct InsertCtx {
    ep_idxs_in_layer: Vec<u32>,
    select_neighbors_res: Vec<SearchLayerRes>,
    search_ctx: SearchCtx,
}

impl InsertCtx {
    pub fn new(params: &HnswParams) -> Self {
        let ep_idxs_in_layer: Vec<u32> = Vec::with_capacity(params.ef_construction);
        let neighbors_out: Vec<SearchLayerRes> = vec![];
        let search_ctx: SearchCtx = SearchCtx::new(params.ef_construction);
        Self {
            ep_idxs_in_layer,
            select_neighbors_res: neighbors_out,
            search_ctx,
        }
    }

    pub fn clear(&mut self) {
        self.ep_idxs_in_layer.clear();
        self.select_neighbors_res.clear();
    }
}

fn insert(hnsw: &mut Hnsw, q: ID, distance: &DistanceTracker, ctx: &mut InsertCtx) {
    let q_data = hnsw.data.get(q as usize);
    ctx.clear();

    // /////////////////////////////////////////////////////////////////////////////
    // Phase 0: insert the element on all levels (with empty neighbors)
    // /////////////////////////////////////////////////////////////////////////////
    let mut rng = ChaCha20Rng::seed_from_u64(q as u64);

    let top_l = hnsw.layers.len() - 1; // (previous top l)
    let insert_l = pick_level(hnsw.params.level_norm_param, &mut rng);
    let mut lower_level_idx: u32 = u32::MAX;
    for l in 0..=insert_l {
        let entry = LayerEntry::new(q, lower_level_idx);
        if let Some(layer) = hnsw.layers.get_mut(l) {
            lower_level_idx = layer.entries.len() as u32;
            layer.entries.push(entry);
        } else {
            let layer = Layer {
                level: hnsw.layers.len(),
                entries: vec![entry],
            };
            hnsw.layers.push(layer);
            lower_level_idx = 0;
        }
    }

    // /////////////////////////////////////////////////////////////////////////////
    // Phase 1: find the idx of the entry point at level `insert_l`
    // /////////////////////////////////////////////////////////////////////////////

    // this loop only runs, if insert_l < top_l, otherwise, the entry point is just the first point of the highest level.
    let ep_l = top_l.min(insert_l);
    let mut ep_idx_at_ep_l = 0;
    for l in (insert_l + 1..=top_l).rev() {
        let res = closest_point_in_layer(
            &hnsw.layers[l],
            &*hnsw.data,
            q_data,
            ep_idx_at_ep_l,
            distance,
        );
        ep_idx_at_ep_l = res.idx_in_lower_layer;
    }

    // /////////////////////////////////////////////////////////////////////////////
    // Phase 2
    // /////////////////////////////////////////////////////////////////////////////

    ctx.ep_idxs_in_layer.push(ep_idx_at_ep_l);
    for l in (0..=ep_l).rev() {
        let layer = &mut hnsw.layers[l];
        closest_points_in_layer(
            &layer.entries,
            &*hnsw.data,
            q_data,
            &ctx.ep_idxs_in_layer,
            hnsw.params.ef_construction,
            &mut ctx.search_ctx,
            distance,
        );
        select_neighbors(
            layer,
            &mut ctx.search_ctx.search_res,
            NEIGHBORS_LIST_MAX_LEN,
            &mut ctx.select_neighbors_res,
        );
        // add bidirectional connections from neighbors to q at layer l:
        let idx_of_q_in_l = layer.entries.len() as u32 - 1;
        let m_max = hnsw.params.m_max_on_level(l);
        for n in ctx.select_neighbors_res.iter() {
            // add connection from q to n:

            layer.entries[idx_of_q_in_l as usize]
                .neighbors
                .insert_asserted(n.idx_in_layer, n.d_to_q);

            // add connection from n to q:
            layer.entries[n.idx_in_layer as usize]
                .neighbors
                .insert_if_better(idx_of_q_in_l, n.d_to_q, m_max);
        }

        // set new ep_idxs_in_layer:
        ctx.clear();
        for e in ctx.search_ctx.search_res.iter() {
            ctx.ep_idxs_in_layer.push(e.i)
        }
    }
}

pub fn pick_level(level_norm_param: f32, rng: &mut ChaCha20Rng) -> usize {
    let f = rng.gen::<f32>();
    (-f.ln() * level_norm_param).floor() as usize
}

#[test]
fn testlevel() {
    let rng = &mut ChaCha20Rng::seed_from_u64(42);
    for i in 0..100 {
        println!("{}", pick_level(10.0, rng))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SearchLayerRes {
    pub idx_in_layer: u32,
    pub idx_in_lower_layer: u32,
    pub id: ID,
    pub d_to_q: f32,
}

struct SearchCtx {
    visited_idxs: ahash::AHashSet<u32>,
    /// we need to be able to extract the closest element from this (so we use Reverse<IdxAndDist> to have a min-heap)
    candidates: BinaryHeap<Reverse<IAndDist<u32>>>,
    /// we need to be able to extract the furthest element from this: this is a max heap, the root is the max distance.
    search_res: BinaryHeap<IAndDist<u32>>,
    // Note: seemingly both rayon and this threadpool add immense overhead (10x) when applied to finding the closest neighbors in a layer each in a seperate thread.
    // pool: rayon::ThreadPool,
    // thread_pool: threadpool::ThreadPool,
}

impl SearchCtx {
    fn new(capacity: usize) -> Self {
        SearchCtx {
            visited_idxs: ahash::AHashSet::with_capacity(capacity),
            candidates: BinaryHeap::with_capacity(capacity),
            search_res: BinaryHeap::with_capacity(capacity),
            // thread_pool: threadpool::Builder::new()
            //     .num_threads(MAX_NEIGHBORS)
            //     .build(),
            // pool: rayon::ThreadPoolBuilder::new().build().unwrap(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct IAndDist<T: PartialEq + Copy> {
    pub i: T,
    pub dist: f32,
}

impl<T: PartialEq + Copy> std::fmt::Debug for IAndDist<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {:.3}", self.i, self.dist)
    }
}

impl<T: PartialEq + Copy> PartialEq for IAndDist<T> {
    fn eq(&self, other: &Self) -> bool {
        self.i == other.i && self.dist == other.dist
    }
}
impl<T: PartialEq + Copy> Eq for IAndDist<T> {}
impl<T: PartialEq + Copy> PartialOrd for IAndDist<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}
impl<T: PartialEq + Copy> Ord for IAndDist<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist.total_cmp(&other.dist)
    }
}

/// greedy routing trough the graph, going to the closest neighbor all the time.
fn closest_point_in_layer(
    layer: &Layer,
    data: &dyn DatasetT,
    q_data: &[f32],
    ep_idx_in_layer: u32,
    distance: &DistanceTracker,
) -> SearchLayerRes {
    // let visited_idxs: HashSet<usize> = HashSet::new(); // prob. not needed???

    // initialize best entry to the entry point (at ep_idx_in_layer)
    let mut best_entry_idx_in_layer = ep_idx_in_layer;
    let mut best_entry = &layer.entries[best_entry_idx_in_layer as usize];
    let mut best_entry_d = distance.distance(data.get(best_entry.id as usize), q_data);

    // iterate over all neighbors of best_entry, go to the one with lowest distance to q.
    // if none of them better than current best entry return (greedy routing).
    loop {
        track!(Point {
            id: best_entry.id as usize,
            level: layer.level
        });
        #[cfg(feature = "tracking")]
        let best_entry_id = best_entry.id;

        let mut found_a_better_neighbor = false;
        for idx_and_dist in best_entry.neighbors.iter() {
            let n = &layer.entries[idx_and_dist.i as usize];
            let n_d = distance.distance(data.get(n.id as usize), q_data);
            if n_d < best_entry_d {
                best_entry_d = n_d;
                best_entry = n;
                found_a_better_neighbor = true;
                best_entry_idx_in_layer = idx_and_dist.i;
            }
        }
        if !found_a_better_neighbor {
            return SearchLayerRes {
                idx_in_layer: best_entry_idx_in_layer,
                idx_in_lower_layer: best_entry.lower_level_idx,
                id: best_entry.id,
                d_to_q: best_entry_d,
            };
        }
        track!(EdgeHorizontal {
            from: best_entry_id as usize,
            to: best_entry.id as usize,
            level: layer.level,
            comment: "search"
        });
    }
}

/// after doing this, out.found will contain the relevant found points.
fn closest_points_in_layer(
    entries: &[LayerEntry],
    data: &dyn DatasetT,
    q_data: &[f32],
    ep_idxs_in_layer: &[u32],
    ef: usize, // max number of found items
    ctx: &mut SearchCtx,
    distance: &DistanceTracker,
) {
    // #[cfg(feature = "tracking")]
    // let mut track_to_idx_from_idx: HashMap<u32, u32> = HashMap::new();

    ctx.visited_idxs.clear();
    ctx.candidates.clear();
    ctx.search_res.clear();
    for idx_in_layer in ep_idxs_in_layer.iter().copied() {
        let id = entries[idx_in_layer as usize].id;
        let dist = distance.distance(data.get(id as usize), q_data);
        ctx.visited_idxs.insert(idx_in_layer);
        ctx.candidates.push(Reverse(IAndDist {
            i: idx_in_layer,
            dist,
        }));
        ctx.search_res.push(IAndDist {
            i: idx_in_layer,
            dist,
        })
    }

    while ctx.candidates.len() > 0 {
        let c = ctx.candidates.pop().unwrap(); // remove closest element.
        let mut f = *ctx.search_res.peek().unwrap();
        if c.0.dist > f.dist {
            break; // all elements in found are evaluated (see paper).
        }
        let c_entry = &entries[c.0.i as usize];
        for idx_and_dist in c_entry.neighbors.iter() {
            let idx_in_layer = idx_and_dist.i;
            if ctx.visited_idxs.insert(idx_in_layer) {
                let n_entry = &entries[idx_in_layer as usize];
                let n_data = data.get(n_entry.id as usize);
                let dist = distance.distance(q_data, n_data);
                f = *ctx.search_res.peek().unwrap();
                if dist < f.dist || ctx.search_res.len() < ef {
                    ctx.candidates.push(Reverse(IAndDist {
                        i: idx_in_layer,
                        dist,
                    }));
                    if ctx.search_res.len() < ef {
                        ctx.search_res.push(IAndDist {
                            i: idx_in_layer,
                            dist,
                        });
                    } else {
                        // compare dist to the currently furthest away,
                        // if further than this dist, kick it out and insert the new one instead.
                        if dist < f.dist {
                            ctx.search_res.pop().unwrap();
                            ctx.search_res.push(IAndDist {
                                i: idx_in_layer,
                                dist,
                            });
                        }
                    }
                }
            }
        }
    }
}

// /// after doing this, ctx.found will contain the relevant found points.
// fn closest_points_in_layer_threadpool(
//     entries: &[LayerEntry],
//     data: &dyn DatasetT,
//     q_data: &[f32],
//     ep_idxs_in_layer: &[u32],
//     ef: usize, // max number of found items
//     ctx: &mut SearchCtx,
//     distance: &DistanceTracker,
// ) {
//     // #[cfg(feature = "tracking")]
//     // let mut track_to_idx_from_idx: HashMap<u32, u32> = HashMap::new();
//     ctx.initialize(ep_idxs_in_layer, |idx| {
//         let id = entries[idx as usize].id;
//         distance.distance(data.get(id as usize), q_data)
//     });

//     struct IdxInLayerAndDist {}

//     let mut tmp_indices_and_distances: [IAndDist<u32>; MAX_NEIGHBORS] =
//         [IAndDist { i: 0, dist: 0.0 }; MAX_NEIGHBORS];

//     while ctx.candidates.len() > 0 {
//         // tmp_indices_and_distances = UnsafeCell::new([(u32::MAX, 0.0); MAX_NEIGHBORS]); // maybe not necessary

//         let c = ctx.candidates.pop().unwrap(); // remove closest element.
//         let mut f = *ctx.search_res.peek().unwrap();
//         if c.0.dist > f.dist {
//             break; // all elements in found are evaluated (see paper).
//         }
//         let c_entry = &entries[c.0.i as usize];

//         // compute the distances of all neighbors in parallel:

//         //

//         // Note: checking with this, we see that in 99% of cases the MAX_NEIHGBORS is actually reached.
//         // std::io::stdout().write_fmt(format_args!(" {}\n", c_entry.neighbors.len()));

//         let mut num_unvisited_neighbors: usize = 0;
//         for idx_and_dist in c_entry.neighbors.iter() {
//             let idx_in_layer = idx_and_dist.i;
//             let newly_visited = ctx.visited_idxs.insert(idx_in_layer);
//             if newly_visited {
//                 unsafe {
//                     tmp_indices_and_distances
//                         .get_unchecked_mut(num_unvisited_neighbors)
//                         .i = idx_in_layer;
//                 }
//                 num_unvisited_neighbors += 1;
//             }
//         }
//         let tmp_start = tmp_indices_and_distances.as_mut_ptr();
//         let barrier = Arc::new(Barrier::new(num_unvisited_neighbors + 1));
//         for i in 0..num_unvisited_neighbors {
//             let idx_and_dist_ptr = unsafe { tmp_start.add(i) };
//             let idx = unsafe { *(idx_and_dist_ptr as *mut u32) };
//             let neighbor_id = &entries[idx as usize].id;

//             let n_data = extend_lifetime(data.get(*neighbor_id as usize));
//             let q_data = extend_lifetime(q_data);
//             let distance = extend_lifetime(distance);
//             let write_dist_ptr = idx_and_dist_ptr as usize + std::mem::size_of::<u32>();

//             let barrier = barrier.clone();
//             ctx.thread_pool.execute(move || unsafe {
//                 let write_dist_ptr = write_dist_ptr as *mut f32;
//                 let distance = distance.distance(q_data, n_data);
//                 (write_dist_ptr as *mut f32).write(distance);
//                 barrier.wait();
//             });
//         }
//         barrier.wait();

//         for idx_and_dist in &tmp_indices_and_distances[0..num_unvisited_neighbors] {
//             let i = idx_and_dist.i;
//             let dist = idx_and_dist.dist;
//             f = *ctx.search_res.peek().unwrap();
//             if dist < f.dist || ctx.search_res.len() < ef {
//                 ctx.candidates.push(Reverse(IAndDist { i, dist }));
//                 if ctx.search_res.len() < ef {
//                     ctx.search_res.push(IAndDist { i, dist });
//                 } else {
//                     // compare dist to the currently furthest away,
//                     // if further than this dist, kick it out and insert the new one instead.
//                     if dist < f.dist {
//                         ctx.search_res.pop().unwrap();
//                         ctx.search_res.push(IAndDist { i, dist });
//                     }
//                 }
//             }
//         }
//     }
// }

// /// after doing this, ctx.found will contain the relevant found points.
// fn closest_points_in_layer_rayon(
//     entries: &[LayerEntry],
//     data: &dyn DatasetT,
//     q_data: &[f32],
//     ep_idxs_in_layer: &[u32],
//     ef: usize, // max number of found items
//     ctx: &mut SearchCtx,
//     distance: &DistanceTracker,
// ) {
//     // #[cfg(feature = "tracking")]
//     // let mut track_to_idx_from_idx: HashMap<u32, u32> = HashMap::new();
//     ctx.initialize(ep_idxs_in_layer, |idx| {
//         let id = entries[idx as usize].id;
//         distance.distance(data.get(id as usize), q_data)
//     });

//     let mut tmp_num_unvisited_neighbors: usize = 0;
//     let mut tmp_indices_and_distances: [(u32, f32); MAX_NEIGHBORS] =
//         [(u32::MAX, 0.0); MAX_NEIGHBORS];

//     while ctx.candidates.len() > 0 {
//         // tmp_indices_and_distances = UnsafeCell::new([(u32::MAX, 0.0); MAX_NEIGHBORS]); // maybe not necessary

//         let c = ctx.candidates.pop().unwrap(); // remove closest element.
//         let mut f = *ctx.search_res.peek().unwrap();
//         if c.0.dist > f.dist {
//             break; // all elements in found are evaluated (see paper).
//         }
//         let c_entry = &entries[c.0.i as usize];

//         // compute the distances of all neighbors in parallel:

//         let tmp_indices_and_distances_ptr = tmp_indices_and_distances.as_mut_ptr() as usize; // as usize to get around the restriction of Send/Sync for ptrs.

//         // Note: checking with this, we see that in 99% of cases the MAX_NEIHGBORS is actually reached.
//         // std::io::stdout().write_fmt(format_args!(" {}\n", c_entry.neighbors.len()));
//         ctx.pool.scope(|s| {
//             let mut i: usize = 0;
//             for idx_and_dist in c_entry.neighbors.iter() {
//                 let idx_in_layer = idx_and_dist.i;
//                 let newly_visited = ctx.visited_idxs.insert(idx_in_layer);
//                 if newly_visited {
//                     s.spawn(move |_| {
//                         let i = i;
//                         let n_entry = &entries[idx_in_layer as usize];
//                         let n_data = data.get(n_entry.id as usize);
//                         let dist = distance.distance(q_data, n_data);
//                         // write the distance back to the stack-allocated buffer:
//                         unsafe {
//                             let addr = (tmp_indices_and_distances_ptr as *mut (u32, f32)).add(i);
//                             addr.write((idx_in_layer, dist));
//                         };
//                     });
//                     i += 1;
//                 }
//             }
//             tmp_num_unvisited_neighbors = i;
//         });

//         for (idx_in_layer, dist_to_q) in &tmp_indices_and_distances[0..tmp_num_unvisited_neighbors]
//         {
//             let idx_in_layer = *idx_in_layer;
//             let dist = *dist_to_q;
//             f = *ctx.search_res.peek().unwrap();
//             if dist < f.dist || ctx.search_res.len() < ef {
//                 ctx.candidates.push(Reverse(DistAnd {
//                     i: idx_in_layer,
//                     dist,
//                 }));
//                 if ctx.search_res.len() < ef {
//                     ctx.search_res.push(DistAnd {
//                         i: idx_in_layer,
//                         dist,
//                     });
//                 } else {
//                     // compare dist to the currently furthest away,
//                     // if further than this dist, kick it out and insert the new one instead.
//                     if dist < f.dist {
//                         ctx.search_res.pop().unwrap();
//                         ctx.search_res.push(DistAnd {
//                             i: idx_in_layer,
//                             dist,
//                         });
//                     }
//                 }
//             }
//         }
//     }
// }

// #[cfg(feature = "tracking")]
// {
//     for e in out.found.iter() {
//         let to = layer.entries[e.idx_in_layer as usize].id;
//         let Some(from_idx) = track_to_idx_from_idx.get(&e.idx_in_layer) else {
//             continue;
//         };
//         let from = layer.entries[*from_idx as usize].id;
//         track!(EdgeHorizontal {
//             from,
//             to,
//             level: layer.level
//         })
//     }
// }

/// todo! what if less neighbors there? will fail??
fn select_neighbors(
    layer: &Layer,
    candidates: &mut BinaryHeap<IAndDist<u32>>, // a max-heap where the root is the largest-dist element.
    n: usize,
    out: &mut Vec<SearchLayerRes>,
) {
    while candidates.len() > n {
        // removes the furthest element from candidates, leaving only the n closest ones in it.
        candidates.pop();
    }

    out.clear();
    for c in candidates.iter() {
        let entry = &layer.entries[c.i as usize];
        out.push(SearchLayerRes {
            idx_in_layer: c.i,
            idx_in_lower_layer: entry.lower_level_idx,
            id: entry.id,
            d_to_q: c.dist,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::distance::l2;

    use super::Hnsw;

    #[test]
    fn build_hnsw() {
        let data = Arc::new(vec![
            [0.0, 0.0],
            [2.0, 1.0],
            [2.0, 6.0],
            [3.0, 6.0],
            [4.0, 6.0],
            [7.0, 7.0],
        ]);

        let hnsw = Hnsw::new(
            data,
            super::HnswParams {
                level_norm_param: 2.0,
                ef_construction: 4,
                m_max: 2,
                m_max_0: 3,
                distance_fn: l2,
            },
        );

        dbg!(hnsw);
    }
}

/*



fn insert_first(hnsw: &mut Hnsw, q: ID, mL: f32) {
     let l = pick_level(mL);
     // insert point at all levels
}


fn insert(
    hnsw: &mut Hnsw,
    q: ID,             id (idx) of pt
    M: usize,               number of establishedconnections M ??? , I think this means, how many neighbors to search for and compare on each layer????
    Mmax: usize,            maximum number of connections for each element per layer above layer 0
    Mmax_0: usize           maximum number of connections for each element on layer 0
    efConstruction: ??,
    mL: f32                  normalization factor for level generation
)   {
    let mut W : [ID]  = [];
    let mut ep : [ID] = [get_entry(hnsw)]  // entry pt for hnsw
    let L =   // level of ep
    let l = pick_level(mL);

    for lc in (l+1..=L).rev() {
        W = search_layer(hnsw, q, ep, ef = 1, lc)
        ep = [get_nearest_element(W, q)]
    }


    for lc in (0..=min(l,L)).rev() {
        let Mmax = if lc == 0 {  Mmax_0 } else { Mmax }
        W : [ID]  = search_layer(hnsw, q, ep, efConstruction, lc);
        neighbors = select_neighbors(hnsw, q, W, M, lc);

        // add bidirectional connections from neighbors to q at layer lc:

        for e in neighbors {
            // shrink connections if needed
            e_conn: [ID] = neighbourhood(hnsw, e, lc)
            if e_conn.len() > Mmax {
                // shrink connections of e:
                eNewConn = select_neighbors(hnsw, e, eConn, Mmax, lc);
                set_connections(hnsw, e, lc, eNewConn);
            }
        }

        ep = W
    }

    if l > L
        // set enter point for hnsw to q
        set_new_enter_point(hnsw, q, l)
}




fn set_new_enter_point(hnsw: &mut Hnsw, q: ID, l: usize) {
    // insert layers such that layer l can exist.
}

fn get_connections(hnsw: &Hnsw, e: ID, lc: usize) -> [ID] {




}

fn set_connections(hnsw: &mut Hnsw, e: ID, lc: usize, new_connections: [ID]) {


}

fn get_nearest_element(W: &[ID], q: &[f32]) -> ID {


}


fn select_neighbors(
    hnsw: &Hnsw,
    W: [ID],
    M: ??,
    lc: usize,        // layer we are looking at
) -> [ID]  {


}

fn search_layer(
    hnsw: &Hnsw,
    q: ID,
    ep: [ID]     // entry points
    ef: usize,   // number of elements to return
    lc: usize    // layer we are looking at
) -> ?? {
    let visited   : HashSet<ID>   =   ep; // visited elements
    let frontier                  =   ep; // candidates
    let found                     =   ep; // dyncamic list of found neighbors
    while candidates.len() > 0{
        c = extract nearest element from frontier to q
        f = get furthest element from found to q
        if distance(c, q) > distance(f,q){
            break; // all elements in W are evaluated
        }
        for e in get_connections(hnsw, c, lc) {
            if !visited.contains(e){
                visited.insert(e);
                f = get furthest element from found to q
                if distance(e, q) < distance (f,q)     || found.len()  <   ef{
                    frontier.push(e);
                    found.push(e);
                    if frontier.len() > ef {
                        // remove the furthest element from found to q.
                    }
                }
            }
        }
    }
    return found;
}


fn pick_level(mL: f32) -> usize {
    let r : f32 = rng.gen(); // 0 to 1
    let l = -ln(r)*mL;
    return floor(l);
}

fn top_level(hnsw: &Hnsw) -> usize  {
    hnsw.layers.len() - 1
}


fn get_entry(hnsw: &Hnsw) -> ID {
    self.layers.last().unwrap()[0]
}



*/
