use std::{
    cell::UnsafeCell,
    cmp::Reverse,
    collections::BinaryHeap,
    fmt::Debug,
    hash::Hash,
    io::Write,
    process::id,
    sync::{Arc, Barrier},
    time::{Duration, Instant},
};

use super::if_tracking;
use arrayvec::ArrayVec;
use heapless::binary_heap::{Max, Min};
use nanoserde::{DeJson, SerJson};
use rand::{Rng, SeedableRng};
use rand_chacha::{ChaCha20Rng, ChaChaRng};
use rayon::iter::Rev;

use crate::{
    dataset::DatasetT,
    distance::{l2, Distance, DistanceFn, DistanceTracker},
    utils::extend_lifetime,
    utils::Stats,
};

#[derive(Debug, Clone, Copy, PartialEq, SerJson, DeJson)]
pub struct HnswParams {
    /// normalization factor for level generation
    /// Influences the chance of at which level a point is interted.
    pub level_norm_param: f32,
    pub ef_construction: usize,
    pub m_max: usize,
    pub m_max_0: usize,
    pub distance: Distance,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            level_norm_param: 0.5,
            ef_construction: 30,
            m_max: 10,
            m_max_0: 20,
            distance: Distance::L2,
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

pub const NEIGHBORS_LIST_MAX_LEN: usize = 40;

#[derive(Debug, Clone)]
pub struct LayerEntry {
    pub id: usize,
    /// pos where we can find this entry at a lower level.
    /// insignificat on level 0, just set to usize::MAX.
    pub lower_level_idx: usize,
    /// a Max-Heap, such that we can easily pop off the item with the largest distance to make space.
    /// DistAnd<usize> is distances to, and idx's of neighbors
    pub neighbors: Neighbors, // the usize stores the index in the layer
}

// TODO! I should probably change the usize back to a u32, because together with the f32,
// each element then will only take 8 bytes instead of 16 bytes.
#[derive(Debug, Clone, Default)]
pub struct Neighbors(heapless::BinaryHeap<DistAnd<usize>, Max, NEIGHBORS_LIST_MAX_LEN>);

impl Neighbors {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn iter<'a>(&'a self) -> std::slice::Iter<'a, DistAnd<usize>> {
        self.0.iter()
    }

    /// # Panics
    ///
    /// if no space anymore
    #[inline]
    pub fn insert_asserted(&mut self, idx_in_layer: usize, dist: f32) {
        self.0
            .push(DistAnd(dist, idx_in_layer))
            .expect("no more space in neighbors list, use insert_if_better instead!")
    }

    /// Returns true if insert was successful
    #[inline]
    pub fn insert_if_better(
        &mut self,
        idx_in_layer: usize,
        dist: f32,
        max_neighbors: usize,
    ) -> bool {
        if self.0.len() < max_neighbors {
            self.0
                .push(DistAnd(dist, idx_in_layer))
                .expect("should have space too");
            return true;
        } else {
            // replace max dist element if distance is smaller.
            let mut max_d_element = self.0.peek_mut().unwrap();
            if dist < max_d_element.0 {
                *max_d_element = DistAnd(dist, idx_in_layer);
                return true;
            } else {
                return false;
            }
        }
    }
}

impl LayerEntry {
    fn new(id: usize, lower_level_idx: usize) -> LayerEntry {
        LayerEntry {
            id,
            lower_level_idx,
            neighbors: Default::default(),
        }
    }
}

impl Hnsw {
    pub fn new(data: Arc<dyn DatasetT>, params: HnswParams, seed: u64) -> Self {
        construct_hnsw(data, params, seed)
    }
    pub fn new_empty(data: Arc<dyn DatasetT>, params: HnswParams) -> Self {
        Hnsw {
            params,
            data,
            layers: vec![],
            build_stats: Stats::default(),
        }
    }

    pub fn knn_search(&self, q_data: &[f32], k: usize, ef: usize) -> (Vec<SearchLayerRes>, Stats) {
        assert_eq!(q_data.len(), self.data.dims());

        let distance = DistanceTracker::new(self.params.distance);
        let start = Instant::now();

        let mut ep_idx_in_layer: usize = 0;
        for i in (1..self.layers.len()).rev() {
            let layer = &self.layers[i];
            let res: SearchLayerRes =
                closest_point_in_layer(layer, &*self.data, q_data, ep_idx_in_layer, &distance);
            ep_idx_in_layer = res.idx_in_lower_layer;

            if_tracking!(Tracking.add_event(Event::EdgeDown {
                from: res.id,
                upper_level: i,
            }));
        }

        let ef = ef.max(k);
        let mut ctx = SearchCtx::new(ef);
        closest_points_in_layer(
            &self.layers[0].entries,
            &*self.data,
            q_data,
            &[ep_idx_in_layer],
            ef,
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

fn construct_hnsw(data: Arc<dyn DatasetT>, params: HnswParams, seed: u64) -> Hnsw {
    let tracker = DistanceTracker::new(params.distance);
    let start = Instant::now();

    let mut hnsw = Hnsw::new_empty(data, params);
    let len = hnsw.data.len();
    if len == 0 {
        return hnsw;
    }

    // insert a first layer with a first entry
    // let mut entries = Vec::with_capacity(len);
    // entries.push(LayerEntry {
    //     id: 0,
    //     lower_level_idx: usize::MAX,
    //     neighbors: Default::default(),
    // });
    // hnsw.layers.push(Layer {
    //     level: hnsw.layers.len(),
    //     entries,
    // });
    // insert the rest of the points one by one
    let mut insert_ctx = InsertCtx::new(&hnsw.params);
    for id in 1..len {
        insert(&mut hnsw, id, &tracker, &mut insert_ctx, seed);
    }

    hnsw.build_stats = Stats {
        num_distance_calculations: tracker.num_calculations(),
        duration: start.elapsed(),
    };

    hnsw
}

pub struct InsertCtx {
    ep_idxs_in_layer: Vec<usize>,
    select_neighbors_res: Vec<SearchLayerRes>,
    search_ctx: SearchCtx,
}

impl InsertCtx {
    pub fn new(params: &HnswParams) -> Self {
        let ep_idxs_in_layer: Vec<usize> = Vec::with_capacity(params.ef_construction);
        let neighbors_out: Vec<SearchLayerRes> = vec![];
        let search_ctx: SearchCtx = SearchCtx::new(params.ef_construction);
        Self {
            ep_idxs_in_layer,
            select_neighbors_res: neighbors_out,
            search_ctx,
        }
    }
}

fn insert(hnsw: &mut Hnsw, q: usize, distance: &DistanceTracker, ctx: &mut InsertCtx, seed: u64) {
    let q_data = hnsw.data.get(q);

    ctx.ep_idxs_in_layer.clear();
    ctx.select_neighbors_res.clear(); // todo! unclear if necessary here

    // /////////////////////////////////////////////////////////////////////////////
    // Phase 0: insert the element on all levels (with empty neighbors)
    // /////////////////////////////////////////////////////////////////////////////
    let mut rng = ChaCha20Rng::seed_from_u64(seed ^ q as u64);

    let top_l = if hnsw.layers.is_empty() {
        0
    } else {
        hnsw.layers.len() - 1
    }; // (previous top l)
    let insert_l = pick_level(hnsw.params.level_norm_param, &mut rng);
    let mut lower_level_idx: usize = usize::MAX;
    for l in 0..=insert_l {
        let entry = LayerEntry::new(q, lower_level_idx);
        if let Some(layer) = hnsw.layers.get_mut(l) {
            lower_level_idx = layer.entries.len();
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
        // set new ep_idxs_in_layer to indices below:
        ctx.ep_idxs_in_layer.clear();
        for e in ctx.search_ctx.search_res.iter() {
            let e_entry = &layer.entries[e.1];
            ctx.ep_idxs_in_layer.push(e_entry.lower_level_idx);
        }
        ctx.select_neighbors_res.clear();
        select_neighbors(
            layer,
            &mut ctx.search_ctx.search_res,
            hnsw.params.m_max, // Note: m_max not m_max_0, even if on bottom layer!!!
            &mut ctx.select_neighbors_res,
        );
        // select_neighbors_heuristic(
        //     q_data,
        //     &*hnsw.data,
        //     distance,
        //     layer,
        //     &mut ctx.search_ctx.search_res,
        //     &mut ctx.search_ctx.candidates,
        //     NEIGHBORS_LIST_MAX_LEN,
        //     &mut ctx.select_neighbors_res,
        // );

        // add bidirectional connections from neighbors to q at layer l:
        let idx_of_q_in_layer = layer.entries.len() - 1;
        let m_max = hnsw.params.m_max_on_level(l);
        for n in ctx.select_neighbors_res.iter() {
            // this is an unsolved issue:
            // if n.id == q {
            //     // println!("try to put {q} twice!!  (layer {l})")
            //     continue;
            // }
            // add connection from q to n:

            // ATTENTION: There is likely an issue here, that the neighbors found can be the idx itself.
            // THis might case the bad recall compared to the other implementations.
            // assert!(n.idx_in_layer != idx_of_q_in_layer);

            layer.entries[idx_of_q_in_layer].neighbors.insert_if_better(
                n.idx_in_layer,
                n.d_to_q,
                m_max,
            );

            // add connection from n to q:
            layer.entries[n.idx_in_layer].neighbors.insert_if_better(
                idx_of_q_in_layer,
                n.d_to_q,
                m_max,
            );
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
    pub idx_in_layer: usize,
    pub idx_in_lower_layer: usize,
    pub id: usize,
    pub d_to_q: f32,
}

struct SearchCtx {
    visited_idxs: ahash::AHashSet<usize>,
    /// we need to be able to extract the closest element from this (so we use Reverse<IdxAndDist> to have a min-heap)
    candidates: BinaryHeap<Reverse<DistAnd<usize>>>,
    /// we need to be able to extract the furthest element from this: this is a max heap, the root is the max distance.
    search_res: BinaryHeap<DistAnd<usize>>,
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
pub struct DistAnd<T: PartialEq + Copy>(pub f32, pub T);
impl<T: PartialEq + Copy> DistAnd<T> {
    #[inline(always)]
    pub fn dist(&self) -> f32 {
        self.0
    }
}

impl<T: PartialEq + Copy> std::fmt::Debug for DistAnd<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {:.3}", self.1, self.0)
    }
}

impl<T: PartialEq + Copy> PartialEq for DistAnd<T> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1 && self.0 == other.0
    }
}
impl<T: PartialEq + Copy> Eq for DistAnd<T> {}
impl<T: PartialEq + Copy> PartialOrd for DistAnd<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl<T: PartialEq + Copy> Ord for DistAnd<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// greedy routing trough the graph, going to the closest neighbor all the time.
fn closest_point_in_layer(
    layer: &Layer,
    data: &dyn DatasetT,
    q_data: &[f32],
    ep_idx_in_layer: usize,
    distance: &DistanceTracker,
) -> SearchLayerRes {
    // let visited_idxs: HashSet<usize> = HashSet::new(); // prob. not needed???

    // initialize best entry to the entry point (at ep_idx_in_layer)
    let mut best_entry_idx_in_layer = ep_idx_in_layer;
    let mut best_entry = &layer.entries[best_entry_idx_in_layer];
    let mut best_entry_d = distance.distance(data.get(best_entry.id), q_data);

    // iterate over all neighbors of best_entry, go to the one with lowest distance to q.
    // if none of them better than current best entry return (greedy routing).
    loop {
        if_tracking!(Tracking.add_event(Event::Point {
            id: best_entry.id,
            level: layer.level
        }));
        #[cfg(feature = "tracking")]
        let best_entry_id = best_entry.id;

        let mut found_a_better_neighbor = false;
        for idx_and_dist in best_entry.neighbors.iter() {
            let n = &layer.entries[idx_and_dist.1];
            let n_d = distance.distance(data.get(n.id), q_data);
            if n_d < best_entry_d {
                best_entry_d = n_d;
                best_entry = n;
                found_a_better_neighbor = true;
                best_entry_idx_in_layer = idx_and_dist.1;
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

        if_tracking!(Tracking.add_event(Event::EdgeHorizontal {
            from: best_entry_id,
            to: best_entry.id,
            level: layer.level,
            comment: "search"
        }));
    }
}

/// after doing this, ctx.search_res will contain the relevant found points.
fn closest_points_in_layer(
    entries: &[LayerEntry],
    data: &dyn DatasetT,
    q_data: &[f32],
    ep_idxs_in_layer: &[usize],
    ef: usize, // max number of found items
    ctx: &mut SearchCtx,
    distance: &DistanceTracker,
) {
    // #[cfg(feature = "tracking")]
    // let mut track_to_idx_from_idx: HashMap<usize, usize> = HashMap::new();

    ctx.visited_idxs.clear();
    ctx.candidates.clear();
    ctx.search_res.clear();
    for idx_in_layer in ep_idxs_in_layer.iter().copied() {
        let id = entries[idx_in_layer].id;
        let dist = distance.distance(data.get(id), q_data);
        ctx.visited_idxs.insert(idx_in_layer);
        ctx.candidates.push(Reverse(DistAnd(dist, idx_in_layer)));
        ctx.search_res.push(DistAnd(dist, idx_in_layer))
    }

    while ctx.candidates.len() > 0 {
        let c = ctx.candidates.pop().unwrap(); // remove closest element.
        let f = *ctx.search_res.peek().unwrap();
        if c.0.dist() > f.dist() {
            break; // all elements in found are evaluated (see paper).
        }
        let c_entry = &entries[c.0 .1];
        for idx_and_dist in c_entry.neighbors.iter() {
            let idx_in_layer = idx_and_dist.1;
            if ctx.visited_idxs.insert(idx_in_layer) {
                let n_entry = &entries[idx_in_layer];
                let n_data = data.get(n_entry.id);
                let dist = distance.distance(q_data, n_data);
                let worst = *ctx.search_res.peek().unwrap(); // f in paper
                let search_res_not_full = ctx.search_res.len() < ef;
                let dist_better_than_worst = dist < worst.dist();
                if search_res_not_full {
                    ctx.candidates.push(Reverse(DistAnd(dist, idx_in_layer)));
                    ctx.search_res.push(DistAnd(dist, idx_in_layer));
                } else if dist_better_than_worst {
                    ctx.candidates.push(Reverse(DistAnd(dist, idx_in_layer)));
                    ctx.search_res.pop().unwrap();
                    ctx.search_res.push(DistAnd(dist, idx_in_layer));
                }
            }
        }
    }
}

fn select_neighbors(
    layer: &Layer,
    candidates: &mut BinaryHeap<DistAnd<usize>>, // a max-heap where the root is the largest-dist element.
    k: usize,
    out: &mut Vec<SearchLayerRes>,
) {
    while candidates.len() > k {
        // removes the furthest element from candidates, leaving only the n closest ones in it.
        candidates.pop();
    }

    out.clear();
    for c in candidates.iter() {
        let entry = &layer.entries[c.1];
        out.push(SearchLayerRes {
            idx_in_layer: c.1,
            idx_in_lower_layer: entry.lower_level_idx,
            id: entry.id,
            d_to_q: c.dist(),
        })
    }
}

fn select_neighbors_heuristic(
    q_data: &[f32],
    data: &dyn DatasetT,
    distance: &DistanceTracker,
    layer: &Layer,
    candidates: &mut BinaryHeap<DistAnd<usize>>, // a max-heap where the root is the largest-dist element.
    w: &mut BinaryHeap<Reverse<DistAnd<usize>>>, // working queue (???)
    k: usize,                                    // number of neighbors to put into out
    out: &mut Vec<SearchLayerRes>,
) {
    out.clear();
    w.clear();
    for c in candidates.iter() {
        w.push(Reverse(*c));
    }

    let extend_candidates = false;
    let keep_pruned_candidates = false;

    if extend_candidates {
        for c in candidates.iter() {
            let c_entry = &layer.entries[c.1];
            for nei in c_entry.neighbors.iter() {
                if !w.iter().any(|e| e.0 .1 == nei.1) {
                    let nei_id = layer.entries[nei.1].id;
                    let nei_data = data.get(nei_id);
                    let dist = distance.distance(q_data, nei_data);
                    w.push(Reverse(DistAnd(dist, nei.1)))
                }
            }
        }
    }

    let mut wd: BinaryHeap<Reverse<DistAnd<usize>>> = Default::default();

    while w.len() > 0 && out.len() < k {
        let e = w.pop().unwrap().0;
        // if e closer to q than any element from R:
        if out.iter().any(|o| e.0 < o.d_to_q) {
            let e_entry = &layer.entries[e.1];
            out.push(SearchLayerRes {
                idx_in_layer: e.1,
                idx_in_lower_layer: e_entry.lower_level_idx,
                id: e_entry.id,
                d_to_q: e.0,
            })
        } else if keep_pruned_candidates {
            wd.push(Reverse(e))
        }
    }

    if keep_pruned_candidates {
        while wd.len() > 0 && out.len() < k {
            let e = wd.pop().unwrap().0;
            let e_entry = &layer.entries[e.1];
            out.push(SearchLayerRes {
                idx_in_layer: e.1,
                idx_in_lower_layer: e_entry.lower_level_idx,
                id: e_entry.id,
                d_to_q: e.0,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::distance::{l2, Distance};

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
                distance: Distance::L2,
            },
            42,
        );

        dbg!(hnsw);
    }
}
