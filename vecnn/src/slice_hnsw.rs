use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
    ptr::NonNull,
    sync::Arc,
    time::Instant,
    usize,
};

use rand::{Rng, SeedableRng};
use rand_chacha::{rand_core::le, ChaCha20Rng};

use crate::{
    dataset::DatasetT,
    distance::DistanceTracker,
    hnsw::{DistAnd, HnswParams},
    utils::Stats,
    utils::{SliceBinaryHeap, SlicesMemory},
};

pub const MAX_LAYERS: usize = 16;
pub struct SliceHnsw {
    pub data: Arc<dyn DatasetT>,
    pub layers: heapless::Vec<Layer, MAX_LAYERS>,
    pub params: HnswParams,
    pub build_stats: Stats,
}

pub type Layers = heapless::Vec<Layer, MAX_LAYERS>;

impl SliceHnsw {
    pub fn new(data: Arc<dyn DatasetT>, params: HnswParams, seed: u64) -> Self {
        Self::new_strategy_1(data, params, seed)
    }

    pub fn new_strategy_1(data: Arc<dyn DatasetT>, params: HnswParams, seed: u64) -> Self {
        s1_construct_hnsw(data, params, seed)
    }

    pub fn new_strategy_2(data: Arc<dyn DatasetT>, params: HnswParams, seed: u64) -> Self {
        s2_construct_hnsw(data, params, seed)
    }

    pub fn new_empty(data: Arc<dyn DatasetT>, params: HnswParams) -> Self {
        SliceHnsw {
            params,
            data,
            layers: Default::default(),
            build_stats: Stats::default(),
        }
    }

    /// the usize's returned are actual data ids, not layers indices!
    pub fn knn_search(&self, q_data: &[f32], k: usize, ef: usize) -> (Vec<DistAnd<usize>>, Stats) {
        let mut buffers = SearchBuffers::new();
        let distance = DistanceTracker::new(self.params.distance);
        let start_time = Instant::now();

        let mut ep_idx: usize = 0;
        let top_layer = self.layers.len() - 1;
        for l in (1..=top_layer).rev() {
            let layer = &self.layers[l];
            search_layer(
                &*self.data,
                &distance,
                &mut buffers,
                layer,
                q_data,
                1,
                &[ep_idx],
            );
            let this_layer_ep_idx = buffers.found.as_slice()[0].1;
            // let this_layer_ep_idx = search_layer_ef_1(
            //     &*self.data,
            //     &distance,
            //     &mut buffers.visited,
            //     layer,
            //     q_data,
            //     ep_idx,
            // );
            ep_idx = layer.entries[this_layer_ep_idx].lower_level_idx;
        }
        let layer_0 = &self.layers[0];
        search_layer(
            &*self.data,
            &distance,
            &mut buffers,
            layer_0,
            q_data,
            ef,
            &[ep_idx],
        );

        // keep best k of the ef found elements:
        while buffers.found.len() > k {
            buffers.found.pop();
        }
        let mut results: Vec<DistAnd<usize>> = vec![];
        for DistAnd(dist, layer_0_idx) in buffers.found.drain() {
            let id = layer_0.entries[layer_0_idx].id;
            results.push(DistAnd(dist, id));
        }

        let stats = Stats {
            duration: start_time.elapsed(),
            num_distance_calculations: distance.num_calculations(),
        };

        (results, stats)
    }
}

/// This layer can be used in two different ways:
///
/// Way 1:
/// - set `entries_cap`
/// - call `allocate_neighbors_memory`
/// - for each entry that should be added, call `add_entry_assuming_allocated_memory`
///
/// Way 2:
/// - for each entry call `add_entry_with_uninit_neighbors`
/// - once all entries are added, call `allocate_neighbors_memory_and_suballocate_neighbors_lists`
#[derive(Debug)]
pub struct Layer {
    pub memory: SlicesMemory<Neighbor>,
    pub m_max: usize,       // max many neighbors per entry
    pub entries_cap: usize, // how many entries should this layer be max able to hold
    pub entries: Vec<LayerEntry>,
}

#[derive(Debug)]
#[repr(C)]
pub struct LayerEntry {
    pub id: usize,
    pub lower_level_idx: usize,
    pub neighbors: Neighbors,
}

pub type Neighbor = DistAnd<usize>;
pub type Neighbors = SliceBinaryHeap<'static, Neighbor>;

impl Layer {
    /// just leave entries_cap_hint at 0 if you don't know yet.
    pub fn new(m_max: usize) -> Self {
        assert!(m_max != 0);
        // Potential improvement: could maybe estimate based on layer number how many elements make it up to here and preallocate entries Vec with okay cap.
        Layer {
            memory: SlicesMemory::new_uninit(),
            m_max,
            entries_cap: 0,
            entries: vec![],
        }
    }

    /// Set self.entries_cap first!
    /// After calling this, the number of elements in this layer should not change anymore, so you can start modifying the neighbors of the elements.
    pub fn allocate_neighbors_memory(&mut self) {
        self.memory = SlicesMemory::new(self.entries_cap, self.m_max);
        if self.entries.capacity() == 0 {
            self.entries = Vec::with_capacity(self.entries_cap);
        }
    }

    /// returns the idx of this entry in this layer
    pub fn add_entry_assuming_allocated_memory(
        &mut self,
        id: usize,
        lower_level_idx: usize,
    ) -> usize {
        // allocate a new neighbors list from the big slice that this layer owns (which contains enough memory from the start to hold exactly all elements)
        let slice = unsafe { self.memory.static_slice_at(self.entries.len()) };
        let entry = LayerEntry {
            id,
            lower_level_idx,
            neighbors: Neighbors::new(slice),
        };
        let idx = self.entries.len();
        self.entries.push(entry);
        idx
    }

    /// returns the idx of this entry in this layer
    #[inline]
    pub fn add_entry_with_uninit_neighbors(&mut self, id: usize, lower_level_idx: usize) -> usize {
        let entry = LayerEntry {
            id,
            lower_level_idx,
            neighbors: unsafe { Neighbors::new_uninitialized() },
        };
        let idx = self.entries.len();
        self.entries.push(entry);
        idx
    }

    /// Assumes all the neighbors lists are uninitialized.
    /// Looks at how many entries are there, allocated the needed memory,
    /// then gives out a slice of this memory to each entry to keep its neighbors list.
    pub fn allocate_neighbors_memory_and_suballocate_neighbors_lists(&mut self) {
        self.entries_cap = self.entries.len();
        self.allocate_neighbors_memory();
        for (i, entry) in self.entries.iter_mut().enumerate() {
            let slice = unsafe { self.memory.static_slice_at(i) };
            entry.neighbors = Neighbors::new(slice)
        }
    }
}

// /////////////////////////////////////////////////////////////////////////////
// SECTION: Insert Strategy 1: preallocate layers memory and determine levels, but insert pts sequentially
// /////////////////////////////////////////////////////////////////////////////

fn s1_construct_hnsw(data: Arc<dyn DatasetT>, params: HnswParams, seed: u64) -> SliceHnsw {
    let start_time = Instant::now();
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let data_len = data.len();

    let (mut layers, insert_levels) =
        s1_determine_insert_levels_and_allocate_layers_memory(data_len, &params, &mut rng);

    let mut ctx = InsertCtx {
        distance: DistanceTracker::new(params.distance),
        params,
        data: &*data,
        layers: &mut layers,
        search_buffers: SearchBuffers::new(),
        selected_neighbors: vec![],
        entry_points: vec![],
        top_level: 0,
    };
    for id in 0..data_len {
        s1_insert_element(&mut ctx, id, insert_levels[id]);
    }
    assert_eq!(ctx.top_level + 1, ctx.layers.len());

    let build_stats = Stats {
        num_distance_calculations: ctx.distance.num_calculations(),
        duration: start_time.elapsed(),
    };

    SliceHnsw {
        data,
        layers,
        params,
        build_stats,
    }
}

/// returns layers and insert level of each node in the dataset
fn s1_determine_insert_levels_and_allocate_layers_memory(
    data_len: usize,
    params: &HnswParams,
    rng: &mut ChaCha20Rng,
) -> (heapless::Vec<Layer, MAX_LAYERS>, Vec<usize>) {
    let mut layers: heapless::Vec<Layer, MAX_LAYERS> = Default::default();
    let mut insert_levels: Vec<usize> = Vec::with_capacity(data_len);
    for _ in 0..data_len {
        let level = random_level(params.level_norm_param, rng);
        insert_levels.push(level);
        for l in 0..=level {
            if l >= layers.len() {
                let m_max = if layers.len() == 0 {
                    params.m_max_0
                } else {
                    params.m_max
                };
                let mut layer = Layer::new(m_max);
                layer.entries_cap = 1;
                layers.push(layer).unwrap();
            } else {
                layers[l].entries_cap += 1;
            }
        }
    }
    for layer in layers.iter_mut() {
        layer.allocate_neighbors_memory();
    }
    (layers, insert_levels)
}

// assumes the levels are known, the layers have all been filled with preallocated blocks of memory, etc.
fn s1_insert_element(ctx: &mut InsertCtx<'_>, id: usize, insert_level: usize) {
    // /////////////////////////////////////////////////////////////////////////////
    // SECTION 0: add entries and sub-allocate neighbors lists on all levels 0..=insert_level
    // /////////////////////////////////////////////////////////////////////////////
    let mut lower_level_idx: usize = usize::MAX;
    for l in 0..=insert_level {
        lower_level_idx = ctx.layers[l].add_entry_assuming_allocated_memory(id, lower_level_idx);
    }

    // /////////////////////////////////////////////////////////////////////////////
    // SECTION 1: Search with ef=1 until the insert level is reached.
    // /////////////////////////////////////////////////////////////////////////////
    let q_data = ctx.data.get(id);
    let search_buffers = &mut ctx.search_buffers;
    let mut ep_idx: usize = 0;
    for l in (insert_level + 1..=ctx.top_level).rev() {
        // println!("  SECTION 1, level {l}");
        let layer = &ctx.layers[l];
        search_layer(
            ctx.data,
            &ctx.distance,
            search_buffers,
            layer,
            q_data,
            1,
            &[ep_idx],
        );
        let this_layer_ep_idx = search_buffers.found.as_slice()[0].1;
        // let this_layer_ep_idx = search_layer_ef_1(
        //     ctx.data,
        //     &ctx.distance,
        //     &mut search_buffers.visited,
        //     layer,
        //     q_data,
        //     ep_idx,
        // );
        ep_idx = layer.entries[this_layer_ep_idx].lower_level_idx;
    }

    // /////////////////////////////////////////////////////////////////////////////
    // SECTION 2: now we have the ep_idx at layer insert_level. Search with ef=ef_construction to find neighbors for the new point here
    // /////////////////////////////////////////////////////////////////////////////
    let selected_neighbors = &mut ctx.selected_neighbors;
    let entry_points = &mut ctx.entry_points;
    entry_points.clear();
    entry_points.push(ep_idx);
    for l in (0..=ctx.top_level.min(insert_level)).rev() {
        let layer = &mut ctx.layers[l];
        if layer.entries.len() == 1 {
            assert!(l == 0); // could come here, if top_level should actually be -1 (not possible because of usize, then no other nodes exist at this level and we can break out)
            break;
        }
        search_layer(
            ctx.data,
            &ctx.distance,
            search_buffers,
            layer,
            q_data,
            ctx.params.ef_construction,
            &entry_points,
        );
        // the next entry points are the found elements. Lower the idx of each of them for search in the next layer:
        if l != 0 {
            entry_points.clear();
            for f in search_buffers.found.iter() {
                let f_idx_this_level = f.1;
                let f_idx_lower_level = layer.entries[f_idx_this_level].lower_level_idx;
                entry_points.push(f_idx_lower_level);
            }
        }
        // select the m_max neighbors from found that should be used for bidirectional connections. (because m_max << ef usually).
        // Attention: select_neighbors mutates the scratch space of search_buffers.found! (for efficiency reasons)
        select_neighbors(
            &mut search_buffers.found,
            selected_neighbors,
            ctx.params.m_max,
        ); // Note: m_max better than m_max_0 here for recall even on layer 0!
        let q_idx = layer.entries.len() - 1; // q is the last entry in the layer

        // add bidirectional connections for each neighbor:
        for &DistAnd(nei_dist_to_q, nei_idx) in selected_neighbors.iter() {
            assert!(nei_idx != q_idx);
            layer.entries[nei_idx]
                .neighbors
                .insert_if_better(DistAnd(nei_dist_to_q, q_idx));
            layer.entries[q_idx]
                .neighbors
                .insert_asserted(DistAnd(nei_dist_to_q, nei_idx)); // should always have space.
        }
    }

    if ctx.top_level < insert_level {
        ctx.top_level = insert_level;
    }
}

// /////////////////////////////////////////////////////////////////////////////
// SECTION: Insert Strategy 2: Predetermine levels and insert all elements with empty neighbors lists first.
// This establishes links to lower lists already. Afterwards the insertion does not need to add layer entries anymore.
// -> Good step towards more parallelism without locks on layers.
// /////////////////////////////////////////////////////////////////////////////

fn s2_construct_hnsw(data: Arc<dyn DatasetT>, params: HnswParams, seed: u64) -> SliceHnsw {
    assert!(params.m_max_0 >= params.m_max);
    let start_time = Instant::now();
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let (mut layers, insert_positions) =
        s2_make_layers_with_empty_neighbors(data.len(), params, &mut rng);
    assert_eq!(insert_positions.len(), data.len());
    let mut ctx = InsertCtx {
        distance: DistanceTracker::new(params.distance),
        params,
        data: &*data,
        layers: &mut layers,
        search_buffers: SearchBuffers::new(),
        selected_neighbors: vec![],
        entry_points: vec![],
        top_level: 0,
    };

    for id in 1..data.len() {
        let pos = insert_positions[id];
        s2_insert_element(&mut ctx, id, pos.level, pos.idx_in_layer);
    }
    assert_eq!(ctx.top_level + 1, ctx.layers.len());

    let build_stats = Stats {
        num_distance_calculations: ctx.distance.num_calculations(),
        duration: start_time.elapsed(),
    };

    SliceHnsw {
        data,
        layers,
        params,
        build_stats,
    }
}

#[derive(Debug, Clone, Copy)]
struct InsertPosition {
    level: usize,
    idx_in_layer: usize,
}

fn s2_make_layers_with_empty_neighbors(
    data_len: usize,
    params: HnswParams,
    rng: &mut ChaCha20Rng,
) -> (heapless::Vec<Layer, MAX_LAYERS>, Vec<InsertPosition>) {
    let mut layers: heapless::Vec<Layer, MAX_LAYERS> = heapless::Vec::new();
    for l in 0..MAX_LAYERS {
        let m_max = if l == 0 { params.m_max_0 } else { params.m_max };
        layers.push(Layer::new(m_max)).unwrap();
    }

    let mut insert_positions: Vec<InsertPosition> = Vec::with_capacity(data_len);
    for id in 0..data_len {
        let level = random_level(params.level_norm_param, rng);
        let mut lower_level_idx = usize::MAX;
        for l in 0..=level {
            lower_level_idx = layers[l].add_entry_with_uninit_neighbors(id, lower_level_idx);
        }
        insert_positions.push(InsertPosition {
            level,
            idx_in_layer: lower_level_idx,
        })
    }
    while let Some(layer) = layers.last() {
        // remove unused empty layers
        if layer.entries.is_empty() {
            layers.pop();
        } else {
            break;
        }
    }
    for layer in layers.iter_mut() {
        layer.allocate_neighbors_memory_and_suballocate_neighbors_lists()
    }

    (layers, insert_positions)
}

// Note: Assumes that the first element ever is skipped (no need to do stuff anyway)
// Panics if the first element is not skipped.
// assumes the levels are known, the layers have all been filled with preallocated blocks of memory, etc.
fn s2_insert_element(
    ctx: &mut InsertCtx<'_>,
    id: usize,
    insert_level: usize,
    mut idx_in_layer: usize,
) {
    // walk the graph down to the level where search starts (only relevant if insert_level > top_level)
    for l in (ctx.top_level + 1..=insert_level).rev() {
        idx_in_layer = ctx.layers[l].entries[idx_in_layer].lower_level_idx;
    }

    // /////////////////////////////////////////////////////////////////////////////
    // SECTION 1: Search with ef=1 until the insert level is reached.
    // /////////////////////////////////////////////////////////////////////////////
    let q_data = ctx.data.get(id);
    let search_buffers = &mut ctx.search_buffers;
    let mut ep_idx: usize = 0;
    for l in (insert_level + 1..=ctx.top_level).rev() {
        let layer = &ctx.layers[l];
        search_layer(
            ctx.data,
            &ctx.distance,
            search_buffers,
            layer,
            q_data,
            1,
            &[ep_idx],
        );
        let this_layer_ep_idx = search_buffers.found.as_slice()[0].1;
        ep_idx = layer.entries[this_layer_ep_idx].lower_level_idx;
    }

    // /////////////////////////////////////////////////////////////////////////////
    // SECTION 2: now we have the ep_idx at layer insert_level. Search with ef=ef_construction to find neighbors for the new point here
    // /////////////////////////////////////////////////////////////////////////////
    let selected_neighbors = &mut ctx.selected_neighbors;
    let entry_points = &mut ctx.entry_points;
    entry_points.clear();
    entry_points.push(ep_idx);
    for l in (0..=ctx.top_level.min(insert_level)).rev() {
        let layer = &mut ctx.layers[l];
        search_layer(
            ctx.data,
            &ctx.distance,
            search_buffers,
            layer,
            q_data,
            ctx.params.ef_construction,
            &entry_points,
        );
        // the next entry points are the found elements. Lower the idx of each of them for search in the next layer:
        if l != 0 {
            entry_points.clear();
            for f in search_buffers.found.iter() {
                let f_idx_this_level = f.1;
                let f_idx_lower_level = layer.entries[f_idx_this_level].lower_level_idx;
                entry_points.push(f_idx_lower_level);
            }
        }

        // select the m_max neighbors from found that should be used for bidirectional connections. (because m_max << ef usually).
        // Attention: select_neighbors mutates the scratch space of search_buffers.found! (for efficiency reasons)
        select_neighbors(
            &mut search_buffers.found,
            selected_neighbors,
            ctx.params.m_max,
        ); // TODO! figure out why params.m_max is better than layer.m_max for recall! (ignoring m_max_0)

        // add bidirectional connections for each neighbor:
        for &DistAnd(nei_dist_to_q, nei_idx) in selected_neighbors.iter() {
            if nei_idx == idx_in_layer {
                println!("{nei_idx} != {idx_in_layer}, {selected_neighbors:?}")
            }
            assert!(nei_idx != idx_in_layer);
            layer.entries[nei_idx]
                .neighbors
                .insert_if_better(DistAnd(nei_dist_to_q, idx_in_layer));
            layer.entries[idx_in_layer]
                .neighbors
                .insert_asserted(DistAnd(nei_dist_to_q, nei_idx)); // should always have space.
        }
        idx_in_layer = layer.entries[idx_in_layer].lower_level_idx;
    }

    if ctx.top_level < insert_level {
        ctx.top_level = insert_level;
    }
}

// /////////////////////////////////////////////////////////////////////////////
// SECTION: helper structs and functions
// /////////////////////////////////////////////////////////////////////////////

fn select_neighbors(
    found: &mut BinaryHeap<DistAnd<usize>>,
    selected_neighbors: &mut Vec<DistAnd<usize>>,
    m_max: usize,
) {
    while found.len() > m_max {
        found.pop();
    }
    selected_neighbors.clear();
    selected_neighbors.extend(found.drain());
}

fn select_neighbors_heuristic(
    q_data: &[f32],
    data: &dyn DatasetT,
    distance: &DistanceTracker,
    layer: &Layer,
    found: &mut BinaryHeap<DistAnd<usize>>, // the candidates, a max-heap where the root is the largest-dist element.
    w: &mut BinaryHeap<Reverse<DistAnd<usize>>>, // working queue (???)
    k: usize,                               // number of neighbors to put into out
    selected_neighbors: &mut Vec<DistAnd<usize>>,
) {
    selected_neighbors.clear();
    w.clear();
    for c in found.iter() {
        w.push(Reverse(*c));
    }

    let extend_candidates = true;
    let keep_pruned_candidates = false;

    if extend_candidates {
        for c in found.iter() {
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

    while w.len() > 0 && selected_neighbors.len() < k {
        let e = w.pop().unwrap().0;
        // if e closer to q than any element from R:
        if selected_neighbors.iter().any(|o| e.0 < o.dist()) {
            let e_entry = &layer.entries[e.1];
            selected_neighbors.push(e)
        } else if keep_pruned_candidates {
            wd.push(Reverse(e))
        }
    }

    if keep_pruned_candidates {
        while wd.len() > 0 && selected_neighbors.len() < k {
            let e = wd.pop().unwrap().0;
            selected_neighbors.push(DistAnd(e.0, e.1))
        }
    }
}

/// puts the ef closest elements to q_data into `buffers.found`. (or less than ef if less found).
fn search_layer(
    data: &dyn DatasetT,
    distance: &DistanceTracker,
    buffers: &mut SearchBuffers,
    layer: &Layer,
    q_data: &[f32],
    ef: usize,
    ep: &[usize],
) {
    buffers.clear();
    for &ep_idx in ep {
        let ep_id = layer.entries[ep_idx].id;
        let ep_data = data.get(ep_id);
        let ep_dist = distance.distance(ep_data, q_data);
        buffers.visited.insert(ep_idx);
        buffers.found.push(DistAnd(ep_dist, ep_idx));
        buffers.frontier.push(Reverse(DistAnd(ep_dist, ep_idx)));
    }

    while buffers.frontier.len() > 0 {
        let DistAnd(c_dist, c_idx) = buffers.frontier.pop().unwrap().0;
        let worst_dist_found = buffers.found.peek().unwrap().0;
        if c_dist > worst_dist_found {
            break;
        };
        for nei in layer.entries[c_idx].neighbors.iter() {
            let nei_idx = nei.1;
            if buffers.visited.insert(nei_idx) {
                // only jumps here if was not visited before (newly inserted -> true)
                let nei_id = layer.entries[nei_idx].id;
                let nei_data = data.get(nei_id);
                let nei_dist_to_q = distance.distance(nei_data, q_data);

                if buffers.found.len() < ef {
                    // always insert if found still has space:
                    buffers
                        .frontier
                        .push(Reverse(DistAnd(nei_dist_to_q, nei_idx)));
                    buffers.found.push(DistAnd(nei_dist_to_q, nei_idx));
                } else {
                    // otherwise only insert, if it is better than the worst found element:
                    let mut worst_found = buffers.found.peek_mut().unwrap();
                    if nei_dist_to_q < worst_found.dist() {
                        buffers
                            .frontier
                            .push(Reverse(DistAnd(nei_dist_to_q, nei_idx)));
                        *worst_found = DistAnd(nei_dist_to_q, nei_idx)
                    }
                }
            }
        }
    }
}

/// pefrorms a greedy search from a single entry point and returns the idx in layer that is closest to q
fn search_layer_ef_1(
    data: &dyn DatasetT,
    distance: &DistanceTracker,
    visited: &mut HashSet<usize>,
    layer: &Layer,
    q_data: &[f32],
    ep_idx: usize,
) -> usize {
    visited.clear();

    let mut best_entry = &layer.entries[ep_idx];
    let ep_id = best_entry.id;
    let ep_data = data.get(ep_id);
    let ep_dist = distance.distance(ep_data, q_data);
    let mut best = DistAnd(ep_dist, ep_idx);

    visited.insert(ep_idx);
    loop {
        let mut updated_best = false;
        for &DistAnd(_, nei_idx) in best_entry.neighbors.iter() {
            if visited.insert(nei_idx) {
                let nei_entry = &layer.entries[nei_idx];
                let nei_data = data.get(nei_entry.id);
                let nei_dist_to_q = distance.distance(nei_data, q_data);
                if nei_dist_to_q < best.dist() {
                    best = DistAnd(nei_dist_to_q, nei_idx);
                    best_entry = nei_entry;
                    updated_best = true;
                }
            }
        }
        if !updated_best {
            return best.1;
        }
    }
}

struct InsertCtx<'a> {
    params: HnswParams,
    distance: DistanceTracker,
    data: &'a dyn DatasetT,
    layers: &'a mut heapless::Vec<Layer, MAX_LAYERS>,
    search_buffers: SearchBuffers, // scratch space for greedy search in a layer
    entry_points: Vec<usize>,
    /// scratch space for idx of entry points for search in a specific layer
    selected_neighbors: Vec<DistAnd<usize>>,
    /// scratch space for idx of selected neighbors in a specific layer
    top_level: usize,
}

/// Note: all fields refer to indices in layer, not data ids!
#[derive(Debug, Clone)]
struct SearchBuffers {
    visited: HashSet<usize>,
    frontier: BinaryHeap<Reverse<DistAnd<usize>>>, // root is min dist element.
    found: BinaryHeap<DistAnd<usize>>,             // root is max dist element.
}

impl SearchBuffers {
    pub fn new() -> Self {
        SearchBuffers {
            visited: HashSet::new(),
            frontier: BinaryHeap::new(),
            found: BinaryHeap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.visited.clear();
        self.frontier.clear();
        self.found.clear();
    }
}

#[inline]
pub fn random_level(level_norm_param: f32, rng: &mut ChaCha20Rng) -> usize {
    let f = rng.gen::<f32>();
    let level = (-f.ln() * level_norm_param).floor() as usize;
    level.min(MAX_LAYERS - 1)
}

// /////////////////////////////////////////////////////////////////////////////
// SECTION: Exposed functions for VP-Tree to HNSW Transition
// /////////////////////////////////////////////////////////////////////////////

/// Note: the elements will get inserted at random heights and point to lower level indices.
/// Memory for the neighbors will also be allocated sufficiently, but all neighbors lists will be fresh and empty.
pub fn create_hnsw_layers_with_empty_neighbors(
    data_len: usize,
    m_max: usize,
    m_max_0: usize,
    level_norm_param: f32,
    rng: &mut ChaCha20Rng,
) -> heapless::Vec<Layer, MAX_LAYERS> {
    let mut layers: heapless::Vec<Layer, MAX_LAYERS> = Default::default();

    // exit early for the special case that only 1 layer is needed (skip all the random level selection stuff):
    if level_norm_param == 0.0 {
        let mut layer = Layer::new(m_max_0);
        layer.entries_cap = data_len;
        layer.allocate_neighbors_memory();
        for id in 0..data_len {
            layer.add_entry_assuming_allocated_memory(id, usize::MAX);
        }
        layers.push(layer).unwrap();
        return layers;
    }

    // create as much layers as possible, such that no checks are needed in the hot loop below:
    for level in 0..MAX_LAYERS {
        let m_max = if level == 0 { m_max_0 } else { m_max };
        layers.push(Layer::new(m_max)).unwrap();
    }
    // determine a random level for each data point and insert it with uninitialized neighbors into the hnsw:
    for id in 0..data_len {
        let level = random_level(level_norm_param, rng);
        let mut lower_level_idx = usize::MAX;
        for l in 0..=level {
            lower_level_idx = layers[l].add_entry_with_uninit_neighbors(id, lower_level_idx);
        }
    }
    // remove empty layers that were not used:
    for level in (0..MAX_LAYERS).rev() {
        if layers[level].entries.len() == 0 {
            layers.pop();
        } else {
            break;
        }
    }
    // now all layers (below as well) should have at least one entry. Allocate all the neighbors lists per layer.
    for layer in layers.iter_mut() {
        assert!(layer.entries.len() > 0);
        layer.allocate_neighbors_memory_and_suballocate_neighbors_lists();
    }

    layers
}
