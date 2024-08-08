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
    utils::{SliceBinaryHeap, SlicesMemory, Stats},
};

pub const MAX_LAYERS: usize = 16;

pub const M_MAX: usize = 20;
pub const M_MAX_0: usize = M_MAX * 2;

pub struct ConstHnsw {
    pub data: Arc<dyn DatasetT>,
    pub bottom_layer: Vec<LayerEntry<M_MAX_0>>,
    pub layers: heapless::Vec<Vec<LayerEntry<M_MAX>>, MAX_LAYERS>,
    pub params: HnswParams,
    pub build_stats: Stats,
}

#[derive(Debug)]
#[repr(C)]
pub struct LayerEntry<const M: usize> {
    pub id: usize,
    pub lower_level_idx: usize,
    pub neighbors: Neighbors<M>,
}

type Neighbors<const M: usize> =
    heapless::BinaryHeap<DistAnd<usize>, heapless::binary_heap::Max, M>;

impl ConstHnsw {
    pub fn new(data: Arc<dyn DatasetT>, params: HnswParams, seed: u64) -> Self {
        construct_hnsw(data, params, seed)
    }

    /// the usize's returned are actual data ids, not layers indices!
    pub fn knn_search(&self, q_data: &[f32], k: usize, ef: usize) -> (Vec<DistAnd<usize>>, Stats) {
        let mut buffers = SearchBuffers::new();
        let distance = DistanceTracker::new(self.params.distance);
        let start_time = Instant::now();

        let mut ep_idx: usize = 0;
        let top_layer = self.layers.len() - 1;
        for l in (1..=top_layer).rev() {
            let layer = &self.layers[l - 1];
            let this_layer_ep_idx = search_layer_ef_1(
                &*self.data,
                &distance,
                &mut buffers.visited,
                layer,
                q_data,
                ep_idx,
            );
            ep_idx = layer[this_layer_ep_idx].lower_level_idx;
        }
        let layer_0 = &self.bottom_layer;
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
            let id = layer_0[layer_0_idx].id;
            results.push(DistAnd(dist, id));
        }

        let stats = Stats {
            duration: start_time.elapsed(),
            num_distance_calculations: distance.num_calculations(),
        };

        (results, stats)
    }
}

fn construct_hnsw(data: Arc<dyn DatasetT>, params: HnswParams, seed: u64) -> ConstHnsw {
    let start_time = Instant::now();
    // /////////////////////////////////////////////////////////////////////////////
    // Step 1: Create all the layers and fill them with empty elements. Store for each id, the position in the hsnw that it is inserted at.
    // /////////////////////////////////////////////////////////////////////////////
    let data_len = data.len();
    let mut bottom_layer: Vec<LayerEntry<M_MAX_0>> = Vec::with_capacity(data_len);
    let mut layers: heapless::Vec<Vec<LayerEntry<M_MAX>>, MAX_LAYERS> = heapless::Vec::new();
    for _ in 1..MAX_LAYERS {
        layers.push(Vec::new()).unwrap();
    }

    let mut insert_positions: Vec<InsertPosition> = Vec::with_capacity(data.len());

    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut highest_level = 0;
    for id in 0..data.len() {
        let level = random_level(params.level_norm_param, &mut rng);

        let mut lower_level_idx = bottom_layer.len();
        bottom_layer.push(LayerEntry {
            id,
            lower_level_idx: usize::MAX,
            neighbors: Default::default(),
        });

        for i in 1..=level {
            let layer = &mut layers[i - 1];
            let entry = LayerEntry {
                id,
                lower_level_idx,
                neighbors: Default::default(),
            };
            lower_level_idx = layer.len();
            layer.push(entry);
        }

        let level = level as u8;
        insert_positions.push(InsertPosition {
            level,
            highest_level,
            idx_in_layer: lower_level_idx as u32,
        });
        if level > highest_level {
            highest_level = level
        }
    }
    while layers.last().is_some_and(|e| e.is_empty()) {
        layers.pop(); // remove unused empty layers
    }

    // Note: preinserting all the elements without doing any actual work lets us
    // insert them now in a mainly lockfree way: locks can now only occur if two threads want to read or write the same neighbors list
    // otherwise the structure remains completely static.

    // /////////////////////////////////////////////////////////////////////////////
    // SECTION: Insertion:
    // /////////////////////////////////////////////////////////////////////////////

    let mut ctx = InsertCtx {
        distance: DistanceTracker::new(params.distance),
        params,
        data: &*data,
        search_buffers: SearchBuffers::new(),
        selected_neighbors: vec![],
        entry_points: vec![],
        bottom_layer: &mut bottom_layer,
        layers: &mut layers,
    };
    for id in 0..data_len {
        insert_element(&mut ctx, id, insert_positions[id]);
    }

    let num_distance_calculations = ctx.distance.num_calculations();
    drop(ctx);
    let build_stats = Stats {
        num_distance_calculations,
        duration: start_time.elapsed(),
    };

    ConstHnsw {
        data,
        bottom_layer,
        layers,
        params,
        build_stats,
    }
}

#[derive(Debug, Clone, Copy)]
struct InsertPosition {
    level: u8,         // pack for lower memory footprint
    highest_level: u8, // the highest level BEFORE this point was inserted -> informs where to start the search (at first element of this layer).
    idx_in_layer: u32,
}

// assumes that the element is already present in HNSW as empty preallocated stub.
fn insert_element(ctx: &mut InsertCtx<'_>, id: usize, insert_position: InsertPosition) {
    let insert_level = insert_position.level as usize;
    let highest_level = insert_position.highest_level as usize;
    let idx_in_layer = insert_position.idx_in_layer as usize;
    if idx_in_layer == 0 && highest_level == 0 {
        // first element ever inserted, nothing to do.
        return;
    }

    // /////////////////////////////////////////////////////////////////////////////
    // SECTION 1: Search with ef=1 until the insert level is reached.
    // /////////////////////////////////////////////////////////////////////////////
    let q_data = ctx.data.get(id);
    let search_buffers = &mut ctx.search_buffers;
    let mut ep_idx: usize = 0;
    for l in (insert_level + 1..=highest_level).rev() {
        let layer = &ctx.layers[l - 1];
        let this_layer_ep_idx = search_layer_ef_1(
            ctx.data,
            &ctx.distance,
            &mut search_buffers.visited,
            layer,
            q_data,
            ep_idx,
        );
        ep_idx = layer[this_layer_ep_idx].lower_level_idx;
    }

    // /////////////////////////////////////////////////////////////////////////////
    // SECTION 2: connect neighbors on every layer except bottom layer
    // /////////////////////////////////////////////////////////////////////////////
    let selected_neighbors = &mut ctx.selected_neighbors;
    let entry_points = &mut ctx.entry_points;
    entry_points.clear();
    entry_points.push(ep_idx);
    let mut q_idx = idx_in_layer;

    let start_search_level = highest_level.min(insert_level);
    // go down the tree:
    for l in (start_search_level + 1..=insert_level).rev() {
        let layer = &mut ctx.layers[l - 1];
        q_idx = layer[q_idx].lower_level_idx;
    }

    for l in (1..=start_search_level).rev() {
        let layer = &mut ctx.layers[l - 1];

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
        entry_points.clear();
        for f in search_buffers.found.iter() {
            let f_idx_this_level = f.1;
            let f_idx_lower_level = layer[f_idx_this_level].lower_level_idx;
            entry_points.push(f_idx_lower_level);
        }

        // select the m_max neighbors from found that should be used for bidirectional connections. (because m_max << ef usually).
        // Attention: select_neighbors mutates the scratch space of search_buffers.found! (for efficiency reasons)
        select_neighbors(&mut search_buffers.found, selected_neighbors, M_MAX);

        // add bidirectional connections for each neighbor:
        for &DistAnd(nei_dist_to_q, nei_idx) in selected_neighbors.iter() {
            assert!(nei_idx != q_idx);
            neighbors_insert_if_better(
                &mut layer[nei_idx].neighbors,
                DistAnd(nei_dist_to_q, q_idx),
            );
            neighbors_insert_asserted(&mut layer[q_idx].neighbors, DistAnd(nei_dist_to_q, nei_idx));
        }
        q_idx = layer[q_idx].lower_level_idx;
    }

    // /////////////////////////////////////////////////////////////////////////////
    // SECTION 3: insert on bottom layer
    // /////////////////////////////////////////////////////////////////////////////
    let layer = &mut *ctx.bottom_layer;

    search_layer(
        ctx.data,
        &ctx.distance,
        search_buffers,
        layer,
        q_data,
        ctx.params.ef_construction,
        &entry_points,
    );

    // interesting observation: M_MAX -> better recall
    select_neighbors(&mut search_buffers.found, selected_neighbors, M_MAX); // TODO: explain why M_MAX instead of M_MAX_0 leads to better recall here!!
    for &DistAnd(nei_dist_to_q, nei_idx) in selected_neighbors.iter() {
        assert!(nei_idx != q_idx);
        neighbors_insert_if_better(&mut layer[nei_idx].neighbors, DistAnd(nei_dist_to_q, q_idx));
        neighbors_insert_asserted(&mut layer[q_idx].neighbors, DistAnd(nei_dist_to_q, nei_idx));
    }
}

#[inline(always)]
fn neighbors_insert_if_better<const M: usize>(neighbors: &mut Neighbors<M>, item: DistAnd<usize>) {
    if neighbors.len() < neighbors.capacity() {
        neighbors.push(item).unwrap();
    } else {
        let mut worst = neighbors.peek_mut().unwrap();
        if item < *worst {
            *worst = item;
        }
    }
}

#[inline(always)]
fn neighbors_insert_asserted<const M: usize>(neighbors: &mut Neighbors<M>, item: DistAnd<usize>) {
    neighbors.push(item).unwrap();
}

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

/// puts the ef closest elements to q_data into `buffers.found`. (or less than ef if less found).
fn search_layer<const M: usize>(
    data: &dyn DatasetT,
    distance: &DistanceTracker,
    buffers: &mut SearchBuffers,
    layer: &[LayerEntry<M>],
    q_data: &[f32],
    ef: usize,
    ep: &[usize],
) {
    buffers.clear();
    for &ep_idx in ep {
        let ep_id = layer[ep_idx].id;
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
        for nei in layer[c_idx].neighbors.iter() {
            let nei_idx = nei.1;
            if buffers.visited.insert(nei_idx) {
                // only jumps here if was not visited before (newly inserted -> true)
                let nei_id = layer[nei_idx].id;
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
fn search_layer_ef_1<const M: usize>(
    data: &dyn DatasetT,
    distance: &DistanceTracker,
    visited: &mut HashSet<usize>,
    layer: &[LayerEntry<M>],
    q_data: &[f32],
    ep_idx: usize,
) -> usize {
    visited.clear();

    let mut best_entry = &layer[ep_idx];
    let ep_id = best_entry.id;
    let ep_data = data.get(ep_id);
    let ep_dist = distance.distance(ep_data, q_data);
    let mut best = DistAnd(ep_dist, ep_idx);

    visited.insert(ep_idx);
    loop {
        let mut updated_best = false;
        for &DistAnd(_, nei_idx) in best_entry.neighbors.iter() {
            if visited.insert(nei_idx) {
                let nei_entry = &layer[nei_idx];
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
    bottom_layer: &'a mut Vec<LayerEntry<M_MAX_0>>,
    layers: &'a mut heapless::Vec<Vec<LayerEntry<M_MAX>>, MAX_LAYERS>,
    search_buffers: SearchBuffers, // scratch space for greedy search in a layer
    entry_points: Vec<usize>,
    /// scratch space for idx of entry points for search in a specific layer
    selected_neighbors: Vec<DistAnd<usize>>,
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
