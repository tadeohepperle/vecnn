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
    utils::SliceBinaryHeap,
    vp_tree::Stats,
};

const MAX_LAYERS: usize = 30;
pub struct Hnsw2 {
    pub data: Arc<dyn DatasetT>,
    layers: heapless::Vec<Layer, MAX_LAYERS>,
    pub params: HnswParams,
    pub build_stats: Stats,
}

impl Hnsw2 {
    pub fn new(data: Arc<dyn DatasetT>, params: HnswParams) -> Self {
        construct_hnsw(data, params)
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

#[derive(Debug)]
pub struct Layer {
    allocation_ptr: NonNull<Neighbor>, // ptr to memory region of size: entries_cap * m_max * size_of<Neighbor>
    entries_cap: usize,                // how many entries is the layer gonna hold?
    m_max: usize,                      // max many neighbors per entry
    entries: Vec<LayerEntry>,
}

impl Layer {
    pub fn new() -> Self {
        // could maybe estimate based on layer number how many elements make it up to here and preallocate entries Vec with okay cap.
        Layer {
            allocation_ptr: NonNull::dangling(),
            entries_cap: 0,
            m_max: 0,
            entries: vec![],
        }
    }

    fn allocation_layout(&self) -> std::alloc::Layout {
        std::alloc::Layout::array::<Neighbor>(self.m_max * self.entries_cap).unwrap()
    }

    /// should be called *AFTER* the layer already contains all of the elements.
    /// After calling this, the number of elements in this layer should not change anymore, so you can start modifying the neighbors of the elements.
    fn allocate_neighbors_memory(&mut self) {
        assert!(self.m_max != 0);
        assert!(self.entries_cap != 0);
        let ptr = unsafe { std::alloc::alloc(self.allocation_layout()) };
        self.allocation_ptr = NonNull::new(ptr as *mut Neighbor).expect("Allocation failed.");
        self.entries = Vec::with_capacity(self.entries_cap);
    }

    /// returns the idx of this entry in `self.entries`
    fn push_entry(&mut self, id: usize, lower_level_idx: usize) -> usize {
        // allocate a new neighbors list from the big slice that this layer owns (which contains enough memory from the start to hold exactly all elements)
        let slice = unsafe {
            std::slice::from_raw_parts_mut(
                self.allocation_ptr
                    .as_ptr()
                    .add(self.entries.len() * self.m_max),
                self.m_max,
            )
        };
        let entry = LayerEntry {
            id,
            lower_level_idx,
            neighbors: Neighbors::new(slice),
        };
        let idx = self.entries.len();
        self.entries.push(entry);
        idx
    }
}

impl Drop for Layer {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(
                self.allocation_ptr.as_ptr() as *mut u8,
                self.allocation_layout(),
            );
        }
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct LayerEntry {
    id: usize,
    lower_level_idx: usize,
    neighbors: Neighbors,
}

pub type Neighbor = DistAnd<usize>;
pub type Neighbors = SliceBinaryHeap<'static, Neighbor>;

fn construct_hnsw(data: Arc<dyn DatasetT>, params: HnswParams) -> Hnsw2 {
    let start_time = Instant::now();
    // /////////////////////////////////////////////////////////////////////////////
    // Step 1: Create all the layers
    // /////////////////////////////////////////////////////////////////////////////
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let data_len = data.len();
    let (mut layers, insert_levels) =
        determine_insert_levels_and_prepare_layers(data_len, &params, &mut rng);
    // we assume that the layers have len 0, and just ignore all the empty layers that already have been preallocated.

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
        insert_element(&mut ctx, id, insert_levels[id]);
    }
    assert_eq!(ctx.top_level + 1, ctx.layers.len());

    let build_stats = Stats {
        num_distance_calculations: ctx.distance.num_calculations(),
        duration: start_time.elapsed(),
    };

    Hnsw2 {
        data,
        layers,
        params,
        build_stats,
    }
}

// assumes the levels are known, the layers have all been filled with preallocated blocks of memory, etc.
fn insert_element(ctx: &mut InsertCtx<'_>, id: usize, insert_level: usize) {
    // println!(
    //     "insert element {id} at {insert_level} (top level: {})",
    //     ctx.top_level
    // );
    // /////////////////////////////////////////////////////////////////////////////
    // SECTION 0: add entries and sub-allocate neighbors lists on all levels 0..=insert_level
    // /////////////////////////////////////////////////////////////////////////////
    let mut lower_level_idx: usize = usize::MAX;
    for l in 0..=insert_level {
        lower_level_idx = ctx.layers[l].push_entry(id, lower_level_idx);
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
        // println!("  SECTION 2, level {l}");
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
        select_neighbors(&mut search_buffers.found, selected_neighbors, layer.m_max);
        // select_neighbors_heuristic(
        //     q_data,
        //     ctx.data,
        //     &ctx.distance,
        //     layer,
        //     &mut search_buffers.found,
        //     &mut search_buffers.frontier,
        //     layer.m_max,
        //     selected_neighbors,
        // );
        let q_idx = layer.entries.len() - 1; // q is the last entry in the layer

        // add bidirectional connections for each neighbor:
        for &DistAnd(nei_dist_to_q, nei_idx) in selected_neighbors.iter() {
            assert!(nei_idx != q_idx);
            layer.entries[nei_idx]
                .neighbors
                .insert_if_better(DistAnd(nei_dist_to_q, q_idx));
            layer.entries[q_idx]
                .neighbors
                .push_asserted(DistAnd(nei_dist_to_q, nei_idx)); // should always have space.
        }
    }

    if ctx.top_level < insert_level {
        ctx.top_level = insert_level;
    }
    // println!("Layers:");
    // for (i, layer) in ctx.layers[0..=ctx.top_level].iter().enumerate().rev() {
    //     println!("    Layer {i}");
    //     for e in layer.entries.iter() {
    //         println!("      {:?}", e);
    //     }
    // }
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

/// returns layers and insert level of each node in the dataset
fn determine_insert_levels_and_prepare_layers(
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
                let mut layer = Layer::new();
                layer.entries_cap = 1;
                layer.m_max = if layers.len() == 0 {
                    params.m_max_0
                } else {
                    params.m_max
                };
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

#[inline]
pub fn random_level(level_norm_param: f32, rng: &mut ChaCha20Rng) -> usize {
    let f = rng.gen::<f32>();
    let level = (-f.ln() * level_norm_param).floor() as usize;
    level.min(MAX_LAYERS - 1)
}
