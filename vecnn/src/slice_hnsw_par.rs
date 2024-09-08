use std::{
    cell::UnsafeCell,
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
    mem::MaybeUninit,
    ptr::NonNull,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, RwLock,
    },
    time::Instant,
    usize,
};

use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::{rand_core::le, ChaCha20Rng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    dataset::DatasetT,
    distance::DistanceTracker,
    hnsw::{DistAnd, HnswParams},
    utils::{extend_lifetime, SliceBinaryHeap, SlicesMemory, Stats},
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
        construct_hnsw(data, params, seed, InsertMode::Rayon)
    }

    pub fn new_with_thread_pool(data: Arc<dyn DatasetT>, params: HnswParams, seed: u64) -> Self {
        construct_hnsw(data, params, seed, InsertMode::ThreadPool)
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
            // search_layer(
            //     &*self.data,
            //     &distance,
            //     &mut buffers,
            //     layer,
            //     q_data,
            //     1,
            //     &[ep_idx],
            // );
            // let this_layer_ep_idx = buffers.found.as_slice()[0].1;
            let this_layer_ep_idx = search_layer_ef_1(
                &*self.data,
                &distance,
                &mut buffers.visited,
                layer,
                q_data,
                ep_idx,
            );
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

    /// This exists, because it sucks to to knn search when you have to call read() on the RwLocks all the time.
    ///
    /// And because Rust does not have a defined memory layout for RwLock (no repr(C) on it) AND no compile time type information telling us e.g. byte offsets of fields,
    /// we have no chance of accessing the data inside the RwLocks with unsafe after building. Just sad.
    ///
    /// An alternative would be to ditch the RwLocks wrapping the neighbors lists and instead maintain a symbolic Vec<RwLock<()>>
    /// for each layer during the build process with an entry for each neighbors list. Can be thrown away once the HNSW is built.
    pub fn convert_to_slice_hnsw_without_locks(&self) -> super::slice_hnsw::SliceHnsw {
        let mut layers: heapless::Vec<super::slice_hnsw::Layer, MAX_LAYERS> = Default::default();

        for layer in self.layers.iter() {
            let mut new_layer = super::slice_hnsw::Layer::new(layer.m_max);
            new_layer.entries_cap = layer.entries.len();
            new_layer.allocate_neighbors_memory();
            for e in layer.entries.iter() {
                let idx = new_layer.add_entry_assuming_allocated_memory(e.id, e.lower_level_idx);
                let new_entry = &mut new_layer.entries[idx];
                let e_neighbors = e.neighbors.read().unwrap();
                unsafe {
                    e_neighbors.clone_into(&mut new_entry.neighbors);
                }
            }
            layers.push(new_layer).unwrap();
        }

        return super::slice_hnsw::SliceHnsw {
            data: self.data.clone(),
            layers,
            params: self.params,
            build_stats: self.build_stats,
        };
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
    pub entry_point_idx: AtomicUsize,
}

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
            entry_point_idx: AtomicUsize::new(0),
        }
    }

    /// should be called *AFTER* the layer already contains all of the elements.
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
            neighbors: RwLock::new(SliceBinaryHeap::new(slice)),
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
            neighbors: unsafe { MaybeUninit::uninit().assume_init() },
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
            entry.neighbors = RwLock::new(SliceBinaryHeap::new(slice))
        }
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct LayerEntry {
    pub id: usize,
    pub lower_level_idx: usize,
    pub neighbors: Neighbors,
}

pub type Neighbor = DistAnd<usize>;
pub type Neighbors = RwLock<SliceBinaryHeap<'static, Neighbor>>;

#[derive(Debug, Clone, Copy)]
struct InsertPosition {
    level: usize,
    idx_in_layer: usize,
}

pub enum InsertMode {
    Sequential,
    Rayon,
    ThreadPool,
}

fn construct_hnsw(
    data: Arc<dyn DatasetT>,
    params: HnswParams,
    seed: u64,
    insert_mode: InsertMode,
) -> SliceHnsw {
    let start_time = Instant::now();
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let (layers, insert_positions) = make_layers_with_empty_neighbors(data.len(), params, &mut rng);
    assert_eq!(insert_positions.len(), data.len());

    // this trick lets us not insert the 0th data point (would have empty neighbors anyway),
    // instead we just detect how tall the 0 goes and use it as an entry point for searches up to this level.
    let mut top_level: usize = 0;
    assert!(layers[0].entries[0].id == 0);
    for layer in layers.iter().skip(1) {
        if layer.entries[0].id == 0 {
            top_level += 1;
        }
    }
    let ctx = InsertCtx {
        distance: DistanceTracker::new(params.distance),
        params,
        data: &*data,
        layers: &layers,
        top_level: AtomicUsize::new(top_level),
    };

    // (1..data.len()).into_par_iter().for_each(|id| {
    //     insert_element(&ctx, id, insert_positions[id]);
    // });

    match insert_mode {
        InsertMode::Sequential => {
            // just insert sequentially, still uses all the RwLocks but they should never be contended
            for id in 1..data.len() {
                insert_element(&ctx, id, insert_positions[id]);
            }
        }
        InsertMode::Rayon => {
            // Work stealing approach, but seems to have quite some overhead in the flamegraph??
            (1..data.len()).into_par_iter().for_each(|id| {
                insert_element(&ctx, id, insert_positions[id]);
            });
        }
        InsertMode::ThreadPool => {
            // Good old thread pool with equal work for every thread.
            let data_len = data.len();
            let mut num_threads: usize = std::thread::available_parallelism().unwrap().into();
            const MAX_THREADS: usize = 64;
            if num_threads > MAX_THREADS {
                println!(
                    "Warning! Your machine has {num_threads} cores, but only {MAX_THREADS} will be used."
                );
                num_threads = MAX_THREADS;
            }
            let elements_per_thread = data_len / num_threads;
            std::thread::scope(|scope| {
                let mut threads: heapless::Vec<std::thread::ScopedJoinHandle<'_, ()>, MAX_THREADS> =
                    Default::default();
                let ctx = &ctx;
                let insert_positions = &insert_positions;
                for t_idx in 0..num_threads - 1 {
                    let handle = scope.spawn(move || {
                        let start_idx = t_idx * elements_per_thread;
                        let end_idx = (t_idx + 1) * elements_per_thread;
                        for id in start_idx..end_idx {
                            insert_element(ctx, id, insert_positions[id]);
                        }
                    });
                    threads.push(handle).unwrap();
                }

                // run some work on this very thread as well:
                let start_idx = (num_threads - 1) * elements_per_thread;
                let end_idx = data_len;
                for id in start_idx..end_idx {
                    insert_element(ctx, id, insert_positions[id]);
                }

                // wait for all threads to finish:
                for t in threads.into_iter() {
                    t.join().unwrap();
                }
            });
        }
    }

    // let mut random_order: Vec<usize> = (1..data.len()).collect();
    // random_order.shuffle(&mut rng);
    // // println!("Layers ({}):", layers.len());
    // // for (i, layer) in layers.iter().enumerate() {
    // //     println!("layer {i}: {:?}", layer.entries)
    // // }
    // // println!("random_order: {random_order:?}");
    // random_order.into_iter().for_each(|id| {
    //     let pos = insert_positions[id];
    //     insert_element(&ctx, id, pos.level, pos.idx_in_layer);
    // });

    assert_eq!(ctx.top_level.load(Ordering::SeqCst) + 1, ctx.layers.len());

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

fn make_layers_with_empty_neighbors(
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
fn insert_element(ctx: &InsertCtx<'_>, id: usize, pos: InsertPosition) {
    let insert_level: usize = pos.level;
    let idx_at_insert_level = pos.idx_in_layer;

    // sets the top_level to the insert_level is insert_level is > top_level. The value returned is the previous level, so might be higher or lower than insert_level
    let top_level = ctx.top_level.load(Ordering::Acquire);
    let mut ep_idx: usize = ctx.layers[top_level]
        .entry_point_idx
        .load(Ordering::Acquire);

    let mut idx_in_layer = idx_at_insert_level;
    // walk the graph down to the level where search starts (only relevant if insert_level > top_level)
    for l in (top_level + 1..=insert_level).rev() {
        let layer = &ctx.layers[l];
        idx_in_layer = layer.entries[idx_in_layer].lower_level_idx;
    }

    // /////////////////////////////////////////////////////////////////////////////
    // SECTION 1: Search with ef=1 until the insert level is reached.
    // /////////////////////////////////////////////////////////////////////////////
    let thread = thread_local_buffers();
    let q_data = ctx.data.get(id);
    for l in (insert_level + 1..=top_level).rev() {
        let layer = &ctx.layers[l];
        search_layer(
            ctx.data,
            &ctx.distance,
            &mut thread.search_buffers,
            layer,
            q_data,
            1,
            &[ep_idx],
        );
        let this_layer_ep_idx = thread.search_buffers.found.as_slice()[0].1;
        ep_idx = layer.entries[this_layer_ep_idx].lower_level_idx;
    }

    // /////////////////////////////////////////////////////////////////////////////
    // SECTION 2: now we have the ep_idx at layer insert_level. Search with ef=ef_construction to find neighbors for the new point here
    // /////////////////////////////////////////////////////////////////////////////
    let selected_neighbors = &mut thread.selected_neighbors;
    let entry_points = &mut thread.entry_points;
    entry_points.clear();
    entry_points.push(ep_idx);
    for l in (0..=top_level.min(insert_level)).rev() {
        let layer = &ctx.layers[l];
        search_layer(
            ctx.data,
            &ctx.distance,
            &mut thread.search_buffers,
            layer,
            q_data,
            ctx.params.ef_construction,
            &entry_points,
        );
        // the next entry points are the found elements. Lower the idx of each of them for search in the next layer:
        if l != 0 {
            entry_points.clear();
            for f in thread.search_buffers.found.iter() {
                let f_idx_this_level = f.1;
                let f_idx_lower_level = layer.entries[f_idx_this_level].lower_level_idx;
                entry_points.push(f_idx_lower_level);
            }
        }

        // select the m_max neighbors from found that should be used for bidirectional connections. (because m_max << ef usually).
        // Attention: select_neighbors mutates the scratch space of search_buffers.found! (for efficiency reasons)
        select_neighbors(
            &mut thread.search_buffers.found,
            selected_neighbors,
            ctx.params.m_max,
        ); // TODO! figure out why params.m_max is better than layer.m_max for recall! (ignoring m_max_0)

        // add bidirectional connections for each neighbor:
        for &DistAnd(nei_dist_to_q, nei_idx) in selected_neighbors.iter() {
            if nei_idx == idx_in_layer {
                // this can happen (very rarely (about 5/20000)), if other threads access the neighbors lists between the `insert_if_better` calls in this loop.
                continue;
            }
            let mut nei_neighbors = layer.entries[nei_idx].neighbors.write().unwrap();
            nei_neighbors.insert_if_better(DistAnd(nei_dist_to_q, idx_in_layer));

            let mut q_neighbors = layer.entries[idx_in_layer].neighbors.write().unwrap();
            q_neighbors.insert_if_better(DistAnd(nei_dist_to_q, nei_idx)); // should always have space. // push asserted!
        }

        idx_in_layer = layer.entries[idx_in_layer].lower_level_idx;
    }

    // this happens at most MAX_LAYERS times across all inserts.
    if insert_level > top_level {
        ctx.top_level.fetch_max(insert_level, Ordering::Release);
        let mut idx_in_layer = idx_at_insert_level;
        for l in (top_level + 1..=insert_level).rev() {
            let layer = &ctx.layers[l];
            // set the entry point of this layer to this element, because we are the first who made it here:
            // this should only happen exactly once (or zero times) per layer, maybe we can use a once_lock instead or something, to be sure.
            ctx.layers[l]
                .entry_point_idx
                .store(idx_in_layer, Ordering::Release);
            idx_in_layer = layer.entries[idx_in_layer].lower_level_idx;
        }
    }
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
        for nei in layer.entries[c_idx].neighbors.read().unwrap().iter() {
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
        for &DistAnd(_, nei_idx) in best_entry.neighbors.read().unwrap().iter() {
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
    layers: &'a heapless::Vec<Layer, MAX_LAYERS>,
    top_level: AtomicUsize,
}

/// if we run the Insertion in multiple threads, we can reuse these buffers stored in TLS, such that we cut down on allocations.
/// Just make sure to clear before using!
struct ThreadLocalBuffers {
    /// scratch space for greedy search in a layer
    search_buffers: SearchBuffers,
    /// scratch space for idx of entry points for search in a specific layer
    entry_points: Vec<usize>,
    /// scratch space for idx of selected neighbors in a specific layer
    selected_neighbors: Vec<DistAnd<usize>>,
}

thread_local! {
    static THREAD_LOCAL_BUFFERS:UnsafeCell<ThreadLocalBuffers> = UnsafeCell::new(ThreadLocalBuffers{ search_buffers: SearchBuffers::new(), entry_points: vec![], selected_neighbors: vec![]});
}

fn thread_local_buffers() -> &'static mut ThreadLocalBuffers {
    THREAD_LOCAL_BUFFERS.with(|e| unsafe { &mut *e.get() })
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
