use ahash::{HashMap, HashMapExt, HashSet};
use core::f32;
use std::{
    cell::UnsafeCell,
    collections::BinaryHeap,
    hash::Hash,
    ops::Range,
    sync::{Arc, Mutex},
    time::Instant,
    usize,
};

use nanoserde::{DeJson, SerJson};
use rand::{seq::SliceRandom, thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::{
    iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSlice,
};

use crate::{
    dataset::DatasetT,
    distance::{l2, Distance, DistanceFn, DistanceTracker},
    hnsw::{
        self, pick_level, DistAnd, Hnsw, HnswParams, Layer, LayerEntry, Neighbors,
        NEIGHBORS_LIST_MAX_LEN,
    },
    if_tracking,
    relative_nn_descent::{RNNConstructionBuffers, RNNGraphParams},
    slice_hnsw::{self, search_layer, search_layer_ef_1, SearchBuffers, SliceHnsw, MAX_LAYERS},
    utils::{
        extend_lifetime, extend_lifetime_mut, sanititze_num_threads, slice_binary_heap_arena,
        BinaryHeapExt, SliceBinaryHeap, SlicesMemory, Stats, YoloCell,
    },
    vp_tree::{
        self, arrange_into_vp_tree_parallel_with_n_candidates,
        arrange_into_vp_tree_with_n_candidates, left, left_with_root, right, DistAndIdT, Node,
        VpTree,
    },
};

#[derive(Debug, Clone, Copy, DeJson, SerJson, PartialEq)]
pub struct EnsembleParams {
    pub n_vp_trees: usize,
    pub max_chunk_size: usize,
    pub same_chunk_m_max: usize,
    pub m_max: usize, // see hnsw params, max number of neighbors at any layer > 0 in hnsw
    pub m_max_0: usize, // see hnsw params, max number of neighbors at layer 0 in hnsw
    pub level_norm: f32, // see hnsw params, controls height
    pub distance: Distance,
    pub strategy: EnsembleStrategy,
    pub n_candidates: usize, // referring to vptree candidates selection. 0 and 1 mean just pick a random point as vantage point
}

/// Describes how WITHIN each chunk elements are connected.
#[derive(Debug, Clone, Copy, DeJson, SerJson, PartialEq)]
pub enum EnsembleStrategy {
    /// Calculates all connections between all elements in chunk and keeps the best `same_chunk_m_max` ones.
    /// These are then inserted into the HNSW (if better than existing connections).
    /// Note: k is same_chunk_m_max
    BruteForceKNN,
    /// builds a small Relative NN descent graph on each chunk.
    RNNDescent {
        o_loops: usize,
        i_loops: usize,
    },
    BruteForceExactRNG,
}

pub fn build_single_layer_hnsw_by_vp_tree_ensemble(
    data: Arc<dyn DatasetT>,
    params: EnsembleParams,
    seed: u64,
) -> SliceHnsw {
    let mut hnsw_layer = slice_hnsw::Layer::new(params.m_max_0);
    hnsw_layer.entries_cap = data.len();
    hnsw_layer.allocate_neighbors_memory();
    for id in 0..data.len() {
        hnsw_layer.add_entry_assuming_allocated_memory(id, usize::MAX);
    }

    let max_chunk_size = params.max_chunk_size;
    let same_chunk_m_max = params.same_chunk_m_max;

    let mut distance = DistanceTracker::new(params.distance);
    let start = Instant::now();
    let mut rng = ChaCha20Rng::seed_from_u64(seed);

    // setup buffers to operate in:
    let chunks = make_chunks(data.len(), max_chunk_size);
    let mut chunk_dst_mat: Vec<f32> = vec![0.0; max_chunk_size * max_chunk_size]; // memory reused for all chunks
    let (_memory, mut vp_tree_neighbors) =
        slice_binary_heap_arena::<DistAnd<usize>>(data.len(), same_chunk_m_max);

    let mut vp_tree: Vec<vp_tree::Node> = Vec::with_capacity(data.len());
    for idx in 0..data.len() {
        vp_tree.push(vp_tree::Node { id: idx, dist: 0.0 });
    }

    let data_get = |e: &vp_tree::Node| data.get(e.id);
    for vp_tree_iteration in 0..params.n_vp_trees {
        // build the vp_tree, chunk it and brute force search the best neighbors per chunk, put them in vp_tree_neighbors list
        arrange_into_vp_tree_with_n_candidates(
            &mut vp_tree,
            &data_get,
            &mut distance,
            &mut rng,
            params.n_candidates,
        );
        for (chunk_i, chunk) in chunks.iter().enumerate() {
            let chunk_size = chunk.range.len();
            let chunk_dst_mat_idx = |i: usize, j: usize| i + j * chunk_size;
            let chunk_nodes = &vp_tree[chunk.range.clone()];

            assert!(chunk_size >= max_chunk_size / 2);
            assert!(chunk_size <= max_chunk_size);

            // calculate distances between all the nodes in this chunk. (clearing chunk_dst_mat not necessary, because everything relevant should be overwritten).
            for i in 0..chunk_size {
                chunk_dst_mat[chunk_dst_mat_idx(i, i)] = 0.0;
                for j in i + 1..chunk_size {
                    let i_data = data.get(chunk_nodes[i].id);
                    let j_data = data.get(chunk_nodes[j].id);
                    let dist = distance.distance(i_data, j_data);
                    chunk_dst_mat[chunk_dst_mat_idx(i, j)] = dist;
                    chunk_dst_mat[chunk_dst_mat_idx(j, i)] = dist;
                }
            }
            // connect each node in the chunk to each other node (except itself)
            for i in 0..chunk_size {
                let i_id = chunk_nodes[i].id;
                for j in 0..chunk_size {
                    if j == i {
                        continue;
                    }
                    let j_id = chunk_nodes[j].id;
                    let dist = chunk_dst_mat[chunk_dst_mat_idx(i, j)];
                    vp_tree_neighbors[i_id].insert_if_better(DistAnd(dist, j_id));
                }

                if_tracking! {
                    Tracking.pt_meta(i_id).chunk_on_level.push(chunk_i);
                }
            }
        }

        // add the connections of this vp-tree into the hsnw:
        for (hnsw_entry, vp_neighbors) in hnsw_layer
            .entries
            .iter_mut()
            .zip(vp_tree_neighbors.iter_mut())
        {
            for v in vp_neighbors.as_slice().iter() {
                if !hnsw_entry.neighbors.iter().any(|e| e.1 == v.1) {
                    hnsw_entry.neighbors.insert_if_better(*v);
                }
            }
            vp_neighbors.clear();
        }
    }

    let mut layers: heapless::Vec<slice_hnsw::Layer, MAX_LAYERS> = Default::default();
    layers.push(hnsw_layer).unwrap(); // single layer
    let build_stats = Stats {
        num_distance_calculations: distance.num_calculations(),
        duration: start.elapsed(),
    };
    SliceHnsw {
        data,
        layers,
        params: HnswParams {
            level_norm_param: params.level_norm,
            ef_construction: 0,
            m_max: params.m_max,
            m_max_0: params.m_max_0,
            distance: params.distance,
        },
        build_stats,
    }
}

pub fn build_hnsw_by_vp_tree_ensemble_multi_layer(
    data: Arc<dyn DatasetT>,
    params: EnsembleParams,
    threaded: bool,
    seed: u64,
) -> SliceHnsw {
    let start_time = Instant::now();
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut layers = super::slice_hnsw::create_hnsw_layers_with_empty_neighbors(
        data.len(),
        params.m_max,
        params.m_max_0,
        params.level_norm,
        &mut rng,
    );
    let distance = DistanceTracker::new(params.distance);
    // do the ensemble approach for each layer:
    for (level, layer) in layers.iter_mut().enumerate() {
        // && false //for disabling it
        if layer.entries.len() <= params.max_chunk_size * 2 {
            // brute force connect neighbors!
            fill_hnsw_layer_range_by_brute_force(&*data, &distance, layer);
        } else {
            if !threaded {
                fill_hnsw_layer_by_vp_tree_ensemble(&*data, &distance, layer, &params, seed);
            } else {
                const USE_CORE_AFFINITY_INSTEAD_OF_RAYON: bool = false;
                if USE_CORE_AFFINITY_INSTEAD_OF_RAYON {
                    fill_hnsw_layer_by_vp_tree_ensemble_threaded_by_chunk_with_core_affinity(
                        &*data, &distance, layer, &mut rng, params,
                    );
                } else {
                    fill_hnsw_layer_by_vp_tree_ensemble_threaded_by_chunk(
                        &*data, &distance, layer, &mut rng, params,
                    );
                }
            }
        }
    }
    let build_stats = Stats {
        duration: start_time.elapsed(),
        num_distance_calculations: distance.num_calculations(),
    };
    SliceHnsw {
        data,
        layers,
        params: HnswParams {
            level_norm_param: params.level_norm,
            ef_construction: 0,
            m_max: params.m_max,
            m_max_0: params.m_max_0,
            distance: params.distance,
        },
        build_stats,
    }
}

fn fill_hnsw_layer_range_by_brute_force(
    data: &dyn DatasetT,
    distance: &DistanceTracker,
    layer: &mut super::slice_hnsw::Layer,
) {
    let n_entries = layer.entries.len();
    let mut dist_mat: Vec<f32> = vec![0.0; n_entries * n_entries];
    let dist_mat_idx = |i: usize, j: usize| i + j * n_entries;
    for i in 0..n_entries {
        for j in i + 1..n_entries {
            let i_data = data.get(layer.entries[i].id);
            let j_data = data.get(layer.entries[j].id);
            let dist = distance.distance(i_data, j_data);
            dist_mat[dist_mat_idx(i, j)] = dist;
            dist_mat[dist_mat_idx(j, i)] = dist;
        }
    }
    for i in 0..n_entries {
        let i_neighbors = &mut layer.entries[i].neighbors;
        for j in 0..n_entries {
            if j == i {
                continue;
            }
            let dist = dist_mat[dist_mat_idx(i, j)];
            i_neighbors.insert_if_better(DistAnd(dist, j));
        }
    }
}

#[derive(Debug, Clone)]
struct VpTreeNode {
    id: u32, // u32 for better align
    idx_in_hnsw_layer: u32,
    dist: f32,
}

impl DistAndIdT for VpTreeNode {
    #[inline(always)]
    fn dist(&self) -> f32 {
        self.dist
    }
    #[inline(always)]
    fn set_dist(&mut self, dist: f32) {
        self.dist = dist
    }
    fn id(&self) -> usize {
        self.id as usize
    }
}

fn fill_hnsw_layer_by_vp_tree_ensemble(
    data: &dyn DatasetT,
    distance: &DistanceTracker,
    layer: &mut super::slice_hnsw::Layer,
    params: &EnsembleParams,
    seed: u64,
) {
    let n_entries = layer.entries.len();
    let mut vp_tree: Vec<VpTreeNode> = Vec::with_capacity(n_entries);
    for (i, entry) in layer.entries.iter().enumerate() {
        vp_tree.push(VpTreeNode {
            id: entry.id as u32,
            idx_in_hnsw_layer: i as u32,
            dist: 0.0,
        });
    }
    let max_chunk_size = params.max_chunk_size;
    let same_chunk_m_max = params.same_chunk_m_max;
    let chunks = make_chunks(n_entries, max_chunk_size);
    let data_get = |e: &VpTreeNode| data.get(e.id as usize);

    let layer_cell = YoloCell::new(std::mem::replace(layer, super::slice_hnsw::Layer::new(1)));
    let unsafe_get_mut_neighbors_in_hnsw_at_idx = |idx_in_hnsw: usize| unsafe {
        &mut layer_cell
            .get_mut()
            .entries
            .get_unchecked_mut(idx_in_hnsw)
            .neighbors
    };

    let mut brute_force_buffers = if params.strategy == EnsembleStrategy::BruteForceKNN {
        EnsembleBruteForceBuffers::new(max_chunk_size, same_chunk_m_max)
    } else {
        EnsembleBruteForceBuffers::new_uninit() // ignored anyway
    };
    let mut rnn_buffers = RNNConstructionBuffers::new();

    // Note: The vp_tree_neighbors are aligned with the nodes in the vp_tree (vp_tree[i] has neighbors stored in vp_tree_neighbors[i]),
    // but the indices stored in the neighbors lists, refer to indices of the hnsw layer!
    for vp_tree_idx in 0..params.n_vp_trees {
        let before = Instant::now();
        let vp_tree_seed = seed + vp_tree_idx as u64;

        #[cfg(feature = "tracking")]
        let mut vp_tree = vp_tree.clone(); // if tracking mode is enabled (e.g. for visualizations, clone the tree, such that all iterations use the same starting tree )

        arrange_into_vp_tree_with_n_candidates(
            &mut vp_tree,
            &data_get,
            distance,
            &mut ChaCha20Rng::seed_from_u64(vp_tree_seed),
            params.n_candidates,
        );

        let between = before.elapsed();
        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            if_tracking!(Tracking.current_chunk = chunk_idx);
            let chunk_nodes = &vp_tree[chunk.range.clone()];
            match params.strategy {
                EnsembleStrategy::BruteForceKNN => {
                    connect_vp_tree_chunk_and_update_neighbors_in_hnsw_brute_force(
                        chunk_nodes,
                        data,
                        distance,
                        &mut brute_force_buffers,
                        &unsafe_get_mut_neighbors_in_hnsw_at_idx,
                        same_chunk_m_max,
                    );
                }
                EnsembleStrategy::RNNDescent {
                    i_loops: inner_loops,
                    o_loops: outer_loops,
                } => {
                    connect_vp_tree_chunk_and_update_neighbors_in_hnsw_rnn_descent(
                        chunk_nodes,
                        data,
                        distance,
                        &mut rnn_buffers,
                        &unsafe_get_mut_neighbors_in_hnsw_at_idx,
                        same_chunk_m_max,
                        inner_loops,
                        outer_loops,
                    );
                }
                EnsembleStrategy::BruteForceExactRNG => {
                    connect_vp_tree_chunk_and_update_neighbors_in_hnsw_exact_rng(
                        chunk_nodes,
                        data,
                        distance,
                        &unsafe_get_mut_neighbors_in_hnsw_at_idx,
                    )
                }
            }
        }
        println!(
            "Non-Threaded Chunks ({}) loop took: {}ms + {}ms",
            chunks.len(),
            between.as_secs_f32() * 1000.0,
            (before.elapsed() - between).as_secs_f32() * 1000.0
        );
    }
    *layer = layer_cell.into_inner();
}

fn fill_hnsw_layer_by_vp_tree_ensemble_threaded_by_vp_tree(
    data: &dyn DatasetT,
    distance: &DistanceTracker,
    layer: &mut super::slice_hnsw::Layer,
    rng: &mut ChaCha20Rng,
    params: EnsembleParams,
) {
    let n_entries = layer.entries.len();
    let max_chunk_size = params.max_chunk_size;
    let same_chunk_m_max = params.same_chunk_m_max;
    let chunks = make_chunks(n_entries, max_chunk_size);
    let data_get = |e: &VpTreeNode| data.get(e.id as usize);

    // Now we officially get the permission to mutate the layer from multiple threads (MUST wrap in UnsafeCell if mutable accesses to this shared reference can happen):
    let layer_cell = YoloCell::new(std::mem::replace(layer, super::slice_hnsw::Layer::new(1)));
    let mut layer_mutexes: Vec<Mutex<()>> = Vec::with_capacity(n_entries);
    for _ in 0..n_entries {
        layer_mutexes.push(Mutex::new(()))
    }

    let seed: u64 = rng.gen();
    // (0..params.n_vp_trees)
    //     .into_par_iter()
    //     .for_each(|vp_tree_idx| {

    for vp_tree_idx in 0..params.n_vp_trees {
        let mut rng = ChaCha20Rng::seed_from_u64(seed + vp_tree_idx as u64);
        let mut vp_tree: Vec<VpTreeNode> = Vec::with_capacity(data.len());
        for (i, entry) in layer_cell.entries.iter().enumerate() {
            vp_tree.push(VpTreeNode {
                id: entry.id as u32,
                idx_in_hnsw_layer: i as u32,
                dist: 0.0,
            });
        }
        arrange_into_vp_tree_with_n_candidates(
            &mut vp_tree,
            &data_get,
            distance,
            &mut rng,
            params.n_candidates,
        );
        let mut tls = EnsembleBruteForceBuffers::new(max_chunk_size, same_chunk_m_max);
        for chunk in chunks.iter() {
            let chunk_size = chunk.range.len();
            let chunk_dst_mat_idx = |i: usize, j: usize| i + j * chunk_size;
            let chunk_nodes = &vp_tree[chunk.range.clone()];
            assert!(chunk_size >= max_chunk_size / 2);
            assert!(chunk_size <= max_chunk_size);

            // calculate distances between all the nodes in this chunk. (clearing chunk_dst_mat not necessary, because everything relevant should be overwritten).
            for i in 0..chunk_size {
                for j in i + 1..chunk_size {
                    let i_data = data.get(chunk_nodes[i].id as usize);
                    let j_data = data.get(chunk_nodes[j].id as usize);
                    let dist = distance.distance(i_data, j_data);
                    tls.chunk_dst_mat[chunk_dst_mat_idx(i, j)] = dist;
                    tls.chunk_dst_mat[chunk_dst_mat_idx(j, i)] = dist;
                }
            }
            // try to connect each node in the chunk to each other node (except itself) (limiting number of neighbors to same_chunk_m_max with the insert_if_better)
            for i in 0..chunk_size {
                // since all chunks are disjunct and each element is only in one chunk, this access should be fine and cannot lead to race conditions.
                let i_neighbors = &mut tls.chunk_neighbors[i];
                for j in 0..chunk_size {
                    if j == i {
                        continue;
                    }
                    let dist = tls.chunk_dst_mat[chunk_dst_mat_idx(i, j)];
                    let j_idx_in_hnsw = chunk_nodes[j].idx_in_hnsw_layer as usize;
                    i_neighbors.insert_if_better(DistAnd(dist, j_idx_in_hnsw));
                }
            }

            // merge the neighbors of the vp-tree in with the hnsw layer, locking neighbors lists that get currently edited:
            for i in 0..chunk_size {
                let i_idx_in_hnsw = chunk_nodes[i].idx_in_hnsw_layer as usize;
                let i_neighbors = &mut tls.chunk_neighbors[i];
                let hnsw_neighbors_lock = layer_mutexes[i_idx_in_hnsw].lock().unwrap();
                let hnsw_neighbors =
                    unsafe { &mut layer_cell.get_mut().entries[i_idx_in_hnsw].neighbors };
                for &DistAnd(dist, idx_in_hnsw) in i_neighbors.iter() {
                    if !hnsw_neighbors.iter().any(|e| e.1 == idx_in_hnsw) {
                        hnsw_neighbors.insert_if_better(DistAnd(dist, idx_in_hnsw));
                    }
                }
                i_neighbors.clear();
                drop(hnsw_neighbors_lock);
            }
        }
    }

    // put the layer back where it was before (move out of the UnsafeCell we used above)
    *layer = layer_cell.into_inner();
}

fn fill_hnsw_layer_by_vp_tree_ensemble_threaded_by_chunk(
    data: &dyn DatasetT,
    distance: &DistanceTracker,
    layer: &mut super::slice_hnsw::Layer,
    rng: &mut ChaCha20Rng,
    params: EnsembleParams,
) {
    let n_entries = layer.entries.len();
    let mut vp_tree: Vec<VpTreeNode> = Vec::with_capacity(data.len());
    for (i, entry) in layer.entries.iter().enumerate() {
        vp_tree.push(VpTreeNode {
            id: entry.id as u32,
            idx_in_hnsw_layer: i as u32,
            dist: 0.0,
        });
    }
    let max_chunk_size = params.max_chunk_size;
    let same_chunk_m_max = params.same_chunk_m_max;
    let chunks = make_chunks(n_entries, max_chunk_size);
    let data_get = |e: &VpTreeNode| data.get(e.id as usize);

    let num_threads: usize = sanititze_num_threads(8);
    const MIN_CHUNK_CHUNK_SIZE: usize = 4; // to not have chunks that are too small
    let chunk_chunk_size = MIN_CHUNK_CHUNK_SIZE.max(chunks.len() / num_threads);
    let chunk_chunks: Vec<&[Chunk]> = chunks.chunks(chunk_chunk_size).collect();

    // Now we officially get the permission to mutate the layer from multiple threads (MUST wrap in UnsafeCell if mutable accesses to this shared reference can happen):
    let layer_cell = YoloCell::new(std::mem::replace(layer, super::slice_hnsw::Layer::new(1)));
    let unsafe_get_mut_neighbors_in_hnsw_at_idx =
        |idx_in_hnsw: usize| unsafe { &mut layer_cell.get_mut().entries[idx_in_hnsw].neighbors };

    for _tree_idx in 0..params.n_vp_trees {
        let before = Instant::now();
        arrange_into_vp_tree_parallel_with_n_candidates(
            &mut vp_tree,
            &data_get,
            distance,
            rng,
            max_chunk_size / 2,
            params.n_candidates,
        );

        let between = before.elapsed();

        chunk_chunks.par_iter().for_each(|chunk_chunk| {
            let start = Instant::now();
            for chunk in chunk_chunk.iter() {
                // Attention: This needs to be ABSOLUTELY sure, that all threads operate on distinct regions of
                // the HNSW. We cannot risk accessing the same neighbor lists at the same time, all hell would break loose.
                // But because all the chunks are non-overlapping and we parallelize only across chunks, all
                // references HNSW elements should be unique and no race conditions should occur reading or editing them.
                let chunk_nodes = &vp_tree[chunk.range.clone()];
                let chunk_size = chunk.range.len();
                assert!(chunk_size >= max_chunk_size / 2);
                assert!(chunk_size <= max_chunk_size);

                match params.strategy {
                    EnsembleStrategy::BruteForceKNN => {
                        connect_vp_tree_chunk_and_update_neighbors_in_hnsw_brute_force(
                            chunk_nodes,
                            data,
                            distance,
                            ensemble_brute_force_buffers_tls(),
                            &unsafe_get_mut_neighbors_in_hnsw_at_idx,
                            same_chunk_m_max,
                        );
                    }
                    EnsembleStrategy::RNNDescent {
                        i_loops: inner_loops,
                        o_loops: outer_loops,
                    } => {
                        connect_vp_tree_chunk_and_update_neighbors_in_hnsw_rnn_descent(
                            chunk_nodes,
                            data,
                            distance,
                            ensemble_rnn_descent_tls(),
                            &unsafe_get_mut_neighbors_in_hnsw_at_idx,
                            same_chunk_m_max,
                            inner_loops,
                            outer_loops,
                        );
                    }
                    EnsembleStrategy::BruteForceExactRNG => {
                        connect_vp_tree_chunk_and_update_neighbors_in_hnsw_exact_rng(
                            chunk_nodes,
                            data,
                            distance,
                            &unsafe_get_mut_neighbors_in_hnsw_at_idx,
                        );
                    }
                }
            }
            println!(
                "    Thread finished work for {} chunks in {}ms",
                chunk_chunk.len(),
                start.elapsed().as_secs_f32() * 1000.0
            );
        });

        println!(
            "Chunks ({}) loop took: {}ms + {}ms",
            chunks.len(),
            between.as_secs_f32() * 1000.0,
            (before.elapsed() - between).as_secs_f32() * 1000.0
        );
    }

    // put the layer back where it was before (move out of the UnsafeCell we used above)
    *layer = layer_cell.into_inner();
}

fn fill_hnsw_layer_by_vp_tree_ensemble_threaded_by_chunk_with_core_affinity(
    data: &dyn DatasetT,
    distance: &DistanceTracker,
    layer: &mut super::slice_hnsw::Layer,
    rng: &mut ChaCha20Rng,
    params: EnsembleParams,
) {
    let n_entries = layer.entries.len();
    let mut vp_tree: Vec<VpTreeNode> = Vec::with_capacity(data.len());
    for (i, entry) in layer.entries.iter().enumerate() {
        vp_tree.push(VpTreeNode {
            id: entry.id as u32,
            idx_in_hnsw_layer: i as u32,
            dist: 0.0,
        });
    }
    let max_chunk_size = params.max_chunk_size;
    let same_chunk_m_max = params.same_chunk_m_max;
    let chunks = make_chunks(n_entries, max_chunk_size);
    let data_get = |e: &VpTreeNode| data.get(e.id as usize);

    let mut num_threads: usize = sanititze_num_threads(0);
    const MIN_CHUNK_CHUNK_SIZE: usize = 4; // to not have chunks that are too small
    let chunk_chunk_size = MIN_CHUNK_CHUNK_SIZE.max(chunks.len() / num_threads);
    let chunk_chunks: Vec<&[Chunk]> = chunks.chunks(chunk_chunk_size).collect();
    if chunk_chunks.len() < num_threads {
        num_threads = chunk_chunks.len();
    }

    // Now we officially get the permission to mutate the layer from multiple threads (MUST wrap in UnsafeCell if mutable accesses to this shared reference can happen):
    let layer_cell = YoloCell::new(std::mem::replace(layer, super::slice_hnsw::Layer::new(1)));
    let unsafe_get_mut_neighbors_in_hnsw_at_idx =
        |idx_in_hnsw: usize| unsafe { &mut layer_cell.get_mut().entries[idx_in_hnsw].neighbors };

    for _tree_idx in 0..params.n_vp_trees {
        let before = Instant::now();
        arrange_into_vp_tree_parallel_with_n_candidates(
            &mut vp_tree,
            &data_get,
            distance,
            rng,
            max_chunk_size / 2,
            params.n_candidates,
        );

        let between = before.elapsed();

        let core_ids = core_affinity::get_core_ids().unwrap();
        assert!(num_threads <= core_ids.len());

        std::thread::scope(|s| {
            let mut t_handles: Vec<std::thread::ScopedJoinHandle<'_, ()>> = vec![];
            for t_idx in 0..num_threads {
                let chunk_chunk = chunk_chunks[t_idx];
                let core_id = &core_ids[t_idx];
                let handle = s.spawn(|| {
                    let affinity_is_set = core_affinity::set_for_current(*core_id);
                    assert!(affinity_is_set);

                    let start = Instant::now();
                    for chunk in chunk_chunk.iter() {
                        // Attention: This needs to be ABSOLUTELY sure, that all threads operate on distinct regions of
                        // the HNSW. We cannot risk accessing the same neighbor lists at the same time, all hell would break loose.
                        // But because all the chunks are non-overlapping and we parallelize only across chunks, all
                        // references HNSW elements should be unique and no race conditions should occur reading or editing them.
                        let chunk_nodes = &vp_tree[chunk.range.clone()];
                        let chunk_size = chunk.range.len();
                        assert!(chunk_size >= max_chunk_size / 2);
                        assert!(chunk_size <= max_chunk_size);

                        match params.strategy {
                            EnsembleStrategy::BruteForceKNN => {
                                connect_vp_tree_chunk_and_update_neighbors_in_hnsw_brute_force(
                                    chunk_nodes,
                                    data,
                                    distance,
                                    ensemble_brute_force_buffers_tls(),
                                    &unsafe_get_mut_neighbors_in_hnsw_at_idx,
                                    same_chunk_m_max,
                                );
                            }
                            EnsembleStrategy::RNNDescent {
                                i_loops: inner_loops,
                                o_loops: outer_loops,
                            } => {
                                connect_vp_tree_chunk_and_update_neighbors_in_hnsw_rnn_descent(
                                    chunk_nodes,
                                    data,
                                    distance,
                                    ensemble_rnn_descent_tls(),
                                    &unsafe_get_mut_neighbors_in_hnsw_at_idx,
                                    same_chunk_m_max,
                                    inner_loops,
                                    outer_loops,
                                );
                            }
                            EnsembleStrategy::BruteForceExactRNG => {
                                connect_vp_tree_chunk_and_update_neighbors_in_hnsw_exact_rng(
                                    chunk_nodes,
                                    data,
                                    distance,
                                    &unsafe_get_mut_neighbors_in_hnsw_at_idx,
                                );
                            }
                        }
                    }
                    println!(
                        "    Thread finished work for {} chunks in {}ms",
                        chunk_chunk.len(),
                        start.elapsed().as_secs_f32() * 1000.0
                    );
                });
                t_handles.push(handle);
            }
            for handle in t_handles {
                handle.join();
            }
        });

        println!(
            "Chunks ({}) loop took: {}ms + {}ms",
            chunks.len(),
            between.as_secs_f32() * 1000.0,
            (before.elapsed() - between).as_secs_f32() * 1000.0
        );
    }

    // put the layer back where it was before (move out of the UnsafeCell we used above)
    *layer = layer_cell.into_inner();
}

thread_local! {
    static ENSEMPLE_RNN_DESCENT_TLS:UnsafeCell<RNNConstructionBuffers> = const {UnsafeCell::new(RNNConstructionBuffers::new())};
}

fn ensemble_rnn_descent_tls() -> &'static mut RNNConstructionBuffers {
    ENSEMPLE_RNN_DESCENT_TLS.with(|e| unsafe { &mut *e.get() })
}

fn connect_vp_tree_chunk_and_update_neighbors_in_hnsw_rnn_descent<'task, 'total>(
    chunk_nodes: &'task [VpTreeNode],
    data: &'total dyn DatasetT,
    distance: &'total DistanceTracker,
    buffers: &'task mut RNNConstructionBuffers,
    unsafe_get_mut_neighbors_in_hnsw_at_idx: &impl Fn(
        usize,
    ) -> &'total mut SliceBinaryHeap<
        'static,
        DistAnd<usize>,
    >,
    same_chunk_m_max: usize,
    inner_loops: usize,
    outer_loops: usize,
) {
    let distance_idx_to_idx = |idx_a: usize, idx_b: usize| -> f32 {
        let id_a = chunk_nodes[idx_a].id as usize;
        let id_b = chunk_nodes[idx_b].id as usize;
        let data_a = data.get(id_a);
        let data_b = data.get(id_b);
        distance.distance(data_a, data_b)
    };

    const SEED: u64 = 42;
    let chunk_size = chunk_nodes.len();
    crate::relative_nn_descent::construct_relative_nn_graph(
        RNNGraphParams {
            outer_loops,
            inner_loops,
            max_neighbors_after_reverse_pruning: same_chunk_m_max,
            initial_neighbors: same_chunk_m_max,
            distance: Distance::Dot, // does not matter, is not used.
        },
        SEED,
        &distance_idx_to_idx,
        buffers,
        chunk_size,
        false, // no threading here, because this should already run in a threaded loop!
    );

    // merge the neighbors of the vp-tree in with the hnsw layer:
    for i in 0..chunk_size {
        let i_idx_in_hnsw = chunk_nodes[i].idx_in_hnsw_layer as usize;
        let i_neighbors = &mut buffers.neighbors[i];
        let hnsw_neighbors = unsafe_get_mut_neighbors_in_hnsw_at_idx(i_idx_in_hnsw);
        for nei in i_neighbors.iter() {
            let nei_idx_in_hnsw = chunk_nodes[nei.idx].idx_in_hnsw_layer as usize;
            if !hnsw_neighbors.iter().any(|e| e.1 == nei_idx_in_hnsw) {
                hnsw_neighbors.insert_if_better(DistAnd(nei.dist, nei_idx_in_hnsw));
            }
        }
    }
}

/// Experimental, just for illustration!!! inefficient, not optimized for threading, buffer reuse, etc!!!
fn connect_vp_tree_chunk_and_update_neighbors_in_hnsw_exact_rng<'task, 'total>(
    chunk_nodes: &'task [VpTreeNode],
    data: &'total dyn DatasetT,
    distance: &'total DistanceTracker,
    unsafe_get_mut_neighbors_in_hnsw_at_idx: &impl Fn(
        usize,
    ) -> &'total mut SliceBinaryHeap<
        'static,
        DistAnd<usize>,
    >,
) {
    let chunk_size = chunk_nodes.len();
    let mut chunk_dst_mat: Vec<f32> = vec![0.0; chunk_size * chunk_size];
    let chunk_dst_mat_idx = |i: usize, j: usize| i + j * chunk_size;

    // calculate distances between all the nodes in this chunk. (clearing chunk_dst_mat not necessary, because everything relevant should be overwritten).
    for i in 0..chunk_size {
        for j in i + 1..chunk_size {
            let i_data = data.get(chunk_nodes[i].id as usize);
            let j_data = data.get(chunk_nodes[j].id as usize);
            let dist = distance.distance(i_data, j_data);
            chunk_dst_mat[chunk_dst_mat_idx(i, j)] = dist;
            chunk_dst_mat[chunk_dst_mat_idx(j, i)] = dist;
        }
    }

    // idx-idx in chunk nodes and their distance:
    // let mut rng_graph_edges: HashMap<(usize, usize), f32> = HashMap::new();
    for i in 0..chunk_size {
        for j in i + 1..chunk_size {
            // RNG strategy: insert edge if d(i,j) <= max(d(v,i), d(v,j)) for all v in chunk
            let d_ij = chunk_dst_mat[chunk_dst_mat_idx(i, j)];
            let mut min_of_max_d_vi_d_vj = f32::MAX;
            for v in 0..chunk_size {
                if v == i || v == j {
                    continue;
                }
                let d_vi = chunk_dst_mat[chunk_dst_mat_idx(v, i)];
                let d_vj = chunk_dst_mat[chunk_dst_mat_idx(v, j)];
                let max_d_vi_d_vj = d_vi.max(d_vj);
                if max_d_vi_d_vj < min_of_max_d_vi_d_vj {
                    min_of_max_d_vi_d_vj = max_d_vi_d_vj;
                }
            }
            if d_ij < min_of_max_d_vi_d_vj {
                // RNG condition fulfilled.
                let i_idx_in_hnsw = chunk_nodes[i].idx_in_hnsw_layer as usize;
                let j_idx_in_hnsw = chunk_nodes[j].idx_in_hnsw_layer as usize;

                let i_hnsw_neighbors = unsafe_get_mut_neighbors_in_hnsw_at_idx(i_idx_in_hnsw);
                if !i_hnsw_neighbors.iter().any(|e| e.1 == j_idx_in_hnsw) {
                    i_hnsw_neighbors.insert_if_better(DistAnd(d_ij, j_idx_in_hnsw));
                }

                let j_hnsw_neighbors = unsafe_get_mut_neighbors_in_hnsw_at_idx(i_idx_in_hnsw);
                if !j_hnsw_neighbors.iter().any(|e| e.1 == i_idx_in_hnsw) {
                    j_hnsw_neighbors.insert_if_better(DistAnd(d_ij, i_idx_in_hnsw));
                }
            }
        }
    }
}

/// Thread-Local storage to help build VP-trees and connect neighbors in chunks.
struct EnsembleBruteForceBuffers {
    chunk_dst_mat: Vec<f32>,
    chunk_neighbors_memory: SlicesMemory<DistAnd<usize>>,
    chunk_neighbors: Vec<SliceBinaryHeap<'static, DistAnd<usize>>>,
}

impl EnsembleBruteForceBuffers {
    fn new_uninit() -> Self {
        Self {
            chunk_dst_mat: vec![],
            chunk_neighbors_memory: SlicesMemory::new_uninit(),
            chunk_neighbors: vec![],
        }
    }

    fn new(max_chunk_size: usize, same_chunk_m_max: usize) -> Self {
        let (memory, neighbors) =
            slice_binary_heap_arena::<DistAnd<usize>>(max_chunk_size, same_chunk_m_max);
        Self {
            chunk_dst_mat: vec![0.0; max_chunk_size * max_chunk_size],
            chunk_neighbors_memory: memory,
            chunk_neighbors: neighbors,
        }
    }

    fn init(&mut self, max_chunk_size: usize, same_chunk_m_max: usize) {
        self.chunk_dst_mat.clear();
        self.chunk_dst_mat.reserve(max_chunk_size * max_chunk_size);
        // for _ in 0..(max_chunk_size * max_chunk_size) {
        //     self.chunk_dst_mat.push(0.0);
        // }
        unsafe {
            self.chunk_dst_mat.set_len(max_chunk_size * max_chunk_size);
        }

        // Reuse memory if exactly same size and number of neighbors is already available, otherwise allocate new memory:
        if self.chunk_neighbors.len() == max_chunk_size
            && self
                .chunk_neighbors_memory
                .is_allocated_for(max_chunk_size, same_chunk_m_max)
        {
            for neighbors in self.chunk_neighbors.iter_mut() {
                neighbors.clear();
            }
        } else {
            let (memory, neighbors) =
                slice_binary_heap_arena::<DistAnd<usize>>(max_chunk_size, same_chunk_m_max);
            self.chunk_neighbors_memory = memory;
            self.chunk_neighbors = neighbors;
        }
    }
}

thread_local! {
    static ENSEMPLE_BRUTE_FORCE_TLS:UnsafeCell<EnsembleBruteForceBuffers> = const {UnsafeCell::new(EnsembleBruteForceBuffers{chunk_dst_mat:vec![],chunk_neighbors_memory:SlicesMemory::new_uninit(), chunk_neighbors: vec![] })};
}

fn ensemble_brute_force_buffers_tls() -> &'static mut EnsembleBruteForceBuffers {
    ENSEMPLE_BRUTE_FORCE_TLS.with(|e| unsafe { &mut *e.get() })
}

fn heavy_work_dummy_task() {
    let mut x: i64 = 0;
    for i in 0..10000000i64 {
        for j in 0..100i64 {
            x += j + i % 113;
        }
    }
    std::hint::black_box(x);
}

fn connect_vp_tree_chunk_and_update_neighbors_in_hnsw_brute_force<'task, 'total>(
    chunk_nodes: &'task [VpTreeNode],
    data: &'total dyn DatasetT,
    distance: &'total DistanceTracker,
    buffers: &'task mut EnsembleBruteForceBuffers,
    unsafe_get_mut_neighbors_in_hnsw_at_idx: &impl Fn(
        usize,
    ) -> &'total mut SliceBinaryHeap<
        'static,
        DistAnd<usize>,
    >,
    same_chunk_m_max: usize,
) {
    // heavy_work_dummy_task();
    // return;
    let chunk_size = chunk_nodes.len();
    buffers.init(chunk_size, same_chunk_m_max);
    let chunk_dst_mat_idx = |i: usize, j: usize| i + j * chunk_size;

    // calculate distances between all the nodes in this chunk. (clearing chunk_dst_mat not necessary, because everything relevant should be overwritten).
    for i in 0..chunk_size {
        if_tracking!(Tracking.pt_meta(chunk_nodes[i].id as usize).chunk = Tracking.current_chunk);
        for j in i + 1..chunk_size {
            let i_data = data.get(chunk_nodes[i].id as usize);
            let j_data = data.get(chunk_nodes[j].id as usize);
            let dist = distance.distance(i_data, j_data);
            buffers.chunk_dst_mat[chunk_dst_mat_idx(i, j)] = dist;
            buffers.chunk_dst_mat[chunk_dst_mat_idx(j, i)] = dist;
        }
    }
    // try to connect each node in the chunk to each other node (except itself) (limiting number of neighbors to same_chunk_m_max with the insert_if_better)
    for i in 0..chunk_size {
        // since all chunks are disjunct and each element is only in one chunk, this access should be fine and cannot lead to race conditions.
        let i_neighbors = &mut buffers.chunk_neighbors[i];
        for j in 0..chunk_size {
            if j == i {
                continue;
            }
            let dist = buffers.chunk_dst_mat[chunk_dst_mat_idx(i, j)];
            let j_idx_in_hnsw = chunk_nodes[j].idx_in_hnsw_layer as usize;
            i_neighbors.insert_if_better(DistAnd(dist, j_idx_in_hnsw));
        }
    }

    // merge the neighbors of the vp-tree in with the hnsw layer:

    for i in 0..chunk_size {
        let i_idx_in_hnsw = chunk_nodes[i].idx_in_hnsw_layer as usize;
        let i_neighbors: &mut SliceBinaryHeap<'_, DistAnd<usize>> = &mut buffers.chunk_neighbors[i];
        let hnsw_neighbors = unsafe_get_mut_neighbors_in_hnsw_at_idx(i_idx_in_hnsw);
        for &DistAnd(dist, idx_in_hnsw) in i_neighbors.iter() {
            if !hnsw_neighbors.iter().any(|e| e.1 == idx_in_hnsw) {
                hnsw_neighbors.insert_if_better(DistAnd(dist, idx_in_hnsw));
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, SerJson, DeJson)]
pub struct StitchingParams {
    pub max_chunk_size: usize,
    pub same_chunk_m_max: usize,
    pub neg_fraction: f32,
    pub keep_fraction: f32,
    pub m_max: usize, // max number of neighbors in hnsw
    pub x_or_ef: usize,
    pub only_n_chunks: Option<usize>, // this is only for debugging, could theoretically be behind a feature flag
    pub distance: Distance,
    pub stitch_mode: StitchMode,
    pub n_candidates: usize,
}

pub fn build_hnsw_by_vp_tree_stitching(
    data: Arc<dyn DatasetT>,
    params: StitchingParams,
    seed: u64,
) -> SliceHnsw {
    let max_chunk_size = params.max_chunk_size;
    let same_chunk_m_max = params.same_chunk_m_max;
    let mut distance = DistanceTracker::new(params.distance);
    let start = Instant::now();
    let mut rng = ChaCha20Rng::seed_from_u64(seed);

    let mut vp_tree: Vec<vp_tree::Node> = Vec::with_capacity(data.len());
    for idx in 0..data.len() {
        vp_tree.push(vp_tree::Node { id: idx, dist: 0.0 });
    }

    let data_get = |e: &vp_tree::Node| data.get(e.id);
    arrange_into_vp_tree_with_n_candidates(
        &mut vp_tree,
        &data_get,
        &mut distance,
        &mut rng,
        params.n_candidates,
    );

    // create an hnsw layer with same order as vp-tree but no neighbors:
    let mut layer = slice_hnsw::Layer::new(params.m_max);
    layer.entries_cap = data.len();
    layer.allocate_neighbors_memory();
    for node in vp_tree.iter() {
        layer.add_entry_assuming_allocated_memory(node.id, usize::MAX);
    }

    let mut chunks = make_chunks(vp_tree.len(), max_chunk_size);
    let mut chunk_dst_mat: Vec<f32> = vec![0.0; max_chunk_size * max_chunk_size]; // memory reused for each chunk.
    for (chunk_i, chunk) in chunks.iter().enumerate() {
        let chunk_size = chunk.range.len();
        let chunk_dst_mat_idx = |i: usize, j: usize| i + j * chunk_size;
        let chunk_nodes = &vp_tree[chunk.range.clone()];

        assert!(chunk_size >= max_chunk_size / 2);
        assert!(chunk_size <= max_chunk_size);

        // calculate distances between all the nodes in this chunk. (clearing chunk_dst_mat not necessary, because everything relevant should be overwritten).
        for i in 0..chunk_size {
            chunk_dst_mat[chunk_dst_mat_idx(i, i)] = 0.0;
            for j in i + 1..chunk_size {
                let i_data = data.get(chunk_nodes[i].id);
                let j_data = data.get(chunk_nodes[j].id);
                let dist = distance.distance(i_data, j_data);
                chunk_dst_mat[chunk_dst_mat_idx(i, j)] = dist;
                chunk_dst_mat[chunk_dst_mat_idx(j, i)] = dist;
            }
        }

        // connect each node in the chunk to each other node (except itself)
        for i in 0..chunk_size {
            let entry = &mut layer.entries[chunk.range.start + i];

            if_tracking!(Tracking.pt_meta(entry.id).chunk = chunk_i);

            for j in 0..chunk_size {
                if j == i {
                    continue;
                }
                let neighbor_idx_in_layer = chunk.range.start + j;
                let neighbor_dist = chunk_dst_mat[chunk_dst_mat_idx(i, j)];

                entry.neighbors.insert_if_better_with_max_len(
                    DistAnd(neighbor_dist, neighbor_idx_in_layer),
                    same_chunk_m_max,
                );
            }
        }
    }
    // stitch positive and negative halves together.
    let mut i = 0;
    while chunks.len() > 1 {
        let pos_idx = chunk_idx_of_pos_half_to_stitch(&chunks);
        let neg_idx = pos_idx + 1;
        if_tracking!(
          println!(
              "Stitch positive half (idx = {pos_idx}, size = {}) with negative half (idx = {neg_idx}, size = {})",
              chunks[pos_idx].range.len(),
              chunks[neg_idx].range.len()
          );
        );
        let merged_chunk = stitch_chunks(
            &*data,
            &distance,
            &chunks[pos_idx],
            &chunks[neg_idx],
            &mut layer.entries,
            &params,
            &mut rng,
        );
        i += 1;
        if let Some(n) = params.only_n_chunks {
            if i >= n {
                break;
            }
        }
        chunks.remove(neg_idx);
        chunks[pos_idx] = merged_chunk;
    }

    // insert a representative of each cluster (first element?)

    let mut layers = slice_hnsw::Layers::new();
    layers.push(layer);

    let build_stats = Stats {
        num_distance_calculations: distance.num_calculations(),
        duration: start.elapsed(),
    };
    SliceHnsw {
        params: Default::default(),
        data,
        layers,
        build_stats,
    }
}

fn stitch_chunks(
    data: &dyn DatasetT,
    distance: &DistanceTracker,
    pos_chunk: &Chunk,
    neg_chunk: &Chunk,
    entries: &mut [slice_hnsw::LayerEntry],
    params: &StitchingParams,
    rng: &mut ChaCha20Rng,
) -> Chunk {
    let len_diff = neg_chunk.range.len() - pos_chunk.range.len();
    assert!(len_diff == 1 || len_diff == 0); // todo: this assertion sometimes fails, figure out why
    assert_eq!(pos_chunk.level, neg_chunk.level);
    assert_eq!(pos_chunk.range.end, neg_chunk.range.start);

    let merged_chunk = Chunk {
        range: pos_chunk.range.start..neg_chunk.range.end,
        level: pos_chunk.level + 1,
        pos_center_idx_offset: if pos_chunk.pos_center_idx_offset == 0 {
            0
        } else {
            pos_chunk.pos_center_idx_offset - 1 // go one step up in the vp-tree hierarchy, so the pos center index moves to the next element (somewhere in the far left of the slice anyway)
        },
    };

    let stitching_candidates =
        generate_stitching_candidates(data, distance, pos_chunk, neg_chunk, entries, params, rng);

    for c in stitching_candidates.iter() {
        let pos_entry = &mut entries[c.pos_cand_idx];
        let _pos_to_neg_inserted = pos_entry
            .neighbors
            .insert_if_better(DistAnd(c.dist, c.neg_cand_idx));

        let neg_entry = &mut entries[c.neg_cand_idx];
        let _neg_to_pos_inserted = neg_entry
            .neighbors
            .insert_if_better(DistAnd(c.dist, c.pos_cand_idx));

        if_tracking!(
            let pos_cand_id = entries[c.pos_cand_idx].id;
            let neg_cand_id = entries[c.neg_cand_idx].id;
            if _pos_to_neg_inserted {
                Tracking
                    .edge_meta(pos_cand_id, neg_cand_id)
                    .is_pos_to_neg = true;
            }
            if _neg_to_pos_inserted {
                Tracking
                    .edge_meta(neg_cand_id, pos_cand_id)
                    .is_neg_to_pos = true;
            }
            // println!("    potential connection: pos:{pos_cand_id} - neg:{neg_cand_id}  (pos_to_neg_inserted={pos_to_neg_inserted}, neg_to_pos_inserted={neg_to_pos_inserted})")
        );
    }

    merged_chunk
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct StitchCandidate {
    pos_cand_idx: usize,
    neg_cand_idx: usize,
    dist: f32,
}
impl Hash for StitchCandidate {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.pos_cand_idx.hash(state);
        self.neg_cand_idx.hash(state);
    }
}
impl Eq for StitchCandidate {}
impl PartialOrd for StitchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}
impl Ord for StitchCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist.total_cmp(&other.dist)
    }
}

fn random_idx_sample(range: Range<usize>, fraction: f32, rng: &mut ChaCha20Rng) -> Vec<usize> {
    let len = range.len();
    let mut ids_random_order: Vec<usize> = range.collect();
    ids_random_order.shuffle(rng);
    let mut max_candidate_count = (fraction * len as f32) as usize;
    if max_candidate_count > len {
        max_candidate_count = len;
    }
    ids_random_order.truncate(max_candidate_count);
    ids_random_order
}

#[derive(Debug, Clone, Copy, PartialEq, SerJson, DeJson, strum::EnumIter)]
pub enum StitchMode {
    /// - select `neg_fraction` random points in negative half.
    /// - search from each of them in neg half towards center of positive half -> neg candidates
    /// - for each neg candidate: search in pos half from center of positive half towards neg candidate -> pos candidate
    /// - connect pos and neg candidates if the connection would be good.
    RandomNegToPosCenterAndBack,
    RandomNegToRandomPosAndBack,
    /// select neg_fraction points randomly in pos and neg half:
    /// - compute all distances and sort them
    /// - suggest the best `keep_fraction` distances as candidates
    RandomSubsetOfSubset,
    /// Select in many iterations X candidates in negative and positive half: (so that in total we look at `neg_fraction` pts roughly)
    /// - compute all X^2 distances between them, select the best X of these distances and return the pairs as candidates.
    BestXofRandomXTimesX,
    /// Select X candidates in negative and positive half:
    /// - for each positive candidate, search towards it, from a negative candidate if it is closer to it than each other negative candidate.
    /// -> leads to X searches from points in neg, towards pos half.
    ///
    /// Afterwards, search from the X pos candidates in pos half towards the X found neg candidates.
    DontStarveXXSearch,
    /// search exactly like in RandomNegToRandomPosAndBack but search with higher ef to find multiple points near the seam.
    /// idea: could also tie ef loosely to chunk size???
    /// ef == x
    MultiEf,
}

impl StitchMode {
    pub fn name(&self) -> &'static str {
        match self {
            StitchMode::RandomNegToPosCenterAndBack => "RandomNegToPosCenterAndBack",
            StitchMode::RandomNegToRandomPosAndBack => "RandomNegToRandomPosAndBack",
            StitchMode::RandomSubsetOfSubset => "RandomSubsetOfSubset",
            StitchMode::BestXofRandomXTimesX => "BestXofRandomXTimesX",
            StitchMode::DontStarveXXSearch => "DontStarveXXSearch",
            StitchMode::MultiEf => "MultiEf",
        }
    }
}

fn generate_stitching_candidates(
    data: &dyn DatasetT,
    distance: &DistanceTracker,
    pos_chunk: &Chunk,
    neg_chunk: &Chunk,
    entries: &[slice_hnsw::LayerEntry],
    params: &StitchingParams,
    rng: &mut ChaCha20Rng,
) -> HashSet<StitchCandidate> {
    let mut result: HashSet<StitchCandidate> = HashSet::default();

    // reusable buffer
    let visited = &mut HashSet::<usize>::default();

    match params.stitch_mode {
        StitchMode::RandomNegToPosCenterAndBack => {
            let pos_center_idx = pos_chunk.range.start + pos_chunk.pos_center_idx_offset;
            let pos_center_id = entries[pos_center_idx].id;
            let pos_center_data = data.get(pos_center_id);

            if_tracking!(
               println!("    center of pos half: {pos_center_id}");
               let mut t = Tracking;
               let meta = t.pt_meta(pos_center_id);
               meta.is_pos_center = true;
               meta.annotation = Some(meta.chunk.to_string());
            );

            let mut neg_candidates = HashSet::<usize>::default();
            let neg_random_indixes =
                random_idx_sample(neg_chunk.range.clone(), params.neg_fraction, rng);
            for random_idx in neg_random_indixes {
                if_tracking!(
                    let random_id = entries[random_idx].id;
                    Tracking.pt_meta(random_id).is_neg_random = true;
                );
                let DistAnd(_, neg_cand_idx) = search_layer_ef_1(
                    data,
                    distance,
                    visited,
                    entries,
                    pos_center_data,
                    random_idx,
                );

                if neg_candidates.insert(neg_cand_idx) {
                    let neg_cand_id = entries[neg_cand_idx].id;
                    if_tracking!(
                        Tracking.pt_meta(neg_cand_id).is_neg_cand = true;
                    );
                    let neg_cand_data = data.get(neg_cand_id);
                    let DistAnd(dist, pos_cand_idx) = search_layer_ef_1(
                        data,
                        distance,
                        visited,
                        entries,
                        neg_cand_data,
                        pos_center_idx,
                    );
                    result.insert(StitchCandidate {
                        neg_cand_idx,
                        pos_cand_idx,
                        dist,
                    });
                }
            }
        }
        StitchMode::RandomNegToRandomPosAndBack => {
            let mut neg_random_indices =
                random_idx_sample(neg_chunk.range.clone(), params.neg_fraction, rng);
            let mut pos_random_indices =
                random_idx_sample(pos_chunk.range.clone(), params.neg_fraction, rng);
            let len = neg_random_indices.len().min(pos_random_indices.len());
            neg_random_indices.truncate(len);
            pos_random_indices.truncate(len);

            for i in 0..len {
                let neg_random_idx = neg_random_indices[i];
                let pos_random_idx = pos_random_indices[i];
                let pos_random_id = entries[pos_random_idx].id;
                let pos_random_data = data.get(pos_random_id);
                // let (neg_cand_idx, _) = greedy_search_in_range(
                //     data,
                //     distance,
                //     entries,
                //     neg_chunk.range.clone(),
                //     neg_random_idx,
                //     pos_random_data,
                // );

                let DistAnd(_, neg_cand_idx) = search_layer_ef_1(
                    data,
                    distance,
                    visited,
                    entries,
                    pos_random_data,
                    neg_random_idx,
                );

                let neg_cand_id = entries[neg_cand_idx].id;
                let neg_cand_data = data.get(neg_cand_id);
                // let (pos_cand_idx, dist) = greedy_search_in_range(
                //     data,
                //     distance,
                //     entries,
                //     pos_chunk.range.clone(),
                //     pos_random_idx,
                //     neg_cand_data,
                // );

                let DistAnd(dist, pos_cand_idx) = search_layer_ef_1(
                    data,
                    distance,
                    visited,
                    entries,
                    neg_cand_data,
                    pos_random_idx,
                );

                result.insert(StitchCandidate {
                    neg_cand_idx,
                    pos_cand_idx,
                    dist,
                });
            }
        }
        StitchMode::RandomSubsetOfSubset => {
            let mut neg_random_indices =
                random_idx_sample(neg_chunk.range.clone(), params.neg_fraction, rng);
            let mut pos_random_indices =
                random_idx_sample(pos_chunk.range.clone(), params.neg_fraction, rng);
            let len = neg_random_indices.len().min(pos_random_indices.len());
            neg_random_indices.truncate(len);
            pos_random_indices.truncate(len);

            let mut max_candidates =
                (params.keep_fraction * neg_chunk.range.len() as f32).floor() as usize;
            if max_candidates == 0 {
                max_candidates = 1;
            }
            let mut candidates: BinaryHeap<StitchCandidate> = Default::default();

            for neg_cand_idx in neg_random_indices {
                let neg_id = entries[neg_cand_idx].id;
                let neg_data = data.get(neg_id);
                for &pos_cand_idx in pos_random_indices.iter() {
                    let pos_id = entries[pos_cand_idx].id;
                    let pos_data = data.get(pos_id);
                    let dist = distance.distance(neg_data, pos_data);
                    let cand = StitchCandidate {
                        pos_cand_idx,
                        neg_cand_idx,
                        dist,
                    };
                    candidates.insert_if_better(cand, max_candidates);
                }
            }
            for c in candidates {
                result.insert(c);
            }
        }

        StitchMode::BestXofRandomXTimesX => {
            let mut neg_random_indices =
                random_idx_sample(neg_chunk.range.clone(), params.neg_fraction, rng);
            let mut pos_random_indices =
                random_idx_sample(pos_chunk.range.clone(), params.neg_fraction, rng);
            let x = params.x_or_ef;
            let outer_loops = neg_random_indices.len().min(pos_random_indices.len()) / x;

            neg_random_indices.truncate(outer_loops * x);
            pos_random_indices.truncate(outer_loops * x);

            for i in 0..outer_loops {
                // perform the XX search, it is like football players running towards each other, but they can starve or duplicate.

                let range = i * x..(i + 1) * x;

                let neg_indices = &neg_random_indices[range.clone()];
                let pos_indices = &pos_random_indices[range];

                let mut x_searches: BinaryHeap<StitchCandidate> = Default::default();
                for &neg_cand_idx in neg_indices.iter() {
                    let neg_id = entries[neg_cand_idx].id;
                    let neg_data = data.get(neg_id);
                    for &pos_cand_idx in pos_indices.iter() {
                        let pos_id = entries[pos_cand_idx].id;
                        let pos_data = data.get(pos_id);
                        let dist = distance.distance(neg_data, pos_data);
                        x_searches.insert_if_better(
                            StitchCandidate {
                                pos_cand_idx,
                                neg_cand_idx,
                                dist,
                            },
                            x,
                        );
                    }
                }
                assert_eq!(x_searches.len(), x);
                for x in x_searches {
                    result.insert(x);
                }
            }
        }
        StitchMode::DontStarveXXSearch => {
            let mut neg_random_indices =
                random_idx_sample(neg_chunk.range.clone(), params.neg_fraction, rng);
            let mut pos_random_indices =
                random_idx_sample(pos_chunk.range.clone(), params.neg_fraction, rng);
            let x = params.x_or_ef;
            let outer_loops = neg_random_indices.len().min(pos_random_indices.len()) / x;

            neg_random_indices.truncate(outer_loops * x);
            pos_random_indices.truncate(outer_loops * x);

            let mut x_searches: BinaryHeap<StitchCandidate> = Default::default();

            for i in 0..outer_loops {
                x_searches.clear();
                // perform the XX search, it is like football players running towards each other, but they can starve or duplicate. What??!

                let range = i * x..(i + 1) * x;
                let neg_indices = &neg_random_indices[range.clone()];
                let pos_indices = &pos_random_indices[range];

                // X^2 distance calculations (random in neg, random in pos), to determine which X point pairs are closest and should have searches between them:
                for &neg_cand_idx in neg_indices.iter() {
                    let neg_id = entries[neg_cand_idx].id;
                    let neg_data = data.get(neg_id);
                    for &pos_cand_idx in pos_indices.iter() {
                        let pos_id = entries[pos_cand_idx].id;
                        let pos_data = data.get(pos_id);
                        let dist = distance.distance(neg_data, pos_data);
                        x_searches.insert_if_better(
                            StitchCandidate {
                                pos_cand_idx,
                                neg_cand_idx,
                                dist,
                            },
                            x,
                        );
                    }
                }
                assert_eq!(x_searches.len(), x);
                // peform the x different searches and attempt to connect the neighbors:
                for x_search in x_searches.iter() {
                    let neg_start_idx = x_search.neg_cand_idx;
                    let pos_start_idx = x_search.pos_cand_idx;

                    let pos_id = entries[pos_start_idx].id;
                    let pos_data = data.get(pos_id);
                    let DistAnd(_, neg_cand_idx) = search_layer_ef_1(
                        data,
                        distance,
                        visited,
                        entries,
                        pos_data,
                        neg_start_idx,
                    );

                    let neg_id = entries[neg_cand_idx].id;
                    let neg_data = data.get(neg_id);

                    let DistAnd(dist, pos_cand_idx) = search_layer_ef_1(
                        data,
                        distance,
                        visited,
                        entries,
                        neg_data,
                        pos_start_idx,
                    );
                    result.insert(StitchCandidate {
                        pos_cand_idx,
                        neg_cand_idx,
                        dist,
                    });
                }
            }
        }
        StitchMode::MultiEf => {
            let mut neg_random_indices =
                random_idx_sample(neg_chunk.range.clone(), params.neg_fraction, rng);
            let mut pos_random_indices =
                random_idx_sample(pos_chunk.range.clone(), params.neg_fraction, rng);
            let len = neg_random_indices.len().min(pos_random_indices.len());
            neg_random_indices.truncate(len);
            pos_random_indices.truncate(len);

            let ef = params.x_or_ef;

            let buffers = &mut SearchBuffers::new();
            let mut neg_cand_indices: Vec<usize> = vec![];

            for i in 0..len {
                let pos_start_idx = pos_random_indices[i];
                let pos_start_id = entries[pos_start_idx].id;
                let pos_start_data = data.get(pos_start_id);

                let neg_start_idx = neg_random_indices[i];

                // search from neg to pos: (stores result in buffers.found)
                search_layer(
                    data,
                    distance,
                    buffers,
                    entries,
                    pos_start_data,
                    ef,
                    &[neg_start_idx],
                );
                // save the ef found points in negative half:
                neg_cand_indices.clear();
                for DistAnd(_, idx) in buffers.found.as_slice() {
                    neg_cand_indices.push(*idx);
                }
                // pick the best one as search target and search to it in positive half:
                let target_neg_cand_idx =
                    *neg_cand_indices.iter().min().expect("min 1 element found");
                let neg_cand_id = entries[target_neg_cand_idx].id;
                let neg_cand_data = data.get(neg_cand_id);
                search_layer(
                    data,
                    distance,
                    buffers,
                    entries,
                    neg_cand_data,
                    ef,
                    &[pos_start_idx],
                );

                // insert all ef^2 candidates
                for &DistAnd(dist, pos_cand_idx) in buffers.found.iter() {
                    for &neg_cand_idx in neg_cand_indices.iter() {
                        // calculate distance between the point pair
                        let dist = if neg_cand_idx == target_neg_cand_idx {
                            dist // already calculated if this was the neg candidate searched towards
                        } else {
                            let pos_cand_id = entries[pos_cand_idx].id;
                            let pos_cand_data = data.get(pos_cand_id);
                            let neg_cand_id = entries[neg_cand_idx].id;
                            let neg_cand_data = data.get(neg_cand_id);
                            distance.distance(pos_cand_data, neg_cand_data)
                        };
                        result.insert(StitchCandidate {
                            pos_cand_idx,
                            neg_cand_idx,
                            dist,
                        });
                    }
                }
            }
        }
    }
    result
}

/// Note: the negative half with the same level as the positive half is
/// always at index returned +1;
fn chunk_idx_of_pos_half_to_stitch(chunks: &[Chunk]) -> usize {
    assert!(chunks.len() > 1);

    let mut neg_idx = chunks.len() - 1;

    loop {
        let pos_idx = neg_idx - 1;
        let n_level = chunks[neg_idx].level;
        let p_level = chunks[pos_idx].level;

        if n_level == p_level {
            return pos_idx;
        };
        if pos_idx == 0 {
            panic!("no chunk indices found to stitch")
        }
        neg_idx -= 1;
    }
}

#[derive(Debug)]
struct Chunk {
    /// indices of entries in the vp tree slice are covered by this chunk
    range: Range<usize>,
    /// how many branches have to be gone down into the tree to get to this chunk.
    level: usize,
    pos_center_idx_offset: usize,
}

impl Chunk {
    pub fn len(&self) -> usize {
        self.range.end - self.range.start
    }
}

fn make_chunks(data_len: usize, max_chunk_size: usize) -> Vec<Chunk> {
    let mut chunks: Vec<Chunk> = Vec::with_capacity(data_len / (max_chunk_size / 2));
    /// What does `pos_center_idx_offset` mean?
    /// -> When you see this chunks as a positive half, there is always another chunk that is the negative half right next to it.
    /// -> all points in the negative half, are at least as far from the pos_center than all points in the positive half.
    /// -> this pos-center is the first element of the pos chunk at the top level, see memory layout of vp-tree.
    /// This would be pos_center_idx_offset = 0
    /// But because the vantage point of a higher level is also always part of the positive chunk at a lower level,
    /// The first element can be this point instead of the real center of the positive chunk.
    ///
    /// See drawing pos_chunk_center.png
    ///
    /// Note: in extreme cases this means that the first point (or first few points)
    /// of a positive chunk can be extremely disconnected from the rest of the chunk (even far in the negative half)
    /// This should be rare enough that we deliberatly don't care about it.
    ///
    /// Every time we enter a positive chunk, the pos_center_idx_offset needs to be increased,
    /// Every time we enter a negative chunk it is set back to 0.
    fn collect_chunks(
        range: Range<usize>,
        level: usize,
        max_chunk_size: usize,
        chunks: &mut Vec<Chunk>,
        pos_center_idx_offset: usize,
    ) {
        let len = range.end - range.start;
        if len <= max_chunk_size {
            chunks.push(Chunk {
                range,
                level,
                pos_center_idx_offset,
            });
        } else {
            let first_idx_right = range.start + len / 2;
            let left = Range {
                start: range.start,
                end: first_idx_right,
            };
            let right = Range {
                start: first_idx_right,
                end: range.end,
            };
            // collect positive half:
            collect_chunks(
                left,
                level + 1,
                max_chunk_size,
                chunks,
                pos_center_idx_offset + 1,
            );
            // collect negative half:
            collect_chunks(right, level + 1, max_chunk_size, chunks, 0);
        }
    }
    collect_chunks(0..data_len, 0, max_chunk_size, &mut chunks, 0);
    chunks
}

#[cfg(test)]
#[test]
fn chunk_collection_matches_vp_tree_subtrees() {
    let len = 35;

    let nodes: Vec<Node> = (0..len)
        .map(|e| Node { id: e, dist: 0.0 })
        .collect::<Vec<_>>();

    fn collect_slices<'a>(
        nodes: &'a [Node],
        start: usize,
        max_size: usize,
        all_slices: &mut Vec<&'a [Node]>,
    ) {
        if nodes.len() <= max_size {
            all_slices.push(nodes);
        } else {
            // println!("{nodes:?}");
            collect_slices(left_with_root(nodes), start, max_size, all_slices);
            collect_slices(
                right(nodes),
                start + left_with_root(nodes).len(),
                max_size,
                all_slices,
            );
        }
    }

    let mut slices: Vec<&[Node]> = Vec::new();
    collect_slices(&nodes, 0, 4, &mut slices);
    let chunks = make_chunks(len, 4);
    assert_eq!(chunks.len(), slices.len());
    for (chunk, s) in chunks.iter().zip(slices.iter()) {
        assert_eq!(chunk.range.len(), s.len());

        println!("{chunk:?}      {} ", chunk.range.len());
    }
}

// experimental and probably stupid
pub fn build_hnsw_by_rnn_descent(
    data: Arc<dyn DatasetT>,
    params: RNNGraphParams,
    level_norm_param: f32,
    seed: u64,
) -> SliceHnsw {
    let start_time = Instant::now();
    let mut rng = ChaCha20Rng::seed_from_u64(seed);

    let m_max = params.max_neighbors_after_reverse_pruning / 2;
    let m_max_0 = params.max_neighbors_after_reverse_pruning;

    let mut layers = super::slice_hnsw::create_hnsw_layers_with_empty_neighbors(
        data.len(),
        m_max,
        m_max_0,
        level_norm_param,
        &mut rng,
    );
    let distance = DistanceTracker::new(params.distance);
    // do the ensemble approach for each layer:
    for (l, layer) in layers.iter_mut().enumerate() {
        let distance_idx_to_idx = |idx_a: usize, idx_b: usize| -> f32 {
            let id_a = layer.entries[idx_a].id as usize;
            let id_b = layer.entries[idx_b].id as usize;
            let data_a = data.get(id_a);
            let data_b = data.get(id_b);
            distance.distance(data_a, data_b)
        };

        let mut buffers = RNNConstructionBuffers::new();
        let initial_neighbors = if l == 0 {
            params.initial_neighbors
        } else {
            params.initial_neighbors / 2
        };
        crate::relative_nn_descent::construct_relative_nn_graph(
            RNNGraphParams {
                max_neighbors_after_reverse_pruning: layer.m_max,
                initial_neighbors,
                ..params
            },
            seed,
            &distance_idx_to_idx,
            &mut buffers,
            layer.entries.len(),
            false,
        );
        for (i, entry) in layer.entries.iter_mut().enumerate() {
            let rnn_neighbors = &buffers.neighbors[i];
            for nei in rnn_neighbors.iter() {
                entry.neighbors.insert_if_better(DistAnd(nei.dist, nei.idx));
            }
        }
    }
    let build_stats = Stats {
        duration: start_time.elapsed(),
        num_distance_calculations: distance.num_calculations(),
    };
    SliceHnsw {
        data,
        layers,
        params: HnswParams {
            level_norm_param,
            ef_construction: 0,
            m_max,
            m_max_0,
            distance: params.distance,
        },
        build_stats,
    }
}

// /////////////////////////////////////////////////////////////////////////////
// SECTION: alternative to rayon to process chunks in vp-tree ensemble in parallel
// /////////////////////////////////////////////////////////////////////////////

// std::thread::scope(|scope| {
//     let mut threads: Vec<ScopedJoinHandle<'_, ()>> = vec![];
//     for &chunk_chunk in chunk_chunks.iter() {
//         let handle = scope.spawn(|| {
//             let start = Instant::now();
//             for chunk in chunk_chunk.iter() {
//                 let tls = ensemble_tls();
//                 // let mut tls = EnsembleTLS::new(max_chunk_size, same_chunk_m_max);
//                 tls.init(max_chunk_size, same_chunk_m_max);
//                 let chunk_size = chunk.range.len();
//                 let chunk_dst_mat_idx = |i: usize, j: usize| i + j * chunk_size;
//                 let chunk_nodes = &vp_tree[chunk.range.clone()];
//                 assert!(chunk_size >= max_chunk_size / 2);
//                 assert!(chunk_size <= max_chunk_size);

//                 // calculate distances between all the nodes in this chunk. (clearing chunk_dst_mat not necessary, because everything relevant should be overwritten).
//                 for i in 0..chunk_size {
//                     for j in i + 1..chunk_size {
//                         let i_data = data.get(chunk_nodes[i].id as usize);
//                         let j_data = data.get(chunk_nodes[j].id as usize);
//                         let dist = distance.distance(i_data, j_data);
//                         tls.chunk_dst_mat[chunk_dst_mat_idx(i, j)] = dist;
//                         tls.chunk_dst_mat[chunk_dst_mat_idx(j, i)] = dist;
//                     }
//                 }
//                 // try to connect each node in the chunk to each other node (except itself) (limiting number of neighbors to same_chunk_m_max with the insert_if_better)
//                 for i in 0..chunk_size {
//                     // since all chunks are disjunct and each element is only in one chunk, this access should be fine and cannot lead to race conditions.
//                     let i_neighbors = &mut tls.chunk_neighbors[i];
//                     for j in 0..chunk_size {
//                         if j == i {
//                             continue;
//                         }
//                         let dist = tls.chunk_dst_mat[chunk_dst_mat_idx(i, j)];
//                         let j_idx_in_hnsw = chunk_nodes[j].idx_in_hnsw_layer as usize;
//                         i_neighbors.insert_if_better(DistAnd(dist, j_idx_in_hnsw));
//                     }
//                 }

//                 // merge the neighbors of the vp-tree in with the hnsw layer:

//                 for i in 0..chunk_size {
//                     let i_idx_in_hnsw = chunk_nodes[i].idx_in_hnsw_layer as usize;
//                     let i_neighbors = &mut tls.chunk_neighbors[i];
//                     let hnsw_neighbors = unsafe {
//                         &mut layer_cell.get_mut().entries[i_idx_in_hnsw].neighbors
//                     };
//                     for &DistAnd(dist, idx_in_hnsw) in i_neighbors.iter() {
//                         if !hnsw_neighbors.iter().any(|e| e.1 == idx_in_hnsw) {
//                             hnsw_neighbors.insert_if_better(DistAnd(dist, idx_in_hnsw));
//                         }
//                     }
//                 }
//             }
//             println!(
//                 "    Thread finished work for {} chunks in {}ms",
//                 chunk_chunk.len(),
//                 start.elapsed().as_secs_f32() * 1000.0
//             );
//         });

//         threads.push(handle);
//     }

//     for t in threads {
//         t.join().unwrap();
//     }
// });
