use std::{
    collections::{BinaryHeap, HashMap, HashSet},
    hash::Hash,
    ops::Range,
    sync::Arc,
    time::{Duration, Instant},
    usize,
};

use nanoserde::{DeJson, SerJson};
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::{
    dataset::DatasetT,
    distance::{l2, Distance, DistanceFn, DistanceTracker},
    hnsw::{
        self, pick_level, DistAnd, Hnsw, HnswParams, Layer, LayerEntry, Neighbors,
        NEIGHBORS_LIST_MAX_LEN,
    },
    if_tracking,
    slice_hnsw::{self, SliceHnsw, MAX_LAYERS},
    utils::{slice_binary_heap_arena, BinaryHeapExt, SliceBinaryHeap, Stats},
    vp_tree::{self, arrange_into_vp_tree, left, left_with_root, right, Node, StoresDistT, VpTree},
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
}

pub fn build_hnsw_by_vp_tree_ensemble(
    data: Arc<dyn DatasetT>,
    params: EnsembleParams,
) -> SliceHnsw {
    let mut hnsw_layer = slice_hnsw::Layer::new(params.m_max_0);
    hnsw_layer.entries_cap = data.len();
    hnsw_layer.allocate_neighbors_memory();
    for id in 0..data.len() {
        hnsw_layer.add_entry_assuming_allocated_memory(id, usize::MAX);
    }

    let max_chunk_size = params.max_chunk_size;
    let same_chunk_m_max = params.same_chunk_m_max;
    assert!(same_chunk_m_max <= NEIGHBORS_LIST_MAX_LEN);

    let mut distance = DistanceTracker::new(params.distance);
    let start = Instant::now();
    let mut rng = ChaCha20Rng::seed_from_u64(42);

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
        arrange_into_vp_tree(&mut vp_tree, &data_get, &mut distance, &mut rng);
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
) -> SliceHnsw {
    let start_time = Instant::now();
    let mut rng = ChaCha20Rng::seed_from_u64(42);
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
        if layer.entries.len() <= params.max_chunk_size * 2 {
            // brute force connect neighbors!
            fill_hnsw_layer_range_by_brute_force(&*data, &distance, layer);
        } else {
            fill_hnsw_layer_by_vp_tree_ensemble(&*data, &distance, layer, &params, rng.clone());
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

fn fill_hnsw_layer_by_vp_tree_ensemble(
    data: &dyn DatasetT,
    distance: &DistanceTracker,
    layer: &mut super::slice_hnsw::Layer,
    params: &EnsembleParams,
    mut rng: ChaCha20Rng,
) {
    struct VpTreeNode {
        id: u32, // u32 for better align
        idx_in_hnsw_layer: u32,
        dist: f32,
    }

    impl StoresDistT for VpTreeNode {
        #[inline(always)]
        fn dist(&self) -> f32 {
            self.dist
        }
        #[inline(always)]
        fn set_dist(&mut self, dist: f32) {
            self.dist = dist
        }
    }

    let mut vp_tree: Vec<VpTreeNode> = Vec::with_capacity(data.len());
    for (i, entry) in layer.entries.iter().enumerate() {
        vp_tree.push(VpTreeNode {
            id: entry.id as u32,
            idx_in_hnsw_layer: i as u32,
            dist: 0.0,
        });
    }
    let n_entries = layer.entries.len();
    let max_chunk_size = params.max_chunk_size;
    let same_chunk_m_max = params.same_chunk_m_max;
    let chunks = make_chunks(n_entries, max_chunk_size);
    let mut chunk_dst_mat: Vec<f32> = vec![0.0; max_chunk_size * max_chunk_size];
    let data_get = |e: &VpTreeNode| data.get(e.id as usize);

    let (_memory, mut vp_tree_neighbors) =
        slice_binary_heap_arena::<DistAnd<usize>>(n_entries, same_chunk_m_max);
    // Note: The vp_tree_neighbors are aligned with the nodes in the vp_tree (vp_tree[i] has neighbors stored in vp_tree_neighbors[i]),
    // but the indices stored in the neighbors lists, refer to indices of the hnsw layer!

    for _ in 0..params.n_vp_trees {
        arrange_into_vp_tree(&mut vp_tree, &data_get, distance, &mut rng);

        for (chunk_i, chunk) in chunks.iter().enumerate() {
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
                    chunk_dst_mat[chunk_dst_mat_idx(i, j)] = dist;
                    chunk_dst_mat[chunk_dst_mat_idx(j, i)] = dist;
                }
            }
            // connect each node in the chunk to each other node (except itself)
            for i in 0..chunk_size {
                let i_neighbors = &mut vp_tree_neighbors[chunk.range.start + i];
                for j in 0..chunk_size {
                    if j == i {
                        continue;
                    }
                    let dist = chunk_dst_mat[chunk_dst_mat_idx(i, j)];
                    let j_idx_in_hnsw = chunk_nodes[j].idx_in_hnsw_layer as usize;
                    i_neighbors.insert_if_better(DistAnd(dist, j_idx_in_hnsw));
                }

                if_tracking! {
                    Tracking.pt_meta(chunk_nodes[i].id as usize).chunk_on_level.push(chunk_i);
                }
            }
        }

        for (node, neighbors) in vp_tree.iter().zip(vp_tree_neighbors.iter_mut()) {
            let hnsw_entry = &mut layer.entries[node.idx_in_hnsw_layer as usize];
            for &DistAnd(dist, idx_in_hnsw) in neighbors.iter() {
                // awkward, but idk how to filter out duplicates mor efficiently
                if !hnsw_entry.neighbors.iter().any(|e| e.1 == idx_in_hnsw) {
                    hnsw_entry
                        .neighbors
                        .insert_if_better(DistAnd(dist, idx_in_hnsw));
                }
            }
            neighbors.clear();
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, SerJson, DeJson)]
pub struct TransitionParams {
    pub max_chunk_size: usize,
    pub same_chunk_max_neighbors: usize,
    pub neg_fraction: f32,
    pub keep_fraction: f32,
    pub x: usize,
    pub stop_after_stitching_n_chunks: Option<usize>,
    pub distance: Distance,
    pub stitch_mode: StitchMode,
}

pub fn build_hnsw_by_transition(data: Arc<dyn DatasetT>, params: TransitionParams) -> Hnsw {
    let max_chunk_size = params.max_chunk_size;
    let same_chunk_max_neighbors = params.same_chunk_max_neighbors;
    assert!(same_chunk_max_neighbors <= NEIGHBORS_LIST_MAX_LEN);
    let mut distance = DistanceTracker::new(params.distance);
    let start = Instant::now();
    let mut rng = ChaCha20Rng::seed_from_u64(42);

    let mut vp_tree: Vec<vp_tree::Node> = Vec::with_capacity(data.len());
    for idx in 0..data.len() {
        vp_tree.push(vp_tree::Node { id: idx, dist: 0.0 });
    }

    let data_get = |e: &vp_tree::Node| data.get(e.id);
    arrange_into_vp_tree(&mut vp_tree, &data_get, &mut distance, &mut rng);

    // create an hnsw layer with same order as vp-tree but no neighbors:
    let mut entries: Vec<hnsw::LayerEntry> = Vec::with_capacity(data.len());
    for node in vp_tree.iter() {
        entries.push(LayerEntry {
            id: node.id,
            lower_level_idx: usize::MAX,
            neighbors: Neighbors::new(),
        })
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
            let entry = &mut entries[chunk.range.start + i];

            if_tracking!(Tracking.pt_meta(entry.id).chunk = chunk_i);

            for j in 0..chunk_size {
                if j == i {
                    continue;
                }
                let neighbor_idx_in_layer = (chunk.range.start + j);
                let neighbor_dist = chunk_dst_mat[chunk_dst_mat_idx(i, j)];

                entry.neighbors.insert_if_better(
                    neighbor_idx_in_layer,
                    neighbor_dist,
                    same_chunk_max_neighbors,
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
            &mut entries,
            &params,
            &mut rng,
        );
        i += 1;
        if let Some(n) = params.stop_after_stitching_n_chunks {
            if i >= n {
                break;
            }
        }
        chunks.remove(neg_idx);
        chunks[pos_idx] = merged_chunk;
    }

    // insert a representative of each cluster (first element?)

    Hnsw {
        params: Default::default(),
        data,
        layers: vec![hnsw::Layer {
            level: 0,
            entries: entries,
        }],
        build_stats: Stats {
            num_distance_calculations: distance.num_calculations(),
            duration: start.elapsed(),
        },
    }
}

fn stitch_chunks(
    data: &dyn DatasetT,
    distance: &DistanceTracker,
    pos_chunk: &Chunk,
    neg_chunk: &Chunk,
    entries: &mut [LayerEntry],
    params: &TransitionParams,
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

    let stitch_candidates =
        generate_stitch_candidates(data, distance, pos_chunk, neg_chunk, entries, params, rng);

    for c in stitch_candidates.iter() {
        let pos_entry = &mut entries[c.pos_cand_idx];
        let pos_to_neg_inserted =
            pos_entry
                .neighbors
                .insert_if_better(c.neg_cand_idx, c.dist, NEIGHBORS_LIST_MAX_LEN); // todo! no simple push!!! check len of neighbors and if better

        let neg_entry = &mut entries[c.neg_cand_idx];
        let neg_to_pos_inserted =
            neg_entry
                .neighbors
                .insert_if_better(c.pos_cand_idx, c.dist, NEIGHBORS_LIST_MAX_LEN);

        if_tracking!(
            let pos_cand_id = entries[c.pos_cand_idx].id;
            let neg_cand_id = entries[c.neg_cand_idx].id;
            if pos_to_neg_inserted {
                Tracking
                    .edge_meta(pos_cand_id, neg_cand_id)
                    .is_pos_to_neg = true;
            }
            if neg_to_pos_inserted {
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
}

impl StitchMode {
    pub fn name(&self) -> &'static str {
        match self {
            StitchMode::RandomNegToPosCenterAndBack => "RandomNegToPosCenterAndBack",
            StitchMode::RandomNegToRandomPosAndBack => "RandomNegToRandomPosAndBack",
            StitchMode::RandomSubsetOfSubset => "RandomSubsetOfSubset",
            StitchMode::BestXofRandomXTimesX => "BestXofRandomXTimesX",
            StitchMode::DontStarveXXSearch => "DontStarveXXSearch",
        }
    }
}

fn generate_stitch_candidates(
    data: &dyn DatasetT,
    distance: &DistanceTracker,
    pos_chunk: &Chunk,
    neg_chunk: &Chunk,
    entries: &[LayerEntry],
    params: &TransitionParams,
    rng: &mut ChaCha20Rng,
) -> HashSet<StitchCandidate> {
    let mut result: HashSet<StitchCandidate> = HashSet::new();
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

            let mut neg_candidates = HashSet::<usize>::new();
            let neg_random_indixes =
                random_idx_sample(neg_chunk.range.clone(), params.neg_fraction, rng);
            for random_idx in neg_random_indixes {
                if_tracking!(
                    let random_id = entries[random_idx].id;
                    Tracking.pt_meta(random_id).is_neg_random = true;
                );
                let (neg_cand_idx, _) = greedy_search_in_range(
                    data,
                    distance,
                    entries,
                    neg_chunk.range.clone(),
                    random_idx,
                    pos_center_data,
                );

                if neg_candidates.insert(neg_cand_idx) {
                    let neg_cand_id = entries[neg_cand_idx].id;
                    if_tracking!(
                        Tracking.pt_meta(neg_cand_id).is_neg_cand = true;
                    );
                    let neg_cand_data = data.get(neg_cand_id);
                    let (pos_cand_idx, dist) = greedy_search_in_range(
                        data,
                        distance,
                        entries,
                        pos_chunk.range.clone(),
                        pos_center_idx,
                        neg_cand_data,
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
                let (neg_cand_idx, _) = greedy_search_in_range(
                    data,
                    distance,
                    entries,
                    neg_chunk.range.clone(),
                    neg_random_idx,
                    pos_random_data,
                );
                let neg_cand_id = entries[neg_cand_idx].id;
                let neg_cand_data = data.get(neg_cand_id);
                let (pos_cand_idx, dist) = greedy_search_in_range(
                    data,
                    distance,
                    entries,
                    pos_chunk.range.clone(),
                    pos_random_idx,
                    neg_cand_data,
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
            let x = params.x;
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
            let x = params.x;
            let outer_loops = neg_random_indices.len().min(pos_random_indices.len()) / x;

            neg_random_indices.truncate(outer_loops * x);
            pos_random_indices.truncate(outer_loops * x);

            let mut neg_search_results: Vec<usize> = vec![];

            for i in 0..outer_loops {
                neg_search_results.clear();
                // perform the XX search, it is like football players running towards each other, but they can starve or duplicate.

                let range = i * x..(i + 1) * x;
                let neg_indices = &neg_random_indices[range.clone()];
                let pos_indices = &pos_random_indices[range];

                // X^2 distance calculations (random in neg, random in pos), then X searches from neg to pos.
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
                for x_search in x_searches.iter() {
                    let pos_id = entries[x_search.pos_cand_idx].id;
                    let pos_data = data.get(pos_id);
                    let (neg_search_res_idx, dist) = greedy_search_in_range(
                        data,
                        distance,
                        entries,
                        neg_chunk.range.clone(),
                        x_search.neg_cand_idx,
                        pos_data,
                    );
                    neg_search_results.push(neg_search_res_idx)
                }

                // X^2 distance calculations (found in neg, random in pos), then X searches from pos to neg.
                x_searches.clear();
                for &neg_search_res_idx in neg_search_results.iter() {
                    let neg_id = entries[neg_search_res_idx].id;
                    let neg_data = data.get(neg_id);
                    for &pos_cand_idx in pos_indices.iter() {
                        let pos_id = entries[pos_cand_idx].id;
                        let pos_data = data.get(pos_id);
                        let dist = distance.distance(neg_data, pos_data);
                        x_searches.insert_if_better(
                            StitchCandidate {
                                pos_cand_idx,
                                neg_cand_idx: neg_search_res_idx,
                                dist,
                            },
                            x,
                        );
                    }
                }
                assert_eq!(x_searches.len(), x);
                for x_search in x_searches.iter() {
                    let neg_id = entries[x_search.neg_cand_idx].id;
                    let neg_data = data.get(neg_id);
                    let (pos_search_res_idx, dist) = greedy_search_in_range(
                        data,
                        distance,
                        entries,
                        pos_chunk.range.clone(),
                        x_search.pos_cand_idx,
                        neg_data,
                    );

                    result.insert(StitchCandidate {
                        pos_cand_idx: pos_search_res_idx,
                        neg_cand_idx: x_search.neg_cand_idx,
                        dist,
                    });
                }
            }
        }
    }
    result
}

/// returns the index in range that was found. This is an index into the entire layer entries.
/// Returns also the best distance.
fn greedy_search_in_range(
    data: &dyn DatasetT,
    distance: &DistanceTracker,
    entries: &[LayerEntry],
    range: Range<usize>,
    start_idx: usize,
    query: &[f32],
) -> (usize, f32) {
    assert!(start_idx >= range.start);
    assert!(start_idx < range.end);

    let mut visited_indices: HashSet<usize> = HashSet::new(); // to not calculate distances twice. // todo! reuse
    let mut best_idx = start_idx;
    let start_data = data.get(entries[best_idx].id);
    let mut best_dist = distance.distance(start_data, query);
    loop {
        let neighbors = &entries[best_idx].neighbors;
        let best_idx_before = best_idx;
        for n in neighbors.iter() {
            let n_idx = n.1;
            if !visited_indices.insert(n_idx) {
                continue;
            }
            let n_data = data.get(entries[n_idx].id);
            let n_dist = distance.distance(n_data, query);
            if n_dist < best_dist {
                best_dist = n_dist;
                best_idx = n_idx;
            }
        }
        if best_idx == best_idx_before {
            return (best_idx, best_dist);
        }
    }
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

// pub fn old_naive_vp_tree_to_hnsw(tree: &VpTree) -> Hnsw {
//     let n = tree.nodes.len();
//     let mut entries = Vec::<LayerEntry>::with_capacity(n);

//     let mut sum_dists = Vec::<f32>::with_capacity(n);

//     for i in 0..n {
//         let node = tree.nodes[i];
//         let id = node.id;
//         let id_data = tree.data.get(id);

//         let mut neighbors = Neighbors::new();
//         // vielleicht nodes die eine hohe mittlere Distanz haben auf hoheren levels einfugen.
//         let mut sum_dist = 0.0;
//         for j in 1..=20 {
//             let i2 = (i + j) % n;
//             let other_id = tree.nodes[i2].id;
//             let other_id_data = tree.data.get(other_id);
//             let dist = l2(id_data, other_id_data);
//             sum_dist += dist;
//             neighbors.insert_asserted(other_id, dist);
//         }
//         sum_dists.push(sum_dist);

//         entries.push(LayerEntry {
//             id: node.id,
//             lower_level_idx: usize::MAX,
//             neighbors,
//         });
//     }

//     let layers = vec![Layer { level: 0, entries }];
//     // connect nodes on the higher levels randomly:
//     Hnsw {
//         params: HnswParams {
//             level_norm_param: 0.0,
//             ef_construction: 40,
//             m_max: 40,
//             m_max_0: 40,
//             distance: Distance::L2,
//         },
//         data: tree.data.clone(),
//         layers,
//         build_stats: Stats::default(),
//     }
// }
