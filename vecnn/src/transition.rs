use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    ops::Range,
    sync::Arc,
    time::{Duration, Instant},
};

use heapless::{binary_heap::Max, BinaryHeap};
use nanoserde::{DeJson, SerJson};
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::{
    dataset::DatasetT,
    distance::{l2, Distance, DistanceFn, DistanceTracker},
    hnsw::{
        self, pick_level, Hnsw, HnswParams, IAndDist, Layer, LayerEntry, Neighbors,
        NEIGHBORS_LIST_MAX_LEN,
    },
    if_tracking,
    vp_tree::{self, arrange_into_vp_tree, left, left_with_root, right, Node, Stats, VpTree},
};

/// we can just create the hnsw by setting just one lowest level.
pub fn vp_tree_to_hnsw(tree: &VpTree) -> Hnsw {
    let n = tree.nodes.len();
    let mut entries = Vec::<LayerEntry>::with_capacity(n);

    let mut sum_dists = Vec::<f32>::with_capacity(n);

    for i in 0..n {
        let node = tree.nodes[i];
        let id = node.idx;
        let id_data = tree.data.get(id);

        let mut neighbors = Neighbors::new();
        // vielleicht nodes die eine hohe mittlere Distanz haben auf hoheren levels einfugen.
        let mut sum_dist = 0.0;
        for j in 1..=20 {
            let i2 = (i + j) % n;
            let other_id = tree.nodes[i2].idx;
            let other_id_data = tree.data.get(other_id);
            let dist = l2(id_data, other_id_data);
            sum_dist += dist;
            neighbors.insert_asserted(other_id, dist);
        }
        sum_dists.push(sum_dist);

        entries.push(LayerEntry {
            id: node.idx,
            lower_level_idx: usize::MAX,
            neighbors,
        });
    }

    let layers = vec![Layer { level: 0, entries }];

    // let avg_sum_dist = sum_dists.iter().copied().sum::<f32>() / sum_dists.len() as f32;

    // // insert also at higher levels:
    // for i in 0..n {
    //     let i_norm = sum_dists[i] / avg_sum_dist;
    //     let highest_level = pick_level(i_norm);

    //     let mut i_in_lower = i;
    //     for l in 1..=highest_level {
    //         if l >= layers.len() {
    //             layers.push(Layer {
    //                 level: l,
    //                 entries: vec![],
    //             })
    //         }
    //         let entries = &mut layers[l].entries;
    //         // entries.push(value)
    //     }
    // }

    // connect nodes on the higher levels randomly:

    Hnsw {
        params: HnswParams {
            level_norm_param: 0.0,
            ef_construction: 40,
            m_max: 40,
            m_max_0: 40,
            distance: Distance::L2,
        },
        data: tree.data.clone(),
        layers,
        build_stats: Stats::default(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, SerJson, DeJson)]
pub struct TransitionParams {
    pub max_chunk_size: usize,
    pub same_chunk_max_neighbors: usize,
    pub neg_fraction: f32,
    pub distance: Distance,
    pub stitch_mode: StitchMode,
}

#[derive(Debug, Clone, Copy, PartialEq, SerJson, DeJson)]
pub enum StitchMode {
    /// - select `neg_fraction` random points in negative half.
    /// - search from each of them in neg half towards center of positive half -> neg candidates
    /// - for each neg candidate: search in pos half from center of positive half towards neg candidate -> pos candidate
    /// - connect pos and neg candidates if the connection would be good.
    RandomNegToPosCenterAndBack,
    RandomNegToRandomPosAndBack,
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
        vp_tree.push(vp_tree::Node { idx, dist: 0.0 });
    }
    arrange_into_vp_tree(&mut vp_tree, &*data, &mut distance);

    // create an hnsw layer with same order as vp-tree but no neighbors:
    let mut entries: Vec<hnsw::LayerEntry> = Vec::with_capacity(data.len());
    for node in vp_tree.iter() {
        entries.push(LayerEntry {
            id: node.idx,
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
                let i_data = data.get(chunk_nodes[i].idx);
                let j_data = data.get(chunk_nodes[j].idx);
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
        if i >= 4 {
            break;
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
        generate_stitch_pair_candidates(data, distance, pos_chunk, neg_chunk, entries, params, rng);

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
            println!("    potential connection: pos:{pos_cand_id} - neg:{neg_cand_id}  (pos_to_neg_inserted={pos_to_neg_inserted}, neg_to_pos_inserted={neg_to_pos_inserted})")
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

fn generate_stitch_pair_candidates(
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
                Tracking.pt_meta(pos_center_id).is_pos_center = true;
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
            let n_idx = n.i;
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
        .map(|e| Node { idx: e, dist: 0.0 })
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
