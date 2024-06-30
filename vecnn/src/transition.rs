use std::{
    collections::HashSet,
    ops::Range,
    sync::Arc,
    time::{Duration, Instant},
};

use heapless::{binary_heap::Max, BinaryHeap};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::{
    dataset::DatasetT,
    distance::{l2, DistanceFn, DistanceTracker},
    hnsw::{
        self, pick_level, Hnsw, HnswParams, IAndDist, Layer, LayerEntry, Neighbors,
        NEIGHBORS_LIST_MAX_LEN,
    },
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
            neighbors.insert_asserted(other_id as u32, dist);
        }
        sum_dists.push(sum_dist);

        entries.push(LayerEntry {
            id: node.idx as u32,
            lower_level_idx: u32::MAX,
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
            distance_fn: l2,
        },
        data: tree.data.clone(),
        layers,
        build_stats: Stats::default(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransitionParams {
    pub max_chunk_size: usize,
    pub same_chunk_max_neighbors: usize,
    pub neg_fraction: f32,
    pub distance_fn: DistanceFn,
}

pub fn build_hnsw_by_transition(data: Arc<dyn DatasetT>, params: TransitionParams) -> Hnsw {
    let max_chunk_size = params.max_chunk_size;
    let same_chunk_max_neighbors = params.same_chunk_max_neighbors;
    assert!(same_chunk_max_neighbors <= NEIGHBORS_LIST_MAX_LEN);

    let mut distance = DistanceTracker::new(params.distance_fn);

    let start = Instant::now();

    let mut vp_tree: Vec<vp_tree::Node> = Vec::with_capacity(data.len());
    for idx in 0..data.len() {
        vp_tree.push(vp_tree::Node { idx, dist: 0.0 });
    }
    arrange_into_vp_tree(&mut vp_tree, &*data, &mut distance);

    let mut rng = ChaCha20Rng::seed_from_u64(42);

    let mut entries: Vec<hnsw::LayerEntry> = Vec::with_capacity(data.len());
    let mut chunks = make_chunks(vp_tree.len(), max_chunk_size);
    let mut chunk_dst_mat: Vec<f32> = vec![0.0; max_chunk_size * max_chunk_size]; // memory reused for each chunk.
    for chunk in chunks.iter() {
        let chunk_size = chunk.range.len();
        let chunk_dst_mat_idx = |i: usize, j: usize| i + j * chunk_size;
        let chunk_nodes = &vp_tree[chunk.range.clone()];

        assert!(chunk_size >= max_chunk_size / 2);
        assert!(chunk_size <= max_chunk_size);
        for i in chunk.range.clone() {
            entries.push(LayerEntry {
                id: vp_tree[i].idx as u32,
                lower_level_idx: u32::MAX,
                neighbors: Neighbors::new(),
            });
        }

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
            for j in 0..chunk_size {
                if j == i {
                    continue;
                }
                let neighbor_idx_in_layer = (chunk.range.start + j) as u32;
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
    while chunks.len() > 1 {
        let pos_idx = chunk_idx_of_pos_half_to_stitch(&chunks);
        let neg_idx = pos_idx + 1;
        let merged_chunk = stitch_chunks(
            &*data,
            &distance,
            &chunks[pos_idx],
            &chunks[neg_idx],
            &mut entries,
            params.neg_fraction,
            &mut rng,
        );
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
    neg_fraction: f32,
    rng: &mut ChaCha20Rng,
) -> Chunk {
    let len_diff = neg_chunk.range.len() - pos_chunk.range.len();
    assert!(len_diff == 1 || len_diff == 0); // todo: this assertion sometimes fails, figure out why
    assert_eq!(pos_chunk.level, neg_chunk.level);
    assert_eq!(pos_chunk.range.end, neg_chunk.range.start);

    let pos_center_idx = pos_chunk.range.start;
    let pos_center_data = data.get(entries[pos_center_idx].id as usize);

    let max_candidate_count = (neg_fraction * neg_chunk.len() as f32) as usize;
    let mut searched_from_neg = HashSet::<usize>::new();
    let mut neg_candidates = HashSet::<usize>::new();

    for _ in 0..max_candidate_count {
        let random_idx = loop {
            let idx = rng.gen_range(neg_chunk.range.clone());
            if searched_from_neg.insert(idx) {
                break idx;
            }
        };
        let (closest_to_center, _) = greedy_search_in_range(
            data,
            distance,
            entries,
            neg_chunk.range.clone(),
            random_idx,
            pos_center_data,
        );
        neg_candidates.insert(closest_to_center);
    }
    for neg_cand_idx in neg_candidates {
        let neg_cand_data = data.get(entries[neg_cand_idx].id as usize);
        let (pos_cand_idx, dist) = greedy_search_in_range(
            data,
            distance,
            entries,
            pos_chunk.range.clone(),
            pos_center_idx,
            neg_cand_data,
        );

        // stitch neg_cand_idx and pos_cand_idx together:
        let pos_entry = &mut entries[pos_cand_idx];
        _ = pos_entry
            .neighbors
            .insert_if_better(neg_cand_idx as u32, dist, NEIGHBORS_LIST_MAX_LEN); // todo! no simple push!!! check len of neighbors and if better
        let neg_entry = &mut entries[neg_cand_idx];
        _ = neg_entry
            .neighbors
            .insert_if_better(pos_cand_idx as u32, dist, NEIGHBORS_LIST_MAX_LEN);
        // todo! no simple push!!! check len of neighbors and if better
    }

    Chunk {
        range: pos_chunk.range.start..neg_chunk.range.end,
        level: pos_chunk.level + 1,
    }
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
    let start_data = data.get(entries[best_idx].id as usize);
    let mut best_dist = distance.distance(start_data, query);
    loop {
        let neighbors = &entries[best_idx].neighbors;
        let best_idx_before = best_idx;
        for n in neighbors.iter() {
            let n_idx = n.i as usize;
            if !visited_indices.insert(n_idx) {
                continue;
            }
            let n_data = data.get(entries[n_idx].id as usize);
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
    range: Range<usize>,
    level: usize,
}

impl Chunk {
    pub fn len(&self) -> usize {
        self.range.end - self.range.start
    }
}

fn make_chunks(data_len: usize, max_chunk_size: usize) -> Vec<Chunk> {
    let mut chunks: Vec<Chunk> = Vec::with_capacity(data_len / (max_chunk_size / 2));
    fn collect_chunks(
        range: Range<usize>,
        level: usize,
        max_chunk_size: usize,
        chunks: &mut Vec<Chunk>,
    ) {
        let len = range.end - range.start;
        if len <= max_chunk_size {
            chunks.push(Chunk { range, level });
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
            collect_chunks(left, level + 1, max_chunk_size, chunks);
            collect_chunks(right, level + 1, max_chunk_size, chunks);
        }
    }
    collect_chunks(0..data_len, 0, max_chunk_size, &mut chunks);
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
