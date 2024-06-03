use std::{ops::Range, sync::Arc};

use heapless::{binary_heap::Max, BinaryHeap};

use crate::{
    dataset::DatasetT,
    distance::{DistanceT, DistanceTracker, SquaredDiffSum},
    hnsw::{self, pick_level, DistAnd, Hnsw, HnswParams, Layer, LayerEntry},
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

        let mut neighbors: BinaryHeap<DistAnd<u32>, Max, 40> = Default::default();
        // vielleicht nodes die eine hohe mittlere Distanz haben auf hoheren levels einfugen.
        let mut sum_dist = 0.0;
        for j in 1..=20 {
            let i2 = (i + j) % n;
            let other_id = tree.nodes[i2].idx;
            let other_id_data = tree.data.get(other_id);
            let dist = SquaredDiffSum::distance(id_data, other_id_data);
            sum_dist += dist;
            neighbors
                .push(DistAnd {
                    dist,
                    i: other_id as u32,
                })
                .unwrap();
        }
        sum_dists.push(sum_dist);

        entries.push(LayerEntry {
            id: node.idx as u32,
            lower_level_idx: u32::MAX,
            neighbors,
        });
    }

    let mut layers = vec![Layer { level: 0, entries }];

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
        },
        data: tree.data.clone(),
        layers,
        build_stats: Stats::default(),
    }
}

pub fn build_hnsw_alternative(data: Arc<dyn DatasetT>, max_chunk_size: usize) -> Hnsw {
    let mut tracker = DistanceTracker::new(SquaredDiffSum::distance);

    let mut vp_tree: Vec<vp_tree::Node> = Vec::with_capacity(data.len());
    for idx in 0..data.len() {
        vp_tree.push(vp_tree::Node { idx, dist: 0.0 });
    }

    arrange_into_vp_tree(&mut vp_tree, &*data, &mut tracker);

    let mut layer_entries: Vec<hnsw::LayerEntry> = Vec::with_capacity(data.len());

    // let max_chunk_size: usize = 64;
    let chunk_ranges = make_chunk_ranges(vp_tree.len(), max_chunk_size);

    let mut chunk_dst_mat: Vec<f32> = vec![0.0; max_chunk_size * max_chunk_size];

    // each chunk should now
    for range in chunk_ranges.iter() {
        println!("chunk range: {range:?}");
        let chunk_size = range.len();
        let chunk_dst_mat_idx = |i: usize, j: usize| i + j * chunk_size;
        let chunk_nodes = &vp_tree[range.clone()];

        assert!(chunk_size >= max_chunk_size / 2);
        assert!(chunk_size <= max_chunk_size);

        let layer_idx_first = layer_entries.len();
        for i in range.clone() {
            let node = &vp_tree[i]; //unsafe { vp_tree.get_unchecked(i) };
            layer_entries.push(LayerEntry {
                id: node.idx as u32,
                lower_level_idx: u32::MAX,
                neighbors: BinaryHeap::new(),
            });
        }

        // calculate distances between all the nodes in this chunk. (clearing chunk_dst_mat not necessary, because everything relevant should be overwritten).
        for i in 0..chunk_size {
            chunk_dst_mat[chunk_dst_mat_idx(i, i)] = 0.0;
            for j in i + 1..chunk_size {
                let i_data = data.get(chunk_nodes[i].idx);
                let j_data = data.get(chunk_nodes[j].idx);
                let dist = tracker.distance(i_data, j_data);
                chunk_dst_mat[chunk_dst_mat_idx(i, j)] = dist;
                chunk_dst_mat[chunk_dst_mat_idx(j, i)] = dist;
            }
        }

        // connect each node in the chunk to each other node (except itself)
        for i in 0..chunk_size {
            let entry = &mut layer_entries[layer_idx_first + i];
            for j in 0..chunk_size {
                if j == i {
                    continue;
                }
                let neighbor = DistAnd {
                    dist: chunk_dst_mat[chunk_dst_mat_idx(i, j)],
                    i: (layer_idx_first + j) as u32,
                };
                let peek_mut = entry.neighbors.peek_mut();
                if let Some(mut top) = peek_mut {
                    if neighbor.dist < top.dist {
                        *top = neighbor;
                    }
                } else {
                    drop(peek_mut);
                    entry.neighbors.push(neighbor).unwrap();
                }
            }
        }
    }

    // now we need to decide how to

    // insert each chunk into an hnsw layer:

    Hnsw {
        params: Default::default(),
        data,
        layers: vec![hnsw::Layer {
            level: 0,
            entries: layer_entries,
        }],
        build_stats: Default::default(),
    }
}

fn make_chunk_ranges(data_len: usize, max_chunk_size: usize) -> Vec<Range<usize>> {
    let mut chunks: Vec<Range<usize>> = Vec::with_capacity(data_len / (max_chunk_size / 2));
    fn collect_chunks(range: Range<usize>, max_chunk_size: usize, chunks: &mut Vec<Range<usize>>) {
        let len = range.end - range.start;
        if len <= max_chunk_size {
            chunks.push(range);
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
            collect_chunks(left, max_chunk_size, chunks);
            collect_chunks(right, max_chunk_size, chunks);
        }
    }
    collect_chunks(0..data_len, max_chunk_size, &mut chunks);
    chunks
}

#[cfg(test)]
#[test]
fn chunk_collection_matches_vp_tree_subtrees() {
    let nodes: Vec<Node> = (0..37)
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
    let ranges = make_chunk_ranges(37, 4);
    assert_eq!(ranges.len(), slices.len());
    for (r, s) in ranges.iter().zip(slices.iter()) {
        assert_eq!(r.len(), s.len())
    }
}
