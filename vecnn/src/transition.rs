use heapless::{binary_heap::Max, BinaryHeap};

use crate::{
    distance::{DistanceT, SquaredDiffSum},
    hnsw::{pick_level, DistAnd, Hnsw, HnswParams, Layer, LayerEntry},
    vp_tree::{Stats, VpTree},
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
