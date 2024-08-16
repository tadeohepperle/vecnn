use std::{collections::HashSet, sync::Arc, time::Instant};

use vecnn::{
    const_hnsw::ConstHnsw,
    distance::{self, cos, dot, l2, Distance},
    hnsw::{Hnsw, HnswParams},
    slice_hnsw::SliceHnsw,
    transition::{build_hnsw_by_vp_tree_ensemble_multi_layer, EnsembleParams, EnsembleStrategy},
    utils::{linear_knn_search, random_data_set},
};

fn main() {
    let dims = 768;
    let data = random_data_set(120000, dims);

    let distance = Distance::L2;

    // let hnsw = vecnn::slice_hnsw_par::SliceHnsw::new(
    //     data.clone(),
    //     HnswParams {
    //         level_norm_param: 0.8,
    //         ef_construction: 40,
    //         m_max: 20,
    //         m_max_0: 40,
    //         distance,
    //     },
    //     2342,
    // );

    let start = Instant::now();
    let hnsw = build_hnsw_by_vp_tree_ensemble_multi_layer(
        data,
        EnsembleParams {
            n_vp_trees: 3,
            max_chunk_size: 512,
            same_chunk_m_max: 20,
            m_max: 30,
            m_max_0: 40,
            level_norm: 0.0,
            distance,
            strategy: EnsembleStrategy::BruteForce,
        },
        true,
        210,
    );
    println!("Total time: {}ms", start.elapsed().as_secs_f32() * 1000.0);

    // println!(
    //     "Build HNSW.       time:{}       n-distance-calculations: {}",
    //     hnsw.build_stats.duration.as_secs_f32(),
    //     hnsw.build_stats.num_distance_calculations
    // );

    // let n_queries = 300;
    // let queries = random_data_set(n_queries, dims);
    // let k = 100;

    // let mut recall = 0.0;
    // for i in 0..n_queries {
    //     let q_data = queries.get(i);
    //     let true_res = linear_knn_search(&*data, q_data, k, distance.to_fn())
    //         .iter()
    //         .map(|e| e.1)
    //         .collect::<HashSet<usize>>();
    //     let s_res = hnsw
    //         .knn_search(q_data, k, 0)
    //         .0
    //         .iter()
    //         .map(|e| e.id as usize)
    //         .collect::<HashSet<usize>>();
    //     let f = true_res.iter().filter(|i| s_res.contains(i)).count();
    //     let r = f as f64 / s_res.len() as f64;
    //     print!("{r}, ");
    //     recall += r;
    // }
    // recall /= n_queries as f64;
    // println!("\nOverall recall: {recall}");

    // std::fs::write(
    //     "./stats.txt",
    //     format!(
    //         "{}   {}",
    //         hnsw.build_stats.duration.as_secs_f32(),
    //         hnsw.build_stats.num_distance_calculations
    //     ),
    // );
}
