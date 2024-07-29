use std::{collections::HashSet, sync::Arc};

use vecnn::{
    distance::{self, cos, dot, l2, Distance},
    hnsw::{Hnsw, HnswParams},
    utils::{linear_knn_search, random_data_set},
};

fn main() {
    let dims = 2;
    let data = random_data_set(100000, dims);

    let distance = Distance::L2;

    let hnsw = Hnsw::new(
        data.clone(),
        HnswParams {
            level_norm_param: 0.5,
            ef_construction: 40,
            m_max: 20,
            m_max_0: 20,
            distance,
        },
    );

    println!(
        "Build HNSW.       time:{}       n-distance-calculations: {}",
        hnsw.build_stats.duration.as_secs_f32(),
        hnsw.build_stats.num_distance_calculations
    );

    let n_queries = 300;
    let queries = random_data_set(n_queries, dims);
    let k = 100;

    let mut recall = 0.0;
    for i in 0..n_queries {
        let q_data = queries.get(i);
        let true_res = linear_knn_search(&*data, q_data, k, distance.to_fn())
            .iter()
            .map(|e| e.1)
            .collect::<HashSet<usize>>();
        let s_res = hnsw
            .knn_search(q_data, k, 0)
            .0
            .iter()
            .map(|e| e.id as usize)
            .collect::<HashSet<usize>>();
        let f = true_res.iter().filter(|i| s_res.contains(i)).count();
        let r = f as f64 / s_res.len() as f64;
        print!("{r}, ");
        recall += r;
    }
    recall /= n_queries as f64;
    println!("\nOverall recall: {recall}");

    // std::fs::write(
    //     "./stats.txt",
    //     format!(
    //         "{}   {}",
    //         hnsw.build_stats.duration.as_secs_f32(),
    //         hnsw.build_stats.num_distance_calculations
    //     ),
    // );
}
