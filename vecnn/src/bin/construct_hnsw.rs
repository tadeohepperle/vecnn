use std::sync::Arc;

use vecnn::{
    distance::{cos, cos_for_spherical, l2},
    hnsw::{Hnsw, HnswParams},
    utils::random_data_set,
};

fn main() {
    let data = random_data_set(4000, 768);

    let hnsw = Hnsw::new(
        data.clone(),
        HnswParams {
            level_norm_param: 0.7,
            ef_construction: 40,
            m_max: 20,
            m_max_0: 20,
            distance_fn: cos_for_spherical,
        },
    );

    println!(
        "{}   {}",
        hnsw.build_stats.duration.as_secs_f32(),
        hnsw.build_stats.num_distance_calculations
    )
    // std::fs::write(
    //     "./stats.txt",
    //     format!(
    //         "{}   {}",
    //         hnsw.build_stats.duration.as_secs_f32(),
    //         hnsw.build_stats.num_distance_calculations
    //     ),
    // );
}
