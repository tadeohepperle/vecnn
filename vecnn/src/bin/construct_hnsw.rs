use std::sync::Arc;

use vecnn::{
    hnsw::{Hnsw, HnswParams},
    utils::random_data_set,
};

fn main() {
    let data = random_data_set(1000, 768);

    let hnsw = Hnsw::new(
        data.clone(),
        HnswParams {
            level_norm_param: 0.7,
            ef_construction: 40,
            m_max: 20,
            m_max_0: 20,
        },
    );
    dbg!(hnsw.build_stats.duration.as_secs_f32());
    dbg!(hnsw.build_stats);
}
