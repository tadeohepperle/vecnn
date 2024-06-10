use std::sync::Arc;

use vecnn::{
    distance::l2,
    hnsw::{Hnsw, HnswParams},
    utils::random_data_set,
};

fn main() {
    let data = random_data_set(10000, 768);
    let hnsw = Hnsw::new(
        data.clone(),
        HnswParams {
            level_norm_param: 0.7,
            ef_construction: 40,
            m_max: 20,
            m_max_0: 20,
            distance_fn: l2,
        },
    );
    for _ in 0..10 {
        for i in 0..data.dims() {
            let q_data = data.get(i);
            hnsw.knn_search(q_data, 30);
        }
    }
}
