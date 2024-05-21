use std::sync::Arc;

use vecnn::{
    hnsw::{Hnsw, HnswParams},
    utils::random_data_set,
};

fn main() {
    let data = random_data_set::<600>(2000);
    for i in 0..500 {
        let hnsw = Hnsw::new(
            data.clone(),
            HnswParams {
                level_norm_param: 0.7,
                ef_construction: 20,
                m_max: 10,
                m_max_0: 10,
            },
        );
    }
}
