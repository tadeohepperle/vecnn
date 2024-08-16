use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{Rng, SeedableRng};
use vecnn::{
    distance::{dot, l2, Distance::*},
    hnsw::{Hnsw, HnswParams},
    utils::random_data_set,
};

pub fn criterion_benchmark(c: &mut Criterion) {
    let data = random_data_set(4000, 768);

    c.bench_function("build_hnsw", |b| {
        b.iter(|| {
            let hnsw = Hnsw::new(
                data.clone(),
                HnswParams {
                    level_norm_param: 0.7,
                    ef_construction: 20,
                    m_max: 10,
                    m_max_0: 10,
                    distance: L2,
                },
                42,
            );
        })
    });

    let mut rng = rand::thread_rng();
    // c.bench_function("distance", |b| {
    //     b.iter(|| {
    //         let i = rng.gen_range(0..data.len());
    //         let j = rng.gen_range(0..data.len());
    //         SquaredDiffSum::distance(data.get(i), data.get(j));
    //     })
    // });

    // c.bench_function("distance_simd", |b| {
    //     b.iter(|| {
    //         let i = rng.gen_range(0..data.len());
    //         let j = rng.gen_range(0..data.len());
    //         dot(data.get(i), data.get(j));
    //     })
    // });
    // c.bench_function("remove_duplicates", |b| {
    //     b.iter_custom(|iterations| {
    //         let mut total_time = Duration::default();
    //         for _ in 0..iterations {
    //             let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    //             let mut neighbors: Vec<vecnn::nn_descent::Neighbor> = vec![];
    //             for idx in 0..100000 {
    //                 for _ in 0..rng.gen_range(0..10) {
    //                     neighbors.push(vecnn::nn_descent::Neighbor {
    //                         idx,
    //                         dist: rng.gen(),
    //                         is_new: true,
    //                     })
    //                 }
    //             }

    //             let start = Instant::now();
    //             remove_duplicates_for_sorted(&mut neighbors);
    //             total_time += start.elapsed();
    //         }
    //         total_time
    //     })
    // });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
