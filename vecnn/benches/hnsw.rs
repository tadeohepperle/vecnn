use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use vecnn::{
    distance::{dot, l2, Distance::*},
    hnsw::{Hnsw, HnswParams},
    utils::random_data_set,
};

pub fn criterion_benchmark(c: &mut Criterion) {
    let data = random_data_set(124, 768);

    // c.bench_function("build_hnsw", |b| {
    //     b.iter(|| {
    //         let hnsw = Hnsw::new(
    //             data.clone(),
    //             HnswParams {
    //                 level_norm_param: 0.7,
    //                 ef_construction: 20,
    //                 m_max: 10,
    //                 m_max_0: 10,
    //                 distance: L2,
    //             },
    //         );
    //     })
    // });

    let mut rng = rand::thread_rng();
    // c.bench_function("distance", |b| {
    //     b.iter(|| {
    //         let i = rng.gen_range(0..data.len());
    //         let j = rng.gen_range(0..data.len());
    //         SquaredDiffSum::distance(data.get(i), data.get(j));
    //     })
    // });

    c.bench_function("distance_simd", |b| {
        b.iter(|| {
            let i = rng.gen_range(0..data.len());
            let j = rng.gen_range(0..data.len());
            dot(data.get(i), data.get(j));
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
