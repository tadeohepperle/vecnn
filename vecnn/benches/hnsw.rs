use criterion::{black_box, criterion_group, criterion_main, Criterion};
use vecnn::{
    hnsw::{Hnsw, HnswParams},
    utils::random_data_set,
};

pub fn criterion_benchmark(c: &mut Criterion) {
    let data = random_data_set::<768>(500);

    c.bench_function("build_hnsw", |b| {
        b.iter(|| {
            let hnsw = Hnsw::new(
                data.clone(),
                HnswParams {
                    level_norm_param: 0.7,
                    ef_construction: 20,
                    m_max: 10,
                    m_max_0: 10,
                },
            );
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
