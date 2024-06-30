use std::{collections::HashSet, sync::Arc};

use prettytable::{
    format::{FormatBuilder, LineSeparator, TableFormat},
    row, Table,
};
use rand::{seq::SliceRandom, thread_rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use vecnn::{
    dataset::{DatasetT, FlatDataSet},
    distance::{cos, dot, l2, DistanceFn},
    hnsw::{Hnsw, HnswParams},
    nn_descent::{RNNGraph, RNNGraphParams},
    transition::{build_hnsw_by_transition, TransitionParams},
    utils::{linear_knn_search, random_data_set},
    vp_tree::Stats,
};

fn main() {
    let mut models: Vec<ModelParams> = vec![ModelParams::Hnsw(HnswParams {
        level_norm_param: 0.5,
        ef_construction: 40,
        m_max: 20,
        m_max_0: 20,
        distance_fn: dot,
    })];
    // for max_chunk_size in 60..99 {
    //     models.push(ModelParams::Transition(TransitionParams {
    //         max_chunk_size,
    //         same_chunk_max_neighbors: 40,
    //         neg_fraction: 0.2,
    //         distance_fn: dot,
    //     }))
    // }

    // eval_models_random_data(10000, 2, 200, 100, dot, &models)
    eval_models_on_laion(4000, 100, 100, dot, &models)
}

#[derive(Debug, Clone, Copy)]
enum ModelParams {
    Hnsw(HnswParams),
    Transition(TransitionParams),
    RNNGraph(RNNGraphParams),
}

impl ModelParams {
    pub fn to_string(&self) -> String {
        match self {
            ModelParams::Hnsw(e) => format!("{e:?}"),
            ModelParams::Transition(e) => format!("{e:?}"),
            ModelParams::RNNGraph(e) => format!("{e:?}"),
        }
    }
}

enum Model {
    Hnsw(Hnsw),
    HnswTransition(Hnsw),
    RNNGraph(RNNGraph),
}

impl Model {
    pub fn hnsw(data: Arc<dyn DatasetT>, params: HnswParams) -> Self {
        Self::Hnsw(Hnsw::new(data, params))
    }

    pub fn hnsw_transition(data: Arc<dyn DatasetT>, params: TransitionParams) -> Self {
        Self::HnswTransition(build_hnsw_by_transition(data, params))
    }

    pub fn rnn_graph(data: Arc<dyn DatasetT>, params: RNNGraphParams) -> Self {
        Self::RNNGraph(RNNGraph::new(data, params))
    }

    /// returns recall and search stats
    pub fn knn_search(
        &self,
        q_data: &[f32],
        true_res: &HashSet<usize>,
        start_candidates: Option<usize>,
    ) -> (f64, Stats) {
        let k = true_res.len();
        let found: HashSet<usize>;
        let stats: Stats;
        match self {
            Model::Hnsw(hnsw) | Model::HnswTransition(hnsw) => {
                let res = hnsw.knn_search(q_data, k);
                found = res
                    .0
                    .iter()
                    .map(|e| e.id as usize)
                    .collect::<HashSet<usize>>();
                stats = res.1
            }
            Model::RNNGraph(rnn_graph) => {
                let res = rnn_graph.knn_search(q_data, k, start_candidates.unwrap());
                found = res.0.iter().map(|e| e.i).collect::<HashSet<usize>>();
                stats = res.1;
            }
        };
        let recall_n = true_res.iter().filter(|i| found.contains(i)).count();
        let r = recall_n as f64 / k as f64;
        (r, stats)
    }

    pub fn build_stats(&self) -> Stats {
        match self {
            Model::Hnsw(e) => e.build_stats,
            Model::HnswTransition(e) => e.build_stats,
            Model::RNNGraph(e) => e.build_stats,
        }
    }
}

fn eval_models_on_laion(
    data_subsample_n: usize,
    queries_subsample_n: usize,
    k: usize,
    truth_distance: DistanceFn,
    params: &[ModelParams],
) {
    let data_path = "../vecnnpy/laion_data_(300000, 768).bin";
    let queries_path = "../vecnnpy/laion_queries_(10000, 768).bin";

    let data_len = 300000;
    let dims = 768;
    let queries_len = 10000;

    fn data_set_from_path(
        path: &str,
        len: usize,
        dims: usize,
        subsample_n: usize,
    ) -> Arc<dyn DatasetT> {
        assert!(subsample_n < len);
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(bytes.len(), len * dims * std::mem::size_of::<f32>());

        let mut indices: Vec<usize> = (0..len).collect();
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        indices.shuffle(&mut rng);

        let mut flat_float_arr: Vec<f32> = vec![0.0; subsample_n * dims];
        dbg!(indices[0]);
        for i in 0..subsample_n {
            let idx = indices[i];
            let slice = &bytes[dims * idx * std::mem::size_of::<f32>()
                ..dims * (idx + 1) * std::mem::size_of::<f32>()];
            unsafe {
                std::ptr::copy(
                    slice.as_ptr(),
                    &mut flat_float_arr[i * dims] as *mut f32 as *mut u8,
                    slice.len(),
                )
            }
        }
        Arc::new(FlatDataSet {
            dims,
            len: subsample_n,
            data: flat_float_arr,
        })
    }

    let data = data_set_from_path(data_path, data_len, dims, data_subsample_n);
    let queries = data_set_from_path(queries_path, queries_len, dims, queries_subsample_n);

    dbg!(&data.get(0)[0..10]);
    eval_models(data, queries, k, truth_distance, params)
}

fn eval_models_random_data(
    n_data: usize,
    dims: usize,
    n_queries: usize,
    k: usize,
    truth_distance: DistanceFn,
    params: &[ModelParams],
) {
    let data = random_data_set(n_data, dims);
    let queries = random_data_set(n_queries, dims);
    eval_models(data, queries, k, truth_distance, params)
}

fn eval_models(
    data: Arc<dyn DatasetT>,
    queries: Arc<dyn DatasetT>,
    k: usize,
    truth_distance: DistanceFn,
    params: &[ModelParams],
) {
    let n_queries = queries.len();

    let mut true_knns: Vec<HashSet<usize>> = vec![];
    for i in 0..n_queries {
        let q_data = queries.get(i);
        let knn = linear_knn_search(&*data, q_data, k, truth_distance)
            .iter()
            .map(|e| e.i)
            .collect::<HashSet<usize>>();
        true_knns.push(knn);
    }

    let mut models: Vec<Model> = vec![];
    for &param in params.iter() {
        let model = match param {
            ModelParams::Hnsw(params) => Model::hnsw(data.clone(), params),
            ModelParams::Transition(params) => Model::hnsw_transition(data.clone(), params),
            ModelParams::RNNGraph(params) => Model::rnn_graph(data.clone(), params),
        };
        models.push(model);
    }

    #[derive(Debug, Clone)]
    struct Results {
        recall_mean: f64,
        recall_std: f64,
        ndc_mean: f64,
        ndc_std: f64,
        time_ms_mean: f64,
        time_ms_std: f64,
    }
    let mut results: Vec<Results> = vec![];

    let mut recalls: Vec<f64> = Vec::with_capacity(n_queries);
    let mut ndcs: Vec<f64> = Vec::with_capacity(n_queries);
    let mut time_mss: Vec<f64> = Vec::with_capacity(n_queries);
    for m in models.iter() {
        recalls.clear();
        ndcs.clear();
        time_mss.clear();

        for i in 0..n_queries {
            let q_data = queries.get(i);
            let (r, s) = m.knn_search(q_data, &true_knns[i], Some(1));
            recalls.push(r);
            ndcs.push(s.num_distance_calculations as f64);
            time_mss.push(s.duration.as_secs_f64() * 1000.0);
        }

        let mut recall_mean: f64 = 0.0;
        let mut ndc_mean: f64 = 0.0;
        let mut time_ms_mean: f64 = 0.0;
        let mut recall_std: f64 = 0.0;
        let mut ndc_std: f64 = 0.0;
        let mut time_ms_std: f64 = 0.0;

        for i in 0..n_queries {
            recall_mean += recalls[i];
            ndc_mean += ndcs[i];
            time_ms_mean += time_mss[i];
        }
        recall_mean /= n_queries as f64;
        ndc_mean /= n_queries as f64;
        time_ms_mean /= n_queries as f64;

        for i in 0..n_queries {
            recall_std += (recalls[i] - recall_mean).powi(2);
            ndc_std += (ndcs[i] - ndc_mean).powi(2);
            time_ms_std += (time_mss[i] - time_ms_mean).powi(2);
        }

        recall_std = recall_std.sqrt();
        ndc_std = ndc_std.sqrt();
        time_ms_std = time_ms_std.sqrt();

        results.push(Results {
            recall_mean,
            recall_std,
            ndc_mean,
            ndc_std,
            time_ms_mean,
            time_ms_std,
        })
    }

    let mut table = Table::new();
    table.add_row(row![
        "Kind",
        "build_ndc",
        "build_ms",
        "recall_mean",
        // "recall_std",
        "ndc_mean",
        // "ndc_std",
        "time_ms_mean",
        // "time_ms_std",
    ]);
    for ((params, model), results) in params.iter().zip(models.iter()).zip(results.iter()) {
        let build_stats = model.build_stats();
        let build_ndc = build_stats.num_distance_calculations.to_string();
        let build_ms = (build_stats.duration.as_secs_f64() * 1000.0).to_string();
        table.add_row(row![
            params.to_string(),
            build_ndc,
            build_ms,
            format!("{:.3}", results.recall_mean),
            // format!("{:.3}", results.recall_std),
            format!("{:.3}", results.ndc_mean),
            // format!("{:.3}", results.ndc_std),
            format!("{:.3}", results.time_ms_mean),
            // format!("{:.3}", results.time_ms_std),
        ]);
    }
    table.set_format(FormatBuilder::new().padding(2, 2).build());
    table.printstd();
    // println!("{table}");
}
