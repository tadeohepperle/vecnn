use std::{
    collections::HashSet,
    fmt::Display,
    fs::{File, OpenOptions},
    io::Write,
    sync::Arc,
};

use prettytable::{
    format::{FormatBuilder, LineSeparator, TableFormat},
    row, AsTableSlice, Slice, Table,
};
use rand::{seq::SliceRandom, thread_rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use vecnn::{
    dataset::{DatasetT, FlatDataSet},
    distance::{cos, dot, l2, Distance::*, DistanceFn},
    hnsw::{Hnsw, HnswParams},
    nn_descent::{RNNGraph, RNNGraphParams},
    transition::{
        build_hnsw_by_transition, build_hnsw_by_vp_tree_ensemble, EnsembleTransitionParams,
        StitchMode, TransitionParams,
    },
    utils::{linear_knn_search, random_data_set},
    vp_tree::Stats,
};

fn main() {
    let same_chunk_max_neighbors: usize = 20;
    let neg_fraction: f32 = 0.3;

    let n = 300000;
    let k = 30;
    let k_samples = 200;
    let mut models: Vec<ModelParams> = vec![];
    for ef_construction in [60, 120] {
        for n in [10000] {
            eval_models_on_laion(
                n,
                k_samples,
                &[ModelParams::Hnsw(HnswParams {
                    level_norm_param: 0.5,
                    ef_construction: ef_construction,
                    m_max: 20,
                    m_max_0: 40,
                    distance: Dot,
                })],
                SearchParams {
                    k,
                    truth_distance: dot,
                    start_candidates: 1,
                    ef: 100,
                },
            )
        }
    }
}

// ModelParams::Hnsw(HnswParams {
//     level_norm_param: 0.5,
//     ef_construction: 60,
//     m_max: 20,
//     m_max_0: 40,
//     distance: Dot,
// }),
// ModelParams::RNNGraph(RNNGraphParams {
//     distance: Dot,
//     outer_loops: 2,
//     inner_loops: 3,
//     max_neighbors_after_reverse_pruning: 20,
//     initial_neighbors: 30,
// }),
// ModelParams::Transition(TransitionParams {
//     max_chunk_size: 200,
//     same_chunk_max_neighbors,
//     neg_fraction,
//     distance: Dot,
//     keep_fraction: 0.2,
//     stop_after_stitching_n_chunks: None,
//     stitch_mode: StitchMode::RandomNegToRandomPosAndBack,
//     x: 5,
// }),
// ModelParams::EnsembleTransition(EnsembleTransitionParams {
//     max_chunk_size: 400,
//     distance: Dot,
//     num_vp_trees: 10,
//     max_neighbors_same_chunk: 20,
//     max_neighbors_hnsw: 40,
// }),
// ModelParams::RNNGraph(RNNGraphParams {
//     distance: Dot,
//     outer_loops: 2,
//     inner_loops: 3,
//     max_neighbors_after_reverse_pruning: 20-+,
//     initial_neighbors: 30,
// }),
#[derive(Debug, Clone, Copy)]
enum ModelParams {
    Hnsw(HnswParams),
    Transition(TransitionParams),
    EnsembleTransition(EnsembleTransitionParams),
    RNNGraph(RNNGraphParams),
}

impl ModelParams {
    pub fn to_string(&self) -> String {
        match self {
            ModelParams::Hnsw(e) => format!("{e:?}"),
            ModelParams::Transition(e) => format!("{e:?}"),
            ModelParams::EnsembleTransition(e) => format!("{e:?}"),
            ModelParams::RNNGraph(e) => format!("{e:?}"),
        }
    }
}

enum Model {
    Hnsw(Hnsw),
    HnswTransition(Hnsw),
    VpTreeEnsemble(Hnsw),
    RNNGraph(RNNGraph),
}

#[derive(Debug, Clone, Copy)]
struct SearchParams {
    truth_distance: DistanceFn,
    k: usize,
    start_candidates: usize, // for RNN graph
    ef: usize,               // for HNSW
}
impl Display for SearchParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{{k:{}, ef: {}, start_candidates: {}}}",
            self.k, self.ef, self.start_candidates
        ))
    }
}

impl Model {
    pub fn hnsw(data: Arc<dyn DatasetT>, params: HnswParams) -> Self {
        Self::Hnsw(Hnsw::new(data, params))
    }

    pub fn hnsw_transition(data: Arc<dyn DatasetT>, params: TransitionParams) -> Self {
        Self::HnswTransition(build_hnsw_by_transition(data, params))
    }

    pub fn vp_tree_ensemble_transition(
        data: Arc<dyn DatasetT>,
        params: EnsembleTransitionParams,
    ) -> Self {
        Self::VpTreeEnsemble(build_hnsw_by_vp_tree_ensemble(data, params))
    }

    pub fn rnn_graph(data: Arc<dyn DatasetT>, params: RNNGraphParams) -> Self {
        Self::RNNGraph(RNNGraph::new(data, params))
    }

    /// returns recall and search stats
    pub fn knn_search(
        &self,
        q_data: &[f32],
        true_res: &HashSet<usize>,
        search_params: SearchParams,
    ) -> (f64, Stats) {
        assert_eq!(search_params.k, true_res.len());
        let found: HashSet<usize>;
        let stats: Stats;
        match self {
            Model::Hnsw(hnsw) | Model::HnswTransition(hnsw) | Model::VpTreeEnsemble(hnsw) => {
                let res = hnsw.knn_search(q_data, search_params.k, search_params.ef);
                found = res
                    .0
                    .iter()
                    .map(|e| e.id as usize)
                    .collect::<HashSet<usize>>();
                stats = res.1
            }
            Model::RNNGraph(rnn_graph) => {
                let res =
                    rnn_graph.knn_search(q_data, search_params.k, search_params.start_candidates);
                found = res.0.iter().map(|e| e.1).collect::<HashSet<usize>>();
                stats = res.1;
            }
        };
        let recall_n = true_res.iter().filter(|i| found.contains(i)).count();
        let r = recall_n as f64 / search_params.k as f64;
        (r, stats)
    }

    pub fn build_stats(&self) -> Stats {
        match self {
            Model::Hnsw(e) | Model::HnswTransition(e) | Model::VpTreeEnsemble(e) => e.build_stats,
            Model::RNNGraph(e) => e.build_stats,
        }
    }
}

fn eval_models_on_laion(
    data_subsample_n: usize,
    queries_subsample_n: usize,
    params: &[ModelParams],
    search_params: SearchParams,
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
        assert!(subsample_n <= len);
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(bytes.len(), len * dims * std::mem::size_of::<f32>());

        let mut indices: Vec<usize> = (0..len).collect();
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        indices.shuffle(&mut rng);

        let mut flat_float_arr: Vec<f32> = vec![0.0; subsample_n * dims];
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
    eval_models(data, queries, params, search_params)
}

fn eval_models_random_data(
    n_data: usize,
    dims: usize,
    n_queries: usize,
    params: &[ModelParams],
    search_params: SearchParams,
) {
    let data = random_data_set(n_data, dims);
    let queries = random_data_set(n_queries, dims);
    eval_models(data, queries, params, search_params)
}

fn eval_models(
    data: Arc<dyn DatasetT>,
    queries: Arc<dyn DatasetT>,
    params: &[ModelParams],
    search_params: SearchParams,
) {
    let n_queries = queries.len();

    let mut true_knns: Vec<HashSet<usize>> = vec![];
    for i in 0..n_queries {
        let q_data = queries.get(i);
        let knn = linear_knn_search(
            &*data,
            q_data,
            search_params.k,
            search_params.truth_distance,
        )
        .iter()
        .map(|e| e.1)
        .collect::<HashSet<usize>>();
        true_knns.push(knn);
    }

    let mut models: Vec<Model> = vec![];
    for &param in params.iter() {
        let model = match param {
            ModelParams::Hnsw(params) => Model::hnsw(data.clone(), params),
            ModelParams::Transition(params) => Model::hnsw_transition(data.clone(), params),
            ModelParams::RNNGraph(params) => Model::rnn_graph(data.clone(), params),
            ModelParams::EnsembleTransition(params) => {
                Model::vp_tree_ensemble_transition(data.clone(), params)
            }
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
            let (r, s) = m.knn_search(q_data, &true_knns[i], search_params);
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
        "n",
        "search_params",
        "params",
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
            data.len().to_string(),
            search_params.to_string(),
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

    let file_path = "experiments.csv";
    if std::fs::File::open(file_path).is_err() {
        let mut file = File::create(file_path).unwrap();
        table.slice(0..1).to_csv(&mut file).unwrap();
    }
    let mut file = OpenOptions::new()
        .append(true) // Enable appending
        .open(file_path)
        .unwrap(); // Open the file
    table.slice(1..).to_csv(&mut file).unwrap();

    table.printstd();
}
