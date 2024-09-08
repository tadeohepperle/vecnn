use std::{
    collections::HashSet,
    fmt::Display,
    fs::{File, OpenOptions},
    hash::Hash,
    io::Write,
    ops::{Add, AddAssign},
    sync::Arc,
};

use ahash::{HashMap, HashMapExt};
use prettytable::{
    format::{FormatBuilder, LineSeparator, TableFormat},
    row, AsTableSlice, Slice, Table,
};
use rand::{seq::SliceRandom, thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use vecnn::{
    const_hnsw::ConstHnsw,
    dataset::{DatasetT, FlatDataSet},
    distance::{
        cos, dot, l2,
        Distance::{self, *},
        DistanceFn,
    },
    hnsw::{self, Hnsw, HnswParams},
    relative_nn_descent::{RNNGraph, RNNGraphParams},
    slice_hnsw::SliceHnsw,
    transition::{
        build_hnsw_by_rnn_descent, build_hnsw_by_vp_tree_ensemble_multi_layer,
        build_hnsw_by_vp_tree_stitching, build_single_layer_hnsw_by_vp_tree_ensemble,
        EnsembleParams, EnsembleStrategy, StitchMode, StitchingParams,
    },
    utils::{linear_knn_search, random_data_set, Stats},
    vp_tree::{VpTree, VpTreeParams},
};
use HnswStructure::*;

const DATA_PATH: &str = "../eval/laion_data_(300000, 768).bin";
const DATA_LEN: usize = 300000;

const QUERIES_PATH: &str = "../eval/laion_queries_(10000, 768).bin";
const QUERIES_LEN: usize = 10000;

const DIMS: usize = 768;
const RNG_SEED: u64 = 21321424198;

fn main() {
    let ensemble_params = EnsembleParams {
        n_vp_trees: 6,
        max_chunk_size: 256,
        same_chunk_m_max: 16,
        m_max: 20,
        m_max_0: 40,
        distance: Distance::Dot,
        level_norm: 0.0,
        strategy: EnsembleStrategy::BruteForceKNN,
    };

    let hnsw_params = HnswParams {
        level_norm_param: 0.5,
        ef_construction: 40,
        m_max: 20,
        m_max_0: 40,
        distance: Dot,
    };

    let rnn_params = RNNGraphParams {
        distance: Dot,
        outer_loops: 2,
        inner_loops: 3,
        max_neighbors_after_reverse_pruning: 20,
        initial_neighbors: 40,
    };

    let stitch_params = StitchingParams {
        max_chunk_size: 128,
        same_chunk_m_max: 20,
        neg_fraction: 0.4,
        keep_fraction: 0.0,
        x: 2,
        only_n_chunks: None,
        distance: Distance::Dot,
        stitch_mode: StitchMode::BestXofRandomXTimesX,
        m_max: 20,
    };

    let test_setup = ExperimentSetup {
        n: 100000,
        n_queries: 100,
        params: vec![
            ModelParams::Hnsw(
                HnswParams {
                    level_norm_param: 0.3,
                    ef_construction: 20,
                    m_max: 8,
                    m_max_0: 16,
                    distance: Dot,
                },
                SliceS2,
            ),
            ModelParams::Hnsw(
                HnswParams {
                    level_norm_param: 0.3,
                    ef_construction: 20,
                    m_max: 8,
                    m_max_0: 16,
                    distance: Dot,
                },
                SliceParralelRayon,
            ),
            // ModelParams::RNNGraph(RNNGraphParams {
            //     distance: Dot,
            //     outer_loops: 2,
            //     inner_loops: 3,
            //     max_neighbors_after_reverse_pruning: 10,
            //     initial_neighbors: 12,
            // }),
            // ModelParams::Stitching(StitchingParams {
            //     stitch_mode: StitchMode::RandomNegToRandomPosAndBack,
            //     ..stitch_params
            // }),
            // ModelParams::Stitching(StitchingParams {
            //     stitch_mode: StitchMode::DontStarveXXSearch,
            //     ..stitch_params
            // }),
            ModelParams::VpTreeEnsemble(
                EnsembleParams {
                    n_vp_trees: 3,
                    max_chunk_size: 256,
                    same_chunk_m_max: 16,
                    m_max: 16,
                    m_max_0: 16,
                    distance: Distance::Dot,
                    level_norm: 0.0,
                    strategy: EnsembleStrategy::BruteForceKNN,
                },
                true,
            ),
            ModelParams::VpTreeEnsemble(
                EnsembleParams {
                    n_vp_trees: 3,
                    max_chunk_size: 256,
                    same_chunk_m_max: 16,
                    m_max: 16,
                    m_max_0: 16,
                    distance: Distance::Dot,
                    level_norm: 0.0,
                    strategy: EnsembleStrategy::BruteForceKNN,
                },
                false,
            ),
            // ModelParams::Stitching(StitchingParams {
            //     stitch_mode: StitchMode::RandomNegToPosCenterAndBack,
            //     ..stitch_params
            // }),
            // ModelParams::Stitching(StitchingParams {
            //     stitch_mode: StitchMode::RandomSubsetOfSubset,
            //     ..stitch_params
            // }),
            // ModelParams::Stitching(StitchingParams {
            //     stitch_mode: StitchMode::BestXofRandomXTimesX,
            //     ..stitch_params
            // }),

            // ModelParams::Hnsw(
            //     HnswParams {
            //         level_norm_param: 0.3,
            //         ef_construction: 50,
            //         m_max: 20,
            //         m_max_0: 40,
            //         distance: Dot,
            //     },
            //     SliceS2,
            // ),
            // ModelParams::VpTreeEnsemble(
            //     EnsembleParams {
            //         level_norm: 0.5,
            //         strategy: EnsembleStrategy::BruteForceKNN,
            //         ..ensemble_params
            //     },
            //     false,
            // ),
            // ModelParams::VpTreeEnsemble(
            //     EnsembleParams {
            //         level_norm: 0.0,
            //         strategy: EnsembleStrategy::BruteForceKNN,
            //         ..ensemble_params
            //     },
            //     true,
            // ),
        ],
        search_params: vec![SearchParams {
            truth_distance: Dot,
            k: 30,
            ef: 100,
            start_candidates: 1,
            vp_max_visits: 0,
        }],
        random_seeds: true,
        repeats: 3,
        title: "try it out",
    };

    let experiments: Vec<ExperimentSetup> = vec![
        test_setup,
        // try_hnsw_effect_of_ef_construction(),
        // try_hnsw_effect_of_level_norm(),
        // try_hnsw_effect_of_ef_search(),
        // try_hnsw_effect_of_m_max(),
    ];
    for e in experiments.iter() {
        println!("Start experiment {}", e.to_string());
        eval_models_on_laion(e)
    }
}

const SMALL_N: usize = 1_000;
const MEDIUM_N: usize = 1_000_000;
const LARGE_N: usize = 10_000_000;

fn try_hnsw_effect_of_ef_construction() -> ExperimentSetup {
    let ef_construction: Vec<usize> = vec![
        30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
    ];
    ExperimentSetup {
        n: SMALL_N,
        n_queries: 100,
        params: ef_construction
            .into_iter()
            .map(|ef_construction| {
                ModelParams::Hnsw(
                    HnswParams {
                        level_norm_param: 0.3,
                        ef_construction,
                        m_max: 20,
                        m_max_0: 40,
                        distance: Dot,
                    },
                    SliceS2,
                )
            })
            .collect(),
        search_params: vec![SearchParams {
            truth_distance: Dot,
            k: 30,
            ef: 60,
            start_candidates: 0,
            vp_max_visits: 0,
        }],
        random_seeds: false,
        repeats: 10,
        title: "hnsw_effect_of_ef_construction",
    }
}

fn try_hnsw_effect_of_level_norm() -> ExperimentSetup {
    let level_norm_params: Vec<f32> = (0..=20).map(|e| (e as f32) / 20.0).collect();
    ExperimentSetup {
        n: SMALL_N,
        n_queries: 100,
        params: level_norm_params
            .into_iter()
            .map(|level_norm_param| {
                ModelParams::Hnsw(
                    HnswParams {
                        level_norm_param,
                        ef_construction: 40,
                        m_max: 20,
                        m_max_0: 40,
                        distance: Dot,
                    },
                    SliceS2,
                )
            })
            .collect(),
        search_params: vec![SearchParams {
            truth_distance: Dot,
            k: 30,
            ef: 60,
            start_candidates: 0,
            vp_max_visits: 0,
        }],
        random_seeds: false,
        repeats: 10,
        title: "hnsw_effect_of_level_norm_param",
    }
}

fn try_hnsw_effect_of_m_max() -> ExperimentSetup {
    ExperimentSetup {
        n: SMALL_N,
        n_queries: 100,
        params: (8..=48usize)
            .step_by(4)
            .map(|m_max| {
                ModelParams::Hnsw(
                    HnswParams {
                        level_norm_param: 0.3,
                        ef_construction: 40,
                        m_max,
                        m_max_0: m_max * 2,
                        distance: Dot,
                    },
                    SliceS2,
                )
            })
            .collect(),
        search_params: vec![SearchParams {
            truth_distance: Dot,
            k: 30,
            ef: 60,
            start_candidates: 0,
            vp_max_visits: 0,
        }],
        random_seeds: false,
        repeats: 10,
        title: "hnsw_effect_of_m_max",
    }
}

fn try_hnsw_effect_of_ef_search() -> ExperimentSetup {
    ExperimentSetup {
        n: SMALL_N,
        n_queries: 100,
        params: vec![ModelParams::Hnsw(
            HnswParams {
                level_norm_param: 0.3,
                ef_construction: 40,
                m_max: 20,
                m_max_0: 40,
                distance: Dot,
            },
            SliceS2,
        )],
        search_params: (30..=200usize)
            .step_by(10)
            .map(|ef| SearchParams {
                truth_distance: Dot,
                k: 30,
                ef,
                start_candidates: 0,
                vp_max_visits: 0,
            })
            .collect(),
        random_seeds: false,
        repeats: 10,
        title: "hnsw_effect_of_level_norm_param",
    }
}

#[derive(Debug, Clone)]
struct ExperimentSetup {
    n: usize,
    n_queries: usize,
    params: Vec<ModelParams>,
    search_params: Vec<SearchParams>,
    random_seeds: bool,
    repeats: usize,
    title: &'static str,
}

impl ExperimentSetup {
    pub fn to_string(&self) -> String {
        format!("{}_n={}_queries_n={}", self.title, self.n, self.n_queries)
    }
}
#[derive(Debug, Clone, Copy, PartialEq)]
enum ModelParams {
    Hnsw(HnswParams, HnswStructure),
    Stitching(StitchingParams),
    VpTreeEnsemble(EnsembleParams, bool), // bool = threaded
    RNNGraph(RNNGraphParams),
    VpTree(VpTreeParams),
    HnswByRnnDescent(RNNGraphParams, f32), // f32 = level_norm_param
}
impl Eq for ModelParams {}

impl ModelParams {
    pub fn to_string(&self) -> String {
        match self {
            ModelParams::Hnsw(e, kind) => format!("{kind:?}{e:?}"),
            ModelParams::Stitching(e) => format!("{e:?}"),
            ModelParams::VpTreeEnsemble(e, threaded) => {
                if *threaded {
                    format!("Threaded {e:?}")
                } else {
                    format!("{e:?}")
                }
            }
            ModelParams::RNNGraph(e) => format!("{e:?}"),
            ModelParams::VpTree(e) => format!("{e:?}"),
            ModelParams::HnswByRnnDescent(e, level_norm_param) => {
                format!("{e:?}, level_norm: {level_norm_param}")
            }
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq)]
enum HnswStructure {
    Old,
    Const,
    SliceS1,
    SliceS2,
    SliceParralelRayon,
    SliceParralelThreadPool,
}

enum Model {
    ConstHnsw(ConstHnsw),
    OldHnsw(Hnsw),
    SliceHnsw(SliceHnsw),
    SliceHnswParallel(vecnn::slice_hnsw_par::SliceHnsw),
    RNNGraph(RNNGraph),
    VpTree(VpTree),
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SearchParams {
    truth_distance: Distance,
    k: usize,
    ef: usize,               // for HNSW and others
    start_candidates: usize, // for RNN graph
    vp_max_visits: usize,
}
impl Display for SearchParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.start_candidates == 0, self.vp_max_visits == 0) {
            (true, true) => f.write_fmt(format_args!("k={} ef={}", self.k, self.ef,)),
            (true, false) => f.write_fmt(format_args!(
                "dist={:?} k={} ef={} vp_max_visits={}",
                self.truth_distance, self.k, self.ef, self.vp_max_visits
            )),
            (false, true) => f.write_fmt(format_args!(
                "dist={:?} k={} ef={} start_candidates={}",
                self.truth_distance, self.k, self.ef, self.start_candidates
            )),
            (false, false) => f.write_fmt(format_args!(
                "dist={:?} k={} ef={} start_candidates={} vp_max_visits={}",
                self.truth_distance, self.k, self.ef, self.start_candidates, self.vp_max_visits
            )),
        }
    }
}

impl Model {
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
            Model::OldHnsw(hnsw) => {
                let res = hnsw.knn_search(q_data, search_params.k, search_params.ef);
                found = res
                    .0
                    .iter()
                    .map(|e| e.id as usize)
                    .collect::<HashSet<usize>>();
                stats = res.1
            }
            Model::SliceHnsw(hnsw) => {
                let res = hnsw.knn_search(q_data, search_params.k, search_params.ef);
                found = res
                    .0
                    .iter()
                    .map(|e| e.1 as usize)
                    .collect::<HashSet<usize>>();
                stats = res.1
            }
            Model::SliceHnswParallel(hnsw) => {
                let res = hnsw.knn_search(q_data, search_params.k, search_params.ef);
                found = res
                    .0
                    .iter()
                    .map(|e| e.1 as usize)
                    .collect::<HashSet<usize>>();
                stats = res.1
            }
            Model::RNNGraph(rnn_graph) => {
                let res = rnn_graph.knn_search(
                    q_data,
                    search_params.k,
                    search_params.ef,
                    search_params.start_candidates,
                );
                found = res.0.iter().map(|e| e.1).collect::<HashSet<usize>>();
                stats = res.1;
            }
            Model::ConstHnsw(hnsw) => {
                let res = hnsw.knn_search(q_data, search_params.k, search_params.ef);
                found = res
                    .0
                    .iter()
                    .map(|e| e.1 as usize)
                    .collect::<HashSet<usize>>();
                stats = res.1
            }
            Model::VpTree(vp_tree) => {
                let res = if search_params.vp_max_visits == 0 {
                    vp_tree.knn_search(q_data, search_params.k)
                } else {
                    vp_tree.knn_search_approximative(
                        q_data,
                        search_params.k,
                        search_params.vp_max_visits,
                    )
                };
                found = res
                    .0
                    .iter()
                    .map(|e| e.1 as usize)
                    .collect::<HashSet<usize>>();
                stats = res.1
            }
        };
        let recall_n = true_res.iter().filter(|i| found.contains(i)).count();
        let r = recall_n as f64 / search_params.k as f64;
        (r, stats)
    }

    pub fn build_stats(&self) -> Stats {
        match self {
            Model::SliceHnsw(e) => e.build_stats,
            Model::RNNGraph(e) => e.build_stats,
            Model::ConstHnsw(e) => e.build_stats,
            Model::OldHnsw(e) => e.build_stats,
            Model::SliceHnswParallel(e) => e.build_stats,
            Model::VpTree(e) => e.build_stats,
        }
    }
}

fn eval_models_on_laion(setup: &ExperimentSetup) {
    if setup.params.len() == 0 {
        return;
    }

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
        let mut rng = ChaCha20Rng::seed_from_u64(RNG_SEED);
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

    let data = data_set_from_path(DATA_PATH, DATA_LEN, DIMS, setup.n);
    let queries = data_set_from_path(QUERIES_PATH, QUERIES_LEN, DIMS, setup.n_queries);
    eval_models(data, queries, setup)
}

fn eval_models_random_data(dims: usize, setup: &ExperimentSetup) {
    let data = random_data_set(setup.n, dims);
    let queries = random_data_set(setup.n_queries, dims);
    eval_models(data, queries, setup)
}

fn eval_models(data: Arc<dyn DatasetT>, queries: Arc<dyn DatasetT>, setup: &ExperimentSetup) {
    let n_queries = queries.len();

    let mut true_knns_by_search_params: Vec<Vec<HashSet<usize>>> = vec![];
    for search_params in setup.search_params.iter() {
        let mut true_knns: Vec<HashSet<usize>> = vec![];
        for i in 0..n_queries {
            let q_data = queries.get(i);
            let knn = linear_knn_search(
                &*data,
                q_data,
                search_params.k,
                search_params.truth_distance.to_fn(),
            )
            .iter()
            .map(|e| e.1)
            .collect::<HashSet<usize>>();
            true_knns.push(knn);
        }
        true_knns_by_search_params.push(true_knns);
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct Result {
        recall_mean: f64,
        recall_std: f64,
        ndc_mean: f64,
        ndc_std: f64,
        time_ms_mean: f64,
        time_ms_std: f64,
        build_ndc: usize,
        build_duration_ms: f32,
    }
    impl Result {
        fn add(&mut self, other: &Result) {
            self.recall_mean += other.recall_mean;
            self.recall_std += other.recall_std;
            self.ndc_mean += other.ndc_mean;
            self.ndc_std += other.ndc_std;
            self.time_ms_mean += other.time_ms_mean;
            self.time_ms_std += other.time_ms_std;
            self.build_ndc += other.build_ndc;
            self.build_duration_ms += other.build_duration_ms;
        }

        fn divide(&mut self, n: usize) {
            self.recall_mean /= n as f64;
            self.recall_std /= n as f64;
            self.ndc_mean /= n as f64;
            self.ndc_std /= n as f64;
            self.time_ms_mean /= n as f64;
            self.time_ms_std /= n as f64;
            self.build_ndc /= n;
            self.build_duration_ms /= n as f32;
        }
    }

    let mut results: Vec<(ModelParams, SearchParams, Vec<Result>)> = vec![];
    for _ in 0..setup.repeats {
        for &model_params in setup.params.iter() {
            let seed: u64 = if setup.random_seeds {
                thread_rng().gen()
            } else {
                42
            };
            let data = data.clone();
            let mut model = match model_params {
                ModelParams::Hnsw(params, kind) => match kind {
                    HnswStructure::Old => Model::OldHnsw(Hnsw::new(data, params, seed)),
                    HnswStructure::Const => Model::ConstHnsw(ConstHnsw::new(data, params, seed)),
                    HnswStructure::SliceS1 => {
                        Model::SliceHnsw(SliceHnsw::new_strategy_1(data, params, seed))
                    }
                    HnswStructure::SliceS2 => {
                        Model::SliceHnsw(SliceHnsw::new_strategy_2(data, params, seed))
                    }
                    HnswStructure::SliceParralelRayon => Model::SliceHnswParallel(
                        vecnn::slice_hnsw_par::SliceHnsw::new(data, params, seed),
                    ),
                    HnswStructure::SliceParralelThreadPool => Model::SliceHnswParallel(
                        vecnn::slice_hnsw_par::SliceHnsw::new_with_thread_pool(data, params, seed),
                    ),
                },
                ModelParams::Stitching(params) => {
                    Model::SliceHnsw(build_hnsw_by_vp_tree_stitching(data, params, seed))
                }
                ModelParams::RNNGraph(params) => Model::RNNGraph(RNNGraph::new(data, params, seed)),
                ModelParams::VpTreeEnsemble(params, threaded) => {
                    // todo! a bit hacky!! move the specialization to the transition module instead.
                    if params.level_norm == -1.0 {
                        Model::SliceHnsw(build_single_layer_hnsw_by_vp_tree_ensemble(
                            data, params, seed,
                        ))
                    } else {
                        Model::SliceHnsw(build_hnsw_by_vp_tree_ensemble_multi_layer(
                            data, params, threaded, seed,
                        ))
                    }
                }
                ModelParams::VpTree(params) => Model::VpTree(VpTree::new(data, params, seed)),
                ModelParams::HnswByRnnDescent(params, level_norm_param) => Model::SliceHnsw(
                    build_hnsw_by_rnn_descent(data, params, level_norm_param, seed),
                ),
            };

            const CONVERT_INTO_NON_LOCK_VERSION: bool = true;
            if CONVERT_INTO_NON_LOCK_VERSION {
                if let Model::SliceHnswParallel(par_hnsw) = &model {
                    let non_lock_hnsw = par_hnsw.convert_to_slice_hnsw_without_locks();
                    model = Model::SliceHnsw(non_lock_hnsw);
                }
            }

            for (search_params_idx, search_params) in setup.search_params.iter().enumerate() {
                let true_knns = &true_knns_by_search_params[search_params_idx];

                let mut recalls: Vec<f64> = Vec::with_capacity(n_queries);
                let mut ndcs: Vec<f64> = Vec::with_capacity(n_queries);
                let mut time_mss: Vec<f64> = Vec::with_capacity(n_queries);

                for i in 0..n_queries {
                    let q_data = queries.get(i);
                    let (r, s) = model.knn_search(q_data, &true_knns[i], *search_params);
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

                let model_result = Result {
                    recall_mean,
                    recall_std,
                    ndc_mean,
                    ndc_std,
                    time_ms_mean,
                    time_ms_std,
                    build_ndc: model.build_stats().num_distance_calculations,
                    build_duration_ms: model.build_stats().duration.as_secs_f32() * 1000.0,
                };

                if let Some((_, s, r)) = results
                    .iter_mut()
                    .find(|(m, s, _)| *m == model_params && s == search_params)
                {
                    r.push(model_result);
                } else {
                    results.push((model_params, *search_params, vec![model_result]))
                }
            }
        }
    }

    let mut table = Table::new();
    table.add_row(row![
        "rep", // repeats
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

    for (model_params, search_params, model_params_results) in results.iter() {
        let mut mean_result: Result = Result::default();
        for r in model_params_results.iter() {
            mean_result.add(r);
        }
        mean_result.divide(model_params_results.len());

        table.add_row(row![
            model_params_results.len(),
            data.len().to_string(),
            search_params.to_string(),
            model_params.to_string(),
            mean_result.build_ndc,
            mean_result.build_duration_ms,
            format!("{:.3}", mean_result.recall_mean),
            // format!("{:.3}", results.recall_std),
            format!("{:.3}", mean_result.ndc_mean),
            // format!("{:.3}", results.ndc_std),
            format!("{:.3}", mean_result.time_ms_mean),
            // format!("{:.3}", results.time_ms_std),
        ]);
    }
    table.set_format(FormatBuilder::new().padding(2, 2).build());

    let file_path = format!("experiments/{}.csv", setup.to_string());
    if std::fs::File::open(&file_path).is_err() {
        let mut file = File::create(&file_path).unwrap();
        table.slice(0..1).to_csv(&mut file).unwrap();
    }
    let mut file = OpenOptions::new()
        .append(true) // Enable appending
        .open(&file_path)
        .unwrap(); // Open the file
    table.slice(1..).to_csv(&mut file).unwrap();

    table.printstd();
}
