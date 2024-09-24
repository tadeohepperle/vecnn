use std::{
    collections::HashSet,
    fmt::Display,
    fs::{File, OpenOptions},
    hash::Hash,
    io::Write,
    ops::{Add, AddAssign, DerefMut},
    os, slice,
    sync::Arc,
    time::Instant,
};

use ahash::{HashMap, HashMapExt};
use prettytable::{
    format::{FormatBuilder, LineSeparator, TableFormat},
    row, AsTableSlice, Slice, Table,
};
use rand::{seq::SliceRandom, thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use vecnn::{
    const_hnsw::ConstHnsw,
    dataset::{self, DatasetT, FlatDataSet},
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
    utils::{linear_knn_search, random_data_set, Stats, YoloCell},
    vp_tree::{VpTree, VpTreeParams},
};
use HnswStructure::*;

const IS_ON_SERVER: bool = false; // Uni Server for testing 10M dataset
const DATA_PATH: &str = const {
    if IS_ON_SERVER {
        "/data/hepperle"
    } else {
        "../data"
    }
};
const FILE_NAME_300K: &str = "laion_300k_(300000, 768).bin";
const FILE_NAME_10M: &str = "laion_10m_(10120191, 768).bin";
const FILE_NAME_QUERIES: &str = "laion_queries_(10000, 768).bin";
const FILE_NAME_GOLD_300K: &str = "laion_gold_300k_(10000, 1000).bin";
const FILE_NAME_GOLD_10M: &str = "laion_gold_10m_(10000, 1000).bin";

#[derive(Debug, Clone, Copy)]
enum DataSetSize {
    _300K,
    _10M,
    Sampled(usize), // anything below 10M
}

/// for benchmarks we need 3 pieces of data: the big dataset the index is built on, the query set and the the true knn indices in the dataset for each query.
#[derive(Debug)]
struct RequiredData {
    data: Arc<dyn DatasetT>,
    queries: Arc<dyn DatasetT>,
    true_knns: Vec<usize>, // true_knns.len = `queries.len * 1000`, indices between 0 and `data.len`
}
impl RequiredData {
    // returns slice of the k true knn indices in the dataset
    pub fn get_true_knn(&self, query_idx: usize, k: usize) -> &[usize] {
        assert!(k <= 1000);
        let start_idx = query_idx * 1000;
        let end_idx = start_idx + k;
        &self.true_knns[start_idx..end_idx]
    }
}

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
        n_candidates: 0,
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
        max_chunk_size: 512,
        same_chunk_m_max: 20,
        neg_fraction: 0.4,
        keep_fraction: 0.0,
        x_or_ef: 2,
        only_n_chunks: None,
        distance: Distance::Dot,
        stitch_mode: StitchMode::BestXofRandomXTimesX,
        m_max: 20,
        n_candidates: 0,
    };

    let test_setup = ExperimentSetup {
        n: 10000,
        n_queries: 100,
        params: vec![ModelParams::Hnsw(
            HnswParams {
                level_norm_param: 0.3,
                ef_construction: 20,
                m_max: 20,
                m_max_0: 40,
                distance: Dot,
            },
            SliceS2,
        )],
        search_params: vec![SearchParams {
            truth_distance: Dot,
            k: 30,
            ef: 100,
            start_candidates: 1,
            vp_max_visits: 0,
        }],
        random_seeds: true,
        repeats: 1,
        title: "try it out",
    };

    let experiments: Vec<ExperimentSetup> = vec![test_setup];
    // let experiments = final_experiment_collection();
    for e in experiments.iter() {
        println!("Start experiment {}", e.to_string());
        eval_models_on_laion(e)
    }
}
const N_10K: usize = 10_000;
const N_100K: usize = 100_000;
const N_1M: usize = 1_000_000;
const N_10M: usize = 10_000_000;

fn final_experiment_collection() -> Vec<ExperimentSetup> {
    fn search_params() -> Vec<SearchParams> {
        vec![SearchParams {
            truth_distance: Dot,
            k: 30,
            ef: 60,
            start_candidates: 0,
            vp_max_visits: 0,
        }]
    }

    fn search_params_varied_k_and_ef() -> Vec<SearchParams> {
        (30..=200usize)
            .step_by(10)
            .map(|ef| SearchParams {
                truth_distance: Dot,
                k: 30,
                ef,
                start_candidates: 0,
                vp_max_visits: 0,
            })
            .chain((30..=200usize).step_by(10).map(|k_and_ef| SearchParams {
                truth_distance: Dot,
                k: k_and_ef,
                ef: k_and_ef,
                start_candidates: 0,
                vp_max_visits: 0,
            }))
            .collect()
    }
    fn n_log_steps_per_magnitude(
        min_n: usize,
        max_n: usize,
        steps_per_magnitude: usize,
    ) -> Vec<usize> {
        let mut ns: Vec<usize> = vec![];
        let mut n: f64 = min_n as f64;
        let factor = (10.0f64).powf(1.0 / steps_per_magnitude as f64);
        while n as usize <= max_n {
            ns.push(n as usize);
            n *= factor;
        }
        assert!(ns[0] == min_n);
        assert!(*ns.last().unwrap() == max_n);
        return ns;
    }
    // /////////////////////////////////////////////////////////////////////////////
    // SECTION: HNSW
    // /////////////////////////////////////////////////////////////////////////////
    fn exp_hnsw_effect_of_m_max() -> ExperimentSetup {
        ExperimentSetup {
            n: N_100K,
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
            search_params: search_params(),
            random_seeds: false,
            repeats: 10,
            title: "exp_hnsw_effect_of_m_max",
        }
    }
    fn exp_hnsw_effect_of_ef_construction() -> ExperimentSetup {
        let ef_construction: Vec<usize> = vec![
            30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
        ];
        ExperimentSetup {
            n: N_100K,
            n_queries: 1000,
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
            search_params: search_params(),
            random_seeds: false,
            repeats: 1,
            title: "exp_hnsw_effect_of_ef_construction",
        }
    }
    fn exp_hnsw_effect_of_level_norm() -> ExperimentSetup {
        let level_norm_params: Vec<f32> = (0..=20).map(|e| (e as f32) / 20.0).collect();
        ExperimentSetup {
            n: N_100K,
            n_queries: 1000,
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
            search_params: search_params(),
            random_seeds: false,
            repeats: 1,
            title: "exp_hnsw_effect_of_level_norm",
        }
    }
    fn exp_hnsw_effect_of_ef_search_and_k() -> ExperimentSetup {
        ExperimentSetup {
            n: N_100K,
            n_queries: 1000,
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
            search_params: search_params_varied_k_and_ef(),
            random_seeds: false,
            repeats: 1,
            title: "exp_hnsw_effect_of_ef_search_and_k",
        }
    }
    // WARNING! CAN BE FAIRLY EXPENSIVE AND LONG RUNNING!
    fn exp_hnsw_effect_of_n() -> Vec<ExperimentSetup> {
        return n_log_steps_per_magnitude(N_10K, N_10M, 5)
            .into_iter()
            .map(|n| ExperimentSetup {
                n,
                n_queries: 100,
                params: vec![ModelParams::Hnsw(
                    HnswParams {
                        level_norm_param: 0.3,
                        ef_construction: 60,
                        m_max: 20,
                        m_max_0: 40,
                        distance: Dot,
                    },
                    SliceS2,
                )],
                search_params: search_params(),
                random_seeds: false,
                repeats: 1,
                title: "exp_hnsw_effect_of_n",
            })
            .collect();
    }
    // /////////////////////////////////////////////////////////////////////////////
    // SECTION: RNN Graphs
    // /////////////////////////////////////////////////////////////////////////////
    fn exp_rnn_effect_of_inner_loops() -> ExperimentSetup {
        ExperimentSetup {
            n: N_100K,
            n_queries: 10000,
            params: (1..=24)
                .map(|t_inner| {
                    ModelParams::RNNGraph(RNNGraphParams {
                        outer_loops: 1,
                        inner_loops: t_inner,
                        max_neighbors_after_reverse_pruning: 40,
                        initial_neighbors: 40,
                        distance: Dot,
                    })
                })
                .collect(),
            search_params: search_params(),
            random_seeds: false,
            repeats: 1,
            title: "exp_rnn_effect_of_inner_loops",
        }
    }
    fn exp_rnn_effect_of_outer_loops() -> ExperimentSetup {
        fn loops(inner_loops: usize, outer_loops: usize) -> ModelParams {
            ModelParams::RNNGraph(RNNGraphParams {
                inner_loops,
                outer_loops,
                max_neighbors_after_reverse_pruning: 40,
                initial_neighbors: 40,
                distance: Dot,
            })
        }
        ExperimentSetup {
            n: N_100K,
            n_queries: 10000,
            params: vec![
                loops(48, 1),
                loops(24, 2),
                loops(16, 3),
                loops(12, 4),
                loops(8, 6),
                loops(6, 8),
                loops(4, 12),
                loops(2, 24),
                loops(48, 1),
            ],
            search_params: search_params(),
            random_seeds: false,
            repeats: 1,
            title: "exp_rnn_effect_of_outer_loops",
        }
    }
    fn exp_rnn_effect_of_ef_search_and_k() -> ExperimentSetup {
        ExperimentSetup {
            n: N_100K,
            n_queries: 100,
            params: vec![ModelParams::RNNGraph(RNNGraphParams {
                inner_loops: 4,
                outer_loops: 3,
                max_neighbors_after_reverse_pruning: 40,
                initial_neighbors: 40,
                distance: Dot,
            })],
            search_params: search_params_varied_k_and_ef(),
            random_seeds: false,
            repeats: 5,
            title: "exp_rnn_effect_of_ef_search_and_k",
        }
    }
    fn exp_rnn_effect_of_multi_start_points() -> ExperimentSetup {
        ExperimentSetup {
            n: N_100K,
            n_queries: 1000,
            params: vec![ModelParams::RNNGraph(RNNGraphParams {
                inner_loops: 4,
                outer_loops: 3,
                max_neighbors_after_reverse_pruning: 40,
                initial_neighbors: 40,
                distance: Dot,
            })],
            search_params: (1usize..=20)
                .map(|start_candidates| SearchParams {
                    truth_distance: Dot,
                    k: 30,
                    ef: 60,
                    start_candidates,
                    vp_max_visits: 0,
                })
                .collect(),
            random_seeds: false,
            repeats: 5,
            title: "exp_rnn_effect_of_ef_search_and_k",
        }
    }
    fn exp_rnn_effect_of_n() -> Vec<ExperimentSetup> {
        return n_log_steps_per_magnitude(N_10K, N_10M, 5)
            .into_iter()
            .map(|n| ExperimentSetup {
                n,
                n_queries: 100,
                params: vec![ModelParams::RNNGraph(RNNGraphParams {
                    inner_loops: 4,
                    outer_loops: 3,
                    max_neighbors_after_reverse_pruning: 40,
                    initial_neighbors: 40,
                    distance: Dot,
                })],
                search_params: search_params(),
                random_seeds: false,
                repeats: 1,
                title: "exp_rnn_effect_of_n",
            })
            .collect();
    }

    // /////////////////////////////////////////////////////////////////////////////
    // SECTION: VP Tree (just for construction, of course not for search)
    // /////////////////////////////////////////////////////////////////////////////

    // /////////////////////////////////////////////////////////////////////////////
    // SECTION: Chunk Stitching
    // /////////////////////////////////////////////////////////////////////////////

    // /////////////////////////////////////////////////////////////////////////////
    // SECTION: VP Tree Ensemble
    // /////////////////////////////////////////////////////////////////////////////
    fn exp_ensemble_effect_of_n_vp_trees() -> ExperimentSetup {
        ExperimentSetup {
            n: N_100K,
            n_queries: 1000,
            params: (2..=20)
                .map(|n_vp_trees| {
                    ModelParams::VpTreeEnsemble(
                        EnsembleParams {
                            n_vp_trees,
                            max_chunk_size: 256,
                            same_chunk_m_max: 16,
                            m_max: 20,
                            m_max_0: 40,
                            distance: Distance::Dot,
                            level_norm: 0.0,
                            strategy: EnsembleStrategy::BruteForceKNN,
                            n_candidates: 0,
                        },
                        false,
                    )
                })
                .collect(),
            search_params: search_params(),
            random_seeds: false,
            repeats: 1,
            title: "exp_ensemble_effect_of_n_vp_trees",
        }
    }

    fn exp_ensemble_effect_of_chunk_size() -> ExperimentSetup {
        ExperimentSetup {
            n: N_100K,
            n_queries: 1000,
            params: [64usize, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048]
                .into_iter()
                .map(|chunk_size| {
                    ModelParams::VpTreeEnsemble(
                        EnsembleParams {
                            n_vp_trees: 6,
                            max_chunk_size: chunk_size,
                            same_chunk_m_max: 16,
                            m_max: 20,
                            m_max_0: 40,
                            distance: Distance::Dot,
                            level_norm: 0.0,
                            strategy: EnsembleStrategy::BruteForceKNN,
                            n_candidates: 0,
                        },
                        false,
                    )
                })
                .collect(),
            search_params: search_params(),
            random_seeds: false,
            repeats: 1,
            title: "exp_ensemble_effect_of_chunk_size",
        }
    }
    fn exp_ensemble_effect_of_multiple_vantage_points() -> ExperimentSetup {
        ExperimentSetup {
            n: N_100K,
            n_queries: 500,
            params: [1, 4, 8, 16, 32, 64]
                .into_iter()
                .map(|n_candidates| {
                    ModelParams::VpTreeEnsemble(
                        EnsembleParams {
                            n_vp_trees: 6,
                            max_chunk_size: 256,
                            same_chunk_m_max: 16,
                            m_max: 20,
                            m_max_0: 40,
                            distance: Distance::Dot,
                            level_norm: 0.0,
                            strategy: EnsembleStrategy::BruteForceKNN,
                            n_candidates,
                        },
                        false,
                    )
                })
                .collect(),
            search_params: search_params(),
            random_seeds: false,
            repeats: 1,
            title: "exp_ensemble_effect_of_multiple_vantage_points",
        }
    }
    fn exp_ensemble_effect_of_level_norm() -> ExperimentSetup {
        let level_norm_params: Vec<f32> = (0..=20).map(|e| (e as f32) / 20.0).collect();
        ExperimentSetup {
            n: N_100K,
            n_queries: 100,
            params: level_norm_params
                .into_iter()
                .map(|level_norm| {
                    ModelParams::VpTreeEnsemble(
                        EnsembleParams {
                            n_vp_trees: 6,
                            max_chunk_size: 256,
                            same_chunk_m_max: 16,
                            m_max: 20,
                            m_max_0: 40,
                            distance: Distance::Dot,
                            level_norm,
                            strategy: EnsembleStrategy::BruteForceKNN,
                            n_candidates: 0,
                        },
                        false,
                    )
                })
                .collect(),
            search_params: search_params(),
            random_seeds: false,
            repeats: 1,
            title: "exp_ensemble_effect_of_level_norm",
        }
    }

    // /////////////////////////////////////////////////////////////////////////////

    const WHAT_TO_RUN: WhatToRun = WhatToRun {
        hnsw_100k: true,
        rnn_100k: true,
        ensemble_100k: true,
        hnsw_by_n: false,
        rnn_graph_by_n: false,
    };
    struct WhatToRun {
        hnsw_100k: bool,
        hnsw_by_n: bool,
        rnn_100k: bool,
        rnn_graph_by_n: bool,
        ensemble_100k: bool,
    }

    let mut res = vec![];
    if WHAT_TO_RUN.hnsw_100k {
        res.extend([
            // exp_hnsw_effect_of_m_max(),
            // exp_hnsw_effect_of_ef_search_and_k(),
            // exp_hnsw_effect_of_ef_construction(),
            // exp_hnsw_effect_of_level_norm(),
        ]);
    }

    if WHAT_TO_RUN.rnn_100k {
        res.extend([
            // exp_rnn_effect_of_inner_loops(),
            // exp_rnn_effect_of_outer_loops(),
            // exp_rnn_effect_of_ef_search_and_k(),
            // exp_rnn_effect_of_multi_start_points(),
        ]);
    }
    if WHAT_TO_RUN.ensemble_100k {
        res.extend([
            // exp_ensemble_effect_of_chunk_size(),
            // exp_ensemble_effect_of_n_vp_trees(),
            exp_ensemble_effect_of_multiple_vantage_points(),
            exp_ensemble_effect_of_level_norm(),
        ]);
    }
    if WHAT_TO_RUN.hnsw_by_n {
        res.extend(exp_hnsw_effect_of_n());
    }
    if WHAT_TO_RUN.rnn_graph_by_n {
        res.extend(exp_rnn_effect_of_n());
    }

    return res;
}

fn try_stitching_n_candidates(n: usize) -> ExperimentSetup {
    let stitch_params = StitchingParams {
        max_chunk_size: 128,
        same_chunk_m_max: 20,
        neg_fraction: 0.4,
        keep_fraction: 0.0,
        m_max: 20,
        x_or_ef: 2,
        only_n_chunks: None,
        distance: Dot,
        stitch_mode: StitchMode::RandomNegToRandomPosAndBack,
        n_candidates: 0,
    };
    ExperimentSetup {
        n,
        n_queries: 100,
        params: vec![
            ModelParams::Stitching(stitch_params),
            ModelParams::Stitching(StitchingParams {
                n_candidates: 20,
                ..stitch_params
            }),
            ModelParams::Stitching(StitchingParams {
                n_candidates: 40,
                ..stitch_params
            }),
            // ModelParams::VpTreeEnsemble(
            //     EnsembleParams {
            //         n_candidates: 0,
            //         ..params
            //     },
            //     false,
            // ),
            // ModelParams::VpTreeEnsemble(
            //     EnsembleParams {
            //         n_candidates: 3,
            //         ..params
            //     },
            //     false,
            // ),
        ],
        search_params: vec![SearchParams {
            truth_distance: Dot,
            k: 30,
            ef: 100,
            start_candidates: 0,
            vp_max_visits: 0,
        }],
        random_seeds: false,
        repeats: 1,
        title: "stitching_n_candidates",
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
        true_res: &[usize],
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

impl RequiredData {
    // compare to true knns
}

// e.g. take "laion_300k_(300000,768).bin" and return 300000, 768
fn extract_two_numbers_from_tuple_in_file_name(file_name: &str) -> (usize, usize) {
    let mut iter = file_name
        .split("(")
        .skip(1)
        .next()
        .unwrap()
        .split(")")
        .next()
        .unwrap()
        .split(",")
        .map(|e| e.trim());
    let a = iter.next().unwrap().parse::<usize>().unwrap();
    let b = iter.next().unwrap().parse::<usize>().unwrap();
    (a, b)
}

fn load_data(data_set_size: DataSetSize, n_queries: usize) -> RequiredData {
    assert!(n_queries <= 10000);
    let data_file_name: &str = match data_set_size {
        DataSetSize::Sampled(n) => {
            assert!(n <= 10_000_000); // should be adjusted later maybe
            if n < 300000 {
                FILE_NAME_300K
            } else {
                FILE_NAME_10M
            }
        }
        DataSetSize::_300K => FILE_NAME_300K,
        DataSetSize::_10M => FILE_NAME_10M,
    };
    let (data_n, dims) = extract_two_numbers_from_tuple_in_file_name(data_file_name);
    let data_file_path = format!("{DATA_PATH}/{data_file_name}");
    let mut data = FlatDataSet {
        dims,
        len: data_n,
        floats: load_binary_data::<f32>(&data_file_path, data_n, dims),
    };

    let (queries_n, dims_q) = extract_two_numbers_from_tuple_in_file_name(FILE_NAME_QUERIES);
    assert!(dims_q == dims);
    let queries_file_path = format!("{DATA_PATH}/{FILE_NAME_QUERIES}");
    let mut queries = FlatDataSet {
        dims,
        len: queries_n,
        floats: load_binary_data::<f32>(&queries_file_path, queries_n, dims),
    };

    // maybe subsample the loaded data:
    if let DataSetSize::Sampled(n) = data_set_size {
        const DATA_SUBSAMPLE_SEED: u64 = 123;
        let mut rng = ChaCha20Rng::seed_from_u64(DATA_SUBSAMPLE_SEED);
        let data_indices = rand::seq::index::sample(&mut rng, data.len, n).into_vec();
        data.floats = subsample_flat(&data.floats, dims, &data_indices);
        data.len = n;
    }

    // load the 1000 true knns for the dataset size. For fixed sizes use the ones provided by sisap, for custom sizes calculate new ones and cache them for future use (next to the official SISAP gold knns)
    let true_knns_file_name: String = match data_set_size {
        DataSetSize::_300K => FILE_NAME_GOLD_300K.to_string(), // from sisap website:  https://sisap-challenges.github.io/2024/datasets/#gold_standard_files_for_the_2024_private_queries
        DataSetSize::_10M => FILE_NAME_GOLD_10M.to_string(),   // from sisap website.
        DataSetSize::Sampled(n) => {
            format!("computed_true_knns_n={n}_(10000,1000).bin")
        }
    };
    let true_knns_path = format!("{DATA_PATH}/{true_knns_file_name}");
    let true_knns_path = std::path::Path::new(&true_knns_path);
    let mut true_knns: Vec<usize>; // flat vec of 10000 * 1000 indices

    const K: usize = 1000; // hard coded because the sisap files also come with k = 1000
    if !true_knns_path.exists() {
        let start_time = Instant::now();
        // calculate real 1000 nearest neighbors for all 10k queries and save them to file:
        let true_knns_cell = YoloCell::new(Vec::<usize>::with_capacity(queries.len * K));
        unsafe {
            true_knns_cell.get_mut().set_len(queries.len * K);
        }
        let queries_ds: &dyn DatasetT = &queries;

        let n_done = std::sync::atomic::AtomicU64::new(0);
        (0..queries.len).into_par_iter().for_each(|q_i| {
            let found = linear_knn_search(&data, queries_ds.get(q_i), K, dot);
            // note: found comed back sorted already
            assert!(found.len() == K);
            for nei_i in 0..K {
                let nei_id_in_data = found[nei_i].1;
                // we are only interested in the idx in data not in dist:
                unsafe { true_knns_cell.get_mut()[K * q_i + nei_i] = nei_id_in_data }
            }
            let n_done = n_done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            println!(
                "    Calculated {}/{} true knns for n={}",
                n_done, queries.len, data.len
            );
        });
        println!(
            "Determined true KNN for {} queries in {}s.",
            queries.len,
            start_time.elapsed().as_secs_f32()
        );
        true_knns = true_knns_cell.into_inner();
        let true_knn_bytes = unsafe {
            slice::from_raw_parts(
                true_knns.as_ptr() as *const u8,
                true_knns.len() * size_of::<usize>(),
            )
        };
        std::fs::write(true_knns_path, true_knn_bytes)
            .expect("could not create file for true knns!");
    } else {
        // load the true knns from file:
        let mut true_knn_bytes =
            std::fs::read(true_knns_path).expect("could not create file for true knns!");
        assert!(true_knn_bytes.len() == queries.len * K * size_of::<usize>());
        assert!(true_knn_bytes.capacity() % size_of::<usize>() == 0);
        true_knns = unsafe {
            Vec::from_raw_parts(
                true_knn_bytes.as_mut_ptr() as *mut usize,
                queries.len * K,
                true_knn_bytes.capacity() / size_of::<usize>(),
            )
        };
        std::mem::forget(true_knn_bytes); // die in dignity, you are now a vec of usizes already.
    }
    assert!(true_knns.len() == queries.len * K);

    // subsample queries and true knn if needed:
    if n_queries < 10000 {
        const QUERY_SUBSAMPLE_SEED: u64 = 123;
        // only keep queries from a sampled indices list:
        let mut rng = ChaCha20Rng::seed_from_u64(QUERY_SUBSAMPLE_SEED);
        let query_indices = rand::seq::index::sample(&mut rng, queries.len, n_queries).into_vec();
        queries.floats = subsample_flat(&queries.floats, dims, &query_indices);
        queries.len = n_queries;
        // keep only the same indices from the true knns list:
        true_knns = subsample_flat(&true_knns, K, &query_indices);
    }

    RequiredData {
        data: Arc::new(data),
        queries: Arc::new(queries),
        true_knns,
    }
}

fn load_binary_data<T: Copy>(path: &str, len: usize, elements_per_row: usize) -> Vec<T> {
    dbg!(path);
    let mut bytes = std::fs::read(path).unwrap();
    assert_eq!(
        bytes.len(),
        len * elements_per_row * std::mem::size_of::<T>()
    );
    assert!(bytes.capacity() % std::mem::size_of::<T>() == 0);
    let numbers = unsafe {
        Vec::<T>::from_raw_parts(
            bytes.as_mut_ptr() as *mut T,
            len * elements_per_row,
            bytes.capacity() / std::mem::size_of::<T>(),
        )
    };
    std::mem::forget(bytes); // the memory is now owned by the floats vec
    numbers
}

fn subsample_flat<T: Copy>(flat: &Vec<T>, elements_per_row: usize, indices: &Vec<usize>) -> Vec<T> {
    assert!(flat.len() % elements_per_row == 0);
    let n_rows_before = flat.len() / elements_per_row;
    let n_rows_after = indices.len();
    assert!(n_rows_after <= n_rows_before);
    let mut new_flat: Vec<T> = Vec::with_capacity(n_rows_after * elements_per_row);
    unsafe { new_flat.set_len(n_rows_after * elements_per_row) };
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        let old_ptr = &flat[old_idx * elements_per_row] as *const T;
        let new_ptr = &mut new_flat[new_idx * elements_per_row] as *mut T;
        unsafe {
            std::ptr::copy_nonoverlapping::<T>(old_ptr, new_ptr, elements_per_row);
        }
    }

    new_flat
}

fn eval_models_on_laion(setup: &ExperimentSetup) {
    if setup.params.len() == 0 {
        return;
    }
    let data_set_size = match setup.n {
        300_000 => DataSetSize::_300K,
        10_000_000 => DataSetSize::_10M,
        n => DataSetSize::Sampled(n),
    };
    let data = load_data(data_set_size, setup.n_queries);
    eval_models(&data, setup)
}

// fn eval_models_random_data(dims: usize, setup: &ExperimentSetup) {
//     let data = random_data_set(setup.n, dims);
//     let queries = random_data_set(setup.n_queries, dims);
//     eval_models(data, queries, setup)
// }

fn eval_models(req_data: &RequiredData, setup: &ExperimentSetup) {
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
            let data = req_data.data.clone();
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
                ModelParams::VpTree(params) => Model::VpTree(VpTree::new(data, params, seed, 0)),
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

            println!(
                "    Built model for {:?} in {} secs",
                model_params,
                model.build_stats().duration.as_secs_f32()
            );

            let model_search_start_time = Instant::now();
            let n_queries = req_data.queries.len();
            for (search_params_idx, search_params) in setup.search_params.iter().enumerate() {
                let mut recalls: Vec<f64> = Vec::with_capacity(n_queries);
                let mut ndcs: Vec<f64> = Vec::with_capacity(n_queries);
                let mut time_mss: Vec<f64> = Vec::with_capacity(n_queries);

                for q_i in 0..n_queries {
                    let q_data = req_data.queries.get(q_i);
                    let true_knn = req_data.get_true_knn(q_i, search_params.k);
                    let (r, s) = model.knn_search(q_data, true_knn, *search_params);
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
            println!(
                "        Finished searches for model in {} secs",
                model_search_start_time.elapsed().as_secs_f32()
            );
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
            req_data.data.len().to_string(),
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
