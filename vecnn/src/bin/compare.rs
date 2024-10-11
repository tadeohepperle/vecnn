use std::{
    collections::HashSet,
    env::args,
    fmt::Display,
    fs::{File, OpenOptions},
    slice,
    sync::Arc,
    time::Instant,
};

use prettytable::{format::FormatBuilder, row, AsTableSlice, Slice, Table};
use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use vecnn::{
    const_hnsw::ConstHnsw,
    dataset::{self, DatasetT, FlatDataSet},
    distance::{
        cos, dot, l2,
        Distance::{self, *},
    },
    hnsw::{Hnsw, HnswParams},
    relative_nn_descent::{RNNGraph, RNNGraphParams},
    slice_hnsw::SliceHnsw,
    transition::{
        build_hnsw_by_rnn_descent, build_hnsw_by_vp_tree_ensemble_multi_layer,
        build_hnsw_by_vp_tree_stitching, build_single_layer_hnsw_by_vp_tree_ensemble,
        EnsembleParams, EnsembleStrategy, StitchMode, StitchingParams,
    },
    utils::{linear_knn_search, Stats, YoloCell},
    vp_tree::{VpTree, VpTreeParams},
};
use HnswStructure::*;

const IS_ON_SERVER: bool = false; // modify on uni Server for testing 10M dataset!
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
const N_10K: usize = 10_000;
const N_100K: usize = 100_000;
const N_300K: usize = 300_000;
const N_1M: usize = 1_000_000;
const N_10M: usize = 10_000_000;

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

fn main() {
    if args().len() > 1 && args().any(|e| e == "gen") {
        // allows you to call `cargo run --bin compare --release -- gen 10000 500000` to just generate subsets of he 10m data
        // and write it to the data folder for e.g. use in python. Makes sure true knns are also either calculated or already present.
        gen_binary_datasets_by_subsampling_main();
        return;
    }

    let test_setup = ExperimentSetup {
        n: N_100K,
        n_queries: 100,
        params: vec![
            // ModelParams::VpTreeEnsemble(
            //     EnsembleParams {
            //         n_vp_trees: 4,
            //         max_chunk_size: 256,
            //         same_chunk_m_max: 10,
            //         m_max: 20,
            //         m_max_0: 40,
            //         level_norm: 0.0,
            //         distance: Dot,
            //         strategy: EnsembleStrategy::BruteForceKNN,
            //         n_candidates: 0,
            //     },
            //     false,
            // ),
            ModelParams::VpTreeEnsemble(
                EnsembleParams {
                    n_vp_trees: 4,
                    max_chunk_size: 1024,
                    same_chunk_m_max: 10,
                    m_max: 20,
                    m_max_0: 40,
                    level_norm: 0.0,
                    distance: Dot,
                    strategy: EnsembleStrategy::RNNDescent {
                        o_loops: 2,
                        i_loops: 3,
                    },
                    n_candidates: 0,
                },
                false,
            ),
        ],
        search_params: vec![SearchParams {
            truth_distance: Dot,
            k: 30,
            ef: 60,
            start_candidates: 0,
            vp_max_visits: 0,
        }],
        random_seeds: true,
        repeats: 1,
        title: "test_setup_rnn_threaded",
    };

    let experiments: Vec<ExperimentSetup> = vec![test_setup];
    // let experiments = final_experiment_collection();
    for e in experiments.iter() {
        eval_models_on_laion(e);
    }
}
/// Specify a bunch of experiments in here, comment in/out what is needed for a particular run
fn final_experiment_collection() -> Vec<ExperimentSetup> {
    let mut res = vec![];
    // // hnsw:
    res.extend([
        // _hnsw_effect_of_m_max(),
        // _hnsw_effect_of_ef_search_and_k(),
        // _hnsw_effect_of_ef_construction(),
        // _hnsw_effect_of_level_norm(),
    ]);
    //  res.extend(_hnsw_effect_of_n().into_iter().skip(11));
    // // rnn:
    res.extend([
        // _rnn_effect_of_inner_loops(),
        // _rnn_effect_of_outer_loops(),
        // _rnn_effect_of_ef_search_and_k(),
        // _rnn_effect_of_multi_start_points(),
        // _rnn_effect_of_num_neighbors(),
    ]);
    // res.extend(_rnn_effect_of_n());
    // // ensemble:
    res.extend([
        //  _ensemble_effect_of_chunk_size(),
        // _ensemble_effect_of_n_vp_trees(),
        // _ensemble_effect_of_multiple_vantage_points(),
        // _ensemble_effect_of_level_norm(),
        _ensemble_effect_of_brute_force_vs_rnn(),
        // _ensemble_effect_of_m_max(),
        // _ensemble_effect_of_same_chunk_m_max(),
    ]);
    // res.extend(_ensemble_effect_of_n());
    // stitching:
    res.extend([
        // _stitching_effect_of_fraction(),
        // _stitching_effect_of_max_chunk_size(),
        // _stitching_effect_of_multi_ef(),
        // _stitching_effect_of_m_max(),
    ]);
    return res;

    // /////////////////////////////////////////////////////////////////////////////
    // SECTION: Utils
    // /////////////////////////////////////////////////////////////////////////////
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
    const STITCHING_MODES: [StitchMode; 4] = [
        StitchMode::RandomNegToPosCenterAndBack,
        StitchMode::RandomNegToRandomPosAndBack,
        StitchMode::DontStarveXXSearch,
        StitchMode::MultiEf,
    ];
    const STITCHING_MODES_OK: [StitchMode; 3] = [
        StitchMode::RandomNegToRandomPosAndBack,
        StitchMode::DontStarveXXSearch,
        StitchMode::MultiEf,
    ];
    // /////////////////////////////////////////////////////////////////////////////
    // SECTION: HNSW
    // /////////////////////////////////////////////////////////////////////////////
    fn _hnsw_effect_of_m_max() -> ExperimentSetup {
        ExperimentSetup {
            n: N_100K,
            n_queries: N_10K,
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
    fn _hnsw_effect_of_ef_construction() -> ExperimentSetup {
        let ef_construction: Vec<usize> = vec![
            30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
        ];
        ExperimentSetup {
            n: N_100K,
            n_queries: N_10K,
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
    fn _hnsw_effect_of_level_norm() -> ExperimentSetup {
        let level_norm_params: Vec<f32> = (0..=18).map(|e| (e as f32) / 20.0).collect();
        ExperimentSetup {
            n: N_1M,
            n_queries: N_10K,
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
                        SliceParralelRayon,
                    )
                })
                .collect(),
            search_params: search_params(),
            random_seeds: true,
            repeats: 1,
            title: "exp_hnsw_effect_of_level_norm_1m",
        }
    }
    fn _hnsw_effect_of_ef_search_and_k() -> ExperimentSetup {
        ExperimentSetup {
            n: N_100K,
            n_queries: N_10K,
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
    fn _hnsw_effect_of_n() -> Vec<ExperimentSetup> {
        return n_log_steps_per_magnitude(N_10K, N_10M, 5)
            .into_iter()
            .map(|n| ExperimentSetup {
                n,
                n_queries: N_10K,
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
    fn _rnn_effect_of_inner_loops() -> ExperimentSetup {
        ExperimentSetup {
            n: N_100K,
            n_queries: N_10K,
            params: (1..=24)
                .map(|t_inner| {
                    ModelParams::RNNGraph(
                        RNNGraphParams {
                            outer_loops: 1,
                            inner_loops: t_inner,
                            max_neighbors_after_reverse_pruning: 40,
                            initial_neighbors: 40,
                            distance: Dot,
                        },
                        false,
                    )
                })
                .collect(),
            search_params: search_params(),
            random_seeds: false,
            repeats: 1,
            title: "exp_rnn_effect_of_inner_loops",
        }
    }
    fn _rnn_effect_of_outer_loops() -> ExperimentSetup {
        fn loops(inner_loops: usize, outer_loops: usize) -> ModelParams {
            ModelParams::RNNGraph(
                RNNGraphParams {
                    inner_loops,
                    outer_loops,
                    max_neighbors_after_reverse_pruning: 40,
                    initial_neighbors: 40,
                    distance: Dot,
                },
                false,
            )
        }
        ExperimentSetup {
            n: N_100K,
            n_queries: N_10K,
            params: vec![
                loops(12, 1),
                loops(6, 2),
                loops(4, 3),
                loops(3, 4),
                loops(2, 6),
                loops(1, 12),
            ],
            search_params: search_params(),
            random_seeds: false,
            repeats: 1,
            title: "exp_rnn_effect_of_outer_loops",
        }
    }
    fn _rnn_effect_of_num_neighbors() -> ExperimentSetup {
        ExperimentSetup {
            n: N_100K,
            n_queries: N_10K,
            params: (8..=48usize)
                .step_by(4)
                .map(|m_max| {
                    ModelParams::RNNGraph(
                        RNNGraphParams {
                            inner_loops: 3,
                            outer_loops: 4,
                            max_neighbors_after_reverse_pruning: m_max,
                            initial_neighbors: m_max,
                            distance: Dot,
                        },
                        false,
                    )
                })
                .collect(),
            search_params: search_params(),
            random_seeds: false,
            repeats: 2,
            title: "exp_rnn_effect_of_num_neighbors",
        }
    }
    fn _rnn_effect_of_ef_search_and_k() -> ExperimentSetup {
        ExperimentSetup {
            n: N_100K,
            n_queries: N_10K,
            params: vec![ModelParams::RNNGraph(
                RNNGraphParams {
                    inner_loops: 4,
                    outer_loops: 3,
                    max_neighbors_after_reverse_pruning: 40,
                    initial_neighbors: 40,
                    distance: Dot,
                },
                false,
            )],
            search_params: search_params_varied_k_and_ef(),
            random_seeds: false,
            repeats: 5,
            title: "exp_rnn_effect_of_ef_search_and_k",
        }
    }
    fn _rnn_effect_of_multi_start_points() -> ExperimentSetup {
        ExperimentSetup {
            n: N_100K,
            n_queries: N_10K,
            params: vec![ModelParams::RNNGraph(
                RNNGraphParams {
                    inner_loops: 4,
                    outer_loops: 3,
                    max_neighbors_after_reverse_pruning: 40,
                    initial_neighbors: 40,
                    distance: Dot,
                },
                false,
            )],
            search_params: (0usize..=6)
                .map(|start_candidates| SearchParams {
                    truth_distance: Dot,
                    k: 30,
                    ef: 60,
                    start_candidates: 2usize.pow(start_candidates as u32),
                    vp_max_visits: 0,
                })
                .collect(),
            random_seeds: false,
            repeats: 5,
            title: "exp_rnn_effect_of_multi_start_points",
        }
    }
    fn _rnn_effect_of_n() -> Vec<ExperimentSetup> {
        return n_log_steps_per_magnitude(N_10K, N_10M, 5)
            .into_iter()
            .map(|n| ExperimentSetup {
                n,
                n_queries: N_10K,
                params: vec![ModelParams::RNNGraph(
                    RNNGraphParams {
                        inner_loops: 4,
                        outer_loops: 3,
                        max_neighbors_after_reverse_pruning: 40,
                        initial_neighbors: 40,
                        distance: Dot,
                    },
                    false,
                )],
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
    fn _ensemble_effect_of_n() -> Vec<ExperimentSetup> {
        return n_log_steps_per_magnitude(N_10K, N_10M, 5)
            .into_iter()
            .map(|n| ExperimentSetup {
                n,
                n_queries: N_10K,
                params: vec![ModelParams::VpTreeEnsemble(
                    EnsembleParams {
                        n_vp_trees: 6,
                        max_chunk_size: 256,
                        same_chunk_m_max: 10,
                        m_max: 20,
                        m_max_0: 40,
                        distance: Distance::Dot,
                        level_norm: 0.0,
                        strategy: EnsembleStrategy::BruteForceKNN,
                        n_candidates: 0,
                    },
                    true,
                )],
                search_params: search_params(),
                random_seeds: false,
                repeats: 1,
                title: "exp_ensemple_effect_of_n",
            })
            .collect();
    }
    fn _ensemble_effect_of_n_vp_trees() -> ExperimentSetup {
        ExperimentSetup {
            n: N_100K,
            n_queries: N_10K,
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
    fn _ensemble_effect_of_brute_force_vs_rnn() -> ExperimentSetup {
        let mut params: Vec<ModelParams> = vec![];

        for max_chunk_size in [64usize, 128, 256, 512, 1024, 2048] {
            for strategy in [
                EnsembleStrategy::BruteForceKNN,
                EnsembleStrategy::RNNDescent {
                    o_loops: 1,
                    i_loops: 3,
                },
                EnsembleStrategy::RNNDescent {
                    o_loops: 2,
                    i_loops: 3,
                },
                EnsembleStrategy::RNNDescent {
                    o_loops: 3,
                    i_loops: 3,
                },
            ] {
                params.push(ModelParams::VpTreeEnsemble(
                    EnsembleParams {
                        n_vp_trees: 6,
                        max_chunk_size,
                        same_chunk_m_max: 8,
                        m_max: 20,
                        m_max_0: 40,
                        distance: Distance::Dot,
                        level_norm: 0.0,
                        strategy,
                        n_candidates: 0,
                    },
                    false,
                ));
            }
        }
        ExperimentSetup {
            n: N_100K,
            n_queries: N_10K,
            params,
            search_params: search_params(),
            random_seeds: false,
            repeats: 1,
            title: "exp_ensemble_effect_of_brute_force_vs_rnn_lowsame",
        }
    }
    fn _ensemble_effect_of_chunk_size() -> ExperimentSetup {
        ExperimentSetup {
            n: N_100K,
            n_queries: N_10K,
            params: [64usize, 128, 256, 512, 768, 1024] // 1280, 1536, 1792, 2048
                .into_iter()
                .map(|chunk_size| {
                    ModelParams::VpTreeEnsemble(
                        EnsembleParams {
                            n_vp_trees: 6,
                            max_chunk_size: chunk_size,
                            same_chunk_m_max: 40,
                            m_max: 40,
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
    fn _ensemble_effect_of_multiple_vantage_points() -> ExperimentSetup {
        ExperimentSetup {
            n: N_100K,
            n_queries: N_10K,
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
    fn _ensemble_effect_of_level_norm() -> ExperimentSetup {
        let level_norm_params: Vec<f32> = (0..=20).map(|e| (e as f32) / 20.0).collect();
        ExperimentSetup {
            n: N_100K,
            n_queries: N_10K,
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
    fn _ensemble_effect_of_m_max() -> ExperimentSetup {
        let mut params: Vec<ModelParams> = vec![];
        for m_max in (4..=40usize).step_by(4) {
            params.push(ModelParams::VpTreeEnsemble(
                EnsembleParams {
                    n_vp_trees: 6,
                    max_chunk_size: 256,
                    same_chunk_m_max: m_max,
                    m_max,
                    m_max_0: m_max,
                    distance: Distance::Dot,
                    level_norm: 0.0,
                    strategy: EnsembleStrategy::BruteForceKNN,
                    n_candidates: 0,
                },
                false,
            ));
        }
        ExperimentSetup {
            n: N_100K,
            n_queries: N_10K,
            params,
            search_params: search_params(),
            random_seeds: false,
            repeats: 1,
            title: "exp_ensemble_effect_of_m_max",
        }
    }

    fn _ensemble_effect_of_same_chunk_m_max() -> ExperimentSetup {
        ExperimentSetup {
            n: N_100K,
            n_queries: N_10K,
            params: (5usize..=7usize)
                .step_by(1)
                .into_iter()
                .map(|same_chunk_m_max| {
                    ModelParams::VpTreeEnsemble(
                        EnsembleParams {
                            n_vp_trees: 8,
                            max_chunk_size: 256,
                            same_chunk_m_max,
                            m_max: 20,
                            m_max_0: 36,
                            distance: Distance::Dot,
                            level_norm: 0.0,
                            strategy: EnsembleStrategy::BruteForceKNN,
                            n_candidates: 0,
                        },
                        true,
                    )
                })
                .collect(),
            search_params: search_params(),
            random_seeds: false,
            repeats: 6,
            title: "exp_ensemble_effect_of_same_chunk_m_max",
        }
    }
    // /////////////////////////////////////////////////////////////////////////////
    // SECTION: Chunk Stitching
    // /////////////////////////////////////////////////////////////////////////////
    fn _stitching_effect_of_fraction() -> ExperimentSetup {
        let mut params: Vec<ModelParams> = vec![];
        for fraction in 0..=20 {
            let fraction = fraction as f32 / 20.0;
            for mode in STITCHING_MODES {
                params.push(ModelParams::Stitching(StitchingParams {
                    max_chunk_size: 256,
                    same_chunk_m_max: 40,
                    neg_fraction: fraction,
                    keep_fraction: 0.0,
                    m_max: 40,
                    x_or_ef: 8,
                    only_n_chunks: None,
                    distance: Dot,
                    stitch_mode: mode,
                    n_candidates: 0,
                }));
            }
        }
        ExperimentSetup {
            n: N_100K,
            n_queries: N_10K,
            params,
            search_params: search_params(),
            random_seeds: false,
            repeats: 1,
            title: "exp_stitching_effect_of_fraction",
        }
    }
    fn _stitching_effect_of_m_max() -> ExperimentSetup {
        let mut params: Vec<ModelParams> = vec![];
        for m_max in (4..=40usize).step_by(4) {
            for stitch_mode in STITCHING_MODES_OK {
                let x_or_ef = 8;
                let neg_fraction = if stitch_mode == StitchMode::MultiEf {
                    0.2
                } else {
                    0.6
                };
                params.push(ModelParams::Stitching(StitchingParams {
                    max_chunk_size: 256,
                    same_chunk_m_max: 40,
                    neg_fraction,
                    keep_fraction: 0.0,
                    m_max,
                    x_or_ef,
                    only_n_chunks: None,
                    distance: Dot,
                    stitch_mode,
                    n_candidates: 0,
                }));
            }
        }
        ExperimentSetup {
            n: N_100K,
            n_queries: N_10K,
            params,
            search_params: search_params(),
            random_seeds: false,
            repeats: 1,
            title: "exp_stitching_effect_of_m_max",
        }
    }
    fn _stitching_effect_of_max_chunk_size() -> ExperimentSetup {
        let mut params: Vec<ModelParams> = vec![];
        for max_chunk_size in [64, 128, 256, 512, 1024] {
            for stitch_mode in STITCHING_MODES_OK {
                let x_or_ef = 8;
                let neg_fraction = if stitch_mode == StitchMode::MultiEf {
                    0.2
                } else {
                    0.6
                };
                params.push(ModelParams::Stitching(StitchingParams {
                    max_chunk_size,
                    same_chunk_m_max: 40,
                    neg_fraction,
                    keep_fraction: 0.0,
                    m_max: 40,
                    x_or_ef,
                    only_n_chunks: None,
                    distance: Dot,
                    stitch_mode,
                    n_candidates: 0,
                }));
            }
        }
        ExperimentSetup {
            n: N_100K,
            n_queries: N_10K,
            params,
            search_params: search_params(),
            random_seeds: false,
            repeats: 1,
            title: "exp_stitching_effect_of_max_chunk_size",
        }
    }
    fn _stitching_effect_of_multi_ef() -> ExperimentSetup {
        let mut params: Vec<ModelParams> = vec![];
        for max_chunk_size in [256usize, 512usize] {
            for ef in [1, 2, 4, 8, 16, 32, 64, 128] {
                let stitch_mode = if ef == 1 {
                    StitchMode::RandomNegToRandomPosAndBack
                } else {
                    StitchMode::MultiEf
                };
                params.push(ModelParams::Stitching(StitchingParams {
                    max_chunk_size,
                    same_chunk_m_max: 40,
                    neg_fraction: 0.2,
                    keep_fraction: 0.0,
                    m_max: 40,
                    x_or_ef: ef,
                    only_n_chunks: None,
                    distance: Dot,
                    stitch_mode,
                    n_candidates: 0,
                }));
            }
        }
        ExperimentSetup {
            n: N_100K,
            n_queries: N_10K,
            params,
            search_params: search_params(),
            random_seeds: false,
            repeats: 1,
            title: "exp_stitching_effect_of_multi_ef",
        }
    }
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
    RNNGraph(RNNGraphParams, bool),       // bool = threaded
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
            ModelParams::RNNGraph(e, threaded) => {
                if *threaded {
                    format!("Threaded {e:?}")
                } else {
                    format!("{e:?}")
                }
            }
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

fn load_data(data_set_size: DataSetSize, n_queries: usize, store_subsampled: bool) -> RequiredData {
    assert!(n_queries <= N_10K);
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
    println!("Loaded dataset from {data_file_path}");

    let (queries_n, dims_q) = extract_two_numbers_from_tuple_in_file_name(FILE_NAME_QUERIES);
    assert!(dims_q == dims);
    let queries_file_path = format!("{DATA_PATH}/{FILE_NAME_QUERIES}");
    let mut queries = FlatDataSet {
        dims,
        len: queries_n,
        floats: load_binary_data::<f32>(&queries_file_path, queries_n, dims),
    };
    println!("Loaded dataset from {queries_file_path}");

    // maybe subsample the loaded data:
    if let DataSetSize::Sampled(n) = data_set_size {
        const DATA_SUBSAMPLE_SEED: u64 = 123;
        let mut rng = ChaCha20Rng::seed_from_u64(DATA_SUBSAMPLE_SEED);
        let data_indices = rand::seq::index::sample(&mut rng, data.len, n).into_vec();
        data.floats = subsample_flat(&data.floats, dims, &data_indices);
        data.len = n;

        if store_subsampled {
            let subsampled_path = format!("{DATA_PATH}/laion_subsampled_({}, {}).bin", n, dims);
            let data_floats_bytes = unsafe {
                std::slice::from_raw_parts(
                    data.floats.as_ptr() as *const u8,
                    data.floats.len() * size_of::<f32>(),
                )
            };
            std::fs::write(&subsampled_path, data_floats_bytes);
            let mb = data_floats_bytes.len() / (1024usize * 1024usize);
            println!("Wrote subsampled data for n={n} ({mb} MB) to {subsampled_path}");
        }
    }

    // load the 1000 true knns for the dataset size. For fixed sizes use the ones provided by sisap, for custom sizes calculate new ones and cache them for future use (next to the official SISAP gold knns)
    let true_knns_file_name: String = match data_set_size {
        DataSetSize::_300K => FILE_NAME_GOLD_300K.to_string(), // from sisap website:  https://sisap-challenges.github.io/2024/datasets/#gold_standard_files_for_the_2024_private_queries
        DataSetSize::_10M => FILE_NAME_GOLD_10M.to_string(),   // from sisap website.
        DataSetSize::Sampled(n) => {
            format!("computed_true_knns_n={n}_(10000, 1000).bin")
        }
    };
    let true_knns_path_str = format!("{DATA_PATH}/{true_knns_file_name}");
    let true_knns_path = std::path::Path::new(&true_knns_path_str);
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
            if n_done % 100 == 0 {
                println!(
                    "    Calculated {}/{} true knns for n={}  ({}s)",
                    n_done,
                    queries.len,
                    data.len,
                    start_time.elapsed().as_secs_f32()
                );
            }
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
        println!(
            "Load true KNN for {} queries from file {}",
            queries.len, true_knns_path_str,
        );
        let start_time = Instant::now();
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
        println!("    Loading took {}s", start_time.elapsed().as_secs_f32(),);
    }
    assert!(true_knns.len() == queries.len * K);

    // subsample queries and true knn if needed:
    if n_queries < N_10K {
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
    println!("Start experiment {}", setup.to_string());
    let data_set_size = match setup.n {
        300_000 => DataSetSize::_300K,
        10_000_000 => DataSetSize::_10M, // Note: is not exactly 10M
        n => DataSetSize::Sampled(n),
    };
    let data = load_data(data_set_size, setup.n_queries, false);
    eval_models(&data, setup)
}

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
                ModelParams::RNNGraph(params, threaded) => {
                    Model::RNNGraph(RNNGraph::new(data, params, seed, threaded))
                }
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

fn gen_binary_datasets_by_subsampling_main() {
    for a in args() {
        if let Ok(n) = a.parse::<usize>() {
            let _data = load_data(DataSetSize::Sampled(n), N_10K, true);
            println!("Subsampled and stored data and true knn for n={n}");
        }
    }
}
