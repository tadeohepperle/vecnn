"""
Generates graphs and tables from the experiment results.
"""

import matplotlib.pyplot as plt
from squasher import Experiment, save_all_experiments_as_latex_tables

# Section 1: hnsw questions
# how does varying ef affect recall, build time, and search time?

# how does varying n affect recall, build time, and search time?

# HNSW_EFFECT_OF_N_PATHS = [
#     "../vecnn/experiments_server/hnsw_effect_of_n_n=10000_queries_n=1000.csv",
#     "../vecnn/experiments_server/hnsw_effect_of_n_n=30000_queries_n=1000.csv",
#     "../vecnn/experiments_server/hnsw_effect_of_n_n=100000_queries_n=1000.csv",
#     "../vecnn/experiments_server/hnsw_effect_of_n_n=300000_queries_n=1000.csv",
#     "../vecnn/experiments_server/hnsw_effect_of_n_n=1000000_queries_n=1000.csv",
#     "../vecnn/experiments_server/hnsw_effect_of_n_n=3000000_queries_n=1000.csv",
#     "../vecnn/experiments_server/hnsw_effect_of_n_n=10000000_queries_n=1000.csv",
# ]

def thesis_exp(prefix: str, name: str |None = None) -> Experiment:
    if name is None:
        name = prefix
    return Experiment("./thesis_experiments/" + prefix, name = name)

hnsw_n = thesis_exp("hnsw_effect_of_n").filter("model", "SliceS2HnswParams").sort_by("n").discard_cols(["build_ndc", "ndc"])
hnsw_n.print()
# todo: run again with more log steps (5) between each magnitude

hnsw_m_max = thesis_exp("exp_hnsw_effect_of_m_max")
hnsw_m_max.print()
# todo: step m_max in 1 steps instead of 4 steps

hnsw_ef_construction = thesis_exp("exp_hnsw_effect_of_ef_construction")
hnsw_ef_construction.print()

hnsw_ef_search = thesis_exp("exp_hnsw_effect_of_ef_search", "hnsw_ef_search").filter("k", 30)
hnsw_ef_search.print()

hnsw_k = thesis_exp("exp_hnsw_effect_of_ef_search", "hnsw_k").filter_col_eq("ef", "k")
hnsw_k.print()

hnsw_level_norm = thesis_exp("exp_hnsw_effect_of_level_norm")
hnsw_level_norm.print()


rnn_ef_search = thesis_exp("exp_rnn_effect_of_ef_search", "rnn_ef_search").filter("k", 30)
rnn_ef_search.print()

rnn_k = thesis_exp("exp_rnn_effect_of_ef_search", "rnn_k").filter_col_eq("ef", "k")
rnn_k.print()

rnn_inner_loops = thesis_exp("exp_rnn_effect_of_inner", "rnn_inner_loops")
rnn_inner_loops.print()

rnn_outer_loops = thesis_exp("exp_rnn_effect_of_outer", "rnn_outer_loops")
rnn_outer_loops.print()


ensemble_chunk_size = thesis_exp("exp_ensemble_effect_of_chunk_size")
ensemble_chunk_size.print()

ensemble_level_norm = thesis_exp("exp_ensemble_effect_of_level_norm")
ensemble_level_norm.print()

ensemble_n_vp_trees = thesis_exp("exp_ensemble_effect_of_n_vp_trees")
ensemble_n_vp_trees.print()
# todo: search time_ms seems a bit weird, run with more repeats

ensemble_multiple_vantage_points = thesis_exp("exp_ensemble_effect_of_multiple_vantage_points")
ensemble_multiple_vantage_points.print()

compare_10m = thesis_exp("compare_10m")
compare_10m.print()

compare_100m = thesis_exp("compare_100m")
compare_100m.print()

save_all_experiments_as_latex_tables()
