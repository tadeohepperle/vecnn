"""
Generates graphs and tables from the experiment results.
"""

import matplotlib.pyplot as plt
from squasher import Experiment

# Section 1: hnsw questions
# how does varying ef affect recall, build time, and search time?

# how does varying n affect recall, build time, and search time?

HNSW_EFFECT_OF_N_PATHS = [
    "../vecnn/experiments_server/hnsw_effect_of_n_n=10000_queries_n=1000.csv",
    "../vecnn/experiments_server/hnsw_effect_of_n_n=30000_queries_n=1000.csv",
    "../vecnn/experiments_server/hnsw_effect_of_n_n=100000_queries_n=1000.csv",
    "../vecnn/experiments_server/hnsw_effect_of_n_n=300000_queries_n=1000.csv",
    "../vecnn/experiments_server/hnsw_effect_of_n_n=1000000_queries_n=1000.csv",
    "../vecnn/experiments_server/hnsw_effect_of_n_n=3000000_queries_n=1000.csv",
    "../vecnn/experiments_server/hnsw_effect_of_n_n=10000000_queries_n=1000.csv",
]

ex = Experiment(HNSW_EFFECT_OF_N_PATHS, "hnsw_effect_of_n").filter("model", "SliceS2HnswParams")
ex.print()
# ex.print_latex(["n", "model", "recall"])
ex.print_latex()
ex.plot("n", "build_ms", log_x=True, log_y=True)
ex.plot("n", "time_ms", log_x=True, log_y=False)
ex.plot("n", "recall", log_x=True, log_y=False)

ex.plot()
