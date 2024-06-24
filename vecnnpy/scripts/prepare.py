import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import vecnn
import time
import hnswlib
import faiss
from typing import Any, Tuple

import h5py

DATA_PATH = '../../data'

# import os
# print(os.path.isdir(DATA_PATH))

laion_path = f'{DATA_PATH}/laion2B-en-clip768v2-n=300K.h5'
laion_gold_queries_path = f'{DATA_PATH}/public-queries-2024-laion2B-en-clip768v2-n=10k.h5'
laion_gold_path = f'{DATA_PATH}/gold-standard-dbsize=300K--public-queries-2024-laion2B-en-clip768v2-n=10k.h5'

f = h5py.File(laion_path, 'r')
laion_data = np.array(f["emb"]).astype("float32") # shape: (300000, 768)

f = h5py.File(laion_gold_queries_path) 
laion_gold_queries = np.array(f["emb"]).astype("float32") # shape: (10000, 768)

f = h5py.File(laion_gold_path) 
laion_gold_dists = np.array(f["dists"]).astype("float32") # shape: (10000, 1000), seem to be sorted in ascending order
laion_gold_knns = np.array(f["knns"]).astype("uint64") # shape: (10000, 1000), same shape as dists.

DATA_N = 10000
QUERIES_N = 600
K = laion_gold_knns.shape[1]

# np.random.seed(42)
# small_laion_data = laion_data[np.random.choice(laion_data.shape[0], DATA_N, replace=False)]

query_indices = np.random.choice(laion_gold_queries.shape[0], QUERIES_N, replace=False)
small_laion_queries = laion_gold_queries[query_indices]
small_laion_gold_knns = laion_gold_knns[query_indices]


def linear_search_knn(data: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Args:
    data: 2-d np.ndarray of float32
    queries: 2-d np.ndarray of float32

    Returns: 
    truth_indices: 2-d np.ndarray of uint64
    search_time: float
    """
    n_queries = queries.shape[0]
    assert(data.shape[1] == queries.shape[1])
    truth_indices = np.zeros((n_queries,k)).astype("uint64")  
    dataset = vecnn.Dataset(data)
    for i in range(n_queries):
        print(i, "/", n_queries)
        res = vecnn.linear_knn(dataset, queries[i,:], k, "cos")
        truth_indices[i,:] = res.indices
    return truth_indices

found = linear_search_knn(laion_data, small_laion_queries, K)

res = []
for i in range(QUERIES_N):
    a = np.intersect1d(small_laion_gold_knns[i,:], found[i,:], assume_unique=False, return_indices=False)
    res.append(len(a) / K)

print(res)
print(np.array(res).mean())
# print(true_knn_of_small_laion_queries)