import pandas as pd
import numpy as np
from typing import Any, Tuple

import h5py

# returns np.ndarray of shape (10000, 1000) and dtype uint64
def load_true_knns(path: str) -> np.ndarray:
    print(f"load_true_knns from", path)
    if path.endswith(".bin"):
        arr = np.fromfile(path, dtype=np.uint64).reshape((10000, 1000))
    elif path.endswith(".h5"):
        f = h5py.File(path, 'r')
        arr = np.array(f["knns"]).astype("uint64")
    else:
        raise Exception("Invalid path, expects h5 or .bin file")
    assert arr.shape == (10000, 1000)
    return arr

# returns np.ndarray of shape (????, 768) and dtype f32
def load_data_or_queries(path: str) -> np.ndarray:
    print(f"load_data_or_queries from", path)
    DIMS = 768
    if path.endswith(".bin"):
        flat = np.fromfile(path, dtype=np.float32)
        assert flat.shape[0] % DIMS == 0
        n = flat.shape[0] // DIMS
        assert f"({n}," in path
        arr = flat.reshape((n, DIMS))
    elif path.endswith(".h5"):
        f = h5py.File(path, 'r')
        arr = np.array(f["emb"]).astype("float32")
    else:
        raise Exception("Invalid path, expects h5 or .bin file")
    assert arr.shape[1] == DIMS
    return arr
    

# class LaionData:
#     data: np.ndarray
#     queries: np.ndarray
#     # not needed, can be computed instead:
#     # gold_dists: np.ndarray
#     # gold_knns: np.ndarray

#     def subset(self,n_data: int, n_queries: int) -> Tuple[np.ndarray,np.ndarray]:
#         np.random.seed(32)
#         if n_data == -1:
#             data = self.data
#         else:
#             data = self.data[np.random.choice(self.data.shape[0], n_data, replace=False)]
#         if n_queries == -1:
#             queries = self.queries
#         else:
#             queries =  self.queries[np.random.choice(self.queries.shape[0], n_queries, replace=False)]
#         return (data, queries)

# def load_laion_data(laion_data_path, laion_queries_path) -> LaionData:
#     res = LaionData()

#     f = h5py.File(laion_data_path, 'r')
#     res.data = np.array(f["emb"]).astype("float32") # shape: (300000, 768)
    
#     f = h5py.File(laion_queries_path) 
#     res.queries = np.array(f["emb"]).astype("float32") # shape: (10000, 768)
    
#     # not needed, can be computed instead:
#     # laion_gold_path = f'{DATA_PATH}/gold-standard-dbsize=300K--public-queries-2024-laion2B-en-clip768v2-n=10k.h5'
#     # f = h5py.File(laion_gold_path) 
#     # res.gold_dists = np.array(f["dists"]).astype("float32") # shape: (10000, 1000), seem to be sorted in ascending order
#     # res.gold_knns = np.array(f["knns"]).astype("uint64") # shape: (10000, 1000), same shape as dists.

#     return res

# # binary data is easy to read from rust, returns dimensions of the data
# def convert_h5_emb_to_binary_f32(h5_path: str, out_path_prefix: str) -> Tuple[int, int]:
#     f = h5py.File(h5_path, 'r')
#     data = np.array(f["emb"]).astype("float32") # shape: (300000, 768)
#     file_path = f"{out_path_prefix}_{data.shape}.bin"
#     data.tofile(file_path)


# # binary data is easy to read from rust
# def convert_h5_knns_to_binary_usize(h5_path: str, out_path_prefix: str):
#     f = h5py.File(h5_path, 'r')
#     data = np.array(f["knns"]).astype("uint64") # shape: (300000, 768)
#     file_path = f"{out_path_prefix}_{data.shape}.bin"
#     data.tofile(file_path)
