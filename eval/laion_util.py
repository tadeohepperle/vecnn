import pandas as pd
import numpy as np
from typing import Any, Tuple

import h5py


class LaionData:
    data: np.ndarray
    gold_queries: np.ndarray
    # not needed, can be computed instead:
    # gold_dists: np.ndarray
    # gold_knns: np.ndarray

    def subset(self,n_data: int, n_queries: int) -> Tuple[np.ndarray,np.ndarray]:
        np.random.seed(32)
        if n_data == -1:
            data = self.data
        else:
            data = self.data[np.random.choice(self.data.shape[0], n_data, replace=False)]
        if n_queries == -1:
            queries = self.gold_queries
        else:
            queries =  self.gold_queries[np.random.choice(self.gold_queries.shape[0], n_queries, replace=False)]
        return (data, queries)

def load_laion_data(laion_data_path, laion_gold_queries_path) -> LaionData:
    res = LaionData()

    f = h5py.File(laion_data_path, 'r')
    res.data = np.array(f["emb"]).astype("float32") # shape: (300000, 768)
    
    f = h5py.File(laion_gold_queries_path) 
    res.gold_queries = np.array(f["emb"]).astype("float32") # shape: (10000, 768)
    
    # not needed, can be computed instead:
    # laion_gold_path = f'{DATA_PATH}/gold-standard-dbsize=300K--public-queries-2024-laion2B-en-clip768v2-n=10k.h5'
    # f = h5py.File(laion_gold_path) 
    # res.gold_dists = np.array(f["dists"]).astype("float32") # shape: (10000, 1000), seem to be sorted in ascending order
    # res.gold_knns = np.array(f["knns"]).astype("uint64") # shape: (10000, 1000), same shape as dists.

    return res

# binary data is easy to read from rust
def convert_h5_emb_to_binary(h5_path: str, out_path_prefix: str):
    f = h5py.File(h5_path, 'r')
    data = np.array(f["emb"]).astype("float32") # shape: (300000, 768)
    file_path = f"{out_path_prefix}_{data.shape}.bin"
    data.tofile(file_path)
