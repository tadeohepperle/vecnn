import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import vecnn
import time
import hnswlib
import faiss
from typing import Any, Tuple

import h5py




class LaionData:
    data: np.ndarray
    gold_queries: np.ndarray
    gold_dists: np.ndarray
    gold_knns: np.ndarray

    def subset(self,n_data: int, n_queries: int) -> Tuple[np.ndarray,np.ndarray]:
        data = self.data[np.random.choice(self.data.shape[0], n_data, replace=False)]
        queries =  self.gold_queries[np.random.choice(self.gold_queries.shape[0], n_queries, replace=False)]
        return (data, queries)

def load_laion_data() -> LaionData:
    res = LaionData()
    DATA_PATH = '../data'
    laion_path = f'{DATA_PATH}/laion2B-en-clip768v2-n=300K.h5'
    laion_gold_queries_path = f'{DATA_PATH}/public-queries-2024-laion2B-en-clip768v2-n=10k.h5'
    laion_gold_path = f'{DATA_PATH}/gold-standard-dbsize=300K--public-queries-2024-laion2B-en-clip768v2-n=10k.h5'
    
    f = h5py.File(laion_path, 'r')
    res.data = np.array(f["emb"]).astype("float32") # shape: (300000, 768)
    
    f = h5py.File(laion_gold_queries_path) 
    res.gold_queries = np.array(f["emb"]).astype("float32") # shape: (10000, 768)
    
    f = h5py.File(laion_gold_path) 
    res.gold_dists = np.array(f["dists"]).astype("float32") # shape: (10000, 1000), seem to be sorted in ascending order
    res.gold_knns = np.array(f["knns"]).astype("uint64") # shape: (10000, 1000), same shape as dists.

    return res