import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import vecnn
import time
import hnswlib
import faiss
from typing import Any, Tuple, Literal, Optional, Callable, Tuple
from dataclasses import dataclass, field

class Table:
    runs: list[Tuple[str, Any]]
    
    def __init__(self, name = "Unnamed") -> None:
        self.name = name
        self.runs = []
        # todo! date
        pass

    def add(self, **kwargs):
        run = []
        for key, value in kwargs.items():
            run.append((key, value))
        self.runs.append(run)
        
    def df(self) -> pd.DataFrame:
        # Extract all keys in order of their first appearance
        all_keys = []
        for run in self.runs:
            for key, _ in run:
                if key not in all_keys:
                    all_keys.append(key)
                    
        rows = []
        for run in self.runs:
            row_dict = {key: None for key in all_keys}
            for key, value in run:
                row_dict[key] = value
            rows.append(row_dict)
        
        df = pd.DataFrame(rows, columns=all_keys)
        return df
    
    @(staticmethod)
    def from_df(df: pd.DataFrame):
        table = Table()
        table.runs = []
        for index, row in df.iterrows():
            run = [(key, row[key]) for key in row.index if pd.notna(row[key])]
            table.runs.append(run)
        return table
    
    def save(self, filename: str) -> None:
        df = self.df()
        df.to_csv(filename, index=False)
    
    @(staticmethod)
    def load(self, filename: str):
        df = pd.read_csv(filename)
        return Table.from_df(df)

@dataclass
class ModelParams:
    model: Literal['vecnn_hsnw', 'vecnn_transition', 'vecnn_vptree', 'scipy_kdtree', 'hnswlib_hnsw', 'rustcv_hnsw', 'faiss_hnsw', 'rustcv_hnsw']
    level_norm_param: Optional[float] = None
    ef_construction: Optional[int] = None
    m_max: Optional[int] = None
    m_max_0: Optional[int] = None
    max_chunk_size: Optional[int] = None
    same_chunk_max_neighbors: Optional[int] = None
    neg_fraction: Optional[float] = None 

@dataclass
class BuildMetrics: 
    build_time: float # in seconds
    num_distance_calculations: Optional[int] = None

@dataclass
class SearchMetrics: 
    search_time: float # in seconds
    recall: float # from 0.0 to 1.0
    num_distance_calculations: Optional[int] = None

class Model:
    build_metrics: BuildMetrics
    params: ModelParams

    def __init__(self, data: np.ndarray, params: ModelParams):
        n, dim = data.shape
        self.params = params
        if params.model == 'vecnn_vptree':
            dataset = vecnn.Dataset(data) # not ideal
            start = time.time()
            vptree = vecnn.VpTree(dataset)
            build_time =  time.time() - start
            self.build_metrics = BuildMetrics(build_time=build_time, num_distance_calculations=vptree.num_distance_calculations_in_build)
            self.vecnn_vptree = vptree
        elif params.model == 'vecnn_hnsw':
            if params.level_norm_param is None or params.ef_construction is None or params.m_max is None or params.m_max_0 is None:
                raise f"for vecnn_hsnw, these params cannot be None: (level_norm_param: {params.level_norm_param}, ef_construction: {params.ef_construction}, m_max: {params.m_max}, m_max_0: {params.m_max_0})"
            dataset = vecnn.Dataset(data) # not ideal
            start = time.time()
            hnsw = vecnn.Hnsw(dataset, params=vecnn.HnswParams(params.level_norm_param, params.ef_construction, params.m_max, params.m_max_0))
            build_time =  time.time() - start
            self.build_metrics =  BuildMetrics(build_time=build_time, num_distance_calculations=hnsw.num_distance_calculations_in_build)
            self.vecnn_hsnw = hnsw
        elif params.model == 'vecnn_transition':
            dataset = vecnn.Dataset(data) # not ideal
            start = time.time()
            hnsw = vecnn.build_hnsw_by_transition(dataset, vecnn.TransitionParams(params.max_chunk_size, params.same_chunk_max_neighbors, params.neg_fraction))
            build_time = time.time() - start
            self.build_metrics =  BuildMetrics(build_time=build_time, num_distance_calculations=hnsw.num_distance_calculations_in_build)
            self.vecnn_hsnw = hnsw
        elif params.model == 'rustcv_hnsw':
            dataset = vecnn.Dataset(data) # not ideal
            start = time.time()
            hnsw = vecnn.RustCvHnsw(dataset, params.ef_construction)
            build_time =  time.time() - start
            self.build_metrics =  BuildMetrics(build_time=build_time)
            self.rustcv_hnsw = hnsw
        elif params.model == 'scipy_kdtree':
            start = time.time()
            scipy_kdtree = cKDTree(data)
            build_time =  time.time() - start
            self.build_metrics =  BuildMetrics(build_time=build_time)
            self.scipy_kdtree = scipy_kdtree
        elif params.model == 'faiss_hnsw':
            start = time.time()
            faiss_hnsw = faiss.IndexHNSWFlat(dim, params.m_max) 
            faiss_hnsw.hnsw.efConstruction = params.ef_construction
            faiss_hnsw.add(data)
            build_time = time.time() - start
            self.build_metrics =  BuildMetrics(build_time=build_time)
            self.faiss_hnsw = faiss_hnsw
            faiss_hnsw.hnsw.efSearch = 20 #todo!
        elif params.model == 'hnswlib_hnsw':
            ids = np.arange(n)
            start = time.time()
            hnswlib_hnsw = hnswlib.Index(space = 'l2', dim = dim) 
            hnswlib_hnsw.init_index(max_elements = n, ef_construction = params.ef_construction, M = params.m_max)
            hnswlib_hnsw.add_items(data, ids)
            hnswlib_hnsw.set_ef(20) #todo!
            build_time =  time.time() - start
            self.build_metrics =  BuildMetrics(build_time=build_time)
            self.hnswlib_hnsw = hnswlib_hnsw
        
            
        else: 
            raise Exception(f"Invalid 'model' type string provided: {params.model}")


    def single_row_knn_fn(self) -> Callable[[np.array, int], Tuple[np.array, float, Optional[int]]]:
        """depending on model inside this class, returns a closure with this signature:

        fn(q: 1-d np.array of float32, k: int) -> 
            indices: 1-d np.array of uint64,
            search_tiem: float, 
            num_distance_calculations: int | None
        
        """
        model = self.params.model
        if model == 'vecnn_vptree':
            def knn(query: np.ndarray, k: int) -> Tuple[np.ndarray, float, Optional[int]]:
                start = time.time()
                res = self.vecnn_vptree.knn(query, k)
                search_time = time.time() - start
                return (res.indices, search_time, res.num_distance_calculations)
            return knn
        elif model == 'vecnn_hnsw' or model == 'vecnn_transition':
            def knn(query: np.ndarray, k: int) -> Tuple[np.ndarray, float, Optional[int]]:
                start = time.time()
                res = self.vecnn_hsnw.knn(query, k)
                search_time = time.time() - start
                return (res.indices, search_time, res.num_distance_calculations)
            return knn
        elif model == 'rustcv_hnsw':
            def knn(query: np.ndarray, k: int) -> Tuple[np.ndarray, float, Optional[int]]:
                start = time.time()
                res = self.rustcv_hnsw.knn(query, k, ef=k)
                search_time = time.time() - start
                return (res.indices, search_time, res.num_distance_calculations)
            return knn
        elif model == 'scipy_kdtree':
            def knn(query: np.ndarray, k: int) -> Tuple[np.ndarray, float, Optional[int]]:
                start = time.time()
                _, indices = self.scipy_kdtree.query(query, k=k)
                search_time = time.time() - start
                return (indices.astype("uint64"), search_time, None)
            return knn
        elif model == 'faiss_hnsw':
            def knn(query: np.ndarray, k: int) -> Tuple[np.ndarray, float, Optional[int]]:
                # reshaped_query = np.reshape(query, (1, query.shape[0]))
                start = time.time()
                _, indices = self.faiss_hnsw.search(query, k=k)
                search_time = time.time() - start
                return (indices.astype("uint64")[0,:], search_time, None)
            return knn
        elif model == 'hnswlib_hnsw':
            def knn(query: np.ndarray, k: int) -> Tuple[np.ndarray, float, Optional[int]]:
                start = time.time()
                indices, _ = self.hnswlib_hnsw.knn_query(query, k=k)
                search_time = time.time() - start
                return (indices[0,:], search_time, None)
            return knn
        else: 
            raise Exception(f"Invalid 'model' type string provided: {model}")

    def knn(self, queries: np.ndarray, k: int, truth_indices: np.ndarray) -> SearchMetrics:
        """
        query: 2-d-ndarray of float32
        truth_indices: 2-d-ndarray of uint64
        """
        n_queries = queries.shape[0]
        fn: Callable[[np.ndarray, int], Tuple[np.ndarray, float, Optional[int]]] = self.single_row_knn_fn()

        search_time = 0
        recall = 0
        num_distance_calculations = 0
        for i in range(n_queries):
            if self.params.model == 'faiss_hnsw':
                (indices, duration, ndc) = fn(queries[i:i+1,:], k)
            else:
                (indices, duration, ndc) = fn(queries[i,:], k)
            search_time += duration
            recall += vecnn.knn_recall(truth_indices[i,:], indices)
            if ndc is not None:
                num_distance_calculations += ndc
        search_time /= n_queries
        recall /= n_queries
        num_distance_calculations /= n_queries

        if num_distance_calculations == 0:
            num_distance_calculations = None
        return SearchMetrics(search_time=search_time, recall=recall, num_distance_calculations=num_distance_calculations)
    
def linear_search_true_knn(data: np.ndarray, queries: np.ndarray, k: int) -> Tuple[np.ndarray, float]:
    """Args:
    data: 2-d np.ndarray of float32
    queries: 2-d np.ndarray of float32

    Returns: 
    truth_indices: 2-d np.ndarray of uint64
    search_time: float
    """
    n_queries = queries.shape[0]
    truth_indices = np.zeros((n_queries,k)).astype("uint64")  
    search_time = 0.0
    dataset = vecnn.Dataset(data)
    for i in range(n_queries):
        start = time.time()
        res = vecnn.linear_knn(dataset, queries[i,:], k)
        search_time += time.time() - start
        truth_indices[i:] = res.indices
    search_time /= n_queries
    return (truth_indices, search_time)

def benchmark_models(model_params: list[ModelParams], data: np.ndarray, queries: np.ndarray, knn_ks: list[int]) -> Table:
    table = Table()
    n, dim = data.shape  
    models: list[Model] = []
    for params in model_params:
        model = Model(data, params)
        models.append(model)

    for k in knn_ks:
        (truth_indices, linear_time) = linear_search_true_knn(data, queries, k)
        for model in models:
            metrics = model.knn(queries, k, truth_indices)
            table.add(model_name = model.params.model, 
                         data_n = n, 
                         data_dims = dim,
                         knn_k = k,
                         neg_fraction = model.params.neg_fraction,
                         build_time = model.build_metrics.build_time,
                         build_ndc = model.build_metrics.num_distance_calculations,
                         search_time = metrics.search_time,
                         search_ndc = metrics.num_distance_calculations,
                         search_recall = metrics.recall,
                      
                        )
    return table

dims = 100
data = np.random.random((1000,dims)).astype("float32")
queries = np.random.random((768,dims)).astype("float32")
k = 10
(truth_indices, search_time) = linear_search_true_knn(data, queries, k)

model_params = [
    ModelParams(model='vecnn_vptree'),
    ModelParams(model='vecnn_hnsw', level_norm_param=0.5, ef_construction=20, m_max=10, m_max_0=10),
    ModelParams(model='vecnn_transition', max_chunk_size = 64, same_chunk_max_neighbors = 30, neg_fraction = 0.0),
    ModelParams(model='vecnn_transition', max_chunk_size = 64, same_chunk_max_neighbors = 30, neg_fraction = 0.5),
    ModelParams(model='vecnn_transition', max_chunk_size = 64, same_chunk_max_neighbors = 30, neg_fraction = 1.0),
    ModelParams(model='rustcv_hnsw', level_norm_param=0.5, ef_construction=20, m_max=10, m_max_0=10),
    # ModelParams(model='faiss_hnsw', level_norm_param=0.5, ef_construction=20, m_max=10, m_max_0=10),
    ModelParams(model='hnswlib_hnsw', level_norm_param=0.5, ef_construction=20, m_max=10, m_max_0=10),
    ModelParams(model='scipy_kdtree'),
]

table = benchmark_models(model_params, data, queries, [k])
print(table.df().to_string())