import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import vecnn
import time
import datetime
import hnswlib
import faiss
from typing import Any, Tuple, Literal, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import laion


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
    
    def save(self,filename: str = "") -> str:
        df = self.df()
        if filename == "":
            filename = f"experiments/experiment{datetime.datetime.now()}.csv"
        df.to_csv(filename, index=False)
        return filename
    
    @(staticmethod)
    def load(self, filename: str):
        df = pd.read_csv(filename)
        return Table.from_df(df)
    

@dataclass
class BuildMetrics: 
    build_time: float # in seconds
    num_distance_calculations: Optional[int] = None

@dataclass
class SearchMetrics: 
    search_time: float # in seconds
    recall: float # from 0.0 to 1.0
    num_distance_calculations: Optional[int] = None




# type ModelKind = Literal['vecnn_hsnw', 'vecnn_transition', 'vecnn_rnn_descent' 'vecnn_vptree', 'scipy_kdtree', 'hnswlib_hnsw', 'rustcv_hnsw', 'jpboth_hnsw', 'faiss_hnsw']
class ModelParams:
    kind: str
    def __init__(self, kind: str, **kwargs) -> None:
        self.kind = kind
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> dict:
        return vars(self)

class SearchParams:
    k: int
    distance_fn: str
    def __init__(self, k: int, distance_fn: str, **kwargs) -> None:
        self.k = k
        self.distance_fn = distance_fn
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> dict:
        return vars(self)


class Model:
    build_metrics: BuildMetrics
    params: ModelParams

    def __init__(self, data: np.ndarray, params: ModelParams):
        n, dim = data.shape
        self.params = params
        if params.kind == 'vecnn_vptree':
            dataset = vecnn.Dataset(data) # not ideal
            start = time.time()
            vptree = vecnn.VpTree(dataset)
            build_time =  time.time() - start
            self.build_metrics = BuildMetrics(build_time=build_time, num_distance_calculations=vptree.num_distance_calculations_in_build)
            self.vecnn_vptree = vptree
        elif params.kind == 'vecnn_hnsw':
            if params.level_norm_param is None or params.ef_construction is None or params.m_max is None or params.m_max_0 is None:
                raise f"for vecnn_hsnw, these params cannot be None: (level_norm_param: {params.level_norm_param}, ef_construction: {params.ef_construction}, m_max: {params.m_max}, m_max_0: {params.m_max_0})"
            dataset = vecnn.Dataset(data) # not ideal
            start = time.time()
            hnsw = vecnn.Hnsw(dataset, params.level_norm_param, params.ef_construction, params.m_max, params.m_max_0, params.distance_fn)
            build_time =  time.time() - start
            self.build_metrics =  BuildMetrics(build_time=build_time, num_distance_calculations=hnsw.num_distance_calculations_in_build)
            self.vecnn_hsnw = hnsw
        elif params.kind == 'vecnn_transition':
            dataset = vecnn.Dataset(data) # not ideal
            start = time.time()
            hnsw = vecnn.build_hnsw_by_transition(dataset, vecnn.TransitionParams(params.max_chunk_size, params.same_chunk_max_neighbors, params.neg_fraction))
            build_time = time.time() - start
            self.build_metrics =  BuildMetrics(build_time=build_time, num_distance_calculations=hnsw.num_distance_calculations_in_build)
            self.vecnn_hsnw = hnsw
        elif params.kind == 'rustcv_hnsw':
            dataset = vecnn.Dataset(data) # not ideal
            start = time.time()
            hnsw = vecnn.RustCvHnsw(dataset, params.ef_construction)
            build_time =  time.time() - start
            self.build_metrics =  BuildMetrics(build_time=build_time)
            self.rustcv_hnsw = hnsw
        elif params.kind == 'jpboth_hnsw':
            dataset = vecnn.Dataset(data) # not ideal
            start = time.time()
            hnsw = vecnn.JpBothHnsw(dataset, params.ef_construction, params.m_max)
            build_time =  time.time() - start
            self.build_metrics = BuildMetrics(build_time=build_time)
            self.jpboth_hnsw = hnsw
        elif params.kind == 'vecnn_rnn_descent':
            dataset = vecnn.Dataset(data) # not ideal
            start = time.time()
            rnn_graph = vecnn.RNNGraph(dataset, params.outer_loops, params.inner_loops, params.max_neighbors_after_reverse_pruning, params.initial_neighbors, params.distance_fn )
            build_time =  time.time() - start
            self.build_metrics =  BuildMetrics(build_time=build_time, num_distance_calculations=rnn_graph.num_distance_calculations_in_build)
            self.vecnn_rnn_descent = rnn_graph
            pass
        elif params.kind == 'scipy_kdtree':
            start = time.time()
            scipy_kdtree = cKDTree(data)
            build_time =  time.time() - start
            self.build_metrics =  BuildMetrics(build_time=build_time)
            self.scipy_kdtree = scipy_kdtree
        elif params.kind == 'faiss_hnsw':
            start = time.time()
            faiss_hnsw = faiss.IndexHNSWFlat(dim, params.m_max) 
            faiss_hnsw.hnsw.efConstruction = params.ef_construction
            faiss_hnsw.add(data)
            build_time = time.time() - start
            self.build_metrics =  BuildMetrics(build_time=build_time)
            self.faiss_hnsw = faiss_hnsw
        elif params.kind == 'hnswlib_hnsw':
            ids = np.arange(n)
            start = time.time()
            space = ''
            if params.distance_fn == "l2":
                space = "l2"
            elif params.distance_fn == "dot":
                space = "ip" # see https://github.com/nmslib/hnswlib
            elif params.distance_fn == "cos":
                space = 'cosine'
            hnswlib_hnsw = hnswlib.Index(space = space, dim = dim) 
            hnswlib_hnsw.init_index(max_elements = n, ef_construction = params.ef_construction, M = params.m_max)
            hnswlib_hnsw.add_items(data, ids, num_threads = 1) #   # add_items(data, ids, num_threads = -1, replace_deleted = False)
            build_time =  time.time() - start
            self.build_metrics =  BuildMetrics(build_time=build_time)
            self.hnswlib_hnsw = hnswlib_hnsw
        else: 
            raise Exception(f"Invalid 'model' type string provided: {params.kind}")


    def single_row_knn_fn(self) -> Callable[[np.array, SearchParams], Tuple[np.array, float, Optional[int]]]:
        """depending on model inside this class, returns a closure with this signature:

        fn(q: 1-d np.array of float32, k: int) -> 
            indices: 1-d np.array of uint64,
            search_tiem: float, 
            num_distance_calculations: int | None
        
        """
        model = self.params.kind
        if model == 'vecnn_vptree':
            def knn(query: np.ndarray, params: SearchParams) -> Tuple[np.ndarray, float, Optional[int]]:
                start = time.time()
                res = self.vecnn_vptree.knn(query, params.k)
                search_time = time.time() - start
                return (res.indices, search_time, res.num_distance_calculations)
            return knn
        elif model == 'vecnn_hnsw' or model == 'vecnn_transition':
            def knn(query: np.ndarray, params: SearchParams) -> Tuple[np.ndarray, float, Optional[int]]:
                start = time.time()
                res = self.vecnn_hsnw.knn(query, params.k, params.ef)
                search_time = time.time() - start
                return (res.indices, search_time, res.num_distance_calculations)
            return knn
        elif model == 'rustcv_hnsw':
            def knn(query: np.ndarray, params: SearchParams) -> Tuple[np.ndarray, float, Optional[int]]:
                start = time.time()
                res = self.rustcv_hnsw.knn(query, params.k, params.ef)
                search_time = time.time() - start
                return (res.indices, search_time, res.num_distance_calculations)
            return knn
        elif model == 'jpboth_hnsw':
            def knn(query: np.ndarray, params: SearchParams) -> Tuple[np.ndarray, float, Optional[int]]:
                start = time.time()
                res = self.jpboth_hnsw.knn(query, params.k, params.ef)
                search_time = time.time() - start
                return (res.indices, search_time, res.num_distance_calculations)
            return knn
        elif model == 'vecnn_rnn_descent':
            def knn(query: np.ndarray, params: SearchParams) -> Tuple[np.ndarray, float, Optional[int]]:
                start = time.time()
                res = self.vecnn_rnn_descent.knn(query, k=params.k, start_candidates=params.start_candidates) # todo! not hardcode the 10 initial neighbors
                search_time = time.time() - start
                return (res.indices, search_time, res.num_distance_calculations)
            return knn
        elif model == 'scipy_kdtree':
            def knn(query: np.ndarray, params: SearchParams) -> Tuple[np.ndarray, float, Optional[int]]:
                start = time.time()
                _, indices = self.scipy_kdtree.query(query, k=params.k)
                search_time = time.time() - start
                return (indices.astype("uint64"), search_time, None)
            return knn
        elif model == 'faiss_hnsw':
            def knn(query: np.ndarray, params: SearchParams) -> Tuple[np.ndarray, float, Optional[int]]:
                # reshaped_query = np.reshape(query, (1, query.shape[0]))
                start = time.time()
                _, indices = self.faiss_hnsw.search(query, k=params.k)
                search_time = time.time() - start
                return (indices.astype("uint64")[0,:], search_time, None)
            return knn
        elif model == 'hnswlib_hnsw':
            def knn(query: np.ndarray, params: SearchParams) -> Tuple[np.ndarray, float, Optional[int]]:
                start = time.time()
                indices, _ = self.hnswlib_hnsw.knn_query(query, k=params.k)
                search_time = time.time() - start
                return (indices[0,:], search_time, None)
            return knn
        else: 
            raise Exception(f"Invalid 'model' type string provided: {model}")

    def knn(self, queries: np.ndarray, params: SearchParams, truth_indices: np.ndarray) -> SearchMetrics:
        """
        query: 2-d-ndarray of float32
        truth_indices: 2-d-ndarray of uint64
        """
        n_queries = queries.shape[0]

        if hasattr(self, "hnswlib_hnsw"):
            print("set ef to ", params.ef)
            self.hnswlib_hnsw.set_ef(params.ef)
        elif hasattr(self, "faiss_hnsw"):
            self.faiss_hnsw.hnsw.efSearch = params.ef #todo!

        fn = self.single_row_knn_fn()

        search_time = 0
        recall = 0
        num_distance_calculations = 0
        for i in range(n_queries):
            if self.params.kind == 'faiss_hnsw':
                (indices, duration, ndc) = fn(queries[i:i+1,:], params)
            else:
                (indices, duration, ndc) = fn(queries[i,:], params)
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
    
def linear_search_true_knn(data: np.ndarray, queries: np.ndarray, k: int, distance_fn: str) -> Tuple[np.ndarray, float]:
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
        res = vecnn.linear_knn(dataset, queries[i,:], k, distance_fn)
        search_time += time.time() - start
        truth_indices[i:] = res.indices
    search_time /= n_queries
    return (truth_indices, search_time)

def benchmark_models(model_params: list[ModelParams], data: np.ndarray, queries: np.ndarray, search_params: list[SearchParams]) -> Table:
    table = Table()
    n, dim = data.shape  
    models: list[Model] = []
    i = 0
    print(f"Build {len(model_params)} models")
    for params in model_params:
        model = Model(data, params)
        models.append(model)
        print(f"    Built model {i+1}/{len(model_params)} in {int(model.build_metrics.build_time)}s:  ({params.to_dict()})")
        i+=1

    for s in search_params:
        print(f"Benchmark {len(models)} models for search params: {s.to_dict()}")
        (truth_indices, linear_time) = linear_search_true_knn(data, queries, s.k, s.distance_fn)
        i = -1
        for model in models:
            i+=1
            print(f"    Model {i+1}/{len(models)}")
            metrics = model.knn(queries, s, truth_indices)
            s_dict = s.to_dict().copy()
            del s_dict["distance_fn"]

            table.add(
                **model.params.to_dict(),
                **s_dict,
                data_n = n, 
                data_dims = dim,
                build_time = model.build_metrics.build_time,
                build_ndc = model.build_metrics.num_distance_calculations,
                search_time = metrics.search_time,
                search_ndc = metrics.num_distance_calculations,
                search_recall = metrics.recall
            )
    return table

# dims = 100
# data = np.random.random((1000,dims)).astype("float32")
# queries = np.random.random((300,dims)).astype("float32")
# k = 10

laion_data = laion.load_laion_data()
data, queries = laion_data.subset(100000, 1000)


ef_construction =20
m_max = 20
# (truth_indices, search_time) = linear_search_true_knn(data, queries, k, "dot")
model_params: list[ModelParams] = [
    ModelParams('rustcv_hnsw', ef_construction=ef_construction),
    ModelParams('jpboth_hnsw', ef_construction=ef_construction, m_max=m_max),
    ModelParams('vecnn_hnsw', level_norm_param=0.5, ef_construction=ef_construction, m_max=m_max, m_max_0=m_max*2, distance_fn = "dot"),
    ModelParams('hnswlib_hnsw', ef_construction=ef_construction, m_max=m_max, distance_fn = "dot"),
    ModelParams('faiss_hnsw', ef_construction=ef_construction, m_max=m_max, distance_fn = "dot"),
    # ModelParams('vecnn_rnn_descent',outer_loops=50, inner_loops=1, max_neighbors_after_reverse_pruning=4, initial_neighbors = 10, distance_fn = "dot"),
    # ModelParams('vecnn_vptree'),
]
search_params: list[SearchParams] = [
    SearchParams(30, "dot", ef = 60, start_candidates = 10) # make sure ef >= k
]

start = time.time()
table = benchmark_models(model_params, data, queries, search_params)
total = time.time() -start
print(table.df().to_string())

filename = table.save(f"experiments/experiment{datetime.datetime.now()} Total time: {int(total)}s.csv")

HNSW_LIB_BEST_CONSTRUCTION_TIME = [
    ModelParams('rustcv_hnsw', ef_construction=200),
    ModelParams('vecnn_hnsw', level_norm_param=0.3, ef_construction=200, m_max=30, m_max_0=30, distance_fn = "dot"),
    ModelParams('hnswlib_hnsw', ef_construction=200, m_max=30, distance_fn = "dot"),
]