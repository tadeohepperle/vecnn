import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import vecnn
import time
import datetime
import hnswlib
import faiss
from typing import Any, Tuple, Literal, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field, asdict
import laion_util
import h5py

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
    build_secs: float
    num_distance_calculations: Optional[int] = None

@dataclass
class SearchMetrics: 
    search_ms: float # in ms
    recall: float # from 0.0 to 1.0
    num_distance_calculations: Optional[int] = None

@dataclass
class VpTreeParams: 
    n_candidates: int = 0
    threaded: bool = False

@dataclass
class HnswParams:
    implementation: Literal["vecnn", "rustcv", "jpboth", "hnswlib", "faiss"]
    ef_construction: int = 60
    m_max: int = 20
    m_max_0: int = 40
    level_norm: float = 0.3
    threaded: bool = False

@dataclass
class RNNGraphParams:
    inner_loops: bool = 4
    outer_loops: int = 3
    m_initial: int = 40
    m_pruned: int = 40
    threaded: bool = False

@dataclass
class StitchingParams:
    """
    method1: random negative to positive center
    method2: random negative to random positive
    method3: x candidates in pos and neg half, do x searches between the closest ones.
    method4: method2 but with mofe than 1 ef, uses x param as ef
    """
    method: Literal["method1", "method2", "method3", "method4"]
    n_candidates: int = 0      # for vptree  construction
    max_chunk_size: int = 256  # max size of the chunks the vptree is split into
    same_chunk_m_max: int = 20 # max neighbors within each chunk.
    m_max: int = 20            # of the resulting graph
    fraction: float = 0.3      # of negative half sampled
    x_or_ef: int = 3                 # ef or x for method2 or method 3
    threaded: bool = False

@dataclass
class EnsembleParams:
    level_norm: float = 0.0
    n_vp_trees: int = 6
    n_candidates: int = 0      # for vptree  construction
    max_chunk_size: int = 256  # max size of the chunks the vptree is split into
    same_chunk_m_max: int = 20 # max neighbors within each chunk.
    m_max: int = 20            # of the resulting graph
    m_max_0: int = 40          # of the resulting graph
    rnn_inner_loops: int = 0
    rnn_outer_loops: int = 0
    threaded: bool = False

@dataclass
class SearchParams:
    k: int
    ef: int
    start_candidates: int

Distance = Literal["l2", "dot", "cos"]
ModelParams = Union[HnswParams, RNNGraphParams, StitchingParams, EnsembleParams, VpTreeParams]

def model_params_to_str(params: ModelParams) -> str:
    if isinstance(params, HnswParams):
        return f"Hnsw_{params.implementation} {{ef_constr: {params.ef_construction}, m_max: {params.m_max}, m_max0: {params.m_max_0}, level_norm: {params.level_norm}, threaded: {params.threaded}}}"
    elif isinstance(params, RNNGraphParams):
        return f"RNNGraph {{outer_loops: {params.outer_loops}, inner_loops: {params.inner_loops}, m_pruned: {params.m_pruned}, m_initial: {params.m_initial}, threaded: {params.threaded}}}"
    elif isinstance(params, StitchingParams):
        return f"Stitching {{method: {params.method}, n_candidates: {params.n_candidates}, max_chunk_size: {params.max_chunk_size}, same_chunk_m_max: {params.same_chunk_m_max}, m_max: {params.m_max}, fraction: {params.fraction}, x_or_ef: {params.x_or_ef}, threaded: {params.threaded}}}"
    elif isinstance(params, EnsembleParams):
        if params.rnn_inner_loops == 0 or params.rnn_outer_loops == 0:
            return f"Ensemble {{level_norm: {params.level_norm}, n_vp_trees: {params.n_vp_trees}, n_candidates: {params.n_candidates}, max_chunk_size: {params.max_chunk_size}, same_chunk_m_max: {params.same_chunk_m_max}, m_max: {params.m_max}, m_max0: {params.m_max_0}, threaded: {params.threaded}}}"
        else: 
            return f"Ensemble {{level_norm: {params.level_norm}, n_vp_trees: {params.n_vp_trees}, n_candidates: {params.n_candidates}, max_chunk_size: {params.max_chunk_size}, same_chunk_m_max: {params.same_chunk_m_max}, m_max: {params.m_max}, m_max0: {params.m_max_0}, threaded: {params.threaded}, rnn_inner_loops: {params.rnn_inner_loops}, rnn_outer_loops: {params.rnn_outer_loops}}}"
    elif isinstance(params, VpTreeParams):
        return f"VpTree {{n_candidates: {params.n_candidates}, threaded: {params.threaded}}}"
    else: 
        raise Exception(f"Invalid ModelParams")

def check_params(params: ModelParams):
    if isinstance(params, HnswParams):
        if params.implementation == "vecnn":
            pass
        elif params.implementation == "jpboth":
            # maybe assert that dot distance is used?
            pass
        elif params.implementation == "rustcv":
            assert(params.threaded == False)
            assert(params.m_max_0 == 40) # fixed by const generics in rust code!
            assert(params.m_max == 20)   # fixed by const generics in rust code!
            pass
        elif params.implementation == "faiss":
            assert(params.threaded == True)
            pass
        elif params.implementation == "hnswlib":
            pass
        else: 
            raise Exception(f"Invalid hnsw implementation: {params.implementation}")
    elif isinstance(params, RNNGraphParams):
        pass
    elif isinstance(params, StitchingParams):
        pass
    elif isinstance(params, EnsembleParams):
        pass
    elif isinstance(params, VpTreeParams):
        pass
    else: 
        raise Exception(f"Invalid ModelParams")


class Model:
    build_metrics: BuildMetrics

    params: ModelParams
    seed: int
    distance: Distance

    vecnn_vptree: Optional[vecnn.VpTree] = None
    vecnn_rnn_graph: Optional[vecnn.RNNGraph] = None
    vecnn_hnsw: Optional[vecnn.Hnsw] = None
    jpboth_hnsw: Optional[vecnn.JpBothHnsw] = None
    rustcv_hnsw: Optional[vecnn.RustCvHnsw] = None
    hnswlib_hnsw: Optional[hnswlib.Index] = None
    faiss_hnsw: Optional[faiss.IndexHNSWFlat] = None

    def __init__(self, data: np.ndarray, params: ModelParams, distance: Distance = "dot", seed: int = 32, ):
        n, dim = data.shape
        self.params = params
        self.seed = seed
        self.distance = distance

        if isinstance(params, HnswParams):
            if params.implementation == "vecnn":
                dataset = vecnn.Dataset(data) # not ideal
                start = time.time()
                use_const_impl = False # TODO: remove this later, is more for checking if the slice HNSW impl is worth it.
                if params.m_max_0 == 0:
                    params.m_max_0 = params.m_max*2
                hnsw = vecnn.Hnsw(dataset, params.level_norm, params.ef_construction, params.m_max, params.m_max_0, distance, params.threaded, use_const_impl, seed)
                build_secs =  time.time() - start
                self.build_metrics =  BuildMetrics(build_secs, hnsw.num_distance_calculations_in_build)
                self.vecnn_hnsw = hnsw
            elif params.implementation == "jpboth":
                dataset = vecnn.Dataset(data) # not ideal
                start = time.time()
                hnsw = vecnn.JpBothHnsw(dataset, params.ef_construction, params.m_max, params.threaded)
                build_secs =  time.time() - start
                self.build_metrics = BuildMetrics(build_secs)
                self.jpboth_hnsw = hnsw
            elif params.implementation == "rustcv":
                dataset = vecnn.Dataset(data) # not ideal
                start = time.time()
                hnsw = vecnn.RustCvHnsw(dataset, params.ef_construction)
                build_secs =  time.time() - start
                self.build_metrics =  BuildMetrics(build_secs)
                self.rustcv_hnsw = hnsw
            elif params.implementation == "faiss":
                start = time.time()
                faiss_hnsw = faiss.IndexHNSWFlat(dim, params.m_max) 
                faiss_hnsw.hnsw.efConstruction = params.ef_construction
                faiss_hnsw.add(data)
                build_secs = time.time() - start
                self.build_metrics =  BuildMetrics(build_secs)
                self.faiss_hnsw = faiss_hnsw
            elif params.implementation == "hnswlib":
                ids = np.arange(n)
                start = time.time()
                space = ''
                if distance == "l2":
                    space = "l2"
                elif distance == "dot":
                    space = "ip" # see https://github.com/nmslib/hnswlib
                elif distance == "cos":
                    space = 'cosine'
                hnswlib_hnsw = hnswlib.Index(space = space, dim = dim) 
                hnswlib_hnsw.init_index(max_elements = n, ef_construction = params.ef_construction, M = params.m_max)
                hnswlib_hnsw.add_items(data, ids, num_threads = -1 if params.threaded else 1) #   # add_items(data, ids, num_threads = -1, replace_deleted = False)
                build_secs =  time.time() - start
                self.build_metrics =  BuildMetrics(build_secs)
                self.hnswlib_hnsw = hnswlib_hnsw
            else: 
                raise Exception(f"Invalid hnsw implementation: {params.implementation}")
        elif isinstance(params, RNNGraphParams):
            dataset = vecnn.Dataset(data) # not ideal
            start = time.time()
            rnn_graph = vecnn.RNNGraph(dataset, params.outer_loops, params.inner_loops, params.m_pruned, params.m_initial, params.threaded, distance, seed)
            build_secs =  time.time() - start
            self.build_metrics =  BuildMetrics(build_secs, rnn_graph.num_distance_calculations_in_build)
            self.vecnn_rnn_graph = rnn_graph
        elif isinstance(params, StitchingParams):
            dataset = vecnn.Dataset(data) # not ideal
            start = time.time()
            hnsw = vecnn.build_hnsw_by_chunk_stitching(dataset, params.method, params.n_candidates, params.max_chunk_size, params.same_chunk_m_max, params.m_max, params.fraction, params.x_or_ef, params.threaded, distance, seed)
            build_secs =  time.time() - start
            self.build_metrics =  BuildMetrics(build_secs, hnsw.num_distance_calculations_in_build)
            self.vecnn_hnsw = hnsw
        elif isinstance(params, EnsembleParams):
            dataset = vecnn.Dataset(data) # not ideal
            start = time.time()
            hnsw = vecnn.build_hnsw_by_vp_tree_ensemble(dataset, params.level_norm, params.n_vp_trees, params.n_candidates, params.max_chunk_size, params.same_chunk_m_max, params.m_max, params.m_max_0, params.rnn_inner_loops, params.rnn_outer_loops, params.threaded, distance, seed)
            build_secs =  time.time() - start
            self.build_metrics =  BuildMetrics(build_secs, hnsw.num_distance_calculations_in_build)
            self.vecnn_hnsw = hnsw
        elif isinstance(params, VpTreeParams):
            dataset = vecnn.Dataset(data) # not ideal
            start = time.time()
            vptree = vecnn.VpTree(dataset, params.n_candidates, params.threaded, distance, seed)
            build_secs =  time.time() - start
            self.build_metrics = BuildMetrics(build_secs, vptree.num_distance_calculations_in_build)
            self.vecnn_vptree = vptree
        else: 
            raise Exception(f"Invalid ModelParams")

    def single_row_knn_fn(self) -> Callable[[np.array, SearchParams], Tuple[np.array, float, Optional[int]]]:
        """depending on model inside this class, returns a closure with this signature:

        fn(q: 1-d np.array of float32, k: int) -> 
            indices: 1-d np.array of uint64,
            search_tiem: float, 
            num_distance_calculations: int | None
        
        """
        if self.vecnn_hnsw is not None:
            def knn(query: np.ndarray, params: SearchParams) -> Tuple[np.ndarray, float, Optional[int]]:
                start = time.time()
                res = self.vecnn_hnsw.knn(query, params.k, params.ef)
                search_time = time.time() - start
                return (res.indices, search_time, res.num_distance_calculations)
            return knn
        elif self.vecnn_vptree is not None:
            def knn(query: np.ndarray, params: SearchParams) -> Tuple[np.ndarray, float, Optional[int]]:
                start = time.time()
                res = self.vecnn_vptree.knn(query, params.k)
                search_time = time.time() - start
                return (res.indices, search_time, res.num_distance_calculations)
            return knn
        elif self.vecnn_rnn_graph is not None:
            def knn(query: np.ndarray, params: SearchParams) -> Tuple[np.ndarray, float, Optional[int]]:
                start = time.time()
                res = self.vecnn_rnn_graph.knn(query, params.k, params.ef, params.start_candidates) # todo! not hardcode the 10 initial neighbors
                search_time = time.time() - start
                return (res.indices, search_time, res.num_distance_calculations)
            return knn
        elif self.rustcv_hnsw is not None:
            def knn(query: np.ndarray, params: SearchParams) -> Tuple[np.ndarray, float, Optional[int]]:
                start = time.time()
                res = self.rustcv_hnsw.knn(query, params.k, params.ef)
                search_time = time.time() - start
                return (res.indices, search_time, res.num_distance_calculations)
            return knn
        elif self.jpboth_hnsw is not None:
            def knn(query: np.ndarray, params: SearchParams) -> Tuple[np.ndarray, float, Optional[int]]:
                start = time.time()
                res = self.jpboth_hnsw.knn(query, params.k, params.ef)
                search_time = time.time() - start
                return (res.indices, search_time, res.num_distance_calculations)
            return knn
        elif self.faiss_hnsw is not None:
            def knn(query: np.ndarray, params: SearchParams) -> Tuple[np.ndarray, float, Optional[int]]:
                # reshaped_query = np.reshape(query, (1, query.shape[0]))
                start = time.time()
                _, indices = self.faiss_hnsw.search(query, k=params.k)
                search_time = time.time() - start
                return (indices.astype("uint64")[0,:], search_time, None)
            return knn
        elif self.hnswlib_hnsw is not None:
            def knn(query: np.ndarray, params: SearchParams) -> Tuple[np.ndarray, float, Optional[int]]:
                start = time.time()
                indices, _ = self.hnswlib_hnsw.knn_query(query, k=params.k)
                search_time = time.time() - start
                return (indices[0,:], search_time, None)
            return knn
        else: 
            raise Exception(f"No model contained in this instance of the Model class: {model_params_to_str(self.params)}")

    def knn(self, queries: np.ndarray, params: SearchParams, true_knns: np.ndarray) -> SearchMetrics:
        """
        query: 2-d-ndarray of float32
        true_knns: 2-d-ndarray of uint64 where each row has 1000 elements! 
        """
        n_queries = queries.shape[0]

        if self.hnswlib_hnsw is not None:
            self.hnswlib_hnsw.set_ef(params.ef)
        elif self.faiss_hnsw is not None:
            self.faiss_hnsw.hnsw.efSearch = params.ef

        fn = self.single_row_knn_fn()
        search_secs = 0
        recall = 0
        num_distance_calculations = 0
        for i in range(n_queries):
            if self.faiss_hnsw is not None:
                (indices, duration, ndc) = fn(queries[i:i+1,:], params)
            else:
                (indices, duration, ndc) = fn(queries[i,:], params)
            search_secs += duration
            recall += vecnn.knn_recall(true_knns[i,:params.k], indices) # select onlu the k first (nearest) elements from the true knn row with 1000 items.
            if ndc is not None:
                num_distance_calculations += ndc
        search_secs /= n_queries
        recall /= n_queries
        num_distance_calculations /= n_queries

        if num_distance_calculations == 0:
            num_distance_calculations = None
        return SearchMetrics(search_ms=search_secs*1000.0, recall=recall, num_distance_calculations=num_distance_calculations)

def benchmark_models(params_list: list[ModelParams], data: np.ndarray, queries: np.ndarray, true_knns: np.ndarray, search_params: list[SearchParams], distance: Distance = "dot", seed: int = 42) -> Table:
    # simple sanity checks if all params are valid, before we sink time into building the models and maybe crash
    for params in params_list:
        check_params(params)
        
    table = Table()
    n, dim = data.shape  
    models: list[Model] = []
    i = 0
    print(f"Build {len(params_list)} models")
    for params in params_list:
        print(f"    Start building model {i+1}/{len(params_list)}")
        model = Model(data, params, distance, seed)
        models.append(model)
        print(f"    Built model {i+1}/{len(params_list)} in {int(model.build_metrics.build_secs)}s:  ({asdict(model.params)})")
        i+=1

    for s in search_params:
        search_params_dict = asdict(s)
        print(f"Benchmark {len(models)} models for search params: {search_params_dict}")
        i = -1
        for model in models:
            i+=1
            print(f"    Model {i+1}/{len(models)}")
            metrics = model.knn(queries, s, true_knns)
            
            table.add(
                n = n, 
                dims = dim,
                **search_params_dict,
                params = model_params_to_str(model.params),
                build_secs = model.build_metrics.build_secs,
                build_ndc = model.build_metrics.num_distance_calculations,
                search_ms = metrics.search_ms,
                search_ndc = metrics.num_distance_calculations,
                search_recall = metrics.recall
            )
    return table

# NOTE: we have data from sisap website for n=300k and n=10M with the corresponding gold standard true knns.
# If you want to e.g. run for n=1M and n=50k, you need to generate the subsampled data first 
# and also calculate the true knns and put them into a file.
# to do that go into ../vecnn and run `cargo run --bin compare --release -- gen 1000000 50000`.
# this will place files `laion_subsampled_(50000, 768).bin` and `computed_true_knns_n=50000_(10000,1000).bin`
# into the data folder. You can then load them with the functions from laion_util.py 
IS_ON_SERVER = False
DATA_PATH =  "/data/hepperle" if IS_ON_SERVER else "../data"
QUERIES_FILE_NAME = "laion_queries_(10000, 768).bin"
DATA_FILE_NAMES = {
    "100k": ("laion_subsampled_(100000, 768).bin", "computed_true_knns_n=100000_(10000, 1000).bin"), # cargo run --bin compare --release -- gen 1000000
    "300k": ("laion_300k.h5", "laion_gold_300k_(10000, 1000).bin"),
    "1m": ("laion_subsampled_(1000000, 768).bin", "computed_true_knns_n=1000000_(10000, 1000).bin"), # cargo run --bin compare --release -- gen 1000000
    "10m": ("laion_10m.h5", "laion_gold_10m_(10000, 1000).bin"),
}

N : Literal["100k", "300k", "1m", "10m"]= "100k" # choose one of the keys from DATA_FILE_NAMES

# n_queries is always 10k, the entire public queries from SISAP 2024
load_start = time.time()
queries = laion_util.load_data_or_queries(f"{DATA_PATH}/{QUERIES_FILE_NAME}")
data = laion_util.load_data_or_queries(f"{DATA_PATH}/{DATA_FILE_NAMES[N][0]}")
true_knns = laion_util.load_true_knns(f"{DATA_PATH}/{DATA_FILE_NAMES[N][1]}")
print(f"Loading data took {time.time()-load_start} seconds")
model_params: list[ModelParams] = [
    # HnswParams("jpboth", threaded=False, level_norm=0.3, ef_construction=20, m_max=20, m_max_0=40),
    # HnswParams("vecnn", threaded=False, level_norm=0.3, ef_construction=20, m_max=20, m_max_0=40),
    # HnswParams("jpboth", threaded=True, level_norm=0.3, ef_construction=20, m_max=20, m_max_0=40),
    # HnswParams("hnswlib", threaded=True, level_norm=0.3, ef_construction=20, m_max=20, m_max_0=40),
    # HnswParams("faiss", threaded=True, level_norm=0.3, ef_construction=20, m_max=20, m_max_0=40),
    # VpTreeParams(n_candidates  = 0, threaded=False),
    # RNNGraphParams(outer_loops=3, inner_loops=3, m_initial=40, m_pruned=40),
    # RNNGraphParams(outer_loops=4, inner_loops=5, m_initial=40, m_pruned=40),
    # HnswParams("hnswlib", threaded=True, level_norm=0.3, ef_construction=10, m_max=8, m_max_0=16),

    # RNNGraphParams(outer_loops=6, inner_loops=3, m_initial=40, m_pruned=40),
    #  StitchingParams("method2", n_candidates=0, max_chunk_size=256, same_chunk_m_max=40, m_max=40, fraction=0.6, x_or_ef=3, threaded=False),
    # EnsembleParams(threaded=False, level_norm=0.3, n_vp_trees=6, n_candidates=0, max_chunk_size=256, same_chunk_m_max=10, m_max=20, m_max_0=40, level_norm = 0.0),
    # HnswParams("vecnn", threaded=True, level_norm=0.3, ef_construction=60, m_max=20, m_max_0=40),
    # HnswParams("vecnn", threaded=False, level_norm=0.3, ef_construction=60, m_max=20, m_max_0=40),
    
    EnsembleParams(threaded=True, n_vp_trees=6, max_chunk_size=256, same_chunk_m_max=10, m_max=40, m_max_0=40, level_norm = 0.0),
    # EnsembleParams(threaded=True, n_vp_trees=7, max_chunk_size=256, same_chunk_m_max=10, m_max=40, m_max_0=40, level_norm = 0.0),
    # EnsembleParams(threaded=True, n_vp_trees=8, max_chunk_size=256, same_chunk_m_max=10, m_max=40, m_max_0=40, level_norm = 0.0),
    # EnsembleParams(threaded=True, n_vp_trees=9, max_chunk_size=256, same_chunk_m_max=10, m_max=40, m_max_0=40, level_norm = 0.0),
    # EnsembleParams(threaded=True, n_vp_trees=10, max_chunk_size=256, same_chunk_m_max=10, m_max=40, m_max_0=40, level_norm = 0.0),
    # EnsembleParams(threaded=True, n_vp_trees=7, max_chunk_size=256, same_chunk_m_max=10, m_max=40, m_max_0=40, level_norm = 0.0),
    # EnsembleParams(threaded=True, n_vp_trees=12, max_chunk_size=256, same_chunk_m_max=10, m_max=40, m_max_0=40, level_norm = 0.0),
    # EnsembleParams(threaded=True, n_vp_trees=12,max_chunk_size=256, same_chunk_m_max=10, m_max=40, m_max_0=40, level_norm = 0.0),
    # EnsembleParams(threaded=True, n_vp_trees=6, max_chunk_size=2048, same_chunk_m_max=6, m_max=40, m_max_0=40, level_norm = 0.0, rnn_inner_loops = 3, rnn_outer_loops = 3),
    # EnsembleParams(threaded=False, n_vp_trees=6, max_chunk_size=256, same_chunk_m_max=10, m_max=60, m_max_0=60, level_norm = 0.0),
    
    # HnswParams("rustcv", threaded=False, level_norm=0.3, ef_construction=20, m_max=20, m_max_0=40),
    # EnsembleParams(threaded=True, n_vp_trees=10, max_chunk_size=1024, same_chunk_m_max=8, m_max=40, m_max_0=40, level_norm = 0.0, rnn_inner_loops = 3, rnn_outer_loops = 2),
    
    # HnswParams("jpboth", threaded=False, level_norm=0.3, ef_construction=20, m_max=20, m_max_0=40),
    # HnswParams("vecnn", threaded=False, level_norm=0.3, ef_construction=20, m_max=20, m_max_0=40),
    # HnswParams("hnswlib", threaded=False, level_norm=0.3, ef_construction=20, m_max=20, m_max_0=40),
    
    # HnswParams("jpboth", threaded=False, level_norm=0.3, ef_construction=60, m_max=20, m_max_0=40),
    # HnswParams("vecnn", threaded=False, level_norm=0.3, ef_construction=60, m_max=20, m_max_0=40),
    # HnswParams("hnswlib", threaded=False, level_norm=0.3, ef_construction=60, m_max=20, m_max_0=40),
    # HnswParams("faiss", threaded=False, level_norm=0.3, ef_construction=60, m_max=20, m_max_0=40),
    # HnswParams("jpboth", threaded=True, level_norm=0.3, ef_construction=80, m_max=20, m_max_0=40),
    # HnswParams("vecnn", threaded=True, level_norm=0.3, ef_construction=80, m_max=20, m_max_0=40),
    # HnswParams("hnswlib", threaded=True, level_norm=0.3, ef_construction=80, m_max=20, m_max_0=40),
    # HnswParams("faiss", threaded=True, level_norm=0.3, ef_construction=80, m_max=20, m_max_0=40),
 ]
# EnsembleParams { n_vp_trees: 6, max_chunk_size: 256, same_chunk_m_max: 16, m_max: 20, m_max_0: 40, level_norm: 0.3, distance: Dot, strategy: BruteForceKNN }  
# SliceS2HnswParams { level_norm_param: 0.3, ef_construction: 20, m_max: 20, m_max_0: 40, distance: Dot }                                                       
search_params: list[SearchParams] = [
    # SearchParams(k=30, ef = 30, start_candidates = 1), # make sure ef >= k
    # SearchParams(k=30, ef = 40, start_candidates = 1), # make sure ef >= k
    SearchParams(k=30, ef = 50, start_candidates = 1), # make sure ef >= k
    # SearchParams(k=30, ef = 60, start_candidates = 1), # make sure ef >= k
    # SearchParams(k=30, ef = 70, start_candidates = 1), # make sure ef >= k
    # SearchParams(k=30, ef = 80, start_candidates = 1), # make sure ef >= k
    # SearchParams(k=30, ef = 90, start_candidates = 1), # make sure ef >= k
    # SearchParams(k=30, ef = 100, start_candidates = 1), # make sure ef >= k
    # SearchParams(k=30, ef = 110, start_candidates = 1), # make sure ef >= k
    # SearchParams(k=30, ef = 120, start_candidates = 1), # make sure ef >= k
    # SearchParams(k=30, ef = 130, start_candidates = 1), # make sure ef >= k
    # SearchParams(k=30, ef = 140, start_candidates = 1), # make sure ef >= k
    # SearchParams(k=30, ef = 150, start_candidates = 1) # make sure ef >= k
]

start = time.time()
table = benchmark_models(model_params, data, queries, true_knns, search_params, seed = 123)
total = time.time() -start
print(table.df().to_string())
filename = table.save(f"experiments/experiment{datetime.datetime.now()} Total time: {int(total)}s.csv")

HNSW_LIB_BEST_CONSTRUCTION_TIME = [
    HnswParams(implementation='rustcv', ef_construction=200),
    HnswParams(implementation='vecnn', level_norm=0.3, ef_construction=200, m_max=30, m_max_0=30, threaded=False),
    HnswParams(implementation='hnswlib', ef_construction=200, m_max=30, threaded=False),
]