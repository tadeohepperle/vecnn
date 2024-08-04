from typing import Literal, Optional, Callable, Tuple
import numpy as np

type Distance = Literal['l1', 'l2', 'cos', 'dot']

def sum_as_string(a: int, b: int) -> str:
    "adds two numbers and returns their sum as a string"


def build_hnsw_by_transition(data: Dataset, max_chunk_size: int, same_chunk_m_max: int, m_max: int, neg_fraction: float,  keep_fraction: float, distance: Distance, seed: int) -> Hnsw:
    pass



def build_hnsw_by_vp_tree_ensemble(data: Dataset, max_chunk_size: int, same_chunk_m_max: int, m_max: int,  m_max_0: int, n_vp_trees: int, level_norm: float, distance: Distance, seed: int) -> Hnsw:
    pass


class Dataset(): 
    '''A dataset, wrapping a 2d-numpy array'''
    def __init__(self, data: np.ndarray) -> Dataset:
        """Create a Rust-Owned dataset.
    
        data: 2-dimensional numpy array.
        """
        pass


    @staticmethod
    def new_random(len: int, dims: int) -> Dataset:
        """Creates a dataset that lives in Rust memory 
        (creates a FlatDataSet, this is just a Vec<f32> that can be indexed returning slices of length `dims`)"""
        pass
    # def to_numpy(self) -> np.ndarray:
    #     pass

    def len(self) -> int:
        pass

    def dims(self) -> int:
        pass

    
class VpTree(): 
    '''A vp-tree, built on a DataSet'''
    num_distance_calculations_in_build: int
    
    def __init__(self, data: Dataset, seed: int):
        """constructs a new vp-tree on the dataset passed in."""
        pass

    def knn(self, query: np.ndarray, k: int) -> KnnResult:
        """performs knn search on the vp-tree

        query: 1-dimensional numpy array of type float32.       
        k: number of nearest neighbors to find.
        """
        pass

class KnnResult():
    indices: np.ndarray
    """1-dimensional numpy array of type uint64. Dataset indices of the k nearest neighbors to the query."""
    # distances: np.ndarray
    # """ 1-dimensional numpy array of type float32. Distances of the k nearest neighbors to the query. Belonging to the respective elements of indices array, same size as indices"""
    num_distance_calculations: int
    """how many distance calculations were performed during build."""


def linear_knn(data: Dataset, query: np.ndarray, k: int, distance: Distance) -> KnnResult:
    """performs a linear knn search thorugh the entire data.

    query: 1-dimensional numpy array of type float32."""
    pass



def knn_recall(truth: np.ndarray, reported: np.ndarray) -> float:
    """returns fraction of truth, that was found in reported. From 0.0 to 1.0.
    truth and reported can have different sizes

    truth: 1-dimensional numpy array of type uint64 containing the true indices of the k neighrest neighbors
    reported: 1-dimensional numpy array of type uint64 containing the indices found in knn"""


class Hnsw(): 
    '''An HNSW, built on a DataSet'''
    num_distance_calculations_in_build: int

    
    def __init__(self, data: Dataset, level_norm_param: float, ef_construction: int, m_max: int, m_max_0: int, distance: Distance, use_const_impl: bool, seed: int):
        """constructs a new hnsw on the dataset."""
        pass

    def knn(self, query: np.ndarray, k: int, ef: int) -> KnnResult:
        """performs knn search on the hnsw

        query: 1-dimensional numpy array of type float32.
        k: number of nearest neighbors to find.
        ef: size of priority queue holding found elements, top k of these are returned.
        """
        pass


class RustCvHnsw():
    '''An HNSW, from the rust-cv/hnsw crate.'''

    
    def __init__(self, data: Dataset, ef_construction: int):
        """constructs a new hnsw on the dataset.

        ef_construction: size of priority queue when searching for insertion point in each layer.
        """
        pass

    def knn(self, query: np.ndarray, k: int, ef: int) -> KnnResult:
        """performs knn search on the hnsw

        query: 1-dimensional numpy array of type float32.
        k: number of nearest neighbors to find.
        ef: number of candidates in candidate set.
        """
        pass

class JpBothHnsw():
    '''An HNSW, from https://github.com/jean-pierreBoth/hnswlib-rs'''

    def __init__(self, data: Dataset, ef_construction: int, m_max: int):
        """constructs a new hnsw on the dataset.
        
        ef_construction: size of priority queue when searching for insertion point in each layer.
        m_max: max number of neighbors in each layer
        """
        pass

    def knn(self, query: np.ndarray, k: int, ef: int) -> KnnResult:
        """performs knn search on the hnsw

        query: 1-dimensional numpy array of type float32.
        k: number of nearest neighbors to find.
        ef: number of candidates in candidate set.
        """
        pass

class RNNGraph:
    num_distance_calculations_in_build: int

    def __init__(self, data: Dataset, outer_loops: int, inner_loops: int, max_neighbors_after_reverse_pruning: int, initial_neighbors: int, distance: Distance, seed: int):
        pass

    def knn(self, query: np.ndarray, k: int, start_candidates: int) -> KnnResult:
        pass

