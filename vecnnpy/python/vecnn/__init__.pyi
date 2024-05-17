from typing import Tuple
import numpy as np

def sum_as_string(a: int, b: int) -> str:
    "adds two numbers and returns their sum as a string"

class Dataset(): 
    '''A dataset, wrapping a 2d-numpy array'''
    def __init__(self, data: np.ndarray) -> Dataset:
        """Create a Rust-Owned dataset.
    
        data: 2-dimensional numpy array.
        """
        pass

    def to_numpy(self) -> np.ndarray:
        pass

    def len(self) -> int:
        pass

    def dims(self) -> int:
        pass

    
class VpTree(): 
    '''A vp-tree, built on a DataSet'''
    num_distance_calculations_in_build: int
    
    def __init__(self, data: Dataset):
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
    distances: np.ndarray
    """ 1-dimensional numpy array of type float32. Distances of the k nearest neighbors to the query. Belonging to the respective elements of indices array, same size as indices"""
    num_distance_calculations: int
    """how many distance calculations were performed during build."""


def linear_knn(data: Dataset, query: np.ndarray, k: int) -> KnnResult:
    """performs a linear knn search thorugh the entire data.

    query: 1-dimensional numpy array of type float32."""
    pass


class HnswParams:
    """Params used for building the Hnsw"""
    level_norm_param: float
    ef_construction: int
    m_max: int
    m_max_0: int

    def __init__(self, level_norm_param: float, ef_construction: int, m_max: int, m_max_0: int):
        pass


class Hnsw(): 
    '''An HNSW, built on a DataSet'''
    num_distance_calculations_in_build: int

    
    def __init__(self, data: Dataset, params: HnswParams):
        """constructs a new hnsw on the dataset."""
        pass

    def knn(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """performs knn search on the hnsw

        query: 1-dimensional numpy array of type float32.
        k: number of nearest neighbors to find.
        """
        pass
