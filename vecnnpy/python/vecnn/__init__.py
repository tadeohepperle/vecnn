"""A nice little py module wrapping the vecnn rust library"""

from ._lib import VpTree, Dataset, linear_knn, knn_recall, KnnResult, Hnsw, RustCvHnsw, JpBothHnsw, build_hnsw_by_transition, RNNGraph # export public parts of the binary extension

__all__ = ["linear_knn", "knn_recall", "VpTree", "Dataset", "KnnResult", "Hnsw", "RustCvHnsw", "JpBothHnsw","build_hnsw_by_transition", "RNNGraph"]

def pyfoo(a: int):
    """A fn that prints foo a lot"""

    print(f"foo foo foo {a}")