"""A nice little py module wrapping the vecnn rust library"""

from ._lib import sum_as_string, VpTree, Dataset, linear_knn, knn_recall, KnnResult, Hnsw, HnswParams  # export public parts of the binary extension

__all__ = ["sum_as_string", "linear_knn", "knn_recall", "VpTree", "Dataset", "KnnResult", "Hnsw", "HnswParams",]

def pyfoo(a: int):
    """A fn that prints foo a lot"""

    print(f"foo foo foo {a}")