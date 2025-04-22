import numpy as np

from abc import ABC, abstractmethod
from typing import Iterator, Literal

from torch import Tensor
from sklearn.cluster import KMeans as SKL_KMeans, HDBSCAN as SKL_HDBSCAN

# Supported clustering types
class ClusteringAlgoritm(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @abstractmethod
    def _cluster(self, vectors: np.ndarray) -> list[list[int]]:
        ...

    def cluster(self, vectors: np.ndarray | Tensor | list[list[float]]) -> list[list[int]]:
        """
        Clusters the given 2D-array-like object. Suppose that `vectors` is an
        N x M array. Then, it codifies N vectors, each of M dimensions.

        Returns a list of lists with the indices of the vectors corresponding
        to each cluster. Indices must go from 0 to N-1, without repetitions and
        all of the values must be present. In other words, the result must be a
        partition of the interval [0, ..., N-1].
        """

        if isinstance(vectors, list):
            vectors = np.array(vectors)
        elif isinstance(vectors, Tensor):
            vectors = vectors.numpy()

        assert len(vectors.shape) == 2

        return self._cluster(vectors)

class KMeans(ClusteringAlgoritm):
    def __init__(self, nclusters: int):
        self.__instance = SKL_KMeans(nclusters)

    def _cluster(self, vectors) -> list[list[int]]:
        return self.__instance.fit_predict(vectors)

class HDBSCAN(ClusteringAlgoritm):
    def __init__(self, min_cluster_size = 3):
        self.__instance = SKL_HDBSCAN(min_cluster_size)

    def _cluster(self, vectors) -> list[list[int]]:
        return self.__instance.fit_predict(vectors)

