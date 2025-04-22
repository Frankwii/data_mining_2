import numpy as np

from abc import ABC, abstractmethod
from itertools import chain

from torch import Tensor
from sklearn.cluster import KMeans as SKL_KMeans, HDBSCAN as SKL_HDBSCAN

Clusterable = np.ndarray | Tensor | list[list]
# Supported clustering types
class ClusteringAlgoritm(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @abstractmethod
    def _assign_labels(self, vectors: np.ndarray) -> list[int]:
        ...

    def _to_ndarray(self, vectors: Clusterable) -> np.ndarray:
        if isinstance(vectors, list):
            v = np.array(vectors)
        elif isinstance(vectors, Tensor):
            v = vectors.numpy()
        elif isinstance(vectors, np.ndarray):
            v = vectors
        else:
            raise TypeError("Unexpected vectors array")

        assert len(v.shape) == 2

        return v

    def assign_labels(self, vectors: Clusterable) -> list[int]:
        """
        Clusters the given 2D-array-like object. Suppose that `vectors` is an
        N x M array. Then, it codifies N vectors, each of M dimensions.

        Returns a list of length N with the labels of the clusters to whom
        vectors belong.
        """

        # Cast to regular Python int to avoid np.int(n) representation.
        return [int(i) for i in self._assign_labels(self._to_ndarray(vectors))]
    
    def _process_singleton_labels(self, labels: list[int]) -> list[int]:
        """
        In some algorithms, a label of -1 signifies that the element forms a
        singleton cluster. For our purposes, we wish to assign a unique label
        to that element.
        """
        singleton_cluster_indices = list(
            map(
                lambda t: t[0],
                filter(lambda t: t[1]==-1, enumerate(labels))
            )
        )

        unique_labels = set(labels)

        maximum_label = max(unique_labels)

        new_singleton_labels = dict(map(
            lambda t: (t[1], t[0] + maximum_label),
            enumerate(singleton_cluster_indices)
        ))

        return [new_singleton_labels[i] if labels[i]==-1 else labels[i] for i in range(len(labels))]

    def cluster(self, vectors: Clusterable) -> list[list[int]]:
        """
        Clusters the given 2D-array-like object. Suppose that `vectors` is an
        N x M array. Then, it codifies N vectors, each of M dimensions.

        Returns a list of lists with the indices of the vectors corresponding
        to each cluster. Indices must go from 0 to N-1, without repetitions and
        all of the values must be present. In other words, the result must be a
        partition of the interval [0, ..., N-1].
        """
        labels = self._process_singleton_labels(self.assign_labels(vectors))

        unique_labels = set(labels)

        return list(map(
            lambda unique_label: list(map(
                lambda t: t[0],
                filter(
                    lambda t: t[1] == unique_label,
                    enumerate(labels)
                )
            )),
            unique_labels
        ))

    def format_clustering(self, functions: list[str], clustering: list[list[int]]) -> list[list[str]]:
        return list(map(
            lambda cluster: list(map(
                lambda index: functions[index],
                cluster
            )),
            clustering
        ))

    def cluster_functions_formatted(self, function_embeddings: dict[str, Clusterable]):
        functions = list(function_embeddings.keys())
        embeddings = np.array(list(function_embeddings.values()))

        return self.format_clustering(functions, self.cluster(embeddings))


class KMeans(ClusteringAlgoritm):
    def __init__(self, nclusters: int):
        self.__kmeans = SKL_KMeans(nclusters)

    def _assign_labels(self, vectors) -> list[int]:
        return self.__kmeans.fit_predict(vectors)

class HDBSCAN(ClusteringAlgoritm):
    def __init__(self, min_cluster_size = 3):
        self.__hdbscan = SKL_HDBSCAN(min_cluster_size)

    def _assign_labels(self, vectors) -> list[int]:
        return self.__hdbscan.fit_predict(vectors)

