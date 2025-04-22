from itertools import chain
from typing import Literal
import numpy as np

from abc import ABC, abstractmethod
from collections import Counter

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

ScoringMethod = Literal[
                "min-min", "min-avg", "min-max",
                "avg-min", "avg-avg", "avg-max",
                "max-min", "max-avg", "max-max"
]

class Retriever:
    def __init__(
            self,
            bacteria_functions: dict[str, set[str]],
            functions_bacteria: dict[str, set[str]],
            function_clustering: list[list[int]],
            scoring_method: ScoringMethod
        ):
        self.__function_clusters = function_clustering
        self.__scoring_method = scoring_method

        self.__bacterium_to_index = {bacterium:index for index, bacterium in enumerate(bacteria_functions.keys())}
        self.__function_to_index = {function:index for index, function in enumerate(functions_bacteria.keys())}

        self.__bacterium_index_to_function_indices = {self.__bacterium_to_index[bacterium]: set(map(lambda f: self.__function_to_index[f], functions)) for bacterium, functions in bacteria_functions.items()}
        self.__inverted_index = {self.__function_to_index[function]: set(map(lambda b: self.__bacterium_to_index[b], bacteria)) for function, bacteria in functions_bacteria.items()}

        self.__tfidf = np.matmul(self.__get_bacteria_frequencies().reshape((-1, 1)), self.__get_inverse_document_frequencies().reshape((1,-1)))

    def __get_bacteria_frequencies(self) -> np.ndarray:
        number_of_bacteria = len(self.__bacterium_to_index)
        number_of_functions = len(self.__function_to_index)

        bacteria_counts = np.zeros(number_of_functions)

        for function_idx, affected_bacteria in self.__bacterium_index_to_function_indices.items():
            bacteria_counts[function_idx] = len(affected_bacteria)

        return bacteria_counts / number_of_bacteria

    def __get_inverse_document_frequencies(self) -> np.ndarray:
        number_of_clusters = len(self.__function_clusters)
        number_of_bacteria = len(self.__bacterium_to_index)

        bacteria_in_clusters = map(
            lambda cluster: set(chain.from_iterable(map(
                lambda function_index: self.__inverted_index[function_index],
                cluster
            ))),
            self.__function_clusters
        )

        cluster_counts = np.zeros(number_of_clusters)

        for i, bacteria_in_cluster in enumerate(bacteria_in_clusters):
            cluster_counts[i] = len(bacteria_in_cluster)

        return np.log(number_of_clusters + 1) - np.log(cluster_counts + 1)

    def __assign_score_to_cluster(self, cluster: list[int]) -> float:
        tfidf_slice = self.__tfidf[cluster, :]

        second, first = self.__scoring_method.split("-")

        match first:
            case "max":
                intermediate = tfidf_slice.max(axis=1)
            case "avg":
                intermediate = tfidf_slice.mean(axis=1)
            case _:
                intermediate = tfidf_slice.min(axis=1)

        match second:
            case "max":
                return float(intermediate.max())
            case "avg":
                return float(intermediate.mean())
            case "min":
                return float(intermediate.min())

    def sort_clusters(self):
        return sorted(self.__function_clusters, key=self.__assign_score_to_cluster, reverse=True)
