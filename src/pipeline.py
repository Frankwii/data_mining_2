import os
import numpy as np

from typing import Literal, Optional
from torch import Tensor

from clustering import ClusteringAlgoritm, KMeans, Retriever, ScoringMethod
from file_handling import IOHandler
from pathlib import Path

from language_model import LLMHandler

def centered_dot(a1: np.ndarray, a2: np.ndarray):
    return np.dot(a1 - a1.mean(), a2 - a2.mean())


def pearson(a1: np.ndarray, a2: np.ndarray):
    m1 = a1.mean()
    m2 = a2.mean()

    return np.dot(a1-m1, a2-m2)/(a1.std(mean=m1) * a2.std(mean=m2))

def cosine(a1: np.ndarray, a2: np.ndarray):
    return np.dot(a1, a2)/(np.linalg.norm(a1) * np.linalg.norm(a2))

class Pipeline:
    def __init__(self, model_name: str, similarity_function: Literal["pearson", "centered_dot", "cosine", "dot"] = "cosine"):
        self.__IO_handler = IOHandler(self.__get_repo_path())
        self.__bacteria_functions = self.__IO_handler.get_bacteria_functions()
        self.__bacteria = list(self.__bacteria_functions.keys())
        self.__functions_bacteria = self.__IO_handler.get_bacteria_inverted_document()
        self.__functions = list(self.__functions_bacteria.keys())

        self.__model_name = model_name
        # Don't load the model itself until needed; this is a bit time-consuming
        # and, for the most part, only required once.
        self.__model_handler : Optional[LLMHandler] = None

        self.__function_embeddings = self.__get_function_embeddings()

        match similarity_function:
            case "dot":
                self.__similarity_function = np.dot
            case "centered_dot":
                self.__similarity_function = centered_dot
            case "pearson":
                self.__similarity_function = pearson
            case "cosine":
                self.__similarity_function = cosine

    def __load_model_handler(self):
        if self.__model_handler is None:
            self.__model_handler = LLMHandler(self.__model_name)

    def __get_repo_path(self) -> Path:
        current_file_path = Path(os.path.abspath(__file__))
    
        return current_file_path.parent.parent

    def __get_function_embeddings(self) -> dict[str, Tensor]:
        try:
            return self.__IO_handler.get_function_embeddings()
        except FileNotFoundError:
            return self.__compute_function_embeddings_and_save()

    def __compute_function_embeddings_and_save(self) -> dict[str, Tensor]:
        self.__load_model_handler()

        pairs = map(
            lambda function: (function, self.__model_handler.get_classification_embedding(function).flatten()),
            self.__functions
        )

        d = dict(pairs)

        self.__IO_handler.jsonify(d, Path("function_embeddings.json"))

        return d

    def get_function_embeddings(self) -> dict[str, Tensor]:
        return self.__function_embeddings

    def get_functions(self) -> list[str]:
        return self.__functions

    def get_bacteria(self) -> list[str]:
        return self.__bacteria

    def get_affected_bacteria(self, bacterium: str) -> set[str]:
        return self.__functions_bacteria[bacterium]

    def get_all_affected_bacteria(self, bacteria: list[str]) -> set[str]:
        return set().union(*(self.get_affected_bacteria(bacterium) for bacterium in bacteria))

    def instantiate_retriever(self, clustering_algorithm: ClusteringAlgoritm, scoring_method: ScoringMethod) -> Retriever:
        embeddings = np.array(list(self.__function_embeddings.values()))
        clusters = clustering_algorithm.cluster(embeddings)
        return Retriever(
            self.__bacteria_functions,
            self.__functions_bacteria,
            clusters,
            scoring_method = scoring_method
        )


    def function_semantic_search(self, query: str) -> list[str]:
        self.__load_model_handler()
        embedding_query = self.__model_handler.get_classification_embedding(query).flatten()

        return sorted(
            self.__function_embeddings.keys(),
            key=lambda k: self.__similarity_function(np.array(self.__function_embeddings[k]), embedding_query),
            reverse=True
        )

    def bacterium_semantic_search(self, query: str, aggregation: Literal["min", "avg", "max"] = "min") -> list[tuple[str, float]]:
        self.__load_model_handler()
        embedding_query = self.__model_handler.get_classification_embedding(query).flatten().numpy()

        bacterium_scores = {}

        for bacterium in self.__bacteria_functions.keys():
            similarities = np.zeros(len(self.__bacteria_functions[bacterium]))
            for idx, function in enumerate(self.__bacteria_functions[bacterium]):
                similarity = self.__similarity_function(
                    self.__function_embeddings[function].numpy(), embedding_query
                )
                similarities[idx] = similarity

            match aggregation:
                case "min":
                    value = similarities.min()
                case "avg":
                    value = similarities.mean()
                case "max":
                    value = similarities.max()
                case _:
                    raise ValueError(f"Unsupported aggregation method: {aggregation}")

            bacterium_scores[bacterium] = float(value)
        
        sorted_bacteria_dict = dict(sorted(bacterium_scores.items(), key=lambda item: item[1], reverse=True))

        return list(sorted_bacteria_dict.items())
