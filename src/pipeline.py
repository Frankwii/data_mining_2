import os
from typing import Optional

from torch import Tensor

from file_handling import IOHandler
from pathlib import Path

from language_model import LLMHandler


class Pipeline:
    def __init__(self, model_name: str):
        self.__IO_handler = IOHandler(self.__get_repo_path())
        self.__bacteria_functions = self.__IO_handler.get_bacteria_functions()
        self.__functions_bacteria = self.__IO_handler.get_bacteria_inverted_document()

        self.__model_name = model_name
        # Don't load the model itself until needed; this is a bit time-consuming
        # and, for the most part, only required once.
        self.__model_handler : Optional[LLMHandler] = None

        self.__function_embeddings = self.__get_function_embeddings()

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
            self.__functions_bacteria.keys()
        )

        d = dict(pairs)

        self.__IO_handler.jsonify(d, Path("function_embeddings.json"))

        return d

    def get_function_embeddings(self) -> dict[str, Tensor]:
        return self.__function_embeddings
