import json

from pathlib import Path
from typing import Any
from itertools import chain

from torch import Tensor

class _CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, set):
            return list(o)
        elif isinstance(o, Tensor):
            return [float(n) for n in o.view(-1)]
        else:
            return super().default(o)

class IOHandler:
    def __init__(self, repo_path: Path):
        self.__resources_path = repo_path.joinpath("resources")

        self.__input_files_path = self.__resources_path.joinpath("input_raw")

        assert self.__input_files_path.is_dir()

    @staticmethod
    def __get_name(path: Path) -> str:
        return path.name

    @staticmethod
    def __get_contents_by_lines(path: Path) -> set[str]:
        with open(path, 'r') as f:
            contents = f.readlines()

        return set(
            filter(
                lambda line: bool(line),
                map(
                    lambda line: line.strip(),
                    contents
                )
            )
        )

    @staticmethod
    def __parse_bacterium(path: Path) -> tuple[str, set[str]]:
        return IOHandler.__get_name(path), IOHandler.__get_contents_by_lines(path)

    def __parse_bacteria_and_save(self) -> dict[str, set[str]]:

        d = dict(map(self.__parse_bacterium, self.__input_files_path.iterdir()))

        self.jsonify(d, Path("input.json"))

        return d

    def get_bacteria_functions(self) -> dict[str, set[str]]:
        """
        Return a dictionary whose keys are the bacteria names and values 
        are the functions.
        """
        expected_location = self.__resources_path.joinpath("input.json")
        if expected_location.exists():
            return self.__parse_json(expected_location)
        else:
            return self.__parse_bacteria_and_save()
        
    def __compute_bacteria_inverted_document_and_save(self) -> dict[str, set[str]]:
        bacteria_functions = self.get_bacteria_functions()

        functions = set(chain.from_iterable(bacteria_functions.values()))

        pairs = map(
            lambda function: (function, set(
                filter(
                    lambda bacterium: function in bacteria_functions[bacterium],
                    bacteria_functions.keys()
                )
                )),
            functions
        )

        d = dict(pairs)

        self.jsonify(d, Path("inverted_document.json"))

        return d

    def get_bacteria_inverted_document(self) -> dict[str, set[str]]:
        """
        Return a dictionary whose keys are the biological functions values 
        are the bacteria which have those functions.
        """
        expected_location = self.__resources_path.joinpath("inverted_document.json")
        if expected_location.exists():
            return self.__parse_json(expected_location)
        else:
            return self.__compute_bacteria_inverted_document_and_save()

    def get_function_embeddings(self) -> dict[str, Tensor]:
        expected_location = self.__resources_path.joinpath("function_embeddings.json")
        if expected_location.exists():
            d = self.__parse_json(expected_location)

            return {k: Tensor(values).flatten() for k, values in d.items()}
        else:
            raise FileNotFoundError("Unable to locate `function_embeddings.json`! Be sure to compute and save embeddings before calling this method.")

    def jsonify(self, d: dict, relpath: Path) -> None:
        """
        Turns a dictionary into a json and writes it to disk inside the
        `resources` folder, under the specified relative path.
        """
        assert not relpath.is_absolute()

        path = self.__resources_path.joinpath(relpath)

        with open(path, 'w') as f:
            json.dump(d, f, cls=_CustomEncoder)

    def __parse_json(self, relpath: Path) -> dict[str, Any]:
        """
        Parses json under the `resources` folder into a Python dictionary.
        """
        with open(relpath, 'r') as f:
            d = json.load(f)

        return d
