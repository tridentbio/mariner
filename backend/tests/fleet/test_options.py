import re
from typing import List, Literal

from pydantic import BaseModel

import fleet.options


def test_options_manager():
    manager = fleet.options.ComponentOptionsManager()

    @manager.config_transformer()
    class Foo(BaseModel):
        type: Literal[
            "sklearn.preprocessing.StandardScaler"
        ] = "sklearn.preprocessing.StandardScaler"

    @manager.configoption(
        fleet.options.ComponentType.TRANSFORMER, "class_path"
    )
    class Baz(BaseModel):
        class_path: Literal[
            "sklearn.preprocessing.LabelEncoder"
        ] = "sklearn.preprocessing.LabelEncoder"

    assert len(manager.options) == 2
    assert (
        manager.options[0].class_path == "sklearn.preprocessing.StandardScaler"
    )


def test_options_manager_import_libs():
    manager = fleet.options.options_manager
    manager.import_libs()
    assert len(manager.options) > 0


def test_options_knn_args_options():
    manager = fleet.options.options_manager
    manager.import_libs()

    knn_options: List[fleet.options.ComponentOption] = []
    class_with_options_regex = [
        re.compile(regex)
        for regex in [
            r".*KNeighbors.*",
            r"sklearn.ensemble.*",
        ]
    ]

    assert len(manager.options) > 0

    for option in manager.options:
        if any(
            [
                regex.search(option.class_path)
                for regex in class_with_options_regex
            ]
        ):
            knn_options.append(option)
    assert len(knn_options) == 6, f"Should be 6, got {len(knn_options)}"
    for option in knn_options:
        assert (
            option.args_options
        ), f"Should be a nom-empty dict or list, got {option.args_options}"
