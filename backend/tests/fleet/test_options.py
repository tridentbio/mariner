from typing import Literal

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
