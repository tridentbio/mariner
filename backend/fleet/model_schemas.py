"""
Exports the grouped schemas from many frameworks a annotated union types.

They are used by pydantic models to correctly parse a payload into the spec
of the respective framework.
"""
from typing import Annotated, Union

from pydantic import Field

from fleet.scikit_.schemas import SklearnModelSpec
from fleet.torch_.schemas import TorchModelSpec

FleetModelSpec = Annotated[
    Union[TorchModelSpec, SklearnModelSpec], Field(discriminator="framework")
]
