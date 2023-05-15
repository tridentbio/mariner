"""
This package holds the underlying pydantic classes used by
:py:class:`fleet.torch_.schemas.TorchModelSpec and the code to generate part
of those classes :py:mod:`fleet.model_builder.generate`.
"""
from . import featurizers, layers
