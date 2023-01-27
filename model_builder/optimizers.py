"""
Package defines optimizers that can be used for model training
"""
from typing import Any, Literal, Union

from pydantic import BaseModel, Field

from model_builder.utils import CamelCaseModel


class AdamParams(BaseModel):
    """
    Arguments of the Adam torch optimizer
    """

    lr: Union[float, None] = None
    beta1: Union[float, None] = None
    beta2: Union[float, None] = None
    eps: Union[float, None] = None


class InputDescription(BaseModel):
    """
    Describes a parameter of some payload
    """

    param_type: Literal["float", "float?"]
    default: Any = None


class AdamParamsSchema(BaseModel):
    """
    Describes how are the in puts of the Adam optimizer
    so the frontend knows how to capture the input
    """

    class_path: Literal["torch.optim.Adam"] = Field(
        default="torch.optim.Adam", alias="classPath"
    )
    lr = InputDescription(param_type="float?", default=0.001)
    beta1 = InputDescription(param_type="float?", default=0.9)
    beta2 = InputDescription(param_type="float?", default=0.999)
    eps = InputDescription(param_type="float?", default=1e-8)


class AdamOptimizer(CamelCaseModel):
    """
    The Adam optimizer

    https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#adam
    """

    classPath = "torch.optim.Adam"
    params: AdamParams = AdamParams()


class SGDParams(BaseModel):
    """
    Arguments of the SGD torch optimizer
    """

    lr: Union[float, None] = None
    momentum: Union[float, None] = None


class SGDParamsSchema(BaseModel):
    """
    Describe how are the inputs of the SGD optimizer
    so the frontend knows how to capture the input

    Description of the vlaues
        - float: the input is a float

    """

    class_path: Literal["torch.optim.SGD"] = Field(
        default="torch.optim.SGD", alias="classPath"
    )
    lr = InputDescription(param_type="float")
    momentum = InputDescription(param_type="float?", default=0)


class SGDOptimizer(CamelCaseModel):
    """
    The SGD optimizer

    https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#sgd
    """

    class_path = "torch.optim.SGD"
    params: SGDParams


Optimizer = Union[AdamOptimizer, SGDOptimizer]
