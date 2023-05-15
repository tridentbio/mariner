"""
Fleet is the Data Science package that powers Mariner.

It has functions to train neural networks with torch, as well as different algorithms
provided by scikit and xgboost libraries.
"""
from . import data_types
from .base_schemas import DatasetConfig, FleetModelSpec
from .model_functions import fit
from .torch_.schemas import TorchTrainingConfig
