"""
This file contains constants used in the model builder.
"""
from enum import Enum


class TrainingStep(Enum):
    """
    Enum defining index used in dataset splitting
    """

    TRAIN = 1
    VAL = 2
    TEST = 3
