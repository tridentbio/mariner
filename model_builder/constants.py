"""
This file contains constants used in the model builder.
"""
from enum import Enum


class TrainingStep(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3
