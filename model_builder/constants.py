"""
This file contains constants used in the model builder.
"""
from enum import Enum


class TrainingStep(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


class BiologicalValidChars(Enum):
    """Used to store list of valid chars
    for each biological sequences.
    """

    AMBIGUOUS_DNA_RNA = ["R", "Y", "S", "W", "K", "M", "B", "D", "H", "V", "N"]
    UNAMBIGUOUS_DNA_RNA = ["A", "C", "G", "T", "U"]
    ONLY_DNA = "T"
    ONLY_RNA = "U"
    PROTEIN = [
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
        "-",
        "*",
    ]
