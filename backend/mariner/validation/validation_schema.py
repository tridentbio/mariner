"""
Schema to validate the dataset
Validators are a tuple of a function and a informative message about the check.
The function should be applied to a pd.Series and return a boolean.
    True if the series is valid
    False if the series is invalid
"""
from re import search

from mariner.schemas.dataset_schemas import SchemaType

from .functions import (
    _is_instance,
    check_biological_sequence,
    is_not_float,
    is_valid_smiles_value,
)

VALIDATION_SCHEMA: SchemaType = {
    "categorical": (is_not_float, "columns $ is categorical and can not be a float"),
    "numeric": (
        lambda x: not x or search(r"^[-\d\.][\.,\d]*$", str(x)) is not None,
        "column $ should be numeric",
    ),
    "smiles": [
        _is_instance(str, msg="smile column $ should be str"),
        (is_valid_smiles_value, "column $ should be a valid smiles"),
    ],
    "string": _is_instance(str),
    "dna": (
        lambda x: check_biological_sequence(x).get("type") == "dna",
        "column $ should be a valid DNA sequence",
    ),
    "rna": (
        lambda x: check_biological_sequence(x).get("type") == "rna",
        "column $ should be a valid RNA sequence",
    ),
    "protein": (
        lambda x: bool(check_biological_sequence(x).get("valid")),
        "column $ should be a valid protein sequence",
    ),
}
