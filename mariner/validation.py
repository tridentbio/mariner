"""
This module is responsible for data validation. It has functions to validate
single values and dataframes.
"""
from re import search
from typing import List, NewType, Tuple, Union

import pandas as pd
from rdkit import Chem, RDLogger

from mariner.schemas.dataset_schemas import ColumnsMeta

Mol = Chem.rdchem.Mol

RDLogger.DisableLog("rdApp.*")


def make_mol(smiles: str) -> Mol:
    """Transforms smiles to rdkit Mol

    Args:
        smiles: str to check if is valid smiles

    Returns:
        str: the same str

    Raises:
        ValueError: if smiles is not syntactically valid or has invalid chemistry
    """
    if not isinstance(smiles, str):
        raise ValueError("smiles should be str")

    mol: Mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f'SMILES "{smiles}" is not syntacticaly valid.')
    try:
        Chem.SanitizeMol(mol)
    except:  # noqa: E722
        raise ValueError(f'SMILES "{smiles}" does not have valid chemistry.')
    return mol


def is_valid_smiles_value(smiles: str) -> bool:
    """Validates a single smiles string

    Args:
        smiles: str to check if is valid smiles

    Returns:
        bool: True if smiles is a valid smiles string
    """
    try:
        make_mol(smiles)
    except ValueError:
        return False
    return True


def is_valid_smiles_series(smiles_series: pd.Series, weak_check=False) -> bool:
    """Validates a pandas string series checking if all elements are valid smiles

    if weak_check is True, only check first 20 rows

    Args:
        smiles_series: pd.Series of strings
        weak_check: Flag to skip checking the whole data series

    Returns:
        bool: True if series is a of valid smile strings
    """
    for val in smiles_series[: 20 if weak_check else len(smiles_series)]:
        try:
            make_mol(val)
        except ValueError:
            return False
    return True


def _is_instance(type: type, msg: str, nullable=True) -> Tuple[callable, str]:
    """
    Function validator creator
    """

    def func(x):
        return isinstance(x, type) or (nullable and not x)

    return (func, msg)


def find_column_i(df: pd.DataFrame, column_name: str) -> int:
    def compare_insensitive(x, y):
        return x.lower() == y.lower()

    def compare_regex(x, y):
        return search(y, x) is not None

    for i, col in enumerate(df.columns):
        if compare_insensitive(col, column_name) or compare_regex(col, column_name):
            return i

    return -1


SchemaType = NewType(
    "schema_type", dict[str, Union[Tuple[callable, str], List[Tuple[callable, str]]]]
)

VALIDATION_SCHEMA: SchemaType = {
    "categorical": [_is_instance(str, "column $ should be str")],
    "numeric": (
        lambda x: not x or search(r"^\d[\.,\d]*$", x) is not None,
        "column $ should be numeric",
    ),
    "smiles": [
        _is_instance(str, "smile column $ should be str"),
        (is_valid_smiles_value, "column $ should be a valid smiles"),
    ],
    "string": _is_instance(str, "column $ should be str"),
}


def check_is_compatible(df: pd.DataFrame, columns_metadata: List[ColumnsMeta]):
    """
    Checks if the columns metadata is compatible with the dataset

    :param df DataFrame: dataset
    :param columns_metadata List[ColumnsMeta]: columns metadata
    """

    errors = []
    for column_metadata in columns_metadata:
        i = find_column_i(df, column_metadata.pattern)

        if i == -1:
            errors.append(f"Column {column_metadata.pattern} not found in dataset")
        else:
            validations = VALIDATION_SCHEMA.get(
                column_metadata.data_type.domain_kind, []
            )
            if isinstance(validations, tuple):
                validations = [validations]

            for validation in validations:
                func, msg = validation
                is_valid = df.iloc[:, i].apply(func).all()
                if not is_valid:
                    errors.append(msg.replace("$", df.columns[i]))
                    break

    return errors
