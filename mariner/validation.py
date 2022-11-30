"""
This module is responsible to defined validatores for datasets and dataframes
"""


import pandas as pd
from rdkit import Chem

Mol = Chem.rdchem.Mol


def validate_smiles(smiles: str) -> Mol:
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


def validate_smiles_series(smiles_series: pd.Series) -> bool:
    """Validates a pandas string series checking if all elements are valid smiles

    [TODO:description]

    Args:
        smiles_series: pd.Series of strings

    Returns:
        bool: True if series is a of valid smile strings
    """
    for val in smiles_series:
        try:
            validate_smiles(val)
        except ValueError:
            return False
    return True
