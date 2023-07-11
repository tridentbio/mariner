"""Validation functions for mariner"""
from re import search
from typing import Any, Callable, Dict, Literal, Optional, Set, Tuple, Union

import pandas as pd
from numba import njit
from rdkit import Chem, RDLogger

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
        if not is_valid_smiles_value(val):
            return False
    return True


def validate_column_pattern(column: str, pattern: str) -> bool:
    """Validates if a column name matches a pattern

    Args:
        column (str): column name
        pattern (str): column pattern

    Returns:
        bool: True if column matches the pattern
    """

    def compare_insensitive(x, y):
        return x.lower() == y.lower()

    def compare_regex(x, y):
        return search(y, x) is not None

    return compare_insensitive(column, pattern) or compare_regex(
        column, pattern
    )


def find_column_by_name(df: pd.DataFrame, column_name: str) -> int:
    """Finds the index of a column in a dataframe by its name

    The search is case insensitive and can use regex

    Args:
        df (pd.DataFrame): dataframe to search
        column_name (str): column name to search

    Returns:
        int: index of the column
    """
    for i, col in enumerate(df.columns):
        if validate_column_pattern(col, column_name):
            return i

    return -1


def is_not_float(x: Any) -> bool:
    """Checks if a string is not a float

    Args:
        x (str): string to check

    Returns:
        bool: False if string is a float
    """
    try:
        x_float = float(x)
        return x_float == int(x_float)
    except Exception:
        return True


def _is_instance(
    type: type, msg: Optional[str] = None, nullable=True
) -> Tuple[Callable[..., bool], str]:
    """Function factory to create a function that checks if a value is an instance of a type

    Can be used in validation schema to create a validator based on "isinstance" check

    Args:
        type (type): instance type to check.
        msg (Optional[str], optional):
            informative message about the check.
            Defaults to f"column $ should be {type.__name__}".
        nullable (bool, optional): True if null values is valid. Defaults to True.

    Returns:
        Tuple[callable, str]: validation_schema validator
    """
    msg = f"column $ should be {type.__name__}" if not msg else msg

    def func(x):
        return isinstance(x, type) or (nullable and not x)

    return (func, msg)


@njit
def determine_seq_type(seq: str) -> Tuple[bool, str, bool, int]:
    """
    Determine sequence type.

    Args:
        seq (str): Sequence to identify.

    Returns:
        Tuple of valid (bool), type (str), possible_ambiguous (bool), count_unambiguous (int).
    """
    AMBIGUOUS_DNA_RNA = "RYSWKMBDHVN"
    UNAMBIGUOUS_DNA_RNA = "ACGTU-"
    ONLY_DNA = "T"
    ONLY_RNA = "U"
    PROTEIN = "ACDEFGHIKLMNPQRSTVWY-*"

    seq = seq.upper()
    domain_kind: Literal["dna", "rna", "protein"] = "dna"  # priority
    possible_ambiguous = False
    count_unambiguous = 0

    for i, char in enumerate(seq):
        # check for unambiguous chars
        if char in UNAMBIGUOUS_DNA_RNA and domain_kind in ["dna", "rna"]:
            # need to have chars A, C, G or T >50% of sequence to be DNA or RNA
            count_unambiguous += 1

            # dna is priority so domain_kind can be changed to rna or protein
            if char in ONLY_RNA and domain_kind == "dna":
                # need to check if it has any only_dna value in checked values
                domain_kind = (
                    "rna" if ONLY_DNA not in set(seq[:i]) else "protein"
                )

            # if domain_kind already changed to rna it's not a valid dna
            if char in ONLY_DNA and domain_kind == "rna":
                domain_kind = "protein"

        # check if it's possible to be ambiguous or protein
        else:
            if char in AMBIGUOUS_DNA_RNA and domain_kind in ["dna", "rna"]:
                possible_ambiguous = True

            elif char in PROTEIN:
                domain_kind = "protein"

            else:
                # invalid char for any biological domain kind
                return False, "none", False, 0

    return True, domain_kind, possible_ambiguous, count_unambiguous


def check_biological_sequence(
    seq: Union[str, Any]
) -> Dict[str, Union[str, bool]]:
    """Check if a sequence is valid as DNA, RNA or Protein
    Rules (ordered by priority):
        DNA:
            unambiguous nucleotides: A, C, G, T, -
            ambiguous nucleotides: R, Y, S, W, K, M, B, D, H, V, N
                presence of A, C, G or T needs to be >50% of sequence
        RNA:
            unanbiguous nucleotides: A, C, G, U, -
            ambiguous nucleotides: R, Y, S, W, K, M, B, D, H, V, N
                presence of A, C, G or U needs to be >50% of sequence
        Protein:
            - not a DNA or RNA
            - chars: - A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y, -, *
    Args:
        seq (str): sequence to check
    Returns:
        Dict[str, str]: dictionary with the following keys:
            - valid: True if sequence is a valid biological sequence
            - type?: dna, rna or protein
            - is_ambiguous?: True if sequence contains ambiguous nucleotides
    """

    # at least 5 chars to be a valid sequence
    if not isinstance(seq, str) or len(seq) < 5:
        return {"valid": False}

    (
        valid,
        domain_kind,
        possible_ambiguous,
        count_unambiguous,
    ) = determine_seq_type(seq)

    if not valid:
        return {"valid": False}

    result = dict(valid=True)

    if possible_ambiguous:
        # Need to check possibility to be a protein
        if count_unambiguous / len(seq) >= 0.5:
            result.update({"type": domain_kind, "is_ambiguous": True})
        else:
            result.update(
                {"type": "protein", "kwargs": {"domain_kind": "protein"}}
            )

    else:
        result.update({"type": domain_kind, "is_ambiguous": False})

    return result


def check_biological_sequence_series(
    series: pd.Series,
) -> Dict[str, Union[str, Dict[str, str]]]:
    """Makes a check if a sequence is valid as DNA, RNA
    or Protein for each element in a series
    Returns best result for sequence following the rules

    Rules:
        if at least one seq is invalid, return invalid
        if at least one seq is protein, return protein
        if at least one seq is dna and other is rna, return protein
        if at least one seq is ambiguous, return ambiguous

    Args:
        series: pd.Series of strings

    Returns:
        dict: dictionary with the following keys:
            - valid: True if sequence is a valid biological sequence
            - type?: dna, rna or protein
            - kwargs?:
                dictionary with kwargs to pass to the biotype class constructor
    """
    types_found: Set[Literal["dna", "rna", "protein"]] = set()
    ambiguity: Set[Literal["ambiguous", "unanbiguous"]] = set()

    try:
        for seq in series:
            seq_result = check_biological_sequence(seq)
            if not seq_result["valid"]:
                raise ValueError("Invalid sequence")

            # store all different types found in the series
            types_found.add(seq_result["type"])
            # store all different ambiguity found in the series
            ambiguity.add(
                "ambiguous"
                if seq_result.get("is_ambiguous")
                else "unanbiguous"
            )

        # if more than one type found or at least one protein, return protein
        if len(types_found) > 1 or "protein" in types_found:
            return {
                "valid": True,
                "type": "protein",
                "kwargs": {"domain_kind": "protein"},
            }

        # if only one type found, return it
        domain_kind = types_found.pop()
        # if at least one ambiguous, return ambiguous
        is_ambiguous = "ambiguous" in ambiguity

        return {
            "valid": True,
            "type": domain_kind,
            "kwargs": {
                "domain_kind": domain_kind,
                "is_ambiguous": is_ambiguous,
            },
        }
    except (AttributeError, ValueError):
        return {"valid": False}
