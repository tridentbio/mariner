"""
This module is responsible for data validation. It has functions to validate
single values and dataframes.
"""
from io import BytesIO
from re import search
from typing import Dict, List, Literal, Optional, Set, Tuple, Union

import pandas as pd
from rdkit import Chem, RDLogger

from mariner.core.aws import Bucket, upload_s3_compressed
from mariner.schemas.dataset_schemas import ColumnsMeta, SchemaType

from .rules import BiologicalValidChars

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


def check_biological_sequence(seq: str) -> Dict[str, Union[str, bool]]:
    """Check if a sequence is valid as DNA, RNA or Protein
    Rules (ordered by priority):
        DNA:
            unanbiguous nucleotides: A, C, G, T, -
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
    if len(seq) == 0:
        return {"valid": False}

    seq = seq.upper()
    domain_kind: Literal["dna", "rna", "protein"] = "dna"  # priority
    possible_ambiguous = False
    count_unambiguous = 0

    for i, char in enumerate(seq):
        # check for unambiguous chars
        if char in BiologicalValidChars.UNAMBIGUOUS_DNA_RNA.value and domain_kind in [
            "dna",
            "rna",
        ]:
            # need to have chars A, C, G or T >50% of sequence to be DNA or RNA
            count_unambiguous += 1

            # dna is priority so domain_kind can be changed to rna or protein
            if char in BiologicalValidChars.ONLY_RNA.value and domain_kind == "dna":
                # need to check if it has any only_dna value in checked values
                domain_kind = (
                    "rna"
                    if BiologicalValidChars.ONLY_DNA.value not in set(seq[:i])
                    else "protein"
                )

            # if domain_kind already changed to rna it's not a valid dna
            if char in BiologicalValidChars.ONLY_DNA.value and domain_kind == "rna":
                domain_kind = "protein"

        # check if it's possible to be ambiguous or protein
        else:
            if (
                char in BiologicalValidChars.AMBIGUOUS_DNA_RNA.value
                and domain_kind in ["dna", "rna"]
            ):
                possible_ambiguous = True

            elif char in BiologicalValidChars.PROTEIN.value:
                domain_kind = "protein"

            else:
                # invalid char for any biological domain kind
                return {"valid": False}

    result = dict(valid=True)

    if possible_ambiguous:
        # Need to check possibility to be a protein
        if count_unambiguous / len(seq) >= 0.5:
            result.update({"type": domain_kind, "is_ambiguous": True})
        else:
            result.update({"type": "protein", "kwargs": {"domain_kind": "protein"}})

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
                "ambiguous" if seq_result.get("is_ambiguous") else "unanbiguous"
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
            "kwargs": {"domain_kind": domain_kind, "is_ambiguous": is_ambiguous},
        }
    except (AttributeError, ValueError):
        return {"valid": False}


def _is_instance(
    type: type, msg: Optional[str] = None, nullable=True
) -> Tuple[callable, str]:
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

    return compare_insensitive(column, pattern) or compare_regex(column, pattern)


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


def is_not_float(x: str) -> bool:
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


VALIDATION_SCHEMA: SchemaType = {
    # Schema to validate the dataset
    # Key is the column metadata pattern and the value is a list of validators.
    # Validators are a tuple of a function and a informative message about the check.
    # The function should be applied to a pd.Series and return a boolean.
    #     True if the series is valid
    #     False if the series is invalid
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
        lambda x: check_biological_sequence(x).get("valid"),
        "column $ should be a valid protein sequence",
    ),
}


class CompatibilityChecker:
    """Class to check if the columns metadata is compatible with the dataset provided.

    Validations will be based on the validate_schema generating a dict with the
    errors sample and a csv with the errors details.

    Attributes:
        df (pd.DataFrame): dataset
        columns_metadata (List[ColumnsMeta]): columns metadata
        validate_schema (SchemaType): schema to validate the dataset
        row_error_limit (int): max number of rows to show in the errors sample dict
        has_error (bool): flag to indicate if the dataset has errors
        error_column (pd.Series): column with the errors details
    """

    def __init__(
        self,
        columns_metadata: List[ColumnsMeta],
        df: pd.DataFrame,
        validate_schema: SchemaType = VALIDATION_SCHEMA,
    ):
        self.df = df
        self.columns_metadata = columns_metadata
        self.validate_schema = validate_schema
        self.errors = {"columns": [], "rows": [], "logs": [], "dataset_error_key": None}
        self.row_error_limit = 10
        self.has_error = False
        self.error_column = pd.Series(data=[""] * len(self.df.index), dtype=str)

    def add_error(
        self,
        msg: str,
        type: Optional[Literal["columns", "rows"]] = None,
        new_log: bool = True,
    ):
        """
        Add an new error to the errors dict

        Args:
            msg (str): error message
            type (Optional[Literal["columns", "rows"]], optional):
                type of error.
                Defaults to None.
            new_log (bool, optional):
                flag to indicate if the error should be added to the logs.
                Defaults to True.
        """
        if type:
            self.errors[type].append(msg)
        if new_log:
            self.errors["logs"].append(msg)

    def check_is_compatible(self):
        """
        Checks if the columns metadata is compatible with the dataset

        All check will be based on column_metadata pattern found
        in validate_schema

        The validation functions will be applied to the column and if
        some row is invalid (False) the error will be added to the errors
        dict and has_error flag will be set to True

        If some validation raises an exception, the error will be added
        to the errors dict as a log and has_error flag will be set to True
        """
        for column_metadata in self.columns_metadata:
            # Find the column index by the column_metadata pattern
            i = find_column_by_name(self.df, column_metadata.pattern)

            # If the column is not found, add an error to the errors dict
            # else, validate the column
            if i == -1:
                self.add_error(
                    f"Column {column_metadata.pattern} not found in dataset", "columns"
                )
            else:
                validations = self.validate_schema.get(
                    column_metadata.data_type.domain_kind, []
                )
                if isinstance(validations, tuple):
                    validations = [validations]

                # Some validations can have multiple functions to validate the column
                for validation in validations:
                    # Unpack the validation tuple
                    # func: (...) -> bool, msg: str
                    func, msg = validation
                    is_valid = None

                    try:
                        valid_row_serie = self.df.iloc[:, i].apply(func)
                        is_valid = valid_row_serie.all()

                        # If some row is invalid, handle the error
                        if not is_valid:
                            # List of invalid rows
                            invalid_rows: list = self.df.loc[
                                ~valid_row_serie, self.df.columns[i]
                            ].to_list()

                            # Concatenate the error message to the error_column
                            self.error_column.loc[~valid_row_serie] += (
                                self.column_message(self.df.columns[i], msg) + " | "
                            )

                            # Add the errors to the errors dict
                            for index, val in enumerate(
                                invalid_rows[: self.row_error_limit]
                            ):
                                self.add_error(
                                    self.row_message(index, val, self.df.columns[i]),
                                    "rows",
                                    new_log=False,
                                )

                    except Exception as e:
                        # When an unexpected error occurs, save error as a log
                        self.errors["logs"].append(
                            (f"Error validating column {self.df.columns[i]}: {e}")
                        )

                    if not is_valid:
                        self.errors["columns"].append(
                            self.column_message(self.df.columns[i], msg)
                        )
                        self.has_error = True

    def generate_errors_dataset(self):
        """Generate a new dataset with the error_column

        upload the dataset to s3 and add the key to the errors dict
        """
        df = self.df.copy()
        df["errors"] = self.error_column
        file = BytesIO()
        df.to_csv(file, index=False)
        file.seek(0)
        file_key, _ = upload_s3_compressed(file, Bucket.Datasets)
        self.errors["dataset_error_key"] = file_key

    @staticmethod
    def column_message(column: str, msg: str) -> str:
        """Static method to generate the column message

        This message will be based on the msg provided by validate_schema
        and the column name

        Args:
            column (str): column name
            msg (str): message from validate_schema

        Returns:
            str: column message
        """
        return msg.replace("$", f'"{column}"')

    @staticmethod
    def row_message(index: int, row: str, column: str) -> str:
        """Static method to generate the row message

        This message will be based on the row value and index and the column name

        Args:
            index (int): row index
            row (str): row value
            column (str): column name

        Returns:
            str: row message
        """
        return f"Value {row} in row '{index}' of column" f" '{column}' is invalid"
