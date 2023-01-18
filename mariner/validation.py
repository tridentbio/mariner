"""
This module is responsible for data validation. It has functions to validate
single values and dataframes.
"""
from io import BytesIO
from re import search
from typing import Callable, List, Literal, Optional, Tuple

import pandas as pd
from rdkit import Chem, RDLogger

from mariner.core.aws import Bucket, upload_s3_compressed
from mariner.schemas.dataset_schemas import ColumnsDescription, SchemaType

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


def _is_instance(
    type: type, msg: Optional[str] = None, nullable=True
) -> Tuple[Callable, str]:
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


def find_column_by_name(df: pd.DataFrame, column_name: str) -> int:
    """Finds the index of a column in a dataframe by its name

    The search is case insensitive and can use regex

    Args:
        df (pd.DataFrame): dataframe to search
        column_name (str): column name to search

    Returns:
        int: index of the column
    """

    def compare_insensitive(x, y):
        return x.lower() == y.lower()

    def compare_regex(x, y):
        return search(y, x) is not None

    for i, col in enumerate(df.columns):
        if compare_insensitive(col, column_name) or compare_regex(col, column_name):
            return i

    return -1


VALIDATION_SCHEMA: SchemaType = {
    # Schema to validate the dataset
    # Key is the column metadata pattern and the value is a list of validators.
    # Validators are a tuple of a function and a informative message about the check.
    # The function should be applied to a pd.Series and return a boolean.
    #     True if the series is valid
    #     False if the series is invalid
    "categorical": _is_instance(str),
    "numeric": (
        lambda x: not x or search(r"^[-\d\.][\.,\d]*$", str(x)) is not None,
        "column $ should be numeric",
    ),
    "smiles": [
        _is_instance(str, msg="smile column $ should be str"),
        (is_valid_smiles_value, "column $ should be a valid smiles"),
    ],
    "string": _is_instance(str),
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
        columns_metadata: List[ColumnsDescription],
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
        file_key, _ = upload_s3_compressed(file, Bucket.Datasets.value)
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
