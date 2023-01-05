"""
This module is responsible for data validation. It has functions to validate
single values and dataframes.
"""
from io import BytesIO
from re import search
from typing import List, Literal, NewType, Optional, Tuple, Union

import pandas as pd
from rdkit import Chem, RDLogger

from mariner.core.aws import Bucket, upload_s3_compressed
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


def _is_instance(
    type: type, msg: Optional[str] = None, nullable=True
) -> Tuple[callable, str]:
    """
    Function validator creator
    """
    msg = f"column $ should be {type.__name__}" if not msg else msg

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
    """
    Class to check if the columns metadata is compatible with the dataset provided
    Validations will be based on the validate_schema
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
        """
        if type:
            self.errors[type].append(msg)
        if new_log:
            self.errors["logs"].append(msg)

    def check_is_compatible(self):
        """
        Checks if the columns metadata is compatible with the dataset

        :param df DataFrame: dataset
        :param columns_metadata List[ColumnsMeta]: columns metadata
        """
        for column_metadata in self.columns_metadata:
            i = find_column_i(self.df, column_metadata.pattern)

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

                for validation in validations:

                    func, msg = validation
                    is_valid = None

                    try:
                        valid_row_serie = self.df.iloc[:, i].apply(func)
                        is_valid = valid_row_serie.all()
                        if not is_valid:
                            invalid_rows: list = self.df.loc[
                                ~valid_row_serie, self.df.columns[i]
                            ].to_list()
                            self.error_column.loc[~valid_row_serie] += (
                                self.column_message(self.df.columns[i], msg) + " | "
                            )
                            for index, val in enumerate(
                                invalid_rows[: self.row_error_limit]
                            ):
                                self.add_error(
                                    self.row_message(index, val, self.df.columns[i]),
                                    "rows",
                                    new_log=False,
                                )

                    except Exception as e:
                        self.errors["logs"].append(
                            (f"Error validating column {self.df.columns[i]}: {e}")
                        )

                    if not is_valid:
                        self.errors["columns"].append(
                            self.column_message(self.df.columns[i], msg)
                        )
                        self.has_error = True

    def generate_errors_dataset(self):
        """
        Generate a new dataset with a column with the errors
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
    def column_message(column: str, msg: str):
        return msg.replace("$", f'"{column}"')

    @staticmethod
    def row_message(index: int, row: str, column: str):
        return f"Value {row} in row '{index}' of column" f" '{column}' is invalid"
