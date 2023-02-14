from io import BytesIO
from typing import Dict, List, Literal, NewType, Optional, Union, cast

import pandas as pd

from mariner.core.aws import Bucket, upload_s3_compressed
from mariner.schemas.dataset_schemas import ColumnsDescription, SchemaType

from .functions import find_column_by_name
from .validation_schema import VALIDATION_SCHEMA

ErrorsType = NewType(
    "ErrorsType",
    Dict[
        Literal["rows", "columns", "logs", "dataset_error_key"],
        Union[list, str, None],
    ],
)


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
        self.errors = cast(
            ErrorsType,
            {
                "columns": [],
                "rows": [],
                "logs": [],
                "dataset_error_key": None,
            },
        )
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
                    f"Column {column_metadata.pattern} not found in dataset",
                    type="columns",
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
                                    type="rows",
                                    new_log=False,
                                )

                    except Exception as e:
                        # When an unexpected error occurs, save error as a log
                        self.add_error(
                            f"Error validating column {self.df.columns[i]}: {e}",
                            new_log=True,
                        )

                    if not is_valid:
                        self.add_error(
                            self.column_message(self.df.columns[i], msg),
                            type="columns",
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
