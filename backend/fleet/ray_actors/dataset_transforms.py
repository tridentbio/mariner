"""
Actors for dataset processing
"""
import io
import logging
from io import BytesIO
from typing import Any, List, Literal, Optional, Tuple, Union, get_args

import pandas as pd
import ray

from fleet.dataset_schemas import StatsType
from fleet.model_builder.splitters import apply_split_indexes
from fleet.stats import get_metadata, get_stats
from fleet.utils.dataset import converts_file_to_dataframe
from fleet.validation.checker import CompatibilityChecker, ErrorsType
from fleet.validation.functions import (
    check_biological_sequence_series,
    is_valid_smiles_series,
    validate_column_pattern,
)
from mariner.core.aws import Bucket, upload_s3_compressed
from mariner.schemas.dataset_schemas import (
    BiologicalDataType,
    CategoricalDataType,
    ColumnsDescription,
    ColumnsMeta,
    DNADataType,
    NumericalDataType,
    ProteinDataType,
    RNADataType,
    SmileDataType,
    StringDataType,
)

LOG = logging.getLogger(__name__)


def infer_domain_type_from_series(
    series: pd.Series, strict=False, len_uniques_to_categorical=100
) -> Any:

    """Infers the domain type from a pd.Series

    Checks some type of series and returns the corresponding
    domain type based on the priority:
        1. If the series is a float:
            it will be considered as a numerical data type
        2. If the series is an object:
            it will be checked if it is a smiles first, then if it is a
            categorical data type
        3. If the series is an int:
            it will be checked if it is a categorical data type

    Args:
        series (pd.Series): the series to infer the domain type from

    Returns:
        Any: inferred domain type
    """
    if series.dtype == float:
        return NumericalDataType(domain_kind="numeric")
    elif series.dtype == object:
        # check if it is a biological sequence
        bio_info = check_biological_sequence_series(series)
        if bio_info["valid"]:
            TypeClass: BiologicalDataType = {
                "dna": DNADataType,
                "rna": RNADataType,
                "protein": ProteinDataType,
            }[bio_info["type"]]
            return TypeClass(**bio_info["kwargs"])

        # check if it is smiles
        if is_valid_smiles_series(series):
            return SmileDataType(domain_kind="smiles")

        # check if it is likely to be categorical
        series = series.sort_values()
        uniques = series.unique()
        if len(uniques) / len(series) < 0.4:
            return CategoricalDataType(
                domain_kind="categorical",
                # Classes will need to be overwritten by complete data type checking
                classes={val: idx for idx, val in enumerate(uniques)},
            )

        return StringDataType(
            domain_kind="string",
        )
    elif series.dtype == int:
        series = series.sort_values()
        uniques = series.unique()
        if len(uniques) <= len_uniques_to_categorical:
            return CategoricalDataType(
                domain_kind="categorical",
                classes={val: idx for idx, val in enumerate(uniques)},
            )
    else:
        if strict:
            raise ValueError(f"Unknown dtype {series.dtype}")


def get_columns_metadata(
    df: pd.DataFrame,
    first_n_rows=20,
    strict=False,
    len_uniques_to_categorical=100,
) -> List[ColumnsMeta]:
    """Returns data type about columns based on the underlying dataframe types

    Returns:
        List[ColumnsMeta]
    """
    metadata = [
        ColumnsMeta(
            name=key,
            dtype=infer_domain_type_from_series(
                df[key][: min(first_n_rows, len(df))],
                strict=strict,
                len_uniques_to_categorical=len_uniques_to_categorical,
            ),
        )
        for key in df
    ]

    return metadata


@ray.remote
class DatasetTransforms:
    """Dataset transformations and queries to be performed by ray cluster

    Each instance of this class is responsible for a single dataset file
    transformations. This implementation uses :class:`pandas.Dataframe`
    for operating on the dataset :func:`DatasetTransforms.write_dataset_buffer` and
    :func:`DatasetTransforms.set_is_dataset_fully_loaded(True)` once the dataset
    is fully loaded
    """

    def __init__(
        self,
        df: Union[pd.DataFrame, None] = None,
    ):
        self._file_input = io.BytesIO()
        self._is_dataset_fully_loaded = False
        self._df = df

    def write_dataset_buffer(self, chunk: bytes):
        """Writes to the underlying csv file

        Bytes are written in sequence

        TODO: Could be optimized like the following:
            1. Create buffer with the size of the file input
            2. Change signature of this method to have have offset of type int
            3. Parellize writing of chunks to the buffer

        Args:
            chunk: bytes to write on the buffer
        """
        self._file_input.write(chunk)

    @property
    def is_dataset_fully_loaded(self):
        return self._is_dataset_fully_loaded

    @is_dataset_fully_loaded.setter
    def is_dataset_fully_loaded(self, value: bool):
        """Sets the dataset as fully loaded

        Once the dataset is fully loaded, self.df will be populated
        with the dataset
        If the dataset is already loaded, this method will do nothing
        If the dataset is compressed, it will be decompressed before
        loading into the dataframe

        Args:
            value (bool): True if the dataset is fully loaded
        """
        if self._is_dataset_fully_loaded:
            LOG.warning(
                "Can't update a dataset already loaded into dataframe'"
            )
            return
        self._is_dataset_fully_loaded = value
        if value:
            self.df = converts_file_to_dataframe(self._file_input)

    def get_dataframe(self) -> pd.DataFrame:
        """Returns the underlying dataframe

        Returns:
            pd.DataFrame: the underlying dataframe
        """
        return self.df

    def set_is_dataset_fully_loaded(self, val: bool):
        """Sets the dataset as fully loaded

        Args:
            val (bool): True if the dataset is fully loaded
        """
        self.is_dataset_fully_loaded = val

    def get_is_dataset_fully_loaded(self) -> bool:
        """Returns if the dataset is fully loaded

        Returns:
            bool: True if the dataset is fully loaded
        """
        return self.is_dataset_fully_loaded

    @property
    def df(self) -> pd.DataFrame:
        """Checks if the dataframe exists before returning it

        Raises:
            RuntimeError: If the dataset is not fully loaded
            AssertionError: If the dataframe is None

        Returns:
            pd.DataFrame: the underlying dataframe
        """
        if not self._is_dataset_fully_loaded:
            raise RuntimeError(
                "Must set dataset as loaded before using dataframe"
            )
        assert self._df is not None, "loading dataset as dataframe failed"
        return self._df

    @df.setter
    def df(self, value: pd.DataFrame):
        """Sets the underlying dataframe

        Args:
            value (pd.DataFrame): the dataframe to set
        """
        self._df = value

    def get_columns_metadata(self, first_n_rows=20) -> List[ColumnsMeta]:
        """Returns data type about columns based on the underlying dataframe types

        Returns:
            List[ColumnsMeta]
        """
        return get_columns_metadata(self.df, first_n_rows=first_n_rows)

    def apply_split_indexes(
        self,
        split_type: Literal["random", "scaffold"],
        split_target: str,
        split_column: Union[str, None] = None,
    ):
        """Separates dataframe row into training, testing and validation

        Adds a 'step' column to the current dataframe, associating the row to
        some data science model stage, i.e. training, testing or validation

        Args:
            split_type: "random" or "scaffold", strategy used to to split examples
            split_target: The wanted distribution of training/validation/testing
            split_column: Only required for scaffold, must be a columns of Smiles
            data type

        Raises:
            NotImplementedError: in case the split_type is not recognized
        """
        apply_split_indexes(
            self.df,
            split_type=split_type,
            split_target=split_target,
            split_column=split_column,
        )

    def get_entity_info_from_csv(self):
        """Gets the row count, column count, and a dictionary
        with basic statistics for each columns data series (median, percentiles,
                                                            max, min, ...)
        """
        stats = get_metadata(self.df)
        return len(self.df), len(self.df.columns), stats

    def get_dataset_summary(
        self, columns_metadata: List[ColumnsDescription] = []
    ) -> StatsType:
        """Gets histogram for dataset columns according to it's inferred type

        Columns for which histograms are generated must be of type int, float,
        smiles, categorical or biological.

        Args:
            columns_metadata (List[ColumnsDescription], optional):
                List of columns metadata. Only used with biological data types.

        Returns:
            Dict[str, Dict[str, Any]]:
                Dictionary with histograms for each column in the dataset
        """
        # Detect the smiles column name
        smiles_columns = []
        for col in self.df.columns:
            if is_valid_smiles_series(self.df[col], weak_check=True):
                smiles_columns.append(col)

        # Get the biological and categorical columns
        biological_columns = []
        categorical_columns = []
        for metadata in columns_metadata:
            if type(metadata.data_type) in get_args(BiologicalDataType):
                biological_columns.extend(
                    [
                        {"col": col, "metadata": metadata}
                        for col in self.df.columns
                        if validate_column_pattern(col, metadata.pattern)
                    ]
                )
            if isinstance(metadata.data_type, CategoricalDataType):
                categorical_columns.extend(
                    [
                        col
                        for col in self.df.columns
                        if validate_column_pattern(col, metadata.pattern)
                    ]
                )

        stats = get_stats(
            self.df, smiles_columns, biological_columns, categorical_columns
        )
        return stats

    def upload_s3(self, old_data_url=None):
        """
        Uploads transformed dataframe to s3 in the csv format compressed with gzip

        TODO:
            Delete old dataset if it exists
        """
        file = BytesIO()
        self.df.to_csv(file, index=False)
        file.seek(0)
        key, file_size = upload_s3_compressed(file, bucket=Bucket.Datasets)

        # TODO check why it is not working (permission denied error)
        # if old_data_url:
        #     delete_s3_file(key=old_data_url, bucket=Bucket.Datasets)

        return key, file_size

    def check_data_types(
        self, columns: List[ColumnsDescription]
    ) -> Tuple[List[ColumnsDescription], Optional[ErrorsType]]:
        """Checks if underlying dataset conforms to columns data types

        If validation succeeds, updates categorical data types to the right
        number of classes
        If validation fails, generates a dataset with the errors details

        Args:
            columns: objects containing information about data types

        Returns:
            Tuple[ColumnsMeta, Optional[List[str]]]:
                updated columns and errors if has error
        """
        checker = CompatibilityChecker(columns_metadata=columns, df=self.df)
        checker.check_is_compatible()

        if checker.has_error:
            checker.generate_errors_dataset()
        else:
            for i, col in enumerate(columns):
                if col.data_type.domain_kind in ["categorical", "dna", "rna"]:
                    columns[i].data_type = infer_domain_type_from_series(
                        self.df[col.pattern]
                    )

        return columns, checker.errors if checker.has_error else None
