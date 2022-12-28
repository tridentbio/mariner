import io
import logging
from io import BytesIO
from typing import List, Literal, Tuple, Union

import pandas as pd
import ray

from mariner.core.aws import Bucket, upload_s3_file
from mariner.schemas.dataset_schemas import (
    CategoricalDataType,
    ColumnsMeta,
    NumericalDataType,
    SmileDataType,
)
from mariner.stats import get_metadata, get_stats
from mariner.utils import decompress_file, hash_md5
from mariner.validation import check_is_compatible, is_valid_smiles_series
from model_builder.splitters import RandomSplitter, ScaffoldSplitter

LOG = logging.getLogger(__name__)


@ray.remote
class DatasetTransforms:
    """Dataset transformations and queries to be performed by ray cluster

    Each instance of this class is responsible for a single dataset file
    transformations. To use the actor, the programmer should write the dataset
    to the remote ray actors by chunks using.

    This implementation uses :class:`pandas.Dataframe` for operating on the dataset
    :func:`DatasetTransforms.write_dataset_buffer` and
    :func:`DatasetTransforms.set_is_dataset_fully_loaded(True)` once the dataset
    is fully loaded
    """

    def __init__(
        self,
        is_compressed: bool = False,
    ):
        self._file_input = io.BytesIO()
        self._is_dataset_fully_loaded = False
        self._df = None
        self.is_compressed = is_compressed

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
        if self._is_dataset_fully_loaded:
            LOG.warning("Can't update a dataset already loaded into dataframe'")
            return
        self._is_dataset_fully_loaded = value
        if value:
            self._file_input.seek(0)
            self.df = self._df = pd.read_csv(
                decompress_file(self._file_input)
                if self.is_compressed
                else self._file_input
            )

    def get_dataframe(self):
        return self.df

    def set_is_dataset_fully_loaded(self, val: bool):
        self.is_dataset_fully_loaded = val

    def get_is_dataset_fully_loaded(self):
        return self.is_dataset_fully_loaded

    @property
    def df(self) -> pd.DataFrame:
        if not self._is_dataset_fully_loaded:
            raise RuntimeError("Must set dataset as loaded before using dataframe")
        assert self._df is not None, "loading dataset as dataframe failed"
        return self._df

    @df.setter
    def df(self, value: pd.DataFrame):
        self._df = value

    def get_columns_metadata(self, first_n_rows=20) -> List[ColumnsMeta]:
        """Returns data type about columns based on the underlying dataframe types

        Returns:
            List[ColumnsMeta]
        """
        metadata = [
            ColumnsMeta(
                name=key,
                dtype=self._infer_domain_type_from_series(
                    self.df[key][: min(first_n_rows, len(self.df))]
                ),
            )
            for key in self.df
        ]

        return metadata

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
        train_size, val_size, test_size = map(
            lambda x: int(x) / 100, split_target.split("-")
        )
        if split_type == "random":
            splitter = RandomSplitter()
            self.df = splitter.split(self.df, train_size, test_size, val_size)
        elif split_type == "scaffold":
            splitter = ScaffoldSplitter()
            assert (
                split_column is not None
            ), "split column can't be none when split_type is scaffold"
            self.df = splitter.split(
                self.df, split_column, train_size, test_size, val_size
            )
        else:
            raise NotImplementedError(f"{split_type} splitting is not implemented")

    def _infer_domain_type_from_series(self, series: pd.Series):
        if series.dtype == float:
            return NumericalDataType(domain_kind="numeric")
        elif series.dtype == object:
            # check if it is smiles
            if is_valid_smiles_series(series):
                return SmileDataType(domain_kind="smiles")
            # check if it is likely to be categorical
            series = series.sort_values()
            uniques = series.unique()
            return CategoricalDataType(
                domain_kind="categorical",
                # Classes will need to be overwritten by complete data type checking
                classes={val: idx for idx, val in enumerate(uniques)},
            )
        elif series.dtype == int:
            series = series.sort_values()
            uniques = series.unique()
            if len(uniques) <= 100:
                return CategoricalDataType(
                    domain_kind="categorical",
                    classes={val: idx for idx, val in enumerate(uniques)},
                )

    def get_entity_info_from_csv(self):
        """Gets the row count, column count, dataset file size, and a dictionary
        with basic statistics for each columns data series (median, percentiles,
                                                            max, min, ...)
        """
        stats = get_metadata(self.df)
        filesize = self.df.memory_usage(deep=True).sum()
        return len(self.df), len(self.df.columns), filesize, stats

    def get_dataset_summary(self):
        """Get's histogram for dataset columns according to it's inferred type

        Columns for which histograms are generated must be of type int or float,
        or must be valid smiles columns
        """
        # Detect the smiles column name
        smiles_columns = []
        for col in self.df.columns:
            # Must go to dataset actor
            if is_valid_smiles_series(self.df[col], weak_check=True):
                smiles_columns.append(col)
        # Must go to dataset actor
        stats = get_stats(self.df, smiles_columns)
        return stats

    def upload_s3(self):
        """Uploads transformed dataframe to s3 in the csv format

        TODO: Compress data and adjust dataframe getter methods in the
        dataset
        """
        file = BytesIO()
        self.df.to_csv(file)
        file.seek(0)
        file_md5 = hash_md5(file=file)
        key = f"datasets/{file_md5}.csv"
        file.seek(0)
        upload_s3_file(file=file, bucket=Bucket.Datasets, key=key)
        return key

    def check_data_types(self, columns: ColumnsMeta) -> Tuple[ColumnsMeta, List[str]]:
        """Checks if underlying dataset conforms to columns data types

        If validation succeeds, updates categorical data types to the right
        number of classes


        Args:
            columns: objects containing information about data types
        """
        errors = check_is_compatible(self.df, columns)
        return columns, errors
