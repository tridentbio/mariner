import io
import logging
from io import BytesIO
from typing import List, Literal, Union

import pandas as pd
import ray

from mariner.core.aws import Bucket, upload_s3_file
from mariner.schemas.dataset_schemas import (
    CategoricalDataType,
    ColumnsMeta,
    NumericalDataType,
    SmileDataType,
    StringDataType,
)
from mariner.stats import get_metadata, get_stats
from mariner.utils import hash_md5
from mariner.validation import is_valid_smiles_series
from model_builder.splitters import RandomSplitter, ScaffoldSplitter

LOG = logging.getLogger(__name__)


@ray.remote
class DatasetTransforms:
    """Dataset transformations and queries to be performed by ray cluster"""

    def __init__(
        self,
    ):
        self._file_input = io.BytesIO()
        self._is_dataset_fully_loaded = False
        self._df = None

    def write_dataset_buffer(self, chunk: bytes):
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
            self.df = self._df = pd.read_csv(self._file_input)

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
        return self._df

    @df.setter
    def df(self, value: pd.DataFrame):
        self._df = value

    def get_columns_metadata(self) -> List[ColumnsMeta]:
        metadata = [
            ColumnsMeta(
                name=key, dtype=self._infer_domain_type_from_series(self.df[key])
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
            if len(uniques) <= 100:
                return CategoricalDataType(
                    domain_kind="categorical",
                    classes={val: idx for idx, val in enumerate(uniques)},
                )
            return StringDataType(domain_kind="string")
        elif series.dtype == int:
            series = series.sort_values()
            uniques = series.unique()
            if len(uniques) <= 100:
                return CategoricalDataType(
                    domain_kind="categorical",
                    classes={val: idx for idx, val in enumerate(uniques)},
                )

    def get_entity_info_from_csv(self):
        stats = get_metadata(self.df)
        filesize = self.df.memory_usage(deep=True).sum()
        return len(self.df), len(self.df.columns), filesize, stats

    def get_dataset_summary(self):
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
        file = BytesIO()
        self.df.to_csv(file)
        file.seek(0)
        file_md5 = hash_md5(file=file)
        key = f"datasets/{file_md5}.csv"
        file.seek(0)
        upload_s3_file(file=file, bucket=Bucket.Datasets, key=key)
        return key
