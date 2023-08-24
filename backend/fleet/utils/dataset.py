import io
from gzip import decompress
from typing import List, Literal, Tuple, Union

import pandas as pd

from fleet.data_types import SmileDataType
from fleet.dataset_schemas import TorchDatasetConfig
from fleet.file_utils import is_compressed
from fleet.validation.functions import is_valid_smiles_series


def decompress_file(fileBytes: bytes) -> io.BytesIO:
    """
    Converts a csv file in bytes compressed by gzip to a BytesIO object
    """
    fileBytes = decompress(fileBytes)
    return io.BytesIO(fileBytes)


def converts_file_to_dataframe(file: Union[bytes, io.BytesIO]) -> pd.DataFrame:
    """Converts a file to a dataframe

    Args:
        file: file to convert to dataframe

    Returns:
        pd.DataFrame: dataframe of the file
    """
    if isinstance(file, io.BytesIO):
        file.seek(0)
        file = file.read()

    return pd.read_csv(
        decompress_file(file) if is_compressed(file) else io.BytesIO(file)
    )


InvalidReason = Literal["invalid-smiles-series"]


def check_dataframe_conforms_dataset(
    df: pd.DataFrame, dataset_config: "TorchDatasetConfig"
) -> List[Tuple[str, InvalidReason]]:
    """Checks if dataframe conforms to data types specified in dataset_config

    Args:
        df: dataframe to be checked
        dataset_config: model builder dataset configuration used in model building
    """
    column_configs = (
        dataset_config.feature_columns + dataset_config.target_columns
    )
    result: List[Tuple[str, InvalidReason]] = []
    for column_config in column_configs:
        data_type = column_config.data_type
        if isinstance(data_type, SmileDataType):
            series = df[column_config.name]
            if not is_valid_smiles_series(series):
                result.append((column_config.name, "invalid-smiles-series"))
    return result
