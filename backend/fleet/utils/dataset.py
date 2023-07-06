import io
from gzip import decompress
from typing import Union

import pandas as pd


def is_compressed(file: bytes) -> bool:
    """Check if file is compressed by checking the first two bytes

    Gzip compressed files start with b'\x1f\x8b'
    """
    return file[0:2] == b"\x1f\x8b"


def decompress_file(fileBytes: bytes) -> pd.DataFrame:
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
