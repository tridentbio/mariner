import hashlib
import io
from gzip import compress, decompress
from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd
from fastapi.datastructures import UploadFile as FAUploadFile
from starlette.datastructures import UploadFile as SUploadFile
from wonderwords import RandomWord


def hash_md5(
    data: Union[bytes, None] = None,
    file: Union[str, Path, BinaryIO, FAUploadFile, None] = None,
    chunk_size=4096,
) -> str:
    """Calculate md5 hash of a file

    Need to provide either data or file

    Args:
        data (Union[bytes, None], optional):
            file instance. Defaults to None.
        file (Union[str, Path, BinaryIO, FAUploadFile, None], optional):
            file instance. Defaults to None.
        chunk_size (int, optional):
            chunk to calculate hash by iteration. Defaults to 4096.

    Raises:
        TypeError: if neither data nor file is provided

    Returns:
        str: md5 hash as a string
    """
    hash_md5 = hashlib.md5()
    if data:
        hash_md5.update(data)
    elif file:
        if isinstance(file, (str, Path)):
            with open(file, "rb") as f:
                # type: ignore
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    hash_md5.update(chunk)
        elif isinstance(file, io.BytesIO):

            for chunk in iter(lambda: file.read(chunk_size), b""):  # type: ignore
                hash_md5.update(chunk)
        elif isinstance(file, (FAUploadFile, SUploadFile)):

            for chunk in iter(lambda: file.file.read(chunk_size), b""):  # type: ignore
                hash_md5.update(bytes(chunk))
        else:
            raise TypeError("file must be UploadFile")
    else:
        raise TypeError('Either "file" or "data" should be provided')
    return hash_md5.hexdigest()


def random_pretty_name() -> str:
    """Generate a random name"""
    return RandomWord().word(word_min_length=4)


def is_compressed(file: Union[FAUploadFile, bytes, io.BytesIO, BinaryIO]) -> bool:
    """Check if file is compressed by checking the first two bytes

    Gzip compressed files start with b'\x1f\x8b'

    Args:
        file (Union[FAUploadFile, bytes, io.BytesIO]):
            file to check if it is compressed

    Raises:
        TypeError: if the file param is not a file

    Returns:
        bool: True if file is compressed, False otherwise
    """
    try:
        if isinstance(file, bytes):
            return file[0:2] == b"\x1f\x8b"

        file.seek(0)
        result = file.read(2) == b"\x1f\x8b"
        file.seek(0)
        return result
    except AttributeError as e:
        raise TypeError(f"file must be instance of file {e}")


def read_compressed_csv(fileBytes: bytes) -> pd.DataFrame:
    """Read csv file in bytes compressed by gzip

    First check if the file is compressed
    if it is then decompress it and read it as a csv file

    Args:
        fileBytes (bytes): bytes of the file to read

    Raises:
        TypeError: if the file is not compressed

    Returns:
        pd.DataFrame: dataframe of the csv file
    """
    if is_compressed(fileBytes):
        fileBytes = decompress(fileBytes)
        return pd.read_csv(io.BytesIO(fileBytes))
    else:
        raise TypeError("file must be UploadFile")


def decompress_file(file: io.BytesIO) -> io.BytesIO:
    """
    Decompress a gziped file

    :param file (io.BytesIO)
    """
    try:
        file = io.BytesIO(decompress(file.read()))
        file.seek(0)
        return file

    except AttributeError:
        raise TypeError("file must be instance of file")

    except Exception as e:
        raise e


def get_size(file: Union[io.BytesIO, BinaryIO]) -> int:
    """Get the size of a file in bytes

    Args:
        file (io.BytesIO): file to get size of

    Returns:
        int: size of file in bytes
    """
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    return size


def compress_file(file: Union[io.BytesIO, BinaryIO]) -> io.BytesIO:
    """Compress a file using gzip

    Args:
        file (io.BytesIO): file to compress

    Raises:
        TypeError: raised if file param is not instance of file
        e: raised if any other error occurs

    Returns:
        io.BytesIO: compressed file
    """
    try:
        file.seek(0)
        file = io.BytesIO(compress(file.read()))
        file.seek(0)
        return file

    except AttributeError:
        raise TypeError("file must be instance of file")

    except Exception as e:
        raise e
