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
    hash_md5 = hashlib.md5()
    if data:
        hash_md5.update(data)
    elif file:
        if isinstance(file, (str, Path)):
            with open(file, "rb") as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    hash_md5.update(chunk)
        elif isinstance(file, io.BytesIO):
            for chunk in iter(lambda: file.read(chunk_size), b""):
                hash_md5.update(chunk)
        elif isinstance(file, (FAUploadFile, SUploadFile)):
            for chunk in iter(lambda: file.file.read(chunk_size), b""):
                hash_md5.update(bytes(chunk))
        else:
            raise TypeError("file must be UploadFile")
    else:
        raise TypeError('Either "file" or "data" should be provided')
    return hash_md5.hexdigest()


def random_pretty_name() -> str:
    return RandomWord().word(word_min_length=4)


def is_compressed(file: Union[FAUploadFile, bytes]) -> bool:
    """
    Check if file is compressed by checking the first two bytes
    gzip compressed files start with b'\x1f\x8b'

    :param file: file to check
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
    """
    Read a compressed csv file into a pandas dataframe

    :param fileBytes: bytes of the file
    :raise TypeError if file is not compressed
    """
    if is_compressed(fileBytes):
        fileBytes = decompress(fileBytes)
        return pd.read_csv(io.BytesIO(fileBytes), dtype=str)
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


def get_size(file: io.BytesIO):
    """
    Get the size of a file in bytes
    """
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    return size


def compress_file(file: io.BytesIO) -> io.BytesIO:
    """
    Compress a file into gzip format
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
