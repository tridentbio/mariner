"""
Mariner utility functions
"""
import hashlib
import io
from gzip import compress, decompress
from pathlib import Path
from typing import IO, BinaryIO, Union

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

    except AttributeError as e:
        raise TypeError("file must be instance of file") from e
