import hashlib
import io
from gzip import decompress
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


def read_compressed_csv(fileBytes: bytes) -> pd.DataFrame:
    if fileBytes[0:2] == b"\x1f\x8b":
        fileBytes = decompress(fileBytes)
        return pd.read_csv(io.BytesIO(fileBytes), dtype=str)
    else:
        raise TypeError("file must be UploadFile")
