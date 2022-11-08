import hashlib
from pathlib import Path
from typing import BinaryIO, Optional, Union

from fastapi.datastructures import UploadFile
from wonderwords import RandomWord


def hash_md5(
    data: Optional[bytes] = None,
    file: Optional[Union[str, Path, BinaryIO, UploadFile]] = None,
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
        elif isinstance(file, UploadFile):
            for chunk in iter(lambda: file.file.read(chunk_size), b""):
                hash_md5.update(bytes(chunk))
        elif isinstance(file, BinaryIO):
            for chunk in iter(lambda: file.read(chunk_size), b""):
                hash_md5.update(chunk)
    else:
        raise TypeError('Either "content" or "fname" should be provided')
    return hash_md5.hexdigest()


def random_pretty_name() -> str:
    return RandomWord().word(word_min_length=4)
