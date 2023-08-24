"""
Package with utility functions for file handling.

todo: Move this package as all other common utility functions between
mariner and fleet to a 'common' package outside both.
"""
from typing import IO, Union


def is_compressed(file: Union[bytes, IO]) -> bool:
    """
    Gzip compressed files start with b'\x1f\x8b'
    """
    gzip_mark = b"\x1f\x8b"

    if hasattr(file, "read"):
        file_prefix = file.read(2)  # type: ignore
        file.seek(0)  # type: ignore
        return file_prefix == gzip_mark
    elif isinstance(file, bytes):
        return file[0:2] == gzip_mark
    else:
        raise TypeError(
            f"file must be of type io.TextIOBase or bytes, not {type(file)}"
        )
