from pathlib import Path

import pytest
from fastapi.datastructures import UploadFile

from mariner.utils import hash_md5


def test_hash_md5(tmp_path: Path) -> None:
    file1 = tmp_path / "file1"
    file2 = tmp_path / "file2"
    file1.touch()
    file2.touch()
    file1.write_bytes(b"foo")
    file2.write_bytes(b"foo")
    with open(file1, "rb") as file3:
        uploadfile = UploadFile("file3", file3)
        assert hash_md5(data=b"foo") == hash_md5(file=file1)
        assert hash_md5(file=uploadfile) == hash_md5(file=file1)
        assert hash_md5(file=file1) == hash_md5(file=file1)
        assert hash_md5(file=file1) == hash_md5(file=file2)
        file2.write_text("baz")
        assert hash_md5(file=file1) != hash_md5(file=file2)


def test_hash_md5_no_args() -> None:
    with pytest.raises(TypeError):
        hash_md5()
