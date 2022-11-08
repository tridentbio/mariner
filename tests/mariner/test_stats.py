from pathlib import Path

import pandas as pd
import pytest
from fastapi import UploadFile
from fastapi.encoders import jsonable_encoder

from mariner.datasets import get_entity_info_from_csv
from mariner.stats import get_metadata


def test_get_stats():
    df = pd.DataFrame(
        {
            "a": [1, 1, 1, 1],
            "b": ["a", "b", "c", "d"],
        }
    )
    stats = get_metadata(df)
    assert len(df.columns) == 2
    assert "min" in stats.index
    assert "max" in stats.index
    assert "count" in stats.index
    assert "na_count" in stats.index
    assert "types" in stats.index
    statsdict = jsonable_encoder(stats.to_json())
    assert "a" in statsdict
    assert "b" in statsdict


@pytest.fixture(scope="function")
def dataset_file(tmp_path: Path):
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.4, 5.1, 2.3],
            "b": ["a", "a", "b", "c", "c"],
            "c": [100, 100, 200, 300, 300],
        }
    )
    df.to_csv(tmp_path / "ds.csv")
    with open(tmp_path / "ds.csv", "rb") as f:
        yield UploadFile("dataset", f)


def test_get_entity_info_from_csv(dataset_file: UploadFile):
    nrows, ncols, fsize, stats = get_entity_info_from_csv(dataset_file)
    assert nrows == 5
    assert "a" in stats
    assert "b" in stats
    assert "c" in stats
