from pathlib import Path

import pandas as pd
import pytest
from fastapi import UploadFile
from fastapi.encoders import jsonable_encoder

from app.features.dataset.controller import get_entity_info_from_csv
from app.features.dataset.schema import (
    CategoricalDataType,
    ColumnsDescription,
    NumericalDataType,
)
from app.features.dataset.utils import get_stats


def test_get_stats():
    df = pd.DataFrame(
        {
            "a": [1, 1, 1, 1],
            "b": ["a", "b", "c", "d"],
        }
    )
    stats = get_stats(df)
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
    columns_descriptions = [
        ColumnsDescription(
            pattern="a",
            description="THIS IS A",
            dataset_id=3,
            data_type=NumericalDataType(domain_kind="numerical"),
        ),
        ColumnsDescription(
            pattern="b",
            description="THIS IS B",
            dataset_id=3,
            data_type=CategoricalDataType(
                domain_kind="categorical", classes={"a": 0, "b": 1, "c": 2}
            ),
        ),
        ColumnsDescription(
            pattern="c",
            description="THIS IS C",
            dataset_id=3,
            data_type=CategoricalDataType(
                domain_kind="categorical", classes={100: 0, 200: 1, 300: 2}
            ),
        ),
    ]
    nrows, ncols, fsize, stats = get_entity_info_from_csv(
        dataset_file, columns_descriptions
    )
    assert nrows == 5
    assert "a" in stats
    assert "b" in stats
    assert "c" in stats
    assert stats["b"]["categories"] == {
        "a": 0,
        "b": 1,
        "c": 2,
    }
    assert stats["c"]["categories"] == {100: 0, 200: 1, 300: 2}
    assert "categories" not in stats["a"]
