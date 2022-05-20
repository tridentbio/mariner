import pandas as pd

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
