import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series


def get_stats(df: DataFrame) -> DataFrame:
    """
    Extracts metadata from the dataset to be cached on
    the entity
    """
    dtypes: Series = pd.Series(
        list(map(str, df.dtypes)), name="types", index=df.columns
    )
    stats = df.describe(include="all")
    nacount: Series = df.isna().sum()
    nacount.name = "na_count"
    nacount.index = df.columns
    stats = stats.append(dtypes)
    stats = stats.append(nacount)
    stats = stats.fillna("NaN")
    return stats
