import numpy as np
import pandas as pd


def get_stats():
    df = pd.read_csv("Lipophilicity.csv")
    n_cols = len(df.columns)
    statsdf = df.describe(include="all")
    statsdf.to_numpy()
    types = np.array(list(map(lambda x: str(x), df.dtypes)))
    na_count = df.isna().sum(axis=0).to_numpy()
    matrix = np.concatenate(
        [
            types.reshape(1, n_cols),
            na_count.reshape(1, n_cols),
        ]
    )
    extradf = pd.DataFrame(matrix, index=["type", "na_count"])
    print(statsdf.shape)
    return pd.concat([statsdf, extradf], axis=0)


stats = get_stats()
print(stats)
