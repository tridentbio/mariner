import numpy as np


class NpConcatenate:
    def fit(self, *_):
        return self

    def transform(self, *args):
        return np.concatenate(*args, axis=-1)

    fit_transform = transform
    __call__ = transform
