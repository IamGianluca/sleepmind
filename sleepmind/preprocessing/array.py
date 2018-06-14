import numpy as np
from sleepmind.base import BaseTransformer


class ArrayTransformer(BaseTransformer):

    def transform(self, X):
        return X.as_matrix()


class Squeeze(BaseTransformer):
    """Remove single-dimensional entries from the shape of an array."""

    @staticmethod
    def transform(X):
        return np.squeeze(np.asarray(X))


class DenseTransformer(BaseTransformer):
    """Transform sparse matrix to a dense one."""

    @staticmethod
    def transform(X):
        return X.todense()
