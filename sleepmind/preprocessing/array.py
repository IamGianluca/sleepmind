import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ArrayTransformer(BaseEstimator, TransformerMixin):

    def transform(self, X):
        return X.as_matrix()

    def fit(self, X, y=None):
        return self


class Squeeze(BaseEstimator, TransformerMixin):
    """Remove single-dimensional entries from the shape of an array."""

    @staticmethod
    def transform(X):
        return np.squeeze(np.asarray(X))

    def fit(self, X, y=None):
        return self


class DenseTransformer(BaseEstimator, TransformerMixin):
    """Transform sparse matrix to a dense one."""

    @staticmethod
    def transform(X):
        return X.todense()

    def fit(self, X, y=None):
        return self
