import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LengthExtractor(BaseEstimator, TransformerMixin):

    @staticmethod
    def transform(X):
        result = [[len(variation[0])] for variation in X]
        return np.array(result)

    def fit(self, X, y=None):
        return self
