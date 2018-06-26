import numpy as np

from sleepmind.base import BaseTransformer


class LengthExtractor(BaseTransformer):
    @staticmethod
    def transform(X):
        result = [[len(variation[0])] for variation in X]
        return np.array(result)
