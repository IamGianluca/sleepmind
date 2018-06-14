import numpy as np
import pandas as pd

from sleepmind.base import BaseTransformer


class CategoricalImputer(BaseTransformer):
    STRATEGIES = ['most_frequent']
    MISSING_VALUES = ['NaN']

    def __init__(self, missing_values='NaN', strategy='most_frequent'):
        assert strategy in self.STRATEGIES
        assert missing_values in self.MISSING_VALUES
        self.missing_values = missing_values
        self.strategy = strategy
        self.modes = None

    def fit(self, X, y=None):
        self.modes = X.mode().to_dict('records')[0]
        return self

    def transform(self, X):
        Xt = X.fillna(value=self.modes)
        return Xt

