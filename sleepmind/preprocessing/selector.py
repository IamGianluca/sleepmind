from sklearn.base import BaseEstimator, TransformerMixin


class Selector(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def transform(self, X):
        return X.reindex(columns=self.columns)

    def fit(self, X, y=None):
        return self
