from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin


class BaseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def get_params(self, deep=True):
        return dict()
