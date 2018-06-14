import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss

from sleepmind.base import BaseTransformer


class XGBoostClassifier(BaseTransformer):
    """XGBoost classifier."""
    def __init__(self, num_boost_round=10, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'multi:softprob'})

    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        dtrain = xgb.DMatrix(X, label=(y-1))
        self.clf = xgb.train(params=self.params, dtrain=dtrain,
                             num_boost_round=num_boost_round)

    def predict(self, X):
        predictions = self.predict_proba(X)
        y_hat = np.argmax(predictions, axis=1)
        return np.array(y_hat)

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)

    def score(self, X, y):
        predictions = self.predict_proba(X)
        return 1 / log_loss(y, predictions)

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self
