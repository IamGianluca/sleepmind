import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from sleepmind.models import XGBoostClassifier


def test_xgboost():
    # given
    train = pd.DataFrame({
        'city': [4, 1, 4, 2],
        'label': [1, 0, 1, 1],
    })
    cls = XGBoostClassifier(
        eta=0.3,
        objective='binary:logistic',
        eval_metric='auc',
        silent=1,
    )

    # when
    cls.fit(
        X=train.drop(['label'], axis=1),
        y=train.label
    )
    preds = cls.predict(X=train.drop(['label'], axis=1))

    # then
    assert_array_equal(preds, np.ones(4))


def test_get_params():
    params = {
        'num_boost_round': 15,
        'max_depth': 2,
        'eta': 1,
        'silent': 1,
        'objective': 'binary:logistic'
    }
    estimator = XGBoostClassifier(**params)
    assert estimator.get_params() == params


def test_set_params():
    params = {
        'num_boost_round': 15,
        'max_depth': 2,
        'eta': 1,
        'silent': 1,
        'objective': 'binary:logistic'
    }
    estimator = XGBoostClassifier()
    estimator.set_params(**params)
    assert estimator.get_params() == params
