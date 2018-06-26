import logging

import numpy as np
from scipy import sparse
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import binarize

from sleepmind.base import BaseTransformer
from sleepmind.feature_extraction.utils import unsquash, squash

logger = logging.getLogger(__name__)


class ModelTransformer(BaseTransformer):
    """Use one model's prediction as features for another model.

    Wraps other models to perform transformation, e.g. a k-means to obtain
    cluster index as new feature; or for use in a model stack (an estimator
    trained using as features the predictions of other models).
    """

    def __init__(self, model, probs=True):
        self.model = model
        self.probs = probs

    def get_params(self, deep=True):
        return dict(model=self.model, probs=self.probs)

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X):
        if self.probs:
            predictions = self.model.predict_proba(X)[:, 1]
        else:
            predictions = self.model.predict(X)
        return unsquash(predictions)


class FeatureStack(BaseTransformer):
    """Stacks several transformer objects to return concatenated features.
    Similar to FeatureUnion, a list of tuples `(name, estimator)` is passed to
    the constructor. Not parallel. But useful for debugging when
    e.g. FeatureUnion doesn't work.
    """

    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def get_feature_names(self):
        pass

    def fit(self, X, y=None):
        for name, trans in self.transformer_list:
            trans.fit(X, y)
        return self

    def transform(self, X):
        logger.info("Stacking models: {}".format(str(self.transformer_list)))
        features = []
        for name, trans in self.transformer_list:
            logger.info("Stack next step: {}".format(name))
            predictions = trans.transform(X)
            features.append(predictions)
            logger.info(
                "Completed stacking model {}: output type {} , "
                "output shape: {}".format(
                    name, type(predictions), predictions.shape
                )
            )
        issparse = [sparse.issparse(f) for f in features]
        if np.any(issparse):
            # convert to sparse if necessary, otherwise cannot be hstack'ed
            features = [sparse.csr_matrix(unsquash(f)) for f in features]
            features = sparse.hstack(features).tocsr()
        else:
            features = np.column_stack(features)

        logger.info(
            "Completed stacking: output shape: {}".format(features.shape)
        )
        return features

    def get_params(self, deep=True):
        if not deep:
            return super(FeatureStack, self).get_params(deep=False)
        else:
            out = dict(self.transformer_list)
            for name, trans in self.transformer_list:
                for key, value in trans.get_params(deep=True).iteritems():
                    out["%s__%s" % (name, key)] = value
            return out


class EnsembleBinaryClassifier(BaseTransformer, ClassifierMixin):
    """Average or majority-vote several different classifiers. Assumes input is
    a matrix of individual predictions, such as the output of a FeatureUnion
    of ModelTransformers [n_samples, n=predictors]. Also see
    http://sebastianraschka.com/Articles/2014_ensemble_classifier.html.
    """

    def __init__(self, mode, weights=None):
        self.mode = mode
        self.weights = weights

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        """Predict (weighted) probabilities """
        probs = np.average(X, axis=1, weights=self.weights)
        return np.column_stack((1 - probs, probs))

    def predict(self, X):
        """Predict class labels."""
        if self.mode == "average":
            return binarize(self.predict_proba(X)[:, [1]], 0.5)
        else:
            res = binarize(X, 0.5)
            return np.apply_along_axis(
                lambda x: np.bincount(x.astype(int), self.weights).argmax(),
                axis=1,
                arr=res,
            )


class Length(BaseTransformer):
    """

    Notes: Assumes column of data type compatible with len function.
    """

    def __init__(self, column=0):
        self.column = column

    def get_params(self, deep=True):
        return dict(column=self.column)

    def transform(self, X):
        col = X.reindex(self.column)
        res = np.vectorize(len)(col).astype(float)
        return unsquash(res)


# Turns a single array into a matrix with single column
# This allows concatenating the predictions of different estimators to be used
# as features in a feature union (since hstack doesn't produce a 2-ol matrix
# from two arrays).
# Note: this is already been taken care of in ModelTransformer
class Squash(BaseTransformer):
    def transform(self, X, **transform_params):
        return squash(X)


class Unsquash(BaseTransformer):
    def transform(self, X, **transform_params):
        return unsquash(X)


class Float(BaseTransformer):
    def transform(self, X, **transform_params):
        return X.astype(float)


class Cast(BaseTransformer):
    """Extract variation from dataset.

    Args:
        expected_type (type): A valid Python data type.
    """

    def __init__(self, expected_type):
        self.expected_type = expected_type

    def transform(self, X):
        """Cast elements of `X` into expected type.

        Args:
            X (list): The input.

        Returns:
            (np.array) The casted output.
        """
        return np.array(X, dtype=self.expected_type)

    def fit(self, X, y=None):
        return self


class TargetStatisticsEncoding(BaseTransformer):
    def __init__(self, frequencies):
        self.frequencies = frequencies

    def transform(self, X):
        result = []
        for key in X:
            key = key[0]
            if key in self.frequencies.keys():
                result.append([self.frequencies[key]])
            else:
                result.append([self.frequencies[None]])
        return np.array(result)

    def fit(self, X, y=None):
        return self
