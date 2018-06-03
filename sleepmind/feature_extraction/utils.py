import json

import numpy as np


def unsquash(X):
    """Transform vector of dim (n,) into (n,1)."""
    if len(X.shape) == 1 or X.shape[0] == 1:
        return np.asarray(X).reshape((len(X), 1))
    else:
        return X


def squash(X):
    """Transform vector of dim (n,1) into (n,)."""
    return np.squeeze(np.asarray(X))


def extract_json(text, fields):
    """Extract specified fields from text."""
    if not isinstance(fields, list):
        fields = [fields]
    obj = json.loads(text)
    return ' '.join([obj.get(field) for field in fields if obj.get(field)])
