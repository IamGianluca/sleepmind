import numpy as np
import pandas as pd
from sleepmind.base import BaseTransformer


class DateParser(BaseTransformer):

    def __init__(self):
        self.attrs = ['year', 'month', 'week', 'day', 'dayofweek',
                      'dayofyear', 'days_in_month', 'quarter',
                      'is_month_start', 'is_month_end', 'is_quarter_start',
                      'is_quarter_end', 'is_year_start', 'is_year_end',
                      'is_leap_year']

    def transform(self, X):
        cols = np.empty((X.shape[0], 0))
        ll = []
        for col_name, col_values in X.iteritems():
            ll.append(['_'.join([col_name, attr]) for attr in self.attrs])
            c = col_values.values.ravel()
            ct = np.array(
                [getattr(pd.to_datetime(pd.Series(c)).dt, attr).values
                 for attr in self.attrs]).T
            cols = np.append(cols, ct, 1)

        col_names = [c for l in ll for c in l]
        return pd.DataFrame(data=cols, columns=col_names).astype(int)
