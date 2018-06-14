from datetime import date

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from sleepmind.feature_extraction.datetime import DateParser


@pytest.mark.parametrize('X,expected', [
    (pd.DataFrame({'date': [date(2018, 8, 2), date(2014, 4, 16)]}),
     pd.DataFrame({
         'date_year': [2018, 2014],
         'date_month': [8, 4],
         'date_week': [31, 16],
         'date_day': [2, 16],
         'date_dayofweek': [3, 2],
         'date_dayofyear': [214, 106],
         'date_days_in_month': [31, 30],
         'date_quarter': [3, 2],
         'date_is_month_start': [0, 0],
         'date_is_month_end': [0, 0],
         'date_is_quarter_start': [0, 0],
         'date_is_quarter_end': [0, 0],
         'date_is_year_start': [0, 0],
         'date_is_year_end': [0, 0],
         'date_is_leap_year': [0, 0]})),
    # multiple columns to transform
    (pd.DataFrame({
        'date': [date(2018, 8, 2), date(2014, 4, 16)],
        'another_date': [date(2018, 8, 2), date(2014, 4, 16)]}),
     pd.DataFrame({
         'date_year': [2018, 2014],
         'date_month': [8, 4],
         'date_week': [31, 16],
         'date_day': [2, 16],
         'date_dayofweek': [3, 2],
         'date_dayofyear': [214, 106],
         'date_days_in_month': [31, 30],
         'date_quarter': [3, 2],
         'date_is_month_start': [0, 0],
         'date_is_month_end': [0, 0],
         'date_is_quarter_start': [0, 0],
         'date_is_quarter_end': [0, 0],
         'date_is_year_start': [0, 0],
         'date_is_year_end': [0, 0],
         'date_is_leap_year': [0, 0],
         'another_date_year': [2018, 2014],
         'another_date_month': [8, 4],
         'another_date_week': [31, 16],
         'another_date_day': [2, 16],
         'another_date_dayofweek': [3, 2],
         'another_date_dayofyear': [214, 106],
         'another_date_days_in_month': [31, 30],
         'another_date_quarter': [3, 2],
         'another_date_is_month_start': [0, 0],
         'another_date_is_month_end': [0, 0],
         'another_date_is_quarter_start': [0, 0],
         'another_date_is_quarter_end': [0, 0],
         'another_date_is_year_start': [0, 0],
         'another_date_is_year_end': [0, 0],
         'another_date_is_leap_year': [0, 0]})),
])
def test_date_parser(X, expected):
    result = DateParser().fit_transform(X)
    assert_array_equal(result, expected)
    assert_frame_equal(result, expected)
