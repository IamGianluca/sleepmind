from setuptools import setup, find_packages
from codecs import open
from os import path


# get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="sleepmind",
    version="1.0.0",
    description="Collection of utilities for fast Machine Learning experimentation",
    long_description=long_description,
    install_requires=[
        'cython>=0.29',
        'scipy>=1.1.0',
        'numpy>=1.15.4',
        'pandas>=0.23.4',
        'scikit-learn>=0.20.0',
        'xgboost>=0.81',
        'gensim>=3.6.0',
    ],
    packages=find_packages(exclude=["tests"]),
)
