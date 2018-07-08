from setuptools import setup, find_packages
from codecs import open
from os import path


# get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="sleepmind",
    version="0.0.2",
    description="Collection of utilities for fast Machine Learning experimentation",
    long_description=long_description,
    install_requires=[
        'cython>=0.28.3',
        'scipy>=1.1.0',
        'scikit-learn>=0.19.0',
        'numpy>=1.14.5',
        'pandas>=0.23.1',
        'xgboost>=0.72',
        'gensim>=3.4.0',
    ],
    dependency_links=[
        'git+ssh://git@github.com/scikit-learn/scikit-learn.git@5a9ce9fe3fe6cdf2574f0142e3f38698155f707a#egg=scikit-learn'
    ],
    packages=find_packages(exclude=["tests"]),
)
