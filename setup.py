from setuptools import setup, find_packages
from codecs import open
from os import path


# get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='sleepmind',
    version='0.0.1',
    description='Collection of utilities for fast Machine Learning experimentation',
    long_description=long_description,
    packages=find_packages(
        exclude=['digit-recognizer', 'home-credit-default-risk', 'tests']),
)
