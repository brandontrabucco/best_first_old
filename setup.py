"""Author: Brandon Trabucco, Copyright 2019
Implements the Best First Module for image captioning."""


from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['numpy', 'tensorflow']


setup(
    name='best_first',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('best_first')],
    description='Berkeley-CMU Best First Captioning Mechanism',
)