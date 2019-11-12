from __future__ import absolute_import
import os
from glob import glob
from os.path import basename, splitext
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='sagemaker_xgboost_container',
    version='1.0',
    description='Open source library for creating XGBoost containers to run on Amazon SageMaker.',

    packages=find_packages(where='src', exclude=('test',)),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],

    long_description=read('README.rst'),
    author='Amazon Web Services',
    license='Apache License 2.0',

    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        'Programming Language :: Python :: 3.5',
    ],

    install_requires=read("requirements.txt"),

    extras_require={
        'test': read("test-requirements.txt")
    },

    entry_points={
        'console_scripts': 'serve=sagemaker_xgboost_container.serving:main'
    },

    python_requires='>=3.5',
)
