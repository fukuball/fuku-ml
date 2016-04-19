#!/usr/bin/env python

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import FukuML

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

long_description = open('README.rst').read()

license = open('LICENSE').read()

requirements_lines = [line.strip() for line in open('requirements.txt').readlines()]
install_requires = list(filter(None, requirements_lines))

setup(
    name='FukuML',
    version=FukuML.__version__,
    description='Simple machine learning library',
    long_description=long_description,
    keywords='Machine Learning, Perceptron Learning Algorithm, PLA, Pocket Perceptron Learning Algorithm, Pocket PLA, Linear Regression, Logistic Regression, Ridge Regression, Binay Classifier, Multi Classifier',
    author='Fukuball Lin',
    author_email='fukuball@gmail.com',
    url='https://github.com/fukuball/fuku-ml',
    license=license,
    install_requires=install_requires,
    packages=['FukuML'],
    package_dir={'FukuML': 'FukuML'},
    package_data={'FukuML': ['*.*', 'dataset/*.*', 'dataset/digits/*.*', 'dataset/digits/traning_digits/*.*', 'dataset/digits/test_digits/*.*']},
    test_suite='test_fuku_ml',
    zip_safe=False,
    classifiers=(
        'Development Status :: 5 - Production/Stable',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Machine Learning :: Perceptron Learning Algorithm',
        'Machine Learning :: Pocket Perceptron Learning Algorithm',
        'Machine Learning :: Linear Regression Learning Algorithm',
        'Machine Learning :: Logistic Regression Learning Algorithm',
        'Machine Learning :: Ridge Regression Learning Algorithm',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Education',
        'Topic :: Software Development :: Machine Learning',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Machine Learning',
        'Topic :: Utilities',
    ),
)
