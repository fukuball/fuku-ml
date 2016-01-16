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
    keywords='Machine Learning, Perceptron Learning Algorithm, PLA',
    author='Fukuball Lin',
    author_email='fukuball@gmail.com',
    url='https://github.com/fukuball/fuku-ml',
    license=license,
    install_requires=install_requires,
    packages=['FukuML'],
    package_dir={'FukuML': 'FukuML'},
    package_data={'FukuML': ['*.*', 'PLA/*', 'dataset/*']},
    test_suite='test_pla',
    zip_safe=False,
    classifiers=(
        'Development Status :: 5 - Production/Stable',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Natural Language :: Chinese (Traditional)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Education',
        'Topic :: Software Development :: Machine Learning',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Machine Learning',
        'Topic :: Utilities',
    ),
)
