`FukuML`_
=========
.. _FukuML: http://www.fukuball.com/fuku-ml/

.. image:: https://travis-ci.org/fukuball/fuku-ml.svg?branch=master
    :target: https://travis-ci.org/fukuball/fuku-ml

.. image:: https://codecov.io/github/fukuball/fuku-ml/coverage.svg?branch=master
    :target: https://codecov.io/github/fukuball/fuku-ml?branch=master

.. image:: https://badge.fury.io/py/FukuML.svg
    :target: https://badge.fury.io/py/FukuML

.. image:: https://api.codacy.com/project/badge/grade/afc87eff27ab47d6b960ea7b3088c469
    :target: https://www.codacy.com/app/fukuball/fuku-ml

.. image:: https://img.shields.io/badge/made%20with-%e2%9d%a4-ff69b4.svg
    :target: http://www.fukuball.com

Simple machine learning library / 簡單易用的機器學習套件

Installation
============

.. code-block:: bash

    $ pip install FukuML

Tutorial
============

- Lesson 1: `Perceptron Binary Classification Learning Algorithm`_

- Appendix 1: `Play With Your Own Dataset`_

- Appendix 2: `iNDIEVOX Open Data/API 智慧音樂應用：An Introduce to iNDIEVOX Open Data/API and the intelligent music application`_

.. _Perceptron Binary Classification Learning Algorithm: https://github.com/fukuball/FukuML-Tutorial/blob/master/Perceptron%20Binary%20Classification%20Learning%20Algorithm%20Tutorial.ipynb

.. _Play With Your Own Dataset: https://github.com/fukuball/FukuML-Tutorial/blob/master/Play%20With%20Your%20Own%20Dataset%20Tutorial.ipynb

.. _iNDIEVOX Open Data/API 智慧音樂應用：An Introduce to iNDIEVOX Open Data/API and the intelligent music application: https://speakerdeck.com/fukuball/api-and-the-intelligent-music-application

Algorithm
============

- Perceptron
    - Perceptron Binary Classification Learning Algorithm
    - Perceptron Multi Classification Learning Algorithm
    - Pocket Perceptron Binary Classification Learning Algorithm
    - Pocket Perceptron Multi Classification Learning Algorithm
- Regression
    - Linear Regression Learning Algorithm
    - Linear Regression Binary Classification Learning Algorithm
    - Linear Regression Multi Classification Learning Algorithm
    - Ridge Regression Learning Algorithm
    - Ridge Regression Binary Classification Learning Algorithm
    - Ridge Regression Multi Classification Learning Algorithm
    - Kernel Ridge Regression Learning Algorithm
    - Kernel Ridge Regression Binary Classification Learning Algorithm
    - Kernel Ridge Regression Multi Classification Learning Algorithm
- Logistic Regression
    - Logistic Regression Learning Algorithm
    - Logistic Regression Binary Classification Learning Algorithm
    - Logistic Regression One vs All Multi Classification Learning Algorithm
    - Logistic Regression One vs One Multi Classification Learning Algorithm
    - L2 Regularized Logistic Regression Learning Algorithm
    - L2 Regularized Logistic Regression Binary Classification Learning Algorithm
    - Kernel Logistic Regression Learning Algorithm
- Support Vector Machine
    - Primal Hard Margin Support Vector Machine Binary Classification Learning Algorithm
    - Dual Hard Margin Support Vector Machine Binary Classification Learning Algorithm
    - Polynomial Kernel Support Vector Machine Binary Classification Learning Algorithm
    - Gaussian Kernel Support Vector Machine Binary Classification Learning Algorithm
    - Soft Polynomial Kernel Support Vector Machine Binary Classification Learning Algorithm
    - Soft Gaussian Kernel Support Vector Machine Binary Classification Learning Algorithm
    - Polynomial Kernel Support Vector Machine Multi Classification Learning Algorithm
    - Gaussian Kernel Support Vector Machine Multi Classification Learning Algorithm
    - Soft Polynomial Kernel Support Vector Machine Multi Classification Learning Algorithm
    - Soft Gaussian Kernel Support Vector Machine Multi Classification Learning Algorithm
    - Probabilistic Support Vector Machine Learning Algorithm
    - Least Squares Support Vector Machine Binary Classification Learning Algorithm
    - Least Squares Support Vector Machine Multi Classification Learning Algorithm
    - Support Vector Regression Learning Algorithm
- Decision Tree
    - Decision Stump Binary Classification Learning Algorithm
    - AdaBoost Stump Binary Classification Learning Algorithm
    - AdaBoost Decision Tree Classification Learning Algorithm
    - Gradient Boost Decision Tree Regression Learning Algorithm
    - Decision Tree Classification Learning Algorithm
    - Decision Tree Regression Learning Algorithm
    - Random Forest Classification Learning Algorithm
    - Random Forest Regression Learning Algorithm
- Neural Network
    - Neural Network Learning Algorithm
    - Neural Network Binary Classification Learning Algorithm
- Accelerator
    - Linear Regression Accelerator
- Feature Transform
    - Polynomial Feature Transform
    - Legendre Feature Transform
- Validation
    - 10 Fold Cross Validation
- Blending
    - Uniform Blending for Classification
    - Linear Blending for Classification
    - Uniform Blending for Regression
    - Linear Blending for Regression

Usage
============

.. code-block:: py

    >>> import numpy as np
    # we need numpy as a base libray

    >>> import FukuML.PLA as pla
    # import FukuML.PLA to do Perceptron Learning

    >>> your_input_data_file = '/path/to/your/data/file'
    # assign your input data file, please check the data format: https://github.com/fukuball/fuku-ml/blob/master/FukuML/dataset/pla_binary_train.dat

    >>> pla_bc = pla.BinaryClassifier()
    # new a PLA binary classifier

    >>> pla_bc.load_train_data(your_input_data_file)
    # load train data

    >>> pla_bc.set_param()
    # set parameter

    >>> pla_bc.init_W()
    # init the W

    >>> W = pla_bc.train()
    # train by Perceptron Learning Algorithm to find best W

    >>> test_data = 'Each feature of data x separated with spaces. And the ground truth y put in the end of line separated by a space'
    # assign test data, format like this '0.97681 0.10723 0.64385 ........ 0.29556 1'

    >>> prediction = pla_bc.prediction(test_data)
    # prediction by trained W

    >>> print prediction['input_data_x']
    # print test data x

    >>> print prediction['input_data_y']
    # print test data y

    >>> print prediction['prediction']
    # print the prediction, will find out prediction is the same as pla_bc.test_data_y

For detail, please check https://github.com/fukuball/fuku-ml/blob/master/doc/sample_code.rst

Tests
=========

.. code-block:: shell

   python test_fuku_ml.py

PEP8
=========

.. code-block:: shell

   pep8 FukuML/*.py --ignore=E501

Donate
=========

If you find fuku-ml useful, please consider a donation. Thank you!

- bitcoin: 1BbihQU3CzSdyLSP9bvQq7Pi1z1jTdAaq9
- eth: 0x92DA3F837bf2F79D422bb8CEAC632208F94cdE33


License
=========
The MIT License (MIT)

Copyright (c) 2016 fukuball

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.