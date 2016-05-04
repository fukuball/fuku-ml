FukuML
=========

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

Simple machine learning library

Installation
============

.. code-block:: bash

    $ pip install FukuML

Algorithm
============

- Perceptron Binary Classification Learning Algorithm

- Perceptron Binary Classification Learning Algorithm with Linear Regression Accelerator

- Perceptron Multi Classification Learning Algorithm

- Perceptron Multi Classification Learning Algorithm with Linear Regression Accelerator

- Pocket Perceptron Binary Classification Learning Algorithm

- Pocket Perceptron Binary Classification Learning Algorithm with Linear Regression Accelerator

- Pocket Perceptron Multi Classification Learning Algorithm

- Pocket Perceptron Multi Classification Learning Algorithm with Linear Regression Accelerator

- Linear Regression Learning Algorithm

- Linear Regression Binary Classification Learning Algorithm

- Linear Regression Multi Classification Learning Algorithm

- Logistic Regression Learning Algorithm

- Logistic Regression Learning Algorithm with Linear Regression Accelerator

- Logistic Regression Binary Classification Learning Algorithm

- Logistic Regression Binary Classification Learning Algorithm with Linear Regression Accelerator

- Logistic Regression One vs All Multi Classification Learning Algorithm

- Logistic Regression One vs All Multi Classification Learning Algorithm with Linear Regression Accelerator

- Logistic Regression One vs One Multi Classification Learning Algorithm

- Logistic Regression One vs One Multi Classification Learning Algorithm with Linear Regression Accelerator

- Ridge Regression Learning Algorithm

- Ridge Regression Binary Classification Learning Algorithm

- Ridge Regression Multi Classification Learning Algorithm

- Polynomial Feature Transform

- Legendre Feature Transform

- 10 Fold Cross Validation

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