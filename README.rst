fuku-ml
=========

.. image:: https://travis-ci.org/fukuball/fuku-ml.svg?branch=master
    :target: https://travis-ci.org/fukuball/fuku-ml

.. image:: https://codecov.io/github/fukuball/fuku-ml/coverage.svg?branch=master
    :target: https://codecov.io/github/fukuball/fuku-ml?branch=master

.. image:: https://img.shields.io/badge/made%20with-%e2%9d%a4-ff69b4.svg
    :target: http://www.fukuball.com

Simple machine learning library

Installation
============

.. code-block:: bash

    $ python setup.py install

Perceptron Learning Algorithm Usage
============

.. code-block:: py

    >>> import numpy as np
    # we need numpy as a base libray

    >>> import FukuML.PLA as pla
    # import FukuML.PLA to do Perceptron Learning

    >>> your_input_data_file = '/path/to/your/data/file'
    # assign your input data file, please check the data format is the same as this example_.
    .. _example: https://github.com/fukuball/fuku-ml/blob/master/FukuML/dataset/pla_train.dat

    >>> pla.load_train_data(your_input_data_file)
    # load train data

    >>> pla.init_W()
    # init the W

    >>> W = pla.train()
    # train by Perceptron Learning Algorithm to find best W

    >>> test_data_x = np.array([1, 0.15654, 0.75584, 0.01122, 0.42598])
    # assign new test data x

    >>> test_data_y = -1.0
    # assign new test data y

    >>> prediction = np.sign(np.dot(test_data_x, W))
    # prediction by trained W

    >>> print prediction
    # print the prediction, will find prediction is the same as test_data_y

Note
=========

Output the requirements

.. code-block:: bash

    $ pip freeze > requirements.txt


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