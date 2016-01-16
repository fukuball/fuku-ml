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

Basic usage to train from you data (Basic Naive Cycle PLA)
-----------------

.. code-block:: py

    >>> import numpy as np
    # we need numpy as a base libray

    >>> import FukuML.PLA as pla
    # import FukuML.PLA to do Perceptron Learning

    >>> your_input_data_file = '/path/to/your/data/file'
    # assign your input data file, please check the data format: https://github.com/fukuball/fuku-ml/blob/master/FukuML/dataset/pla_train.dat

    >>> pla.load_train_data(your_input_data_file)
    # load train data

    >>> pla.init_W()
    # init the W

    >>> W = pla.train()
    # train by Perceptron Learning Algorithm to find best W

    >>> test_data = 'Each feature of data x separated with spaces. And the ground truth y put in the end of line separated by a space'
    # assign test data, format like this '0.97681 0.10723 0.64385 ........ 0.29556 1'

    >>> prediction = pla.prediction(test_data)
    # prediction by trained W

    >>> print pla.test_data_x
    # print test data x

    >>> print pla.test_data_y
    # print test data y

    >>> print prediction
    # print the prediction, will find out prediction is the same as pla.test_data_y

Run demo dataset: Basic Naive Cycle PLA
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.PLA as pla

    >>> pla.load_train_data()

    >>> pla.init_W()

    >>> pla.train()

    >>> test_data = '0.97681 0.10723 0.64385 0.29556 1'

    >>> prediction = pla.prediction(test_data)

Run demo dataset: Random Cycle PLA
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.PLA as pla

    >>> pla.load_train_data()

    >>> pla.init_W()

    >>> pla.train('random')

    >>> test_data = '0.97681 0.10723 0.64385 0.29556 1'

    >>> prediction = pla.prediction(test_data)

Run demo dataset: Random Cycle PLA alpha=0.5 step correction
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.PLA as pla

    >>> pla.load_train_data()

    >>> pla.init_W()

    >>> pla.train('random', 0.5)

    >>> test_data = '0.97681 0.10723 0.64385 0.29556 1'

    >>> prediction = pla.prediction(test_data)

Pocket Perceptron Learning Algorithm Usage
============

Run demo dataset: Basic Naive Cycle PLA
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.PocketPLA as pocket

    >>> pocket.load_train_data()

    >>> pocket.init_W()

    >>> W = pocket.train(50)

    >>> test_data = '0.62771 0.11513 0.82235 0.14493 -1'

    >>> prediction = pocket.prediction(test_data)

    >>> pocket.load_test_data()

    >>> pocket.calculate_avg_error(pocket.test_X, pocket.test_Y, W)

Note
=========

Output the requirements

.. code-block:: bash

    $ pip freeze > requirements.txt

Run tests

.. code-block:: bash

    $ python test_pla.py

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