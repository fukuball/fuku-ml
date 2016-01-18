FukuML
=========

.. image:: https://travis-ci.org/fukuball/fuku-ml.svg?branch=master
    :target: https://travis-ci.org/fukuball/fuku-ml

.. image:: https://codecov.io/github/fukuball/fuku-ml/coverage.svg?branch=master
    :target: https://codecov.io/github/fukuball/fuku-ml?branch=master

.. image:: https://badge.fury.io/py/FukuML.svg
    :target: https://badge.fury.io/py/FukuML

.. image:: https://img.shields.io/badge/made%20with-%e2%9d%a4-ff69b4.svg
    :target: http://www.fukuball.com

Simple machine learning library

Installation
============

Option 1
-----------------

.. code-block:: bash

    $ pip install FukuML

Option 2
-----------------

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
    # assign your input data file, please check the data format: https://github.com/fukuball/fuku-ml/blob/master/FukuML/dataset/pla_binary_train.dat

    >>> pla_bc = pla.BinaryClassifier()
    # new a PLA binary classifier

    >>> pla_bc.load_train_data(your_input_data_file)
    # load train data

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

Run demo dataset: Basic Naive Cycle PLA
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.PLA as pla

    >>> pla_bc = pla.BinaryClassifier()

    >>> pla_bc.load_train_data()

    >>> pla_bc.init_W()

    >>> pla_bc.train()

    >>> test_data = '0.97681 0.10723 0.64385 0.29556 1'

    >>> prediction = pla_bc.prediction(test_data)

    >>> future_data = '0.97681 0.10723 0.64385 0.29556'

    >>> prediction = pla_bc.prediction(future_data, 'future_data')

Run demo dataset: Random Cycle PLA
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.PLA as pla

    >>> pla_bc = pla.BinaryClassifier()

    >>> pla_bc.load_train_data()

    >>> pla_bc.init_W()

    >>> pla_bc.train('random')

    >>> test_data = '0.97681 0.10723 0.64385 0.29556 1'

    >>> prediction = pla_bc.prediction(test_data)

    >>> future_data = '0.97681 0.10723 0.64385 0.29556'

    >>> prediction = pla_bc.prediction(future_data, 'future_data')

Run demo dataset: Random Cycle PLA alpha=0.5 step correction
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.PLA as pla

    >>> pla_bc = pla.BinaryClassifier()

    >>> pla_bc.load_train_data()

    >>> pla_bc.init_W()

    >>> pla_bc.train('random', 0.5)

    >>> test_data = '0.97681 0.10723 0.64385 0.29556 1'

    >>> prediction = pla_bc.prediction(test_data)

    >>> future_data = '0.97681 0.10723 0.64385 0.29556'

    >>> prediction = pla_bc.prediction(future_data, 'future_data')

Pocket Perceptron Learning Algorithm Usage
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.PocketPLA as pocket

    >>> pocket_bc = pocket.BinaryClassifier()

    >>> pocket_bc.load_train_data()

    >>> pocket_bc.init_W()

    >>> W = pocket_bc.train(50)

    >>> test_data = '0.62771 0.11513 0.82235 0.14493 -1'

    >>> prediction = pocket_bc.prediction(test_data)

    >>> future_data = '0.62771 0.11513 0.82235 0.14493'

    >>> prediction = pocket_bc.prediction(future_data, 'future_data')

    >>> pocket_bc.load_test_data()

    >>> pocket_bc.calculate_avg_error(pocket_bc.test_X, pocket_bc.test_Y, W)

Note
=========

Output the requirements

.. code-block:: bash

    $ pip freeze > requirements.txt

Run tests

.. code-block:: bash

    $ python test_fuku_ml.py

Package

.. code-block:: bash

    $ python setup.py sdist
    $ python setup.py bdist_wheel --universal
    $ twine upload dist/*

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