Perceptron Binary Classification Learning Algorithm
============

Run demo dataset: Basic Naive Cycle PLA
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.PLA as pla

    >>> pla_bc = pla.BinaryClassifier()

    >>> pla_bc.load_train_data()

    >>> pla_bc.set_param()

    >>> pla_bc.init_W()

    >>> pla_bc.train()

    >>> test_data = '0.97681 0.10723 0.64385 0.29556 1'

    >>> prediction = pla_bc.prediction(test_data)

    >>> future_data = '0.97681 0.10723 0.64385 0.29556'

    >>> prediction = pla_bc.prediction(future_data, 'future_data')

Run demo dataset: Basic Naive Cycle PLA with Linear Regression Accelerator
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.PLA as pla

    >>> pla_bc = pla.BinaryClassifier()

    >>> pla_bc.load_train_data()

    >>> pla_bc.set_param()

    >>> pla_bc.init_W('linear_regression_accelerator')

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

    >>> pla_bc.set_param(loop_mode='random')

    >>> pla_bc.init_W()

    >>> pla_bc.train()

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

    >>> pla_bc.set_param(loop_mode='random', step_alpha=0.5)

    >>> pla_bc.init_W()

    >>> pla_bc.train()

    >>> test_data = '0.97681 0.10723 0.64385 0.29556 1'

    >>> prediction = pla_bc.prediction(test_data)

    >>> future_data = '0.97681 0.10723 0.64385 0.29556'

    >>> prediction = pla_bc.prediction(future_data, 'future_data')

Perceptron Multi Classification Learning Algorithm
============

Run demo dataset: One vs One
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.PLA as pla

    >>> pla_mc = pla.MultiClassifier()

    >>> pla_mc.load_train_data()

    >>> pla_mc.set_param()

    >>> pla_mc.init_W()

    >>> W = pla_mc.train()

    >>> test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'

    >>> prediction = pla_mc.prediction(test_data)

Run demo dataset: One vs One with Linear Regression Accelerator
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.PLA as pla

    >>> pla_mc = pla.MultiClassifier()

    >>> pla_mc.load_train_data()

    >>> pla_mc.set_param()

    >>> pla_mc.init_W('linear_regression_accelerator')

    >>> W = pla_mc.train()

    >>> test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'

    >>> prediction = pla_mc.prediction(test_data)

Pocket Perceptron Binary Classification Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.PocketPLA as pocket

    >>> pocket_bc = pocket.BinaryClassifier()

    >>> pocket_bc.load_train_data()

    >>> pocket_bc.set_param(loop_mode='naive_cycle', step_alpha=1, updates=50)

    >>> pocket_bc.init_W()

    >>> W = pocket_bc.train()

    >>> test_data = '0.62771 0.11513 0.82235 0.14493 -1'

    >>> prediction = pocket_bc.prediction(test_data)

    >>> future_data = '0.62771 0.11513 0.82235 0.14493'

    >>> prediction = pocket_bc.prediction(future_data, 'future_data')

    >>> pocket_bc.load_test_data()

    >>> pocket_bc.calculate_avg_error(pocket_bc.test_X, pocket_bc.test_Y, W)

Run demo dataset with Linear Regression Accelerator
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.PocketPLA as pocket

    >>> pocket_bc = pocket.BinaryClassifier()

    >>> pocket_bc.load_train_data()

    >>> pocket_bc.set_param(loop_mode='naive_cycle', step_alpha=1, updates=50)

    >>> pocket_bc.init_W('linear_regression_accelerator')

    >>> W = pocket_bc.train()

    >>> test_data = '0.62771 0.11513 0.82235 0.14493 -1'

    >>> prediction = pocket_bc.prediction(test_data)

    >>> future_data = '0.62771 0.11513 0.82235 0.14493'

    >>> prediction = pocket_bc.prediction(future_data, 'future_data')

    >>> pocket_bc.load_test_data()

    >>> pocket_bc.calculate_avg_error(pocket_bc.test_X, pocket_bc.test_Y, W)

Pocket Perceptron Multi Classification Learning Algorithm
============

Run demo dataset: One vs One
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.PocketPLA as pocket

    >>> pocket_mc = pocket.MultiClassifier()

    >>> pocket_mc.load_train_data()

    >>> pocket_mc.set_param(loop_mode='naive_cycle', step_alpha=1, updates=50)

    >>> pocket_mc.init_W()

    >>> W = pocket_mc.train()

    >>> test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'

    >>> prediction = pocket_mc.prediction(test_data)

Run demo dataset: One vs One with Linear Regression Accelerator
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.PocketPLA as pocket

    >>> pocket_mc = pocket.MultiClassifier()

    >>> pocket_mc.load_train_data()

    >>> pocket_mc.set_param(loop_mode='naive_cycle', step_alpha=1, updates=50)

    >>> pocket_mc.init_W('linear_regression_accelerator')

    >>> W = pocket_mc.train()

    >>> test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'

    >>> prediction = pocket_mc.prediction(test_data)

Linear Regression Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.LinearRegression as linear_regression

    >>> linear = linear_regression.LinearRegression()

    >>> linear.load_train_data()

    >>> linear.init_W()

    >>> W = linear.train()

    >>> test_data = '0.62771 0.11513 0.82235 0.14493 -1'

    >>> prediction = linear.prediction(test_data)

    >>> future_data = '0.62771 0.11513 0.82235 0.14493'

    >>> prediction = linear.prediction(future_data, 'future_data')

    >>> linear.load_test_data()

    >>> linear.calculate_avg_error(linear.test_X, linear.test_Y, W)

Linear Regression Binary Classification Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.LinearRegression as linear_regression

    >>> linear_bc = linear_regression.BinaryClassifier()

    >>> linear_bc.load_train_data()

    >>> linear_bc.init_W()

    >>> W = linear_bc.train()

    >>> test_data = '0.62771 0.11513 0.82235 0.14493 -1'

    >>> prediction = linear_bc.prediction(test_data)

    >>> future_data = '0.62771 0.11513 0.82235 0.14493'

    >>> prediction = linear_bc.prediction(future_data, 'future_data')

    >>> linear_bc.load_test_data()

    >>> linear_bc.calculate_avg_error(linear_bc.test_X, linear_bc.test_Y, W)


Linear Regression Multi Classification Learning Algorithm
============

Run demo dataset: One vs One
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.LinearRegression as linear_regression

    >>> linear_mc = linear_regression.MultiClassifier()

    >>> linear_mc.load_train_data()

    >>> linear_mc.init_W()

    >>> W = linear_mc.train()

    >>> test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'

    >>> prediction = linear_mc.prediction(test_data)

Logistic Regression Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.LogisticRegression as logistic_regression

    >>> logistic = logistic_regression.LogisticRegression()

    >>> logistic.load_train_data()

    >>> logistic.set_param()

    >>> logistic.init_W()

    >>> W = logistic.train()

    >>> test_data = '0.26502 0.5486 0.971 0.19333 0.12207 0.81528 0.46743 0.45889 0.31004 0.3307 0.43078 0.50661 0.57281 0.052715 0.50443 0.78686 0.20099 0.85909 0.26772 0.13751 1'

    >>> prediction = logistic.prediction(test_data)

Run demo dataset with Linear Regression Accelerator
-----------------

    .. code-block:: py

    >>> import numpy as np

    >>> import FukuML.LogisticRegression as logistic_regression

    >>> logistic = logistic_regression.LinearRegression()

    >>> logistic.load_train_data()

    >>> logistic.set_param()

    >>> logistic.init_W('linear_regression_accelerator')

    >>> W = logistic.train()

    >>> test_data = '0.26502 0.5486 0.971 0.19333 0.12207 0.81528 0.46743 0.45889 0.31004 0.3307 0.43078 0.50661 0.57281 0.052715 0.50443 0.78686 0.20099 0.85909 0.26772 0.13751 1'

    >>> prediction = logistic.prediction(test_data)

Logistic Regression Binary Classification Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.LogisticRegression as logistic_regression

    >>> logistic = logistic_regression.BinaryClassifier()

    >>> logistic.load_train_data()

    >>> logistic.set_param()

    >>> logistic.init_W()

    >>> W = logistic.train()

    >>> test_data = '0.26502 0.5486 0.971 0.19333 0.12207 0.81528 0.46743 0.45889 0.31004 0.3307 0.43078 0.50661 0.57281 0.052715 0.50443 0.78686 0.20099 0.85909 0.26772 0.13751 1'

    >>> prediction = logistic.prediction(test_data)

Run demo dataset with Linear Regression Accelerator
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.LogisticRegression as logistic_regression

    >>> logistic = logistic_regression.BinaryClassifier()

    >>> logistic.load_train_data()

    >>> logistic.set_param()

    >>> logistic.init_W('linear_regression_accelerator')

    >>> W = logistic.train()

    >>> test_data = '0.26502 0.5486 0.971 0.19333 0.12207 0.81528 0.46743 0.45889 0.31004 0.3307 0.43078 0.50661 0.57281 0.052715 0.50443 0.78686 0.20099 0.85909 0.26772 0.13751 1'

    >>> prediction = logistic.prediction(test_data)

Logistic Regression Multi Classification Learning Algorithm
============

Run demo dataset: One vs All
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.LogisticRegression as logistic_regression

    >>> logistic = logistic_regression.MultiClassifier()

    >>> logistic.load_train_data()

    >>> logistic.set_param()

    >>> logistic.init_W()

    >>> W = logistic.train()

    >>> test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'

    >>> prediction = logistic.prediction(test_data)

Run demo dataset: One vs All with Linear Regression Accelerator
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.LogisticRegression as logistic_regression

    >>> logistic = logistic_regression.MultiClassifier()

    >>> logistic.load_train_data()

    >>> logistic.set_param()

    >>> logistic.init_W('linear_regression_accelerator')

    >>> W = logistic.train()

    >>> test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'

    >>> prediction = logistic.prediction(test_data)

Run demo dataset: One vs One
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.LogisticRegression as logistic_regression

    >>> logistic = logistic_regression.MultiClassifier()

    >>> logistic.load_train_data()

    >>> logistic.set_param()

    >>> logistic.init_W('normal', 'ovo')

    >>> W = logistic.train()

    >>> test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'

    >>> prediction = logistic.prediction(test_data)

Run demo dataset: One vs One with Linear Regression Accelerator
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.LogisticRegression as logistic_regression

    >>> logistic = logistic_regression.MultiClassifier()

    >>> logistic.load_train_data()

    >>> logistic.set_param()

    >>> logistic.init_W('linear_regression_accelerator', 'ovo')

    >>> W = logistic.train()

    >>> test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'

    >>> prediction = logistic.prediction(test_data)

L2 Regularized Logistic Regression Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.L2RLogisticRegression as l2r_logistic_regression

    >>> l2r_logistic = l2r_logistic_regression.L2RLogisticRegression()

    >>> l2r_logistic.load_train_data()

    >>> l2r_logistic.set_param()

    >>> l2r_logistic.init_W()

    >>> W = l2r_logistic.train()

    >>> test_data = '0.26502 0.5486 0.971 0.19333 0.12207 0.81528 0.46743 0.45889 0.31004 0.3307 0.43078 0.50661 0.57281 0.052715 0.50443 0.78686 0.20099 0.85909 0.26772 0.13751 1'

    >>> prediction = l2r_logistic.prediction(test_data)

L2 Regularized Logistic Regression Binary Classification Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.L2RLogisticRegression as l2r_logistic_regression

    >>> l2r_logistic = l2r_logistic_regression.BinaryClassifier()

    >>> l2r_logistic.load_train_data()

    >>> l2r_logistic.set_param()

    >>> l2r_logistic.init_W()

    >>> W = l2r_logistic.train()

    >>> test_data = '0.26502 0.5486 0.971 0.19333 0.12207 0.81528 0.46743 0.45889 0.31004 0.3307 0.43078 0.50661 0.57281 0.052715 0.50443 0.78686 0.20099 0.85909 0.26772 0.13751 1'

    >>> prediction = l2r_logistic.prediction(test_data)

Ridge Regression Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.RidgeRegression as ridge_regression

    >>> ridge = ridge_regression.RidgeRegression()

    >>> ridge.load_train_data()

    >>> ridge.set_param(lambda_p=pow(10, -3))

    >>> ridge.init_W()

    >>> W = ridge.train()

    >>> test_data = '0.62771 0.11513 0.82235 0.14493 -1'

    >>> prediction = ridge.prediction(test_data)

    >>> future_data = '0.62771 0.11513 0.82235 0.14493'

    >>> prediction = ridge.prediction(future_data, 'future_data')

    >>> ridge.load_test_data()

    >>> ridge.calculate_avg_error(ridge.test_X, ridge.test_Y, W)

Ridge Regression Binary Classification Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.RidgeRegression as ridge_regression

    >>> ridge_bc = ridge_regression.BinaryClassifier()

    >>> ridge_bc.load_train_data()

    >>> ridge_bc.set_param(lambda_p=pow(10, -3))

    >>> ridge_bc.init_W()

    >>> W = ridge_bc.train()

    >>> test_data = '0.402041 0.402048 -1'

    >>> prediction = ridge_bc.prediction(test_data)

    >>> future_data = '0.402041 0.402048'

    >>> prediction = ridge_bc.prediction(future_data, 'future_data')

    >>> ridge_bc.load_test_data()

    >>> ridge_bc.calculate_avg_error(ridge_bc.test_X, ridge_bc.test_Y, W)


Ridge Regression Multi Classification Learning Algorithm
============

Run demo dataset: One vs One
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.RidgeRegression as ridge_regression

    >>> ridge_mc = ridge_regression.MultiClassifier()

    >>> ridge_mc.load_train_data()

    >>> ridge_mc.set_param(lambda_p=pow(10, -3))

    >>> ridge_mc.init_W()

    >>> W = ridge_mc.train()

    >>> test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'

    >>> prediction = ridge_mc.prediction(test_data)

Primal Hard Margin Support Vector Machine Binary Classification Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.SupportVectorMachine as svm

    >>> svm_bc = svm.BinaryClassifier()

    >>> svm_bc.load_train_data()

    >>> svm_bc.set_param(svm_kernel='primal_hard_margin')

    >>> svm_bc.init_W()

    >>> W = svm_bc.train()

    >>> test_data = '0.97681 0.10723 0.64385 0.29556 1'

    >>> prediction = svm_bc.prediction(test_data)

    >>> test_data = '0.97681 0.10723 0.64385 0.29556'

    >>> prediction = svm_bc.prediction(future_data, 'future_data')

    >>> svm_bc.load_test_data()

    >>> svm_bc.calculate_avg_error(svm_bc.test_X, svm_bc.test_Y, W)

Dual Hard Margin Support Vector Machine Binary Classification Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.SupportVectorMachine as svm

    >>> svm_bc = svm.BinaryClassifier()

    >>> svm_bc.load_train_data()

    >>> svm_bc.set_param(svm_kernel='dual_hard_margin')

    >>> svm_bc.init_W()

    >>> W = svm_bc.train()

    >>> test_data = '0.97681 0.10723 0.64385 0.29556 1'

    >>> prediction = svm_bc.prediction(test_data)

    >>> test_data = '0.97681 0.10723 0.64385 0.29556'

    >>> prediction = svm_bc.prediction(future_data, 'future_data')

    >>> svm_bc.load_test_data()

    >>> svm_bc.calculate_avg_error(svm_bc.test_X, svm_bc.test_Y, W)

Polynomial Kernel Support Vector Machine Binary Classification Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.SupportVectorMachine as svm

    >>> svm_bc = svm.BinaryClassifier()

    >>> svm_bc.load_train_data()

    >>> svm_bc.set_param(svm_kernel='polynomial_kernel', zeta=100, gamma=1, Q=3)

    >>> svm_bc.init_W()

    >>> W = svm_bc.train()

    >>> test_data = '0.97681 0.10723 0.64385 0.29556 1'

    >>> prediction = svm_bc.prediction(test_data)

    >>> test_data = '0.97681 0.10723 0.64385 0.29556'

    >>> prediction = svm_bc.prediction(future_data, 'future_data')

    >>> svm_bc.load_test_data()

    >>> svm_bc.calculate_avg_error(svm_bc.test_X, svm_bc.test_Y, W)

Gaussian Kernel Support Vector Machine Binary Classification Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.SupportVectorMachine as svm

    >>> svm_bc = svm.BinaryClassifier()

    >>> svm_bc.load_train_data()

    >>> svm_bc.set_param(svm_kernel='gaussian_kernel', gamma=1)

    >>> svm_bc.init_W()

    >>> W = svm_bc.train()

    >>> test_data = '0.97681 0.10723 0.64385 0.29556 1'

    >>> prediction = svm_bc.prediction(test_data)

    >>> test_data = '0.97681 0.10723 0.64385 0.29556'

    >>> prediction = svm_bc.prediction(future_data, 'future_data')

    >>> svm_bc.load_test_data()

    >>> svm_bc.calculate_avg_error(svm_bc.test_X, svm_bc.test_Y, W)

Soft Polynomial Kernel Support Vector Machine Binary Classification Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.SupportVectorMachine as svm

    >>> svm_bc = svm.BinaryClassifier()

    >>> svm_bc.load_train_data()

    >>> svm_bc.set_param(svm_kernel='soft_polynomial_kernel', zeta=100, gamma=1, Q=3, C=0.1)

    >>> svm_bc.init_W()

    >>> W = svm_bc.train()

    >>> test_data = '0.97681 0.10723 0.64385 0.29556 1'

    >>> prediction = svm_bc.prediction(test_data)

    >>> test_data = '0.97681 0.10723 0.64385 0.29556'

    >>> prediction = svm_bc.prediction(future_data, 'future_data')

    >>> svm_bc.load_test_data()

    >>> svm_bc.calculate_avg_error(svm_bc.test_X, svm_bc.test_Y, W)

Soft Gaussian Kernel Support Vector Machine Binary Classification Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.SupportVectorMachine as svm

    >>> svm_bc = svm.BinaryClassifier()

    >>> svm_bc.load_train_data()

    >>> svm_bc.set_param(svm_kernel='soft_gaussian_kernel', gamma=1, C=0.1)

    >>> svm_bc.init_W()

    >>> W = svm_bc.train()

    >>> test_data = '0.97681 0.10723 0.64385 0.29556 1'

    >>> prediction = svm_bc.prediction(test_data)

    >>> test_data = '0.97681 0.10723 0.64385 0.29556'

    >>> prediction = svm_bc.prediction(future_data, 'future_data')

    >>> svm_bc.load_test_data()

    >>> svm_bc.calculate_avg_error(svm_bc.test_X, svm_bc.test_Y, W)

Probabilistic Support Vector Machine Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.ProbabilisticSVM as probabilistic_svm

    >>> probabilistic = probabilistic_svm.ProbabilisticSVM()

    >>> probabilistic.load_train_data()

    >>> probabilistic.set_param()

    >>> probabilistic.init_W()

    >>> probabilistic.train()

    >>> test_data = '0.26502 0.5486 0.971 0.19333 0.12207 0.81528 0.46743 0.45889 0.31004 0.3307 0.43078 0.50661 0.57281 0.052715 0.50443 0.78686 0.20099 0.85909 0.26772 0.13751 1'

    >>> prediction = probabilistic.prediction(test_data)

Decision Stump Binary Classification Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.DecisionStump as decision_stump

    >>> decision_stump = decision_stump.BinaryClassifier()

    >>> decision_stump.load_train_data()

    >>> decision_stump.set_param()

    >>> decision_stump.init_W()

    >>> decision_stump.train()

    >>> test_data = '-8.451 7.694 -1.887 1.017 3.708 7.244 9.748 -2.362 -3.618 1'

    >>> prediction = decision_stump.prediction(test_data)

AdaBoost Stump Binary Classification Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.AdaBoostStump as adaboost_stump

    >>> adaboost_stump_bc = adaboost_stump.BinaryClassifier()

    >>> adaboost_stump_bc.load_train_data()

    >>> adaboost_stump_bc.set_param(run_t=10)

    >>> adaboost_stump_bc.init_W()

    >>> adaboost_stump_bc.train()

    >>> test_data = '-9.706 1.392 6.562 -6.543 -1.980 -6.261 -6.067 1.254 -1.071 11'

    >>> prediction = adaboost_stump_bc.prediction(test_data)

Decision Tree Classification Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.DecisionTree as decision_tree

    >>> decision_tree_c = decision_tree.CART()

    >>> decision_tree_c.load_train_data()

    >>> decision_tree_c.set_param(learn_type='classifier')

    >>> decision_tree_c.init_W()

    >>> decision_tree_c.train()

    >>> test_data = '6.0 2.2 5.0 1.5 virginica'

    >>> prediction = decision_tree_c.prediction(test_data)

Decision Tree Regression Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.DecisionTree as decision_tree

    >>> input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/linear_regression_train.dat')

    >>> decision_tree_c = decision_tree.CART()

    >>> decision_tree_c.load_train_data(input_train_data_file)

    >>> decision_tree_c.set_param(learn_type='regression')

    >>> decision_tree_c.init_W()

    >>> decision_tree_c.train()

    >>> test_data = '55.7 43 285'

    >>> prediction = decision_tree_c.prediction(test_data)

Kernel Logistic Regression Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.KernelLogisticRegression as kernel_logistic_regression

    >>> kernel_logistic = kernel_logistic_regression.KernelLogisticRegression()

    >>> kernel_logistic.load_train_data()

    >>> kernel_logistic.set_param()

    >>> kernel_logistic.init_W()

    >>> W = kernel_logistic.train()

    >>> test_data = '0.26502 0.5486 0.971 0.19333 0.12207 0.81528 0.46743 0.45889 0.31004 0.3307 0.43078 0.50661 0.57281 0.052715 0.50443 0.78686 0.20099 0.85909 0.26772 0.13751 1'

    >>> prediction = kernel_logistic.prediction(test_data)

Kernel Ridge Regression Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.KernelRidgeRegression as kernel_ridge_regression

    >>> kernel_ridge = kernel_ridge_regression.KernelRidgeRegression()

    >>> kernel_ridge.load_train_data()

    >>> kernel_ridge.set_param(lambda_p=pow(10, -3), gamma=1, C=0.1)

    >>> kernel_ridge.init_W()

    >>> kernel_ridge.train()

    >>> test_data = '0.62771 0.11513 0.82235 0.14493 -1'

    >>> prediction = kernel_ridge.prediction(test_data)

Kernel Ridge Regression Binary Classification Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.KernelRidgeRegression as kernel_ridge_regression

    >>> kernel_ridge_bc = kernel_ridge_regression.BinaryClassifier()

    >>> kernel_ridge_bc.load_train_data()

    >>> kernel_ridge.set_param(lambda_p=pow(10, -3), gamma=1, C=0.1)

    >>> kernel_ridge.init_W()

    >>> kernel_ridge.train()

    >>> test_data = '0.62771 0.11513 0.82235 0.14493 -1'

    >>> prediction = kernel_ridge.prediction(test_data)

Kernel Ridge Regression Multi Classification Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.KernelRidgeRegression as kernel_ridge_regression

    >>> kernel_ridge_mc = kernel_ridge_regression.MultiClassifier()

    >>> kernel_ridge_mc.load_train_data()

    >>> kernel_ridge_mc.set_param(lambda_p=pow(10, -3), gamma=1, C=0.1)

    >>> kernel_ridge_mc.init_W()

    >>> kernel_ridge_mc.train()

    >>> test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'

    >>> prediction = kernel_ridge_mc.prediction(test_data)

Least Squares Support Vector Machine Binary Classification Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.LeastSquaresSVM as least_squares_svm

    >>> least_squares_svm = least_squares_svm.BinaryClassifier()

    >>> least_squares_svm.load_train_data()

    >>> least_squares_svm.set_param(lambda_p=pow(10, -3), gamma=1, C=0.1)

    >>> least_squares_svm.init_W()

    >>> least_squares_svm.train()

    >>> test_data = '0.62771 0.11513 0.82235 0.14493 -1'

    >>> prediction = least_squares_svm.prediction(test_data)

Least Squares Support Vector Machine Multi Classification Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.LeastSquaresSVM as least_squares_svm

    >>> least_squares_mc = least_squares_svm.MultiClassifier()

    >>> least_squares_mc.load_train_data()

    >>> least_squares_mc.set_param(lambda_p=pow(10, -3), gamma=1, C=0.1)

    >>> least_squares_mc.init_W()

    >>> least_squares_mc.train()

    >>> test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'

    >>> prediction = least_squares_mc.prediction(test_data)

Soft Gaussian Kernel Support Vector Machine Multi Classification Learning Algorithm
============

Run demo dataset: One vs One
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.SupportVectorMachine as svm

    >>> svm_mc = svm.MultiClassifier()

    >>> svm_mc.load_train_data()

    >>> svm_mc.set_param(svm_kernel='soft_gaussian_kernel', gamma=1, C=1)

    >>> svm_mc.init_W()

    >>> svm_mc.train()

    >>> test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'

    >>> prediction = svm_mc.prediction(test_data)

Support Vector Regression Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.SupportVectorRegression as svr

    >>> sv_regression = svr.SupportVectorRegression()

    >>> sv_regression.load_train_data()

    >>> sv_regression.set_param(gamma=1, C=1, epsilon=0.1)

    >>> sv_regression.init_W()

    >>> sv_regression.train()

    >>> test_data = '0.62771 0.11513 0.82235 0.14493 -1'

    >>> prediction = sv_regression.prediction(test_data)

Neural Network Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.NeuralNetwork as nn

    >>> neural_network = nn.NeuralNetwork()

    >>> neural_network.load_train_data()

    >>> neural_network.set_param(network_structure=[8, 3, 1], w_range_high=0.1, w_range_low=-0.1, feed_mode='stochastic', step_eta=0.01, updates=50000)

    >>> neural_network.init_W()

    >>> neural_network.train()

    >>> test_data = '0.135592 0.0317051 -1'

    >>> prediction = neural_network.prediction(test_data)

Neural Network Binary Classification Learning Algorithm
============

Run demo dataset
-----------------

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.NeuralNetwork as nn

    >>> neural_network = nn.BinaryClassifier()

    >>> neural_network.load_train_data()

    >>> neural_network.set_param(network_structure=[8, 4, 1], w_range_high=0.1, w_range_low=-0.1, feed_mode='stochastic', step_eta=0.01, updates=50000)

    >>> neural_network.init_W()

    >>> neural_network.train()

    >>> test_data = '0.135592 0.0317051 -1'

    >>> prediction = neural_network.prediction(test_data)

Polynomial Feature Transform
============

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.PLA as pla

    >>> pla_bc = pla.BinaryClassifier()

    >>> pla_bc.load_train_data()

    >>> pla_bc.set_feature_transform('polynomial', 2)

Legendre Feature Transform
============

.. code-block:: py

    >>> import numpy as np

    >>> import FukuML.PLA as pla

    >>> pla_bc = pla.BinaryClassifier()

    >>> pla_bc.load_train_data()

    >>> pla_bc.set_feature_transform('legendre', 2)

10 Fold Cross Validation
============

.. code-block:: py

    >>> cross_validator = utility.CrossValidator()

    >>> pla_mc = pla.MultiClassifier()

    >>> pla_mc.load_train_data()

    >>> pla_mc.set_param()

    >>> pocket_mc = pocket.MultiClassifier()

    >>> pocket_mc.load_train_data()

    >>> pocket_mc.set_param()

    >>> cross_validator.add_model(pla_mc)

    >>> cross_validator.add_model(pocket_mc)

    >>> avg_errors = cross_validator.excute()

    >>> best_model = cross_validator.get_best_model()

Uniform Blending for Classification
============

.. code-block:: py

    >>> input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pocket_pla_binary_train.dat')

    >>> input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pocket_pla_binary_test.dat')

    >>> uniform_blending_classifier = utility.UniformBlendingClassifier()

    >>> pla_bc = pla.BinaryClassifier()

    >>> pla_bc.load_train_data(input_train_data_file)

    >>> pla_bc.load_test_data(input_test_data_file)

    >>> pla_bc.set_param()

    >>> pla_bc.init_W()

    >>> pla_bc.train()

    >>> print("PLA 平均錯誤值（Eout）：")

    >>> print(pla_bc.calculate_avg_error(pla_bc.test_X, pla_bc.test_Y, pla_bc.W))

    >>> pocket_bc = pocket.BinaryClassifier()

    >>> pocket_bc.load_train_data(input_train_data_file)

    >>> pocket_bc.load_test_data(input_test_data_file)

    >>> pocket_bc.set_param()

    >>> pocket_bc.init_W()

    >>> pocket_bc.train()

    >>> print("Pocket 平均錯誤值（Eout）：")

    >>> print(pocket_bc.calculate_avg_error(pocket_bc.test_X, pocket_bc.test_Y, pocket_bc.W))

    >>> linear_bc = linear_regression.BinaryClassifier()

    >>> linear_bc.load_train_data(input_train_data_file)

    >>> linear_bc.load_test_data(input_test_data_file)

    >>> linear_bc.set_param()

    >>> linear_bc.init_W()

    >>> linear_bc.train()

    >>> print("Linear 平均錯誤值（Eout）：")

    >>> print(linear_bc.calculate_avg_error(linear_bc.test_X, linear_bc.test_Y, linear_bc.W))

    >>> uniform_blending_classifier.add_model(pla_bc)

    >>> uniform_blending_classifier.add_model(pocket_bc)

    >>> uniform_blending_classifier.add_model(linear_bc)

    >>> test_data = '0.32368 0.61439 0.42097 0.025626 -1'

    >>> prediction = uniform_blending_classifier.prediction(test_data)

    >>> print("測試資料 x：")

    >>> print(prediction['input_data_x'])

    >>> print("測試資料 y：")

    >>> print(prediction['input_data_y'])

    >>> print("預測結果：")

    >>> print(prediction['prediction'])

    >>> print("平均錯誤率（Ein）：")

    >>> print(uniform_blending_classifier.calculate_avg_error(input_train_data_file))

    >>> print("平均錯誤率（Eout）：")

    >>> print(uniform_blending_classifier.calculate_avg_error(input_test_data_file))
