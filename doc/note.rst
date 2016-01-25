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

Competition

https://www.kaggle.com/