# Project Overview

This is FukuML, a Python library for machine learning. It provides implementations of various algorithms, including:

*   **Classification:**
    *   Perceptron (Binary and Multi-class)
    *   Pocket PLA (Binary and Multi-class)
    *   Logistic Regression (Binary and Multi-class)
    *   Support Vector Machine (SVM) (Binary and Multi-class)
    *   Decision Tree (CART)
    *   Random Forest
    *   AdaBoost
*   **Regression:**
    *   Linear Regression
    *   Ridge Regression
    *   Kernel Ridge Regression
    *   Support Vector Regression (SVR)
    *   Gradient Boost Decision Tree
*   **Other Tools:**
    *   Feature Transformation (Polynomial, Legendre)
    *   Cross-Validation
    *   Blending (Uniform, Linear)

The library is written in Python and uses `numpy`, `scipy`, and `scikit-learn` for numerical operations.

# Building and Running

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Testing

The project uses `unittest` for testing. To run the tests, execute the following command:

```bash
python test_fuku_ml.py
```

Or with coverage:

```bash
coverage run --source=FukuML setup.py test
```

# Development Conventions

## Code Style

The project follows the PEP8 style guide. To check for compliance, run:

```bash
pep8 FukuML/*.py --ignore=E501
```

## Usage

The `doc/sample_code.rst` file contains many examples of how to use the different algorithms in the library. Here is a basic example of how to use the Perceptron Binary Classifier:

```python
import numpy as np
import FukuML.PLA as pla

# Create a binary classifier
pla_bc = pla.BinaryClassifier()

# Load the training data
pla_bc.load_train_data()

# Set the parameters
pla_bc.set_param()

# Initialize the weights
pla_bc.init_W()

# Train the model
pla_bc.train()

# Make a prediction on new data
test_data = '0.97681 0.10723 0.64385 0.29556 1'
prediction = pla_bc.prediction(test_data)

print(prediction)
```
