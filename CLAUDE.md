# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FukuML is a comprehensive machine learning library written in Python that implements numerous ML algorithms from scratch. It focuses on educational purposes and provides clean, understandable implementations of classic machine learning algorithms.

## Development Commands

### Testing
```bash
python test_fuku_ml.py
```
Run the complete test suite covering all ML algorithms.

### Code Quality
```bash
python -m pycodestyle FukuML/*.py --ignore=E501
```
Check PEP8 compliance (ignoring line length violations). Note: Use `pycodestyle` instead of deprecated `pep8`.

### Package Building
```bash
python setup.py sdist
python setup.py bdist_wheel --universal
```

### Package Deployment
```bash
./deploy_to_pip.sh
```
Deploy package to PyPI (removes build artifacts, builds distribution, uploads via twine).

## Architecture

### Core Structure
- **FukuML/**: Main package directory containing all ML algorithm implementations
- **FukuML/MLBase.py**: Abstract base class `Learner` defining common interface for all ML algorithms
- **FukuML/Utility.py**: Shared utilities for data loading, feature transformation, validation, and kernels
- **FukuML/dataset/**: Training and test datasets for various algorithms

### Algorithm Organization
All algorithms follow a consistent pattern inherited from `MLBase.Learner`:
- **Binary/Multi Classifiers**: Support both binary and multi-class classification
- **Regressors**: Implement regression algorithms
- **Common Interface**: `load_train_data()`, `set_param()`, `init_W()`, `train()`, `prediction()`

### Algorithm Categories
1. **Perceptron Family**: PLA, PocketPLA with binary/multi-class variants
2. **Regression**: Linear, Ridge, Kernel Ridge with classification variants
3. **Logistic Regression**: Standard, L2 regularized, Kernel variants
4. **Support Vector Machines**: Hard/soft margin, polynomial/Gaussian kernels, probabilistic, least squares
5. **Tree-based**: Decision Trees, Random Forest, AdaBoost, Gradient Boost
6. **Neural Networks**: Basic feed-forward implementation
7. **Ensemble**: Blending methods for combining models

### Key Design Patterns
- **Abstract Base Class**: All algorithms inherit from `MLBase.Learner`
- **Feature Transformation**: Built-in polynomial and Legendre feature transforms
- **Cross Validation**: 10-fold cross validation support via `Utility.CrossValidator`
- **Data Loading**: Standardized dataset format and loading via `Utility.DatasetLoader`
- **Serialization**: Model persistence through `Utility.Serializer`

### Dependencies
The original requirements.txt specifies older versions, but modern versions work:
- numpy: Core numerical operations
- scipy: Scientific computing functions  
- scikit-learn: Benchmarking and validation
- cvxopt: Convex optimization for SVM
- pycodestyle: Code style checking (replaces deprecated pep8)

### Data Format
Training/test data files use space-separated format:
- Each line represents one data point
- Features separated by spaces
- Ground truth label at end of line
- Example: `0.97681 0.10723 0.64385 1` (features + label)

### Testing Strategy
Single comprehensive test file `test_fuku_ml.py` exercises all algorithms with default datasets. Tests validate training convergence and prediction accuracy across the full algorithm suite.

**Note**: Full test suite takes several minutes to complete due to comprehensive algorithm coverage.

### Installation Requirements
```bash
pip install numpy scipy scikit-learn cvxopt pycodestyle
```
Modern versions work despite older versions specified in requirements.txt.

### Python 2/3 Compatibility
The codebase has been updated for Python 2/3 compatibility:
- **Fixed**: String literal comparisons using `==` instead of `is`
- **Fixed**: Added `from __future__ import division` to files with mathematical operations
- **Fixed**: Added `from __future__ import print_function` to utility scripts
- **Fixed**: Explicit `list()` conversion for `range()` when used with `random.sample()`

### Additional Test Commands
```bash
python test_compatibility.py
```
Quick compatibility test for basic functionality.

### Known Issues
- Original requirements.txt dependencies are outdated but modern versions are compatible
- Full test suite is comprehensive but time-consuming (2+ minutes)
- Python 2 testing requires separate numpy installation in Python 2 environment