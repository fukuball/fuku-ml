# encoding=utf8

import os
import sys
import unittest

# Import modular test cases for method delegation
tests_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests')
sys.path.insert(0, tests_dir)
from test_utility import UtilityTestCase
from test_pla_binary_classifier import PLABinaryClassifierTestCase
from test_pla_multi_classifier import PLAMultiClassifierTestCase
from test_pocket_pla_binary_classifier import PocketPLABinaryClassifierTestCase
from test_pocket_pla_multi_classifier import PocketPLAMultiClassifierTestCase
from test_linear_regression import LinearRegressionTestCase
from test_linear_regression_binary_classifier import LinearRegressionBinaryClassifierTestCase
from test_linear_regression_multi_classifier import LinearRegressionMultiClassifierTestCase
from test_logistic_regression import LogisticRegressionTestCase
from test_logistic_regression_binary_classifier import LogisticRegressionBinaryClassifierTestCase
from test_logistic_regression_multi_classifier import LogisticRegressionMultiClassifierTestCase
from test_l2r_logistic_regression import L2RLogisticRegressionTestCase
from test_l2r_logistic_regression_binary_classifier import L2RLogisticRegressionBinaryClassifierTestCase
from test_binary_classifier_with_feature_transform import BinaryClassifierWithFeatureTransformTestCase
from test_multi_classifier_with_feature_transform import MultiClassifierWithFeatureTransformTestCase
from test_binary_classification_cross_validation import BinaryClassificationCrossValidationTestCase
from test_multi_classification_cross_validation import MultiClassificationCrossValidationTestCase
from test_ridge_regression import RidgeRegressionTestCase
from test_ridge_regression_binary_classifier import RidgeRegressionBinaryClassifierTestCase
from test_ridge_regression_multi_classifier import RidgeRegressionMultiClassifierTestCase
from test_primal_hard_margin_svm_binary_classifier import PrimalHardMarginSVMBinaryClassifierTestCase
from test_dual_hard_margin_svm_binary_classifier import DualHardMarginSVMBinaryClassifierTestCase
from test_non_linear_hard_margin_svm_binary_classifier import NonLinearHardMarginSVMBinaryClassifierTestCase
from test_polynomial_kernel_svm_binary_classifier import PolynomialKernelSVMBinaryClassifierTestCase
from test_gaussian_kernel_svm_binary_classifier import GaussianKernelSVMBinaryClassifierTestCase
from test_soft_polynomial_kernel_svm_binary_classifier import SoftPolynomialKernelSVMBinaryClassifierTestCase
from test_soft_gaussian_kernel_svm_binary_classifier import SoftGaussianKernelSVMBinaryClassifierTestCase
from test_probabilistic_svm import ProbabilisticSVMTestCase
from test_decision_stump_binary_classifier import DecisionStumpBinaryClassifierTestCase
from test_decision_tree import DecisionTreeTestCase
from test_random_forest import RandomForestTestCase
from test_kernel_logistic_regression import KernelLogisticRegressionTestCase
from test_kernel_ridge_regression import KernelRidgeRegressionTestCase
from test_kernel_ridge_regression_binary_classifier import KernelRidgeRegressionBinaryClassifierTestCase
from test_kernel_ridge_regression_multi_classifier import KernelRidgeRegressionMultiClassifierTestCase
from test_least_squares_svm_binary_classifier import LeastSquaresSVMBinaryClassifierTestCase
from test_least_squares_svm_multi_classifier import LeastSquaresSVMMultiClassifierTestCase
from test_svm_multi_classifier import SVMMultiClassifierTestCase
from test_adaboost_stump_classifier import AdaBoostStumpClassifierTestCase
from test_adaboost_decision_tree_classifier import AdaBoostDecisionTreeClassifierTestCase
from test_gradient_boost_decision_tree_regression import GradientBoostDecisionTreeRegressionTestCase
from test_support_vector_regression import SupportVectorRegressionTestCase
from test_neural_network import NeuralNetworkTestCase
from test_neural_network_binary_classifier import NeuralNetworkBinaryClassifierTestCase
from test_classification_uniform_blending import ClassificationUniformBlendingTestCase
from test_classification_bootstrap_bagging_uniform_blending import ClassificationBootstrapBaggingUniformBlendingTestCase
from test_classification_linear_blending import ClassificationLinearBlendingTestCase
from test_regression_uniform_blending import RegressionUniformTestCase
from test_regression_linear_blending import RegressionLinearBlendingTestCase

class FukuMLTestCase(unittest.TestCase):

    '''
    def test_utility_method(self):
        return

    def test_pla_binary_classifier(self):
        return

    def test_pla_multi_classifier(self):
        return

    def test_pocket_pla_binary_classifier(self):
        return
    
    def test_pocket_pla_multi_classifier(self):
        return

    def test_linear_regression(self):
        return

    def test_linear_regression_binary_classifier(self):
        return

    def test_linear_regression_multi_classifier(self):
        return

    def test_logistic_regression(self):
        return

    def test_logistic_regression_binary_classifier(self):
        return

    def test_logistic_regression_multi_classifier(self):
        return

    def test_l2r_logistic_regression(self):
        return

    def test_l2r_logistic_regression_binary_classifier(self):
        return

    def test_binary_classifier_with_feature_transform(self):
        return

    def test_multi_classifier_with_feature_transform(self):
        return

    def test_binary_classification_cross_validation(self):
        return

    def test_multi_classification_cross_validation(self):
        return

    def test_ridge_regression(self):
        return

    def test_ridge_regression_binary_classifier(self):
        return

    def test_ridge_regression_multi_classifier(self):
        return

    def test_primal_hard_margin_svm_binary_classifier(self):
        return

    def test_dual_hard_margin_svm_binary_classifier(self):
        return

    def test_non_linear_hard_margin_svm_binary_classifier(self):
        return

    def test_polynomial_kernel_svm_binary_classifier(self):
        return

    def test_gaussian_kernel_svm_binary_classifier(self):
        return

    def test_soft_polynomial_kernel_svm_binary_classifier(self):
        return

    def test_soft_gaussian_kernel_svm_binary_classifier(self):
        return

    def test_probabilistic_svm(self):
        return

    def test_decision_stump_binary_classifier(self):
        return

    def test_decision_tree(self):
        return

    def test_random_forest(self):
        return

    def test_kernel_logistic_regression(self):
        return

    def test_kernel_ridge_regression(self):
        return

    def test_kernel_ridge_regression_binary_classifier(self):
        return

    def test_kernel_ridge_regression_multi_classifier(self):
        return

    def test_least_squares_svm_binary_classifier(self):
        return

    def test_least_squares_svm_multi_classifier(self):
        return

    def test_svm_multi_classifier(self):
        return

    def test_adaboost_stump_classifier(self):
        return

    def test_adaboost_decision_tree_classifier(self):
        return

    def test_gradient_boost_decision_tree_regression(self):
        return

    def test_support_vector_regression(self):
        return

    def test_neural_network(self):
        return

    def test_neural_network_binary_classifie(self):
        return

    def test_classification_uniform_blending(self):
        return

    def test_classification_bootstrap_bagging_uniform_blending(self):
        return

    def test_classification_linear_blending(self):
        return

    def test_regression_uniform_blending(self):
        return

    def test_regression_linear_blending(self):
        return
    '''

if __name__ == '__main__':

    unittest.main(verbosity=2)
