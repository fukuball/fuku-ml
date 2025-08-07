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
from test_primal_hard_margin_svm_binary_classifier import PrimalHardMarginSVMBinary
from test_dual_hard_margin_svm_binary_classifier import DualHardMarginSVMBinaryClassifierTestCase
from test_non_linear_hard_margin_svm_binary_classifier import NonLinearHardMarginSVMBinaryClassifierTestCase
from test_polynomial_kernel_svm_binary_classifier import PolynomialKernelSVMBinaryClassifierTestCase
from test_gaussian_kernel_svm_binary_classifier import GaussianKernelSVMBinaryClassifierTestCase
from test_soft_polynomial_kernel_svm_binary_classifier import SoftPolynomialKernelSVMBinaryClassifierTest
from test_soft_gaussian_kernel_svm_binary_classifier import SoftGaussianKernelSVMBinaryClassifierTestCase
from test_probabilistic_svm import ProbabilisticSVMTestCase
from test_decision_stump_binary_classifier import DecisionStumpBinaryClassifierTestCase

import FukuML.PLA as pla
import FukuML.PocketPLA as pocket
import FukuML.LinearRegression as linear_regression
import FukuML.LogisticRegression as logistic_regression
import FukuML.L2RLogisticRegression as l2r_logistic_regression
import FukuML.KernelLogisticRegression as kernel_logistic_regression
import FukuML.RidgeRegression as ridge_regression
import FukuML.KernelRidgeRegression as kernel_ridge_regression
import FukuML.SupportVectorMachine as svm
import FukuML.ProbabilisticSVM as probabilistic_svm
import FukuML.LeastSquaresSVM as least_squares_svm
import FukuML.SupportVectorRegression as svr
import FukuML.DecisionStump as decision_stump
import FukuML.DecisionTree as decision_tree
import FukuML.RandomForest as random_forest
import FukuML.AdaBoostStump as adaboost_stump
import FukuML.AdaBoostDecisionTree as adaboost_decision_tree
import FukuML.GradientBoostDecisionTree as gradient_boost_decision_tree
import FukuML.NeuralNetwork as nn
import FukuML.Blending as blending
import FukuML.Utility as utility


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

        #------------------------------------------------------------

        decision_tree_c = decision_tree.CART()
        decision_tree_c.load_train_data()
        decision_tree_c.set_param(learn_type='classifier')
        decision_tree_c.load_test_data()
        decision_tree_c.init_W()
        decision_tree_c.train()

        print("\n訓練得出 Decision Tree：")
        decision_tree_c.plot(decision_tree_c.decision_tree)

        test_data = '6.0 2.2 5.0 1.5 virginica'
        prediction = decision_tree_c.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        test_data = 'None None None 1.5'
        prediction = decision_tree_c.prediction(input_data=test_data, mode='future_data', with_missing_data=True)
        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        print("W 平均錯誤率（Ein）：")
        print(decision_tree_c.calculate_avg_error(decision_tree_c.train_X, decision_tree_c.train_Y, decision_tree_c.W))
        print("W 平均錯誤率（Eout）：")
        print(decision_tree_c.calculate_test_data_avg_error())
        print('-'*70)

        #------------------------------------------------------------

        decision_tree_c = decision_tree.CART()
        decision_tree_c.load_train_data()
        decision_tree_c.set_param(learn_type='classifier', prune_gain=0.1, prune_notify=True)
        decision_tree_c.load_test_data()
        decision_tree_c.init_W()
        decision_tree_c.train()

        print("\n訓練得出 Decision Tree：")
        decision_tree_c.plot(decision_tree_c.decision_tree)

        test_data = '6.0 2.2 5.0 1.5 virginica'
        prediction = decision_tree_c.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        test_data = 'None None None 1.5'
        prediction = decision_tree_c.prediction(input_data=test_data, mode='future_data', with_missing_data=True)
        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        print("W 平均錯誤率（Ein）：")
        print(decision_tree_c.calculate_avg_error(decision_tree_c.train_X, decision_tree_c.train_Y, decision_tree_c.W))
        print("W 平均錯誤率（Eout）：")
        print(decision_tree_c.calculate_test_data_avg_error())
        print('-'*70)

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/decision_tree_2_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/decision_tree_2_test.dat')

        decision_tree_c = decision_tree.CART()
        decision_tree_c.load_train_data(input_train_data_file)
        decision_tree_c.set_param(learn_type='classifier')
        decision_tree_c.load_test_data(input_test_data_file)
        decision_tree_c.init_W()
        decision_tree_c.train()

        print("\n訓練得出 Decision Tree：")
        decision_tree_c.plot(decision_tree_c.decision_tree)

        test_data = 'ohne leicht Streifen normal normal Tuberkulose'
        prediction = decision_tree_c.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        test_data = 'None leicht None Flocken fiepend'
        prediction = decision_tree_c.prediction(input_data=test_data, mode='future_data', with_missing_data=True)
        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        print("W 平均錯誤率（Ein）：")
        print(decision_tree_c.calculate_avg_error(decision_tree_c.train_X, decision_tree_c.train_Y, decision_tree_c.W))
        print("W 平均錯誤率（Eout）：")
        print(decision_tree_c.calculate_test_data_avg_error())
        print('-'*70)

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/linear_regression_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/linear_regression_test.dat')

        decision_tree_r = decision_tree.CART()
        decision_tree_r.load_train_data(input_train_data_file)
        decision_tree_r.set_param(learn_type='regression')
        decision_tree_r.load_test_data(input_test_data_file)
        decision_tree_r.init_W()
        decision_tree_r.train()

        print("\n訓練得出 Decision Tree：")
        decision_tree_r.plot(decision_tree_r.decision_tree)

        test_data = '55.7 43 285'
        prediction = decision_tree_r.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        test_data = 'None 43'
        prediction = decision_tree_r.prediction(input_data=test_data, mode='future_data', with_missing_data=True)
        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        print("W 平均錯誤值（Ein）：")
        print(decision_tree_r.calculate_avg_error(decision_tree_r.train_X, decision_tree_r.train_Y, decision_tree_r.W))
        print("W 平均錯誤值（Eout）：")
        print(decision_tree_r.calculate_test_data_avg_error())
        print('-'*70)

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/linear_regression_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/linear_regression_test.dat')

        decision_tree_r = decision_tree.CART()
        decision_tree_r.load_train_data(input_train_data_file)
        decision_tree_r.set_param(learn_type='regression', prune_gain=0.3, prune_notify=True)
        decision_tree_r.load_test_data(input_test_data_file)
        decision_tree_r.init_W()
        decision_tree_r.train()

        print("\n訓練得出 Decision Tree：")
        decision_tree_r.plot(decision_tree_r.decision_tree)

        test_data = '55.7 43 285'
        prediction = decision_tree_r.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        test_data = 'None 43'
        prediction = decision_tree_r.prediction(input_data=test_data, mode='future_data', with_missing_data=True)
        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        print("W 平均錯誤值（Ein）：")
        print(decision_tree_r.calculate_avg_error(decision_tree_r.train_X, decision_tree_r.train_Y, decision_tree_r.W))
        print("W 平均錯誤值（Eout）：")
        print(decision_tree_r.calculate_test_data_avg_error())
        print('-'*70)

    def test_random_forest(self):

        random_forest_c = random_forest.RandomForest()
        random_forest_c.load_train_data()
        random_forest_c.set_param(learn_type='classifier', blending_model_num=3)
        random_forest_c.load_test_data()
        random_forest_c.init_W()
        random_forest_c.train()

        test_data = '0.94544 0.42842 0.79833 0.16244 -1'
        prediction = random_forest_c.prediction(test_data)
        self.assertEqual(float(prediction['input_data_y']), float(prediction['prediction']))

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        print("W 平均錯誤率（Ein）：")
        print(random_forest_c.calculate_avg_error(random_forest_c.train_X, random_forest_c.train_Y, random_forest_c.W))
        print("W 平均錯誤率（Eout）：")
        print(random_forest_c.calculate_test_data_avg_error())
        print('-'*70)

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/linear_regression_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/linear_regression_test.dat')

        random_forest_r = random_forest.RandomForest()
        random_forest_r.load_train_data(input_train_data_file)
        random_forest_r.set_param(learn_type='regression', blending_model_num=3)
        random_forest_r.load_test_data(input_test_data_file)
        random_forest_r.init_W()
        random_forest_r.train()

        test_data = '61.9 56 348'
        prediction = random_forest_r.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        print("W 平均錯誤值（Ein）：")
        print(random_forest_r.calculate_avg_error(random_forest_r.train_X, random_forest_r.train_Y, random_forest_r.W))
        print("W 平均錯誤值（Eout）：")
        print(random_forest_r.calculate_test_data_avg_error())
        print('-'*70)

    def test_kernel_logistic_regression(self):

        #------------------------------------------------------------

        kernel_logistic = kernel_logistic_regression.KernelLogisticRegression()
        kernel_logistic.load_train_data()
        kernel_logistic.load_test_data()
        kernel_logistic.set_param(feed_mode='stochastic', step_eta=0.126, updates=2000, lambda_p=0.0001, gamma=1, C=0.1)
        kernel_logistic.init_W()
        kernel_logistic.train()

        print("\n訓練得出權重模型：")
        print(kernel_logistic.beta)

        test_data = '0.26502 0.5486 0.971 0.19333 0.12207 0.81528 0.46743 0.45889 0.31004 0.3307 0.43078 0.50661 0.57281 0.052715 0.50443 0.78686 0.20099 0.85909 0.26772 0.13751 1'
        prediction = kernel_logistic.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(kernel_logistic.error_function(prediction['input_data_x'], prediction['input_data_y'], kernel_logistic.W))
        print("W 平均錯誤值（Ein）：")
        print(kernel_logistic.calculate_avg_error(kernel_logistic.train_X, kernel_logistic.train_Y, kernel_logistic.W))
        print("W 平均錯誤值（Eout）：")
        print(kernel_logistic.calculate_avg_error(kernel_logistic.test_X, kernel_logistic.test_Y, kernel_logistic.W))
        print('-'*70)

    def test_kernel_ridge_regression(self):

        #------------------------------------------------------------

        kernel_ridge = kernel_ridge_regression.KernelRidgeRegression()
        kernel_ridge.load_train_data()
        kernel_ridge.load_test_data()
        kernel_ridge.set_param(lambda_p=pow(10, -3), gamma=1, C=0.1)
        kernel_ridge.init_W()
        kernel_ridge.train()

        print("\n訓練得出權重模型：")
        print(kernel_ridge.beta)

        test_data = '0.62771 0.11513 0.82235 0.14493 -1'
        prediction = kernel_ridge.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(kernel_ridge.error_function(prediction['prediction'], prediction['input_data_y']))
        print("W 平均錯誤值（Ein）：")
        print(kernel_ridge.calculate_avg_error(kernel_ridge.train_X, kernel_ridge.train_Y, kernel_ridge.W))
        print("W 平均錯誤值（Eout）：")
        print(kernel_ridge.calculate_avg_error(kernel_ridge.test_X, kernel_ridge.test_Y, kernel_ridge.W))
        print('-'*70)

    def test_kernel_ridge_regression_binary_classifier(self):

        #------------------------------------------------------------

        kernel_ridge_b = kernel_ridge_regression.BinaryClassifier()
        kernel_ridge_b.load_train_data()
        kernel_ridge_b.load_test_data()
        kernel_ridge_b.set_param(lambda_p=pow(10, -3), gamma=1, C=0.1)
        kernel_ridge_b.init_W()
        kernel_ridge_b.train()

        print("\n訓練得出權重模型：")
        print(kernel_ridge_b.beta)

        test_data = '0.62771 0.11513 0.82235 0.14493 -1'
        prediction = kernel_ridge_b.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(kernel_ridge_b.error_function(prediction['prediction'], prediction['input_data_y']))
        print("W 平均錯誤值（Ein）：")
        print(kernel_ridge_b.calculate_avg_error(kernel_ridge_b.train_X, kernel_ridge_b.train_Y, kernel_ridge_b.W))
        print("W 平均錯誤值（Eout）：")
        print(kernel_ridge_b.calculate_avg_error(kernel_ridge_b.test_X, kernel_ridge_b.test_Y, kernel_ridge_b.W))
        print('-'*70)

    def test_kernel_ridge_regression_multi_classifier(self):

        #------------------------------------------------------------

        kernel_ridge_mc = kernel_ridge_regression.MultiClassifier()
        kernel_ridge_mc.load_train_data()
        kernel_ridge_mc.load_test_data()
        kernel_ridge_mc.set_param(lambda_p=pow(10, -3), gamma=1, C=0.1)
        kernel_ridge_mc.init_W()
        kernel_ridge_mc.train()

        test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
        prediction = kernel_ridge_mc.prediction(test_data)
        self.assertEqual(float(prediction['input_data_y']), float(prediction['prediction']))

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("所有類型預測機率值：")
        print(prediction['prediction_list'])
        #print("W 平均錯誤率（Ein）：")
        #print(kernel_ridge_mc.calculate_avg_error_all_class(kernel_ridge_mc.train_X, kernel_ridge_mc.train_Y, kernel_ridge_mc.W))
        #print("W 平均錯誤率（Eout）：")
        #print(kernel_ridge_mc.calculate_avg_error_all_class(kernel_ridge_mc.test_X, kernel_ridge_mc.test_Y, kernel_ridge_mc.W))
        print('-'*70)

    def test_least_squares_svm_binary_classifier(self):

        #------------------------------------------------------------

        least_squares = least_squares_svm.BinaryClassifier()
        least_squares.load_train_data()
        least_squares.load_test_data()
        least_squares.set_param(lambda_p=pow(10, -3), gamma=1, C=0.1)
        least_squares.init_W()
        least_squares.train()

        print("\n訓練得出權重模型：")
        print(least_squares.beta)

        test_data = '0.62771 0.11513 0.82235 0.14493 -1'
        prediction = least_squares.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(least_squares.error_function(prediction['prediction'], prediction['input_data_y']))
        print("W 平均錯誤值（Ein）：")
        print(least_squares.calculate_avg_error(least_squares.train_X, least_squares.train_Y, least_squares.W))
        print("W 平均錯誤值（Eout）：")
        print(least_squares.calculate_avg_error(least_squares.test_X, least_squares.test_Y, least_squares.W))
        print('-'*70)

    def test_least_squares_svm_multi_classifier(self):

        #------------------------------------------------------------

        least_squares_mc = least_squares_svm.MultiClassifier()
        least_squares_mc.load_train_data()
        least_squares_mc.load_test_data()
        least_squares_mc.set_param(lambda_p=pow(10, -3), gamma=1, C=0.1)
        least_squares_mc.init_W()
        least_squares_mc.train()

        test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
        prediction = least_squares_mc.prediction(test_data)
        self.assertEqual(float(prediction['input_data_y']), float(prediction['prediction']))

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("所有類型預測機率值：")
        print(prediction['prediction_list'])
        #print("W 平均錯誤率（Ein）：")
        #print(least_squares_mc.calculate_avg_error_all_class(least_squares_mc.train_X, least_squares_mc.train_Y, least_squares_mc.W))
        #print("W 平均錯誤率（Eout）：")
        #print(least_squares_mc.calculate_avg_error_all_class(least_squares_mc.test_X, least_squares_mc.test_Y, least_squares_mc.W))
        print('-'*70)

    def test_svm_multi_classifier(self):

        #------------------------------------------------------------

        svm_mc = svm.MultiClassifier()
        svm_mc.load_train_data()
        svm_mc.load_test_data()
        svm_mc.set_param(svm_kernel='gaussian_kernel', gamma=0.001)
        svm_mc.init_W()
        svm_mc.train()

        test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
        prediction = svm_mc.prediction(test_data)
        self.assertEqual(float(prediction['input_data_y']), float(prediction['prediction']))

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("所有類型預測機率值：")
        print(prediction['prediction_list'])
        print("W 平均錯誤率（Ein）：")
        print(svm_mc.calculate_avg_error_all_class(svm_mc.train_X, svm_mc.train_Y, svm_mc.W))
        print("W 平均錯誤率（Eout）：")
        print(svm_mc.calculate_avg_error_all_class(svm_mc.test_X, svm_mc.test_Y, svm_mc.W))
        print('-'*70)

    def test_adaboost_stump_classifier(self):

        #------------------------------------------------------------

        adaboost_stump_bc = adaboost_stump.BinaryClassifier()
        adaboost_stump_bc.load_train_data()
        adaboost_stump_bc.load_test_data()
        adaboost_stump_bc.set_param(run_t=10)
        adaboost_stump_bc.init_W()
        adaboost_stump_bc.train()

        test_data = '-9.706 1.392 6.562 -6.543 -1.980 -6.261 -6.067 1.254 -1.071 1'
        prediction = adaboost_stump_bc.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        print("平均錯誤率（Ein）：")
        print(adaboost_stump_bc.calculate_avg_error(adaboost_stump_bc.train_X, adaboost_stump_bc.train_Y, adaboost_stump_bc.W))
        print("平均錯誤率（Eout）：")
        print(adaboost_stump_bc.calculate_avg_error(adaboost_stump_bc.test_X, adaboost_stump_bc.test_Y, adaboost_stump_bc.W))
        print('-'*70)

    def test_adaboost_decision_tree_classifier(self):

        #------------------------------------------------------------

        adaboost_decision_tree_c = adaboost_decision_tree.Classifier()
        adaboost_decision_tree_c.load_train_data()
        adaboost_decision_tree_c.load_test_data()
        adaboost_decision_tree_c.set_param(run_t=10)
        adaboost_decision_tree_c.init_W()
        adaboost_decision_tree_c.train()

        test_data = '-9.706 1.392 6.562 -6.543 -1.980 -6.261 -6.067 1.254 -1.071 1'
        prediction = adaboost_decision_tree_c.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        print("平均錯誤率（Ein）：")
        print(adaboost_decision_tree_c.calculate_avg_error(adaboost_decision_tree_c.train_X, adaboost_decision_tree_c.train_Y, adaboost_decision_tree_c.W))
        print("平均錯誤率（Eout）：")
        print(adaboost_decision_tree_c.calculate_avg_error(adaboost_decision_tree_c.test_X, adaboost_decision_tree_c.test_Y, adaboost_decision_tree_c.W))
        print('-'*70)

    def test_gradient_boost_decision_tree_regression(self):

        #------------------------------------------------------------

        gradient_boost_decision_tree_r = gradient_boost_decision_tree.Regression()
        gradient_boost_decision_tree_r.load_train_data()
        gradient_boost_decision_tree_r.load_test_data()
        gradient_boost_decision_tree_r.set_param(run_t=10)
        gradient_boost_decision_tree_r.init_W()
        gradient_boost_decision_tree_r.train()

        test_data = '0.62771 0.11513 0.82235 0.14493 -1'
        prediction = gradient_boost_decision_tree_r.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        print("平均錯誤值（Ein）：")
        print(gradient_boost_decision_tree_r.calculate_avg_error(gradient_boost_decision_tree_r.train_X, gradient_boost_decision_tree_r.train_Y, gradient_boost_decision_tree_r.W))
        print("平均錯誤值（Eout）：")
        print(gradient_boost_decision_tree_r.calculate_avg_error(gradient_boost_decision_tree_r.test_X, gradient_boost_decision_tree_r.test_Y, gradient_boost_decision_tree_r.W))
        print('-'*70)

    def test_support_vector_regression(self):

        #------------------------------------------------------------

        sv_regression = svr.SupportVectorRegression()
        sv_regression.load_train_data()
        sv_regression.load_test_data()
        sv_regression.set_param(gamma=1, C=1, epsilon=0.1)
        sv_regression.init_W()
        sv_regression.train()

        test_data = '0.62771 0.11513 0.82235 0.14493 -1'
        prediction = sv_regression.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(sv_regression.error_function(prediction['prediction'], prediction['input_data_y']))
        print("W 平均錯誤值（Ein）：")
        print(sv_regression.calculate_avg_error(sv_regression.train_X, sv_regression.train_Y, sv_regression.W))
        print("W 平均錯誤值（Eout）：")
        print(sv_regression.calculate_avg_error(sv_regression.test_X, sv_regression.test_Y, sv_regression.W))
        print('-'*70)

    def test_neural_network(self):

        neural_network = nn.NeuralNetwork()
        neural_network.load_train_data()
        neural_network.load_test_data()
        neural_network.set_param(network_structure=[8, 3, 1], w_range_high=0.1, w_range_low=-0.1, feed_mode='stochastic', step_eta=0.01, updates=50000)
        neural_network.init_W()
        neural_network.train()

        print("\n訓練得出權重模型：")
        print(neural_network.W)

        print("W 平均錯誤值（Ein）：")
        print(neural_network.calculate_avg_error(neural_network.train_X, neural_network.train_Y, neural_network.W))
        print("W 平均錯誤值（Eout）：")
        print(neural_network.calculate_avg_error(neural_network.test_X, neural_network.test_Y, neural_network.W))
        print('-'*70)

    def test_neural_network_binary_classifie(self):

        neural_network = nn.BinaryClassifier()
        neural_network.load_train_data()
        neural_network.load_test_data()
        neural_network.set_param(network_structure=[8, 4, 1], w_range_high=0.1, w_range_low=-0.1, feed_mode='stochastic', step_eta=0.01, updates=50000)
        neural_network.init_W()
        neural_network.train()

        print("\n訓練得出權重模型：")
        print(neural_network.W)

        print("W 平均錯誤率（Ein）：")
        print(neural_network.calculate_avg_error(neural_network.train_X, neural_network.train_Y, neural_network.W))
        print("W 平均錯誤率（Eout）：")
        print(neural_network.calculate_avg_error(neural_network.test_X, neural_network.test_Y, neural_network.W))
        print('-'*70)

    def test_classification_uniform_blending(self):

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pocket_pla_binary_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pocket_pla_binary_test.dat')

        uniform_blending_classifier = blending.UniformBlendingClassifier()

        pla_bc = pla.BinaryClassifier()
        pla_bc.load_train_data(input_train_data_file)
        pla_bc.load_test_data(input_test_data_file)
        pla_bc.set_param()
        pla_bc.init_W()
        pla_bc.train()
        print("PLA 平均錯誤值（Eout）：")
        print(pla_bc.calculate_avg_error(pla_bc.test_X, pla_bc.test_Y, pla_bc.W))
        pocket_bc = pocket.BinaryClassifier()
        pocket_bc.load_train_data(input_train_data_file)
        pocket_bc.load_test_data(input_test_data_file)
        pocket_bc.set_param()
        pocket_bc.init_W()
        pocket_bc.train()
        print("Pocket 平均錯誤值（Eout）：")
        print(pocket_bc.calculate_avg_error(pocket_bc.test_X, pocket_bc.test_Y, pocket_bc.W))
        linear_bc = linear_regression.BinaryClassifier()
        linear_bc.load_train_data(input_train_data_file)
        linear_bc.load_test_data(input_test_data_file)
        linear_bc.set_param()
        linear_bc.init_W()
        linear_bc.train()
        print("Linear 平均錯誤值（Eout）：")
        print(linear_bc.calculate_avg_error(linear_bc.test_X, linear_bc.test_Y, linear_bc.W))

        uniform_blending_classifier.add_model(pla_bc)
        uniform_blending_classifier.add_model(pocket_bc)
        uniform_blending_classifier.add_model(linear_bc)
        test_data = '0.32368 0.61439 0.42097 0.025626 -1'
        prediction = uniform_blending_classifier.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("平均錯誤率（Ein）：")
        print(uniform_blending_classifier.calculate_avg_error(input_train_data_file))
        print("平均錯誤率（Eout）：")
        print(uniform_blending_classifier.calculate_avg_error(input_test_data_file))
        print('-'*70)

    def test_classification_bootstrap_bagging_uniform_blending(self):

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pocket_pla_binary_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pocket_pla_binary_test.dat')

        uniform_blending_classifier = blending.UniformBlendingClassifier()

        pla_bc1 = pla.BinaryClassifier()
        pla_bc1.load_train_data(input_train_data_file)
        pla_bc1.load_test_data(input_test_data_file)
        pla_bc1.set_param()
        pla_bc1.init_W()
        pla_bc1.train_X, pla_bc1.train_Y = utility.DatasetLoader.bootstrap_bagging(pla_bc1.train_X, pla_bc1.train_Y, pla_bc1.data_num)
        pla_bc1.train()
        print("PLA 1 平均錯誤值（Eout）：")
        print(pla_bc1.calculate_avg_error(pla_bc1.test_X, pla_bc1.test_Y, pla_bc1.W))

        pla_bc2 = pla.BinaryClassifier()
        pla_bc2.load_train_data(input_train_data_file)
        pla_bc2.load_test_data(input_test_data_file)
        pla_bc2.set_param()
        pla_bc2.init_W()
        pla_bc2.train_X, pla_bc2.train_Y = utility.DatasetLoader.bootstrap_bagging(pla_bc2.train_X, pla_bc2.train_Y, pla_bc2.data_num)
        pla_bc2.train()
        print("PLA 2 平均錯誤值（Eout）：")
        print(pla_bc2.calculate_avg_error(pla_bc2.test_X, pla_bc2.test_Y, pla_bc2.W))

        pla_bc3 = pla.BinaryClassifier()
        pla_bc3.load_train_data(input_train_data_file)
        pla_bc3.load_test_data(input_test_data_file)
        pla_bc3.set_param()
        pla_bc3.init_W()
        pla_bc3.train_X, pla_bc3.train_Y = utility.DatasetLoader.bootstrap_bagging(pla_bc3.train_X, pla_bc3.train_Y, pla_bc3.data_num)
        pla_bc3.train()
        print("PLA 3 平均錯誤值（Eout）：")
        print(pla_bc3.calculate_avg_error(pla_bc3.test_X, pla_bc3.test_Y, pla_bc3.W))

        uniform_blending_classifier.add_model(pla_bc1)
        uniform_blending_classifier.add_model(pla_bc2)
        uniform_blending_classifier.add_model(pla_bc3)
        test_data = '0.32368 0.61439 0.42097 0.025626 -1'
        prediction = uniform_blending_classifier.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("平均錯誤率（Ein）：")
        print(uniform_blending_classifier.calculate_avg_error(input_train_data_file))
        print("平均錯誤率（Eout）：")
        print(uniform_blending_classifier.calculate_avg_error(input_test_data_file))
        print('-'*70)

    def test_classification_linear_blending(self):

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pocket_pla_binary_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pocket_pla_binary_test.dat')

        linear_blending_classifier = blending.LinearBlendingClassifier()

        pla_bc = pla.BinaryClassifier()
        pla_bc.load_train_data(input_train_data_file)
        pla_bc.load_test_data(input_test_data_file)
        pla_bc.set_param()
        pla_bc.init_W()
        pla_bc.train()
        print("PLA 平均錯誤值（Eout）：")
        print(pla_bc.calculate_avg_error(pla_bc.test_X, pla_bc.test_Y, pla_bc.W))
        pocket_bc = pocket.BinaryClassifier()
        pocket_bc.load_train_data(input_train_data_file)
        pocket_bc.load_test_data(input_test_data_file)
        pocket_bc.set_param()
        pocket_bc.init_W()
        pocket_bc.train()
        print("Pocket 平均錯誤值（Eout）：")
        print(pocket_bc.calculate_avg_error(pocket_bc.test_X, pocket_bc.test_Y, pocket_bc.W))
        linear_bc = linear_regression.BinaryClassifier()
        linear_bc.load_train_data(input_train_data_file)
        linear_bc.load_test_data(input_test_data_file)
        linear_bc.set_param()
        linear_bc.init_W()
        linear_bc.train()
        print("Linear 平均錯誤值（Eout）：")
        print(linear_bc.calculate_avg_error(linear_bc.test_X, linear_bc.test_Y, linear_bc.W))

        linear_blending_classifier.add_model(pla_bc)
        linear_blending_classifier.add_model(pocket_bc)
        linear_blending_classifier.add_model(linear_bc)
        linear_blending_classifier.train()

        test_data = '0.32368 0.61439 0.42097 0.025626 -1'
        prediction = linear_blending_classifier.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("平均錯誤率（Ein）：")
        print(linear_blending_classifier.calculate_avg_error(input_train_data_file))
        print("平均錯誤率（Eout）：")
        print(linear_blending_classifier.calculate_avg_error(input_test_data_file))
        print('-'*70)

    def test_regression_uniform_blending(self):

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pocket_pla_binary_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pocket_pla_binary_test.dat')

        uniform_blending_regression = blending.UniformBlendingRegression()

        linear1 = linear_regression.LinearRegression()
        linear1.load_train_data(input_train_data_file)
        linear1.load_test_data(input_test_data_file)
        linear1.set_param()
        linear1.init_W()
        linear1.train_X, linear1.train_Y = utility.DatasetLoader.bootstrap_bagging(linear1.train_X, linear1.train_Y, linear1.data_num)
        linear1.train()

        print("Linear 1 平均錯誤值（Eout）：")
        print(linear1.calculate_avg_error(linear1.test_X, linear1.test_Y, linear1.W))

        linear2 = linear_regression.LinearRegression()
        linear2.load_train_data(input_train_data_file)
        linear2.load_test_data(input_test_data_file)
        linear2.set_param()
        linear2.init_W()
        linear2.train_X, linear2.train_Y = utility.DatasetLoader.bootstrap_bagging(linear2.train_X, linear2.train_Y, linear2.data_num)
        linear2.train()

        print("Linear 2 平均錯誤值（Eout）：")
        print(linear2.calculate_avg_error(linear2.test_X, linear2.test_Y, linear2.W))

        linear3 = linear_regression.LinearRegression()
        linear3.load_train_data(input_train_data_file)
        linear3.load_test_data(input_test_data_file)
        linear3.set_param()
        linear3.init_W()
        linear3.train_X, linear1.train_Y = utility.DatasetLoader.bootstrap_bagging(linear3.train_X, linear3.train_Y, linear3.data_num)
        linear3.train()

        print("Linear 3 平均錯誤值（Eout）：")
        print(linear3.calculate_avg_error(linear3.test_X, linear3.test_Y, linear3.W))

        uniform_blending_regression.add_model(linear1)
        uniform_blending_regression.add_model(linear2)
        uniform_blending_regression.add_model(linear3)
        test_data = '0.32368 0.61439 0.42097 0.025626 -1'
        prediction = uniform_blending_regression.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("平均錯誤值（Ein）：")
        print(uniform_blending_regression.calculate_avg_error(input_train_data_file))
        print("平均錯誤值（Eout）：")
        print(uniform_blending_regression.calculate_avg_error(input_test_data_file))
        print('-'*70)

    def test_regression_linear_blending(self):

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pocket_pla_binary_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pocket_pla_binary_test.dat')

        linear_blending_regression = blending.LinearBlendingRegression()

        linear1 = linear_regression.LinearRegression()
        linear1.load_train_data(input_train_data_file)
        linear1.load_test_data(input_test_data_file)
        linear1.set_param()
        linear1.init_W()
        linear1.train_X, linear1.train_Y = utility.DatasetLoader.bootstrap_bagging(linear1.train_X, linear1.train_Y, linear1.data_num)
        linear1.train()

        print("Linear 1 平均錯誤值（Eout）：")
        print(linear1.calculate_avg_error(linear1.test_X, linear1.test_Y, linear1.W))

        linear2 = linear_regression.LinearRegression()
        linear2.load_train_data(input_train_data_file)
        linear2.load_test_data(input_test_data_file)
        linear2.set_param()
        linear2.init_W()
        linear2.train_X, linear2.train_Y = utility.DatasetLoader.bootstrap_bagging(linear2.train_X, linear2.train_Y, linear2.data_num)
        linear2.train()

        print("Linear 2 平均錯誤值（Eout）：")
        print(linear2.calculate_avg_error(linear2.test_X, linear2.test_Y, linear2.W))

        linear3 = linear_regression.LinearRegression()
        linear3.load_train_data(input_train_data_file)
        linear3.load_test_data(input_test_data_file)
        linear3.set_param()
        linear3.init_W()
        linear3.train_X, linear1.train_Y = utility.DatasetLoader.bootstrap_bagging(linear3.train_X, linear3.train_Y, linear3.data_num)
        linear3.train()

        print("Linear 3 平均錯誤值（Eout）：")
        print(linear3.calculate_avg_error(linear3.test_X, linear3.test_Y, linear3.W))

        linear_blending_regression.add_model(linear1)
        linear_blending_regression.add_model(linear2)
        linear_blending_regression.add_model(linear3)
        linear_blending_regression.train()

        test_data = '0.32368 0.61439 0.42097 0.025626 -1'
        prediction = linear_blending_regression.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("平均錯誤值（Ein）：")
        print(linear_blending_regression.calculate_avg_error(input_train_data_file))
        print("平均錯誤值（Eout）：")
        print(linear_blending_regression.calculate_avg_error(input_test_data_file))
        print('-'*70)
    '''

if __name__ == '__main__':

    unittest.main(verbosity=2)
