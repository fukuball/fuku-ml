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

        #------------------------------------------------------------

        linear = linear_regression.LinearRegression()
        linear.load_train_data()
        linear.load_test_data()
        linear.set_param()
        linear.init_W()
        W = linear.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0.62771 0.11513 0.82235 0.14493 -1'
        prediction = linear.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(linear.error_function(prediction['prediction'], prediction['input_data_y']))
        print("W 平均錯誤值（Ein）：")
        print(linear.calculate_avg_error(linear.train_X, linear.train_Y, W))
        print("W 平均錯誤值（Eout）：")
        print(linear.calculate_avg_error(linear.test_X, linear.test_Y, W))
        print('-'*70)

    def test_linear_regression_binary_classifier(self):

        #------------------------------------------------------------

        linear_bc = linear_regression.BinaryClassifier()
        linear_bc.load_train_data()
        linear_bc.load_test_data()
        linear_bc.set_param()
        linear_bc.init_W()
        W = linear_bc.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0.62771 0.11513 0.82235 0.14493 -1'
        prediction = linear_bc.prediction(test_data)

        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("W 平均錯誤率（Ein）：")
        print(linear_bc.calculate_avg_error(linear_bc.train_X, linear_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(linear_bc.calculate_avg_error(linear_bc.test_X, linear_bc.test_Y, W))
        print('-'*70)

    def test_linear_regression_multi_classifier(self):

        #------------------------------------------------------------

        linear_mc = linear_regression.MultiClassifier()
        linear_mc.load_train_data()
        linear_mc.load_test_data()
        linear_mc.set_param()
        linear_mc.init_W()
        W = linear_mc.train()
        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
        prediction = linear_mc.prediction(test_data)
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
        print(linear_mc.calculate_avg_error_all_class(linear_mc.train_X, linear_mc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(linear_mc.calculate_avg_error_all_class(linear_mc.test_X, linear_mc.test_Y, W))
        print('-'*70)

    def test_logistic_regression(self):

        #------------------------------------------------------------

        logistic = logistic_regression.LogisticRegression()
        logistic.load_train_data()
        logistic.load_test_data()
        logistic.set_param()
        logistic.init_W()
        W = logistic.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0.26502 0.5486 0.971 0.19333 0.12207 0.81528 0.46743 0.45889 0.31004 0.3307 0.43078 0.50661 0.57281 0.052715 0.50443 0.78686 0.20099 0.85909 0.26772 0.13751 1'
        prediction = logistic.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(logistic.error_function(prediction['input_data_x'], prediction['input_data_y'], W))
        print("W 平均錯誤值（Ein）：")
        print(logistic.calculate_avg_error(logistic.train_X, logistic.train_Y, W))
        print("W 平均錯誤值（Eout）：")
        print(logistic.calculate_avg_error(logistic.test_X, logistic.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        print("使用 Linear Regression 加速器：")

        logistic = logistic_regression.LogisticRegression()
        logistic.load_train_data()
        logistic.load_test_data()
        logistic.set_param()
        logistic.init_W('linear_regression_accelerator')
        W = logistic.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0.26502 0.5486 0.971 0.19333 0.12207 0.81528 0.46743 0.45889 0.31004 0.3307 0.43078 0.50661 0.57281 0.052715 0.50443 0.78686 0.20099 0.85909 0.26772 0.13751 1'
        prediction = logistic.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(logistic.error_function(prediction['input_data_x'], prediction['input_data_y'], W))
        print("W 平均錯誤值（Ein）：")
        print(logistic.calculate_avg_error(logistic.train_X, logistic.train_Y, W))
        print("W 平均錯誤值（Eout）：")
        print(logistic.calculate_avg_error(logistic.test_X, logistic.test_Y, W))
        print('-'*70)

    def test_logistic_regression_binary_classifier(self):

        #------------------------------------------------------------

        logistic = logistic_regression.BinaryClassifier()
        logistic.load_train_data()
        logistic.load_test_data()
        logistic.set_param()
        logistic.init_W()
        W = logistic.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0.26502 0.5486 0.971 0.19333 0.12207 0.81528 0.46743 0.45889 0.31004 0.3307 0.43078 0.50661 0.57281 0.052715 0.50443 0.78686 0.20099 0.85909 0.26772 0.13751 1'
        prediction = logistic.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("W 平均錯誤率（Ein）：")
        print(logistic.calculate_avg_error(logistic.train_X, logistic.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(logistic.calculate_avg_error(logistic.test_X, logistic.test_Y, W))
        print('-'*70)

        print("隨機梯度下降：")
        logistic.set_param(feed_mode='stochastic', step_eta=0.1, updates=2000)
        logistic.init_W()
        W = logistic.train()
        print("訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(logistic.calculate_avg_error(logistic.train_X, logistic.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(logistic.calculate_avg_error(logistic.test_X, logistic.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        print("使用 Linear Regression 加速器：")
        logistic.set_param(feed_mode='batch', step_eta=0.126, updates=2000)
        logistic.init_W('linear_regression_accelerator')
        W = logistic.train()
        print("\n訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(logistic.calculate_avg_error(logistic.train_X, logistic.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(logistic.calculate_avg_error(logistic.test_X, logistic.test_Y, W))
        print('-'*70)

    def test_logistic_regression_multi_classifier(self):

        #------------------------------------------------------------

        logistic = logistic_regression.MultiClassifier()
        logistic.load_train_data()
        logistic.load_test_data()
        logistic.set_param()
        logistic.init_W()
        W = logistic.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
        prediction = logistic.prediction(test_data)
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
        print(logistic.calculate_avg_error_all_class(logistic.train_X, logistic.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(logistic.calculate_avg_error_all_class(logistic.test_X, logistic.test_Y, W))
        print('-'*70)

        print("隨機梯度下降：")
        logistic.set_param(feed_mode='stochastic', step_eta=0.126, updates=2000)
        logistic.init_W()
        W = logistic.train()
        print("訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(logistic.calculate_avg_error_all_class(logistic.train_X, logistic.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(logistic.calculate_avg_error_all_class(logistic.test_X, logistic.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        print("使用 Linear Regression 加速器：")
        logistic.set_param(feed_mode='batch', step_eta=0.126, updates=2000)
        W = logistic.init_W('linear_regression_accelerator')
        print("\n加速初始化權重模型：")
        print(W)
        W = logistic.train()
        print("\n訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(logistic.calculate_avg_error_all_class(logistic.train_X, logistic.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(logistic.calculate_avg_error_all_class(logistic.test_X, logistic.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        print("One to One Multiclassification：")
        logistic.set_param()
        logistic.init_W('normal', 'ovo')
        W = logistic.train()
        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
        prediction = logistic.prediction(test_data)
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
        print(logistic.calculate_avg_error_all_class(logistic.train_X, logistic.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(logistic.calculate_avg_error_all_class(logistic.test_X, logistic.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        print("One to One Multiclassification 使用 Linear Regression 加速器：")
        logistic.set_param()
        logistic.init_W('linear_regression_accelerator', 'ovo')
        W = logistic.train()
        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
        prediction = logistic.prediction(test_data)
        self.assertEqual(float(prediction['input_data_y']), float(prediction['prediction']))

        print("W 平均錯誤率（Ein）：")
        print(logistic.calculate_avg_error_all_class(logistic.train_X, logistic.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(logistic.calculate_avg_error_all_class(logistic.test_X, logistic.test_Y, W))
        print('-'*70)

    def test_l2r_logistic_regression(self):

        #------------------------------------------------------------

        l2r_logistic = l2r_logistic_regression.L2RLogisticRegression()
        l2r_logistic.load_train_data()
        l2r_logistic.load_test_data()
        l2r_logistic.set_param(feed_mode='batch', step_eta=0.126, updates=2000, lambda_p=0.0001)
        l2r_logistic.init_W()
        W = l2r_logistic.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0.26502 0.5486 0.971 0.19333 0.12207 0.81528 0.46743 0.45889 0.31004 0.3307 0.43078 0.50661 0.57281 0.052715 0.50443 0.78686 0.20099 0.85909 0.26772 0.13751 1'
        prediction = l2r_logistic.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(l2r_logistic.error_function(prediction['input_data_x'], prediction['input_data_y'], W))
        print("W 平均錯誤值（Ein）：")
        print(l2r_logistic.calculate_avg_error(l2r_logistic.train_X, l2r_logistic.train_Y, W))
        print("W 平均錯誤值（Eout）：")
        print(l2r_logistic.calculate_avg_error(l2r_logistic.test_X, l2r_logistic.test_Y, W))
        print('-'*70)

    def test_l2r_logistic_regression_binary_classifier(self):

        #------------------------------------------------------------

        l2r_logistic = l2r_logistic_regression.BinaryClassifier()
        l2r_logistic.load_train_data()
        l2r_logistic.load_test_data()
        l2r_logistic.set_param(feed_mode='batch', step_eta=0.126, updates=2000, lambda_p=0.0001)
        l2r_logistic.init_W()
        W = l2r_logistic.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0.26502 0.5486 0.971 0.19333 0.12207 0.81528 0.46743 0.45889 0.31004 0.3307 0.43078 0.50661 0.57281 0.052715 0.50443 0.78686 0.20099 0.85909 0.26772 0.13751 1'
        prediction = l2r_logistic.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("W 平均錯誤率（Ein）：")
        print(l2r_logistic.calculate_avg_error(l2r_logistic.train_X, l2r_logistic.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(l2r_logistic.calculate_avg_error(l2r_logistic.test_X, l2r_logistic.test_Y, W))
        print('-'*70)

        print("隨機梯度下降：")
        l2r_logistic.set_param(feed_mode='stochastic', step_eta=0.1, updates=2000, lambda_p=0.0001)
        l2r_logistic.init_W()
        W = l2r_logistic.train()
        print("訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(l2r_logistic.calculate_avg_error(l2r_logistic.train_X, l2r_logistic.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(l2r_logistic.calculate_avg_error(l2r_logistic.test_X, l2r_logistic.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        print("使用 Linear Regression 加速器：")
        l2r_logistic.set_param(feed_mode='batch', step_eta=0.126, updates=2000, lambda_p=0.0001)
        l2r_logistic.init_W('linear_regression_accelerator')
        W = l2r_logistic.train()
        print("\n訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(l2r_logistic.calculate_avg_error(l2r_logistic.train_X, l2r_logistic.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(l2r_logistic.calculate_avg_error(l2r_logistic.test_X, l2r_logistic.test_Y, W))
        print('-'*70)

    def test_binary_classifier_with_feature_transform(self):

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pocket_pla_binary_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pocket_pla_binary_test.dat')

        pla_bc = pla.BinaryClassifier()
        pla_bc.load_train_data(input_train_data_file)
        pla_bc.set_param()
        pla_bc.init_W()
        W = pla_bc.train()
        pla_bc.load_test_data(input_test_data_file)

        print("\nPLA 訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pla_bc.tune_times)
        print("W 平均錯誤率（Ein）：")
        print(pla_bc.calculate_avg_error(pla_bc.train_X, pla_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pla_bc.calculate_avg_error(pla_bc.test_X, pla_bc.test_Y, W))
        print('-'*70)

        pla_bc = pla.BinaryClassifier()
        pla_bc.load_train_data(input_train_data_file)
        pla_bc.set_feature_transform('polynomial', 2)
        pla_bc.set_param()
        pla_bc.init_W()
        W = pla_bc.train()
        pla_bc.load_test_data(input_test_data_file)

        print("\nPLA Polynomial 非線性轉換訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pla_bc.tune_times)
        print("W 平均錯誤率（Ein）：")
        print(pla_bc.calculate_avg_error(pla_bc.train_X, pla_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pla_bc.calculate_avg_error(pla_bc.test_X, pla_bc.test_Y, W))
        print('-'*70)

        pla_bc = pla.BinaryClassifier()
        pla_bc.load_train_data(input_train_data_file)
        pla_bc.set_feature_transform('legendre', 2)
        pla_bc.set_param()
        pla_bc.init_W()
        W = pla_bc.train()
        pla_bc.load_test_data(input_test_data_file)

        print("\nPLA Legendre 非線性轉換訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pla_bc.tune_times)
        print("W 平均錯誤率（Ein）：")
        print(pla_bc.calculate_avg_error(pla_bc.train_X, pla_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pla_bc.calculate_avg_error(pla_bc.test_X, pla_bc.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        pocket_bc = pocket.BinaryClassifier()
        pocket_bc.load_train_data(input_train_data_file)
        pocket_bc.set_param(loop_mode='naive_cycle', step_alpha=1, updates=50)
        pocket_bc.init_W()
        W = pocket_bc.train()
        pocket_bc.load_test_data(input_test_data_file)

        print("Pocket 訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pocket_bc.tune_times)
        print("W 效果改善次數：")
        print(pocket_bc.put_in_pocket_times)
        print("W 平均錯誤率（Ein）：")
        print(pocket_bc.calculate_avg_error(pocket_bc.train_X, pocket_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pocket_bc.calculate_avg_error(pocket_bc.test_X, pocket_bc.test_Y, W))
        print('-'*70)

        pocket_bc = pocket.BinaryClassifier()
        pocket_bc.load_train_data(input_train_data_file)
        pocket_bc.set_feature_transform('polynomial', 2)
        pocket_bc.set_param(loop_mode='naive_cycle', step_alpha=1, updates=50)
        pocket_bc.init_W()
        W = pocket_bc.train()
        pocket_bc.load_test_data(input_test_data_file)

        print("Pocket Polynomial 非線性轉換訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pocket_bc.tune_times)
        print("W 效果改善次數：")
        print(pocket_bc.put_in_pocket_times)
        print("W 平均錯誤率（Ein）：")
        print(pocket_bc.calculate_avg_error(pocket_bc.train_X, pocket_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pocket_bc.calculate_avg_error(pocket_bc.test_X, pocket_bc.test_Y, W))
        print('-'*70)

        pocket_bc = pocket.BinaryClassifier()
        pocket_bc.load_train_data(input_train_data_file)
        pocket_bc.set_feature_transform('legendre', 2)
        pocket_bc.set_param(loop_mode='naive_cycle', step_alpha=1, updates=50)
        pocket_bc.init_W()
        W = pocket_bc.train()
        pocket_bc.load_test_data(input_test_data_file)

        print("Pocket Legendre 非線性轉換訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pocket_bc.tune_times)
        print("W 效果改善次數：")
        print(pocket_bc.put_in_pocket_times)
        print("W 平均錯誤率（Ein）：")
        print(pocket_bc.calculate_avg_error(pocket_bc.train_X, pocket_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pocket_bc.calculate_avg_error(pocket_bc.test_X, pocket_bc.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        linear_bc = linear_regression.BinaryClassifier()
        linear_bc.load_train_data(input_train_data_file)
        linear_bc.set_param()
        linear_bc.init_W()
        W = linear_bc.train()
        linear_bc.load_test_data(input_test_data_file)

        print("\nLinear Regression 訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(linear_bc.calculate_avg_error(linear_bc.train_X, linear_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(linear_bc.calculate_avg_error(linear_bc.test_X, linear_bc.test_Y, W))
        print('-'*70)

        linear_bc = linear_regression.BinaryClassifier()
        linear_bc.load_train_data(input_train_data_file)
        linear_bc.set_feature_transform('polynomial', 2)
        linear_bc.set_param()
        linear_bc.init_W()
        W = linear_bc.train()
        linear_bc.load_test_data(input_test_data_file)

        print("\nLinear Regression Polynomial 訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(linear_bc.calculate_avg_error(linear_bc.train_X, linear_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(linear_bc.calculate_avg_error(linear_bc.test_X, linear_bc.test_Y, W))
        print('-'*70)

        linear_bc = linear_regression.BinaryClassifier()
        linear_bc.load_train_data(input_train_data_file)
        linear_bc.set_feature_transform('legendre', 2)
        linear_bc.set_param()
        linear_bc.init_W()
        W = linear_bc.train()
        linear_bc.load_test_data(input_test_data_file)

        print("\nLinear Regression Legendre 訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(linear_bc.calculate_avg_error(linear_bc.train_X, linear_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(linear_bc.calculate_avg_error(linear_bc.test_X, linear_bc.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        logistic_bc = logistic_regression.BinaryClassifier()
        logistic_bc.load_train_data(input_train_data_file)
        logistic_bc.set_param()
        logistic_bc.init_W()
        W = logistic_bc.train()
        logistic_bc.load_test_data(input_test_data_file)

        print("\nLogistic Regression 訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(logistic_bc.calculate_avg_error(logistic_bc.train_X, logistic_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(logistic_bc.calculate_avg_error(logistic_bc.test_X, logistic_bc.test_Y, W))
        print('-'*70)

        logistic_bc = logistic_regression.BinaryClassifier()
        logistic_bc.load_train_data(input_train_data_file)
        logistic_bc.set_feature_transform('polynomial', 2)
        logistic_bc.set_param()
        logistic_bc.init_W()
        W = logistic_bc.train()
        logistic_bc.load_test_data(input_test_data_file)

        print("\nLogistic Regression Polynomial 訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(logistic_bc.calculate_avg_error(logistic_bc.train_X, logistic_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(logistic_bc.calculate_avg_error(logistic_bc.test_X, logistic_bc.test_Y, W))
        print('-'*70)

        logistic_bc = logistic_regression.BinaryClassifier()
        logistic_bc.load_train_data(input_train_data_file)
        logistic_bc.set_feature_transform('legendre', 2)
        logistic_bc.set_param()
        logistic_bc.init_W()
        W = logistic_bc.train()
        logistic_bc.load_test_data(input_test_data_file)

        print("\nLogistic Regression Legendre 訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(logistic_bc.calculate_avg_error(logistic_bc.train_X, logistic_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(logistic_bc.calculate_avg_error(logistic_bc.test_X, logistic_bc.test_Y, W))
        print('-'*70)

    def test_multi_classifier_with_feature_transform(self):

        #------------------------------------------------------------

        pla_mc = pla.MultiClassifier()
        pla_mc.load_train_data()
        pla_mc.set_feature_transform('polynomial', 1)
        pla_mc.load_test_data()
        pla_mc.set_param()
        pla_mc.init_W()
        W = pla_mc.train()
        print("\nPLA Polynomial 多類訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(pla_mc.calculate_avg_error_all_class(pla_mc.train_X, pla_mc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pla_mc.calculate_avg_error_all_class(pla_mc.test_X, pla_mc.test_Y, W))
        print('-'*70)

        pla_mc = pla.MultiClassifier()
        pla_mc.load_train_data()
        pla_mc.set_feature_transform('legendre', 1)
        pla_mc.load_test_data()
        pla_mc.set_param()
        pla_mc.init_W()
        W = pla_mc.train()
        print("\nPLA Legendre 多類訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(pla_mc.calculate_avg_error_all_class(pla_mc.train_X, pla_mc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pla_mc.calculate_avg_error_all_class(pla_mc.test_X, pla_mc.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        pocket_mc = pocket.MultiClassifier()
        pocket_mc.load_train_data()
        pocket_mc.set_feature_transform('polynomial', 1)
        pocket_mc.load_test_data()
        pocket_mc.set_param()
        pocket_mc.init_W()
        W = pocket_mc.train()
        print("\nPocket Polynomial 多類訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(pocket_mc.calculate_avg_error_all_class(pocket_mc.train_X, pocket_mc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pocket_mc.calculate_avg_error_all_class(pocket_mc.test_X, pocket_mc.test_Y, W))
        print('-'*70)

        pocket_mc = pocket.MultiClassifier()
        pocket_mc.load_train_data()
        pocket_mc.set_feature_transform('legendre', 1)
        pocket_mc.load_test_data()
        pocket_mc.set_param()
        pocket_mc.init_W()
        W = pocket_mc.train()
        print("\nPocket Legendre 多類訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(pocket_mc.calculate_avg_error_all_class(pocket_mc.train_X, pocket_mc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pocket_mc.calculate_avg_error_all_class(pocket_mc.test_X, pocket_mc.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        linear_mc = linear_regression.MultiClassifier()
        linear_mc.load_train_data()
        linear_mc.set_feature_transform('polynomial', 1)
        linear_mc.load_test_data()
        linear_mc.set_param()
        linear_mc.init_W()
        W = linear_mc.train()
        print("\nLinear Regression Polynomial 多類訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(linear_mc.calculate_avg_error_all_class(linear_mc.train_X, linear_mc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(linear_mc.calculate_avg_error_all_class(linear_mc.test_X, linear_mc.test_Y, W))
        print('-'*70)

        linear_mc = linear_regression.MultiClassifier()
        linear_mc.load_train_data()
        linear_mc.set_feature_transform('legendre', 1)
        linear_mc.load_test_data()
        linear_mc.set_param()
        linear_mc.init_W()
        W = linear_mc.train()
        print("\nLinear Regression Legendre 多類訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(linear_mc.calculate_avg_error_all_class(linear_mc.train_X, linear_mc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(linear_mc.calculate_avg_error_all_class(linear_mc.test_X, linear_mc.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        logistic_mc = logistic_regression.MultiClassifier()
        logistic_mc.load_train_data()
        logistic_mc.set_feature_transform('polynomial', 1)
        logistic_mc.load_test_data()
        logistic_mc.set_param()
        logistic_mc.init_W()
        W = logistic_mc.train()

        print("\nLogistic Regression Polynomial 多類訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(logistic_mc.calculate_avg_error_all_class(logistic_mc.train_X, logistic_mc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(logistic_mc.calculate_avg_error_all_class(logistic_mc.test_X, logistic_mc.test_Y, W))
        print('-'*70)

        logistic_mc = logistic_regression.MultiClassifier()
        logistic_mc.load_train_data()
        logistic_mc.set_feature_transform('legendre', 1)
        logistic_mc.load_test_data()
        logistic_mc.set_param()
        logistic_mc.init_W()
        W = logistic_mc.train()

        print("\nLogistic Regression Legendre 多類訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(logistic_mc.calculate_avg_error_all_class(logistic_mc.train_X, logistic_mc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(logistic_mc.calculate_avg_error_all_class(logistic_mc.test_X, logistic_mc.test_Y, W))
        print('-'*70)

    def test_binary_classification_cross_validation(self):

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pocket_pla_binary_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pocket_pla_binary_test.dat')

        cross_validator = utility.CrossValidator()

        pla_bc = pla.BinaryClassifier()
        pla_bc.load_train_data(input_train_data_file)
        pla_bc.set_param()
        pocket_bc = pocket.BinaryClassifier()
        pocket_bc.load_train_data(input_train_data_file)
        pocket_bc.set_param()
        linear_bc = linear_regression.BinaryClassifier()
        linear_bc.load_train_data(input_train_data_file)
        linear_bc.set_param()
        logistic_bc = logistic_regression.BinaryClassifier()
        logistic_bc.load_train_data(input_train_data_file)
        logistic_bc.set_param()

        print("\n10 fold cross validation：")

        cross_validator.add_model(pla_bc)
        cross_validator.add_model(pocket_bc)
        cross_validator.add_model(linear_bc)
        cross_validator.add_model(logistic_bc)
        avg_errors = cross_validator.excute()

        print("\n各模型驗證平均錯誤：")
        print(avg_errors)
        print("\n最小平均錯誤率：")
        print(cross_validator.get_min_avg_error())

        print("\n取得最佳模型：")
        best_model = cross_validator.get_best_model()
        print(best_model)
        best_model.init_W()
        W = best_model.train()
        best_model.load_test_data(input_test_data_file)

        print("\n訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(best_model.calculate_avg_error(best_model.train_X, best_model.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(best_model.calculate_avg_error(best_model.test_X, best_model.test_Y, W))
        print('-'*70)

    def test_multi_classification_cross_validation(self):

        cross_validator = utility.CrossValidator()

        pla_mc = pla.MultiClassifier()
        pla_mc.load_train_data()
        pla_mc.set_param()
        pocket_mc = pocket.MultiClassifier()
        pocket_mc.load_train_data()
        pocket_mc.set_param()

        print("\n10 fold cross validation：")

        cross_validator.add_model(pla_mc)
        cross_validator.add_model(pocket_mc)
        avg_errors = cross_validator.excute()

        print("\n各模型驗證平均錯誤：")
        print(avg_errors)
        print("\n最小平均錯誤率：")
        print(cross_validator.get_min_avg_error())

        print("\n取得最佳模型：")
        best_model = cross_validator.get_best_model()
        print(best_model)
        best_model.init_W()
        W = best_model.train()
        best_model.load_test_data()

        print("\n訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(best_model.calculate_avg_error_all_class(best_model.train_X, best_model.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(best_model.calculate_avg_error_all_class(best_model.test_X, best_model.test_Y, W))
        print('-'*70)

    def test_ridge_regression(self):

        #------------------------------------------------------------

        ridge = ridge_regression.RidgeRegression()
        ridge.load_train_data()
        ridge.load_test_data()
        ridge.set_param(lambda_p=pow(10, -3))
        ridge.init_W()
        W = ridge.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0.62771 0.11513 0.82235 0.14493 -1'
        prediction = ridge.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(ridge.error_function(prediction['prediction'], prediction['input_data_y']))
        print("W 平均錯誤值（Ein）：")
        print(ridge.calculate_avg_error(ridge.train_X, ridge.train_Y, W))
        print("W 平均錯誤值（Eout）：")
        print(ridge.calculate_avg_error(ridge.test_X, ridge.test_Y, W))
        print('-'*70)

    def test_ridge_regression_binary_classifier(self):

        #------------------------------------------------------------

        cross_validator = utility.CrossValidator()

        lambda_p_set = [i for i in range(2, -11, -1)]

        for lambda_p in lambda_p_set:
            ridge_bc = ridge_regression.BinaryClassifier()
            ridge_bc.load_train_data()
            ridge_bc.set_param(lambda_p=pow(10, lambda_p))
            cross_validator.add_model(ridge_bc)

        print("\n10 fold cross validation：")
        avg_errors = cross_validator.excute()

        print("\n各模型驗證平均錯誤：")
        print(avg_errors)
        print("\n最小平均錯誤率：")
        print(cross_validator.get_min_avg_error())

        print("\n取得最佳模型：")
        best_model = cross_validator.get_best_model()
        print(best_model)

        best_model.load_test_data()
        best_model.init_W()
        W = best_model.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0.402041 0.402048 -1'
        prediction = best_model.prediction(test_data)

        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("W 平均錯誤率（Ein）：")
        print(best_model.calculate_avg_error(best_model.train_X, best_model.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(best_model.calculate_avg_error(best_model.test_X, best_model.test_Y, W))
        print('-'*70)

    def test_ridge_regression_multi_classifier(self):

        #------------------------------------------------------------

        ridge_mc = ridge_regression.MultiClassifier()
        ridge_mc.load_train_data()
        ridge_mc.load_test_data()
        ridge_mc.set_param(lambda_p=pow(10, -3))
        ridge_mc.init_W()
        W = ridge_mc.train()
        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
        prediction = ridge_mc.prediction(test_data)
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
        print(ridge_mc.calculate_avg_error_all_class(ridge_mc.train_X, ridge_mc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(ridge_mc.calculate_avg_error_all_class(ridge_mc.test_X, ridge_mc.test_Y, W))
        print('-'*70)

    def test_primal_hard_margin_svm_binary_classifier(self):

        #------------------------------------------------------------

        svm_bc = svm.BinaryClassifier()
        svm_bc.load_train_data()
        svm_bc.load_test_data()
        svm_bc.set_param(svm_kernel='primal_hard_margin')
        svm_bc.init_W()
        W = svm_bc.train()
        print("\n訓練得出權重模型：")
        print(W)
        print("SVM Marging：")
        print(svm_bc.get_marge())

        test_data = '0.97681 0.10723 0.64385 0.29556 1'
        prediction = svm_bc.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print('-'*70)

        test_data = '0.15654 0.75584 0.01122 0.42598 -1'
        prediction = svm_bc.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("W 平均錯誤率（Ein）：")
        print(svm_bc.calculate_avg_error(svm_bc.train_X, svm_bc.train_Y, W))
        print('-'*70)

    def test_dual_hard_margin_svm_binary_classifier(self):

        #------------------------------------------------------------

        svm_bc = svm.BinaryClassifier()
        svm_bc.load_train_data()
        svm_bc.load_test_data()
        svm_bc.set_param(svm_kernel='dual_hard_margin')
        svm_bc.init_W()
        W = svm_bc.train()

        print("\n訓練得出權重模型：")
        print(W)
        print("SVM Marging：")
        print(svm_bc.get_marge())
        print("Support Vectors")
        print(svm_bc.get_support_vectors())

        test_data = '0.97681 0.10723 0.64385 0.29556 1'
        prediction = svm_bc.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print('-'*70)

        test_data = '0.15654 0.75584 0.01122 0.42598 -1'
        prediction = svm_bc.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("W 平均錯誤率（Ein）：")
        print(svm_bc.calculate_avg_error(svm_bc.train_X, svm_bc.train_Y, W))
        print('-'*70)

    def test_non_linear_hard_margin_svm_binary_classifier(self):

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/non_linear_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/non_linear_test.dat')

        svm_bc = svm.BinaryClassifier()
        svm_bc.load_train_data(input_train_data_file)
        svm_bc.set_feature_transform('legendre', 3)
        svm_bc.load_test_data(input_test_data_file)
        svm_bc.set_param(svm_kernel='primal_hard_margin')
        svm_bc.init_W()
        W = svm_bc.train()
        print("\n訓練得出權重模型：")
        print(W)
        print("SVM Marging：")
        print(svm_bc.get_marge())

        test_data = '-3.53133797 5.23661847 -1'
        prediction = svm_bc.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        print("W 平均錯誤率（Ein）：")
        print(svm_bc.calculate_avg_error(svm_bc.train_X, svm_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(svm_bc.calculate_test_data_avg_error())
        print('-'*70)

        svm_bc = svm.BinaryClassifier()
        svm_bc.load_train_data(input_train_data_file)
        svm_bc.set_feature_transform('legendre', 3)
        svm_bc.load_test_data(input_test_data_file)
        svm_bc.set_param(svm_kernel='dual_hard_margin')
        svm_bc.init_W()
        W = svm_bc.train()
        print("\n訓練得出權重模型：")
        print(W)
        print("SVM Marging：")
        print(svm_bc.get_marge())
        print("Support Vectors")
        print(svm_bc.get_support_vectors())

        print("W 平均錯誤率（Ein）：")
        print(svm_bc.calculate_avg_error(svm_bc.train_X, svm_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(svm_bc.calculate_test_data_avg_error())
        print('-'*70)

    def test_polynomial_kernel_svm_binary_classifier(self):

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/non_linear_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/non_linear_test.dat')

        svm_bc = svm.BinaryClassifier()
        svm_bc.load_train_data(input_train_data_file)
        svm_bc.load_test_data(input_test_data_file)
        svm_bc.set_param(svm_kernel='polynomial_kernel', zeta=100, gamma=1, Q=3)
        svm_bc.init_W()
        W = svm_bc.train()
        print("\n訓練得出權重模型：")
        print(W)
        print("SVM Marging：")
        print(svm_bc.get_marge())
        print("Support Vectors")
        print(svm_bc.get_support_vectors())

        print("W 平均錯誤率（Ein）：")
        print(svm_bc.calculate_avg_error(svm_bc.train_X, svm_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(svm_bc.calculate_test_data_avg_error())
        print('-'*70)

    def test_gaussian_kernel_svm_binary_classifier(self):

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/non_linear_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/non_linear_test.dat')

        svm_bc = svm.BinaryClassifier()
        svm_bc.load_train_data(input_train_data_file)
        svm_bc.load_test_data(input_test_data_file)
        svm_bc.set_param(svm_kernel='gaussian_kernel', gamma=0.001)
        svm_bc.init_W()
        W = svm_bc.train()
        print("\n訓練得出權重模型：")
        print(W)
        print("SVM Marging：")
        print(svm_bc.get_marge())
        print("Support Vectors")
        print(svm_bc.get_support_vectors())

        print("W 平均錯誤率（Ein）：")
        print(svm_bc.calculate_avg_error(svm_bc.train_X, svm_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(svm_bc.calculate_test_data_avg_error())
        print('-'*70)

    def test_soft_polynomial_kernel_svm_binary_classifier(self):

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/overlap_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/overlap_test.dat')

        svm_bc = svm.BinaryClassifier()
        svm_bc.load_train_data(input_train_data_file)
        svm_bc.load_test_data(input_test_data_file)
        svm_bc.set_param(svm_kernel='soft_polynomial_kernel', zeta=0, gamma=1, Q=1, C=0.1)
        svm_bc.init_W()
        W = svm_bc.train()

        print("\n訓練得出權重模型：")
        print(W)
        print("SVM Marging：")
        print(svm_bc.get_marge())
        print("Support Vectors")
        print(svm_bc.get_support_vectors())

        print("W 平均錯誤率（Ein）：")
        print(svm_bc.calculate_avg_error(svm_bc.train_X, svm_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(svm_bc.calculate_test_data_avg_error())
        print('-'*70)

    def test_soft_gaussian_kernel_svm_binary_classifier(self):

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/overlap_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/overlap_test.dat')

        svm_bc = svm.BinaryClassifier()
        svm_bc.load_train_data(input_train_data_file)
        svm_bc.load_test_data(input_test_data_file)
        svm_bc.set_param(svm_kernel='soft_gaussian_kernel', gamma=1, C=1000)
        svm_bc.init_W()
        W = svm_bc.train()

        print("\n訓練得出權重模型：")
        print(W)
        print("SVM Marging：")
        print(svm_bc.get_marge())
        print("Support Vectors")
        print(svm_bc.get_support_vectors())

        print("W 平均錯誤率（Ein）：")
        print(svm_bc.calculate_avg_error(svm_bc.train_X, svm_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(svm_bc.calculate_test_data_avg_error())
        print('-'*70)

    def test_probabilistic_svm(self):

        #------------------------------------------------------------

        probabilistic = probabilistic_svm.ProbabilisticSVM()
        probabilistic.load_train_data()
        probabilistic.load_test_data()
        probabilistic.set_param()
        probabilistic.init_W()
        probabilistic.train()

        test_data = '0.26502 0.5486 0.971 0.19333 0.12207 0.81528 0.46743 0.45889 0.31004 0.3307 0.43078 0.50661 0.57281 0.052715 0.50443 0.78686 0.20099 0.85909 0.26772 0.13751 1'
        prediction = probabilistic.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(probabilistic.error_function(prediction['input_data_x'], prediction['input_data_y'], probabilistic.W))
        print("W 平均錯誤值（Ein）：")
        print(probabilistic.calculate_avg_error(probabilistic.train_X, probabilistic.train_Y, probabilistic.W))
        print("W 平均錯誤值（Eout）：")
        print(probabilistic.calculate_avg_error(probabilistic.test_X, probabilistic.test_Y, probabilistic.W))
        print('-'*70)

    def test_decision_stump_binary_classifier(self):

        #------------------------------------------------------------

        decision_stump_bc = decision_stump.BinaryClassifier()
        decision_stump_bc.load_train_data()
        decision_stump_bc.set_param()
        decision_stump_bc.init_W()
        decision_stump_bc.train()
        decision_stump_bc.load_test_data()

        print("\n訓練得出 sign：")
        print(decision_stump_bc.sign)
        print("訓練得出 feature index：")
        print(decision_stump_bc.feature_index)
        print("訓練得出 theta")
        print(decision_stump_bc.theta)
        print("平均錯誤率（Ein）：")
        print(decision_stump_bc.calculate_avg_error(decision_stump_bc.train_X, decision_stump_bc.train_Y, decision_stump_bc.W))
        print("W 平均錯誤率（Eout）：")
        print(decision_stump_bc.calculate_avg_error(decision_stump_bc.test_X, decision_stump_bc.test_Y, decision_stump_bc.W))
        print('-'*70)

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/decision_stump_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/decision_stump_test.dat')

        decision_tree_c = decision_tree.CART()
        decision_tree_c.load_train_data(input_train_data_file)
        decision_tree_c.set_param(learn_type='classifier', tree_height_limit=1)
        decision_tree_c.load_test_data(input_test_data_file)
        decision_tree_c.init_W()
        decision_tree_c.train()

        print("\n訓練得出 Decision Tree：")
        decision_tree_c.plot(decision_tree_c.decision_tree)

        print("W 平均錯誤率（Ein）：")
        print(decision_tree_c.calculate_avg_error(decision_tree_c.train_X, decision_tree_c.train_Y, decision_tree_c.W))
        print("W 平均錯誤率（Eout）：")
        print(decision_tree_c.calculate_test_data_avg_error())
        print('-'*70)

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
