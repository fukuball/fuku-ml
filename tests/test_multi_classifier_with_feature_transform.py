# encoding=utf8

import os
from test_base import BaseTestCase
import FukuML.PLA as pla
import FukuML.PocketPLA as pocket
import FukuML.LinearRegression as linear_regression
import FukuML.LogisticRegression as logistic_regression


class MultiClassifierWithFeatureTransformTestCase(BaseTestCase):

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


if __name__ == '__main__':
    import unittest
    unittest.main()
