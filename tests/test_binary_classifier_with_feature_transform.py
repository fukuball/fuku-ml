# encoding=utf8

import os
from test_base import BaseTestCase
import FukuML.PLA as pla
import FukuML.PocketPLA as pocket
import FukuML.LinearRegression as linear_regression
import FukuML.LogisticRegression as logistic_regression


class BinaryClassifierWithFeatureTransformTestCase(BaseTestCase):

    def test_binary_classifier_with_feature_transform(self):

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), '../FukuML/dataset/pocket_pla_binary_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), '../FukuML/dataset/pocket_pla_binary_test.dat')

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


if __name__ == '__main__':
    import unittest
    unittest.main()
