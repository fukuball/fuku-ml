# encoding=utf8

import os
from test_base import BaseTestCase
import FukuML.PLA as pla
import FukuML.PocketPLA as pocket
import FukuML.LinearRegression as linear_regression
import FukuML.LogisticRegression as logistic_regression
import FukuML.Utility as utility


class BinaryClassificationCrossValidationTestCase(BaseTestCase):

    def test_binary_classification_cross_validation(self):

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), '../FukuML/dataset/pocket_pla_binary_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), '../FukuML/dataset/pocket_pla_binary_test.dat')

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


if __name__ == '__main__':
    import unittest
    unittest.main()
