# encoding=utf8

import os
from test_base import BaseTestCase
import FukuML.Utility as utility
import FukuML.LinearRegression as linear_regression
import FukuML.Blending as blending


class RegressionUniformTestCase(BaseTestCase):

    def test_regression_uniform_blending(self):

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), '../FukuML/dataset/pocket_pla_binary_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), '../FukuML/dataset/pocket_pla_binary_test.dat')

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


if __name__ == '__main__':
    import unittest
    unittest.main()
