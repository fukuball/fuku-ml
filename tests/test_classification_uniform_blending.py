# encoding=utf8

import os
from test_base import BaseTestCase
import FukuML.PLA as pla
import FukuML.PocketPLA as pocket
import FukuML.LinearRegression as linear_regression
import FukuML.Blending as blending


class ClassificationUniformBlendingTestCase(BaseTestCase):

    def test_classification_uniform_blending(self):

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), '../FukuML/dataset/pocket_pla_binary_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), '../FukuML/dataset/pocket_pla_binary_test.dat')

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


if __name__ == '__main__':
    import unittest
    unittest.main()
