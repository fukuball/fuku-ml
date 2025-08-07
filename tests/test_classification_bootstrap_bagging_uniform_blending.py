# encoding=utf8

import os
from test_base import BaseTestCase
import FukuML.PLA as pla
import FukuML.Utility as utility
import FukuML.Blending as blending


class ClassificationBootstrapBaggingUniformBlendingTestCase(BaseTestCase):

    def test_classification_bootstrap_bagging_uniform_blending(self):

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), '../FukuML/dataset/pocket_pla_binary_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), '../FukuML/dataset/pocket_pla_binary_test.dat')

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


if __name__ == '__main__':
    import unittest
    unittest.main()
