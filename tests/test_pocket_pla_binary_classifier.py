# encoding=utf8

import os
import sys

# Add parent directory to path so we can import FukuML
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FukuML.PLA as pla
import FukuML.PocketPLA as pocket
import FukuML.Utility as utility
from test_base import BaseTestCase


class PocketPLABinaryClassifierTestCase(BaseTestCase):

    def test_pocket_pla_binary_classifier(self):
        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'FukuML/dataset/pocket_pla_binary_train.dat')
        input_test_data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'FukuML/dataset/pocket_pla_binary_test.dat')

        pla_bc = pla.BinaryClassifier()
        pla_bc.load_train_data(input_train_data_file)
        pla_bc.set_param()
        pla_bc.init_W()
        W = pla_bc.train()
        pla_bc.load_test_data(input_test_data_file)

        print("\n訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pla_bc.tune_times)
        print("Dataset 不是線性可分的，所以要使用 Pocket PLA.")
        print("W 平均錯誤率（Ein）：")
        print(pla_bc.calculate_avg_error(pla_bc.train_X, pla_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(pla_bc.calculate_avg_error(pla_bc.test_X, pla_bc.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        pocket_bc = pocket.BinaryClassifier()
        pocket_bc.load_train_data()
        pocket_bc.set_param(loop_mode='naive_cycle', step_alpha=1, updates=50)
        pocket_bc.init_W()
        W = pocket_bc.train()
        pocket_bc.load_test_data()

        print("訓練得出權重模型：")
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

        test_data = '0.62771 0.11513 0.82235 0.14493 -1'
        prediction = pocket_bc.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print('-'*70)

        serialized_pocket_bc = utility.Serializer.serialize(pocket_bc)
        deserialized_pocket_bc = utility.Serializer.deserialize(serialized_pocket_bc)
        self.assertTrue((pocket_bc.train_X == deserialized_pocket_bc.train_X).all())

        #------------------------------------------------------------

        print("使用 Linear Regression 加速器：")

        pocket_bc = pocket.BinaryClassifier()
        pocket_bc.load_train_data()
        pocket_bc.set_param(loop_mode='naive_cycle', step_alpha=1, updates=50)
        pocket_bc.init_W('linear_regression_accelerator')
        W = pocket_bc.train()
        pocket_bc.load_test_data()

        print("訓練得出權重模型：")
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

if __name__ == '__main__':
    import unittest
    unittest.main()