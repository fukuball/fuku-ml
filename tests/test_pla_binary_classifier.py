# encoding=utf8

import os
import sys
import warnings
import numpy as np

# Add parent directory to path so we can import FukuML
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FukuML.Utility as utility
import FukuML.PLA as pla
from test_base import BaseTestCase


class PLABinaryClassifierTestCase(BaseTestCase):

    def test_pla_binary_classifier(self):
        #------------------------------------------------------------

        pla_bc = pla.BinaryClassifier()
        pla_bc.load_train_data()
        pla_bc.set_param()
        pla_bc.init_W()
        W = pla_bc.train()

        print("\n訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pla_bc.tune_times)
        print('-'*70)

        test_data = '0.97681 0.10723 0.64385 0.29556 1'
        prediction = pla_bc.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print('-'*70)

        test_data = '0.15654 0.75584 0.01122 0.42598 -1'
        prediction = pla_bc.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("W 平均錯誤率（Ein）：")
        print(pla_bc.calculate_avg_error(pla_bc.train_X, pla_bc.train_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        print("使用 Linear Regression 加速器：")

        pla_bc = pla.BinaryClassifier()
        pla_bc.load_train_data()
        pla_bc.set_param()
        pla_bc.init_W('linear_regression_accelerator')
        W = pla_bc.train()

        print("\n訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pla_bc.tune_times)
        print('-'*70)

        test_data = '0.97681 0.10723 0.64385 0.29556 1'
        prediction = pla_bc.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print('-'*70)

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'FukuML/dataset/pla_binary_train.dat')
        pla_bc.load_train_data(input_train_data_file)
        pla_bc.set_param()
        pla_bc.init_W()
        W = pla_bc.train()
        print("測試載入 Custom Dataset")
        print("訓練得出權重模型：")
        print(W)
        print('-'*70)

        #------------------------------------------------------------

        print("使用 Random Cycle：")

        pla_bc.set_param(loop_mode='random')
        pla_bc.init_W()
        W = pla_bc.train()

        print("訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pla_bc.tune_times)
        print('-'*70)

        test_data = '0.97681 0.10723 0.64385 0.29556 1'
        prediction = pla_bc.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print('-'*70)

        #------------------------------------------------------------

        print("使用 Random Cycle alpha=0.5 step correction：")

        pla_bc.set_param(loop_mode='random', step_alpha=0.5)
        pla_bc.init_W()
        W = pla_bc.train()

        print("訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pla_bc.tune_times)
        print('-'*70)

        test_data = '0.97681 0.10723 0.64385 0.29556 1'
        prediction = pla_bc.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print('-'*70)

        serialized_pla_bc = utility.Serializer.serialize(pla_bc)
        deserialized_pla_bc = utility.Serializer.deserialize(serialized_pla_bc)
        self.assertTrue((pla_bc.train_X == deserialized_pla_bc.train_X).all())

if __name__ == '__main__':
    import unittest
    unittest.main()