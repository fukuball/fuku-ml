#encoding=utf8

import os
import unittest
import pickle
import FukuML.PLA as pla
import FukuML.PocketPLA as pocket


class FukuMLTestCase(unittest.TestCase):

    def test_pla_binary_classifier(self):

        #------------------------------------------------------------

        pla_bc = pla.BinaryClassifier()
        pla_bc.load_train_data()
        pla_bc.init_W()
        W = pla_bc.train()

        print("\n訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pla_bc.tune_times)
        print('-'*70)

        test_data = '0.97681 0.10723 0.64385 0.29556 1'
        prediction = pla_bc.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print('-'*70)

        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        test_data = '0.15654 0.75584 0.01122 0.42598 -1'
        prediction = pla_bc.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print('-'*70)

        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pla_binary_train.dat')
        pla_bc.load_train_data(input_train_data_file)
        pla_bc.init_W()
        W = pla_bc.train()
        print("測試載入 Custom Dataset")
        print("訓練得出權重模型：")
        print(W)
        print('-'*70)

        #------------------------------------------------------------

        print("使用 Random Cycle：")

        pla_bc.init_W()
        W = pla_bc.train('random')

        print("訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pla_bc.tune_times)
        print('-'*70)

        test_data = '0.97681 0.10723 0.64385 0.29556 1'
        prediction = pla_bc.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print('-'*70)

        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        #------------------------------------------------------------

        print("使用 Random Cycle alpha=0.5 step correction：")

        pla_bc.init_W()
        W = pla_bc.train('random', 0.5)

        print("訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pla_bc.tune_times)
        print('-'*70)

        test_data = '0.97681 0.10723 0.64385 0.29556 1'
        prediction = pla_bc.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print('-'*70)

        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        serialized_pla_bc = pla_bc.serialize()
        deserialized_pla_bc = pickle.loads(serialized_pla_bc)
        self.assertTrue((pla_bc.train_X == deserialized_pla_bc.train_X).all())

    def test_pocket_pla_binary_classifier(self):

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pocket_pla_binary_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pocket_pla_binary_test.dat')

        pla_bc = pla.BinaryClassifier()
        pla_bc.load_train_data(input_train_data_file)
        pla_bc.init_W()
        W = pla_bc.train()
        pla_bc.load_test_data(input_test_data_file)

        print("\n訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pla_bc.tune_times)
        print("Dataset 不是線性可分的，所以要使用 Pocket PLA.")
        print("W 平均錯誤率：")
        print(pla_bc.calculate_avg_error(pla_bc.test_X, pla_bc.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        pocket_bc = pocket.BinaryClassifier()
        pocket_bc.load_train_data()
        pocket_bc.init_W()
        W = pocket_bc.train(50)
        pocket_bc.load_test_data()

        print("訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pocket_bc.tune_times)
        print("W 效果改善次數：")
        print(pocket_bc.put_in_pocket_times)
        print("W 平均錯誤率：")
        print(pocket_bc.calculate_avg_error(pocket_bc.test_X, pocket_bc.test_Y, W))
        print('-'*70)

        test_data = '0.62771 0.11513 0.82235 0.14493 -1'
        prediction = pocket_bc.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print('-'*70)

        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        serialized_pocket_bc = pocket_bc.serialize()
        deserialized_pocket_bc = pickle.loads(serialized_pocket_bc)
        self.assertTrue((pocket_bc.train_X == deserialized_pocket_bc.train_X).all())


if __name__ == '__main__':

    unittest.main(verbosity=2)
