#encoding=utf8

import os
import unittest
import FukuML.PLA as pla
import FukuML.PocketPLA as pocket


class FukuMLTestCase(unittest.TestCase):

    def test_pocket_pla(self):

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pocket_pla_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pocket_pla_test.dat')

        pla.load_train_data(input_train_data_file)
        pla.init_W()
        W = pla.train()
        pla.load_test_data(input_test_data_file)

        print("\n訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pla.tune_times)
        print("Dataset 不是線性可分的，所以要使用 Pocket PLA.")
        print("W 平均錯誤率：")
        print(pocket.calculate_avg_error(pla.test_X, pla.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        pocket.load_train_data()
        pocket.init_W()
        W = pocket.train(50)
        pocket.load_test_data()

        print("訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pocket.tune_times)
        print("W 效果改善次數：")
        print(pocket.put_in_pocket_times)
        print("W 平均錯誤率：")
        print(pocket.calculate_avg_error(pocket.test_X, pocket.test_Y, W))
        print('-'*70)

        test_data = '0.62771 0.11513 0.82235 0.14493 -1'
        prediction = pocket.prediction(test_data)

        print("測試資料 x：")
        print(pocket.test_data_x)
        print("測試資料 y：")
        print(pocket.test_data_y)
        print("預測結果：")
        print(prediction)
        print('-'*70)

        self.assertEqual(pocket.test_data_y, prediction)

    def test_pla(self):

        #------------------------------------------------------------

        pla.load_train_data()
        pla.init_W()
        W = pla.train()

        print("\n訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pla.tune_times)
        print('-'*70)

        test_data = '0.97681 0.10723 0.64385 0.29556 1'
        prediction = pla.prediction(test_data)

        print("測試資料 x：")
        print(pla.test_data_x)
        print("測試資料 y：")
        print(pla.test_data_y)
        print("預測結果：")
        print(prediction)
        print('-'*70)

        self.assertEqual(pla.test_data_y, prediction)

        test_data = '0.15654 0.75584 0.01122 0.42598 -1'
        prediction = pla.prediction(test_data)

        print("測試資料 x：")
        print(pla.test_data_x)
        print("測試資料 y：")
        print(pla.test_data_y)
        print("預測結果：")
        print(prediction)
        print('-'*70)

        self.assertEqual(pla.test_data_y, prediction)

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pla_train.dat')
        pla.load_train_data(input_train_data_file)
        pla.init_W()
        W = pla.train()
        print("測試載入 Custom Dataset")
        print("訓練得出權重模型：")
        print(W)
        print('-'*70)

        #------------------------------------------------------------

        print("使用 Random Cycle：")

        pla.init_W()
        W = pla.train('random')

        print("訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pla.tune_times)
        print('-'*70)

        test_data = '0.97681 0.10723 0.64385 0.29556 1'
        prediction = pla.prediction(test_data)

        print("測試資料 x：")
        print(pla.test_data_x)
        print("測試資料 y：")
        print(pla.test_data_y)
        print("預測結果：")
        print(prediction)
        print('-'*70)

        self.assertEqual(pla.test_data_y, prediction)

        #------------------------------------------------------------

        print("使用 Random Cycle alpha=0.5 step correction：")

        pla.init_W()
        W = pla.train('random', 0.5)

        print("訓練得出權重模型：")
        print(W)
        print("W 更新次數：")
        print(pla.tune_times)
        print('-'*70)

        test_data = '0.97681 0.10723 0.64385 0.29556 1'
        prediction = pla.prediction(test_data)

        print("測試資料 x：")
        print(pla.test_data_x)
        print("測試資料 y：")
        print(pla.test_data_y)
        print("預測結果：")
        print(prediction)
        print('-'*70)

        self.assertEqual(pla.test_data_y, prediction)

        #------------------------------------------------------------

        tune_times_sum = 0
        for i in range(2000):
            pla.init_W()
            W = pla.train('random')
            tune_times_sum += pla.tune_times

        print("W 平均更新次數：")
        print(tune_times_sum/2000)

        #------------------------------------------------------------

        tune_times_sum = 0
        for i in range(2000):
            pla.init_W()
            W = pla.train('random', 0.5)
            tune_times_sum += pla.tune_times

        print("W 平均更新次數：")
        print(tune_times_sum/2000)


if __name__ == '__main__':

    unittest.main(verbosity=2)
