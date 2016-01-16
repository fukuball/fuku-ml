#encoding=utf8

import os
import unittest
import FukuML.PLA as pla


class PLATestCase(unittest.TestCase):

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

        input_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/pla_train.dat')
        pla.load_train_data(input_data_file)
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

        print("使用 Random Cycle alpha=0.5 Step correction：")

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
