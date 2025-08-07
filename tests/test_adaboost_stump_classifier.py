# encoding=utf8

from test_base import BaseTestCase
import FukuML.AdaBoostStump as adaboost_stump


class AdaBoostStumpClassifierTestCase(BaseTestCase):

    def test_adaboost_stump_classifier(self):

        #------------------------------------------------------------

        adaboost_stump_bc = adaboost_stump.BinaryClassifier()
        adaboost_stump_bc.load_train_data()
        adaboost_stump_bc.load_test_data()
        adaboost_stump_bc.set_param(run_t=10)
        adaboost_stump_bc.init_W()
        adaboost_stump_bc.train()

        test_data = '-9.706 1.392 6.562 -6.543 -1.980 -6.261 -6.067 1.254 -1.071 1'
        prediction = adaboost_stump_bc.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        print("平均錯誤率（Ein）：")
        print(adaboost_stump_bc.calculate_avg_error(adaboost_stump_bc.train_X, adaboost_stump_bc.train_Y, adaboost_stump_bc.W))
        print("平均錯誤率（Eout）：")
        print(adaboost_stump_bc.calculate_avg_error(adaboost_stump_bc.test_X, adaboost_stump_bc.test_Y, adaboost_stump_bc.W))
        print('-'*70)


if __name__ == '__main__':
    import unittest
    unittest.main()
