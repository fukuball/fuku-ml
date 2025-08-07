# encoding=utf8

from test_base import BaseTestCase
import FukuML.ProbabilisticSVM as probabilistic_svm


class ProbabilisticSVMTestCase(BaseTestCase):

    def test_probabilistic_svm(self):

        #------------------------------------------------------------

        probabilistic = probabilistic_svm.ProbabilisticSVM()
        probabilistic.load_train_data()
        probabilistic.load_test_data()
        probabilistic.set_param()
        probabilistic.init_W()
        probabilistic.train()

        test_data = '0.26502 0.5486 0.971 0.19333 0.12207 0.81528 0.46743 0.45889 0.31004 0.3307 0.43078 0.50661 0.57281 0.052715 0.50443 0.78686 0.20099 0.85909 0.26772 0.13751 1'
        prediction = probabilistic.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(probabilistic.error_function(prediction['input_data_x'], prediction['input_data_y'], probabilistic.W))
        print("W 平均錯誤值（Ein）：")
        print(probabilistic.calculate_avg_error(probabilistic.train_X, probabilistic.train_Y, probabilistic.W))
        print("W 平均錯誤值（Eout）：")
        print(probabilistic.calculate_avg_error(probabilistic.test_X, probabilistic.test_Y, probabilistic.W))
        print('-'*70)


if __name__ == '__main__':
    import unittest
    unittest.main()
