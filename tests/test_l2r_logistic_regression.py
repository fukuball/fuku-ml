# encoding=utf8

from test_base import BaseTestCase
import FukuML.L2RLogisticRegression as l2r_logistic_regression


class L2RLogisticRegressionTestCase(BaseTestCase):

    def test_l2r_logistic_regression(self):

        #------------------------------------------------------------

        l2r_logistic = l2r_logistic_regression.L2RLogisticRegression()
        l2r_logistic.load_train_data()
        l2r_logistic.load_test_data()
        l2r_logistic.set_param(feed_mode='batch', step_eta=0.126, updates=2000, lambda_p=0.0001)
        l2r_logistic.init_W()
        W = l2r_logistic.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0.26502 0.5486 0.971 0.19333 0.12207 0.81528 0.46743 0.45889 0.31004 0.3307 0.43078 0.50661 0.57281 0.052715 0.50443 0.78686 0.20099 0.85909 0.26772 0.13751 1'
        prediction = l2r_logistic.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(l2r_logistic.error_function(prediction['input_data_x'], prediction['input_data_y'], W))
        print("W 平均錯誤值（Ein）：")
        print(l2r_logistic.calculate_avg_error(l2r_logistic.train_X, l2r_logistic.train_Y, W))
        print("W 平均錯誤值（Eout）：")
        print(l2r_logistic.calculate_avg_error(l2r_logistic.test_X, l2r_logistic.test_Y, W))
        print('-'*70)


if __name__ == '__main__':
    import unittest
    unittest.main()
