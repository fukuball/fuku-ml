# encoding=utf8

from test_base import BaseTestCase
import FukuML.LogisticRegression as logistic_regression


class LogisticRegressionTestCase(BaseTestCase):

    def test_logistic_regression(self):

        #------------------------------------------------------------

        logistic = logistic_regression.LogisticRegression()
        logistic.load_train_data()
        logistic.load_test_data()
        logistic.set_param()
        logistic.init_W()
        W = logistic.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0.26502 0.5486 0.971 0.19333 0.12207 0.81528 0.46743 0.45889 0.31004 0.3307 0.43078 0.50661 0.57281 0.052715 0.50443 0.78686 0.20099 0.85909 0.26772 0.13751 1'
        prediction = logistic.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(logistic.error_function(prediction['input_data_x'], prediction['input_data_y'], W))
        print("W 平均錯誤值（Ein）：")
        print(logistic.calculate_avg_error(logistic.train_X, logistic.train_Y, W))
        print("W 平均錯誤值（Eout）：")
        print(logistic.calculate_avg_error(logistic.test_X, logistic.test_Y, W))
        print('-'*70)

        #------------------------------------------------------------

        print("使用 Linear Regression 加速器：")

        logistic = logistic_regression.LogisticRegression()
        logistic.load_train_data()
        logistic.load_test_data()
        logistic.set_param()
        logistic.init_W('linear_regression_accelerator')
        W = logistic.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0.26502 0.5486 0.971 0.19333 0.12207 0.81528 0.46743 0.45889 0.31004 0.3307 0.43078 0.50661 0.57281 0.052715 0.50443 0.78686 0.20099 0.85909 0.26772 0.13751 1'
        prediction = logistic.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(logistic.error_function(prediction['input_data_x'], prediction['input_data_y'], W))
        print("W 平均錯誤值（Ein）：")
        print(logistic.calculate_avg_error(logistic.train_X, logistic.train_Y, W))
        print("W 平均錯誤值（Eout）：")
        print(logistic.calculate_avg_error(logistic.test_X, logistic.test_Y, W))
        print('-'*70)


if __name__ == '__main__':
    import unittest
    unittest.main()
