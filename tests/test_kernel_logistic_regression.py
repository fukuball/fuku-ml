# encoding=utf8

from test_base import BaseTestCase
import FukuML.KernelLogisticRegression as kernel_logistic_regression


class KernelLogisticRegressionTestCase(BaseTestCase):

    def test_kernel_logistic_regression(self):

        #------------------------------------------------------------

        kernel_logistic = kernel_logistic_regression.KernelLogisticRegression()
        kernel_logistic.load_train_data()
        kernel_logistic.load_test_data()
        kernel_logistic.set_param(feed_mode='stochastic', step_eta=0.126, updates=2000, lambda_p=0.0001, gamma=1, C=0.1)
        kernel_logistic.init_W()
        kernel_logistic.train()

        print("\n訓練得出權重模型：")
        print(kernel_logistic.beta)

        test_data = '0.26502 0.5486 0.971 0.19333 0.12207 0.81528 0.46743 0.45889 0.31004 0.3307 0.43078 0.50661 0.57281 0.052715 0.50443 0.78686 0.20099 0.85909 0.26772 0.13751 1'
        prediction = kernel_logistic.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(kernel_logistic.error_function(prediction['input_data_x'], prediction['input_data_y'], kernel_logistic.W))
        print("W 平均錯誤值（Ein）：")
        print(kernel_logistic.calculate_avg_error(kernel_logistic.train_X, kernel_logistic.train_Y, kernel_logistic.W))
        print("W 平均錯誤值（Eout）：")
        print(kernel_logistic.calculate_avg_error(kernel_logistic.test_X, kernel_logistic.test_Y, kernel_logistic.W))
        print('-'*70)


if __name__ == '__main__':
    import unittest
    unittest.main()
