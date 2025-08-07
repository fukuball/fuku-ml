# encoding=utf8

from test_base import BaseTestCase
import FukuML.KernelRidgeRegression as kernel_ridge_regression


class KernelRidgeRegressionTestCase(BaseTestCase):

    def test_kernel_ridge_regression(self):

        #------------------------------------------------------------

        kernel_ridge = kernel_ridge_regression.KernelRidgeRegression()
        kernel_ridge.load_train_data()
        kernel_ridge.load_test_data()
        kernel_ridge.set_param(lambda_p=pow(10, -3), gamma=1, C=0.1)
        kernel_ridge.init_W()
        kernel_ridge.train()

        print("\n訓練得出權重模型：")
        print(kernel_ridge.beta)

        test_data = '0.62771 0.11513 0.82235 0.14493 -1'
        prediction = kernel_ridge.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(kernel_ridge.error_function(prediction['prediction'], prediction['input_data_y']))
        print("W 平均錯誤值（Ein）：")
        print(kernel_ridge.calculate_avg_error(kernel_ridge.train_X, kernel_ridge.train_Y, kernel_ridge.W))
        print("W 平均錯誤值（Eout）：")
        print(kernel_ridge.calculate_avg_error(kernel_ridge.test_X, kernel_ridge.test_Y, kernel_ridge.W))
        print('-'*70)


if __name__ == '__main__':
    import unittest
    unittest.main()
