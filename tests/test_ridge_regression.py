# encoding=utf8

from test_base import BaseTestCase
import FukuML.RidgeRegression as ridge_regression


class RidgeRegressionTestCase(BaseTestCase):

    def test_ridge_regression(self):

        #------------------------------------------------------------

        ridge = ridge_regression.RidgeRegression()
        ridge.load_train_data()
        ridge.load_test_data()
        ridge.set_param(lambda_p=pow(10, -3))
        ridge.init_W()
        W = ridge.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0.62771 0.11513 0.82235 0.14493 -1'
        prediction = ridge.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(ridge.error_function(prediction['prediction'], prediction['input_data_y']))
        print("W 平均錯誤值（Ein）：")
        print(ridge.calculate_avg_error(ridge.train_X, ridge.train_Y, W))
        print("W 平均錯誤值（Eout）：")
        print(ridge.calculate_avg_error(ridge.test_X, ridge.test_Y, W))
        print('-'*70)


if __name__ == '__main__':
    import unittest
    unittest.main()
