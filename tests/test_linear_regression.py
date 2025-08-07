# encoding=utf8

from test_base import BaseTestCase
import FukuML.LinearRegression as linear_regression


class LinearRegressionTestCase(BaseTestCase):
    
    def test_linear_regression(self):

        #------------------------------------------------------------

        linear = linear_regression.LinearRegression()
        linear.load_train_data()
        linear.load_test_data()
        linear.set_param()
        linear.init_W()
        W = linear.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0.62771 0.11513 0.82235 0.14493 -1'
        prediction = linear.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(linear.error_function(prediction['prediction'], prediction['input_data_y']))
        print("W 平均錯誤值（Ein）：")
        print(linear.calculate_avg_error(linear.train_X, linear.train_Y, W))
        print("W 平均錯誤值（Eout）：")
        print(linear.calculate_avg_error(linear.test_X, linear.test_Y, W))
        print('-'*70)


if __name__ == '__main__':
    import unittest
    unittest.main()