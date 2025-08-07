# encoding=utf8

from test_base import BaseTestCase
import FukuML.LinearRegression as linear_regression


class LinearRegressionBinaryClassifierTestCase(BaseTestCase):
    
    def test_linear_regression_binary_classifier(self):

        #------------------------------------------------------------

        linear_bc = linear_regression.BinaryClassifier()
        linear_bc.load_train_data()
        linear_bc.load_test_data()
        linear_bc.set_param()
        linear_bc.init_W()
        W = linear_bc.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0.62771 0.11513 0.82235 0.14493 -1'
        prediction = linear_bc.prediction(test_data)

        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("W 平均錯誤率（Ein）：")
        print(linear_bc.calculate_avg_error(linear_bc.train_X, linear_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(linear_bc.calculate_avg_error(linear_bc.test_X, linear_bc.test_Y, W))
        print('-'*70)


if __name__ == '__main__':
    import unittest
    unittest.main()