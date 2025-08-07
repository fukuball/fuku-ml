# encoding=utf8

from test_base import BaseTestCase
import FukuML.SupportVectorRegression as svr


class SupportVectorRegressionTestCase(BaseTestCase):

    def test_support_vector_regression(self):

        #------------------------------------------------------------

        sv_regression = svr.SupportVectorRegression()
        sv_regression.load_train_data()
        sv_regression.load_test_data()
        sv_regression.set_param(gamma=1, C=1, epsilon=0.1)
        sv_regression.init_W()
        sv_regression.train()

        test_data = '0.62771 0.11513 0.82235 0.14493 -1'
        prediction = sv_regression.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(sv_regression.error_function(prediction['prediction'], prediction['input_data_y']))
        print("W 平均錯誤值（Ein）：")
        print(sv_regression.calculate_avg_error(sv_regression.train_X, sv_regression.train_Y, sv_regression.W))
        print("W 平均錯誤值（Eout）：")
        print(sv_regression.calculate_avg_error(sv_regression.test_X, sv_regression.test_Y, sv_regression.W))
        print('-'*70)


if __name__ == '__main__':
    import unittest
    unittest.main()
