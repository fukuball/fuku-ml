# encoding=utf8

from test_base import BaseTestCase
import FukuML.LeastSquaresSVM as least_squares_svm


class LeastSquaresSVMBinaryClassifierTestCase(BaseTestCase):

    def test_least_squares_svm_binary_classifier(self):

        #------------------------------------------------------------

        least_squares = least_squares_svm.BinaryClassifier()
        least_squares.load_train_data()
        least_squares.load_test_data()
        least_squares.set_param(lambda_p=pow(10, -3), gamma=1, C=0.1)
        least_squares.init_W()
        least_squares.train()

        print("\n訓練得出權重模型：")
        print(least_squares.beta)

        test_data = '0.62771 0.11513 0.82235 0.14493 -1'
        prediction = least_squares.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("錯誤評估：")
        print(least_squares.error_function(prediction['prediction'], prediction['input_data_y']))
        print("W 平均錯誤值（Ein）：")
        print(least_squares.calculate_avg_error(least_squares.train_X, least_squares.train_Y, least_squares.W))
        print("W 平均錯誤值（Eout）：")
        print(least_squares.calculate_avg_error(least_squares.test_X, least_squares.test_Y, least_squares.W))
        print('-'*70)


if __name__ == '__main__':
    import unittest
    unittest.main()
