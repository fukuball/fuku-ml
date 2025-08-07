# encoding=utf8

from test_base import BaseTestCase
import FukuML.RidgeRegression as ridge_regression
import FukuML.Utility as utility


class RidgeRegressionBinaryClassifierTestCase(BaseTestCase):

    def test_ridge_regression_binary_classifier(self):

        #------------------------------------------------------------

        cross_validator = utility.CrossValidator()

        lambda_p_set = [i for i in range(2, -11, -1)]

        for lambda_p in lambda_p_set:
            ridge_bc = ridge_regression.BinaryClassifier()
            ridge_bc.load_train_data()
            ridge_bc.set_param(lambda_p=pow(10, lambda_p))
            cross_validator.add_model(ridge_bc)

        print("\n10 fold cross validation：")
        avg_errors = cross_validator.excute()

        print("\n各模型驗證平均錯誤：")
        print(avg_errors)
        print("\n最小平均錯誤率：")
        print(cross_validator.get_min_avg_error())

        print("\n取得最佳模型：")
        best_model = cross_validator.get_best_model()
        print(best_model)

        best_model.load_test_data()
        best_model.init_W()
        W = best_model.train()

        print("\n訓練得出權重模型：")
        print(W)

        test_data = '0.402041 0.402048 -1'
        prediction = best_model.prediction(test_data)

        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print("W 平均錯誤率（Ein）：")
        print(best_model.calculate_avg_error(best_model.train_X, best_model.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(best_model.calculate_avg_error(best_model.test_X, best_model.test_Y, W))
        print('-'*70)


if __name__ == '__main__':
    import unittest
    unittest.main()
