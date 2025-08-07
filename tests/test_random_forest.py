# encoding=utf8

import os
from test_base import BaseTestCase
import FukuML.RandomForest as random_forest


class RandomForestTestCase(BaseTestCase):

    def test_random_forest(self):

        random_forest_c = random_forest.RandomForest()
        random_forest_c.load_train_data()
        random_forest_c.set_param(learn_type='classifier', blending_model_num=3)
        random_forest_c.load_test_data()
        random_forest_c.init_W()
        random_forest_c.train()

        test_data = '0.94544 0.42842 0.79833 0.16244 -1'
        prediction = random_forest_c.prediction(test_data)
        self.assertEqual(float(prediction['input_data_y']), float(prediction['prediction']))

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        print("W 平均錯誤率（Ein）：")
        print(random_forest_c.calculate_avg_error(random_forest_c.train_X, random_forest_c.train_Y, random_forest_c.W))
        print("W 平均錯誤率（Eout）：")
        print(random_forest_c.calculate_test_data_avg_error())
        print('-'*70)

        #------------------------------------------------------------

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), '../FukuML/dataset/linear_regression_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), '../FukuML/dataset/linear_regression_test.dat')

        random_forest_r = random_forest.RandomForest()
        random_forest_r.load_train_data(input_train_data_file)
        random_forest_r.set_param(learn_type='regression', blending_model_num=3)
        random_forest_r.load_test_data(input_test_data_file)
        random_forest_r.init_W()
        random_forest_r.train()

        test_data = '61.9 56 348'
        prediction = random_forest_r.prediction(test_data)

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        print("W 平均錯誤值（Ein）：")
        print(random_forest_r.calculate_avg_error(random_forest_r.train_X, random_forest_r.train_Y, random_forest_r.W))
        print("W 平均錯誤值（Eout）：")
        print(random_forest_r.calculate_test_data_avg_error())
        print('-'*70)


if __name__ == '__main__':
    import unittest
    unittest.main()
