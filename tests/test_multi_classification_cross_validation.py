# encoding=utf8

import os
from test_base import BaseTestCase
import FukuML.PLA as pla
import FukuML.PocketPLA as pocket
import FukuML.Utility as utility


class MultiClassificationCrossValidationTestCase(BaseTestCase):

    def test_multi_classification_cross_validation(self):

        cross_validator = utility.CrossValidator()

        pla_mc = pla.MultiClassifier()
        pla_mc.load_train_data()
        pla_mc.set_param()
        pocket_mc = pocket.MultiClassifier()
        pocket_mc.load_train_data()
        pocket_mc.set_param()

        print("\n10 fold cross validation：")

        cross_validator.add_model(pla_mc)
        cross_validator.add_model(pocket_mc)
        avg_errors = cross_validator.excute()

        print("\n各模型驗證平均錯誤：")
        print(avg_errors)
        print("\n最小平均錯誤率：")
        print(cross_validator.get_min_avg_error())

        print("\n取得最佳模型：")
        best_model = cross_validator.get_best_model()
        print(best_model)
        best_model.init_W()
        W = best_model.train()
        best_model.load_test_data()

        print("\n訓練得出權重模型：")
        print(W)
        print("W 平均錯誤率（Ein）：")
        print(best_model.calculate_avg_error_all_class(best_model.train_X, best_model.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(best_model.calculate_avg_error_all_class(best_model.test_X, best_model.test_Y, W))
        print('-'*70)


if __name__ == '__main__':
    import unittest
    unittest.main()
