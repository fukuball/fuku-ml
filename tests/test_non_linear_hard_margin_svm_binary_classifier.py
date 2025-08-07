# encoding=utf8

import os
from test_base import BaseTestCase
import FukuML.SupportVectorMachine as svm


class NonLinearHardMarginSVMBinaryClassifierTestCase(BaseTestCase):

    def test_non_linear_hard_margin_svm_binary_classifier(self):

        input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), '../FukuML/dataset/non_linear_train.dat')
        input_test_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), '../FukuML/dataset/non_linear_test.dat')

        svm_bc = svm.BinaryClassifier()
        svm_bc.load_train_data(input_train_data_file)
        svm_bc.set_feature_transform('legendre', 3)
        svm_bc.load_test_data(input_test_data_file)
        svm_bc.set_param(svm_kernel='primal_hard_margin')
        svm_bc.init_W()
        W = svm_bc.train()
        print("\n訓練得出權重模型：")
        print(W)
        print("SVM Marging：")
        print(svm_bc.get_marge())

        test_data = '-3.53133797 5.23661847 -1'
        prediction = svm_bc.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])

        print("W 平均錯誤率（Ein）：")
        print(svm_bc.calculate_avg_error(svm_bc.train_X, svm_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(svm_bc.calculate_test_data_avg_error())
        print('-'*70)

        svm_bc = svm.BinaryClassifier()
        svm_bc.load_train_data(input_train_data_file)
        svm_bc.set_feature_transform('legendre', 3)
        svm_bc.load_test_data(input_test_data_file)
        svm_bc.set_param(svm_kernel='dual_hard_margin')
        svm_bc.init_W()
        W = svm_bc.train()
        print("\n訓練得出權重模型：")
        print(W)
        print("SVM Marging：")
        print(svm_bc.get_marge())
        print("Support Vectors")
        print(svm_bc.get_support_vectors())

        print("W 平均錯誤率（Ein）：")
        print(svm_bc.calculate_avg_error(svm_bc.train_X, svm_bc.train_Y, W))
        print("W 平均錯誤率（Eout）：")
        print(svm_bc.calculate_test_data_avg_error())
        print('-'*70)


if __name__ == '__main__':
    import unittest
    unittest.main()
