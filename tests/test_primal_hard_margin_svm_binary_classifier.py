# encoding=utf8

from test_base import BaseTestCase
import FukuML.SupportVectorMachine as svm


class PrimalHardMarginSVMBinaryClassifierTestCase(BaseTestCase):

    def test_primal_hard_margin_svm_binary_classifier(self):

        #------------------------------------------------------------

        svm_bc = svm.BinaryClassifier()
        svm_bc.load_train_data()
        svm_bc.load_test_data()
        svm_bc.set_param(svm_kernel='primal_hard_margin')
        svm_bc.init_W()
        W = svm_bc.train()
        print("\n訓練得出權重模型：")
        print(W)
        print("SVM Marging：")
        print(svm_bc.get_marge())

        test_data = '0.97681 0.10723 0.64385 0.29556 1'
        prediction = svm_bc.prediction(test_data)
        self.assertEqual(prediction['input_data_y'], prediction['prediction'])

        print("測試資料 x：")
        print(prediction['input_data_x'])
        print("測試資料 y：")
        print(prediction['input_data_y'])
        print("預測結果：")
        print(prediction['prediction'])
        print('-'*70)

        test_data = '0.15654 0.75584 0.01122 0.42598 -1'
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
        print('-'*70)


if __name__ == '__main__':
    import unittest
    unittest.main()
