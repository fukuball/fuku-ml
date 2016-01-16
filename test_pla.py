#encoding=utf8

import unittest
import FukuML.PLA as pla
import numpy as np

class PLATestCase(unittest.TestCase):

    def test_train_pla(self):
        pla.load_train_data()
        pla.init_W()
        W = pla.train()

        print u"\n訓練得出權重模型："
        print W
        print('-'*70)

        test_data_x = np.array([1, 0.97681, 0.10723, 0.64385, 0.29556])
        test_data_y = 1.0
        prediction = np.sign(np.dot(test_data_x, W))
        self.assertEqual(prediction, test_data_y)

        print u"測試資料 x："
        print test_data_x
        print u"測試資料 y："
        print test_data_y
        print u"預測結果："
        print prediction
        print('-'*70)

        test_data_x = np.array([1, 0.15654, 0.75584, 0.01122, 0.42598])
        test_data_y = -1.0
        prediction = np.sign(np.dot(test_data_x, W))
        self.assertEqual(prediction, test_data_y)

        print u"測試資料 x："
        print test_data_x
        print u"測試資料 y："
        print test_data_y
        print u"預測結果："
        print prediction
        print('-'*70)

if __name__ == '__main__':

    unittest.main(verbosity=2)
