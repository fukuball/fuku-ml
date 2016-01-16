#encoding=utf8

import os
import unittest
import FukuML.PocketPLA as pocket


class PocketPLATestCase(unittest.TestCase):

    def test_pocket_pla(self):

        #------------------------------------------------------------

        pocket.load_train_data()


if __name__ == '__main__':

    unittest.main(verbosity=2)
