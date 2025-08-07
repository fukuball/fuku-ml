# encoding=utf8

import os
import sys

# Add parent directory to path so we can import FukuML
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FukuML.Utility as utility
from test_base import BaseTestCase


class UtilityTestCase(BaseTestCase):

    def test_utility_method(self):
        """Test utility data generation methods"""
        utility.DatasetLoader.gen_lin_separable_data()
        utility.DatasetLoader.gen_non_lin_separable_data()
        utility.DatasetLoader.gen_lin_separable_overlap_data()


if __name__ == '__main__':
    import unittest
    unittest.main()