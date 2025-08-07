# encoding=utf8

import os
import sys
import unittest

# Add parent directory to path so we can import FukuML
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class BaseTestCase(unittest.TestCase):
    """Base class for all FukuML tests"""
    pass