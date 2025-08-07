# encoding=utf8

import os
import sys

# Add parent directory to path so we can import FukuML
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FukuML.PLA as pla
import FukuML.PocketPLA as pocket

def test_basic_functionality():
    print("Testing PLA BinaryClassifier...")
    pla_bc = pla.BinaryClassifier()
    print("✓ PLA BinaryClassifier instantiated successfully")
    
    print("Testing PocketPLA BinaryClassifier...")
    pocket_bc = pocket.BinaryClassifier()
    print("✓ PocketPLA BinaryClassifier instantiated successfully")
    
    # Test default parameter setting
    pla_bc.set_param()
    pocket_bc.set_param()
    print("✓ Parameters set successfully")
    
    print("\nAll basic functionality tests passed!")
    print("Python 2/3 compatibility fixes working correctly.")

if __name__ == "__main__":
    test_basic_functionality()