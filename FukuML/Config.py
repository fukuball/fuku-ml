# encoding=utf8
"""
FukuML Configuration Module
Provides global settings for FukuML behavior including warning control
"""

import warnings
import numpy as np


class FukuMLConfig:
    """Global configuration for FukuML"""

    # Warning control settings
    SUPPRESS_NUMPY_WARNINGS = False

    @classmethod
    def suppress_warnings(cls, suppress=True):
        """
        Control whether to suppress common numpy warnings globally

        Parameters:
        suppress (bool): True to suppress warnings, False to show them
        """
        cls.SUPPRESS_NUMPY_WARNINGS = suppress

        if suppress:
            # Set NumPy's global error handling - this affects ALL numpy operations
            import numpy as np
            np.seterr(divide='ignore', over='ignore', invalid='ignore', under='ignore')
        else:
            import numpy as np
            np.seterr(all='warn')  # Reset to default warning behavior

    @classmethod
    def reset_warnings(cls):
        """Reset all warning filters to default"""
        cls.SUPPRESS_NUMPY_WARNINGS = False
        warnings.resetwarnings()


# Convenience functions for users
def suppress_warnings(suppress=True):
    """Suppress common numerical warnings from FukuML"""
    FukuMLConfig.suppress_warnings(suppress)


def reset_warnings():
    """Reset warning settings to default"""
    FukuMLConfig.reset_warnings()


# 預設啟用警告抑制，讓使用者有更好的體驗
suppress_warnings(True)
