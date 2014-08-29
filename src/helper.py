"""
Helper functions
"""

import numpy as np

def xgrad(gray_image):
    """
    Returns the X gradient of grayscale image,
    imitating MatLab's gradient function
    """

    grad = np.column_stack(((gray_image[:, 1] - gray_image[:, 0]), \
                            (gray_image[:, 2 :] - gray_image[:, 0 : -2]) / 2, \
                            (gray_image[:, -1] - gray_image[:, -2])))

    return grad

def ygrad(gray_image):
    """
    Returns the Y gradient of grayscale image,
    imitatin MatLab's gradient function
    """

    grad = xgrad(gray_image.T).T

    return grad
