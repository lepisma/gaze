"""
Helper functions
"""

import numpy as np

def xgrad(gray_image):
    """
    Returns the X gradient of grayscale image,
    imitating MatLab's gradient function

    Parameters
    ----------
    gray_image : numpy.ndarray
        Grayscale image

    Returns
    -------
    numpy.ndarray
        X gradient of image
    """

    grad = np.column_stack(((gray_image[:, 1] - gray_image[:, 0]), \
                            (gray_image[:, 2 :] - gray_image[:, 0 : -2]) / 2, \
                            (gray_image[:, -1] - gray_image[:, -2])))

    return grad

def ygrad(gray_image):
    """
    Returns the Y gradient of grayscale image,
    imitating MatLab's gradient function

    Parameters
    ----------
    gray_image : numpy.ndarray
        Grayscale image

    Returns
    -------
    numpy.ndarray
        Y gradient of image
    """

    grad = xgrad(gray_image.T).T

    return grad

def test_possible_center(pos, weight, grad, out_image):
    """
    Calculates the dot product between
    - Vector from all possible centers to gradient origin
    - Gradient vector at the given point of gradient origin
    
    Parameters
    ----------
    pos : tuple (x, y)
        Position of gradient origin
    weight : float
        Weight of gradient
    grad : tuple (x, y)
        Value of gradients at pos
    out_image : numpy.ndarray
        Accumulator matrix (of same size as image) to keep track of
        cumulative sum of dot products
    """

    rows, columns = out_image.shape
    x_accu = np.tile(np.linspace(1, columns - 1, columns), [rows, 1])
    y_accu = np.tile(np.linspace(1, rows - 1, rows), [columns, 1]).T
    
    x_accu = pos[0] - x_accu
    y_accu = pos[1] - y_accu
    
    mag = np.sqrt((x_accu ** 2) + (y_accu ** 2))
    
    # Normalize
    x_accu = x_accu / mag
    y_accu = y_accu / mag
    
    # Dot product
    prod = (x_accu * grad[0]) + (y_accu * grad[1])
    prod[prod < 0] = 0
    
    out_image = prod * prod * weight
    
    return
