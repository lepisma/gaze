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

    gray = np.array(gray_image, dtype = np.float32)

    grad = np.column_stack(((gray[:, 1] - gray[:, 0]), \
                            (gray[:, 2 :] - gray[:, 0 : -2]) / 2, \
                            (gray[:, -1] - gray[:, -2])))

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

def test_possible_centers(pos_x, pos_y, weight, grad_x, grad_y, out_image):
    """
    Calculates the dot product between
    - Vector from all possible centers to gradient origin
    - Gradient vector at the given point of gradient origin
    
    Parameters
    ----------
    pos_x, pos_y : int
        Position of gradient origin
    weight : float
        Weight of gradient
    grad_x, grad_y : int
        Value of gradients at pos
    out_image : numpy.ndarray
        Accumulator matrix (of same size as image) to keep track of
        cumulative sum of dot products
    """

    rows, columns = out_image.shape
    x_accu = np.tile(np.linspace(1, columns - 1, columns), [rows, 1])
    y_accu = np.tile(np.linspace(1, rows - 1, rows), [columns, 1]).T
    
    x_accu = pos_x - x_accu
    y_accu = pos_y - y_accu
    
    mag = np.sqrt((x_accu ** 2) + (y_accu ** 2))
    
    # Normalize
    x_accu /= mag
    y_accu /= mag

    x_accu[np.isnan(x_accu)] = 0
    y_accu[np.isnan(y_accu)] = 0

    # Dot product
    prod = (x_accu * grad_x) + (y_accu * grad_y)
    prod[prod < 0] = 0

    out_image += prod * prod * weight
    
    return

def find_center(grad_x, grad_y, out_image):
    """
    Finds the center of eye from given grayscale image's gradients

    Parameters
    ----------
    grad_x : numpy.ndarray
        Array of x gradients
    grad_y : numpy.ndarray
        Array of y gradients

    Returns
    -------
    (x, y) : tuple
        The pixel index of eye's center, relative to grad images
    """

    rows, columns = grad_x.shape
    
    #pos_list = coords(np.arange(rows), np.arange(columns))

    x_pos = np.repeat(np.arange(rows), columns)
    y_pos = np.tile(np.arange(columns), rows)

    x_grad = grad_x.ravel(order = 'F')
    y_grad = grad_y.ravel(order = 'F')
    
    v_possible_centers = np.vectorize(test_possible_centers, excluded = ["out_image"])
    v_possible_centers(x_pos, y_pos, 1.0, x_grad, y_grad, out_image = out_image)

    return np.unravel_index(out_image.argmax(), out_image.shape)
    
    #out_image /= np.max(out_image)
    #out_image *= 255
    
    #return out_image

def coords(*arrays):
    """
    Returns cartesian coordinate combinations from given arrays
    """

    grid = np.meshgrid(*arrays)
    coord_list = [entry.ravel() for entry in grid]
    points = np.vstack(coord_list).T
    return points.tolist()
