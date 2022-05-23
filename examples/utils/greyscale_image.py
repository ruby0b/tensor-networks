from itertools import product
from math import cos, pi, sin, sqrt

from tensor_networks.annotations import *
from tensor_networks.annotations import Array
from tensor_networks.patched_numpy import np


def color_abs_to_percentage(value: AbsColor) -> PartialColor:
    """
    :param value: An integer color value [0;255]
    :return: The color value in percent [0;1]
    """
    return value / 255


def greyscale_feature_sin_cos(percentage: PartialColor) -> Array:
    """
    :param percentage: A grey value
    :return: An array of a black value and a white value with a sum of 1
    """
    return np.array([cos(pi / 2 * percentage), sin(pi / 2 * percentage)])


def greyscale_feature_linear(percentage: PartialColor) -> Array:
    """
    :param percentage: A grey value
    :return: An array of a black value and a white value with a sum of 1
    """
    return np.array([percentage, 1 - percentage])


def image_feature_sin_cos(absolute_colors: Array) -> Array:
    return np.array(list(map(greyscale_feature_sin_cos,
                             map(color_abs_to_percentage,
                                 absolute_colors))))


def image_feature_linear(absolute_colors: Array) -> Array:
    return np.array(list(map(greyscale_feature_linear,
                             map(color_abs_to_percentage,
                                 absolute_colors))))


def shrink_quadratic(image: Array, new_side_length: int) -> Array:
    """Only works if `new_side_length` is a divisor of `image`'s side length for now"""""
    new_image = np.zeros((new_side_length, new_side_length))
    if len(image.shape) == 1:
        old_side_length = int(sqrt(image.shape[0]))
        image = image.reshape(old_side_length, old_side_length)
    else:
        assert len(image.shape) == 2
        assert image.shape[0] == image.shape[1]
        old_side_length = image.shape[0]
    assert old_side_length % new_side_length == 0
    scale = old_side_length // new_side_length
    for i, j in product(range(new_side_length), range(new_side_length)):
        new_image[i, j] = np.average(image[i*scale:i*scale + scale, j*scale:j*scale + scale])
    return new_image
