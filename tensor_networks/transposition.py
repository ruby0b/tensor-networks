from tensor_networks.annotations import *


def reverse_dimensions(array: Array) -> Array:
    """
    :return: array but with all of its indices reversed by transposing them
    """
    return array.transpose(*reversed(list(range(array.ndim))))


def transpose_outer_indices(array: Array) -> Array:
    """
    :return: array but with its first and last index transposed
    """
    if array.ndim < 2:
        return array
    return array.transpose(-1, *range(1, array.ndim - 1), 0)
