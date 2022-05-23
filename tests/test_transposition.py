from tensor_networks.transposition import transpose_outer_indices, reverse_dimensions
from tests.helpers import constant_fixture, arange_from_shape

from tensor_networks.patched_numpy import np


arr = constant_fixture(params=[
    arange_from_shape(9, 9),
    np.random.random((3, 7, 5, 3, 2)),
    np.array([]),
    arange_from_shape(8),
])


def test_transpose_bond_indices(arr):
    assert (arr == transpose_outer_indices(transpose_outer_indices(arr))).all()


def test_reverse_transpose(arr):
    assert (arr == reverse_dimensions(reverse_dimensions(arr))).all()
