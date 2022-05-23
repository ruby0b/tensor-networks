from tensor_networks.patched_numpy import np
from pytest import approx

from tensor_networks.contraction import contract
from tensor_networks.decomposition import truncated_svd, split
from tests.helpers import constant_fixture, arange_from_shape


matrix = constant_fixture(params=[
    arange_from_shape(10, 5),
    arange_from_shape(1, 1),
])
max_chi = constant_fixture(params=[None, 2, 3, 4, 10, 100])


def test_truncated_svd(matrix, max_chi):
    u, s, v = truncated_svd(matrix, max_chi=max_chi)
    reassembled = u @ np.diag(s) @ v
    assert reassembled == approx(matrix)


arr = constant_fixture(params=[
    arange_from_shape(2, 3, 4, 5),
    arange_from_shape(2, 3),
    arange_from_shape(5),
    arange_from_shape(1),
])


def test_split(arr, max_chi):
    a1, a2 = split(arr, 2, max_chi=max_chi)
    reassembled = contract(a1, a2)
    assert reassembled == approx(arr)
