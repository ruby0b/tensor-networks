from copy import copy

from tensor_networks.patched_numpy import np
import pytest
from pytest import approx

from tensor_networks.decomposition import tensor_train_decomposition
from tensor_networks.transposition import reverse_dimensions
from tests.helpers import constant_fixture


data_arrays, data_ds = zip(
    (np.arange(64), 2),
    (np.arange(4), 2),
)
data_ids = [str(x) for x in range(len(data_arrays))]

arr = constant_fixture(params=data_arrays, ids=data_ids)
d = constant_fixture(params=data_ds, ids=data_ids)
max_chi = constant_fixture(params=[2, 4, 8, None])


@pytest.fixture(ids=data_ids)
def tt(arr: np.ndarray, d: int, max_chi: int):
    t = tensor_train_decomposition(arr, d, max_chi=max_chi)
    return t


def test_shape(tt, d, max_chi):
    for s in tt.shape:
        # general shape
        assert len(s) == 3
        # physical indices
        assert s[1] == d
    # bond indices
    if max_chi is not None:
        for s in tt.shape[1:]:
            assert s[0] <= max_chi
        for s in tt.shape[:-1]:
            assert s[-1] <= max_chi
    # mock bond indices
    assert tt.shape[0][0] == 1
    assert tt.shape[-1][-1] == 1


@pytest.fixture(ids=data_ids)
def accumulated(tt):
    return list(tt.contractions())


def test_accumulate(tt):
    for l, t in zip(range(3, len(tt)), tt.contractions()):
        assert len(t.shape) == l


@pytest.fixture(ids=data_ids)
def reduced(tt):
    return tt.contract(fully=True)


def test_reassemble(arr, d, reduced):
    assert np.prod(reduced.shape) == arr.shape
    for i in reduced.shape:
        assert i == d
    assert reduced.flatten() == approx(arr.flatten())


def test_reversed(tt, reduced):
    rev = tt[::-1]
    for x, y in zip(reversed(tt), rev):
        assert (x == y).all()
    rev_reassembled = rev.contract(fully=True)
    assert reverse_dimensions(rev_reassembled) == approx(reduced)


def test_arithmetic(tt):
    assert np.array(tt.cores) - np.array(((tt / 5) * 5).cores) == approx(0)
    assert np.array(tt.cores) - np.array(((5 * tt) / 5).cores) == approx(0)


def test_setitem_and_copy(tt):
    ttc = copy(tt)
    if len(ttc) > 0:
        a = ttc[0]
        b = ttc[-1]
        ttc[0] = np.array([1, 2, 3])
        ttc[-1] = np.array([1, 2, 3])
        ttc[0] = a
        ttc[-1] = b
    assert ttc == tt
    with pytest.raises(IndexError):
        ttc[len(ttc)]
