from tensor_networks.patched_numpy import np
import pytest


def constant_fixture(*args, **kwargs):
    return pytest.fixture(*args, **kwargs)(lambda request: request.param)


def arange_from_shape(*shape: int, **kwargs):
    return np.arange(np.prod(shape)).reshape(shape)
