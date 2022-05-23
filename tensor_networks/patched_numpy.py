"""
This module provides a patched version of NumPy under the `numpy` or `np`
variable.
In this patched version one can set GLOBAL_NUMERIC_DATA_TYPE to e.g. np.float32
to use this as the default data type for NumPy arrays.
This only modifies the behaviour of the local copy of NumPy defined in here.
"""
import functools
import importlib.util
import typing

if typing.TYPE_CHECKING:
    import numpy as _backend_module
    np = numpy = _backend_module
else:
    import numpy as _backend_module
    _backend_spec = importlib.util.find_spec('numpy')
    np = numpy = importlib.util.module_from_spec(_backend_spec)
    _backend_spec.loader.exec_module(np)


GLOBAL_NUMERIC_DATA_TYPE: typing.Optional[np.dtype] = None


def _as_global_dtype(array_constructor: typing.Callable[..., np.ndarray]):
    @functools.wraps(array_constructor)
    def changed_dtype(*args, **kwargs):
        if GLOBAL_NUMERIC_DATA_TYPE is not None and len(args) < 2:
            kwargs.setdefault('dtype', GLOBAL_NUMERIC_DATA_TYPE)
        return array_constructor(*args, **kwargs)
    return changed_dtype


np.array = _as_global_dtype(_backend_module.array)
np.ones = _as_global_dtype(_backend_module.ones)
np.zeros = _as_global_dtype(_backend_module.zeros)
np.full = _as_global_dtype(_backend_module.full)
