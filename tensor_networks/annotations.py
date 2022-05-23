from typing import *

from tensor_networks.patched_numpy import np


__all__ = (
    # typing
    'List', 'Tuple', 'Iterable', 'Optional', 'Union', 'Iterator', 'Sequence',
    'overload', 'Any', 'TypeVar', 'Callable', 'Type', 'TYPE_CHECKING',
    'Generator', 'Reversible', 'MutableSequence', 'NamedTuple',

    # custom
    'Array', 'AbsColor', 'PartialColor',
)

Array = np.ndarray
AbsColor = int
PartialColor = float
