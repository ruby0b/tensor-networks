from enum import Enum
from itertools import tee, cycle, chain, repeat, islice

from tensor_networks.patched_numpy import np
from tensor_networks.annotations import *


_T1 = TypeVar('_T1')
_T2 = TypeVar('_T2')


def get(l: Sequence[_T1], index: int, default: Any = None) -> Union[_T1, Any]:
    return l[index] if index < len(l) and abs(index) <= len(l) else default


def get_last(l: Sequence[_T1], default: Any = None) -> Union[_T1, Any]:
    return get(l, index=-1, default=default)


def identity(x: _T1) -> _T1:
    return x


class Direction(Enum):
    LEFT_TO_RIGHT = 1
    RIGHT_TO_LEFT = -1

    def order(self, *args: _T1) -> Tuple[_T1, ...]:
        """Return all arguments in the order of this direction"""
        return args[::self.value]


def swing_pairwise(seq: Sequence[_T1],
                   start: int = 0,
                   direction: Direction = Direction.LEFT_TO_RIGHT
                   ) -> Iterator[Tuple[_T1, _T1, Direction]]:
    seq_rev = list(reversed(seq))[1:-1]
    iter1, iter2 = tee(cycle(chain(seq, seq_rev)))
    next(iter2, None)

    directions = cycle(chain(repeat(Direction.LEFT_TO_RIGHT, len(seq) - 1),
                             repeat(Direction.RIGHT_TO_LEFT, len(seq) - 1)))

    if direction == Direction.RIGHT_TO_LEFT:
        start = 2 * (len(seq) - 1) - start

    return islice(zip(iter1, iter2, directions), start, None)


def neutral_array(*shape: int, **kwargs):
    arr = np.zeros(shape, **kwargs)
    for i in range(max(shape)):
        ind = tuple(min(i, j-1) for j in shape)
        arr[ind] = 1
    return arr
