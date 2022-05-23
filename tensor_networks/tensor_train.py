from __future__ import annotations

import re
from copy import deepcopy, copy

from tensor_networks import contraction
from tensor_networks.annotations import *
from tensor_networks.transposition import transpose_outer_indices


class TensorTrain(Sequence[Array]):
    """
    This class represents a tensor train decomposition.
    It acts as an array of its cores.
    Every core's indices are assumed to have the following meaning:
        0: left bond index
        1: physical index
        -1: right bond index
    There is also always one core with the additional index:
        2: label index
    """

    def __init__(self, cores: MutableSequence[Array]):
        self.cores = cores

    def contractions(self, keep_mock_indices=True, **kwargs) -> Iterable[Array]:
        cores = copy(self.cores)

        if not keep_mock_indices and len(cores) > 0:
            if cores[0].shape[0] == 1:
                cores[0] = cores[0].squeeze(axis=0)
            if cores[-1].shape[-1] == 1:
                cores[-1] = cores[-1].squeeze(axis=-1)

        return contraction.contractions(*cores, **kwargs)

    def contract(self, fully=False, **kwargs) -> Array:
        """
        :param fully: Whether to contract the outer indices
        :return: The array obtained by contracting every core
        """
        contracted = contraction.contract(*self, **kwargs)
        if fully:
            contracted = contracted.trace(axis1=0, axis2=-1)
        return contracted

    def attach(self, attachments: Iterable[Array], **kwargs) -> TensorTrain:
        """
        :param attachments: Tensors to be contracted with
        :return: Every core contracted with its respective attachment
        """
        cores = [contraction.attach(core, attached, **kwargs)
                 for core, attached in zip(self, attachments)]
        return type(self)(cores)

    @property
    def shape(self):
        return [t.shape for t in self]

    @overload
    def __getitem__(self, item: int) -> Array:
        ...

    @overload
    def __getitem__(self, item: slice) -> TensorTrain:
        ...

    def __getitem__(self, item):
        value = self.cores[item]
        if isinstance(item, slice):
            if item.step is not None and item.step < 0:
                # transpose bond indices if the train gets reversed
                value = [transpose_outer_indices(arr) for arr in value]
            return type(self)(value)
        return value

    def __setitem__(self, key: int, value: Array):
        self.cores[key] = value

    def __reversed__(self) -> Iterator[Array]:
        return iter(self[::-1])

    def __len__(self) -> int:
        return len(self.cores)

    def __iter__(self) -> Iterator[Array]:
        return iter(self.cores)

    def __mul__(self, other):
        return type(self)([other * c for c in self])

    __rmul__ = __mul__

    def __truediv__(self, other):
        return type(self)([c / other for c in self])

    __rtruediv__ = __truediv__

    def __str__(self) -> str:
        return '[' + ', '.join(re.sub(r'\s+', ' ', str(c)) for c in self) + ']'

    def __repr__(self) -> str:
        type_name = type(self).__name__
        cores_string = (f',\n{" " * (len(type_name) + 1)}'
                        .join(re.sub(r'\s+', ' ', repr(c)) for c in self))
        return f'{type_name}({cores_string})'

    def __copy__(self) -> TensorTrain:
        return type(self)(self.cores)

    def __deepcopy__(self, memo=None):
        return type(self)([deepcopy(c, memo=memo) for c in self])

    def __eq__(self, other) -> bool:
        if not isinstance(other, TensorTrain):
            raise NotImplementedError
        return self.cores == other.cores
