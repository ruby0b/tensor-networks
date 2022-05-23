import logging
import math
from functools import partial

from tensor_networks.patched_numpy import np

from tensor_networks.annotations import *
from tensor_networks.tensor_train import TensorTrain


class SVD(NamedTuple):
    u: Array
    s: Array
    v: Array


standard_svd = partial(np.linalg.svd, full_matrices=False)


def truncated_svd(matrix: Array, *,
                  compute_chi: Optional[Callable[[SVD], int]] = None,
                  max_chi: Optional[int] = None,
                  normalize: bool = True
                  ) -> SVD:
    """
    Calculates the Singular Value Decomposition for `matrix` and truncates it
    so that the returned matrices all have dimensions with cardinalities
    of at most `chi`.
    `chi` can be computed using a function `compute_chi` and can also not
    exceed `max_chi`.

    :param matrix: The matrix to decompose
    :param compute_chi: A function that takes an SVD and returns a value for `chi`
    :param max_chi: `chi` cannot exceed this value
    :param normalize: Whether to renormalize the SVD after truncating it
    :return:
    """
    u, s, v = standard_svd(matrix)

    if compute_chi:
        max_chi = min(max_chi, compute_chi(SVD(u, s, v)))

    if max_chi is None or len(s) <= max_chi:
        new_u, new_s, new_v = u, s, v
    elif max_chi < 1:
        raise ValueError('max_chi has to be at least 1')
    else:
        new_u, new_s, new_v = u[:, :max_chi], s[:max_chi], v[:max_chi, :]
        if normalize:
            new_s *= np.linalg.norm(s) / np.linalg.norm(new_s)

    logging.debug(f'{matrix.shape} --SVD--> {u.shape} {s.shape} {v.shape}'
                  f' --truncated--> {new_u.shape} {new_s.shape} {new_v.shape}')
    return SVD(new_u, new_s, new_v)


def split(tensor: Array, before_index: int, *,
          svd: Callable[..., SVD] = truncated_svd, **kwargs) -> Tuple[Array, Array]:
    """
    Split an arbitrary tensor into two tensors before a given index.

    :param tensor: The tensor to split
    :param before_index: Split the tensor before this index
    :param svd: An SVD function to use as a backend for decomposing
    :param kwargs: Any other kwargs are passed through to `svd` for convenience
    :return:
        Two tensors with the following shapes:
            1. All dimensions to the left of `before_index` in `tensor` and
               an additional bond index as the last index
            2. A bond index followed by all dimensions of `tensor`,
               starting at `before_index`
    """
    left_shape = tensor.shape[:before_index]
    right_shape = tensor.shape[before_index:]
    matrix = tensor.reshape(int(np.prod(left_shape)), int(np.prod(right_shape)))
    u, s, v = svd(matrix, **kwargs)
    t1 = u
    t2 = np.diag(s) @ v
    t1.shape = (*left_shape, t1.shape[-1])
    t2.shape = (t2.shape[0], *right_shape)
    return t1, t2


def tensor_train_decomposition(tensor: Array, d: int, *,
                               svd: Callable[..., SVD] = truncated_svd,
                               **svd_kwargs) -> TensorTrain:
    """
    Decompose a tensor into the tensor train format using SVD

    :param tensor: The tensor to decompose
    :param d: The dimension of the bond indices
    :param svd: The function used for singular value decomposition
    :param svd_kwargs:
        Any keyworded arguments are passed through to the svd function
        (for convenience)
    :return: The tensor in tensor train format
    """
    # Amount of elements in the tensor: d^N (= tensor.size)
    # <==> N = log_d(d^N)
    n = int(math.log(tensor.size, d))
    # Add a mock index on the left for the first iteration
    t = tensor.reshape(1, tensor.size)
    cores = []
    for i in range(1, n):
        # Reshape the tensor into a matrix (to calculate the SVD)
        t.shape = (d * t.shape[0], d ** (n - i))
        # Split the tensor using Singular Value Decomposition (SVD)
        u, s, v = svd(t, **svd_kwargs)
        # Split the first index of the matrix u
        u.shape = (u.shape[0] // d, d, u.shape[1])
        # u is part of the tensor train
        cores.append(u)
        # Continue, using the contraction of s and v as the remaining tensor
        t = np.diag(s) @ v
    # The remaining matrix is the right-most tensor in the tensor train
    # and gets a mock index on the right for consistency
    t.shape = (*t.shape, 1)
    cores.append(t)
    return TensorTrain(cores)
