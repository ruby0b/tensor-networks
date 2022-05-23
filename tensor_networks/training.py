from more_itertools import consume

from tensor_networks import utils
from tensor_networks.annotations import *
from tensor_networks.contraction import contract, tensor_product, attach
from tensor_networks.decomposition import truncated_svd, split, SVD
from tensor_networks.inputs import Input
from tensor_networks.tensor_train import TensorTrain
from tensor_networks.transposition import transpose_outer_indices
from tensor_networks.utils import Direction, neutral_array


def update(ideal_output: Array, calculated_output: Array, local_input: Array,
           factor=0.001) -> Array:
    """
    Calculate an update for two contracted cores based on an input.
    The returned update array has the same shape as the two contracted cores
    and values that are relatively small compared to the cores.
    """
    full_update = tensor_product(ideal_output - calculated_output, local_input)\
        .transpose(1, 2, 0, 3, 4)
    small_update = factor * full_update
    assert isinstance(small_update, Array)
    return small_update


class IndexFrame:
    """
    Represents two indices (`label` and `other`)
    as well as the `direction` in which one is iterating.
    """

    def __init__(self, label_index: int, other_index: int, direction: Direction):
        self.label = label_index
        self.other = other_index
        self.direction = direction
        self.left, self.right = (self.label, self.other)[::self.direction.value]


class InputFrame:
    """
    This data structure is unique for each `tensor_train` and `input`.
    It calculates contractions of the left and right side of
    a specified `IndexFrame` and caches the accumulations that are
    produced during the calculation to speed up later contractions.
    """

    def __init__(self, tensor_train: TensorTrain, input_: Input):
        self.tensor_train = tensor_train
        self.input = input_
        self.accumulated_left: List[Array] = []
        self.accumulated_right: List[Array] = []

    def shift(self, indices: IndexFrame):
        """
        Adjust `accumulated_left` and `accumulated_right` to be the accumulated
        contractions on the left and right side of the specified indices.

        Using accumulation instead of reduction
        (which would only save the last element of the accumulation)
        has the advantage of avoiding redundant computation since only the ends
        of the accumulation ever change and the rest of it can be reused.
        """
        while len(self.accumulated_left) > indices.left:
            self.accumulated_left.pop()
        while len(self.accumulated_right) > len(self.tensor_train) - 1 - indices.right:
            self.accumulated_right.pop()
        while len(self.accumulated_left) < indices.left:
            previous_reduced = utils.get_last(self.accumulated_left, default=neutral_array(1))
            next_core = self.tensor_train[len(self.accumulated_left)]
            next_attachment = self.input.array[len(self.accumulated_left)]
            next_attached = attach(next_core, next_attachment)
            new_reduced = contract(previous_reduced, next_attached)
            self.accumulated_left.append(new_reduced)
        while len(self.accumulated_right) < len(self.tensor_train) - 1 - indices.right:
            previous_reduced = utils.get_last(self.accumulated_right, default=neutral_array(1))
            next_core = self.tensor_train[len(self.tensor_train) - 1 - len(self.accumulated_right)]
            next_attachment = self.input.array[
                len(self.tensor_train) - 1 - len(self.accumulated_right)]
            next_attached = attach(next_core, next_attachment)
            new_reduced = contract(next_attached, previous_reduced)
            self.accumulated_right.append(new_reduced)

    def local_input(self, indices: IndexFrame) -> Array:
        """Returns the tensor product of all inputs around the specified `indices`."""
        self.shift(indices)

        # The prefixes 'l' and 'r' stand for 'left' and 'right'.
        # They symbolize that the variable can be used as if
        # it really was on the left/right side
        # (even though it might actually have been on the other side).
        l_label_acc, r_other_acc = indices.direction.order(self.accumulated_left,
                                                           self.accumulated_right)
        l_label_reduced = utils.get_last(l_label_acc, default=neutral_array(1))
        r_other_reduced = utils.get_last(r_other_acc, default=neutral_array(1))

        return tensor_product(l_label_reduced,
                              self.input.array[indices.label],
                              self.input.array[indices.other],
                              r_other_reduced)


def contraction_of_cores_to_optimize(tensor_train: TensorTrain, indices: IndexFrame) -> Array:
    l_label_core = tensor_train[indices.label]
    r_other_core = tensor_train[indices.other]
    if indices.direction == Direction.RIGHT_TO_LEFT:
        # swap bond indices when going backward so that further algorithms
        # only need to handle the case of going left to right
        l_label_core = transpose_outer_indices(l_label_core)
        r_other_core = transpose_outer_indices(r_other_core)

    # core with label is contracted with the next core
    return contract(l_label_core, r_other_core)


def apply_optimized_contraction(tensor_train: TensorTrain, optimized: Array,
                                svd: Callable[..., SVD], indices: IndexFrame) -> None:
    # split the optimized cores
    label_core, other_core = split(optimized, before_index=2, svd=svd)
    # transpose label index into its correct position (from 1 to 2)
    other_core = other_core.swapaxes(1, 2)
    if indices.direction == Direction.LEFT_TO_RIGHT:
        tensor_train[indices.label] = label_core
        tensor_train[indices.other] = other_core
    else:
        tensor_train[indices.label] = transpose_outer_indices(label_core)
        tensor_train[indices.other] = transpose_outer_indices(other_core)


def sweep(tensor_train: TensorTrain,
          inputs: Sequence[Input],
          *,
          label_index: int = 0,
          starting_direction: Direction = Direction.LEFT_TO_RIGHT,
          updater: Callable[[Array, Array, Array], Array] = update,
          svd: Callable[..., SVD] = truncated_svd
          ) -> Iterator[None]:
    """
    Sweep back and forth through the train and optimize the cores

    :param tensor_train: The train to optimize
    :param inputs: Any number of inputs to train the tensor train with
    :param label_index: The index at which we start optimizing
    :param starting_direction: The direction in which we start sweeping
    :param updater: The function used for calculating updates
    :param svd: The function used for singular value decomposition
    """
    assert len(inputs[0].array) == len(tensor_train)

    input_frames = [InputFrame(tensor_train, inp) for inp in inputs]
    index_generator = utils.swing_pairwise(range(len(tensor_train)),
                                           start=label_index,
                                           direction=starting_direction)

    for label_index, other_index, direction in index_generator:
        indices = IndexFrame(label_index, other_index, direction)

        to_optimize = contraction_of_cores_to_optimize(tensor_train, indices)

        local_inputs = [inp_frame.local_input(indices)
                        for inp_frame in input_frames]
        outputs = [contract(to_optimize, local_inp, axes=([0, 1, 3, 4], [0, 1, 2, 3]))
                   for local_inp in local_inputs]
        updates = [updater(inp.label, out, local_inp)
                   for inp, out, local_inp in zip(inputs, outputs, local_inputs)]

        optimized = to_optimize + sum(updates)
        apply_optimized_contraction(tensor_train, optimized, svd=svd, indices=indices)

        # yield to allow the caller of this function to stop
        # the iterations at some point (otherwise this would go on infinitely)
        yield


def sweep_entire_train(tensor_train: TensorTrain, inputs: Sequence[Input],
                       **kwargs) -> Iterator[None]:
    """Each iteration is a sweep of the entire train"""
    sweep_iterator = sweep(tensor_train, inputs, **kwargs)
    while True:
        consume(sweep_iterator, len(tensor_train) - 1)
        yield
