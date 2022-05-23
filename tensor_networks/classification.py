from tensor_networks.annotations import *

from tensor_networks.patched_numpy import np
from tensor_networks.inputs import Input
from tensor_networks.tensor_train import TensorTrain


def cost(labels1: Array, labels2: Array) -> float:
    return np.sum(np.square(labels1 - labels2)) / 2


def classify(tensor_train: TensorTrain, input: Input):
    return tensor_train.attach(input.array).contract(fully=True)
