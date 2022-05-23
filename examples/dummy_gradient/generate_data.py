import random
from itertools import islice

from scipy.io import savemat  # type: ignore[import]

from tensor_networks.patched_numpy import np


def generate_data_set():
    while True:
        b0 = random.randint(0, 140)
        b1 = random.randint(0, 140)
        d0 = min(b0 + random.randint(0, 127), 255)
        d1 = min(b1 + random.randint(0, 127), 255)
        label = random.choice([0, 1])
        if label == 0:
            yield np.array([b0, d0, b1, d1]), 0
        else:
            yield np.array([d0, b0, d1, b1]), 1


def save_data_set(n: int, file_path):
    trainX, trainY = zip(*list(islice(generate_data_set(), n)))
    testX, testY = zip(*list(islice(generate_data_set(), n)))
    trainX = np.array(list(trainX))
    trainY = np.array(list(trainY))
    testX = np.array(list(testX))
    testY = np.array(list(testY))
    savemat(file_path, {
        'trainX': trainX,
        'trainY': trainY,
        'testX': testX,
        'testY': testY,
    }, do_compression=True)
