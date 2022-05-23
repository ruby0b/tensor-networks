from scipy.io import savemat, loadmat  # type: ignore[import]

from examples.utils.greyscale_image import shrink_quadratic
from tensor_networks.patched_numpy import np


def save_shrunken_mnist(new_side_length: int, original_file_path, new_file_path):
    data = loadmat(original_file_path, squeeze_me=True)

    def shrink(images):
        return np.array([shrink_quadratic(img, new_side_length).flatten() for img in images])

    data['trainX'] = shrink(data['trainX'])
    data['testX'] = shrink(data['testX'])
    savemat(new_file_path, data, do_compression=True)
