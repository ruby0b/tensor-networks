import time
from functools import partial

from examples.utils.greyscale_image import image_feature_sin_cos, image_feature_linear
from examples.utils.io import load_mat_data_set, print_test_results
from examples.utils.results import ManyClassificationTests
from tensor_networks.decomposition import truncated_svd
from tensor_networks.inputs import index_label
from tensor_networks.patched_numpy import np
from tensor_networks.training import sweep_entire_train, update
from tensor_networks.weights import starting_weights


TRAIN_AMOUNT = 10000  # max: 60000
TEST_AMOUNT = 10000  # max: 10000
CHI = 20
LAMBDA = 0.001
# e.g. image_feature_sin_cos or image_feature_linear
FEATURE = image_feature_sin_cos

if __name__ == '__main__':
    print(f'{TRAIN_AMOUNT=}, {TEST_AMOUNT=}, {CHI=}, {LAMBDA=}, {FEATURE=}')
    # patch arrays to be float32 to enhance performance
    np.GLOBAL_NUMERIC_DATA_TYPE = np.float32

    # load data set
    train_inputs, test_inputs = load_mat_data_set(
        path='./examples/mnist/mnist-7x7.mat',
        feature=lambda x, y: (FEATURE(x), index_label(y, 10)),
        train_amount=TRAIN_AMOUNT,
        test_amount=TEST_AMOUNT
    )

    # starting weights
    weights = starting_weights(input_length=len(train_inputs[0].array),
                               label_length=10)

    # optimize
    control_results = ManyClassificationTests.create(weights, test_inputs)
    print('### Results before any optimization:')
    print_test_results(control_results, summary_only=True)
    print()
    sweep_iterator = sweep_entire_train(weights, train_inputs,
                                        svd=partial(truncated_svd, max_chi=CHI),
                                        updater=partial(update, factor=LAMBDA))
    for i in range(1, 21):
        print(f'### Sweep {i} ... ', end='')
        start_time = time.time()
        next(sweep_iterator)
        end_time = time.time()
        print(f'Done! ({end_time - start_time:.2f}s) ###')
        results = ManyClassificationTests.create(weights, test_inputs)
        print_test_results(results, summary_only=True)
        print()
