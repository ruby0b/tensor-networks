from scipy.io import loadmat  # type: ignore[import]

from examples.utils.results import ManyClassificationTests
from tensor_networks.inputs import Input
from tensor_networks.patched_numpy import np


def load_mat_data_set(path, feature, train_amount=None, test_amount=None):
    data = loadmat(path, squeeze_me=True)

    trainX = data['trainX'][:train_amount]
    trainY = data['trainY'][:train_amount]
    testX = data['testX'][:test_amount]
    testY = data['testY'][:test_amount]

    train_inputs = [Input(*feature(x, y)) for x, y in zip(trainX, trainY)]
    test_inputs = [Input(*feature(x, y)) for x, y in zip(testX, testY)]
    return train_inputs, test_inputs


def print_test_results(results: ManyClassificationTests, decimal_places=None, summary_only=False):
    if not summary_only:
        for test in results.tests:
            guess_vs_actual_str = (f'✅ {test.output_label}'
                                   if test.correct
                                   else f'❌ {test.output_label} ({test.actual_label=})')
            rounded_label = (test.output_label_vec
                             if decimal_places is None
                             else np.round(test.output_label, decimal_places))
            print(f'{guess_vs_actual_str :15}; '
                  f'cost={test.cost} '
                  f'{list(rounded_label)}')

    print(f'Overall cost: {results.cost_sum}\n'
          f'Success rate: {results.success_rate :.0%} '
          f'({results.correct_guesses}/{len(results.tests)})')
