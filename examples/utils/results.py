from __future__ import annotations

from copy import deepcopy

from tensor_networks.annotations import *
from tensor_networks.classification import classify, cost
from tensor_networks.inputs import Input
from tensor_networks.patched_numpy import np
from tensor_networks.tensor_train import TensorTrain


class ClassificationTest(NamedTuple):
    input_: Input
    output_label_vec: Array
    output_label: int
    actual_label: int
    correct: bool
    cost: float

    @staticmethod
    def create(weights: TensorTrain, test_inp: Input) -> ClassificationTest:
        output_label_vec = classify(weights, test_inp)
        output_label = np.argmax(output_label_vec)
        actual_label = np.argmax(test_inp.label)
        return ClassificationTest(
            input_=test_inp,
            output_label_vec=output_label_vec,
            output_label=output_label,
            actual_label=actual_label,
            cost=cost(output_label_vec, test_inp.label),
            correct=output_label == actual_label,
        )


class ManyClassificationTests(NamedTuple):
    tests: List[ClassificationTest]
    weights_snapshot: TensorTrain
    cost_sum: float
    correct_guesses: int
    failed_guesses: int
    success_rate: float
    error_rate: float

    @staticmethod
    def create(weights: TensorTrain, test_inputs: Iterable[Input]) -> ManyClassificationTests:
        weights_snapshot = deepcopy(weights)
        tests = [ClassificationTest.create(weights, test_inp) for test_inp in test_inputs]
        correct_guesses = [t.correct for t in tests].count(True)
        success_rate = correct_guesses / len(tests)
        return ManyClassificationTests(
            tests=tests,
            weights_snapshot=weights_snapshot,
            cost_sum=sum(t.cost for t in tests),
            correct_guesses=correct_guesses,
            failed_guesses=len(tests) - correct_guesses,
            success_rate=success_rate,
            error_rate=1 - success_rate,
        )
