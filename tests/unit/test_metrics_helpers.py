import numpy as np

import train as train_module


def test_precision_at_k_perfect_ranking():
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert train_module.precision_at_k(y_true, y_score, k_frac=0.2) == 1.0


def test_precision_at_k_multiclass_macro_averages_per_class_precision():
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3] * 5)  # 40 samples, 4 classes
    classes = [0, 1, 2, 3]
    # Each class's own probability column is 1.0 exactly where that class is
    # the true label, so every class's top-k ranking is perfect.
    proba = np.zeros((len(y_true), 4))
    for c in classes:
        proba[y_true == c, c] = 1.0

    result = train_module.precision_at_k_multiclass(y_true, proba, classes, k_frac=0.1)

    assert result == 1.0
