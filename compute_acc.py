from itertools import permutations

import numpy as np


def compute_acc(pred_labels: np.ndarray, true_labels: np.ndarray, n_labels: int, test_permutations: bool = False):
    best_acc = np.NINF
    index_permutations = permutations(range(n_labels)) if test_permutations else [range(n_labels)]
    for index_permutation in index_permutations:
        pred_labels_remapped = np.take(a=index_permutation, indices=pred_labels)
        acc = np.mean(pred_labels_remapped == true_labels)
        if acc > best_acc:
            best_acc = acc
    return best_acc
