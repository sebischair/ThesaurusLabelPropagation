import logging

import numpy as np
from scipy import sparse
from sklearn.utils.extmath import safe_sparse_dot
import pandas as pd


def label_propagation(affinity_matrix, y, options):
    '''
    ## References
    - [1]E. Buchnik and E. Cohen, “Bootstrapped Graph Diffusions: Exposing the Power of Nonlinearity,” arXiv:1703.02618 [cs], Mar. 2017.
    - [2]Y. Bengio, O. Delalleau, and N. Le Roux, “Label Propagation and Quadratic Criterion,” Semi-Supervised Learning, Sep. 2006.

    Implemented according to "Algorithm 1: Label propagation (LP)" from Buchnik and Cohen as their description is
    more detailed, but equivalent to "Algorithm 11.1 Label propagation" described by Bengio et al.

    Code skeleton similar to implementations from [3].

    :param affinity_matrix: sparse matrix in CSR format
    :param y:
    :param options:
    :return:
    '''

    y = np.asarray(y)
    unlabeled = y == -1
    unlabeled = unlabeled[:, np.newaxis]

    # construct a categorical distribution for classification only
    classes = np.unique(y)
    classes = (classes[classes != -1])
    n_samples, n_classes = len(y), len(classes)

    # Initialization
    # Diagonal degree matrix
    diag_deg = sparse.csr_matrix.sum(affinity_matrix, axis=1)
    diag_deg = np.asarray(diag_deg).ravel()  # flatten
    dim = affinity_matrix.shape[0]

    # we want to invert the diagonal degree matrix => the same as dividing 1/(degree) => much more efficient
    # see https://en.wikipedia.org/wiki/Diagonal_matrix#Matrix_operations
    diag_mat_inv = sparse.csc_matrix((dim, dim))
    diag_mat_inv.setdiag(1 / diag_deg)

    Y = np.zeros((n_samples, n_classes))
    for label in classes:
        Y[y == label, classes == label] = 1
    Y_static = np.copy(Y)

    # Iterate:
    for n_iter_ in range(options["iter"]):
        logging.info("Propagation Iteration {} starting".format(n_iter_))
        Y = safe_sparse_dot(safe_sparse_dot(diag_mat_inv, affinity_matrix), Y)

        # Reset learned labels of labeled nodes
        Y = np.where(unlabeled,
                     Y,
                     Y_static)

    # Row-normalize Y so Y acts as a confidence matrix
    normalizer = np.sum(
        Y,
        axis=1)[:, np.newaxis]
    Y = np.nan_to_num(Y / normalizer)

    # Finalize:
    predictions = classes[np.argmax(Y, axis=1)]
    confidences = np.max(Y, axis=1)

    # If confidence for top prediction is 0.0, "predict" the garbage class -1
    for idx, prediction in enumerate(predictions):
        if confidences[idx] == 0.0:
            predictions[idx] = -1

    def top3_candidate_classes_sorted(row):
        inds = np.argpartition(row, -3)[-3:]
        inds_sorted_by_conf = inds[np.argsort(row[inds])][::-1]
        return [classes[ind] if row[ind] != 0.0 else -1 for ind in inds_sorted_by_conf]
    top3_classes = np.apply_along_axis(top3_candidate_classes_sorted, axis=1, arr=Y)

    return predictions.ravel(), confidences.ravel(), top3_classes.tolist()
