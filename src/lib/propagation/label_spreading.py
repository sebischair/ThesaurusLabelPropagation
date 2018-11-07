import logging

import numpy as np
from scipy import sparse
from sklearn.utils.extmath import safe_sparse_dot
import pandas as pd


def label_spreading(affinity_matrix, y, options):
    '''
    ## References
    - [1]E. Buchnik and E. Cohen, “Bootstrapped Graph Diffusions: Exposing the Power of Nonlinearity,” arXiv:1703.02618 [cs], Mar. 2017.
    - [2]Y. Bengio, O. Delalleau, and N. Le Roux, “Label Propagation and Quadratic Criterion,” Semi-Supervised Learning, Sep. 2006.
    - [3]F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.

    Implemented according to "Algorithm 2: Normalized Laplacian LP" from [1] as their description is
    more detailed, but equivalent to "Algorithm 11.3 Label spreading" described by [2] except that
    the hyperparameter \alpha acts inversed between them both. Used \alpha meaning from [2] as this
    seems to be the more accepted way (e.g. also in [3]).

    Code skeleton similar to implementations from [3].

    :param affinity_matrix: sparse matrix in CSR format
    :param y:
    :param options:
    :return:
    '''

    y = np.asarray(y)
    alpha = options["alpha"]

    # construct a categorical distribution for classification only
    classes = np.unique(y)
    classes = (classes[classes != -1])
    n_samples, n_classes = len(y), len(classes)

    # Initialization
    # Diagonal degree matrix
    diag_deg = sparse.csr_matrix.sum(affinity_matrix, axis=1)
    diag_deg = np.asarray(diag_deg).ravel()  # flatten
    dim = affinity_matrix.shape[0]

    diag_mat_pow_min_05 = sparse.csc_matrix((dim, dim))
    diag_mat_pow_min_05.setdiag(diag_deg ** -0.5)

    # Note: [1] and [2] call this matrix "normalized laplacian matrix",
    # probably as this is very similar to the actual symmetrically normalized laplacian matrix (L_sym = I - A_norm)
    A_norm = safe_sparse_dot(safe_sparse_dot(diag_mat_pow_min_05, affinity_matrix), diag_mat_pow_min_05)

    Y = np.zeros((n_samples, n_classes))
    for label in classes:
        Y[y == label, classes == label] = 1
    Y_static = np.copy(Y)

    # Iterate:
    for n_iter_ in range(options["iter"]):
        logging.info("Propagation Iteration {} starting".format(n_iter_))
        Y = alpha * safe_sparse_dot(A_norm, Y) + (1 - alpha) * Y_static

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
