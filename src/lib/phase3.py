import logging
import shutil
from os.path import join

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import neighbors

from lib.general_helpers import prepare_output_dir, find_cached_output

# Phase 3: Graph generation
# Depending on: Phase 2

def run_phase3(_config):
    _output_dir = prepare_output_dir(_config)
    try:
        run_phase3_core(_config["input"], _config["params"], _output_dir)
        return _output_dir
    except Exception as e:
        shutil.rmtree(_output_dir)
        raise e


def run_phase3_core(_input, params, _output_dir):
    _df_embeddings = pd.read_table(
        join(_input, "main.txt"),
        skiprows=1,
        sep=" ",
        header=None,
        index_col=0)
    _df_embeddings.index.names = ["word"]

    _X = _df_embeddings

    if params["variant"] == "knn":
        affinity_matrix = neighbors.kneighbors_graph(_X,
                                                     n_neighbors=params["options"]["n_neighbors"],
                                                     include_self=params["options"]["include_self"],
                                                     mode=params["options"]["mode"],
                                                     n_jobs=-1)
    elif params["variant"] == "radius":
        affinity_matrix = neighbors.radius_neighbors_graph(_X,
                                                           radius=params["options"]["radius"],
                                                           include_self=params["options"]["include_self"],
                                                           mode=params["options"]["mode"],
                                                           n_jobs=-1)
    else:
        raise NotImplementedError

    if params["options"]["mode"] == "distance":
        # convert the weights from euclidean distance to cosine distance via cos_d = (euc_d^2)/2
        # and then to cosine **similarity** via cos_sim = 1 - cos_dist
        affinity_matrix.data = np.subtract(1, np.divide(np.square(affinity_matrix.data), 2))

        # In the distance case, the similarity of a node to itself will be "1".
        # If no self-loops are wanted, we manually set the diagonal to 0.
        if params["options"]["include_self"] is False:
            affinity_matrix.setdiag(0)

    # via https://github.com/scikit-learn/scikit-learn/issues/8008
    # and https://github.com/musically-ut/semi_supervised/blob/master/semi_supervised/label_propagation.py#L136
    # and https://stackoverflow.com/questions/28904411/making-a-numpy-ndarray-matrix-symmetric
    if params["options"]["force_symmetric"] is True:
        # NOTE: Some nodes will now have more than k edges
        affinity_matrix = sparse.csr_matrix.maximum(affinity_matrix, affinity_matrix.T)

    assert (affinity_matrix.format == "csr"), "Not a csr matrix!"
    sparse.save_npz(join(_output_dir, "main.txt"), affinity_matrix)
    _df_embeddings.to_csv(join(_output_dir, "words.txt"), sep=" ", columns=[])


def phase3(_config):
    logging.info("Phase 3 starting")
    _cached_output_dir = find_cached_output(_config["output"], _config["input"], _config["params"])

    if _cached_output_dir is not None:
        return _cached_output_dir

    return run_phase3(_config)
