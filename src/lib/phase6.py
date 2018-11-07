import logging
import shutil
from os.path import join

import pandas as pd
from scipy import sparse
import numpy as np

from lib.general_helpers import prepare_output_dir, find_cached_output

# Phase 6: Graph labeling (with training data)
# Depending on: Phase 3 & Phase 5

def run_phase6(_config):
    _output_dir = prepare_output_dir(_config)
    try:
        run_phase6_core(_config["inputs"], _config["params"], _output_dir)
        return _output_dir
    except Exception as e:
        shutil.rmtree(_output_dir)
        raise e


def run_phase6_core(_inputs, params, _output_dir):
    if params["variant"] != "normal":
        raise Exception

    _corpus_graph = sparse.load_npz(join(_inputs["corpus_graph"], "main.txt.npz"))
    _corpus_graph_words = pd.read_table(
        join(_inputs["corpus_graph"], "words.txt"),
        sep=" ",
        index_col=0)

    _y_train = pd.read_table(
        join(_inputs["thesaurus"], "y_train.txt"),
        sep=" ",
        dtype={"synset": np.int32},
        index_col=0)

    labels = _corpus_graph_words.join(_y_train, how="left").fillna(-1, downcast="infer")
    labels.rename(columns={"synset": "y_train"}, inplace=True)

    labels["y_train"].to_csv(join(_output_dir, "labels.txt"), sep=" ", header=True)
    sparse.save_npz(join(_output_dir, "main.txt"), _corpus_graph)


def phase6(_config):
    logging.info("Phase 6 starting")
    _cached_output_dir = find_cached_output(_config["output"], _config["inputs"], _config["params"])

    if _cached_output_dir is not None:
        return _cached_output_dir

    return run_phase6(_config)
