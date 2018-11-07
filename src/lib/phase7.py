import shutil
from os.path import join
import logging

import pandas as pd
from scipy import sparse

from lib.general_helpers import prepare_output_dir, find_cached_output
from lib.propagation.label_propagation import label_propagation
from lib.propagation.label_spreading import label_spreading

# Phase 7: Label propagation
# Depending on: Phase 6

def run_phase7(_config):
    _output_dir = prepare_output_dir(_config)
    try:
        run_phase7_core(_config["input"], _config["params"], _output_dir)
        return _output_dir
    except Exception as e:
        shutil.rmtree(_output_dir)
        raise e


def run_phase7_core(_input, params, _output_dir):
    _df_labels = pd.read_table(
        join(_input, "labels.txt"),
        sep=" ",
        index_col=0)
    y = _df_labels.iloc[:, -1]
    affinity_matrix = sparse.load_npz(join(_input, "main.txt.npz"))

    if params["variant"] == "label_propagation":
        predictions, confidences, top3_classes = label_propagation(affinity_matrix, y, params["options"])
    elif params["variant"] == "label_spreading":
        predictions, confidences, top3_classes = label_spreading(affinity_matrix, y, params["options"])
    else:
        raise NotImplementedError

    df_predicted = _df_labels.assign(y_pred=predictions)
    df_predicted = df_predicted.assign(y_conf=confidences)
    df_predicted = df_predicted.assign(y_top3_classes=top3_classes)
    df_predicted.to_csv(join(_output_dir, "main.txt"), sep=" ", columns=["y_pred", "y_conf", "y_top3_classes"])


def phase7(_config):
    logging.info("Phase 7 starting")
    _cached_output_dir = find_cached_output(_config["output"], _config["input"], _config["params"])

    if _cached_output_dir is not None:
        return _cached_output_dir

    return run_phase7(_config)
