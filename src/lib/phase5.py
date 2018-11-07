import logging
import shutil
from os.path import join

import pandas as pd
from numpy.random import RandomState

from lib.general_helpers import prepare_output_dir, find_cached_output

# Phase 5: Thesaurus sampling
# Depending on: Phase 4

def run_phase5(_config):
    _output_dir = prepare_output_dir(_config)
    try:
        run_phase5_core(_config["input"], _config["params"], _output_dir)
        return _output_dir
    except Exception as e:
        shutil.rmtree(_output_dir)
        raise e


def run_phase5_core(_input, params, _output_dir):
    rng = RandomState(params["rng_num"])

    _df_thesaurus = pd.read_table(
        join(_input, "main.txt"),
        sep=" ",
        index_col=0)

    _y = _df_thesaurus.copy()
    if params["variant"] == "sample_each_concept_frac":
        # frac: Fraction of elements of each concept that will go into **y_train**
        _y_train = _y.groupby(["synset"], as_index=False).apply(
            lambda x: x.sample(frac=params["options"]["frac"], random_state=rng))
        _y_train.index = _y_train.index.droplevel()
        _y_test = _y.drop(_y_train.index)
    elif params["variant"] == "sample_each_concept_n":
        # n_test: Number of elements of each concept that will go into **y_test**
        _y_test = _y.groupby(["synset"], as_index=False).apply(
            lambda x: x.sample(n=params["options"]["n_test"], random_state=rng))
        _y_test.index = _y_test.index.droplevel()
        _y_train = _y.drop(_y_test.index)
    elif params["variant"] == "all_as_training":
        _y_train = _y.groupby(["synset"], as_index=False).apply(
            lambda x: x.sample(frac=1.0, random_state=rng))
        _y_train.index = _y_train.index.droplevel()
        _y_test = _y.drop(_y_train.index)
    else:
        raise NotImplementedError

    _y_train.to_csv(join(_output_dir, "y_train.txt"), sep=" ")
    _y_test.to_csv(join(_output_dir, "y_test.txt"), sep=" ")


def phase5(_config):
    logging.info("Phase 5 starting")
    _cached_output_dir = find_cached_output(_config["output"], _config["input"], _config["params"])

    if _cached_output_dir is not None:
        return _cached_output_dir

    return run_phase5(_config)
