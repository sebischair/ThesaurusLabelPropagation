import json
import logging
import shutil
from os.path import join

import numpy as np
import pandas as pd
from sklearn import metrics

from lib.general_helpers import prepare_output_dir, find_cached_output

# Phase 8: Evaluation
# Depending on: Phase 5 & Phase 7

def run_phase8(_config):
    _output_dir = prepare_output_dir(_config)
    try:
        run_phase8_core(_config["inputs"], _config["params"], _output_dir)
        return _output_dir
    except Exception as e:
        shutil.rmtree(_output_dir)
        raise e


def evaluate(_df_evaluation):
    df_where_test_possible = _df_evaluation[_df_evaluation["y_test"].notnull()]
    y_true = df_where_test_possible["y_test"]
    y_pred = df_where_test_possible["y_pred"]
    if len(df_where_test_possible) == 0:
        within_top3_mean = None
    else:
        within_top3_mean = df_where_test_possible.apply(lambda row: row["y_test"] in row["y_top3_classes"],
                                                        axis=1).mean()

    pred_val_counts = _df_evaluation.loc[_df_evaluation["y_pred"] != -1, "y_pred"].value_counts()

    _stats = {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "conf_stats": {
            "mean_conf_true": df_where_test_possible[y_pred == y_true]["y_conf"].mean(),
            "mean_conf_false": df_where_test_possible[y_pred != y_true]["y_conf"].mean(),
        },
        "mean_within_top3": within_top3_mean,
        "pred_synsets_stats": pred_val_counts.describe().to_dict(),
        "test_stats": dict()
    }

    _stats["pred_synsets_stats"]["sum_preds"] = int(pred_val_counts.sum())
    _stats["pred_synsets_stats"]["max_group_id"] = int(pred_val_counts.index[0])

    _stats["test_stats"]["count_all_words"] = int(len(_df_evaluation))
    _stats["test_stats"]["count_train_set"] = int(len(_df_evaluation[_df_evaluation["y_train"] != -1]))
    _stats["test_stats"]["count_test_set"] = int(len(df_where_test_possible))
    _stats["test_stats"]["count_correctly_predicted"] = int(len(df_where_test_possible[y_pred == y_true]))

    logging.info("Evaluation stats:\n{}".format(_stats))

    return _stats


def run_phase8_core(_inputs, params, _output_dir):
    if params["variant"] != "normal":
        raise Exception

    def clean(seq_string):
        tmp = list(map(int, seq_string.strip("[]").split(',')))
        return tmp

    _y_pred_conf_top3 = pd.read_table(
        join(_inputs["predictions"], "main.txt"),
        sep=" ",
        dtype={"y_pred": np.int32},
        converters={"y_top3_classes": clean},
        index_col=0)

    _y_train = pd.read_table(
        join(_inputs["thesaurus"], "y_train.txt"),
        sep=" ",
        dtype={"synset": np.int32},
        index_col=0)

    _y_test = pd.read_table(
        join(_inputs["thesaurus"], "y_test.txt"),
        sep=" ",
        dtype={"synset": np.int32},
        index_col=0)

    _df_evaluation = _y_pred_conf_top3.join(_y_train, how="left").fillna(-1, downcast="infer")
    _df_evaluation.rename(columns={"synset": "y_train"}, inplace=True)

    _df_evaluation = _df_evaluation.join(_y_test, how="left")
    _df_evaluation.rename(columns={"synset": "y_test"}, inplace=True)

    _df_evaluation = _df_evaluation[["y_train", "y_pred", "y_conf", "y_top3_classes", "y_test"]]

    _df_evaluation.to_csv(join(_output_dir, "main.txt"), sep=" ")
    with open(join(_output_dir, "stats.json"), "w") as stats_file:
        json.dump(evaluate(_df_evaluation), stats_file, indent=2)
        stats_file.write("\n")  # Add newline because Python JSON does not


def phase8(_config):
    logging.info("Phase 8 starting")
    _cached_output_dir = find_cached_output(_config["output"], _config["inputs"], _config["params"])

    if _cached_output_dir is not None:
        return _cached_output_dir

    return run_phase8(_config)
