from os.path import join

import numpy as np
import pandas as pd


def get_train_test(path):
    train = pd.read_table(
        join(path, "y_train.txt"),
        sep=" ",
        dtype={"synset": np.int32},
        index_col=0)["synset"]
    test = pd.read_table(
        join(path, "y_test.txt"),
        sep=" ",
        dtype={"synset": np.int32},
        index_col=0)["synset"]
    return train, test


def df_evaluation_init(index, train, pred, test):
    df_predicted = pd.DataFrame(index=index).rename_axis("word")

    df_predicted = df_predicted.join(train, how="left").fillna(-1).rename(columns={"synset": "y_train"})

    df_predicted = df_predicted.assign(y_pred=pred["synset"])
    df_predicted = df_predicted.assign(y_conf=pred["current_max_similarity"])

    df_predicted = df_predicted.assign(y_top3_classes=[[-1, -1, -1]] * len(df_predicted))

    df_evaluation = df_predicted.join(test, how="left")
    df_evaluation.rename(columns={"synset": "y_test"}, inplace=True)
    return df_evaluation


def apply_synset_prediction_if_more_similar(pred, synset_id, predictions):
    for (word, similarity) in predictions:
        if pred.loc[word].current_max_similarity < similarity:
            pred.loc[word, "synset"] = synset_id
            pred.loc[word, "current_max_similarity"] = similarity