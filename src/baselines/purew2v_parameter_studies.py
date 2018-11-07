#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import datetime
import logging
import sys
sys.path.append("..")  # so importing a package from the top-folder "lib" works, see https://stackoverflow.com/a/45874916/3327577
import json
import os.path

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from baselineExtendSynsetVector import baseline_synset_vector
from helpers import get_train_test, df_evaluation_init, apply_synset_prediction_if_more_similar
from lib.phase8 import evaluate

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))


if __name__ == "__main__":
    start_time = datetime.datetime.today().strftime('%Y%m%d-%H%M')

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--path-embeddings-file", type=str, required=True)
    parser.add_argument("-p1", type=str, required=True)  # thesaurus path 1
    parser.add_argument("-p2", type=str, required=False)  # thesaurus path 2
    parser.add_argument("-p3", type=str, required=False)  # thesaurus path 3
    parser.add_argument("-m", "--baseline-method", type=str, required=True)
    parser.add_argument("-s", "--save-evaluation-file", type=bool, required=False, default=False)

    args = parser.parse_args()
    stats_file = "{}-stats-baseline{}.json".format(start_time, args.baseline_method)

    if args.baseline_method == "SYNSET_VECTOR":
        baselineMethod = baseline_synset_vector
        parameter = [2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30]
    else:
        raise NotImplementedError

    thesaurus_paths = [args.p1, args.p2, args.p3]
    thesaurus_paths = [p for p in thesaurus_paths if p] # remove paths that are not set

    word_vectors = KeyedVectors.load_word2vec_format(args.path_embeddings_file, binary=False)
    vocab = list(word_vectors.vocab.keys())

    for k in parameter:
        logger.info("k=" + str(k))

        accuracies = []
        for thesaurus_path in thesaurus_paths:
            y_train, y_test = get_train_test(thesaurus_path)

            grouped = y_train.groupby(y_train)
            y_pred = pd.DataFrame(index=vocab,
                                  columns=["synset", "current_max_similarity"],
                                  data={"synset": -1, "current_max_similarity": -100.0})

            for idx, (synset_id, group) in enumerate(grouped):
                synset_predictions = baselineMethod(word_vectors, group.index.tolist(), k)
                apply_synset_prediction_if_more_similar(y_pred, synset_id, synset_predictions)

                if idx % 100 == 0:
                    logger.info(idx)

            df_evaluation = df_evaluation_init(index=vocab, train=y_train, pred=y_pred, test=y_test)
            stats = evaluate(df_evaluation)
            accuracy = stats["accuracy"]
            logger.info("Stats calculated. Accuracy: " + str(accuracy))
            accuracies.append(accuracy)

            if args.save_evaluation_file:
                now = datetime.datetime.today().strftime('%Y%m%d-%H%M')
                df_evaluation = df_evaluation[["y_train", "y_pred", "y_conf", "y_top3_classes", "y_test"]]
                df_evaluation.to_csv("{}-dfEvaluation-baseline{}-{}.txt".format(now, args.baseline_method, k), sep=" ")

        print("For", k, "mean", np.mean(accuracies), ", individual accuracies", accuracies)
        result = {
            "k": k,
            "mean_accuracy": np.mean(accuracies),
            "accuracies": accuracies
        }

        if not os.path.isfile(stats_file):
            stats = {
                "embeddings": args.path_embeddings_file,
                "thesaurus_sampled_paths": thesaurus_paths,
                "baseline_method": args.baseline_method,
                "results": [result]
            }
        else:
            with open(stats_file, "r") as f:
                stats = json.load(f)

        stats["results"].append(result)
        with open(stats_file, "w") as f:
            json.dump(stats, f)
            f.write("\n")  # Add newline because Python JSON does not

    logger.info("Program finished")
