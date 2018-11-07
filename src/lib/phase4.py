import json
import shutil
import subprocess
import tempfile
import os
from os.path import join
import logging

import pandas as pd
import time
from itertools import chain

from lib.general_helpers import prepare_output_dir, find_cached_output

# Phase 4: Thesaurus preprocessing
# Depending on: Phase 2
# Input: Thesaurus by DATEV
# Output: File with list of (word, synset) tuples

# Expected format of the thesaurus file:
# [ { "Class": string, "Concept": number, "Keys": [string], ... } ]
# See `ipynbs/02_thesaurus_stats.ipynb` for the steps in converting the original DATEV file into the required format

def run_phase4(_config):
    _output_dir = prepare_output_dir(_config)
    try:
        run_phase4_core(_config["inputs"], _config["params"], _output_dir)
        return _output_dir
    except Exception as e:
        shutil.rmtree(_output_dir)
        raise e


def run_phase4_core(_inputs, params, _output_dir):
    thesaurus_file_name = _inputs["thesaurus"]
    word_embeddings_file_name = _inputs["word_embeddings"]
    with open(thesaurus_file_name) as f:
        thesaurus_concepts = json.load(f)
    synonym_concepts = [c for c in thesaurus_concepts if c["Class"] == "synonym"]

    # Preprocessing steps 1 and 2
    for synonym_concept in synonym_concepts:
        if "transform_to_lowercase" in params["actions"]:
            synonym_concept["Keys"] = [key.lower() for key
                                       in synonym_concept["Keys"]]

        if "remove_keys_with_space_or_hyphen" in params["actions"]:
            synonym_concept["Keys"] = [key for key
                                       in synonym_concept["Keys"]
                                       if " " not in key
                                       and "-" not in key]
        elif "remove_keys_with_space_keep_hyphen" in params["actions"]:
            synonym_concept["Keys"] = [key for key
                                       in synonym_concept["Keys"]
                                       if " " not in key]

        if "replace_all_sz" in params["actions"]:
            synonym_concept["Keys"] = [key.replace(u"ß", 'ss') for key
                                       in synonym_concept["Keys"]]
    synonym_concepts = [x for x in synonym_concepts if len(x["Keys"]) > 0]

    # Preprocessing step 3
    # we sort synonym_concepts by number of keys (ascending) in the concept
    # so that on the conversion to dict,
    # the word will map to the concept with more keys
    # example: `dict([('B', 1), ('B', 2), ('A', 2), ('C', 3)])` will map 'B' to 2
    if "create_n_1_mapping_largest_sysnet_wins_key" in params["actions"]:
        synonym_concepts = sorted(
            synonym_concepts,
            key=lambda _concept: len(_concept["Keys"]))

        list_of_keylists = [
            [(key, x["Concept"]) for key in x["Keys"]]
            for x in synonym_concepts]
        wordlist = list(chain.from_iterable(list_of_keylists))

        # Wordset is the one we want - every key (word) appears there just once
        wordset = dict(wordlist)  # dict({word: conceptNumber})
    else:
        raise NotImplementedError

    # Preprocessing step 4
    if "remove_keys_not_in_corpus" in params["actions"]:
        thesaurus_words_with_concepts = tempfile.NamedTemporaryFile(mode="w+")
        sorted_thesaurus_words_without_concepts = tempfile.NamedTemporaryFile("w+")

        for word, concept in wordset.items():
            thesaurus_words_with_concepts.write("{} {}\n".format(word, concept))
        thesaurus_words_with_concepts.flush()
        os.fsync(thesaurus_words_with_concepts.fileno())

        cmd1 = subprocess.getoutput(
            "cat {} | awk '{{print $1;}}' | sort > {}".format(
                thesaurus_words_with_concepts.name,
                sorted_thesaurus_words_without_concepts.name))
        logging.info(time.strftime("%H:%M:%S") + " cmd1" + cmd1)

        word_embeddings_sorted_word_list = tempfile.NamedTemporaryFile("w+")
        cmd2 = subprocess.getoutput(
            "cat {} | tail -n +2 | awk '{{print $1;}}' | sort > {}".format(
                join(word_embeddings_file_name, "main.txt"),
                word_embeddings_sorted_word_list.name
            ))
        logging.info(time.strftime("%H:%M:%S") + " cmd2" + cmd2)

        thesaurus_words_also_in_corpus = tempfile.NamedTemporaryFile("w+")
        cmd3 = subprocess.getoutput(
            "comm -12 {} {} > {}".format(
                sorted_thesaurus_words_without_concepts.name,
                word_embeddings_sorted_word_list.name,
                thesaurus_words_also_in_corpus.name
            ))
        logging.info(time.strftime("%H:%M:%S") + " cmd3" + cmd3)

        sorted_thesaurus_words_without_concepts.close()
        word_embeddings_sorted_word_list.close()

        thesaurus_words_also_in_corpus_with_concepts = tempfile.NamedTemporaryFile("w+")
        cmd4 = subprocess.getoutput(
            "awk 'NR==FNR{{a[$0]=$0}}NR>FNR{{if($1==a[$1])print $0}}' {} {} > {}".format(
                thesaurus_words_also_in_corpus.name,
                thesaurus_words_with_concepts.name,
                thesaurus_words_also_in_corpus_with_concepts.name
            ))
        logging.info(time.strftime("%H:%M:%S") + " cmd4" + cmd4)

        thesaurus_words_also_in_corpus.close()
        thesaurus_words_with_concepts.close()

        thesaurus_after4 = pd.read_table(
            thesaurus_words_also_in_corpus_with_concepts.name,
            sep=" ",
            header=None,
            names=["word", "synset"],
            index_col=0)
        thesaurus_words_also_in_corpus_with_concepts.close()
    else:
        raise NotImplementedError

    # Preprocessing step 5
    if "remove_synsets_with_less_than_two_keys" in params["actions"]:
        synset_value_counts = thesaurus_after4["synset"].value_counts()
        matches = thesaurus_after4["synset"].isin(synset_value_counts[synset_value_counts > 1].index)
        thesaurus_after5 = thesaurus_after4[matches]
    else:
        raise NotImplementedError

    thesaurus_after5.to_csv(join(_output_dir, "main.txt"), sep=" ")


def phase4(_config):
    logging.info("Phase 4 starting")
    _cached_output_dir = find_cached_output(_config["output"], _config["inputs"], _config["params"])

    if _cached_output_dir is not None:
        return _cached_output_dir

    return run_phase4(_config)
