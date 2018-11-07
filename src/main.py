#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from os.path import join
from os import mkdir
import logging
import dpath.util
from ast import literal_eval

from base_config import BASE_CONFIG
from lib.general_helpers import make_absolute_input_output, get_start_time, set_up_custom_logger
from lib.phase1 import phase1
from lib.phase2 import phase2
from lib.phase3 import phase3
from lib.phase4 import phase4
from lib.phase5 import phase5
from lib.phase6 import phase6
from lib.phase7 import phase7
from lib.phase8 import phase8

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path-absolute-propagator-folder", type=str, required=True)
    parser.add_argument("-x", "--xx-runs-prefix", type=str, required=False)  # Optional prefix within XX_runs folder
    parser.add_argument("-c", type=str, action="append", nargs="*")  # Config modifier
    args = parser.parse_args()

    config = make_absolute_input_output(args.path_absolute_propagator_folder, BASE_CONFIG)

    if args.c:
        for c in args.c[0]:
            path, raw_value = c.split("=", 1)  # path=value
            value = json.loads(raw_value) if "{" in raw_value or "[" in raw_value else literal_eval(raw_value)
            
            # do a lookup if path exists in config. If not, dpath will raise a KeyError
            dpath.get(config, path, separator=".")

            dpath.set(config, path, value, separator=".")

    if args.xx_runs_prefix:
        output_dir_run = join(args.path_absolute_propagator_folder, "output/XX_runs",
                              args.xx_runs_prefix,
                              get_start_time())
    else:
        output_dir_run = join(args.path_absolute_propagator_folder, "output/XX_runs",
                              get_start_time())
    mkdir(output_dir_run)
    set_up_custom_logger(output_dir_run)

    try:
        # Phase 1: Corpus preprocessing
        # Depending on: Nothing
        # Input: Folder with RW40... JSON files by DATEV
        # Output: Single file with contents from all files, now preprocessed

        config["phase1"]["output"] = phase1(config["phase1"])

        # Phase 2: Embedding generation
        # Depending on: Phase 1

        config["phase2"]["input"] = config["phase1"]["output"]
        config["phase2"]["output"] = phase2(config["phase2"])

        # Phase 3: Graph generation
        # Depending on: Phase 2

        config["phase3"]["input"] = config["phase2"]["output"]
        config["phase3"]["output"] = phase3(config["phase3"])

        # Phase 4: Thesaurus preprocessing
        # Depending on: Phase 2
        # Input: Thesaurus by DATEV
        # Output: File with list of (word, synset) tuples

        config["phase4"]["inputs"]["word_embeddings"] = config["phase2"]["output"]
        config["phase4"]["output"] = phase4(config["phase4"])

        # Phase 5: Thesaurus sampling
        # Depending on: Phase 4

        config["phase5"]["input"] = config["phase4"]["output"]
        config["phase5"]["output"] = phase5(config["phase5"])

        # Phase 6: Graph labeling (with training data)
        # Depending on: Phase 3 & Phase 5

        config["phase6"]["inputs"]["corpus_graph"] = config["phase3"]["output"]
        config["phase6"]["inputs"]["thesaurus"] = config["phase5"]["output"]
        config["phase6"]["output"] = phase6(config["phase6"])

        # Phase 7: Label propagation
        # Depending on: Phase 6

        config["phase7"]["input"] = config["phase6"]["output"]
        config["phase7"]["output"] = phase7(config["phase7"])

        # Phase 8: Evaluation
        # Depending on: Phase 5 & Phase 7

        config["phase8"]["inputs"]["thesaurus"] = config["phase5"]["output"]
        config["phase8"]["inputs"]["predictions"] = config["phase7"]["output"]
        config["phase8"]["output"] = phase8(config["phase8"])
        logging.info("Run finished")
    except Exception as e:
        logging.error("An error occurred")
        logging.error(e)
        raise e
    finally:
        with open(join(output_dir_run, "config.json"), "w") as output_dir_run_config_file:
            json.dump(config, output_dir_run_config_file, indent=2)
            output_dir_run_config_file.write("\n")  # Add newline because Python JSON does not
