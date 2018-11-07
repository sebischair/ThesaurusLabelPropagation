#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from os.path import join
import os
import subprocess
import datetime
import json
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rng-nums", type=int, action="append",
                        nargs="*")  # Specify all rng thesaurus sampling rng_nums that should be run on
    parser.add_argument("-p", "--path-absolute-propagator-folder", type=str, required=True)
    parser.add_argument("-c", type=str, action="append", nargs="*")  # Config modifier
    args = parser.parse_args()

    xx_runs_prefix = "multi_runs_{}".format(
        datetime.datetime.today().strftime('%Y%m%d-%H%M%S'))
    output_dir_xx_prefix = join(args.path_absolute_propagator_folder, "output/XX_runs",
                                xx_runs_prefix)
    os.mkdir(output_dir_xx_prefix)

    if args.rng_nums and len(args.rng_nums) > 0:
        rng_nums = args.rng_nums[0]
    else:
        rng_nums = [1,2,3]

    for rng_num in rng_nums:
        p = subprocess.Popen(
            "{} -x {} -p {} -c '{}' '{}'".format(
                join(args.path_absolute_propagator_folder, "src", "main.py"),
                xx_runs_prefix,
                args.path_absolute_propagator_folder,
                "' '".join(args.c[0]),
                "phase5.params.rng_num={}".format(rng_num)
            ),
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while True:
            line = p.stdout.readline().decode().rstrip()
            print(line)
            if p.poll() is not None:
                break

    # Get accuracies from each run
    dir_names = [join(output_dir_xx_prefix, name) for name in os.listdir(output_dir_xx_prefix) if
                 os.path.isdir(join(output_dir_xx_prefix, name))]
    all_stats = []
    for dir_name in dir_names:
        with open(join(dir_name, "config.json"), "r") as config_file:
            config = json.load(config_file)
        with open(join(config["phase8"]["output"], "stats.json"), "r") as stats_file:
            stats = json.load(stats_file)
        all_stats.append(stats)

    mean_accuracy = np.mean([stats["accuracy"] for stats in all_stats])

    with open(join(output_dir_xx_prefix, "all_stats.json"), "w") as all_stats_file:
        all_stats_summary = {
            "mean_accuracy": mean_accuracy,
            "c": args.c,
            "rng_nums": rng_nums,
            "all_stats": all_stats
        }
        json.dump(all_stats_summary, all_stats_file, indent=2)
        all_stats_file.write("\n")
        print(all_stats_summary)
