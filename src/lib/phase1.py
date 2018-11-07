import json
import logging
import re
import shutil
import subprocess
from genericpath import isfile
from os import listdir
from os.path import join
from tempfile import TemporaryDirectory

from lib.general_helpers import prepare_output_dir, find_cached_output

# Phase 1: Corpus preprocessing
# Depending on: Nothing
# Input: Folder with RW40... JSON files
# Output: Single file with contents from all files, now preprocessed

# Expected format an individual `RW40_**.json` file:
# { "Volltext": string, ... }

def run_phase1(_config):
    _output_dir = prepare_output_dir(_config)
    try:
        run_phase1_core(_config["input"], _config["params"], _output_dir)
        return _output_dir
    except Exception as e:
        shutil.rmtree(_output_dir)
        raise e


def run_phase1_core(_input, params, _output_dir):
    _rw_folders = [f for f in listdir(_input) if isfile(join(_input, f)) and f != ".DS_Store"]

    with TemporaryDirectory(prefix="thesaurus_propagator") as _tmp_cleantxts_dir:
        _counter = 0
        # loop files
        for _f in _rw_folders:
            if _counter % 10000 == 0:
                logging.info(_counter)

            # load data
            with open(join(_input, _f), 'r') as fh:
                data = json.load(fh)

            text_data = data['Volltext']
            _content = text_data.split(_f.replace(".json", "").replace("RW40_", ""))[0]

            # save to file
            if "remove_newline_and_carriage_returns" in params["actions"]:
                _content = _content.replace(u"\n", '')
                _content = _content.replace(u"\r", '')

            if "replace_sz_in_muss" in params["actions"]:
                _content = _content.replace(u"muß", 'muss')
            elif "replace_all_sz" in params["actions"]:
                _content = _content.replace(u"ß", 'ss')

            if "replace_paragraphsign_with_word" in params["actions"]:
                _content = _content.replace(u"§", 'PARAGRAPHSIGN ')

            # discard characters options
            if "replace_non_german_chars_with_space" in params["actions"]:
                # replace non-alphanumeric chars with space, keep only German characters
                _content = re.sub(u"[^a-zA-ZüöäÜÖÄß]", " ", _content)
            elif "replace_punctuation_with_space" in params["actions"]:
                # replace non-alphabetic chars with space, keep all alphabetic characters
                # see https://stackoverflow.com/a/16799238/3327577
                # added replacement of underscore manually as it would not be replaced by original regex
                _content = re.sub(u'[^\w\s]|[0-9]|[_]', " ", _content)
            elif "replace_punctuation_with_space_except_hyphen" in params["actions"]:
                # replace non-alphabetic chars with space, keep all alphabetic characters
                # and hyphens (but only hyphens between characters)
                _content = re.sub(u'[^\w\s-]|[0-9]|[_]', " ", _content) # keep all hyphens
                # discard hyphens that are not between characters
                # Note: For a word `--abc--`, the result will be `-abc-`, not `abc`. Same for `--abc` => `-abc`.
                _content = re.sub(u'(-\s-)|(\s-)|(-\s)', " ", _content)

            # split along spaces and reassemble
            words = _content.split(u" ")
            if "discard_words_less_two_characters" in params["actions"]:
                words = [w for w in words if len(w) >= 2]
            if "transform_to_lowercase" in params["actions"]:
                words = [w.lower() for w in words]
            _content = u" ".join(words)

            _cleantxt_name = _f.replace(".json", "").replace("RW40_", "")
            with open(join(_tmp_cleantxts_dir, _cleantxt_name), "w") as fh:
                fh.write(_content + " ")

            _counter += 1

            if params["DEV_max_texts"] and _counter == params["DEV_max_texts"]:
                logging.info("DEV_max_texts")
                logging.info(params["DEV_max_texts"])
                break

        if "save_as_single_line" in params["actions"]:
            cmd = subprocess.getoutput(
                "find {} -type f | xargs cat > {}".format(
                    _tmp_cleantxts_dir,
                    join(_output_dir, "main.txt")
                ))
        elif "save_as_multiple_lines" in params["actions"]:
            cmd = subprocess.getoutput(
                "find {} -type f | xargs -I{{}} sh -c \"cat {{}}; echo ''\" > {}".format(
                    _tmp_cleantxts_dir,
                    join(_output_dir, "main.txt")
                ))
        else:
            raise Exception
        logging.info(cmd)


def phase1(_config):
    logging.info("Phase 1 starting")
    _cached_output_dir = find_cached_output(_config["output"], _config["input"], _config["params"])

    if _cached_output_dir is not None:
        return _cached_output_dir

    return run_phase1(_config)
