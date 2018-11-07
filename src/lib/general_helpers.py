import logging
from os import listdir, mkdir
import json
from os.path import isdir, join
import datetime

import sys

START_TIME = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')


def get_start_time():
    return START_TIME


def find_cached_output(_output, _input, _params):
    _cached_output_dirs = [join(_output, name) for name in listdir(_output) if isdir(join(_output,name))]

    for _cached_output_dir in _cached_output_dirs:
        with open(join(_cached_output_dir,"config.json")) as _config_file:
            _config = json.load(_config_file)

        # if multiple inputs
        if isinstance(_input, dict):
            _input_check = _config["inputs"] == _input
        else:
            _input_check = _config["input"] == _input

        if _input_check and _config["params"] == _params:
            logging.info("Cached output for current phase exists:\n{}".format(_cached_output_dir))
            return _cached_output_dir

    return None # No cached output yet


def make_absolute_input_output(_root_dir, _config):
    _new_config = dict()
    for _phase, _phase_config in _config.items():
        _new_config[_phase] = dict()

        for _key, _value in _phase_config.items():
            if _key == "input" or _key == "output":
                _value = join(_root_dir,_value)
            elif _key == "inputs":
                _value = {_k: join(_root_dir, _v) for (_k, _v) in _value.items()}

            _new_config[_phase][_key] = _value

    return _new_config


def prepare_output_dir(_config):
    _output_dir = join(_config["output"], get_start_time())
    mkdir(_output_dir)

    _config["output"] = _output_dir
    with open(join(_output_dir, "config.json"), "w") as _output_config_file:
        json.dump(_config, _output_config_file, indent=2)
        _output_config_file.write("\n")  # Add newline because Python JSON does not
    return _output_dir


def set_up_custom_logger(_config_dir):
    logging.captureWarnings(True)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler(join(_config_dir, 'log.txt'), mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    _logger = logging.getLogger()
    _logger.setLevel(logging.INFO)
    _logger.addHandler(handler)
    _logger.addHandler(screen_handler)
    _logger.info("Logger for configDir " + _config_dir + " created")
    _logger.info(sys.path)
    return _logger
