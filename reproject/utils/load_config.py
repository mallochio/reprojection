import json


def load_config(other_file = None):
    config_file = 'config.json'
    if other_file is not None:
        config_file = other_file
    with open(config_file, 'r') as fh:
        return json.load(fh)
