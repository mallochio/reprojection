import os
import json


def load_config(other_file = None):
    here_dir = os.path.dirname(os.path.abspath(__file__))
    default_file = here_dir + '/config.json'

    if other_file is not None:
        config_file = other_file
    else:
        config_file = default_file
    
    print(f'Opening config.json file at: "{config_file}" ...')
    
    with open(config_file, 'r') as fh:
        return json.load(fh)