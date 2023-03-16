#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Convert a pickle file of intrinsics to JSON for HuMoR.
"""

import json
import pickle
import numpy as np
import argparse

def main(pickle_path: str):
    with open(pickle_path, 'rb') as f:
        intrinsics = pickle.load(f)
    intrinsics = np.array(intrinsics['intrinsics'])
    intrinsics = intrinsics.tolist()
    with open(f'{pickle_path.split(".")[0]}.json', 'w') as f:
        json.dump(intrinsics, f)
        print(f"[*] Saved {pickle_path.split('.')[0]}.json")


parser = argparse.ArgumentParser(description='Convert a pickle file of intrinsics to JSON for HuMoR.')
parser.add_argument('pickle', type=str, help='Path to the pickle file.')
args = parser.parse_args()
main(args.pickle)
