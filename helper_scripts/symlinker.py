#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   Untitled-1
@Time    :   2023/03/19 21:21:38
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Go through the file containing the synced filenames from the different directories and make symlinks
'''

import os
import argparse
import numpy as np
from tqdm import tqdm
import shutil
from pprint import pprint


def parse_basedir_for_subsequence(basedir):
    outdir = "/home/Dataset/4a10" 
    shots_array = []
    with open(os.path.join(basedir, "synced_filenames.txt"), "r") as file:
        # Skip the first line, read the second and format it into a list of lists
        for line in file:
            if line.startswith("capture"):
                line = line.strip().split(";")
                num_kinects = len(line) - 1
                continue
            for i in range(num_kinects):
                shots = line.strip().split(";")
            for j in range(num_kinects):
                shots[j] = os.path.join(basedir, f"capture{j}", "rgb", shots[j])
            shots[-1] = os.path.join(basedir, "omni", shots[-1])
            shots_array.append(shots)
    
    # choose the middle of the array and get a 50 image subsequence
    middle = len(shots_array) // 2
    shots_array_subsequence = shots_array[middle + 100 : middle + 150]
    pprint(shots_array_subsequence)
    # Now make four folders for each camera and copy the images accordingly
    for i in range(num_kinects):
        os.makedirs(os.path.join(outdir, f"capture{i}/rgb"), exist_ok=True)
        for j in shots_array_subsequence:
            shutil.copy(j[i], os.path.join(outdir, f"capture{i}/rgb"))

    # Now make a folder for the omnidirectional camera and copy the images accordingly
    os.makedirs(os.path.join(outdir, "omni"), exist_ok=True)
    for j in shots_array_subsequence:
        shutil.copy(j[-1], os.path.join(outdir, "omni"))


def parse_basedir(basedir):
    shots_array = []
    with open(os.path.join(basedir, "synced_filenames.txt"), "r") as file:
        # Skip the first line, read the second and format it into a list of lists
        for line in file:
            if line.startswith("capture"):
                line = line.strip().split(";")
                num_kinects = len(line) - 1
                continue
            for i in range(num_kinects):
                shots = line.split(";")
            for j in range(num_kinects):
                shots[j] = os.path.join(basedir, f"capture{j}", "rgb", shots[j])
            shots[-1] = os.path.join(basedir, "omni", shots[-1])
            shots_array.append(shots)

    return shots_array

def write_synced_files(shots_array, out_dir):
    for i in tqdm(range(0, len(shots_array), 100)):
        shots = shots_array[i]
        out_dir = os.path.join(out_dir, f"set{i}")
        os.makedirs(out_dir, exist_ok=True)
        for j in shots:
            os.symlink(j, os.path.join(out_dir, os.path.basename(j)))
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/sid/Desktop/Paper stuff/a10",
        help="Directory containing the synced filenames file and the directories of the different cameras",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/sid/Desktop/Paper stuff/outputs",
        help="Directory to write the synced filenames file and the directories of the different cameras",
    )
    args = parser.parse_args()
    parse_basedir_for_subsequence(args.base_dir)
    # write_synced_files(parse_basedir(args.base_dir), args.output_dir)


if __name__ == "__main__":
    main()