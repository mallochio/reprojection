#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   Untitled-1
@Time    :   2023/03/09 20:38:11
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Get dataset statistics
'''

import os
import argparse
from pprint import pprint

def count_folders(path, level):
    folder_count = 0
    dirset = set()
    for root, dirs, files in os.walk(path):
        if "calib" in root:
            continue
        _, subsequent = root.split(path)
        depth = len(subsequent.split("/"))
        if depth == level:
            if "calib" in dirs:
                dirs.remove("calib")
            folder_count += len(dirs)
            for i in dirs:
                dirset.add(i)
    return folder_count, dirset


def make_dictionary_of_filecounts(path):
    # Make a dictionary of the number of files in subfolders if it is in the folder named either of "rgb", "ir", "depth", or "omni"
    #Keep a running total of the number of files in the folders
    rgb_count = 0
    ir_count = 0
    depth_count = 0
    omni_count = 0
    for root, dirs, files in os.walk(path):
        # only check the files in the folder if it is not in a subfolder of "calib"
        if  "calib" in root:
            continue
        if "rgb" in dirs:
            rgb_count += len(os.listdir(os.path.join(root, "rgb")))
        if "ir" in dirs:
            ir_count += len(os.listdir(os.path.join(root, "ir")))
        if "depth" in dirs:
            depth_count += len(os.listdir(os.path.join(root, "depth")))
        if "omni" in dirs:
            omni_count += len(os.listdir(os.path.join(root, "omni")))
            
    return {"rgb": rgb_count, "ir": ir_count, "depth": depth_count, "omni": omni_count}


def main(path, level, get_dictionary=False):
    if level:
        pprint(count_folders(path, level))
    if get_dictionary:
        pprint(make_dictionary_of_filecounts(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count the number of folders at a particular level in a directory tree")
    parser.add_argument("path", type=str, help="Path to the directory tree")
    parser.add_argument("--level", type=int, default=0, help="Level of the directory tree to count folders at")
    parser.add_argument("--get_dictionary", action="store_true", default=False, help="Get a dictionary of the number of files in each folder")
    args = parser.parse_args()
    main(args.path, args.level, args.get_dictionary)
    