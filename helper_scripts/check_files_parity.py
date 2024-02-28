#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   check_files_parity.py
@Time    :   2024/02/27 13:08:21
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Script to check if the number of rgb images equal the number of raw images, the number of masks, and the number of keypoints
'''

import os
import sys
import logging
import glob

ROOT_DIR = "/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego"

# Set up basic logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='parity_check.log', 
    filemode='a' 
)

def count_files(directory):
    return len(glob.glob(os.path.join(directory, '*')))

def check_files_parity():
    for root, dirs, files in os.walk(ROOT_DIR):
        if "calib" in root:
            dirs.clear()
            continue
        if "rgb" in dirs:
            n = os.path.basename(root)[-1]
            out_dir = os.path.join(root, f"out_capture{n}")
            paths = {
                'rgb': os.path.join(root, "rgb"),
                'raw_frames': os.path.join(out_dir, "rgb_preprocess", "raw_frames"),
                'masks': os.path.join(out_dir, "rgb_preprocess", "masks"),
                'op_keypoints': os.path.join(out_dir, "rgb_preprocess", "op_keypoints"),
                'op_frames': os.path.join(out_dir, "rgb_preprocess", "op_frames")
            }

            # Check if all directories exist and get the count of files
            file_counts = {key: count_files(path) for key, path in paths.items() if os.path.isdir(path)}
            all_counts_same = len(set(file_counts.values())) <= 1

            if not all_counts_same:
                logging.error("*==============================*")
                logging.error(f"File count mismatch in directory: {out_dir}")
                for key, count in file_counts.items():
                    logging.error(f"{key}: {count}")
                logging.error("*==============================*")
                logging.error("\n")

            dirs.clear()                

def main():
    check_files_parity()

if __name__ == "__main__":
    main()