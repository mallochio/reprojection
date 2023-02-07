#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   sync_files.py
@Time    :   2023/02/07 11:34:48
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Script to generate the synced filenames for the dataset, choosing the omni image as the reference
'''

import os
import argparse
import numpy as np
from icecream import ic
from tqdm import tqdm


def write_synced_filenames(synced_filenames, output_dir, num_kinects):
    with open(os.path.join(output_dir, "synced_filenames.txt"), "w") as file:
        for i in range(num_kinects):
            file.write(f"capture{i};")
        file.write("omni\n")
        for row in synced_filenames:
            file.write(";".join(row) + "\n")


def get_synced_filenames(base_dir, output_dir):
    # shots.txt contains the timestamps of the synchronized images in each directory, this should be starting point for the sync
    with open(os.path.join(base_dir, "shots.txt"), "r") as file:
        # Skip the first line, read the second and format it into a list of lists
        for line in file:
            if line.startswith("/media"):
                line = line.strip().split(";")
                num_kinects = len(line) - 1 
                shots = [os.path.basename(path) for path in line]

    synced_filenames = [shots]

    # Get the timestamps of the images in the omni directory
    omni_filenames = sorted([filename for filename in os.listdir(os.path.join(base_dir, "omni")) if filename.endswith(".jpg")])
    omni_timestamps = np.array([int(omni_filename.split(".")[0]) for omni_filename in omni_filenames])
    
    # Get the timestamps of the images in each kinect directory
    kinect_filenames = [sorted([filename for filename in os.listdir(os.path.join(base_dir, f"capture{i}/rgb")) if filename.endswith(".jpg")]) for i in range(num_kinects)]
    kinect_timestamps = np.array([np.array([int(filename.split(".")[0]) for filename in kinect_filenames[i]]) for i in range(num_kinects)], dtype=object,)

    # Trim timestamps to be after the timestamps in the shots.txt file
    omni_reference_timestamp = int(shots[-1].split('.')[0])
    omni_timestamps = omni_timestamps[omni_timestamps > omni_reference_timestamp]

    for i in range(num_kinects):
        kinect_reference_timestamp = int(shots[i].split('.')[0])
        mask = kinect_timestamps[i] > kinect_reference_timestamp
        kinect_timestamps[i] = kinect_timestamps[i][mask]

    # Run through the omnidirectional images and find the nearest image in each directory, and keep a running delta
    print("[*] Syncing files")
    for omni_timestamp in tqdm(omni_timestamps):
        delta = omni_timestamp - int(synced_filenames[-1][-1].split(".")[0])

        # Now we find the corresponding images in the other directories according to the delta
        for i in range(num_kinects):
            kinect_timestamp_approx = int(synced_filenames[-1][0].split(".")[0]) + delta

            # Find the nearest timestamp in the kinect directory
            kinect_timestamp = min(kinect_timestamps[i], key=lambda x:abs(x-kinect_timestamp_approx))
            kinect_filename = f"{kinect_timestamp}.jpg"
            if i == 0:
                synced_filenames.append([kinect_filename])
            else:
                synced_filenames[-1].append(kinect_filename)
                synced_filenames[-1].append(f"{omni_timestamp}.jpg")

    print("[*] Writing to file")
    write_synced_filenames(synced_filenames, args.output_dir, num_kinects)
    print("[*] Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to generate the synced filenames for the dataset, choosing the omni image as the reference")
    parser.add_argument(
        "--base_dir", 
        type=str, 
        help="The base directory of the dataset",
        default="/home/sid/Projects/OmniScience/dataset/session-recordings/2022-10-07/at-paus/bedroom/sid"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The directory to save the synced filenames",
        default="/home/sid/Projects/OmniScience/dataset/session-recordings/2022-10-07/at-paus/bedroom/sid"
    )
    args = parser.parse_args()
    get_synced_filenames(args.base_dir, args.output_dir)    
