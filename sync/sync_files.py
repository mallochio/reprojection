#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   sync_files.py
@Time    :   2023/02/07 11:34:48
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Script to generate the synced filenames for the dataset, choosing the omni image as the reference
"""

import os
import argparse
import numpy as np
from tqdm import tqdm


filename = "synced_filenames_full.txt"


def write_synced_filenames(synced_filenames, output_dir, num_kinects):
    with open(os.path.join(output_dir, filename), "w") as file:
        for i in range(num_kinects):
            file.write(f"capture{i};")
        file.write("omni\n")
        for row in synced_filenames:
            file.write(";".join(row) + "\n")


def get_synced_filenames_partial(base_dir, output_dir):
    with open(os.path.join(base_dir, "shots.txt"), "r") as file:
        # Skip the first line, read the second and format it into a list of lists
        for ix, line in enumerate(file):
            if ix == 1: #line.startswith("/media"):
                line = line.strip().split(";")
                num_kinects = len(line) - 1
                shots = [os.path.basename(path) for path in line]

    synced_filenames = [shots]

    # Get the timestamps of the images in the omni directory
    omni_filenames = sorted(
        [
            filename
            for filename in os.listdir(os.path.join(base_dir, "omni"))
            if filename.endswith(".jpg")
        ]
    )
    omni_timestamps = np.array(
        [
            int(omni_filename.split(".")[0]) 
            for omni_filename in omni_filenames
        ]
    )

    # Get the timestamps of the images in each kinect directory
    kinect_filenames = [
        sorted(
            [
                filename
                for filename in os.listdir(os.path.join(base_dir, f"capture{i}/rgb"))
                if filename.endswith(".jpg")
            ]
        )
        for i in range(num_kinects)
    ]
    kinect_timestamps = np.array(
        [
            np.array([int(filename.split(".")[0]) for filename in kinect_filenames[i]])
            for i in range(num_kinects)
        ],
        dtype=object,
    )

    # Trim timestamps to be after the timestamps in the shots.txt file
    omni_reference_timestamp = int(shots[-1].split(".")[0])
    omni_timestamps = omni_timestamps[omni_timestamps > omni_reference_timestamp]

    kinect_shots = []
    for i in range(num_kinects):
        kinect_reference_timestamp = int(shots[i].split(".")[0])
        mask = kinect_timestamps[i] > kinect_reference_timestamp
        kinect_timestamps[i] = kinect_timestamps[i][mask]
        kinect_shots.append(kinect_reference_timestamp)

    deltas = np.array(
        [kinect_shots[i] - omni_reference_timestamp for i in range(num_kinects)],
        dtype=object,
    )

    # Now run through the omni images and find the nearest image in each directory after adding the delta
    print("[*] Syncing files")
    for omni_timestamp in tqdm(omni_timestamps):
        for i in range(num_kinects):
            kinect_timestamp_approx = omni_timestamp + deltas[i]
            # get nearest timestamp in the kinect directory
            kinect_timestamp_new = min(
                kinect_timestamps[i], key=lambda x: abs(x - kinect_timestamp_approx)
            )
            kinect_filename = f"{kinect_timestamp_new}.jpg"
            if i == 0:
                synced_filenames.append([kinect_filename])
            else:
                synced_filenames[-1].append(kinect_filename)
        synced_filenames[-1].append(f"{omni_timestamp}.jpg")

    print("[*] Writing to file")
    write_synced_filenames(synced_filenames, args.output_dir, num_kinects)
    print("[*] Done!")


def get_sorted_files(base_dir, num_kinects):
    # Get the timestamps of the images in the omni directory
    omni_filenames = sorted(
        [
            filename
            for filename in os.listdir(os.path.join(base_dir, "omni"))
            if filename.endswith(".jpg")
        ]
    )
    omni_timestamps = np.array([int(omni_filename.split(".")[0]) for omni_filename in omni_filenames])

    # Get the timestamps of the images in each kinect directory
    kinect_filenames = [
        sorted(
            [
                filename
                for filename in os.listdir(os.path.join(base_dir, f"capture{i}/rgb"))
                if filename.endswith(".jpg")
            ]
        )
        for i in range(num_kinects)
    ]
    kinect_timestamps = np.array(
        [
            np.array([int(filename.split(".")[0]) for filename in kinect_filenames[i]])
            for i in range(num_kinects)
        ],
        dtype=object,
    )
    return omni_filenames, omni_timestamps, kinect_filenames, kinect_timestamps    


def get_synced_filenames_full(base_dir, output_dir):
    with open(os.path.join(base_dir, "shots.txt"), "r") as file:
        shots = []
        for ix, line in enumerate(file):
            if ix == 1: #line.startswith("/media"):
                line = line.strip().split(";")
                num_kinects = len(line) - 1
                shots = [os.path.basename(path) for path in line]

    omni_filenames, omni_timestamps, kinect_filenames, kinect_timestamps = get_sorted_files(base_dir, num_kinects)
    
    # Compute deltas based on the shots file
    # omni_start_timestamp = int(omni_filenames[0].split(".")[0])
    kinect_shots = []
    for i in range(num_kinects):
        kinect_start_timestamp = int(shots[i].split(".")[0])
        kinect_shots.append(kinect_start_timestamp)
    omni_reference_timestamp = int(shots[-1].split(".")[0])
    deltas = np.array(
        [kinect_shots[i] - omni_reference_timestamp for i in range(num_kinects)],
        dtype=object,
    )

    synced_filenames = []
    # Now run through the omni images and find the nearest image in each directory after adding the delta
    print("[*] Syncing files")
    for omni_timestamp in tqdm(omni_timestamps):
        omni_filename = f"{omni_timestamp}.jpg"
        synced_shot = []
        for i in range(num_kinects):
            kinect_timestamp_approx = omni_timestamp + deltas[i]
            # get nearest timestamp in the kinect directory
            kinect_timestamp_new = min(
                kinect_timestamps[i], key=lambda x: abs(x - kinect_timestamp_approx)
            )
            kinect_filename = f"{kinect_timestamp_new}.jpg"
            synced_shot.append(kinect_filename)
        synced_shot.append(omni_filename)
        synced_filenames.append(synced_shot)

    print("[*] Writing to file")
    write_synced_filenames(synced_filenames, output_dir, num_kinects)
    print("[*] Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to generate the synced filenames for the dataset, choosing the omni image as the reference"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        help="The base directory of the dataset",
        default="/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The directory to save the synced filenames",
        default="/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/",
    )
    args = parser.parse_args()

    # Recurse through the base directory until we find shots.txt files and then sync the files
    for root, dirs, files in os.walk(args.base_dir):
        if "shots.txt" in files:
            if root == "/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/2023-02-09/at-unis/lab/a06":
                dirs.clear()
                continue
            print(f"[*] Found shots.txt in {root}")
            if "synced_filenames_full.txt" not in files:
                get_synced_filenames_full(root, root)
            dirs.clear()
            # break

    # get_synced_filenames_full(args.base_dir, args.output_dir)
