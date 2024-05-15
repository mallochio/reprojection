#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   train.py
@Time    :   14/4/24 11:28
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2023-2024, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Create an activity recognition dataset from individual activities.txt files
'''
import sys
sys.path.append("/home/sid/Projects/OmniScience/code/reprojection")

print(sys.path)

import argparse
import os
import json
from tqdm import tqdm
import numpy as np
from sync.sync_files import get_sorted_files


def process_dataset(dataset_path):
    activities = []
    for recording_date in tqdm(os.listdir(dataset_path)):
        recording_date_path = os.path.join(dataset_path, recording_date)
        if os.path.isdir(recording_date_path):
            for recording_location in os.listdir(recording_date_path):
                recording_location_path = os.path.join(recording_date_path, recording_location)
                if os.path.isdir(recording_location_path):
                    for room_type in os.listdir(recording_location_path):
                        room_type_path = os.path.join(recording_location_path, room_type)
                        if os.path.isdir(room_type_path):
                            for person_name in os.listdir(room_type_path):
                                person_name_path = os.path.join(room_type_path, person_name)
                                if os.path.isdir(person_name_path):
                                    activities_file_path = os.path.join(person_name_path, "activities.txt")
                                    shots_file_path = os.path.join(person_name_path, "shots.txt")
                                    if os.path.isfile(activities_file_path) and os.path.isfile(shots_file_path):
                                        activity_segments = process_activities_file(activities_file_path)
                                        activities.append({
                                            "recording_date": recording_date,
                                            "recording_location": recording_location,
                                            "room_type": room_type,
                                            "person_name": person_name,
                                            "activity_segments": activity_segments
                                        })

    return activities


def get_closest_omni(k0_timestamp, synced_filenames):
    k0_image = f"{k0_timestamp}.jpg"
    # Find the omni file for the corresponding k0_image in synced filenames file
    for row in synced_filenames:
        if k0_image in row:
            return row[1].split(".")[0]
        
    raise ValueError(f"Could not find the corresponding omni image for {k0_image}")



def get_num_kinects(base_dir):
    return len(
        [
            folder
            for folder in os.listdir(base_dir)
            if folder.startswith("capture")
            and os.path.isdir(os.path.join(base_dir, folder))
        ]
    )


def get_synced_filenames_kinect(base_dir):
    with open(os.path.join(base_dir, "shots.txt"), "r") as file:
        shots = []
        for ix, line in enumerate(file):
            if ix == 1:
                line = line.strip().split(";")
                num_kinects = len(line) - 1
                shots = [os.path.basename(path) for path in line]

    omni_filenames, omni_timestamps, kinect_filenames, kinect_timestamps = get_sorted_files(base_dir, num_kinects)
    reference_timestamp =  int(shots[0].split(".")[0])
    omni_shot = int(shots[-1].split(".")[0])
    delta = omni_shot - reference_timestamp

    synced_filenames = []
    # Now run through the omni images and find the nearest image in the omni directory after adding the delta
    print("[*] Syncing files")
    reference_kinect_timestamps = kinect_timestamps[0]
    for timestamp in tqdm(reference_kinect_timestamps):
        kinect_filename = f"{timestamp}.jpg"
        synced_shot = [kinect_filename]
        omni_timestamp_approx = timestamp + delta
        omni_timestamp_new = min(omni_timestamps, key=lambda x: abs(x - omni_timestamp_approx))
        omni_filename = f"{omni_timestamp_new}.jpg"
        synced_shot.append(omni_filename)
        synced_filenames.append(synced_shot)
    
    return synced_filenames

def process_activities_file(file_path):
    activity_segments = []
    base_dir = os.path.dirname(file_path)
    num_kinects = get_num_kinects(base_dir)
    synced_filenames = get_synced_filenames_kinect(base_dir)

    with open(file_path, "r") as f:
        for line in f:
            data = line.strip().split(";")
            if len(data) == 4:
                start_timestamp, end_timestamp, activity_type, activity_name = data
                start_timestamp = get_closest_omni(int(start_timestamp), synced_filenames)
                end_timestamp = get_closest_omni(int(end_timestamp), scopilotynced_filenames)

                activity_segments.append({
                    "start_timestamp": int(start_timestamp),
                    "end_timestamp": int(end_timestamp),
                    "activity_type": activity_type,
                    "activity_name": activity_name
                })

    return activity_segments


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Process a dataset")
    parser.
    parser.add_argument('dataset_path', type=str, help='The path to the dataset')

    args = parser.parse_args()
    activities = process_dataset(args.dataset_path)

    with open("train_activities_omni_timestamps.json", "w") as file:
        json.dump({"activities": activities}, file, indent=2)
    
    print("train_activities_omni_timestamps.json file generated")