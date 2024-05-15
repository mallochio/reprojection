#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   dataset_file_generator.py
@Time    :   2024/03/07 13:57:45
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Functions to help generate dataset json files
'''
import os
import json


OUTPUT_JSON = "/home/sid/Projects/OmniScience/mount-NAS/train.json"

def get_entry(root, reprojections_folder):
    omni_dir = os.path.join(root, "omni")
    pose_dir = os.path.join(reprojections_folder, "poses")
    mesh_dir = os.path.join(reprojections_folder, "meshes")
    mask_dir = os.path.join(reprojections_folder, "masks")
    return {"omni": omni_dir, "pose": pose_dir, "mesh": mesh_dir, "mask": mask_dir}



def collect_pose_estimates(root_dir):
    data_entries = []
    for root, dirs, files in os.walk(root_dir):
        if "calib" in root:
            dirs.clear()
        if "omni" in dirs:
            capture_dirs = [i for i in dirs if i.startswith("capture")]
            for capture_folder in capture_dirs:
                reprojections_folder = os.path.join(root, capture_folder, f"out_{capture_folder}", "reprojected")
                if os.path.exists(reprojections_folder):
                    print("[*] Found reprojections at ", reprojections_folder)
                    entry = get_entry(root, reprojections_folder)
                    data_entries.append(entry)
            dirs.clear()
    return data_entries

if __name__ == "__main__":
    ROOT_DIRS = [
        "/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/2024-01-12",
        "/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/2023-02-09",
    ]

    for root in ROOT_DIRS:
        data_entries = collect_pose_estimates(root)
        with open(OUTPUT_JSON, "a") as f:  
            json.dump(data_entries, f, indent=4)