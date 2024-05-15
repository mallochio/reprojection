#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   make_segmentation_masks.py
@Time    :   2024/03/15 13:20:59
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Run the pose_to_mask.py script in a loop to generate segmentation masks from 3D pose data
'''

import os

# Walk through the directory until you hit a folder named "reprojected"
# For each folder named "reprojected", run the pose_to_mask.py script

ROOT_DIR = "/home/sid/Projects/NAS-mountpoint/kinect-omni-ego/2023-02-09"

for root, dirs, files in os.walk(ROOT_DIR):
    if "reprojected" in dirs:
        os.system(f"/home/sid/miniforge3/envs/humor/bin/python3 helper_scripts/pose_to_mask.py --base-folder {os.path.join(root, 'reprojected')}")
        dirs.clear()
    elif root.endswith("calib"):
        dirs.clear()
