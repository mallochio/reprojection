#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   calibrate_extrinsics.py
@Time    :   2024/02/26 14:12:42
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Calibrate extrinsics recursively by running get_cam_pose in a loop
'''

from os.path import abspath
import sys
from pathlib import Path
from tqdm import tqdm
import subprocess
from multiprocessing import Pool

sys.path.append(abspath("../.."))
sys.path.append(abspath(".."))

script_path =  "/home/sid/Projects/OmniScience/code/reprojection/calibration/opencv_calibration/get_cam_pose.py"
intrinsics_path = "/home/sid/Projects/OmniScience/code/reprojection/calibration/intrinsics"
base_dir = Path("/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/2022-09-23")

def process_directory(path):
    prefixes = ["k0", "k1", "k2"]
    ix = [ix for ix in prefixes if ix in str(path)][0]
    dirtype = path.name
    if not ix or dirtype not in ["rgb", "ir", "omni"]:
        print(path)
        raise ValueError("No prefix found in path")
    else:
        # run get_cam_pose on the first image in each directory using the corresponding camera matrix
        img_path = next(path.glob("*.jpg"), None)
        if img_path is None:
            return
        if dirtype == "omni":
            camera_matrix_file = "omni_calib.pkl"
        else:
            camera_matrix_file = f"{ix}_rgb_calib.pkl"

        camera_matrix_path = Path(intrinsics_path) / camera_matrix_file
        command = f"python {script_path} {img_path} {camera_matrix_path} --dst {path.parent} --prefix {ix}_{dirtype}"
        if dirtype == "omni":
            command += " --fisheye"
        # print(command)
        
        subprocess.run(command.split(" "))

if __name__ == "__main__":
    subdir = list(base_dir.glob('**'))
    calib_dirs = [x for x in subdir if x.name in ["rgb", "omni"] and "/calib/" in str(x)]

    # print(calib_dirs)

    # Use Pool to parallelize the processing
    with Pool() as pool:
        list(tqdm(pool.imap(process_directory, calib_dirs), total=len(calib_dirs)))