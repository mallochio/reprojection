#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   run_reprojection_loop.py
@Time    :   2024/03/05 11:41:08
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Given a list of directories, run the reprojection script in a loop
'''


import os
import subprocess
import re

folders = [
    "/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/2023-02-09/at-unis/lab/a03/capture1/out_capture1/reprojected",
    "/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/2023-02-09/at-unis/lab/a03/capture2/out_capture2/reprojected",
    "/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/2023-02-09/at-unis/lab/a04-2/capture0/out_capture0/reprojected",
    "/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/2023-02-09/at-unis/lab/a04-2/capture1/out_capture1/reprojected",
    "/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/2023-02-09/at-unis/lab/a04-2/capture2/out_capture2/reprojected",
    "/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/2023-02-09/at-unis/lab/a04/capture1/out_capture1/reprojected",
    "/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/2023-02-09/at-unis/lab/a04/capture2/out_capture2/reprojected",
    "/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/2023-02-09/at-unis/lab/a08/capture1/out_capture1/reprojected",
    "/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/2023-02-09/at-unis/lab/a10/capture0/out_capture0/reprojected",
    "/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/2023-02-09/at-unis/lab/a12/capture1/out_capture1/reprojected",
    "/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/2023-02-09/at-unis/lab/a13/capture0/out_capture0/reprojected",
    "/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/2023-02-09/at-unis/lab/a13/capture1/out_capture1/reprojected",
    "/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/2023-02-09/at-unis/lab/a13/capture2/out_capture2/reprojected",
    "/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/2024-01-12/at-unis/lab/sid/capture2/out_capture2/reprojected",
    "/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/2024-01-12/at-unis/lab/sid/capture1/out_capture1/reprojected",
]

def reproject_in_loop():
    for folder in folders:
        match = re.search(r'capture(\d+)', folder)
        kinect_num = match[1]
        folder_root = folder.split("capture")[0][:-1]
        command = f"/home/sid/miniforge-pypy3/envs/humor/bin/python3 run_reprojection.py {folder_root} --n {kinect_num}"
        print(f"[*] Running {command}")
        subprocess.run(command, shell=True)

def count_file_stats():
    total_files = set()
    for folder in folders:
        print(f"[*] Counting mesh files in {folder}")
        meshfolder = os.path.join(folder, "meshes")
        files = len(os.listdir(meshfolder))
        # print(f"[*] Total files: {files}\n")
        total_files.add(files)
    print(f"[*] Total files: {total_files}")
    print(f"Total number = {sum(total_files)}\n")

if __name__ == "__main__":
    count_file_stats()
    # reproject_in_loop()