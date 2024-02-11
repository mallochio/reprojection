#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   run_preprocessing_sid.py
@Time    :   2024/02/10 20:10:09
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Script to run preprocessing on the data in a loop (uses humor/fitting/run_fitting.py)
'''
import os, sys
import logging
import subprocess


ROOT_DIR = "/openpose/data/mount-NAS/kinect-omni-ego"  # TODO - This is mounted with SSHFS so access is slow, this needs to be sped up

# Configure logging
logging.basicConfig(level=logging.INFO,
                    filename='preprocessing_log.txt',  # Log file name
                    filemode='a',  # Append to existing log
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Get the logger for the script
logger = logging.getLogger(__name__)
python_executable = sys.executable


def main():
    for root, dirs, files in os.walk(ROOT_DIR):
        n = os.path.basename(root)[-1]
        if "rgb" in dirs and "calib" not in root and f"out_capture{n}" not in dirs:
            command = f"{python_executable} humor/fitting/run_fitting.py @./configs/fit_rgb_demo_use_split_looped.cfg --data-path {root}/rgb --out {root}/out_capture{n} --rgb-intrinsics /openpose/data/code/reprojection/calibration/intrinsics/k{n}_rgb_calib.json"
            logger.info(command)
            print(command)
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            logger.info(result.stdout)
            if result.stderr:
                logger.error(result.stderr)

if __name__ == "__main__":
    main()