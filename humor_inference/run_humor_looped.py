#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   run_humor_looped.py
@Time    :   2024/02/10 20:10:09
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Script to run humor on the data in a loop, before or after preprocessing (uses humor/fitting/run_fitting.py)
'''
import os
import sys
import logging
import subprocess

ROOT_DIR = "/openpose/data/dataset/out-2022-10-14"

# Configure logging
logging.basicConfig(level=logging.INFO,
                    filename='humor_log.txt',
                    filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
python_executable = sys.executable


def main():
    for root, dirs, files in os.walk(ROOT_DIR):
        n = os.path.basename(root)[-1]
        if "calib" not in root and f"out_capture{n}" in dirs and not os.path.exists(f"{root}/out_capture{n}/results_out"):
            command = f"{python_executable} humor/fitting/run_fitting.py @./configs/fit_rgb_demo_use_split_looped.cfg --data-path {root}/out_capture{n}/rgb_preprocess/raw_frames --out {root}/out_capture{n} --rgb-intrinsics /openpose/data/reprojection/calibration/intrinsics/k{n}_rgb_calib.json"
            logger.info(command)
            print(command)
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            logger.info(result.stdout)
            if result.stderr:
                logger.error(result.stderr)


if __name__ == "__main__":
    main()