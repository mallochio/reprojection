#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   check_openpose.py
@Time    :   2023/10/23 14:42:19
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Check the openpose keypoints for correctness and run mediapipe 2D pose estimation
'''


import os
import json
import shutil
import mediapipe as mp
import cv2
from pprint import pprint


openpose_path = "/openpose/data/test/capture2/full_run_use_split/rgb_preprocess/op_keypoints/"
imgs_path = "/openpose/data/test/capture2/full_run_use_split/rgb_preprocess/op_frames"
empty_path = "/openpose/data/test/capture2/full_run_use_split/empty"

imgs_path_odin_ctr = "/home/test/capture2/full_run_use_split/rgb_preprocess/op_frames"
mediapipe_out_dir = "/home/test/capture2/full_run_use_split/mediapipe_out"

def move_empty_frames(data, i):
    if len(data['people']) == 0:
        # copy corresponding image to a new folder
        image_name = i.split("_")[0] + "_rendered.png"
        image_path = os.path.join(imgs_path, image_name)
        shutil.copy(image_path, empty_path)
        return True
    return False

def check_keypoint_file(data):
    pprint(len(data["people"][0]["pose_keypoints_2d"]) / 3)


def run_mediapipe_2Dpose(img):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # For static images:
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
        ) as pose:
        pose.use_gpu = True
        image = cv2.imread(img)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print and draw pose landmarks on the image.
    if not results.pose_landmarks:
        print("No landmarks found for image ", img.split("/")[-1])
        return False
    else:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imwrite(os.path.join(mediapipe_out_dir, img.split("/")[-1]), image)
        cv2.waitKey(0)
        return True


def main():   
    for i in os.listdir(imgs_path_odin_ctr):
        if i.endswith(".png"):
            run_mediapipe_2Dpose(os.path.join(imgs_path_odin_ctr, i))

            

if __name__ == "__main__":
    main()
