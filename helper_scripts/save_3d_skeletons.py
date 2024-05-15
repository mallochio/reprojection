#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   save_3d_skeletons.py
@Time    :   2024/05/15 17:23:39
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Script to save 3D skeletons for side view images using mediapipe
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set the GPU device to use
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow GPU memory growth

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import json
from mediapipe.tasks.python import vision
import argparse
from tqdm import tqdm



def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


def serialize_landmarks(landmarks):
    # return np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])
    return [[landmark.x, landmark.y, landmark.z] for landmark in landmarks]


def serialize_detection_result(detection_result):
    return {
        'pose_landmarks': [serialize_landmarks(pose_landmarks) for pose_landmarks in detection_result.pose_landmarks],
        'pose_world_landmarks': [serialize_landmarks(pose_landmarks) for pose_landmarks in detection_result.pose_world_landmarks],
    }


def save_detection_result(detection_result, filename):
    with open(filename, 'w') as f:
        json.dump(serialize_detection_result(detection_result), f)

def main(base_path):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

    base_options = BaseOptions(model_asset_path='/home/sid/Projects/OmniScience/other/pose_landmarker.task')
    options = PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    for root, dirs, files in os.walk(base_path):
        if root.endswith("calib"):
            dirs.clear()
        if "rgb" in dirs and "calib" not in root:
            print(f"[*] Found rgb folder in {root}")
            rgb_folder = os.path.join(root, "rgb")
            images = [i for i in os.listdir(rgb_folder) if i.endswith(".jpg")]
            capture_num = root.split("/")[-1]
            output_folder = os.path.join(root, f"out_{capture_num}", "rgb_preprocess", "mp_keypoints_3d")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            for image in tqdm(images):
                image_path = os.path.join(rgb_folder, image)
                img = mp.Image.create_from_file(image_path)
                detection_result = detector.detect(img)
                out_file_name = image.split(".")[0]
                save_detection_result(detection_result, f'{output_folder}/{out_file_name}.json')
                
            dirs.clear()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save 3D skeletons for side view images using mediapipe")
    parser.add_argument('base_path', type=str, help='The path to the dataset')
    args = parser.parse_args()
    main(args.base_path)

 
