#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   save_3d_skeletons.py
@Time    :   2024/05/15 17:23:39
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Script to save 3D skeletons for side view images using mediapipe
"""
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
import os

os.environ["GLOG_minloglevel"] = "2"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set the GPU device to use
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow GPU memory growth
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import datetime
import mediapipe as mp
from mediapipe import solutions
import concurrent.futures
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
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return annotated_image


def serialize_landmarks(landmarks):
    # return np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])
    return [[landmark.x, landmark.y, landmark.z] for landmark in landmarks]


def serialize_detection_result(detection_result):
    return {
        "pose_landmarks": [
            serialize_landmarks(pose_landmarks)
            for pose_landmarks in detection_result.pose_landmarks
        ],
        "pose_world_landmarks": [
            serialize_landmarks(pose_landmarks)
            for pose_landmarks in detection_result.pose_world_landmarks
        ],
    }


def save_detection_result(detection_result, filename):
    with open(filename, "w") as f:
        json.dump(serialize_detection_result(detection_result), f)


# Function to initialize the detector
def initialize_detector():
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    base_options = BaseOptions(model_asset_path="pose_landmarker.task")
    options = PoseLandmarkerOptions(
        base_options=base_options, output_segmentation_masks=False
    )
    detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    return detector


# Function to process a single image
def process_image(image_path, output_folder):
    # Initialize the detector in each process
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    base_options = BaseOptions(model_asset_path="pose_landmarker.task")
    options = PoseLandmarkerOptions(
        base_options=base_options, output_segmentation_masks=False
    )
    detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    try:
        img = mp.Image.create_from_file(image_path)
        detection_result = detector.detect(img)
        out_file_name = os.path.basename(image_path).split(".")[0]
        save_detection_result(detection_result, f"{output_folder}/{out_file_name}.json")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")


# Main function to process images
def main(base_path):
    num_cores_to_use = (
        64  # min(8, os.cpu_count())  # Use a reasonable number of CPU cores
    )

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_cores_to_use
    ) as executor:
        futures = []

        for root, dirs, files in os.walk(base_path):
            if "calib" in root:
                dirs.clear()
                print(f"Cleared subdirectories in {root}")
            elif "rgb" in dirs and "calib" not in root:
                rgb_folder = os.path.join(root, "rgb")
                images = [
                    os.path.join(rgb_folder, i)
                    for i in os.listdir(rgb_folder)
                    if i.endswith(".jpg")
                ]
                capture_num = root.split("/")[-1]
                output_folder = os.path.join(
                    root, f"out_{capture_num}", "rgb_preprocess", "mp_keypoints_3d"
                )
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                else:
                    print(f"Skipping {output_folder} as it is already processed")
                    continue

                for image_path in images:
                    futures.append(
                        executor.submit(process_image, image_path, output_folder)
                    )

                # Print statement indicating the folder is processed
                print(f"Processing images in {rgb_folder}")

                # Wait for all futures of the current folder to complete
                for future in tqdm(
                    concurrent.futures.as_completed(futures), total=len(futures)
                ):
                    try:
                        future.result()  # Handle exceptions if any
                    except Exception as e:
                        # Handle exceptions
                        print(f"Exception occurred: {e}")

                # Clear the directories to prevent processing subdirectories
                dirs.clear()
                # Get the current time
                now = datetime.datetime.now()

                # Format the time in a human-readable format
                human_readable_time = now.strftime("%Y-%m-%d %H:%M:%S")
                print(f"Finished processing folder: {root} at time {human_readable_time}")

        # Print statement indicating all folders are processed
        print("All folders have been processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save 3D skeletons for side view images using mediapipe"
    )
    parser.add_argument("base_path", type=str, help="The path to the dataset")
    args = parser.parse_args()
    main(args.base_path)
