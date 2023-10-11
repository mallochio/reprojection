#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   validate_intrinsics.py
@Time    :   2023/09/22 15:32:24
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Validate the intrinsics and extrinsics of the pipeline by checking perspective projection of checkerboard corners
"""
import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/local/lib/python3.7/site-packages/cv2/qt/plugins/platforms'

import sys

sys.path.insert(0, "/openpose/data/other/humor/humor")
sys.path.insert(0, "/openpose/data/code/reprojection/")

from humor_inference.reproject_humor_sequence import make_44

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

pattern_size = (9, 6)
side_image_path = "/openpose/data/dataset/session-recordings/test/2022-10-07/at-paus/bedroom/calib/k1-omni/capture1/rgb/1665065606836.jpg"
top_image_path = "/openpose/data/dataset/session-recordings/test/2022-10-07/at-paus/bedroom/calib/k1-omni/omni/1665065677015.jpg"

rgb_cam_to_world_path = "/openpose/data/dataset/session-recordings/test/2022-10-07/at-paus/bedroom/calib/k1-omni/capture1/k1_rgb_cam_to_world.pkl"
omni_world_to_cam_path = "/openpose/data/dataset/session-recordings/test/2022-10-07/at-paus/bedroom/calib/k1-omni/k1_omni_world_to_cam.pkl"

rgb_intrinsics_path = "/openpose/data/code/reprojection/calibration/intrinsics/k1_rgb_intrinsics_new.pkl"
omni_intrinsics_path = "/openpose/data/code/reprojection/calibration/intrinsics/omni_intrinsics_new.pkl"


def load_pickle(path):
    with open(path, "rb") as f:
        matrix = pickle.load(f)
    return matrix


def calculate_distance():
    distances = 0.42 * np.ones((1, 54))
    count = 0
    increment = 0
    # Increase the value of every 10th point by 0.01, 19th point by 0.02 and so on
    for i in range(len(distances[0])):
        if count == 9:
            increment += 0.02
            count = 0
        distances[0][i] = distances[0][i] + increment
        count += 1
    return distances


def find_and_plot_checkerboard(image, corners_img, ret=True, display=False):
    if corners_img is None:
        # Find the checkerboard corners in both images
        ret, corners_img = cv2.findChessboardCorners(image, pattern_size, None)

    # Draw the corners on both images
    image = cv2.drawChessboardCorners(image, pattern_size, corners_img, ret)

    # Display the images
    if display:
        plt.imshow(image)
        plt.show()
    return ret, corners_img, image


def main():
    # Validating extrinsion matrix
    side_image = cv2.imread(side_image_path)
    top_image = cv2.imread(top_image_path)

    cv2.imshow("side_image", side_image)
    cv2.imshow("top_image", top_image)

    ret1, side_corners_img, _ = find_and_plot_checkerboard(side_image, None, True, display=False)
    ret2, top_corners_img, _ = find_and_plot_checkerboard(top_image, None, True, display=False)

    # Load the camera matrices
    rgb_cam_to_world = load_pickle(rgb_cam_to_world_path)
    omni_world_to_cam = load_pickle(omni_world_to_cam_path)
    rgb_intrinsics = load_pickle(rgb_intrinsics_path)
    omni_intrinsics = load_pickle(omni_intrinsics_path)

    # Get the camera coordinates
    cameraMatrix, distCoeffs = (
        rgb_intrinsics["intrinsics"],
        rgb_intrinsics["distortion"],
    )

    # Convert 2D image coordinates to 3D camera coordinates
    # side_corners_img = cv2.undistortPoints(src=side_corners_img, cameraMatrix=np.linalg.inv(cameraMatrix), distCoeffs=distCoeffs)

    homogeneous_image_coordinates = np.concatenate(
        (side_corners_img, np.ones((54, 1, 1))), axis=2
    )
    camera_coordinates = (
        np.linalg.inv(cameraMatrix) @ homogeneous_image_coordinates.reshape(54, 3).T
    )
    distances = calculate_distance()
    homogeneous_camera_coordinates = np.vstack((camera_coordinates, distances))

    # Transform with extrinsic matrix
    R_rgb, t_rgb = rgb_cam_to_world["R"], rgb_cam_to_world["t"]
    R_omni_inv, t_omni_inv = omni_world_to_cam["R"], omni_world_to_cam["t"]

    T_cw = make_44(R_rgb, t_rgb)
    T_wc = make_44(R_omni_inv, t_omni_inv)

    extrinsic_matrix = T_wc @ T_cw
    extrinsic_matrix[:3, 3] = extrinsic_matrix[:3, 3] / 1000.0

    omni_camera_coordinates = extrinsic_matrix @ homogeneous_camera_coordinates
    omni_camera_coordinates = omni_camera_coordinates[:3, :].T
    omni_camera_coordinates = omni_camera_coordinates.reshape(54, 1, 3)

    # Apply omni camera intrinsics
    K = omni_intrinsics["intrinsics"]
    D = omni_intrinsics["distortion"]
    xi = omni_intrinsics["xi"]

    xi = xi.item() if isinstance(xi, np.ndarray) else xi
    xii = np.double(xi)

    rvec, tvec = np.zeros(3), np.zeros(3)

    omni_image_corners, _ = cv2.omnidir.projectPoints(
        objectPoints=omni_camera_coordinates, rvec=rvec, tvec=tvec, K=K, D=D, xi=xii
    )
    omni_image_corners = omni_image_corners.astype(np.float32)

    # Draw omni_image_corners on top_image
    _, _, top_image  = find_and_plot_checkerboard(top_image, omni_image_corners, ret2, display=False)

    #Save the image
    cv2.imwrite("/openpose/data/code/reprojection/calibration/opencv_calibration/omni_image_corners.jpg", top_image)
    


if __name__ == "__main__":
    main()
