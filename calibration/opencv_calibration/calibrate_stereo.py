#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
We are using a 9x6 grid of 96mm.
"""

import argparse
import os
import pickle
from typing import Dict, Tuple

import cv2 as cv
import numpy as np


def find_keypoints(
    files_path_cam1: str,
    files_path_cam2,
    grid_size: Tuple[int, int],
    square_width_mm: float,
):
    if not os.path.exists(files_path_cam1):
        raise ValueError(f"Path {files_path_cam1} does not exist")
    if not os.path.exists(files_path_cam2):
        raise ValueError(f"Path {files_path_cam2} does not exist")
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : grid_size[0], 0 : grid_size[1]].T.reshape(-1, 2)
    objp = objp * square_width_mm
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints_cam1, imgpoints_cam2 = [], []  # 2d points in image plane.
    img_shape1, img_shape2 = None, None

    """ Most of the following code was written by GitHub Copilot. """

    def load_images(path, depth=False) -> Dict[str, any]:
        images = {}
        for file_name in os.listdir(path):
            file = os.path.join(path, file_name)
            if os.path.isfile(file):
                img = cv.imread(file, cv.IMREAD_GRAYSCALE if depth else cv.IMREAD_COLOR)
                if not depth:
                    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                images[file_name] = img
        return images

    images_cam1 = load_images(files_path_cam1)
    images_cam2 = load_images(files_path_cam2, depth=True)

    for (fname1, frame1), (fname2, frame2) in zip(
        images_cam1.items(), images_cam2.items()
    ):
        # Find the chess board corners
        ret1, corners1 = cv.findChessboardCorners(frame1, grid_size)
        ret2, corners2 = cv.findChessboardCorners(frame2, grid_size)
        if img_shape1 is None:
            img_shape1 = frame1.shape
            img_shape2 = frame2.shape
        # If found, add object points, image points (after refining them)
        if ret1 and ret2:
            print(f"-> [{fname1}, {fname2}]: Found the chessboard in both cameras!")
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(frame2, corners2, (11, 11), (-1, -1), criteria)
            imgpoints_cam2.append(corners2)
            corners1 = cv.cornerSubPix(frame1, corners1, (11, 11), (-1, -1), criteria)
            imgpoints_cam1.append(corners1)
        else:
            print(
                f"-> [{fname1}, {fname2}]: Could not find the chessboard in both cameras :("
            )
    return objpoints, imgpoints_cam1, imgpoints_cam2, img_shape1, img_shape2


def calibrate_stereo(
    cam1_files_path: str,
    cam2_files_path: str,
    cam1_calibration_fpath: str,
    cam2_calibration_fpath: str,
    grid_size: Tuple[int, int],
    square_width_mm: float,
):

    objpoints, imgpoints_cam1, imgpoints_cam2, _, _ = find_keypoints(
        cam1_files_path, cam2_files_path, grid_size, square_width_mm
    )

    # We assume that the two cameras are already calibrated
    # and we only need to find the extrinsic parameters.
    print(
        f"[*] Found {len(objpoints)} common valid frames! Calibrating the extrinsics..."
    )
    # Load camera matrices and distortion coefficients for cam 1 & 2 (from pickle files):
    with open(cam1_calibration_fpath, "rb") as f:
        cam1_calib = pickle.load(f)
    with open(cam2_calibration_fpath, "rb") as f:
        cam2_calib = pickle.load(f)

    # https://docs.opencv.org/4.6.0/d9/d0c/group__calib3d.html#ga91018d80e2a93ade37539f01e6f07de5
    retval, _, _, _, _, R, t, _, _, per_view_errors = cv.stereoCalibrateExtended(
        objpoints,
        imgpoints_cam1,
        imgpoints_cam2,
        cam1_calib["intrinsics"],
        cam1_calib["distortion"],
        cam2_calib["intrinsics"],
        cam2_calib["distortion"],
        imageSize=None,
        R=None,
        T=None,
        # flags=cv.CALIB_FIX_INTRINSIC+cv.CALIB_FIX_PRINCIPAL_POINT+cv.CALIB_ZERO_TANGENT_DIST+cv.CALIB_FIX_S1_S2_S3_S4+cv.CALIB_FIX_TAUX_TAUY,
    )
    if not retval:
        raise Exception("Could not calibrate!")
    print("-> Calibration done!")
    print(f"-> R={R}\n-> t={t}")
    # Save the R and T matrices as a pickle dictionary:
    with open("stereo_extrinsics.pkl", "wb") as f:
        pickle.dump({"R": R, "t": t}, f)
    print("-> Extrinsics saved to stereo_extrinsics.pkl")
    # Compute the mean pixel error for each of the two views from the per_view_errors array, where
    # each entry is a tuple of the RGB and Depth view errors.
    mean_pixel_error_cam1 = np.mean([e[0] for e in per_view_errors])
    mean_pixel_error_cam2 = np.mean([e[1] for e in per_view_errors])
    print(f"-> Mean pixel error for cam 1: {mean_pixel_error_cam1:.2f}px")
    print(f"-> Mean pixel error for cam 2: {mean_pixel_error_cam2:.2f}px")


parser = argparse.ArgumentParser()
parser.add_argument("root_of_cam1")
parser.add_argument("root_of_cam2")
parser.add_argument("cam1_calibration_fpath")
parser.add_argument("cam2_calibration_fpath")
parser.add_argument(
    "--grid-size",
    nargs=2,
    type=int,
    default=(9, 6),
    required=False,
    help="Size of the grid",
)
parser.add_argument(
    "--square-width",
    type=float,
    default=96,
    required=False,
    help="Square width in mm",
)
args = parser.parse_args()
calibrate_stereo(
    args.root_of_cam1,
    args.root_of_cam2,
    args.cam1_calibration_fpath,
    args.cam2_calibration_fpath,
    args.grid_size,
    args.square_width,
)
