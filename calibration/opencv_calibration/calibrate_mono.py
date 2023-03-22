#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
We are using a 9x6 grid of 96mm.

A note on omnidirectional camera calibration:
CMei's model is apprently better than Scaramuzza's, which is not implemented in OpenCV.
In their paper (https://www.robots.ox.ac.uk/~cmei/articles/single_viewpoint_calib_mei_07.pdf), they
state "In [Scaramuzza], the authors propose a method relying on a polynomial approximation of the
projection function. With this model, initial values of the projection function are difficult to
obtain so the user has to select each point of the calibration grid independently for the
calibration. We will show that by using an exact model to which we add small errors, only four
points need to be selected for each calibration grid. The parameters that appear in the proposed
model can also be easily interpreted in terms of the optical quality of the sensor."
"""

import argparse
import os
import pickle
from typing import Optional, Tuple

import cv2 as cv
import numpy as np
import random


def calibrate_cam(
    files_path: str,
    grid_size: Tuple[int, int],
    square_width_mm: float,
    fish_eye: bool,
    verbose: bool,
    max_files: Optional[int] = None,
):
    if not os.path.exists(files_path):
        raise ValueError(f"Path {files_path} does not exist")
    intrinsics_fpath = (
        f"intrinsics{'x'.join(f'{int_}' for int_ in grid_size)}_{square_width_mm}.npy"
    )
    calibration_fpath = (
        f"calibration{'x'.join(f'{int_}' for int_ in grid_size)}_{square_width_mm}.pkl"
    )
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, 1e-6)
    fisheye_calibration_flags = (
        None
        # cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        # + cv.fisheye.CALIB_CHECK_COND
        # + cv.fisheye.CALIB_FIX_SKEW
    )
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((1, grid_size[0] * grid_size[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0 : grid_size[0], 0 : grid_size[1]].T.reshape(-1, 2)
    objp = objp * square_width_mm
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    img_shape = None
    idx = None

    print("[*] Finding corners...")
    files = os.listdir(files_path)
    if max_files is not None:
        files = random.sample(files, max_files)

    for file_name in files:
        file = os.path.join(files_path, file_name)
        if os.path.isfile(file):
            img = cv.cvtColor(cv.imread(file), cv.COLOR_BGR2GRAY)
            if img_shape is None:
                img_shape = img.shape
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(
                img,
                grid_size,
            )
            print(f"-> Image {file_name} - Corners found: {ret}")
            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

    assert img_shape is not None
    print(
        f"[*] Found {len(objpoints)} valid frames! Calibrating for a {img_shape[0]}x{img_shape[1]} resolution..."
    )
    xi = None
    if fish_eye:
        (
            ret,
            cam_mat,
            xi,
            dist_coeffs,
            rotation_vec,
            translation_vec,
            idx,
        ) = cv.omnidir.calibrate(
            objpoints,
            imgpoints,
            img_shape[::-1],
            None,
            None,
            None,
            None,
            criteria=(
                cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                200,
                1e-6,
            ),
        )
    else:
        ret, cam_mat, dist_coeffs, rotation_vec, translation_vec = cv.calibrateCamera(
            objpoints,
            imgpoints,
            img_shape[::-1],
            None,
            None,
        )

    if verbose:
        print(f"-> Intrinsics: {cam_mat}")
        print(f"-> Distortion: {dist_coeffs}")
        if fish_eye:
            print(f"-> xi: {xi.item() if isinstance(xi, np.ndarray) else xi}")
    # np.save(intrinsics_fpath, cam_mat)
    with open(calibration_fpath, "wb") as f:
        pickle.dump(
            {
                "intrinsics": cam_mat,
                "distortion": dist_coeffs,
                "xi": xi,
                "img_shape": img_shape,
            },
            f,
        )
    # print(f"-> Camera intrinsics saved as {intrinsics_fpath}")
    print(f"-> Calibration results saved as {calibration_fpath}")
    mean_error = 0
    if fish_eye:
        assert idx is not None
        idx = idx.squeeze()
        used = 0
        for i in range(len(idx)):
            if idx[i] != 0:
                used += 1
                xi = xi.item() if isinstance(xi, np.ndarray) else xi
                imgpoints2, _ = cv.omnidir.projectPoints(
                    objpoints[i],
                    rotation_vec[i],
                    translation_vec[i],
                    cam_mat,
                    xi,
                    dist_coeffs,
                )
                imgpoints2 = np.swapaxes(imgpoints2, 0, 1)
                error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
                mean_error += error
        print(f"-> Used {used}/{len(objpoints)} frames for calibration.")
        print(f"-> Total error: {mean_error/used:.2f}px")
    else:
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(
                objpoints[i], rotation_vec[i], translation_vec[i], cam_mat, dist_coeffs
            )
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
            mean_error += error
        print(f"-> Total error: {mean_error/len(objpoints):.2f}px")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
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
    parser.add_argument(
        "--fisheye",
        action="store_true",
    )
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose")
    parser.add_argument("--max-files", type=int, default=None, required=False)
    args = parser.parse_args()
    calibrate_cam(
        args.root, args.grid_size, args.square_width, args.fisheye, args.verbose, args.max_files
    )
