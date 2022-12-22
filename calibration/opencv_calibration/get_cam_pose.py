#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Optimize the camera pose given an image of a chessboard, the camera intrinsics and the chessboard grid size.
"""

import os
import argparse
import pickle
from typing import Optional, Tuple

import cv2 as cv
import numpy as np

""" Most of this file was written by GitHub Copilot. """


def draw_axes(img, corners, imgpts):
    corner = tuple([int(x) for x in corners[0].ravel()])
    img = cv.line(
        img, corner, tuple([int(x) for x in imgpts[0].ravel()]), (255, 0, 0), 5
    )
    img = cv.line(
        img, corner, tuple([int(x) for x in imgpts[1].ravel()]), (0, 255, 0), 5
    )
    img = cv.line(
        img, corner, tuple([int(x) for x in imgpts[2].ravel()]), (0, 0, 255), 5
    )
    return img


def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)
    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img


def get_cam_pose(
    img_path: str,
    calibrated_img_shape: Tuple[int, int],
    grid_size: Tuple[int, int],
    square_width_mm: float,
    cam_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    xi: Optional[float],
    fish_eye: bool,
    debug: bool,
):
    rvecs, tvecs = None, None
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : grid_size[0], 0 : grid_size[1]].T.reshape(-1, 2)
    objp = objp * square_width_mm
    img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2GRAY)
    assert (
        img.shape == calibrated_img_shape
    ), "Image shape is not the same as the calibrated one."
    print(f"[*] Image shape: {img.shape}")
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(img, grid_size)
    # If found, add object points, image points (after refining them)
    if ret:
        print(f"-> [{img_path}]: Found the chessboard!")
        # Refine them
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        corners = cv.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
        # Find the rotation and translation vectors.
        # cv.SOLVEPNP_SQPNP, cv.SOLVEPNP_ITERATIVE, cv.SOLVEPNP_IPPE? iterative seems best
        solver = cv.SOLVEPNP_ITERATIVE
        if fish_eye:
            print("[*] Using omni camera solver")
            corners = cv.omnidir.undistortPoints(
                corners, cam_matrix, dist_coeffs, xi, np.eye(3)
            )
            K, D = np.eye(3), np.zeros((1, 5))
        else:
            K, D = cam_matrix, dist_coeffs
        """ A note on solvePnP: 'This function returns the rotation and the translation vectors
        that transform a 3D point expressed in the object coordinate frame to the camera coordinate
        frame.' So it is the world-to-camera transformation. """
        ret, rvecs, tvecs = cv.solvePnP(objp, corners, K, D, flags=solver)
        print(f"[*] Found initial pose: R={cv.Rodrigues(rvecs)[0]}, t={tvecs}")
        print("[*] Refining pose...")
        # Refine the pose
        rvecs, tvecs = cv.solvePnPRefineVVS(
            objp,
            corners,
            K,
            D,
            rvecs,
            tvecs,
            criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 500, 1e-8),
        )
        print(f"[*] Refined pose: R={cv.Rodrigues(rvecs)[0]}, t={tvecs}")
        # Or with RANSAC:
        """
        _, rvecs, tvecs, inliers = cv.solvePnPRansac(
            objp,
            corners,
            K,
            D,
        )
        print(f"[*] RANSAC pose: R={cv.Rodrigues(rvecs)[0]}, t={tvecs}")
        # Now let's look at all the solutions:
        retval, prvecs, ptvecs, errors = cv.solvePnPGeneric(
            objp,
            corners,
            K,
            D,
            flags=cv.SOLVEPNP_ITERATIVE,
        )
        # Now go through all rotation vectors and translation vectors and print them (convert  the
        # rotation vectors to rotation matrices first):
        for i in range(len(prvecs)):
            print(f"[*] Solution {i}: R={cv.Rodrigues(prvecs[i])[0]}, t={ptvecs[i]}")
            print(f"-> Error: {errors[i]}")

        """
        if debug:
            axis = (
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32).reshape(
                    -1, 3
                )
                * square_width_mm
            )
            cube = (
                np.array(
                    [
                        [0, 0, 0],
                        [0, 1, 0],
                        [1, 1, 0],
                        [1, 0, 0],
                        [0, 0, -1],
                        [0, 1, -1],
                        [1, 1, -1],
                        [1, 0, -1],
                    ],
                    dtype=np.float32,
                )
                * square_width_mm
            )
            # img = draw_axes(rgbimg, corners, cv.projectPoints(axis, rvecs, tvecs, K, D)[0])
            rgbimg = cv.imread(img_path)
            if fish_eye:
                cube = np.expand_dims(cube, axis=0)
                xi = xi.item() if isinstance(xi, np.ndarray) else xi
                print(cube.shape)
                img_pts, _ = cv.omnidir.projectPoints(
                    cube, rvecs, tvecs, cam_matrix, xi, dist_coeffs
                )
            else:
                img_pts = cv.projectPoints(cube, rvecs, tvecs, cam_matrix, dist_coeffs)[
                    0
                ]
            img = draw_cube(rgbimg, corners, img_pts)
            cv.imshow("img", img)
            k = ""
            while k != ord("q"):
                k = cv.waitKey(0)
            cv.destroyAllWindows()
    else:
        raise Exception("Could not find the chessboard!")
    return rvecs, tvecs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "img_path",
        type=str,
        help="Path to the image of the chessboard",
    )
    parser.add_argument(
        "cam_params",
        type=str,
        default=None,
        help="Path to the camera parameters pickle file",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        nargs=2,
        default=(9, 6),
        help="Number of squares in the chessboard grid",
    )
    parser.add_argument(
        "--square_width_mm",
        type=float,
        default=96,
        help="Width of a square in the chessboard grid",
    )
    parser.add_argument(
        "--fisheye",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        dest="debug",
        help="Draw 3D axes/cube reprojected on the image",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=".",
        help="Path to the directory where the output files will be saved",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix to add to the output filenames",
    )
    args = parser.parse_args()
    # Load the camera parameters from the pickle file
    with open(args.cam_params, "rb") as f:
        cam_params = pickle.load(f)

    print("[*] Running camera pose optimization (PnP)...")
    rvecs, tvecs = get_cam_pose(
        args.img_path,
        cam_params["img_shape"],
        args.grid_size,
        args.square_width_mm,
        cam_params["intrinsics"],
        cam_params["distortion"],
        cam_params["xi"] if args.fisheye else None,
        args.fisheye,
        args.debug,
    )
    # Compose the rotation matrix from the rotation vector
    Rt, _ = cv.Rodrigues(rvecs)
    # Get the global camera pose. The world-to-camera transformation matrix is the inverse of the
    # camera-to-world matrix. The obtained rvecs,tvecs are the world-to-camera transformation, that is
    # it transforms a point from the world origin to the camera frame.
    print("================== World-to-camera transformation ==================")
    print(f"-> R={Rt}")
    print(f"-> t={tvecs}")
    # Save the rotation matrix and translation vector into a pickle file named 'world_to_cam.pkl'
    world_to_cam_path = os.path.join(args.dst, f"{args.prefix}_world_to_cam.pkl")
    with open(world_to_cam_path, "wb") as f:
        pickle.dump({"R": Rt, "t": tvecs}, f)
        print("[*] Transformation saved to 'world_to_cam.pkl'")
    print("====================================================================")

    print("================== Camera-to-world transformation ==================")
    R = Rt.T
    t = -R @ tvecs
    print(f"-> R={R}")
    print(f"-> t={t}")
    # Save the rotation matrix and translation vector into a pickle file named 'cam_to_world.pkl'
    cam_to_world_path = os.path.join(args.dst, f"{args.prefix}_cam_to_world.pkl")
    with open(cam_to_world_path, "wb") as f:
        pickle.dump({"R": R, "t": t}, f)
        print("[*] Transformation saved to 'cam_to_world.pkl'")
    print("====================================================================")
