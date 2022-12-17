#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Project a point from the image plane to the camera frame given the intrinsics parameters and the
distance Z from the camera to the point.
"""

import argparse
from typing import Tuple

import numpy as np
import scipy


def main(intrinsics_path: str, point_img: Tuple[int, int], z: float):
    print("---------------- Simple linear projection (no distortion coefficients) ------------------")
    project_to_camera_frame(intrinsics_path, point_img, z)
    print("-----------------------------------------------------------------------------------------")


def project_to_camera_frame(
    intrinsics_path: str,
    point_img: Tuple[int, int],
    z_world: float,
    # img_size=(3648, 2736), # Photo mode
    img_size=(1080, 1920), # Video mode
):
    # Load the camera matrix (intrinsics)
    cam_mat = np.load(intrinsics_path)
    fx, fy = cam_mat[0, 0], cam_mat[1, 1]
    uc, vc = cam_mat[0, 2], cam_mat[1, 2]
    print(f"fx={fx} / fy={fy} -- uc={uc} / vc={vc}")
    # u = uc + x/width
    x_img = (point_img[0] - uc)# * img_size[0]
    y_img = (point_img[1] - vc)# * img_size[1]
    x_world = (x_img * z_world) / fx
    y_world = (y_img * z_world) / fy
    # print(f"Camera frame coordinates (world): ({x_world}, {y_world}, {z_world}) in mm")
    print(f"Camera frame coordinates (world): ({x_world/10}, {y_world/10}, {z_world/10}) in cm")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cam_mat_path", type=str)
    parser.add_argument("--img-pt", dest="point_img", nargs=2, type=int)
    parser.add_argument("-z", type=float)
    args = parser.parse_args()
    main(args.cam_mat_path, args.point_img, args.z)
