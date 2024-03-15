#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   pose-to-mask.py
@Time    :   2023/12/21 12:10:16
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Script to generate segmentation masks from 3D pose data using dilation on the pose estimate (Incomplete)
'''

import cv2
import os
import sys
import trimesh
from tqdm import tqdm
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from skimage.draw import polygon
from multiprocessing import Pool
import argparse


def process_file(i):
    """
    Process a file by loading the mesh and pose, creating a semantic mask image, and saving it.

    Args:
        i: The file name of the mesh file.

    Returns:
        None
    """
    mesh_file = os.path.join(mesh_folder, i)
    pose_file = os.path.join(pose_folder, i).replace(".obj", ".npy")

    mesh = trimesh.load(mesh_file)
    pose = np.squeeze(np.load(pose_file), axis=1)
    image_name = i.replace(".obj", ".png")
    image_path = os.path.join(out_folder, image_name)
    create_semantic_mask(mesh, pose, image_path, 5)


def create_semantic_mask(mesh, pose_estimates, image_path, dilation_kernel_size):
    """
    Creates a semantic segmentation mask from an SMPL mesh and an image.

    Args:
        smpl_mesh_path (str): Path to the SMPL mesh file (.obj).
        image_path (str): Path to the corresponding image.
        dilation_kernel_size (int): Size of the dilation kernel. Default is 5.

    Returns:y
        mask (numpy.ndarray): The semantic segmentation mask.
    """
    # Initialize an empty image to the size you require. Replace 'width' and 'height' with your image dimensions.
    width, height = 1200, 900  # Example dimensions
    mask = np.zeros((height, width), dtype=np.uint8)

    for face in mesh.faces:
        x_coords = pose_estimates[face, 0]
        y_coords = pose_estimates[face, 1]

        x_coords = np.clip(x_coords, 0, width - 1)
        y_coords = np.clip(y_coords, 0, height - 1)

        rr, cc = polygon(y_coords, x_coords)  # Note: skimage.draw.polygon expects row, col format

        mask[rr, cc] = 1

    # Optionally, save the mask to a file
    plt.imsave(image_path, mask, cmap='gray')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate segmentation masks from 3D pose data.")
    parser.add_argument(
        "--base-folder", 
        type=str, 
        dest="base_folder",
        help="Path to the base folder containing the pose and mesh files."
    )
    args = parser.parse_args()
    base_folder = args.base_folder
    print(f"Processing folder - {base_folder}")
    # base_folder = "/home/sid/Projects/NAS-mountpoint/kinect-omni-ego/2023-02-09/at-unis/lab/a03/capture2/out_capture2/reprojected/"
    mesh_folder = os.path.join(base_folder, "meshes")
    pose_folder = os.path.join(base_folder, "poses")
    out_folder = os.path.join(base_folder, "masks")
    if os.path.exists(out_folder):
        # Exit program
        print("Output folder already exists, already processed. Exiting program.")
        sys.exit(0)
    os.mkdir(out_folder)

    files = os.listdir(mesh_folder)
    with Pool(19) as p:
        for _ in tqdm(p.imap_unordered(process_file, files), total=len(files)):
            pass
        