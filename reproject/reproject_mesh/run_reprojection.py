#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   reproject_mesh.py
@Time    :   2024/02/09 11:04:09
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Script to project 3D meshes from kinect to omni camera
'''

import os
import sys
import json
import torch

sys.path.append('/home/sid/Projects/OmniScience/code/reprojection')
sys.path.append('/home/sid/Projects/OmniScience/other/humor/humor')

from fitting.fitting_utils import load_res, prep_res, run_smpl
from body_model.body_model import BodyModel
from body_model.utils import SMPL_JOINTS
from fitting.eval_utils import SMPL_SIZES
from utils.torch import copy2cpu as c2c

# from reprojection

import cv2
import pickle
import trimesh
import argparse
import numpy as np
from typing import List
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from scipy.optimize import least_squares
from humor_inference.reproject_humor_sequence import make_44, transform_SMPL_sequence, get_camera_params, sanitize_preds
from reproject.reproject_mesh.reprojection_utils import save_dataset_files, get_filepaths, get_transformation_matrix_matlab, get_transformation_matrix_opencv, get_kinect_list, get_calib_paths
from bundle_adjustment import perform_bundle_adjustment


def get_synced_meshes(sync_file, capture_dir, transformed_meshes: List[trimesh.Trimesh]):
    df = pd.read_csv(sync_file, sep=";", header=0)
    capture_files = sorted(os.listdir(capture_dir))

    mesh_indices = []
    for ix, val in df.iterrows():
        # get index of val["capture0"] in capture_files
        mesh_index = capture_files.index(val[f"capture{n}"])
        mesh_indices.append([mesh_index, val["omni"]])

    # Select only the meshes from transformed_meshes which are in mesh_indices
    transformed_meshes_new = [transformed_meshes[i[0]] for i in mesh_indices]
    synced_cam1_files = [i[1] for i in mesh_indices]
    return synced_cam1_files, transformed_meshes_new


def render_mesh(img, img_path, mesh, vertices_2d, output_dir=None):
    try:
        img = ImageOps.mirror(img)
    except Exception as e:
        print("Image: ", img_path)
        return

    draw = ImageDraw.Draw(img)
    for face in mesh.faces:
        face_vertices = vertices_2d[face]

        draw.polygon(
            [tuple(p[0]) for p in face_vertices],
            fill=None,
            outline="gray",
            # outline="black",
            width=1,
        )
    # Save the image
    if output_dir is not None:
        img.save(os.path.join(output_dir, f"{i:08d}.png"))
        
    img = img.resize((int(img.size[0] / 2.), int(img.size[1] / 2.)))
    # Display image in original RGB format
    cv2.imshow("Image", cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit()
    return img


def project_meshes(
    cam0_images_dir: str,
    cam1_images_dir: str,
    mesh_seq: List[trimesh.Trimesh],
    sync_file: str,
    camera_calib: dict,
    render: bool = True,
):
    """
    Project the mesh sequence on the (omni) images.
    """
    # Load the camera intrinsics and distortion coefficients from the pickle file
    use_omni, camera_matrix, xi, dist_coeffs = get_camera_params(camera_calib)
    images = [
        filename
        for filename in sorted(os.listdir(cam1_images_dir))
        if filename.endswith(".png") or filename.endswith(".jpg")
    ]
    synced_cam1_files, mesh_seq = get_synced_meshes(sync_file, cam0_images_dir, mesh_seq)
    images = [i for i in images if i in synced_cam1_files]
    images = [os.path.join(cam1_images_dir, i) for i in images]

    assert len(mesh_seq) == len(images), "Number of images and meshes must be the same!"

    if len(images) > len(mesh_seq):
        images = images[: len(mesh_seq)]
        print("Warning: more images than meshes, truncating images to match.")
    elif len(images) < len(mesh_seq):
        mesh_seq = mesh_seq[: len(images)]
        print("More meshes than images, truncating meshes to match.")

    # Object to save all meshes and projected vertices
    projected_vertices = []
    print("[*] Projecting the meshes on the images - Press 'q' to quit!")
    for i, (mesh, img_path) in tqdm(enumerate(zip(mesh_seq, images)), total=len(images)):
        img = Image.open(img_path)
        assert (img.size[1], img.size[0]) == camera_calib[
            "img_shape"
        ], "Image shape must match the camera calibration!"

        if use_omni:
            vertices_2d, _ = cv2.omnidir.projectPoints(
                np.expand_dims(mesh.vertices, axis=0),
                np.zeros(3),
                np.zeros(3),
                camera_matrix,
                xi,
                dist_coeffs,
            )
            vertices_2d = vertices_2d.swapaxes(0, 1)

        else:
            vertices_2d, _ = cv2.projectPoints(mesh.vertices, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
        if render:
            img = render_mesh(img, img_path, mesh, vertices_2d)

        projected_vertices.append(vertices_2d)

    cv2.destroyAllWindows()
    return projected_vertices, mesh_seq


def associate_partial_meshes(mesh_folder, image_folder, window_size = 60, overlap = 10):
    # Function to associate partial meshes with the corresponding images and returns the list of associated pairs
    mesh_folder_root = str(Path(mesh_folder).parent)
    valid_folders = sorted(os.listdir(mesh_folder_root))
    if "final_results" in valid_folders:
        valid_folders.remove("final_results")
    subfolder_number = valid_folders.index(os.path.basename(mesh_folder)) # Assumes no other folders are present in the root folder containing the mesh_folder

    step_size = window_size - overlap
    start_index = 1 + (subfolder_number - 1) * step_size
    end_index = start_index + window_size - 1

    all_image_files = sorted(os.listdir(image_folder))
    return all_image_files[start_index-1:end_index]


def get_meshes(results_folder: str):
    res_file = os.path.join(results_folder, "stage3_results.npz")
    if not os.path.isfile(res_file):
        raise FileNotFoundError(f"Could not find {res_file}!")

    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    pred_res = np.load(res_file)
    T = pred_res["trans"].shape[0]

    sanitize_preds(pred_res, T) # This function is currently just checking if the predictions have invalid values
    pred_res = prep_res(pred_res, device, T)
    num_pred_betas = pred_res["betas"].size(1)

    # create body models for each
    meta_path = os.path.join(results_folder, "meta.txt")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Could not find {meta_path}!")

    optim_bm_path = None
    with open(meta_path, "r") as f:
        optim_bm_str = f.readline().strip()
        optim_bm_path = optim_bm_str.split(" ")[1]

    if not os.path.exists(optim_bm_path):
        optim_bm_path = "/home/sid/Projects/OmniScience/other/humor/body_models/smplh/male/model.npz"

    # humor model
    pred_bm = BodyModel(bm_path=optim_bm_path, num_betas=num_pred_betas, batch_size=T).to(device)

    # run through SMPL
    print("[*] Running SMPL")
    pred_body = run_smpl(pred_res, pred_bm)
    print("[*] Done")
    return pred_body


def transform_meshes(pred_body, cam0_to_world_pth, world_to_cam1_pth, kinect_jsonpath, omni_jsonpath, use_matlab=False):
    # Apply extrinsic transformation to the meshes to get from camera frames of kinect to omni
    print("[*] Loaded the sequence of SMPL models!")
    if use_matlab:
        transform = get_transformation_matrix_matlab(kinect_jsonpath, omni_jsonpath)
    else:
        transform = get_transformation_matrix_opencv(cam0_to_world_pth, world_to_cam1_pth)
    print("[*] Applying the transform to the SMPL models sequence...")
    transformed_meshes = transform_SMPL_sequence(pred_body, transform)
    print("[*] Done")
    return transformed_meshes


def project_multiple(kinect_list: List[int]):
    # Project multiple captures at once, using bundle adjustment
    pass


def project_single(n: int):
    cam0_images_path, cam1_images_path, sync_file, output_path, results_folder = get_filepaths(root, n, args)
    kinect_jsonpath, omni_jsonpath, cam0_to_world_pth, world_to_cam1_pth = get_calib_paths(root, use_matlab, n)
    pred_body = get_meshes(results_folder)
    transformed_meshes = transform_meshes(pred_body, cam0_to_world_pth, world_to_cam1_pth, kinect_jsonpath, omni_jsonpath, use_matlab=use_matlab)
    with open(omni_intrinsics_file, "rb") as f:
        omni_params = pickle.load(f)

    projected_vertices, transformed_meshes = project_meshes(cam0_images_path, cam1_images_path, transformed_meshes, sync_file, omni_params, render=render_meshes)
    if save_reprojections:
        save_dataset_files(projected_vertices, transformed_meshes, output_path, cam1_images_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproject 3D meshes from kinect to omni camera")
    parser.add_argument(
        "root",
        type=str, 
        help="Root directory of the dataset", 
        default="/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/2024-01-12/at-unis/lab/sid"
    )

    parser.add_argument(
        "--omni_intrinsics", 
        type=str, 
        help="Path to the omni intrinsics file", 
        default="/home/sid/Projects/OmniScience/code/reprojection/calibration/intrinsics/omni_calib.pkl"
    )
    parser.add_argument(
        "--use-matlab",
        action="store_true",
        default=False,
        help="Use matrices from MATLAB instead of OpenCV",
    )
    parser.add_argument(
        "--partial-meshes",
        action="store_true",
        default=False,
    )   
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--n",
        type=int,
        help="Capture number, if only one is to be reprojected",
        default=None,
    )
    parser.add_argument(
        "--save-reprojections",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    root = args.root
    omni_intrinsics_file = args.omni_intrinsics
    use_matlab = args.use_matlab
    render_meshes = args.render
    n = args.n
    save_reprojections = args.save_reprojections

    if n is None:
        kinect_list = get_kinect_list(root)
        if len(kinect_list) > 1:
            print(f"Multiple final_results folders found, using bundle adjustment with captures{kinect_list}")
            for n in kinect_list:
                project_multiple(kinect_list)
        elif len(kinect_list) == 0:
            raise ValueError("No final_results folder found")
        else:
            n = kinect_list[0]
            print(f"Using capture{n} for reprojection")
            project_single(n)
    else:
        project_single(n)
