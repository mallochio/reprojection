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

# os.environ["CUDA_VISIBLE_DEVICES"]="1"
from fitting.fitting_utils import load_res, prep_res, run_smpl
from body_model.body_model import BodyModel
from body_model.utils import SMPL_JOINTS
from fitting.eval_utils import SMPL_SIZES
from utils.torch import copy2cpu as c2c

# from reprojection
from humor_inference.reproject_humor_sequence import make_44, transform_SMPL_sequence, get_camera_params,sanitize_preds
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


def get_synced_meshes(transformed_meshes: List[trimesh.Trimesh]):
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
            outline="red",
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


def save_projected_meshes(
        meshes: List[trimesh.Trimesh],
        vertices_2d_list: List[np.ndarray],
        output_dir: str
    ):
    # Save the meshes that have undergone perspective projection in the original format
    for i, (mesh, vertices_2d) in enumerate(zip(meshes, vertices_2d_list)):
        mesh.vertices = vertices_2d
        mesh.export(os.path.join(output_dir, f"{i:06d}.obj"))    
    return


def project_meshes(
    images_dir: str,
    mesh_seq: List[trimesh.Trimesh],
    camera_calib: dict,
    output_dir: str,
):
    """
    Project the mesh sequence on the (omni) images.
    """
    # Load the camera intrinsics and distortion coefficients from the pickle file
    use_omni, camera_matrix, xi, dist_coeffs = get_camera_params(camera_calib)

    images = [
        filename
        for filename in sorted(os.listdir(images_dir))
        if filename.endswith(".png") or filename.endswith(".jpg")
    ]
    synced_cam1_files, mesh_seq = get_synced_meshes(mesh_seq)
    images = [i for i in images if i in synced_cam1_files]
    images = [os.path.join(images_dir, i) for i in images]

    assert len(mesh_seq) == len(images), "Number of images and meshes must be the same!"

    if len(images) > len(mesh_seq):
        images = images[: len(mesh_seq)]
        print("Warning: more images than meshes, truncating images to match.")
    elif len(images) < len(mesh_seq):
        mesh_seq = mesh_seq[: len(images)]
        print("More meshes than images, truncating meshes to match.")

    # Object to save all meshes and projected vertices
    projected_vertices = []
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
            vertices_2d, _ = cv2.projectPoints(
                mesh.vertices, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs
            )

        img = render_mesh(img, img_path, mesh, vertices_2d)
        projected_vertices.append(vertices_2d)

    cv2.destroyAllWindows()
    print("[*] Done!")
    return projected_vertices, mesh_seq


def get_camera_parameters(params, camera_type):
    camera_params = {}
    if camera_type == 'kinect':
        fx, fy = tuple(params["FocalLength"])
        cx, cy = tuple(params["PrincipalPoint"])
        camera_params["k_params"] = (fx, fy, cx, cy)
        camera_params['K'] = np.asarray(params['IntrinsicMatrix']).T

        Dk = np.asarray(params["RadialDistortion"])
        Dp = np.asarray(params["TangentialDistortion"])
        if len(Dk) == 3:
            D = np.asarray([Dk[0], Dk[1], Dp[0], Dp[1], Dk[2]])
        else:
            D = np.asarray([Dk[0], Dk[1], Dp[0], Dp[1]])
        camera_params['D'] = D

    elif camera_type == 'omni':
        c0, c2, c3, c4 = params['Intrinsics']['MappingCoefficients']
        camera_params['Coeffs'] = np.array([c0, 0., c2, c3, c4])
        cx, cy = params['Intrinsics']['DistortionCenter']
        camera_params['Centre'] = (cx, cy)
        m = np.asarray(params['Intrinsics']['StretchMatrix'])
        camera_params['c'] = m[0, 0]
        camera_params['d'] = m[0, 1]
        camera_params['e'] = m[1, 0]

    else:
        raise ValueError("Unsupported camera type")

    RR = np.asarray(params["RotationMatrices"])
    camera_params["RR"] = [RR[:, :, i].T for i in range(RR.shape[2])]

    tt = np.asarray(params["TranslationVectors"])
    camera_params["tt"] = [tt[i] for i in range(tt.shape[0])]

    return camera_params

def get_transformation_matrix_opencv():
    with open(cam0_to_world_pth, "rb") as f:
        cam0_to_world = make_44(pickle.load(f))

    with open(world_to_cam1_pth, "rb") as f:
        world_to_cam1 = make_44(pickle.load(f))

    transform = world_to_cam1 @ cam0_to_world
    transform[:3, 3] = transform[:3, 3] / 1000.0
    return transform


def get_transformation_matrix_matlab():
    # Get Matlab matrices, make a composite matrix for perspective projection
    with open(kinect_jsonpath, "r") as f:
        kParams = json.load(f)

    with open(omni_jsonpath, "r") as f:
        omniParams = json.load(f)

    kinect_params = get_camera_parameters(kParams, 'kinect')
    omni_params = get_camera_parameters(omniParams, 'omni')

    R_kinect = kinect_params["RR"][0]
    t_kinect = kinect_params["tt"][0].reshape(3, 1)

    R_omni = omni_params["RR"][0]
    t_omni = omni_params["tt"][0].reshape(3, 1)

    meo = make_homogenous_transformation_matrix(R_omni, t_omni)
    mei = make_homogenous_transformation_matrix(R_kinect, t_kinect)
    return np.matmul(meo, np.linalg.pinv(mei))


def make_homogenous_transformation_matrix(R, t):
    homogenous_matrix = np.zeros((4, 4))
    homogenous_matrix[0:3, 0:3] = R
    homogenous_matrix[:3, 3] = t.ravel() / 1000.
    homogenous_matrix[3, :] = np.asarray([0, 0, 0, 1])
    return homogenous_matrix


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
        raise Exception(f"Could not find {res_file}!")

    device = torch.device("cpu")
    pred_res = np.load(res_file)
    T = pred_res["trans"].shape[0]

    sanitize_preds(pred_res, T) # This function is currently just checking if the predictions have invalid values
    pred_res = prep_res(pred_res, device, T)
    num_pred_betas = pred_res["betas"].size(1)

    # create body models for each
    meta_path = os.path.join(results_folder, "meta.txt")
    if not os.path.exists(meta_path):
        raise Exception(f"Could not find {meta_path}!")

    optim_bm_path = None
    with open(meta_path, "r") as f:
        optim_bm_str = f.readline().strip()
        optim_bm_path = optim_bm_str.split(" ")[1]

    if not os.path.exists(optim_bm_path):
        optim_bm_path = "/home/sid/Projects/humor/body_models/smplh/male/model.npz"

    # humor model
    pred_bm = BodyModel(bm_path=optim_bm_path, num_betas=num_pred_betas, batch_size=T).to(device)

    # run through SMPL
    pred_body = run_smpl(pred_res, pred_bm)
    return pred_body


def transform_meshes():
    # Apply extrinsic transformation to the meshes to get from camera frames of kinect to omni
    pred_body = get_meshes(results_folder)
    print("[*] Loaded the sequence of SMPL models!")
    if use_matlab:
        transform = get_transformation_matrix_matlab()
    else:
        transform = get_transformation_matrix_opencv()
    print("[*] Applying the transform to the SMPL models sequence...")
    return transform_SMPL_sequence(pred_body, transform)


def main():
    # To transform the meshes using extrinsic parameters
    transformed_meshes = transform_meshes()
    with open(omni_intrinsics_file, "rb") as f:
        omni_params = pickle.load(f)

    projected_vertices, transformed_meshes = project_meshes(cam1_images_path, transformed_meshes, omni_params, output_path)
    save_projected_meshes(transformed_meshes, projected_vertices, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproject 3D meshes from kinect to omni camera")
    parser.add_argument(
        "--root", 
        type=str, 
        help="Root directory of the dataset", 
        default="/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/2022-08-11/at-a01/living-room/a01"
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
        dest="use_opencv",
        default=False,
        help="Use matrices from OpenCV",
    )

    args = parser.parse_args()
    root = args.root
    omni_intrinsics_file = args.omni_intrinsics
    use_matlab = args.use_matlab

    # Define derivations relative to the basepath
    cam1_images_path = f"{root}/omni"
    n = 0 # kinect number
    capture_dir = f"{root}/capture{n}/rgb"
    sync_file = f"{root}/synced_filenames_full.txt"
    results_folder = f"{root}/capture{n}/out_capture{n}/results_out/final_results"
    output_path = f"{root}/capture{n}/out_capture{n}/reprojected"

    # calib_dir = f"/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/2024-01-12/at-unis/lab/calib/extrinsics/k{n}-extrinsics"
    calib_dir = f"/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/2022-08-11/at-a01/living-room/calib/k{n}-omni"

    if use_matlab:
        kinect_jsonpath = f"{calib_dir}/k{n}Params.json"
        omni_jsonpath = f"{root}/omni{n}Params.json"


    else:
        cam0_to_world_pth = f"{calib_dir}/capture{n}/k{n}_rgb_cam_to_world.pkl"
        world_to_cam1_pth = f"{calib_dir}/k{n}_omni_world_to_cam.pkl"

    main()
