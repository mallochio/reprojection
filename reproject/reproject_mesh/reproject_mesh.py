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
sys.path.append('/home/sid/Projects/OmniScience/other/humor')

# os.environ["CUDA_VISIBLE_DEVICES"]="1"
from humor.fitting.fitting_utils import load_res, prep_res, run_smpl
from humor.body_model.body_model import BodyModel
from humor.body_model.utils import SMPL_JOINTS
from humor.fitting.eval_utils import SMPL_SIZES
from humor.utils.torch import copy2cpu as c2c

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


def get_synced_meshes(transformed_meshes: List[trimesh.Trimesh]):
    df = pd.read_csv(sync_file, sep=";", header=0)
    capture_files = sorted(os.listdir(capture_dir))

    mesh_indices = []
    for ix, val in df.iterrows():
        # get index of val["capture0"] in capture_files
        mesh_index = capture_files.index(val["capture2"])
        mesh_indices.append([mesh_index, val["omni"]])

    # Select only the meshes from transformed_meshes which are in mesh_indices
    transformed_meshes_new = [transformed_meshes[i[0]] for i in mesh_indices]
    synced_cam1_files = [i[1] for i in mesh_indices]
    return synced_cam1_files, transformed_meshes_new


def render_mesh(img, img_path, mesh, vertices_2d, output_dir=None, i=None):
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


def project_meshes(
    images_dir: str,
    mesh_seq: List[trimesh.Trimesh],
    camera_calib: dict,
    output_dir: str,
):
    """
    Project the mesh sequence on the omni images.
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

    for i, (mesh, img_path) in tqdm(
        enumerate(zip(mesh_seq, images)), total=len(images)
    ):
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

    cv2.destroyAllWindows()
    print("[*] Done!")
    return vertices_2d, mesh


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


def get_transformation_matrix():
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

    meo = np.zeros((4, 4))
    meo[0:3, 0:3] = R_omni
    meo[:3, 3] = t_omni.ravel() / 1000.
    meo[3, :] = np.asarray([0, 0, 0, 1])

    mei = np.zeros((4, 4))
    mei[0:3, 0:3] = R_kinect
    mei[:3, 3] = t_kinect.ravel() / 1000.
    mei[3, :] = np.asarray([0, 0, 0, 1])

    return np.matmul(meo, np.linalg.pinv(mei))

def process_meshes():    
    res_file = os.path.join(results_folder, "stage3_results.npz")
    if not os.path.isfile(res_file):
        raise Exception(f"Could not find {res_file}!")

    device = torch.device("cpu")
    pred_res = np.load(res_file)
    T = pred_res["trans"].shape[0]

    sanitize_preds(pred_res, T)
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

    # humor model
    pred_bm = BodyModel(bm_path=optim_bm_path, num_betas=num_pred_betas, batch_size=T).to(device)

    # run through SMPL
    pred_body = run_smpl(pred_res, pred_bm)
    print("[*] Loaded the sequence of SMPL models!")

    transform = get_transformation_matrix()
    print("[*] Applying the transform to the SMPL models sequence...")
    return transform_SMPL_sequence(pred_body, transform)


def main():
    # To render the projected meshes on the images
    transformed_meshes = process_meshes()
    with open(omni_intrinsics_file, "rb") as f:
        omni_params = pickle.load(f)

    vertices_2d, mesh = project_meshes(cam1_images_path, transformed_meshes, omni_params, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproject 3D meshes from kinect to omni camera")
    parser.add_argument(
        "--root", 
        type=str, 
        help="Root directory of the dataset", 
        default="/home/sid/Projects/OmniScience/dataset/session-recordings/2024-01-12/at-unis/lab/sid"
    )
    parser.add_argument(
        "--omni_intrinsics", 
        type=str, 
        help="Path to the omni intrinsics file", 
        default="/home/sid/Projects/OmniScience/code/reprojection/calibration/intrinsics/omni_intrinsics_flipped_2024.pkl"
    )
    args = parser.parse_args()
    root = args.root
    omni_intrinsics_file = args.omni_intrinsics

    # Define derivations relative to the basepath
    cam1_images_path = f"{root}/omni"
    capture_dir = f"{root}/capture2/rgb"
    sync_file = f"{root}/synced_filenames_full.txt"
    results_folder = f"{root}/capture2/results_out/final_results"
    output_path = f"{root}/capture2/results_out/reprojected"
    kinect_jsonpath = f"{root}/k2Params.json"
    omni_jsonpath = f"{root}/omni2Params.json"

    main()
