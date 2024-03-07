#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   create_dataset.py
@Time    :   2024/03/02 13:52:58
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Create a dataset from the reprojected meshes (STUB)
'''

import os
import sys

sys.path.append('/home/sid/Projects/OmniScience/code/reprojection')
sys.path.append('/home/sid/Projects/OmniScience/other/humor/humor')

import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle
import json
from humor_inference.reproject_humor_sequence import make_44


def save_dataset_files(poses, meshes, dataset_dir, cam1_images_path):
    """
    Save the reprojected files to the dataset directory
    Poses: List of reprojected vertices in the frame of the omni camera
    Meshes: List of reprojected (tri)meshes in the frame of the omni camera
    Returns: None
    """
    image_dir = cam1_images_path
    omni_filenames = os.listdir(image_dir)

    poses_dir = os.path.join(dataset_dir, 'poses')
    meshes_dir = os.path.join(dataset_dir, 'meshes')
    # canonical_poses_dir = os.path.join(dataset_dir, 'canonical_poses')
    
    os.makedirs(poses_dir, exist_ok=True)
    os.makedirs(meshes_dir, exist_ok=True)
    # os.makedirs(canonical_poses_dir, exist_ok=True)
    
    # # Load canonical poses
    # canonical_poses_path = os.path.join(results_folder, "canonical_poses.npy")
    # canonical_poses = np.load(canonical_poses_path)
    
    # Assume that poses and canonical poses are in the same order and number
    print("[*] Saving the reprojected files...")
    for i, (pose, mesh) in tqdm(enumerate(zip(poses, meshes)), total=len(poses)):  # , canonical_poses
        # Get timestamp using the omni image file names
        timestamp = omni_filenames[i].split(".")[0]        

        # Save pose
        pose_path = os.path.join(poses_dir, f"{timestamp}.npy")
        # np.save(pose_path, pose.cpu().numpy())  # Assuming pose is a torch tensor
        np.save(pose_path, pose)
        
        # Save mesh as .obj file
        mesh_path = os.path.join(meshes_dir, f"{timestamp}.obj")

        with open(mesh_path, 'w') as file:
            for v in mesh.vertices:  # Assuming mesh.vertices is a list of vertex positions
                file.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in mesh.faces:  # Assuming mesh.faces is a list of vertex indices that make up the face
                file.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        # Save canonical pose
        # canonical_pose_path = os.path.join(canonical_poses_dir, f"{timestamp}.npy")
        # np.save(canonical_pose_path, canonical_pose)

    print("[*] Done saving the reprojected files!")


def get_calib_paths(root, use_matlab, n):
    calib_dir = f"{os.path.dirname(root)}/calib/k{n}-omni"
    kinect_jsonpath, omni_jsonpath, cam0_to_world_pth, world_to_cam1_pth = None, None, None, None
    if use_matlab:
        kinect_jsonpath = f"{calib_dir}/k{n}Params.json"
        omni_jsonpath = f"{calib_dir}/omni{n}Params.json"
    else:
        cam0_to_world_pth = f"{calib_dir}/capture{n}/k{n}_rgb_cam_to_world.pkl"
        world_to_cam1_pth = f"{calib_dir}/k{n}_omni_world_to_cam.pkl"
    
    for path in [kinect_jsonpath, omni_jsonpath, cam0_to_world_pth, world_to_cam1_pth]:
        if path is not None and not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist!")
    
    return kinect_jsonpath, omni_jsonpath, cam0_to_world_pth, world_to_cam1_pth


def get_filepaths(root, n, args):
    cam1_images_path = f"{root}/omni"
    capture_dir = f"{root}/capture{n}/rgb"
    sync_file = f"{root}/synced_filenames_full.txt"
    output_path = f"{root}/capture{n}/out_capture{n}/reprojected"
    results_folder = f"{root}/capture{n}/out_capture{n}/results_out/final_results"
    if args.partial_meshes:
        results_folder = f"{root}/out_capture{n}/results_out"
    
    for path in [cam1_images_path, capture_dir, sync_file, results_folder]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist!")
    
    return capture_dir, cam1_images_path, sync_file, output_path, results_folder


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

def get_transformation_matrix_opencv(cam0_to_world_pth, world_to_cam1_pth):
    with open(cam0_to_world_pth, "rb") as f:
        cam0_to_world = make_44(pickle.load(f))

    with open(world_to_cam1_pth, "rb") as f:
        world_to_cam1 = make_44(pickle.load(f))

    transform = world_to_cam1 @ cam0_to_world
    transform[:3, 3] = transform[:3, 3] / 1000.0
    return transform


def get_transformation_matrix_matlab(kinect_jsonpath, omni_jsonpath):
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


def get_kinect_list(root):
    # check for non-empty final_results folders in root/capture{n}/out_capture{n}/final_results for n in range(3) for a file named stage3_results.npz
    n_list = []
    for i in range(3):
        results_folder = f"{root}/capture{i}/out_capture{i}/results_out/final_results"
        # check for a file named stage3_results.npz
        if os.path.isfile(os.path.join(results_folder, "stage3_results.npz")):
            n_list.append(i)
    return n_list             
