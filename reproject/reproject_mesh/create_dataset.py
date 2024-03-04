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
import numpy as np
from PIL import Image


def save_dataset_files(poses, meshes, dataset_dir):
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