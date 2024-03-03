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


def save_dataset_files(poses, meshes, dataset_dir, results_folder):
    # Create the necessary directories if they do not exist
    images_dir = os.path.join(dataset_dir, 'images')
    poses_dir = os.path.join(dataset_dir, 'poses')
    meshes_dir = os.path.join(dataset_dir, 'meshes')
    canonical_poses_dir = os.path.join(dataset_dir, 'canonical_poses')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(poses_dir, exist_ok=True)
    os.makedirs(meshes_dir, exist_ok=True)
    os.makedirs(canonical_poses_dir, exist_ok=True)
    
    # Load canonical poses
    canonical_poses_path = os.path.join(results_folder, "canonical_poses.npy")
    canonical_poses = np.load(canonical_poses_path)
    
    # Assume that poses and canonical poses are in the same order and number
    for i, (pose, mesh, canonical_pose) in enumerate(zip(poses, meshes, canonical_poses)):
        timestamp = f"{i:08d}"  # Or use another way to generate unique timestamps
        
        # Load and save image
        image_path = os.path.join(dataset_dir, 'omni', f"{timestamp}.jpg")
        try:
            image = Image.open(image_path)
            image.save(os.path.join(images_dir, f"{timestamp}.jpg"))
        except FileNotFoundError:
            print(f"Image file not found: {image_path}")
            continue
        
        # Save pose
        pose_path = os.path.join(poses_dir, f"{timestamp}.npy")
        np.save(pose_path, pose.cpu().numpy())  # Assuming pose is a torch tensor
        
        # Save mesh as .obj file
        mesh_path = os.path.join(meshes_dir, f"{timestamp}.obj")
        with open(mesh_path, 'w') as file:
            for v in mesh.vertices:  # Assuming mesh.vertices is a list of vertex positions
                file.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in mesh.faces:  # Assuming mesh.faces is a list of vertex indices that make up the face
                file.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        # Save canonical pose
        canonical_pose_path = os.path.join(canonical_poses_dir, f"{timestamp}.npy")
        np.save(canonical_pose_path, canonical_pose)

    print("Dataset files saved successfully!")
