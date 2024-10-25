#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   reproject_wham_sequence.py
@Time    :   2024/10/25 17:01:30
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   None
'''
from reproject_humor_sequence import BodyModel, c2c
import torch
import trimesh
import numpy as np
import joblib
import os



def get_wham_mesh_sequence(wham_output, betas, device):
    """
    Generates a sequence of trimesh objects from WHAM output vertices and betas (similar format to the humor output).

    Parameters:
    wham_output (numpy.ndarray): The output vertices from WHAM, of shape (T, 6890, 3).
    betas (numpy.ndarray): The shape coefficients from WHAM, of shape (T, 10).
    device (torch.device): The device to run the SMPL model on (CPU or GPU).

    Returns:
    list: A list of trimesh objects, one for each frame.
    """
    T = wham_output.shape[0]  # Number of frames
    num_betas = betas.shape[1]  # Number of shape coefficients
    batch_size = T

    # Initialize the SMPL model
    pred_bm = BodyModel(
        bm_path="/home/sid/Projects/humor/body_models/smplh/male/model.npz",
        num_betas=num_betas, batch_size=batch_size
    ).to(device)

    # Run SMPL model over the vertices and betas
    pred_body = pred_bm.forward(
        betas=torch.tensor(betas, dtype=torch.float32).to(device),
        pose=torch.zeros((T, 72), dtype=torch.float32).to(device),  # Assuming zero pose
        trans=torch.zeros((T, 3), dtype=torch.float32).to(device)   # Assuming zero translation
    )

    # Extract faces from the predicted body mesh
    faces = c2c(pred_body.f)

    # Create a list of trimeshes
    wham_mesh_sequence = [
        trimesh.Trimesh(
            vertices=wham_output[i],
            faces=faces,
            process=False,
        )
        for i in range(T)
    ]

    return wham_mesh_sequence


def get_wham_parameters(wham_output):
    """
    Extracts vertices, betas, and device information from WHAM output.

    Parameters:
    wham_output (list): The WHAM output containing vertices and betas.

    Returns:
    tuple: A tuple containing:
        - verts (numpy.ndarray): The output vertices from WHAM, of shape (T, 6890, 3).
        - betas (numpy.ndarray): The shape coefficients from WHAM, of shape (T, 10).
        - device (torch.device): The device to run the SMPL model on (CPU or GPU).
    """
    verts = wham_output["verts"]
    betas = wham_output["betas"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return verts, betas, device



if __name__ == "__main__":
    # Load WHAM output
    wham_output_0_pkl_file = "/home/sid/Projects/humor/wham_output_0.pkl"
    with open(wham_output_0_pkl_file, "rb") as file:
        wham_output_0 = joblib.load(file)

    # wham_output_0 is of type collections.defaultDict
    wham_output = wham_output_0[0]
    verts, betas, device = get_wham_parameters(wham_output)
    wham_meshes = get_wham_mesh_sequence(verts, betas, device)

    # # Save the WHAM meshes
    # wham_meshes_dir = "/home/sid/Projects/humor/wham_meshes"
    # os.makedirs(wham_meshes_dir, exist_ok=True)
    # for i, mesh in enumerate(wham_meshes):
    #     mesh.export(os.path.join(wham_meshes_dir, f"wham_mesh_{i}.obj"))

    print("Done!")