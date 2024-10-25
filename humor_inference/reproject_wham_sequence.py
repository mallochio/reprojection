#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   reproject_wham_sequence.py
@Time    :   2024/10/25 17:01:30
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Reproject WHAM pose estimation outputs to camera 1 using world coordinates.
'''

import argparse
import os
import pickle
from typing import Optional, Dict, List

import torch
import trimesh
import numpy as np
import joblib
from tqdm import tqdm

# Import functions and classes from the HUMOR script
from reproject_humor_sequence import (
    BodyModel,
    c2c,
    render_on_images,
    make_44,
    export_timestamped_mesh_seq,
    sanitize_preds  # If needed
)


def get_wham_mesh_sequence(wham_output, betas, device):
    """
    Generates a sequence of trimesh objects from WHAM output vertices and betas.

    Parameters:
    - wham_output (dict): The WHAM output for a single sequence.
    - betas (numpy.ndarray): Shape coefficients.
    - device (torch.device): Device to perform computations on.

    Returns:
    - List[trimesh.Trimesh]: List of mesh objects for each frame.
    """
    T = betas.shape[0]  # Number of frames
    num_betas = betas.shape[1]  # Number of shape coefficients

    # Initialize the BodyModel with world parameters
    pred_bm = BodyModel(
        bm_path="/home/sid/Projects/humor/body_models/smplh/male/model.npz",
        num_betas=num_betas,
        batch_size=T,
        model_type="smplh"  # Adjust based on your model type
    ).to(device)

    # Since WHAM provides the vertices directly in world coordinates, we can construct the meshes directly
    faces = pred_bm.bm.faces_tensor  # Assuming same faces as HUMOR
    faces = c2c(faces)  # Convert to NumPy

    wham_mesh_sequence = [
        trimesh.Trimesh(
            vertices=wham_output["verts"][i],
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
    - wham_output (dict): The WHAM output for a single sequence.

    Returns:
    - tuple:
        - verts (numpy.ndarray): Vertices from WHAM, shape (T, 6890, 3).
        - betas (numpy.ndarray): Shape coefficients, shape (T, 10).
        - device (torch.device): Computation device.
    """
    verts = wham_output["verts"]  # Shape: (T, 6890, 3)
    betas = wham_output["betas"]  # Shape: (T, 10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return verts, betas, device


def sanitize_wham_preds(pred_res, T):
    """
    Sanitize the predictions from WHAM by replacing NaNs and infinite values.

    Args:
        pred_res (dict): Dictionary containing the predictions.
        T (int): Number of frames.

    Returns:
        dict: Sanitized predictions.
    """
    for smpk, size in {"trans_world": 3, "betas": 10, "pose_world": 72}.items():
        if smpk not in pred_res:
            print(f"Warning: Key '{smpk}' not found in predictions. Skipping sanitization for this key.")
            continue

        data = np.array(pred_res[smpk], dtype=np.float32, copy=True)
        invalid_mask = ~np.isfinite(data)

        # Check if there are any invalid values
        if np.any(invalid_mask):
            print(f"Found NaNs or infinite values in prediction for {smpk}.")

            if smpk == "betas":
                # Replace invalid betas with zeros
                data[invalid_mask] = 0.0
            else:
                # For 'trans_world' and 'pose_world', replace invalid entries with the median of valid data
                for t in range(T):
                    if np.all(invalid_mask[t]):
                        # If all values in the slice are invalid, set to zeros
                        data[t] = 0.0
                    elif np.any(invalid_mask[t]):
                        valid_data = data[t][~invalid_mask[t]]
                        if valid_data.size > 0:
                            median = np.nanmedian(valid_data)
                            data[t][invalid_mask[t]] = median
                        else:
                            data[t] = 0.0
            pred_res[smpk] = data

    return pred_res


def transform_mesh_sequence(mesh_seq: List[trimesh.Trimesh], transform: np.ndarray) -> List[trimesh.Trimesh]:
    """
    Apply a transformation matrix to each mesh in the sequence.

    Args:
        mesh_seq (List[trimesh.Trimesh]): List of trimesh.Trimesh objects.
        transform (numpy.ndarray): 4x4 transformation matrix.

    Returns:
        List[trimesh.Trimesh]: List of transformed trimesh.Trimesh objects.
    """
    print(f"[*] Processing sequence of {len(mesh_seq)} frames...")
    transformed_mesh_seq = []
    for mesh in tqdm(mesh_seq, total=len(mesh_seq)):
        transformed_mesh = mesh.copy()  # To avoid modifying the original mesh
        transformed_mesh.apply_transform(transform)
        transformed_mesh_seq.append(transformed_mesh)
    return transformed_mesh_seq


def main(
        cam0_to_world_pth: str,
        world_to_cam1_pth: str,
        wham_output_path: str,
        cam1_images_path: str,
        output_path: Optional[str] = None,
        cam1_calib_pth: Optional[str] = None,
    ) -> Optional[Dict[int, trimesh.Trimesh]]:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load WHAM output
    wham_output_0_pkl_file = os.path.join(wham_output_path, "wham_output.pkl")
    if not os.path.isfile(wham_output_0_pkl_file):
        print(f"Could not find {wham_output_0_pkl_file}!")
        return None

    with open(wham_output_0_pkl_file, "rb") as file:
        wham_output_0 = joblib.load(file)

    # Assuming wham_output_0 is a defaultdict with integer keys
    # Here, we consider key '0'; adjust if multiple sequences are present
    wham_output = wham_output_0[0]  # Adjust based on WHAM's output structure

    # Sanitize predictions for world coordinates
    wham_output = sanitize_wham_preds(wham_output, T=wham_output["verts"].shape[0])

    verts, betas, device = get_wham_parameters(wham_output)
    wham_meshes = get_wham_mesh_sequence(wham_output, betas, device)

    # Load transformation matrices
    with open(cam0_to_world_pth, "rb") as f:
        cam0_to_world = make_44(pickle.load(f))
    with open(world_to_cam1_pth, "rb") as f:
        world_to_cam1 = make_44(pickle.load(f))

    # Compute the transformation: from cam0 world to cam1 world
    transform = world_to_cam1 @ cam0_to_world
    transform[:3, 3] = transform[:3, 3] / 1000.0  # Convert units if necessary

    print("[*] Applying the transform to the WHAM models sequence...")
    transformed_meshes = transform_mesh_sequence(wham_meshes, transform)

    if cam1_calib_pth is None:
        # Return a dictionary of the transformed meshes where the keys are the matching image names (timestamps)
        return export_timestamped_mesh_seq(cam1_images_path, transformed_meshes)
    else:
        with open(cam1_calib_pth, "rb") as f:
            cam1_calib = pickle.load(f)
        output_path = output_path if output_path is not None else "projected_output_viz"
        os.makedirs(output_path, exist_ok=True)
        render_on_images(cam1_images_path, transformed_meshes, cam1_calib, output_path)

    return None



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reproject WHAM pose estimation outputs in world coordinates to another camera."
    )

    parser.add_argument(
        "--wham_results_dir",
        type=str,
        default="/home/sid/Projects/WHAM/output/demo/2023-02-09_at-unis_lab_a01_capture0",
        help="Path to the directory containing the WHAM results.",
    )
    parser.add_argument(
        "--cam1_images_dir",
        type=str,
        default="/home/NAS-mountpoint/kinect-omni-ego/2023-02-09/at-unis/lab/a01/omni",
        help="Path to cam1 images directory for rendering.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/sid/Projects/WHAM/output/demo/2023-02-09_at-unis_lab_a01_capture0/reprojected",
        help="Path to save visualizations to.",
    )
    parser.add_argument(
        "--cam0-to-world",
        type=str,
        # dest="cam0_to_world",
        default="/home/NAS-mountpoint/kinect-omni-ego/2023-02-09/at-unis/lab/calib/k0-omni/capture0/k0_rgb_cam_to_world.pkl",
        help="Pickle file containing the [R|t] matrix from camera 0 to world coordinates.",
    )
    parser.add_argument(
        "--world-to-cam1",
        type=str,
        # dest="world_to_cam1",
        default="/home/NAS-mountpoint/kinect-omni-ego/2023-02-09/at-unis/lab/calib/k0-omni/k0_omni_world_to_cam.pkl",
        help="Pickle file containing the [R|t] matrix from world to camera 1 coordinates.",
    )
    parser.add_argument(
        "--cam1-calib",
        type=str,
        # dest="cam1_calib",
        default="/home/sid/Projects/reprojection/calibration/intrinsics/omni_calib.pkl",
        help="Pickle file containing camera 1 calibration (intrinsics, distortion, etc.). If not set, rendering will be skipped.",
    )
    args = parser.parse_args()
    main(
        cam0_to_world_pth=args.cam0_to_world,
        world_to_cam1_pth=args.world_to_cam1,
        wham_output_path=args.wham_results_dir,
        cam1_images_path=args.cam1_images_dir,
        output_path=args.output_dir,
        cam1_calib_pth=args.cam1_calib,
    )