#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   run_reprojections.py
@Time    :   2024/05/20 16:12:23
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Reproject skeletons to the top view 
"""


import sys; sys.path.append('/home/sid/Projects/reprojection')
import os
import json
import traceback
from pathlib import Path
import numpy as np
import argparse
from typing import List
from tqdm import tqdm
from reproject.reproject_mesh.reprojection_utils import (
    get_calib_paths,
    get_transformation_matrix_opencv
)


def save_reprojections(skeletons: List[List], output_dir: str, filenames: List[str]) -> None:
    """Save reprojected skeletons to JSON files."""
    output_dir = Path(output_dir)
    print(f"[*] Saving reprojected skeletons at {output_dir}...")
    for skel, filename in tqdm(zip(skeletons, filenames)):
        skel_data = {"pose_world_landmarks": skel.tolist()}  # Convert numpy array to list
        json_path = output_dir / f"{Path(filename).stem}.json"
        with open(json_path, 'w') as outfile:
            json.dump(skel_data, outfile)

def transform_skeletons(pred_body: np.ndarray, cam0_to_world_pth: str, world_to_cam1_pth: str) -> np.ndarray:
    """Transform skeleton coordinates using a transformation matrix."""
    transform = get_transformation_matrix_opencv(cam0_to_world_pth, world_to_cam1_pth)
    homo_skel = np.concatenate([pred_body, np.ones((1, 33, 1))], axis=-1)
    homo_skel = homo_skel.transpose(0, 2, 1).reshape(4, -1)
    transformed_skel = transform @ homo_skel
    transformed_skel = transformed_skel[:3].transpose(1, 0).reshape(1, 33, 3)
    return transformed_skel

def get_skeletons(results_folder: str) -> np.ndarray:
    """Load skeletons from JSON files in a directory."""
    world_poses = []
    json_files = Path(results_folder).glob("*.json")
    for json_file in sorted(json_files):
        with open(json_file, 'r') as f:
            data = json.load(f)
            pose_world_landmarks = np.array(data["pose_world_landmarks"])
        world_poses.append(pose_world_landmarks)
    return world_poses

def get_filepaths_skel(root: str, n: int) -> tuple:
    """Construct various paths needed for processing."""
    root = Path(root)
    cam1_images_path = root / "omni"
    capture_dir = root / f"capture{n}" / "rgb"
    sync_file = root / "synced_filenames_full.txt"
    results_folder = root / f"capture{n}" / f"out_capture{n}" / "rgb_preprocess" / "mp_keypoints_3d"
    output_path = root / f"capture{n}" / f"out_capture{n}" / "rgb_preprocess" / "mp_keypoints_3d_reprojected"

    for path in [cam1_images_path, capture_dir, sync_file, results_folder]:
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist!")

    output_path.mkdir(parents=True, exist_ok=True)
    return capture_dir, cam1_images_path, sync_file, output_path, results_folder

def project_single(n: int, root: Path) -> None:
    """Process skeletons for a single capture session."""
    use_matlab = False
    cam0_images_path, cam1_images_path, sync_file, output_path, results_folder = get_filepaths_skel(root, n)
    kinect_jsonpath, omni_jsonpath, cam0_to_world_pth, world_to_cam1_pth = get_calib_paths(root, use_matlab, n)
    skeletons = get_skeletons(results_folder)
    transformed_skeletons = []

    print("[*] Projecting skeletons...")
    for pred_body in tqdm(skeletons):
        if len(pred_body) == 0:
            transformed_skeletons.append(pred_body)
        else:
            skel_projected = transform_skeletons(
                pred_body=pred_body,
                cam0_to_world_pth=cam0_to_world_pth,
                world_to_cam1_pth=world_to_cam1_pth
            )
            transformed_skeletons.append(skel_projected)

    save_reprojections(
        skeletons=transformed_skeletons,
        output_dir=output_path,
        filenames=[file.name for file in sorted(Path(cam0_images_path).glob("*.jpg"))]
    )

def main(args: argparse.Namespace) -> None:
    """Main function to walk through the root directory and process each capture session."""
    for root_dir, dirs, _ in os.walk(args.root):
        root_path = Path(root_dir)
        if "calib" in root_path.parts:
            dirs.clear()
        elif any("capture" in d for d in dirs):
            print(f"[*] Processing {root_path}...")
            capture_dirs = [d for d in dirs if "capture" in d]
            for capture_dir in capture_dirs:
                n = int(capture_dir[-1])
                try:
                    project_single(n, root_path)
                except Exception as e:
                    print(f"Error processing {root_path}: {e}")
                    print(traceback.format_exc())
            dirs.clear()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproject skeletons to the top view")
    parser.add_argument("--root", type=str, required=True, help="Path to the root directory containing the data")
    args = parser.parse_args()
    main(args)
