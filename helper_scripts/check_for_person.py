#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   check_for_person.py
@Time    :   2023/01/12 13:27:31
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Script to check if a person is in the image by using the mocap output
'''

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
from tqdm import tqdm
from typing import Dict


def get_mesh_dict(checkpath: str) -> Dict:
    # Get the pickle files from the mocap output directory
    pickle_dict = {}
    files = [i for i in os.listdir(checkpath) if i.endswith(".pkl")]


    for i in tqdm(files):
        if i.endswith(".pkl"):
            df = pd.read_pickle(str(Path(checkpath, i)))
            # An if condition to check if the pickle has a person in it (this was seen during tests)
            if df["pred_output_list"][0] is not None:
                pickle_dict[i] = df
    return pickle_dict


def check_for_person(
    obj: Dict, 
    threshold: float, 
    image_width: int = 1280, 
    image_height: int = 720
) -> bool:
    # Check if a person is in the image by using frankmocap meshes by checking
    # if the image coordinates of the person are within the image boundaries by above a threshold
    person_image_coordinates = obj["pred_output_list"][0]["pred_vertices_img"]

    # Convert list of coordinates to a NumPy array for easier indexing and boolean indexing
    coordinates = np.array(person_image_coordinates)

    # Check if the mesh vertices are within the image boundaries
    x_in_bounds = np.logical_and(coordinates[:, 0] >= 0, coordinates[:, 0] < image_width)
    y_in_bounds = np.logical_and(coordinates[:, 1] >= 0, coordinates[:, 1] < image_height)

    # Check if the percent of vertices within the image boundaries is above a threshold
    return np.mean(np.logical_and(x_in_bounds, y_in_bounds)) > threshold


def main(picklepath: str, threshold: float, output_path: str) -> None:
    # Check if the person is within the image boundaries by above a threshold
    print("=================================================================")
    print(f"[*] Checking for person in {picklepath}")
    mesh_dict = get_mesh_dict(picklepath)
    print(f"[*] Checking for person in {len(mesh_dict)} images")
    person_in_image = [
        key
        for key, value in tqdm(mesh_dict.items())
        if check_for_person(
            value, threshold, image_width=1280, image_height=720
        )
    ]
    # (Over)Write the list of images with a person in it to a file
    print("=================================================================")
    if not output_path.endswith(".txt"):
        output_path = str(Path(output_path, "person_detected.txt"))

    print(f"[*] Writing list to {output_path}")
    with open(output_path, "w") as f:
        for item in tqdm(person_in_image):
            # Point to the corresponding image that the pickle file is generated from
            image_name = item.split("_")[0] + ".jpg"
            image_name = f"{Path(picklepath).parents[1]}/rgb/{image_name}"
            f.write(f"{image_name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pickle-path", "-i",
        help="Path to the pickle files from running frankmocap",
        dest="picklepath",
        type=str,
        required=True,
        # default="/home/sid/Projects/OmniScience/dataset/2022-10-06/bedroom/sid/round1/capture1/mocap_output/mocap",
    )
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument(
        "--output-path", "-o",
        help="Path to write the list of images with a person in it",
        dest="outputpath",
        type=str,
        required=True,
        # default="/home/sid/Projects/OmniScience/dataset/2022-10-06/bedroom/sid/round1/capture1/mocap_output",
    )
    args = parser.parse_args()
    main(args.picklepath, args.threshold, args.outputpath)
