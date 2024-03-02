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
import trimesh


def save_dataset_files():
    """
    save dataset files in the following format
    dataset
    |-- images
    |   |-- <timestamp1>.jpg
    |   |-- <timestamp2>.jpg
    ├── poses # holds the 3D SMPLH poses as npy files
    │   ├── <timestamp1>.npy
    │   ├── <timestamp2>.npy
    ├── meshes # holds the reprojected 3D meshes as obj files holding vertices and faces
    │   ├── <timestamp1>.obj
    │   ├── <timestamp2>.obj
    |-- canonical_poses # holds the canonical/prior 3D meshes as obj files
    |   |-- <timestamp1>.npy
    |   |-- <timestamp2>.npy
    """
    pass


def main():
    save_dataset_files()

if __name__ == "__main__":
    main()