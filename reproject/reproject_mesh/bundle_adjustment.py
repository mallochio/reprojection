#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   bundle_adjustment.py
@Time    :   2024/03/07 13:04:57
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Functions to help do bundle adjustment on the reprojected meshes (TODO - This is a stub, needs to be filled in)
'''

from typing import List
import trimesh
import numpy as np
import cv2
from scipy.optimize import minimize


def perform_bundle_adjustment(vertices_list: List[np.ndarray], meshes_list: List[trimesh.Trimesh]):
    pass