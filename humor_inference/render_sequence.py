#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Viusalize a dataset sequence with the computed pose annotations on all view points and render
videos.
"""

import argparse
import os
import pickle
import cv2 as cv
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageOps, ImageFile

from tqdm import tqdm

import numpy as np
import trimesh

ImageFile.LOAD_TRUNCATED_IMAGES = True

ANNOTATION_FILENAME = "pose_labels.pkl"
CAM_INTRINSICS_PATH = {
    "capture0": "../calibration/intrinsics/k0_rgb_intrinsics_new.pkl",
    "capture1": "../calibration/intrinsics/k1_rgb_intrinsics_new.pkl",
    "omni": "../calibration/intrinsics/omni_intrinsics_new.pkl",
}

def render_on_images(
    images_dir: str,
    timed_meshes: List[Tuple[int, trimesh.Trimesh]],
    camera_intrinsics: dict,
    output_dir: str,
):
    """
    Render the mesh sequence on the images in the directory.
    """
    # Load the camera intrinsics and distortion coefficients from the pickle file
    camera_matrix = camera_intrinsics["intrinsics"]
    dist_coeffs = camera_intrinsics["distortion"]
    xi = camera_intrinsics.get("xi")
    use_omni = xi is not None
    xi = xi.item() if isinstance(xi, np.ndarray) else xi

    # Load all images in the directory
    images = []
    extension = None
    for filename in os.listdir(images_dir):
        # If the file type is an image, load it
        if filename.endswith(".png") or filename.endswith(".jpg"):
            images.append(os.path.join(images_dir, filename))
        if extension is None:
            extension = filename.split(".")[-1]
    images.sort()
    for timestamp, mesh in tqdm(timed_meshes):
        img_path = os.path.join(images_dir, f"{timestamp:08d}.{extension}")
        img = Image.open(img_path)
        assert (img.size[1], img.size[0]) == camera_intrinsics[
            "img_shape"
        ], "Image shape must match the camera calibration!"
        # Project the vertices on the image
        # We've already transformed the vertices to the camera frame, so we can just 
        # project them on the image without any transformation.
        if not isinstance(mesh, list):
            meshlist = [mesh]
        else:
            print("Rendering multiple meshes")
            meshlist = mesh
        colours = ["gray", "red", "green", "blue", "yellow", "orange", "purple"]
        for i, mesh in enumerate(meshlist):
            if use_omni:
                vertices_2d, _ = cv.omnidir.projectPoints(
                    np.expand_dims(mesh.vertices, axis=0),
                    np.zeros(3),
                    np.zeros(3),
                    camera_matrix,
                    xi,
                    dist_coeffs,
                )
                vertices_2d = np.swapaxes(vertices_2d, 0, 1)
            else:
                vertices_2d, _ = cv.projectPoints(
                    mesh.vertices, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs
                )

            try:
                img = ImageOps.mirror(img) # TODO: Why is this needed?
            except OSError:
                print(f"Error in file {img_path}")
                continue

            draw = ImageDraw.Draw(img)
            for face in mesh.faces:
                # Numpy version:
                # face_vertices = np.clip(vertices_2d[face], 0, img.size)
                # points = [tuple(p[0]) for p in face_vertices]
    
                face_vertices = vertices_2d[face]
                points = [max(min(tuple(p[0]), img.size), (0, 0)) for p in face_vertices]
                draw.polygon(
                    points,
                    fill=None,
                    outline=colours[i],
                    width=1,
                )
        # Save the image
        img.save(os.path.join(output_dir, f"{timestamp:08d}.png"))
    print("[*] Done!")


def main(root: str):
    """
    Sequence structure:
     ̶d̶a̶t̶a̶s̶e̶t̶_̶p̶a̶t̶h̶/̶
        s̶e̶q̶u̶e̶n̶c̶e̶_̶d̶a̶t̶e̶/̶
            l̶o̶c̶a̶t̶i̶o̶n̶/̶
                r̶o̶o̶m̶/̶
                    c̶a̶l̶i̶b̶/̶
                        k̶0̶-̶o̶m̶n̶i̶/̶
                            e̶x̶t̶r̶i̶n̶s̶i̶c̶s̶.̶p̶k̶l̶
                            c̶a̶p̶t̶u̶r̶e̶0̶/̶
                                e̶x̶t̶r̶i̶n̶s̶i̶c̶s̶.̶p̶k̶l̶
                            c̶a̶p̶t̶u̶r̶e̶1̶
                            .̶.̶.
                            o̶m̶n̶i̶
                        k̶1̶-̶o̶m̶n̶i̶/̶
                            e̶x̶t̶r̶i̶n̶s̶i̶c̶s̶.̶p̶k̶l̶
                            .̶.̶.̶
                        .̶.̶.̶
                    participant/ <-- root
                        capture0/ <-- Kinect 0 capture
                            rgb_folder_name/
                                rgb_frames.jpg
                            depth_folder_name/
                                depth_frames.png
                            iuv_folder_name/
                                iuv_frames.png
                        capture1/ <-- Kinect 1 capture
                            ...


                        ...
                        omni/ <-- Omni capture
    """
    assert os.path.isfile(os.path.join(root, ANNOTATION_FILENAME)), 'No annotation file found.'
    with open(os.path.join(root, ANNOTATION_FILENAME), "rb") as f:
        meshes = pickle.load(f)
    for capture_root, dirs, _ in os.walk(root):
        capture_name = os.path.basename(capture_root)
        if capture_name == "rendered":
            dirs.clear()
            continue
        if capture_name not in CAM_INTRINSICS_PATH:
            continue
        output_path = os.path.join(root, "rendered", capture_name)
        cam_timed_meshes: List[Tuple[int, trimesh.Trimesh]] = []
        for frame in meshes:
            if capture_name in frame:
                cam_timed_meshes.append(frame[capture_name])
        with open(CAM_INTRINSICS_PATH[capture_name], "rb") as f:
            cam_intrinsics = pickle.load(f)
        os.makedirs(output_path, exist_ok=True)
        img_dir = os.path.join(capture_root, "rgb") if capture_name != "omni" else capture_root
        print(f"[*] Rendering {capture_name}...")
        render_on_images(img_dir, cam_timed_meshes, cam_intrinsics, output_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, help='Path to the sequence directory.')
    args = parser.parse_args()
    main(args.root)
