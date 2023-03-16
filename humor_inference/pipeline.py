#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
HuMoR-based annotation pipeline.
"""

import argparse
import os
import subprocess
import numpy as np
import trimesh
from contextlib import redirect_stdout
from typing import Dict, List
from reproject_humor_sequence import main as reproject
from preprocess import main as preprocess

RGB_FOLDER = "rgb"
IR_FOLDER = "ir"
DEPTH_FOLDER = "depth"
IUV_FOLDER = "rgb_dp2_iuv"
MASK_FOLDER = "rgb_dp2_mask"
OUTPUT_FOLDER = "humor_output"
PREPROCESS_FOLDER = "preprocessed"
PREPROCESS_THRESHOLD = 0.24

CAM_INTRINSICS_PATH = {
    "k0": "../calibration/intrinsics/k0_rgb_calib.json",
    "k1": "../calibration/intrinsics/k1_rgb_calib.json",
    "omni": "../calibration/intrinsics/omni_calib.json",
}


def annotate_capture(
    capture_path: str,
    humor_docker_script: str,
    base_cam_intrinsics_path: str,
    basecam_to_world_pth: str,
    world_to_destcam_pth: str,
) -> Dict[int, trimesh.Trimesh]:
    """Go through a capture folder, annotate each subsequence where a person was detected, and
    reproject the HuMoR output to the other camera. Then, join the reprojected mesh sequences into
    the final annotation for the sequence.
    Args:
        capture_path (str): Path to the capture folder.
        humor_docker_script (str): Path to the HuMoR Docker script that will run the docker image.
        basecam_to_world_pth (str): Path to the origin camera (which recorded the capture) to world transformation matrix.
        world_to_destcam_pth (str): Path to the world to the destination camera (to which we are reprojecting, i.e. the omni cam) transformation matrix.
    Returns:
        dict: key=timestamp of fitted image (base cam), value=reprojected fitted mesh (dest cam)
    """
    print(f"[*] Annotating capture {capture_path}...")
    if not os.path.isdir(os.path.join(capture_path, PREPROCESS_FOLDER)):
        print("\t-> Preprocessing...")
        # Redirect the prints to a log file
        with open(os.path.join(capture_path, "preprocess.log"), "w") as f:
            with redirect_stdout(f):
                preprocess(
                    capture_path,
                    os.path.join(capture_path, PREPROCESS_FOLDER),
                    debug=False,
                    threshold=PREPROCESS_THRESHOLD,
                )
        print("\t-> Preprocessing done.")
    sequence_meshes = {}
    # Compile each "person" subsequence into a video file and process it
    for root, _, _ in os.walk(os.path.join(capture_path, PREPROCESS_FOLDER)):
        seq_name = os.path.basename(root)
        if "no_person" in seq_name or "person" not in seq_name:
            continue
        print(f"Capture path is {capture_path}")
        print(f"\t-> Processing {seq_name}...")
        # TODO: Maybe patch HuMoR so that it loads the images instead? The only problem is that
        # HuMoR works for 30hz videos, so we would need to interpolate or duplicate the frames
        # maybe? With these parameters, FFMPEG just builds a 30hz video from the images but since
        # we recorded at ~15hz, the video looks sped up. That might be the best way to deal with
        # the disparity because the movements remain smooth, just faster.
        output_vid_file = os.path.join(
            capture_path, PREPROCESS_FOLDER, f"{seq_name}.mp4"
        )
        if not os.path.exists(output_vid_file):
            print(f"\t\t-> Compiling {seq_name} into video file {output_vid_file}...")
            os.system(
                f"ffmpeg -framerate 30 -pattern_type glob -i '{root}/*.jpg' -c:v"
                + f" libx264 -r 30 -loglevel quiet {output_vid_file}"
            )
            print(f"\t\t-> Output file {output_vid_file} created.")
        with open(os.path.join(capture_path, "humor.log"), "w") as f:
            with redirect_stdout(f):
                # TODO: Check if HuMoR was already ran on this subsequence and skip it if so
                humor_was_run = os.path.isfile(os.path.join(capture_path, OUTPUT_FOLDER, "final_results", "stage3_results.npz"))
                # Run the HuMoR Docker script
                # TODO: Write a bash script that will run the docker image and the inference script.
                # For a first single-threaded PoC, the input video file and output folder should
                # probably be fixed.
                humor_output_path = os.path.join(capture_path, OUTPUT_FOLDER)
                if os.path.isdir(humor_output_path):
                    # This line emove the directory with all its contents:
                    os.system(f"rm -rf {humor_output_path}")

                if not humor_was_run:
                    # TODO: Refine this, current workflow is clunky
                    try:
                        os.system(
                            f"bash {humor_docker_script} {output_vid_file} {humor_output_path} {os.path.abspath(base_cam_intrinsics_path)}"
                        )
                    except FileNotFoundError as e:
                        raise Exception("humor failed: ", e)
                    except Exception as e:
                        raise Exception("humor failed: ", e)

                timestamped_meshes = reproject(
                    basecam_to_world_pth,
                    world_to_destcam_pth,
                    humor_output_path,
                    capture_path,
                )  # key=timestamp of fitted image (base cam), value=reprojected fitted mesh (dest cam)
                assert (
                    timestamped_meshes is not None and type(timestamped_meshes) == dict
                ), "reproject_humor_sequence.main() did not return a dict"
                print(f"\t\t-> Deleting {humor_output_path}...")
                os.system(f"rm -rf {humor_output_path}")
            for timestamp, mesh in timestamped_meshes.items():
                if timestamp in sequence_meshes:
                    # TODO: We'll need to average these boys (in the sync function?)
                    sequence_meshes[timestamp] = (
                        sequence_meshes[timestamp] + [mesh]
                        if type(sequence_meshes[timestamp]) == list
                        else [sequence_meshes[timestamp], mesh]
                    )
                else:
                    sequence_meshes[timestamp] = mesh
    return sequence_meshes


def synchronize_annotations(
    sub_sequences: List[Dict[int, trimesh.Trimesh]],
    synced_filenames_array: List[List[str]],
    root: str,
) -> Dict[int, trimesh.Trimesh]:
    """
    We synchronize the annotations from the different sub-sequences of a capture.
    For this, we take the average of the meshes for each of the synced files in the array.
    """
    merged_sequence = {}
    for synced_files in synced_filenames_array:
        # Get the timestamp of the first file
        first_file_timestamp = int(synced_files[0].split(".")[0])
        # Get the meshes for each of the synced files
        meshes = []
        for synced_file in synced_files:
            # Get the timestamp of the synced file
            synced_file_timestamp = int(synced_file.split(".")[0])
            # Get the mesh for the synced file
            for sub_sequence in sub_sequences:
                if synced_file_timestamp in sub_sequence:
                    meshes.append(sub_sequence[synced_file_timestamp])

        # Get the points of the meshes and average them
        points = []
        for mesh in meshes:
            points.append(mesh.vertices)
        points = np.mean(points, axis=0)
        # Create a new mesh with the averaged points
        new_mesh = trimesh.Trimesh(vertices=points)
        # Add the new mesh to the sub-sequences
        merged_sequence[first_file_timestamp] = new_mesh

    return merged_sequence


def get_calibration_files(root) -> Dict[str, str]:
    current_room_calib = {}
    """Get the paths to the calibration files as a dictionary."""
    # Now we are in a room folder, so we can get the extrinsics for the current room
    # Get all the sequence calibration files first by traversing calib folder, we need this to process captures
    calib_folder = os.path.join(root, "calib")
    for calib_root, _, calib_files in os.walk(calib_folder):
        for fpath in calib_files:
            if fpath.endswith(".pkl"):
                current_room_calib[
                    os.path.basename(fpath).split(".")[0]
                ] = f"{calib_root}/{fpath}"

    return current_room_calib


def annotate_participant(root, humor_docker_script, current_room_calib):
    sequence_annotations = []
    for capture_root, _, _ in os.walk(root):
        capture_name = os.path.basename(capture_root)
        if capture_name.startswith("capture"):
            kinect_id = capture_name.split("capture")[1]
            sequence_meshes = annotate_capture(
                capture_root,
                humor_docker_script,
                CAM_INTRINSICS_PATH[f"k{kinect_id}"],
                current_room_calib[f"k{kinect_id}_rgb_cam_to_world"],
                current_room_calib[f"k{kinect_id}_omni_world_to_cam"],
            )
            sequence_annotations.append(sequence_meshes)
    return sequence_annotations


def main(dataset_path: str, humor_docker_script: str):
    """
    Args:
        dataset_path (str): Path to the dataset to annotate.
    """
    """ Dataset structure:
    dataset_path/
        sequence_date/
            location/
                room/
                    calib/
                        k0-omni/
                            extrinsics.pkl
                            capture0/
                                extrinsics.pkl
                            capture1
                            ...
                            omni
                        k1-omni/
                            extrinsics.pkl
                            ...
                        ...
                    participant/
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
    assert os.path.isdir("checkpoints"), "Please run ln -s /path/to/humor/checkpoints ."

    sequence_annotations = []
    current_room_calib = {}
    for root, dirs, files in os.walk(dataset_path):
        folder = os.path.basename(root)
        if "calib" in dirs:
            # Now we are in a room folder, so we can get the
            # extrinsics for the current room and annotate the sequences
            print(f"[*] Processing sequences from {folder}")
            current_room_calib = get_calibration_files(root)
            dirs.pop(dirs.index("calib"))

        elif "capture0" in dirs:
            # We're now in a participant sequence folder
            print(f"\t[*] Processing participant '{folder}'")
            sequence_annotations = annotate_participant(
                root, humor_docker_script, current_room_calib
            )
            # os.walk is depth-first, so we should have all the sequence annotations
            print("\t\t-> Merging sequences...")
            synced_filenames = []
            with open(os.path.join(root, "synced_filenames.txt"), "r") as file:
                for line in file:
                    synced_filenames.append(line.strip().split(";"))
            synced_filenames = synced_filenames[1:]
            final_seq_annotations = synchronize_annotations(
                sequence_annotations, synced_filenames, root
            )
            print(final_seq_annotations)
            # TODO: Save the annotations to a file?
            dirs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HuMoR-based annotation pipeline.")
    parser.add_argument("dataset_path", help="Path to the dataset to annotate.")
    parser.add_argument(
        "--humor-script",
        help="Path to the HuMoR script that will run the inference.",
        required=True,
    )
    args = parser.parse_args()
    main(args.dataset_path, args.humor_script)
