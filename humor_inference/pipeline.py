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
OUTPUT_FOLDER = "preprocessed"

PREPROCESS_THRESHOLD = 0.24


def annotate_capture(
    capture_path: str,
    humor_docker_script: str,
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
    print("\t-> Preprocessing...")
    # Redirect the prints to a log file
    with open(os.path.join(capture_path, "preprocess.log"), "w") as f:
        with redirect_stdout(f):
            preprocess(
                capture_path,
                os.path.join(capture_path, OUTPUT_FOLDER),
                debug=False,
                threshold=PREPROCESS_THRESHOLD,
            )

    sequence_meshes = {}
    # Compile each "person" subsequence into a video file and process it
    for root, _, _ in os.walk(os.path.join(capture_path, OUTPUT_FOLDER)):
        seq_name = os.path.basename(root)
        if "no_person" in seq_name or "person" not in seq_name:
            continue
        print(f"\t-> Processing {seq_name}...")
        # TODO: Maybe patch HuMoR so that it loads the images instead? The only problem is that
        # HuMoR works for 30hz videos, so we would need to interpolate or duplicate the frames
        # maybe? With these parameters, FFMPEG just builds a 30hz video from the images but since
        # we recorded at ~15hz, the video looks sped up. That might be the best way to deal with
        # the disparity because the movements remain smooth, just faster.
        output_vid_file = os.path.join(capture_path, OUTPUT_FOLDER, f"{seq_name}.mp4")
        if not os.path.exists(output_vid_file):
            # print(f"\t\t-> Compiling {seq_name} into video file {output_vid_file}...")
            os.system(
                f"ffmpeg -framerate 30 -pattern_type glob -i '{root}/*.jpg' -c:v"
                + f" libx264 -r 30 {output_vid_file} >"
                + f" {os.path.join(capture_path, OUTPUT_FOLDER, 'ffmpeg.log')} 2>&1"
            )
        with open(os.path.join(capture_path, "humor.log"), "w") as f:
            with redirect_stdout(f):
                # TODO: Check if HuMoR was already ran on this subsequence and skip it if so
                humor_was_run = False
                # Run the HuMoR Docker script
                # TODO: Write a bash script that will run the docker image and the inference script.
                # For a first single-threaded PoC, the input video file and output older should
                # probably be fixed.
                humor_output_path = os.path.join(capture_path, OUTPUT_FOLDER)
                if not humor_was_run:
                    # TODO: define this
                    os.system(
                        f"bash {humor_docker_script} {capture_path} {humor_output_path}"
                    )
                timestamped_meshes = reproject(
                    basecam_to_world_pth,
                    world_to_destcam_pth,
                    humor_output_path,
                    capture_path,
                )  # key=timestamp of fitted image (base cam), value=reprojected fitted mesh (dest cam)
                assert (
                    timestamped_meshes is not None and type(timestamped_meshes) == dict
                ), "reproject_humor_sequence.main() did not return a dict"
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
) -> Dict[int, trimesh.Trimesh]:
    raise NotImplementedError


def main(dataset_path: str, humor_docker_script: str):
    """Main function of the pipeline.

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
                            capture0
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
    current_room_calib = {}
    sequence_annotations = []
    for root, dirs, files in os.walk(dataset_path):
        folder = os.path.basename(root)
        if folder == "calib" or folder == "Calibration":
            current_room_calib = {
                os.path.basename(fpath).split(".")[0]: fpath
                for fpath in files
                if fpath.endswith(".pkl")
            }
            dirs.clear()
        if folder.startswith("capture") and RGB_FOLDER in dirs:
            kinect_id = folder.split("capture")[1]
            sequence_annotations.append(
                annotate_capture(
                    root,
                    humor_docker_script,
                    current_room_calib[f"k{kinect_id}-world"],
                    current_room_calib[f"world-omni"]
                )
            )
        if "omni" in dirs and len(sequence_annotations) > 0:
            # TODO: Detect when we're leaving a sequence and synchronize the annotations
            final_seq_annotations = synchronize_annotations(sequence_annotations)
            # TODO: Save the annotations to a file?


parser = argparse.ArgumentParser(description="HuMoR-based annotation pipeline.")
parser.add_argument("dataset_path", help="Path to the dataset to annotate.")
parser.add_argument(
    "--humor-docker-script",
    help="Path to the HuMoR Docker script that will run the docker image and the inference script.",
)
args = parser.parse_args()
main(args.dataset_path, args.humor_docker_script)
