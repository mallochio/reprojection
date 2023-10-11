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
import pickle
import subprocess
import sys
import numpy as np
import trimesh
from contextlib import redirect_stdout
from typing import Dict, List, Tuple
from reproject_humor_sequence import main as reproject
from preprocess import main as preprocess
import cv2
import glob
import os
from tqdm import tqdm

RGB_FOLDER = "rgb"
IR_FOLDER = "ir"
DEPTH_FOLDER = "depth"
IUV_FOLDER = "rgb_dp2_iuv"
MASK_FOLDER = "rgb_dp2_mask"
OUTPUT_FOLDER = "humor_output"
PREPROCESS_FOLDER = "preprocessed"
SYNC_FILENAME = "synced_filenames.txt"
PREPROCESS_THRESHOLD = 0.26
PREPROCESS_MIN_FRAMES_PER_PERSON = 30

CAM_INTRINSICS_PATH = {
    "k0": "../calibration/intrinsics/k0_rgb_calib.json",
    "k1": "../calibration/intrinsics/k1_rgb_calib.json",
    "omni": "../calibration/intrinsics/omni_c.json",
}


def make_video(roottt, output_vid_file):
    img_array = []
    capt = roottt.split('/')[-3]
    for filename in sorted(glob.glob(f'{roottt}/*.jpg')):
        filename = os.path.join(f'/openpose/data/dataset/session-recordings/test/2022-10-07/at-paus/bedroom/sid/{capt}/rgb',
                                os.path.basename(filename))
        img = cv2.imread(filename)
        img_array.append(img)

    height, width, _ = img.shape
    size = (width, height)

    out = cv2.VideoWriter(output_vid_file, cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
    print(f"Writing video to {output_vid_file}")
    for i in tqdm(range(len(img_array))):
        out.write(img_array[i])
    out.release()

def annotate_capture(
        capture_path: str,
        humor_docker_script: str,
        base_cam_intrinsics_path: str,
        basecam_to_world_pth: str,
        world_to_destcam_pth: str,
        keep_dirty: bool = False,
        verbose: bool = False,
) -> Dict[int, trimesh.Trimesh]:
    """Go through a capture folder for one participant in one location, annotate each subsequence
    where a person was detected, and reproject the HuMoR output to the other camera. Then, join the
    reprojected mesh sequences into the final annotation for the sequence.
    Args:
        capture_path (str): Path to the capture folder.
        humor_docker_script (str): Path to the HuMoR Docker script that will run the docker image.
        basecam_to_world_pth (str): Path to the origin camera (which recorded the capture) to world transformation matrix.
        world_to_destcam_pth (str): Path to the world to the destination camera (to which we are reprojecting, i.e. the omni cam) transformation matrix.
    Returns:
        dict: key=timestamp of fitted image (base cam), value=reprojected fitted mesh (dest cam)
    """
    if verbose:
        print(f"[*] Annotating capture {capture_path}...")
    if not os.path.isdir(os.path.join(capture_path, PREPROCESS_FOLDER)):
        if verbose:
            print("\t-> Preprocessing...")
        with open(os.path.join(capture_path, "..", SYNC_FILENAME), "r") as file:
            cameras = file.readline().strip().split(";")
            start = file.readline().strip().split(";")
            sequence_start_frame: str = start[
                cameras.index(os.path.basename(capture_path))
            ]
        # Redirect the prints to a log file
        with open(os.path.join(capture_path, "preprocess.log"), "w") as f:
            with redirect_stdout(sys.stdout if verbose else f):
                preprocess(
                    capture_path,
                    os.path.join(capture_path, PREPROCESS_FOLDER),
                    debug=False,
                    threshold=PREPROCESS_THRESHOLD,
                    skip_until_frame=sequence_start_frame,
                    min_frames_per_person=PREPROCESS_MIN_FRAMES_PER_PERSON,
                )
        if verbose:
            print("\t-> Preprocessing done.")
    sequence_meshes = {}
    # Compile each "person" subsequence into a video file and process it
    for root, _, _ in os.walk(os.path.join(capture_path, PREPROCESS_FOLDER)):
        seq_name = os.path.basename(root)
        if "no_person" in seq_name or "person" not in seq_name:
            continue
        if verbose:
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
            if verbose:
                print(
                    f"\t\t-> Compiling {seq_name} into video file {output_vid_file}..."
                )
            # os.system(
            #     f"ffmpeg -framerate 30 -pattern_type glob -i '{root}/*.jpg' -c:v"
            #     + f" libx264 -r 30 -loglevel quiet {output_vid_file}"
            # )
            path_imgs = '/openpose/data/dataset/session-recordings/test/2022-10-07/at-paus/bedroom/sid/capture1/preprocessed/5_person'
            make_video(root, output_vid_file)
            # TODO! read all images in each folder. check the order, with a for loop create video by opencv#

            print(f"\t\t-> Output file {output_vid_file} created.")
        with open(os.path.join(capture_path, "humor.log"), "w") as f:
            with redirect_stdout(sys.stdout if verbose else f):
                humor_output_path = os.path.join(capture_path, OUTPUT_FOLDER, seq_name)
                humor_was_run = os.path.isdir(
                    os.path.join(humor_output_path, "results_out", "final_results")
                )
                # Run the HuMoR Docker script
                # TODO: Parallelize this if possible
                if not humor_was_run:
                    try:
                        os.system(
                            f"bash {humor_docker_script} {output_vid_file} {humor_output_path} {os.path.abspath(base_cam_intrinsics_path)}"
                        )
                    except FileNotFoundError as e:
                        raise Exception("humor failed: ", e)
                    except Exception as e:
                        raise Exception("humor failed: ", e)

                # There's still a possibility that HuMoR fails and stage3_results.npz is missing.
                # In that case, we should go on to the next sequence and maybe mark this one as
                # failed somehow.
                timestamped_meshes = reproject(
                    basecam_to_world_pth,
                    world_to_destcam_pth,
                    humor_output_path,
                    root,
                )  # key=timestamp of fitted image (base cam), value=reprojected fitted mesh (dest cam)
                if verbose:
                    print(
                        f"\t\t-> Meshes for {seq_name}: {len(timestamped_meshes.keys()) if timestamped_meshes is not None else 0}"
                    )
                if not keep_dirty:
                    if verbose:
                        print(f"\t\t-> Deleting {humor_output_path}...")
                    os.system(f"rm -rf {humor_output_path}")
                if timestamped_meshes is None:
                    print(
                        f"\t\t-> HuMoR failed to produce a result for {seq_name}. Skipping..."
                    )
                    continue
                assert (
                        type(timestamped_meshes) == dict
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


def average_meshes(meshes: List[trimesh.Trimesh]) -> trimesh.Trimesh:
    """
    Averages a list of meshes by averaging their vertices.
    Args:
        meshes: A list of meshes to average.
    Returns:
        The averaged mesh.
    """
    vertices = np.array([mesh.vertices for mesh in meshes])
    vertices = np.mean(vertices, axis=0)
    # TODO: Rebuild the faces? How?
    return trimesh.Trimesh(vertices=vertices, faces=meshes[0].faces)


def synchronize_annotations(
    sub_sequences: List[Dict[int, trimesh.Trimesh]],
    synced_filenames_array: List[List[str]],
    synced_camera_names: List[str],
) -> List[Dict[str, Tuple[int, trimesh.Trimesh]]]:
    # TODO: Right now, the meshes are ALL in top-view frame! We wanna change that so we have them
    # for all the cameras.
    """
    We synchronize the annotations from the different sub-sequences of a capture.
    For this, we take the average of the meshes for each of the synced files in the array.
    Args:
        sub_sequences: A list of dictionaries, where each dictionary is a sub-sequence of a capture
            with keys being the timestamps of the frames and values being the meshes for that frame.
        synced_filenames_array: A list of lists of filenames, where each list of filenames contains
            the filenames of the frames that are synced together from the available cameras.
            Example: [kinect0_filename, kinect1_filename, omni_filename] or
            [kinect0_filename, kinect1_filename, kinect2_filename, omni_filename].
        synced_camera_names: A list of strings, where each string is the name of the camera that
            corresponds to the synced filenames in the synced_filenames_array.
            Example: ["kinect0", "kinect1", "omni"] or ["kinect0", "kinect1", "kinect2", "omni"].
    Returns:
        A list of dictionaries, where each dictionary is a synchronized frame with key=camera_name,
        value=(timestamp, mesh).
    """
    assert (
        sum([len(sub_sequence) for sub_sequence in sub_sequences]) > 0
    ), "No meshes to synchronize"
    # Step 1, flatten the subsequences into a single dictionary where the keys are the ordered
    # timestamps of the frames and the values are the meshes for that frame:
    mesh_sequence = {}
    for sub_sequence in sub_sequences:
        for timestamp, mesh in sub_sequence.items():
            # It is still possible that there are multiple meshes for a single timestamp, if more
            # than one camera has a frame for that timestamp. In that case, we average the meshes
            # once more.
            if timestamp in mesh_sequence:
                if isinstance(mesh_sequence[timestamp], list):
                    mesh_sequence[timestamp].append(mesh)
                else:
                    mesh_sequence[timestamp] = [mesh_sequence[timestamp], mesh]
            mesh_sequence[timestamp] = mesh
    # Now go through them once more to average all the meshes that need to be averaged
    for timestamp, mesh in mesh_sequence.items():
        if isinstance(mesh, list):
            mesh_sequence[timestamp] = mesh #average_meshes(mesh)
    mesh_sequence = {
        timestamp: mesh_sequence[timestamp]
        for timestamp in sorted(mesh_sequence.keys())
    }
    # Step 2, pick the meshes that correspond to the synced frames and build the final sequence
    merged_sequence = []
    for frame_sync_files in synced_filenames_array:
        omni_ts = int(frame_sync_files[-1].split(".")[0])
        # 1. Find the frame in mesh_sequence that matches one of the 3 timestamps in
        # frame_sync_files. If none, continue
        match = {}
        for ts, mesh in mesh_sequence.items():
            if any([f"{ts}.jpg" in frame_sync_file for frame_sync_file in frame_sync_files]):
                # What's happening here? We are going through each synchronized set of frames
                # (what's the sync strategy here?) and we are looking for the mesh in the
                # annotation sequence that matches one of the timestamps in the synchronized
                # frame set. Why would there be more than one match?
                # It turns out that we can have, for instance, 2 frames from kinect1 that are
                # in mesh_sequence, but only one of them is in the synchronized frame set so
                # that's a match. However, it's possible (and it happens) that we have the
                # other of the 2 frames from kinect1 in mesh_sequence that matches the omni
                # timestamp in the same synchronized frame set. In that case, we have 2
                # matches. We should pick the one that is closest to the omni timestamp.
                match[ts] = mesh
        if match == {}:
            continue
        # 2. Find the match closest in time to the omni in the frame
        match = min(match.items(), key=lambda x: abs(x[0] - omni_ts))
        # 2. Rename that frame to frame_sync_files[-1] (the omni frame) and add it to the merged
        # sequence
        merged_sequence.append({"omni": (omni_ts, match[1])})
        print(omni_ts, match)

    # for frame_sync_files in synced_filenames_array:
        # frame = {}
        # for i, frame_file in enumerate(frame_sync_files):
            # timestamp = int(frame_file.split(".")[0])
            # if timestamp in mesh_sequence:
                # frame[synced_camera_names[i]] = (timestamp, mesh_sequence[timestamp])
        # if frame != {}:
            # merged_sequence.append(frame)
    num_annotated_frames = sum([len(frame) for frame in merged_sequence])
    assert num_annotated_frames > 0, "Resulting sequence is empty"
    print(
        f"-> Merged sequence: {num_annotated_frames}/{len(synced_filenames_array)} frames annotated!"
    )
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


def annotate_participant(
    root, humor_docker_script, current_room_calib, keep_dirty, verbose
) -> List[Dict[int, trimesh.Trimesh]]:
    """
    Returns a list of viewpoint annotations for each capture in the participant folder. The
    annotations are dictionaries with the timestamp of the image as the key and the
    reprojected mesh as the value.
    """
    multi_view_sequence_annotations = []
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
                keep_dirty=keep_dirty,
                verbose=verbose,
            )
            multi_view_sequence_annotations.append(sequence_meshes)
    return multi_view_sequence_annotations


def main(
        dataset_path: str,
        humor_docker_script: str,
        keep_dirty: bool = False,
        verbose: bool = False,
):
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
    assert os.path.isdir("body_models"), "Please run ln -s /path/to/humor/body_models ."

    sequence_annotations = []
    current_room_calib = {}
    for root, dirs, _ in os.walk(dataset_path):
        folder = os.path.basename(root)
        if "calib" in dirs:
            # Now we are in a room folder, so we can get the
            # extrinsics for the current room and annotate the sequences
            print(f"=== Processing sequences from {folder} ===")
            current_room_calib = get_calibration_files(root)
            dirs.pop(dirs.index("calib"))

        elif "capture0" in dirs:
            # We're now in a participant sequence folder
            print(f"[*] Processing participant '{folder}'")
            sequence_annotations = annotate_participant(
                root, humor_docker_script, current_room_calib, keep_dirty, verbose
            )
            # os.walk is depth-first, so we should have all the sequence annotations
            print("[*] Merging sequences...")
            synced_filenames = []
            with open(os.path.join(root, SYNC_FILENAME), "r") as file:
                for line in file:
                    synced_filenames.append(line.strip().split(";"))
            synced_camera_names = synced_filenames[0]
            synced_filenames = synced_filenames[1:]
            final_seq_annotations = synchronize_annotations(
                sequence_annotations, synced_filenames, synced_camera_names
            )
            print(f"-> Saving annotations as {root}/pose_labels.pkl")
            with open(f"{root}/pose_labels.pkl", "wb") as file:
                pickle.dump(final_seq_annotations, file)
            dirs.clear()
            print("============================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HuMoR-based annotation pipeline.")
    parser.add_argument("--dataset-path", help="Path to the dataset to annotate.", default='../dataset/')
    parser.add_argument(
        "--humor-script",
        help="Path to the HuMoR script that will run the inference.",
        default='./run_humor_script.sh'
    )
    parser.add_argument(
        "--clean-up",
        help="Whether to clean the temporary files created during the annotation"
             + " (HuMoR stuff, etc.). This is to be used in production.",
        action="store_true",
    )
    parser.add_argument("--verbose", "-v", help="Verbose mode.", action="store_true")
    args = parser.parse_args()
    main(
        args.dataset_path,
        args.humor_script,
        keep_dirty=not args.clean_up,
        verbose=args.verbose,
    )
