#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Preprocess the RGB view of a sequence, such that it is decomposed into video sequences that are
guaranteed to contain a subject or not, from start to finish.
"""

import argparse
import os
from typing import List, Optional

import cv2
import numpy as np
from scipy.special import kl_div as kl_div
from tqdm import tqdm


def compute_body_coverage(seg_mask, iuv_map, debug=False):
    part_index = {
        "torso": [1, 2],
        "upper_right_arm": [16, 18],
        "lower_right_arm": [3, 20, 22],
        "upper_left_arm": [15, 17],
        "lower_left_arm": [4, 19, 21],
        "right_leg": [5, 7, 9, 11, 13],
        "left_leg": [6, 8, 10, 12, 14],
        "head": [23, 24],
    }  # https://github.com/facebookresearch/DensePose/issues/64

    # Extract the pixels that correspond to the person's body in the segmentation mask
    body_pixels = np.where(
        seg_mask == 255, 1, 0
    )  # 255 is the class index for the person class

    # Create a dictionary to store the surface area of each body part
    part_surface_areas = {}

    # Compute the total surface area of the person in pixels
    total_surface_area = np.sum(body_pixels)
    if total_surface_area == 0:
        return 0

    for part, indices in part_index.items():
        # Extract the pixels of the current body part in the IUV map
        body_part_pixels = np.zeros_like(seg_mask).astype(np.uint8)
        for i in indices:
            body_part_pixels += np.where(iuv_map[:, :, 0] == i, 1, 0).astype(np.uint8)

        # Compute the intersection of the body part pixels and the body pixels
        intersection = body_part_pixels * body_pixels

        # Compute the weighted surface area of the body part
        part_surface_areas[part] = max(50, intersection.sum()) / body_pixels.sum()
        if debug:
            print("{part}: {part_surface_areas[part]}px".format(part=part))

    ideal_distribution = {
        "torso": 0.4,
        "head": 0.2,
        "left_leg": 0.1,
        "right_leg": 0.1,
        "upper_left_arm": 0.05,
        "upper_right_arm": 0.05,
        "lower_left_arm": 0.05,
        "lower_right_arm": 0.05,
    }
    computed_distribution = [part_surface_areas[part] for part in ideal_distribution]
    div = sum(kl_div(list(ideal_distribution.values()), computed_distribution))
    if debug:
        print(ideal_distribution, computed_distribution)
        print("KL divergence: {div}".format(div=div))

    # Move to the [0,1] range via a sigmoid function
    diversity_metric = 1 / (1 + np.exp(div))
    if debug:
        print("diversity: {diversity_metric}".format(diversity_metric=diversity_metric))
    return diversity_metric


def save_subsequence(
    subsequence: List, subsequence_idx: int, contains_person: bool, output_path: str
):
    """Save a sub-sequence.

    Args:
        subsequence (list): List of file names in the sub-sequence.
        subsequence_idx (int): Index of the sub-sequence.
        output_path (str): Path to the output directory.
    """
    print(
        "[*] Saving sub-sequence {subsequence_idx}...".format(
            subsequence_idx=subsequence_idx
        )
    )
    # Create the output directory for the sub-sequence
    subsequence_path = os.path.join(
        output_path,
        str(subsequence_idx) + ("_person" if contains_person else "_no_person"),
    )
    os.makedirs(subsequence_path, exist_ok=True)
    # Copy the frames of the sub-sequence to the output directory
    for frame_path in subsequence:
        if not os.path.exists(
            os.path.join(subsequence_path, os.path.basename(frame_path))
        ):
            os.symlink(
                frame_path, os.path.join(subsequence_path, os.path.basename(frame_path))
            )


def main(
    sequence_path: str,
    output_path: str,
    masks_path: Optional[str] = None,
    iuvs_path: Optional[str] = None,
    threshold: float = 0.24,
    debug: bool = False,
):
    """Preprocess a sequence.

    Args:
        sequence_path (str): Path to the sequence to preprocess (root containing the 'rgb' folder).
        masks_path (str): Path to the masks of the sequence.
        output_path (str): Path to the output directory.
        threshold (float, optional): Threshold to consider a pixel as part of the subject. Defaults to 0.5.
    """
    MIN_FRAMES_FOR_PERSON = 30
    # Load all image absolute file paths in the sequence which follows this tree structure:
    # <sequence_path>/rgb/<frame_id>.jpg
    rgb_path = os.path.join(sequence_path, "rgb")
    rgb_files = sorted([os.path.join(rgb_path, f) for f in os.listdir(rgb_path)])
    if not masks_path:
        masks_path = os.path.join(sequence_path, "rgb_dp2_mask")
    # Load all mask file names in the sequence which follows this tree structure:
    # <masks_path>/<frame_id>.jpg
    mask_files = sorted(os.listdir(masks_path))

    if not iuvs_path:
        iuvs_path = os.path.join(sequence_path, "rgb_dp2_iuv")
    # Load all IUV file names in the sequence which follows this tree structure:
    # <iuvs_path>/<frame_id>.jpg
    iuv_files = sorted(os.listdir(iuvs_path))

    # Remove all entries from mask_files and ivu_files if the corresponding image is not in the
    # sequence of rgb images:
    mask_files = [
        mask_file
        for mask_file in mask_files
        if os.path.join(sequence_path, "rgb", mask_file.split(".")[0] + ".jpg")
        in rgb_files
    ]
    iuv_files = [
        iuv_file
        for iuv_file in iuv_files
        if os.path.join(sequence_path, "rgb", iuv_file.split(".")[0] + ".jpg")
        in rgb_files
    ]

    # Make sure there are no duplicates in the list of rgb files
    assert len(rgb_files) == len(set(rgb_files))
    assert len(mask_files) == len(set(mask_files))
    assert len(iuv_files) == len(set(iuv_files))
    assert len(mask_files) == len(rgb_files)
    assert len(iuv_files) == len(rgb_files)

    # Go through the sequence and the masks, and split the sequence into sub-sequences: assume the
    # subject is initially not visible. When the subject is visible, start a new sub-sequence. When
    # the subject is not visible, end the current sub-sequence. When the end of the sequence is
    # reached, save the last sub-sequence. We want to add the frame to the sub-sequence in all
    # cases, and only save the sub-sequence when the subject visibility changes. This way, we have
    # alternative sub-sequences that are guaranteed to contain the subject or not.
    current_subsequence, subsequence_idx = [], 0
    # Load the first mask and first IUV to see if the subject is visible
    first_mask = cv2.imread(
        os.path.join(masks_path, mask_files[0]), cv2.IMREAD_GRAYSCALE
    )
    first_iuv = cv2.imread(os.path.join(iuvs_path, iuv_files[0]))
    body_part_coverage = compute_body_coverage(first_mask, first_iuv)
    contains_person: bool = body_part_coverage > threshold
    median_filter_size = 5
    median_filter = [body_part_coverage] * median_filter_size
    for rgb_file, mask_file, iuv_file in tqdm(
        zip(rgb_files, mask_files, iuv_files), total=len(rgb_files)
    ):
        mask = cv2.imread(os.path.join(masks_path, mask_file), cv2.IMREAD_GRAYSCALE)
        iuv = cv2.imread(os.path.join(iuvs_path, iuv_file))
        body_part_coverage = compute_body_coverage(mask, iuv, debug)
        median_filter.pop(0)
        median_filter.append(body_part_coverage)
        median = np.median(median_filter)
        # If the person area is above the threshold, the subject is visible
        if median > threshold:
            # if body_part_coverage > threshold:
            # If the subject was not visible in the previous frame, start a new sub-sequence
            if not contains_person:
                assert len(current_subsequence) == len(set(current_subsequence))
                save_subsequence(
                    current_subsequence, subsequence_idx, contains_person, output_path
                )
                contains_person = True
                current_subsequence = []
                subsequence_idx += 1
        elif contains_person:
            # If the subject was only visible for less than MIN_FRAMES_FOR_PERSON though,
            # reject the sequence and consider the subject not visible.
            contains_person = len(current_subsequence) >= MIN_FRAMES_FOR_PERSON
            save_subsequence(
                current_subsequence, subsequence_idx, contains_person, output_path
            )
            contains_person = False
            current_subsequence = []
            subsequence_idx += 1
        # Add the frame to the sub-sequence
        current_subsequence.append(rgb_file)
    # Save the last sub-sequence if there's one remaining
    if contains_person:
        contains_person = len(current_subsequence) >= MIN_FRAMES_FOR_PERSON
    print("[*] Saving all sub-sequences...")
    save_subsequence(current_subsequence, subsequence_idx, contains_person, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a sequence.")
    parser.add_argument(
        "sequence",
        type=str,
        help="The folder containing the sequence to preprocess (containing a directory with RGB files).",
    )
    parser.add_argument(
        "--masks",
        type=str,
        required=False,
        help="The segmentation masks of the sequence, usually inside the sequence folder else explicitly linked.",
    )
    parser.add_argument(
        "--iuvs",
        type=str,
        required=False,
        help="The IUV maps of the sequence, usually inside the sequence folder else explicitly linked.",
    )
    parser.add_argument(
        "--output", "-o", type=str, help="The output directory.", required=True
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Minimum threshold for person visibility.",
        default=0.24,
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    args = parser.parse_args()
    main(
        args.sequence,
        args.output,
        masks_path=args.masks,
        iuvs_path=args.iuvs,
        threshold=args.threshold,
        debug=args.debug,
    )
