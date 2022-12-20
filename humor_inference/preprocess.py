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
from typing import List

import cv2
import numpy as np
from scipy.stats import mode as scipy_mode
from tqdm import tqdm


def compute_body_parts_diversity_bad(seg_mask, iuv_map):
    # Extract the body part pixels from the IUV map
    body_part_pixels = np.zeros_like(seg_mask)
    for i in range(1, 26):  # Skip the background index (0)
        body_part_pixels += np.where(iuv_map[:, :, 0] == i, 1, 0).astype(np.uint8)

    # Extract the pixels that correspond to the person's body in the segmentation mask
    body_pixels = np.where(seg_mask == 255, 1, 0).astype(int)  # 255 is the class index for the person class

    # Compute the intersection of the body part pixels and the body pixels
    intersection = body_part_pixels * body_pixels

    # Compute the sum of the intersection mask
    total_part_pixels = np.sum(intersection)

    # Compute the unique body part indices in the intersection mask
    unique_part_indices = np.unique(iuv_map[:, :, 0][intersection == 1])

    # Compute the diversity metric
    diversity = len(unique_part_indices) / total_part_pixels

    return diversity


def compute_body_coverage(seg_mask, iuv_map):
    # Create a dictionary to store the coverage of each body part
    body_part_coverage = {}

    # Extract the pixels that correspond to the person's body in the segmentation mask
    body_pixels = (seg_mask == 255).astype(
        int
    )  # 1 is the class index for the person class

    n_body_pixels = np.sum(body_pixels)
    total_part_pixels = 0
    if n_body_pixels == 0:
        # return 0, 0, 0, 0
        return 1000
    # Loop over each body part
    for i in range(25):
        # Skip the background index
        if i == 0:
            continue
        # Extract the pixels of the current body part in the IUV map, which has shape (H, W, 3) and
        # where the first channel is the body part index. The other two channels are U and V.
        body_part_pixels = np.where(iuv_map[:,:,0] == i, 1, 0)
        # Compute the intersection of the two masks
        intersection = body_part_pixels * body_pixels
        total_part_pixels += intersection.sum()

        # Compute the ratio of the intersection over the total body area
        body_part_coverage[i] = np.sum(intersection) / n_body_pixels

    # The sum of the body part pixels should equal the number of body pixels
    assert total_part_pixels == n_body_pixels
    # print(body_part_coverage)
    # Display the iuv map with opencv
    # cv2.imshow("iuv", iuv_map)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print(np.prod([np.maximum(0.0001, v) for k, v in body_part_coverage.items()]))
    # Multiply the values of body_part_coverage together, if they are not 0:
    body_part_coverage = -np.log(np.prod([np.maximum(0.0001, v) for k, v in
                                          body_part_coverage.items()]))
    return body_part_coverage
    # Compute the median, mode, and standard deviation of the body part coverage values
    mean = np.mean(list(body_part_coverage.values()))
    median = np.median(list(body_part_coverage.values()))
    mode, _ = scipy_mode(list(body_part_coverage.values()))
    std = np.std(list(body_part_coverage.values()))
    return mean, median, mode.item(), std


def compute_body_parts_diversity(seg_mask, iuv_map):
    # Define the weights for each body part
    weights = {
        "torso": 0.7,
        "head": 0.35,
        "upper_left_arm": 0.6,
        "lower_left_arm": 0.6,
        "upper_right_arm": 0.6,
        "lower_right_arm": 0.6,
        "left_leg": 1,
        "right_leg": 1,
        # "left_hand": 0.2,
        # "right_hand": 0.2,
        # "left_foot": 0.2,
        # "right_foot": 0.2,
    }

    # The part_index dictionary is defined by DensePose as follows:
    part_index = {
        "background": 0,
        "torso": [1, 2],
        "upper_right_arm": [16, 18],
        "lower_right_arm": [3, 20, 22],
        "upper_left_arm": [15, 17],
        "lower_left_arm": [4,  19, 21],
        "right_leg": [5, 7, 9, 11, 13],
        "left_leg": [6, 8, 10, 12, 14],
        "head": [23, 24],
    } # https://github.com/facebookresearch/DensePose/issues/64

    # Extract the pixels that correspond to the person's body in the segmentation mask
    body_pixels = np.where(seg_mask == 255, 1, 0)  # 255 is the class index for the person class

    # Compute the total surface area of the person in pixels
    total_surface_area = np.sum(body_pixels)
    if total_surface_area == 0:
        return 0

    # Compute the weighted surface area of each body part in pixels
    weighted_surface_areas = {}
    for part, weight in weights.items():
        # Extract the pixels of the current body part in the IUV map
        part_pixels = np.zeros_like(seg_mask).astype(np.uint8)
        for idx in part_index[part]:
            part_pixels += np.where(iuv_map[:, :, 0] == idx, 1, 0).astype(np.uint8)
        # print(f"{part}: {part_pixels.sum()}px")

        # Compute the intersection of the body part pixels and the body pixels
        intersection = part_pixels * body_pixels

        # Compute the weighted surface area of the body part
        weighted_surface_areas[part] = weight * np.sum(intersection)

    # Compute the diversity metric as the sum of the weighted surface areas of all body parts divided by the total surface area of the person
    diversity_metric = np.sum(list(weighted_surface_areas.values())) / total_surface_area

    return diversity_metric



def save_subsequence(
    subsequence: List, subsequence_idx: int, contains_person: bool, output_path: str):
    """Save a sub-sequence.

    Args:
        subsequence (list): List of file names in the sub-sequence.
        subsequence_idx (int): Index of the sub-sequence.
        output_path (str): Path to the output directory.
    """
    print("[*] Saving sub-sequence {}...".format(subsequence_idx))
    # Create the output directory for the sub-sequence
    subsequence_path = os.path.join(
        output_path,
        str(subsequence_idx) + ("_person" if contains_person else "_no_person"),
    )
    os.makedirs(subsequence_path, exist_ok=True)
    # Copy the frames of the sub-sequence to the output directory
    for frame_path in subsequence:
        os.symlink(
            frame_path, os.path.join(subsequence_path, os.path.basename(frame_path))
        )


def main(
    sequence_path: str,
    masks_path: str,
    iuvs_path: str,
    output_path: str,
    threshold: float = 0.5,
):
    """Preprocess a sequence.

    Args:
        sequence_path (str): Path to the sequence to preprocess.
        masks_path (str): Path to the masks of the sequence.
        output_path (str): Path to the output directory.
        threshold (float, optional): Threshold to consider a pixel as part of the subject. Defaults to 0.5.
    """
    # Load all image absolute file paths in the sequence which follows this tree structure:
    # <sequence_path>/rgb/<frame_id>.jpg
    rgb_files = sorted(
        [
            os.path.join(sequence_path, "rgb", f)
            for f in os.listdir(os.path.join(sequence_path, "rgb"))
        ]
    )
    # Load all mask file names in the sequence which follows this tree structure:
    # <masks_path>/<frame_id>.jpg
    mask_files = sorted(os.listdir(masks_path))


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
    first_mask = cv2.imread(os.path.join(masks_path, mask_files[0]), cv2.IMREAD_GRAYSCALE)
    first_iuv = cv2.imread(os.path.join(iuvs_path, iuv_files[0]))
    # contains_person = (
        # np.sum(first_mask > 0) / (first_mask.shape[0] * first_mask.shape[1]) > threshold
    # )
    # body_part_coverage = compute_body_coverage(first_mask, first_iuv)
    body_part_coverage = compute_body_parts_diversity(first_mask, first_iuv)
    contains_person: bool = body_part_coverage > threshold
    # mean, median, mode, std = compute_body_coverage(first_mask, first_iuv)
    # contains_person: bool = mean > threshold
    median_filter_size = 5
    median_filter = [body_part_coverage] * median_filter_size
    for rgb_file, mask_file, iuv in tqdm(zip(rgb_files, mask_files, iuv_files), total=len(rgb_files)):
        mask = cv2.imread(os.path.join(masks_path, mask_file), cv2.IMREAD_GRAYSCALE)
        iuv = cv2.imread(os.path.join(iuvs_path, iuv))
        # person_area = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
        # body_part_coverage = compute_body_coverage(mask, iuv)
        body_part_coverage = compute_body_parts_diversity(mask, iuv)
        median_filter.pop(0)
        median_filter.append(body_part_coverage)
        median = np.median(median_filter)
        print(body_part_coverage, median)
        # print(body_part_coverage, median)
        # mean, median, mode, std = compute_body_coverage(mask, iuv)
        # print(median, mode, std)
        # print(mean, median, mode, std)
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
            # Add the frame to the sub-sequence
            current_subsequence.append(rgb_file)
        else:
            # If the subject was visible in the previous frame, save the sub-sequence
            if contains_person:
                save_subsequence(
                    current_subsequence, subsequence_idx, contains_person, output_path
                )
                contains_person = False
                current_subsequence = []
                subsequence_idx += 1
            # Add the frame to the sub-sequence
            current_subsequence.append(rgb_file)


parser = argparse.ArgumentParser(description="Preprocess a sequence.")
parser.add_argument("sequence", type=str, help="The RGB sequence to preprocess.")
parser.add_argument("masks", type=str, help="The segmentation masks of the sequence.")
parser.add_argument("iuvs", type=str, help="The IUV maps of the sequence.")
parser.add_argument("--output", type=str, help="The output directory.", required=True)
parser.add_argument(
    "--threshold",
    type=float,
    help="The minimum person area to use for the segmentation masks.",
    default=0.02,
)
args = parser.parse_args()
main(args.sequence, args.masks, args.iuvs, args.output, args.threshold)
