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
from tqdm import tqdm


def save_subsequence(
    subsequence: List, subsequence_idx: int, contains_person: bool, output_path: str
):
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


def main(sequence_path: str, masks_path: str, output_path: str, threshold: float = 0.5):
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

    # Remove all entries from mask_files if the corresponding image is not in the sequence of rgb
    # images:
    mask_files = [
        mask_file
        for mask_file in mask_files
        if os.path.join(sequence_path, "rgb", mask_file.split(".")[0] + ".jpg")
        in rgb_files
    ]

    # Make sure there are no duplicates in the list of rgb files
    assert len(rgb_files) == len(set(rgb_files))
    assert len(mask_files) == len(set(mask_files))
    assert len(mask_files) == len(rgb_files)

    # Go through the sequence and the masks, and split the sequence into sub-sequences: assume the
    # subject is initially not visible. When the subject is visible, start a new sub-sequence. When
    # the subject is not visible, end the current sub-sequence. When the end of the sequence is
    # reached, save the last sub-sequence. We want to add the frame to the sub-sequence in all
    # cases, and only save the sub-sequence when the subject visibility changes. This way, we have
    # alternative sub-sequences that are guaranteed to contain the subject or not.
    current_subsequence, subsequence_idx = [], 0
    # Load the first mask to see if the subject is visible
    first_mask = cv2.imread(os.path.join(masks_path, mask_files[0]))
    contains_person = (
        np.sum(first_mask > 0) / (first_mask.shape[0] * first_mask.shape[1]) > threshold
    )
    for rgb_file, mask_file in tqdm(zip(rgb_files, mask_files), total=len(rgb_files)):
        mask = cv2.imread(os.path.join(masks_path, mask_file), cv2.IMREAD_GRAYSCALE)
        person_area = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
        # If the person area is above the threshold, the subject is visible
        if person_area > threshold:
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
parser.add_argument("--output", type=str, help="The output directory.", required=True)
parser.add_argument(
    "--threshold",
    type=float,
    help="The minimum person area to use for the segmentation masks.",
    default=0.02,
)
args = parser.parse_args()
main(args.sequence, args.masks, args.output, args.threshold)
