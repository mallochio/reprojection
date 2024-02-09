#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   trim_videos.py
@Time    :   2023/10/25 10:21:34
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Script to select the contents of the processed video that are within a short time interval, and also have predictions for each of the stages of the pipeline. (For testing humor)
'''

import os
from tqdm import tqdm

root_dir = "/openpose/data/a10/copies/out_capture2/rgb_preprocess"
out_dir = os.path.join(root_dir, "trimmed", "rgb_preprocess")
n_minutes = 4
delay_till_first_frame = 5 # minutes


def rename_files(out_dir):
    # Renames files so they start from 000001
    raw_frames_dir = os.path.join(out_dir, "raw_frames")
    op_frames_dir = os.path.join(out_dir, "op_frames")
    op_keypoints_dir = os.path.join(out_dir, "op_keypoints")
    masks_dir = os.path.join(out_dir, "masks")

    raw_frames = sorted(os.listdir(raw_frames_dir))
    op_frames = sorted(os.listdir(op_frames_dir))
    op_keypoints = sorted(os.listdir(op_keypoints_dir))
    masks = sorted(os.listdir(masks_dir))

    for i in tqdm(range(len(raw_frames))):
        old_name = os.path.join(raw_frames_dir, raw_frames[i])
        new_name = os.path.join(raw_frames_dir, "{:06d}.png".format(i+1))
        os.rename(old_name, new_name)

        old_name = os.path.join(op_frames_dir, op_frames[i])
        new_name = os.path.join(op_frames_dir, "{:06d}_rendered.png".format(i+1))
        os.rename(old_name, new_name)

        old_name = os.path.join(op_keypoints_dir, op_keypoints[i])
        new_name = os.path.join(op_keypoints_dir, "{:06d}_keypoints.json".format(i+1))
        os.rename(old_name, new_name)

        old_name = os.path.join(masks_dir, masks[i])
        new_name = os.path.join(masks_dir, "{:06d}.png".format(i+1))
        os.rename(old_name, new_name)        


def select_frames(root_dir, n_minutes, delay_till_first_frame):
    selected_frames = []
    raw_frames_dir = os.path.join(root_dir, "raw_frames")
    op_frames_dir = os.path.join(root_dir, "op_frames")
    op_keypoints_dir = os.path.join(root_dir, "op_keypoints")
    masks_dir = os.path.join(root_dir, "masks")
    raw_frames = sorted(os.listdir(raw_frames_dir))
    start_frame_ix = delay_till_first_frame * 15 * 60
    start_frame = raw_frames[start_frame_ix]    

    n_frames = n_minutes * 15 * 60
    for i in raw_frames[start_frame_ix:]:
        base_name = os.path.splitext(os.path.basename(i))[0]
        # see if corresponding files exist in other folders
        op_frame = os.path.join(op_frames_dir, base_name + "_rendered.png")
        op_keypoint = os.path.join(op_keypoints_dir, base_name + "_keypoints.json")
        mask = os.path.join(masks_dir, base_name + ".png") 
        flag = True
        for j in [op_frame, op_keypoint, mask]:
            if not os.path.exists(j):
                flag = False
                break
        if flag:
            selected_frames.append(i)
            
        else:
            print("Skipping frame: ", i)
            continue

        n_frames -= 1
        if n_frames <= 0:
            break

    return selected_frames


def save_frames_to_out_dir(frames, out_dir):
    for i in tqdm(frames):
        base_name = os.path.splitext(os.path.basename(i))[0]
        raw_frame = os.path.join(root_dir, "raw_frames", i)
        op_frame = os.path.join(root_dir, "op_frames", base_name + "_rendered.png")
        op_keypoint = os.path.join(root_dir, "op_keypoints", f"{base_name}_keypoints.json")
        mask = os.path.join(root_dir, "masks", f"{base_name}.png") 


        out_raw_frame = os.path.join(out_dir, "raw_frames", i)
        out_op_frame = os.path.join(out_dir, "op_frames", f"{base_name}_rendered.png")
        out_op_keypoint = os.path.join(out_dir, "op_keypoints", f"{base_name}_keypoints.json")
        out_mask = os.path.join(out_dir, "masks", f"{base_name}.png") 


        os.makedirs(os.path.dirname(out_raw_frame), exist_ok=True)
        os.makedirs(os.path.dirname(out_op_frame), exist_ok=True)
        os.makedirs(os.path.dirname(out_op_keypoint), exist_ok=True)
        os.makedirs(os.path.dirname(out_mask), exist_ok=True)

        os.system(f"cp {raw_frame} {out_raw_frame}")
        os.system(f"cp {op_frame} {out_op_frame}")
        os.system(f"cp {op_keypoint} {out_op_keypoint}")
        os.system(f"cp {mask} {out_mask}")

        # dump the frame names to a file
        with open(os.path.join(out_dir, "frame_names_original.txt"), "a") as f:
            f.write(i + "\n")


def main():
    print("Selecting frames...")
    frames = select_frames(root_dir, n_minutes, delay_till_first_frame)
    print("Saving frames to out_dir...")
    save_frames_to_out_dir(frames, out_dir)
    print("Renaming files to start from 000001")
    rename_files(out_dir)

if __name__ == "__main__":
    main()