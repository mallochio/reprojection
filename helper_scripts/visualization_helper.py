#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   rename_metafiles.py
@Time    :   2023/10/26 09:00:23
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Scripts to work with the visualizations of the results
'''

import os
import cv2
import pandas as pd
from tqdm import tqdm


root_dir = '/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/2022-08-11/at-a01/living-room/a01/capture0/out_capture0/results_out/'
synced_filename_file = "sftp://sid@bestium:22/home/sid/Projects/OmniScience/dataset/session-recordings/2024-01-12/at-unis/lab/sid/synced_filenames_full.txt"
kinect_dir = "sftp://sid@bestium:22/home/sid/Projects/OmniScience/dataset/session-recordings/2024-01-12/at-unis/lab/sid/capture1/rgb"
omni_dir = "sftp://sid@bestium:22/home/sid/Projects/OmniScience/dataset/session-recordings/2024-01-12/at-unis/lab/sid/omni"
kinect_dst = "/home/sid/Projects/OmniScience/dataset/session-recordings/a10/trimmed_experiment/trimmed/capture0"
omni_dst = "/home/sid/Projects/OmniScience/dataset/session-recordings/a10/trimmed_experiment/trimmed/omni"


def rename_files():
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('meta.txt'):
                with open(os.path.join(root, file), "w") as f:
                    f.write("optim_bm /home/sid/Projects/OmniScience/other/humor/body_models/smplh/male/model.npz")
                    # f.write("optim_bm /openpose/data/other/humor/body_models/smplh/male/model.npz")
                    f.write("\n")
                    f.write("gt_bm /home/sid/Projects/OmniScience/other/humor/body_models/smplh/male/model.npz")
                    # f.write("gt_bm /openpose/data/other/humor/body_models/smplh/male/model.npz")


def stitch_videos(vid_file_name):
    video_files = []
    for root, dirs, files in os.walk(f"{os.path.dirname(root_dir)}"):
        print(root)
        for file in files:
            if file.endswith(vid_file_name):
                video_files.append(os.path.join(root, file))
    output_dir = os.path.dirname(root_dir)
    video_files.sort()
    # stitch videos together to form a single video from the list of videos using ffmpeg
    print("Writing to join_video.txt")
    with open("join_video.txt", "w") as f:
        for video_file in video_files:
            f.write(f"file '{video_file}'\n")

    print("Finished writing")
    os.system(f"sudo ffmpeg -f concat -safe 0 -i join_video.txt -c copy {output_dir}/{vid_file_name}")
    os.system("rm join_video.txt")


def get_kinect_filenames(kinect_dir, delay=4501, end_time =4501+3600):
    files = sorted(os.listdir(kinect_dir))
    start_filename = files[delay] # Move these if not synced correctly
    end_filename = files[end_time]
    return start_filename, end_filename, files


def collate_video(image_list):
    image_list = [os.path.join(kinect_dir, i) for i in image_list]

    # Collate all trimmed file images into a video using cv2
    frame = cv2.imread(image_list[0])
    height, width, channels = frame.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use the correct codec
    out = cv2.VideoWriter("collated_trimmed.mp4", fourcc, 30, (width, height))

    print("[*] Writing video...")
    for image in tqdm(image_list):
        frame = cv2.imread(image)
        out.write(frame)  # Write out frame to video

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()



def associate_omni(synced_filenames):
    start_file, end_file, files = get_kinect_filenames(kinect_dir)
    df = pd.read_csv(synced_filenames, header=0, sep=";")
    df = df.reset_index()
    kinect_lookup = {}
    mask_index = [0] * 3600
    trimmed_files = files[4501:8101]

    collate_video(trimmed_files)

    ## Commented from here on -
    #     if val["capture0"] in trimmed_files:
    #         kinect_lookup[val["capture0"]] = val["omni"]
    #         mask_index[trimmed_files.index(val["capture0"])] = 1
    # return kinect_lookup, mask_index

def copy_dict_files(lookup):
    # create destination directories if they don't exist
    if not os.path.exists(kinect_dst):
        os.makedirs(kinect_dst)
    if not os.path.exists(omni_dst):
        os.makedirs(omni_dst)
    for kinect, omni in tqdm(lookup.items()):
        os.system(f"cp {kinect_dir}/{kinect} {kinect_dst}/{kinect}")
        os.system(f"cp {omni_dir}/{omni} {omni_dst}/{omni}")
        

if __name__ == "__main__":
    rename_files()
    # stitch_videos("comp_og_final_pred.mp4")
    # lookup, mask_index = associate_omni(synced_filename_file)
    # # print(len(lookup))
    # # copy_dict_files(lookup)
    # # # write mask index so it can be loaded as an array in numpy
    # # with open("mask_index.txt", "w") as f:
    # #     f.write(str(mask_index))