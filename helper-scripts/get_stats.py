#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   Untitled-1
@Time    :   2023/03/09 20:38:11
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Get dataset statistics
'''

import os
import argparse
import datetime
import seaborn as sns
import numpy as np
from pprint import pprint
import matplotlib as mpl
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": r"\usepackage{amsmath}"
})
import matplotlib.pyplot as plt
# Set font to Times New Roman
plt.switch_backend("pgf")

def count_folders(path, level):
    folder_count = 0
    dirset = set()
    for root, dirs, files in os.walk(path):
        if "calib" in root:
            continue
        _, subsequent = root.split(path)
        depth = len(subsequent.split("/"))
        if depth == level:
            if "calib" in dirs:
                dirs.remove("calib")
            folder_count += len(dirs)
            for i in dirs:
                dirset.add(i)
    return folder_count, dirset


def make_dictionary_of_filecounts(path):
    # Make a dictionary of the number of files in subfolders if it is in the folder named either of "rgb", "ir", "depth", or "omni"
    #Keep a running total of the number of files in the folders
    rgb_count = 0
    ir_count = 0
    depth_count = 0
    omni_count = 0
    for root, dirs, files in os.walk(path):
        # only check the files in the folder if it is not in a subfolder of "calib"
        if  "calib" in root:
            continue
        if "rgb" in dirs:
            rgb_count += len(os.listdir(os.path.join(root, "rgb")))
        if "ir" in dirs:
            ir_count += len(os.listdir(os.path.join(root, "ir")))
        if "depth" in dirs:
            depth_count += len(os.listdir(os.path.join(root, "depth")))
        if "omni" in dirs:
            omni_count += len(os.listdir(os.path.join(root, "omni")))
            
    return {"rgb": rgb_count, "ir": ir_count, "depth": depth_count, "omni": omni_count}


def calculate_duration(folder_path):
    # function to calculate the time difference between the first and last timestamps.
    # All timestamps are in utc but need to be divided by 1000 to get the correct time

    rgb_folder = os.path.join(folder_path, "rgb")
    if not os.path.exists(rgb_folder):
        return None
    timestamps = []
    for filename in os.listdir(rgb_folder):
        if filename.endswith(".jpg"):
            # timestamp_str = filename.split("_")[0]
            timestamp = int(os.path.splitext(filename)[0]) // 1000
            timestamp = datetime.datetime.fromtimestamp(timestamp) 
            timestamps.append(timestamp)
    if len(timestamps) < 2:
        return None
    duration = max(timestamps) - min(timestamps)
    return duration.total_seconds() / 60.0
  
def calculate_time_and_duration(folder_path):
    # function to calculate the time of day and duration of each recording session
    # based on the timestamps of the image files.
    # All timestamps are in utc but need to be divided by 1000 to get the correct time.

    rgb_folder = os.path.join(folder_path, "rgb")
    if not os.path.exists(rgb_folder):
        return None
    
    timestamps = []
    for filename in os.listdir(rgb_folder):
        if filename.endswith(".jpg"):
            timestamp = int(os.path.splitext(filename)[0]) // 1000
            timestamp = datetime.datetime.fromtimestamp(timestamp)
            timestamps.append(timestamp)
    if len(timestamps) < 2:
        return None, None
    
    duration = max(timestamps) - min(timestamps)
    
    # Calculate the median timestamp and convert it to hours, and then into radians
    median_timestamp = timestamps[len(timestamps) // 2]
    theta = (median_timestamp.hour + median_timestamp.minute / 60.0) / 12.0 * np.pi
    duration = duration.total_seconds() / 60.0

    return theta, duration


def traverse_folder(folder_path):
    # recursively traverse the folder structure and calculate the duration of each recording when an rgb folder is found
    durations = []
    thetas = []
    for dirpath, dirs, files in os.walk(folder_path):
        if "rgb" in dirs and "calib" not in dirpath:
            theta, duration = calculate_time_and_duration(dirpath)
            if theta is not None:
                thetas.append(theta)
            if duration is not None:
                durations.append(duration)
    # remove the highest value from the durations list, it is an outlier
    durations.remove(max(durations))
    return durations, thetas

def plot_histogram(durations):
    # Create a seaborn histogram plot of the durations with 50 bins
    sns.histplot(durations, bins=50, color='skyblue', edgecolor='white',
                 linewidth=1.2, alpha=0.8, kde=True)
    
    # Set plot title and axis labels
    plt.title("Duration of RGB Image Capture")
    plt.xlabel("Duration (minutes)")
    plt.ylabel("Frequency")
    
    # Remove top and right spines from the plot
    sns.despine(top=True, right=True)
    
    # Save the plot as a PDF file with a transparent background
    plt.savefig("myplot.pdf", bbox_inches="tight", facecolor="none")



def plot_TOD(theta):


    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='polar')
    
    # Set the theta values to start at 12 o'clock and go clockwise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    
    # Plot the theta values as a histogram
    sns.histplot(theta, bins=24, color='skyblue', edgecolor='white',
                 linewidth=1.2, alpha=0.8, ax=ax)
    
    # Set the xticks and labels to resemble a clock
    hours = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    ax.set_xticks([h/12*2*np.pi - np.pi/2 for h in hours])
    ax.set_xticklabels([f"{h:d}" for h in hours])
    ax.tick_params(pad=8)
    ax.tick_params(axis='x', colors='black', labelsize=14)
    
    # Set plot title and axis labels
    plt.title("Recording Times", fontsize=16)
    
    # Remove top and right spines from the plot
    ax.spines['polar'].set_visible(False)
    ax.set_rticks([])
    
    plt.savefig("TOD.pdf", bbox_inches="tight", facecolor="none")


def traverse_and_plot(top_folder):
    # function to plot the durations
    durations, thetas = traverse_folder(top_folder)
    # Remove None values from durations list
    durations = [d for d in durations if d is not None]
    thetas = [t for t in thetas if t is not None]
    # Remove the longest duration from the data (an outlier)
    max_duration = max(durations)
    durations.remove(max_duration)
    plot_histogram(durations)
    plot_TOD(thetas)


def main(path, level, get_dictionary, make_plots):
    if level:
        pprint(count_folders(path, level))
    if get_dictionary:
        pprint(make_dictionary_of_filecounts(path))
    if make_plots:
        traverse_and_plot(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count the number of folders at a particular level in a directory tree")
    parser.add_argument("path", type=str, help="Path to the directory tree")
    parser.add_argument("--level", type=int, default=0, help="Level of the directory tree to count folders at")
    parser.add_argument("--get_dictionary", action="store_true", default=False, help="Get a dictionary of the number of files in each folder")
    parser.add_argument("--make_plots", action="store_true", default=False, help="Make plots of the durations of the recordings")
    args = parser.parse_args()
    main(args.path, args.level, args.get_dictionary, args.make_plots)
    