#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   remove-sound.py
@Time    :   2023/02/15 16:52:35
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Recursively traverse  a directory to find mp4 files and remove the sound from them using ffmpeg
'''

import os
import subprocess
import argparse


def remove_sound(file_path, dst_path=None):
    print(f"[*] Removing sound from {file_path}")
    subprocess.call(["ffmpeg", "-i", file_path, "-c:v", "copy", "-an", dst_path])

def main():
    parser = argparse.ArgumentParser(description="Remove sound from mp4 files in a directory")
    parser.add_argument("--directory", "-i", help="Directory to traverse")
    parser.add_argument("--dst", "-o", help="Destination directory")
    args = parser.parse_args()
    directory = args.directory
    for root, _, files in os.walk(directory):
        for file in files:
            print(file)
            if file.endswith(".MP4"):
                remove_sound(os.path.join(root, file) , os.path.join(args.dst, file))

if __name__ == "__main__":
    main()
