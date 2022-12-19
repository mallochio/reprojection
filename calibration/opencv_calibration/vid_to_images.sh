#! /bin/sh
#
# vid_to_images.sh
# Copyright (C) 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.
#


if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <video_file> <output_dir> <FPS>"
    exit 1
fi
if [ ! -d $2 ]; then
    echo "[*] Creating directory $2"
    mkdir $2
fi

echo "[*] Converting $1 to PNGs in $2 at $3 FPS"
ffmpeg -i $1 -vf fps=$3 $2/out%d.png
#echo "[*] Extracted ${ls -l $2 | wc -l} frames"
