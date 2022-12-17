#! /bin/sh
#
# example_run.sh
# Copyright (C) 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.
#



# If the camera intrinsics were calibrated in mm, pass z in mm
./main.py cam_mat.npy --img-pt 524 1592 -z 1090
