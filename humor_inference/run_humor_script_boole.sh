#! /bin/sh
#
# run_humor_script_boole.sh
# Copyright (C) 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.
# For the Boole cluster.

python /home/tmorales/humor/humor/fitting/run_fitting.py \
                                @/home/tmorales/humor/configs/fit_rgb_demo_use_split.cfg \
                                --openpose /openpose \
                                --data-path $1 \
                                --out $2 \
                                --batch-size 32 \
                                --rgb-intrinsics $3

