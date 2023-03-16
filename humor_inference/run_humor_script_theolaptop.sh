#! /bin/sh
#
# run_humor_script_theolaptop.sh
# Copyright (C) 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.
#


python /home/cactus/Code/humor/humor/fitting/run_fitting.py \
                                @/home/cactus/Code/humor/configs/fit_rgb_demo_use_split.cfg \
                                --openpose /usr/bin/openpose \
                                --data-path $1 \
                                --out $2
