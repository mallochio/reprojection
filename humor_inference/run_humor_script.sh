#! /bin/sh
#
# run_humor_script.sh
# Copyright (C) 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.
#


python /openpose/data/other/humor/humor/fitting/run_fitting.py \
                                @/openpose/data/other/humor/configs/fit_rgb_demo_use_split.cfg \
                                --openpose /openpose/ \
                                --data-path $1 \
                                --out $2
