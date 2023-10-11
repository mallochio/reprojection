# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import logging
import os
import pickle
import sys
from typing import Any, ClassVar, Dict, List
import torch
import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0,'/home/appuser/detectron2_repo/projects/DensePose')
print(sys.version)
print(sys.executable)

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor

from densepose import add_densepose_config
# from densepose.data.structures import DensePoseResult
from densepose.structures import quantize_densepose_chart_result
#from densepose.vis.densepose import DensePoseResultsVisualizer
from densepose.vis.densepose_results import DensePoseResultsVisualizer
from densepose.vis.extractor import CompoundExtractor, create_extractor


def _get_input_file_list(input_spec: str):
    if os.path.isdir(input_spec):
        file_list = [
            os.path.join(input_spec, fname)
            for fname in os.listdir(input_spec)
            if os.path.isfile(os.path.join(input_spec, fname))
        ]
    elif os.path.isfile(input_spec):
        file_list = [input_spec]
    else:
        file_list = glob.glob(input_spec)
    return file_list


def setup_config(config_fpath: str, model_fpath: str):
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(config_fpath)
    cfg.MODEL.WEIGHTS = model_fpath
    cfg.freeze()
    return cfg

cfg_file = './model/densepose_rcnn_R_50_FPN_DL_WC1_s1x.yaml'
model_file = './model/model_final_b1e525.pkl'

# dataset = '/media/pau/extern_wd/Datasets/MuHAVI-MAS-proc/images/'
dataset = '/data/rgb/'
pattern = '*.jpg'
input_dir = dataset + pattern
iuv_output_dir = '/data/rgb_dp2_iuv/'
output_dir = '/data/rgb_dp2_mask/'


cfg = setup_config(cfg_file, model_file)
predictor = DefaultPredictor(cfg)
file_list = _get_input_file_list(input_dir)
if len(file_list) == 0:
    print('No files found for provided input')
    exit(-1)
else:
    print('Found %d files to process ...' % len(file_list))

# Create output directories
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(iuv_output_dir):
    os.makedirs(iuv_output_dir)

with torch.no_grad():
    for file_name in tqdm(file_list):
        # Come up with a suitable name for output image
        out_filename = file_name.replace(dataset, '')
        out_filename = out_filename.replace('/', '_')
        mask_out_filename = output_dir + out_filename.replace('.jpg', '.png')
        iuv_out_filename = iuv_output_dir + out_filename.replace('.jpg', '.png')

        if not os.path.exists(mask_out_filename) or not os.path.exists(iuv_out_filename):
            img = read_image(file_name, format="BGR")  # predictor expects BGR image.
            outputs = predictor(img)["instances"]

            visualizer = DensePoseResultsVisualizer()
            extractor = create_extractor(visualizer)
            
            outputs = predictor(img)
            instances = outputs['instances']
            scores = instances._fields['scores'].cpu().numpy()
            data = extractor(instances)

            uv_results, bboxes = data
            # if not uv_results:
            #    continue

            results = [(scores[s], uv_results[s], bboxes[s]) for s in range(len(scores))]

            # if results and not np.all(scores == scores[0]):
            results = sorted(results, key=lambda r: r[0])

            iuv_img = np.zeros_like(img)
            bbox_dict = {}
            for result in results:
                    score, res, bbox = result
                    if score > 0.8:
                        # iuv_arr = DensePoseResult.decode_png_data(*res)
                        qres = quantize_densepose_chart_result(res)
                        iuv_arr = qres.labels_uv_uint8.cpu()
                        x, y, w, h = map(int, bbox)
                        for c in range(3):
                            iuv_img[y:y+h,x:x+w,c] = iuv_arr[c, :, :]
                        # cv2.rectangle(iuv_img, (x,y), (x+w,y+h), (255,255,255))
                        bbox_dict = {"score": float(score), "rect": [x,y,w,h]}

            mask = np.uint8(iuv_img > 0) * 255
            #cv2.imshow('iuv', iuv_img)
            #cv2.imshow('mask', mask)
            #cv2.waitKey(5)

            # Save detectron2/densepose mask images.
            cv2.imwrite(iuv_out_filename, iuv_img)
            cv2.imwrite(mask_out_filename, mask)
            print("  Saved '%s'." % mask_out_filename)
        else:
            print("  Skipped '%s'." % mask_out_filename)
print('Done.')

cv2.destroyAllWindows()
