"""
Generate demo for frames
"""
# First of all, let this module find others.
import os
import sys
from os.path import abspath
from unittest import result

sys.path.append(abspath('.'))
sys.path.append(abspath('..'))

import utils.morphology as morphology
from utils import kinect
from utils import omni
from utils.framekeeper import FrameKeeper
from utils import distance_filter as df
import numpy as np
import cv2
from tqdm import tqdm

import trimesh
import torch
from config import load_config as conf


results_folder = "/home/sid/Projects/OmniScience/dataset/session-recordings/2024-01-12/at-unis/lab/sid/capture2/results_out/final_results"
sys.path.insert(0, '/home/sid/Projects/OmniScience/other/humor')
sys.path.insert(0, '/home/sid/Projects/OmniScience/code/reprojection')

from humor.fitting.fitting_utils import prep_res, run_smpl
from humor.body_model.body_model import BodyModel
from humor.utils.torch import copy2cpu as c2c
from humor_inference.reproject_humor_sequence import sanitize_preds


# Destination image size
uw, uh = 1200, 900
scale = .4
dsize = int(uw*scale), int(uh*scale)

config =  conf.load_config()
# Working directory_config()
base_dir = config['base_dir']
debug = False  # Show additional information


def get_meshes():
    res_file = os.path.join(results_folder, "stage3_results.npz")
    if not os.path.isfile(res_file):
        raise Exception(f"Could not find {res_file}!")

    device = torch.device("cpu")
    pred_res = np.load(res_file)
    T = pred_res["trans"].shape[0]

    sanitize_preds(pred_res, T)
    pred_res = prep_res(pred_res, device, T)
    num_pred_betas = pred_res["betas"].size(1)

    # create body models for each
    meta_path = os.path.join(results_folder, "meta.txt")
    if not os.path.exists(meta_path):
        raise Exception(f"Could not find {meta_path}!")
        
    optim_bm_path = None
    with open(meta_path, "r") as f:
        optim_bm_str = f.readline().strip()
        optim_bm_path = optim_bm_str.split(" ")[1]

    # humor model
    pred_bm = BodyModel(
        bm_path=optim_bm_path, num_betas=num_pred_betas, batch_size=T
    ).to(device)

    # run through SMPL
    pred_body = run_smpl(pred_res, pred_bm)
    print("[*] Loaded the sequence of SMPL models!")
    
    faces = c2c(pred_body.f)
    body_mesh_seq = [
        trimesh.Trimesh(
            vertices=c2c(pred_body.v[i]),
            faces=faces,
            process=False,
        )
        for i in range(pred_body.v.size(0))
    ]
    return body_mesh_seq


def project_kinect_to_omni(frames, k_idx, fk, frame_list, body_mesh_seq):
    depth = np.float32(cv2.flip(frames['capture%d' % k_idx], 1))
    depth = cv2.undistort(depth, fk.kinect_params[k_idx]['K'], fk.kinect_params[k_idx]['D'])

    cv2.imshow('depth_ud', depth)
    background = fk.empty_depth[k_idx].copy()

    depth_visible = np.uint8((depth / 4500.) * 255.)

    # Densepose AVAILABLE
    # mask = cv2.flip(frames['_capture%d_rgb_densepose' % k_idx][:,:,0], 1)
    # mask = np.logical_and(mask, motion_mask)
    
    # No Densepose EXISTS
    motion_mask = morphology.generate_mask(depth, background, cc=True)
    mask = motion_mask
    
    masked_depth = np.zeros_like(depth)
    masked_depth[mask>0] = depth[mask>0]

    #valid = df.proximity_filter_fast(masked_depth, fk.kinect_params[k_idx]['k_params'])
    #masked_depth = valid

    binary = np.uint8(np.uint8(masked_depth/4500.*255.) > 0)*255

    # Get world coordinates (metres) from mediandepth image (pixels x,y and z metres)
    world_coordinates, valid_depth_coordinates = kinect.depth_to_world(masked_depth, depth, fk.kinect_params[k_idx]['k_params'])
    X, Y = omni.world_to_omni_scaramuzza_fast(fk.Ts[k_idx], world_coordinates, fk.omni_params[k_idx], uw, uh, frame_list, body_mesh_seq)

    X = np.real(X)
    Y = np.real(Y)

    return np.int32(np.round(X)), np.int32(np.round(Y)), depth_visible, binary


def main():
    assigned_colors = [(85,0,0), (0,0,85), (0,85,0)]  # Colors assigned to blend with each point cloud.

    # Get all necessary parameters for reprojection
    fk = FrameKeeper(base_dir, capture_Hz=15)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid_size = (dsize[0]*3, dsize[1]*2)
    out = cv2.VideoWriter('out/output.avi', fourcc, 15.0, vid_size)
    side_vid_size = (uw*2, uh)
    side_out = cv2.VideoWriter('out/output_side.avi', fourcc, 15.0, side_vid_size)

    ts0, tsf = fk.get_span()
    step = fk.get_step_ms() / 1000.
    print("Timespan: t0 = %.3f, tf = %.3f (step = %.3f)" % (ts0, tsf, step))
    body_mesh_seq = get_meshes()
    for ts in tqdm(fk.get_lead_span()):
        out_frame = np.zeros((vid_size[1], vid_size[0], 3), dtype=np.uint8)
        side_frame = np.zeros((uh, uw*2, 3), dtype=np.uint8)
        # Get a set of sync'ed frames
        frames, frame_list = fk.get_syncro_frames(ts, debug=debug)

        # Load omnidirectional camera image
        omni_image = frames['omni']

        # Mirror omni image
        omni_image = cv2.flip(omni_image, 1)
        omni_mask = np.zeros(omni_image.shape[:2], dtype=np.uint8)

        # Project each kinect point cloud onto the omnidirectional camera view.
        for k_idx in range(fk.num_kinects):
            if k_idx != 2:
                continue

            Xu, Yu, depth_vis, binary_mask = project_kinect_to_omni(frames, k_idx, fk, frame_list, body_mesh_seq)
            omni_image[Yu, Xu] = omni_image[Yu, Xu] / 1.5 + np.array(assigned_colors[k_idx])
            omni_mask[Yu, Xu] = 255

            out_frame[0:vid_size[1] // 2, k_idx * dsize[0]:k_idx * dsize[0] + dsize[0], :] = \
                cv2.cvtColor(cv2.resize(depth_vis, dsize), cv2.COLOR_GRAY2BGR)
            out_frame[vid_size[1]//2:, k_idx*dsize[0]:k_idx*dsize[0]+dsize[0], :] =\
                cv2.cvtColor(cv2.resize(binary_mask, dsize), cv2.COLOR_GRAY2BGR)

        # cv2.imshow('fisheye view', cv2.resize(omni_image, dsize=dsize))
        out_frame[0:vid_size[1]//2, 2*dsize[0]:, :] = cv2.resize(omni_image, dsize)
        side_frame[:, 0:uw] = cv2.resize(omni_image, (uw, uh))

        omni_mask=morphology.paco(omni_mask)
        # cv2.imshow('fisheye mask', cv2.resize(omni_mask, dsize=dsize))
        out_frame[vid_size[1]//2:, 2*dsize[0]:, :] = cv2.cvtColor(cv2.resize(omni_mask, dsize), cv2.COLOR_GRAY2BGR)
        side_frame[:, uw:] = cv2.cvtColor(cv2.resize(omni_mask, (uw, uh)), cv2.COLOR_GRAY2BGR)

        cv2.imshow('out_frame', out_frame)
        cv2.imshow('side_frame', side_frame)
        out.write(out_frame)
        side_out.write(side_frame)

        cv2.waitKey(1)

    out.release()
    side_out.release()



if __name__ == '__main__':
    main()
