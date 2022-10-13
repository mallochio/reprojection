"""
Generate demo for frames
"""
# First of all, let this module find others.
import sys
from os.path import abspath
sys.path.append(abspath('.'))
sys.path.append(abspath('..'))

import utils.morphology as morphology
from utils import kinect
from utils import omni
from utils.framekeeper import FrameKeeper
import numpy as np
import cv2
from tqdm import tqdm
from utils import distance_filter as df
from config import load_config as conf

# Destination image size
uw, uh = 1200, 900
scale = .4
dsize = int(uw*scale), int(uh*scale)

config =  conf.load_config()
# Working directory_config()
base_dir = config['base_dir']
debug = False  # Show additional information


def project_kinect_to_omni(frames, k_idx, fk):
    depth = np.float32(cv2.flip(frames['capture%d' % k_idx], 1))
    depth = cv2.undistort(depth, fk.kinect_params[k_idx]['K'], fk.kinect_params[k_idx]['D'])

    cv2.imshow('depth_ud', depth)
    background = fk.empty_depth[k_idx].copy()

    depth_visible = np.uint8((depth / 4500.) * 255.)
    # cv2.imshow('depth image%d' % k_idx, cv2.resize(depth_visible, dsize=dsize))

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
    # cv2.imshow('masked_depth%d' % k_idx, cv2.resize(binary, dsize=dsize))

    # Get world coordinates (metres) from mediandepth image (pixels x,y and z metres)
    world_coordinates, valid_depth_coordinates = kinect.depth_to_world(masked_depth, depth,
                                                                           fk.kinect_params[k_idx]['k_params'])

    # X, Y = omni.world_to_omni_scaramuzza(fk.Ts[k_idx], world_coordinates, fk.omni_params[k_idx], uw, uh)
    X, Y = omni.world_to_omni_scaramuzza_fast(fk.Ts[k_idx], world_coordinates, fk.omni_params[k_idx], uw, uh)

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

    #for ts in tqdm(np.arange(ts0, tsf, step=step)):
    for ts in tqdm(fk.get_lead_span()):
        out_frame = np.zeros((vid_size[1], vid_size[0], 3), dtype=np.uint8)
        side_frame = np.zeros((uh, uw*2, 3), dtype=np.uint8)
        # Get a set of sync'ed frames
        frames, _ = fk.get_syncro_frames(ts, debug=debug)
        
        # Load omnidirectional camera image
        omni_image = frames['omni']
        omni_mask = np.zeros(omni_image.shape[:2], dtype=np.uint8)

        # Project each kinect point cloud onto the omnidirectional camera view.
        for k_idx in range(fk.num_kinects):
            Xu, Yu, depth_vis, binary_mask = project_kinect_to_omni(frames, k_idx, fk)
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
