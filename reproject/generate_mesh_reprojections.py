"""
Generate demo using mesh reprojections 
"""

import sys
from os.path import abspath

sys.path.append(abspath("."))
sys.path.append(abspath(".."))


from config import load_config as conf
from utils import distance_filter as df
from tqdm import tqdm
import cv2
import numpy as np
from utils.framekeeper import FrameKeeper
from utils import omni
from utils import kinect
import utils.morphology as morphology
from utils.reproject_mesh import get_mesh_in_depth_coordinates


# Destination image size - This is the omni's image size
uw, uh = 1200, 900
scale = 0.4
dsize = int(uw * scale), int(uh * scale)

config = conf.load_config()
# Working directory_config()
base_dir = config["base_dir"]
debug = False  # Show additional information


def get_depth_visible(frames, k_idx, fk):
    depth = np.float32(cv2.flip(frames["capture%d" % k_idx], 1))

    depth = cv2.undistort(
        depth.astype(np.float64), 
        fk.kinect_params[k_idx]["K"], 
        fk.kinect_params[k_idx]["D"]
    )

    cv2.imshow("depth_ud", depth)
    background = fk.empty_depth[k_idx].copy()
    depth_visible = np.uint8((depth / 4500.0) * 255.0)
    return depth, depth_visible


def get_masked_depth_and_binary(depth, mask):
    masked_depth = np.zeros_like(depth)
    masked_depth[mask > 0] = depth[mask > 0]
    binary = np.uint8(np.uint8(masked_depth / 4500.0 * 255.0) > 0) * 255
    return masked_depth, binary

def get_mask(frames, k_idx, fk, depth):
    # Mask is the intersection of a motion mask and a densepose mask
    background = fk.empty_depth[k_idx].copy()
    motion_mask = morphology.generate_mask(depth, background, cc=True)
    mask = motion_mask
    mask = cv2.flip(frames[f'_capture{k_idx}_rgb_densepose'][:,:,0], 1)
    mask = np.logical_and(mask, motion_mask)
    return mask


def project_kinect_to_omni(frames, k_idx, fk):
    print(frames.keys())
    print(blah)
    depth, depth_visible = get_depth_visible(frames, k_idx, fk)

    # mask = get_mask(frames, k_idx, fk, depth)
    # masked_depth, binary = get_masked_depth_and_binary(depth, mask)
    binary = np.uint8(np.uint8(depth / 4500.0 * 255.0) > 0) * 255
    
    # Get camera coordinates in depth image space
    depthX, depthY, depthZ = get_mesh_in_depth_coordinates(config)
    ones = np.ones_like(depthX)
    depth_camera_coordinates = np.stack([depthX, depthY, depthZ, ones], axis=1)

    X, Y = omni.world_to_omni_scaramuzza_fast(
        fk.Ts[k_idx], 
        depth_camera_coordinates, 
        fk.omni_params[k_idx], 
        uw, uh
    )

    X, Y = np.real(X), np.real(Y)
    return np.int32(np.round(X)), np.int32(np.round(Y)), depth_visible, binary


def main():
    # Colors assigned to blend with each point cloud.
    assigned_colors = [(85, 0, 0), (0, 0, 85), (0, 85, 0)]

    # Get all necessary parameters for reprojection
    fk = FrameKeeper(base_dir, capture_Hz=15)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vid_size = (dsize[0] * 3, dsize[1] * 2)
    out = cv2.VideoWriter("out/output.avi", fourcc, 15.0, vid_size)
    side_vid_size = (uw * 2, uh)
    side_out = cv2.VideoWriter("out/output_side.avi", fourcc, 15.0, side_vid_size)

    ts0, tsf = fk.get_span()
    step = fk.get_step_ms() / 1000.0
    print("Timespan: t0 = %.3f, tf = %.3f (step = %.3f)" % (ts0, tsf, step))

    for ts in tqdm(fk.get_lead_span()):
        out_frame = np.zeros((vid_size[1], vid_size[0], 3), dtype=np.uint8)
        side_frame = np.zeros((uh, uw * 2, 3), dtype=np.uint8)

        # Get a set of sync'ed frames
        frames, _ = fk.get_syncro_frames(ts, debug=debug)

        # Load omnidirectional camera image
        omni_image = frames["omni"]
        omni_mask = np.zeros(omni_image.shape[:2], dtype=np.uint8)

        # Project each kinect point cloud onto the omnidirectional camera view.
        for k_idx in range(fk.num_kinects):
            Xu, Yu, depth_vis, binary_mask = project_kinect_to_omni(frames, k_idx, fk)
            omni_image[Yu, Xu] = omni_image[Yu, Xu] / 1.5 + np.array(assigned_colors[k_idx])
            omni_mask[Yu, Xu] = 255

            out_frame[0 : vid_size[1] // 2, k_idx * dsize[0] : k_idx * dsize[0] + dsize[0], :] = cv2.cvtColor(cv2.resize(depth_vis, dsize), cv2.COLOR_GRAY2BGR)
            out_frame[vid_size[1] // 2 :, k_idx * dsize[0] : k_idx * dsize[0] + dsize[0], :] = cv2.cvtColor(cv2.resize(binary_mask, dsize), cv2.COLOR_GRAY2BGR)

        # cv2.imshow('fisheye view', cv2.resize(omni_image, dsize=dsize))
        out_frame[0 : vid_size[1] // 2, 2 * dsize[0] :, :] = cv2.resize(omni_image, dsize)
        side_frame[:, 0:uw] = cv2.resize(omni_image, (uw, uh))

        omni_mask = morphology.paco(omni_mask)
        # cv2.imshow('fisheye mask', cv2.resize(omni_mask, dsize=dsize))

        out_frame[vid_size[1] // 2 :, 2 * dsize[0] :, :] = cv2.cvtColor(cv2.resize(omni_mask, dsize), cv2.COLOR_GRAY2BGR)
        side_frame[:, uw:] = cv2.cvtColor(cv2.resize(omni_mask, (uw, uh)), cv2.COLOR_GRAY2BGR)

        cv2.imshow("out_frame", out_frame)
        cv2.imshow("side_frame", side_frame)
        out.write(out_frame)
        side_out.write(side_frame)

        cv2.waitKey(1)

    out.release()
    side_out.release()


if __name__ == "__main__":
    main()
