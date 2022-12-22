"""
Generate demo using mesh reprojections 
"""

import sys
from os.path import abspath

sys.path.append(abspath("."))
sys.path.append(abspath(".."))


import cv2
import pickle
from config import load_config as conf
from utils import distance_filter as df
from tqdm import tqdm
import numpy as np
from utils.framekeeper import FrameKeeper
from utils import omni
from utils import kinect
import utils.morphology as morphology
from utils import reproject_mesh
from utils.reproject_mesh import get_mesh_in_depth_coordinates, plot_mesh_3D

############
## For debugging
from icecream import ic, install

install()
###########

# Destination image size - This is the omni's image size
uw, uh = 1200, 900
scale = 0.4
dsize = int(uw * scale), int(uh * scale)

config = conf.load_config()
# Working directory_config()
base_dir = config["base_dir"]
debug = False  # Show additional information


def get_depth_visible(frames, k_idx, fk):
    depth = np.float64(cv2.flip(frames["capture%d" % k_idx], 1))

    depth = cv2.undistort(
        depth,
        fk.kinect_params[k_idx]["K"],
        fk.kinect_params[k_idx]["D"],
    )

    cv2.imshow("depth_ud", depth)
    background = fk.empty_depth[k_idx].copy()
    depth_visible = np.uint8((depth / 4500.0) * 255.0)
    return depth, depth_visible


def get_masked_depth_and_binary(depth, mask=None):
    if mask:
        masked_depth = np.zeros_like(depth)
        masked_depth[mask > 0] = depth[mask > 0]
    else:
        masked_depth = depth

    binary = np.uint8(np.uint8(masked_depth / 4500.0 * 255.0) > 0) * 255
    return masked_depth, binary


def get_mask(frames, k_idx, fk, depth):
    # Mask is the intersection of a motion mask and a densepose mask
    background = fk.empty_depth[k_idx].copy()
    motion_mask = morphology.generate_mask(depth, background, cc=True)
    mask = motion_mask
    mask = cv2.flip(frames[f"_capture{k_idx}_rgb_densepose"][:, :, 0], 1)
    mask = np.logical_and(mask, motion_mask)
    return mask


def make_transformation_matrix(ix):
    cam0_to_world_pth = config[f"k{ix}_depth_to_world"]
    world_to_cam1_pth = config[f"k{ix}_world_to_omni"]

    with open(cam0_to_world_pth, "rb") as f:
        pose = pickle.load(f)
        cam0_to_world = np.vstack((np.hstack((pose["R"], pose["t"])), [0, 0, 0, 1]))
    with open(world_to_cam1_pth, "rb") as f:
        pose = pickle.load(f)
        world_to_cam1 = np.vstack((np.hstack((pose["R"], pose["t"])), [0, 0, 0, 1]))

    transform = world_to_cam1 @ cam0_to_world
    transform[:3, 3] = transform[:3, 3] / 1000.0
    return transform


def project_kinect_to_omni(frames, k_idx, fk):
    mesh_pickle_file = frames[f"_capture{k_idx}_frankmocap"]
    depth, depth_visible = get_depth_visible(frames, k_idx, fk)

    # mask = get_mask(frames, k_idx, fk, depth)
    mask = None
    masked_depth, binary = get_masked_depth_and_binary(depth, mask)

    # Get camera coordinates in depth image space
    depthX, depthY, depthZ = get_mesh_in_depth_coordinates(config, mesh_pickle_file, k_idx, need_image_coordinates_flag=True)
    plot_mesh_3D(depthX, depthY, depthZ, dst_filepath="/home/sid/mesh-depth-cam2.html")
    depth_camera_coordinates = np.stack([depthX, depthY, depthZ, np.ones_like(depthX)], axis=1)

    # Project the mesh onto the omnidirectional camera frame using extrinsics.
    transform = make_transformation_matrix(k_idx)
    omni_camera_coordinates = transform @ depth_camera_coordinates.T
    
    omni_camera_coordinates = omni_camera_coordinates[:3, :].T
    omnicamX, omniCamY, omniCamZ = omni_camera_coordinates.T


    plot_mesh_3D(omnicamX, omniCamY, omniCamZ, dst_filepath="/home/sid/mesh-omni-cam.html")

    omni_camera_coordinates = np.expand_dims(omni_camera_coordinates, axis=0)

    # Project the mesh onto the omnidirectional image frame.
    omni_params_file = config["omni_params"]
    with open(omni_params_file, "rb") as f:
        omni_params = pickle.load(f)

    xi = omni_params["xi"]
    xi = xi.item() if isinstance(xi, np.ndarray) else xi

    omni_image_coordinates, _ = cv2.omnidir.projectPoints(
        omni_camera_coordinates.astype(np.float64),
        np.zeros(3),
        np.zeros(3),
        omni_params["intrinsics"],  
        xi,
        omni_params["distortion"],
    )

    omni_image_coordinates = np.squeeze(omni_image_coordinates, axis=0).T
     

    # omni_image_coordinates = np.swapaxes(omni_image_coordinates, 0, 1)
    omniX, omniY = omni_image_coordinates[0], omni_image_coordinates[1]

    X, Y = np.real(omniX), np.real(omniY)
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
            omni_image[Yu, Xu] = omni_image[Yu, Xu] / 1.5 + np.array(
                assigned_colors[k_idx]
            )
            omni_mask[Yu, Xu] = 255

            out_frame[
                0 : vid_size[1] // 2, k_idx * dsize[0] : k_idx * dsize[0] + dsize[0], :
            ] = cv2.cvtColor(cv2.resize(depth_vis, dsize), cv2.COLOR_GRAY2BGR)
            out_frame[
                vid_size[1] // 2 :, k_idx * dsize[0] : k_idx * dsize[0] + dsize[0], :
            ] = cv2.cvtColor(cv2.resize(binary_mask, dsize), cv2.COLOR_GRAY2BGR)

        # cv2.imshow('fisheye view', cv2.resize(omni_image, dsize=dsize))
        out_frame[0 : vid_size[1] // 2, 2 * dsize[0] :, :] = cv2.resize(
            omni_image, dsize
        )
        side_frame[:, 0:uw] = cv2.resize(omni_image, (uw, uh))

        omni_mask = morphology.paco(omni_mask)
        # cv2.imshow('fisheye mask', cv2.resize(omni_mask, dsize=dsize))

        out_frame[vid_size[1] // 2 :, 2 * dsize[0] :, :] = cv2.cvtColor(
            cv2.resize(omni_mask, dsize), cv2.COLOR_GRAY2BGR
        )
        side_frame[:, uw:] = cv2.cvtColor(
            cv2.resize(omni_mask, (uw, uh)), cv2.COLOR_GRAY2BGR
        )

        cv2.imshow("out_frame", out_frame)
        cv2.imshow("side_frame", side_frame)
        out.write(out_frame)
        side_out.write(side_frame)

        cv2.waitKey(1)

    out.release()
    side_out.release()


if __name__ == "__main__":
    main()
