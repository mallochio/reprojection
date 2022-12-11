import sys
from os.path import abspath

sys.path.append(abspath("."))
sys.path.append(abspath(".."))


import numpy as np
import cv2
import json
import os
import statistics
import random
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
from config.load_config import load_config


params_file = config["stereo_params_k0"]
pickle_dir = config["pickle_dir"]
depth_images_dir = config["depth_images_dir"]
rgb_images_dir = config["rgb_images_dir"]
mesh_img_coord_dict  = {}


def get_distortion_matrix(params):
    Dk = np.asarray(params["RadialDistortion"])
    Dp = np.asarray(params["TangentialDistortion"])
    if len(Dk) == 3:
        distortion = np.asarray([Dk[0], Dk[1], Dp[0], Dp[1], Dk[2]])
    else:
        distortion = np.asarray([Dk[0], Dk[1], Dp[0], Dp[1]])
    return distortion


def undistort_and_project_points(params, x, y):
    # Projects from image coordinates to camera coordinates
    distortion = get_distortion_matrix(params)
    intrinsics = np.array(params["IntrinsicMatrix"]).T
    temp = cv2.undistortPoints(
        src=np.vstack((x, y)).astype(np.float64),
        cameraMatrix=intrinsics,
        distCoeffs=distortion,
    )
    x, y = np.squeeze(np.array(temp), axis=1).T
    return x, y


def transform_RGBimgcoords_to_depthcoords(
    pcloud, depth_frame, depth_params, color_params, R, T, need_image_coordinates=False
):
    """
    # TODO - complete docstring with types
    _summary_: Function to transform the image coordinate system to the depth coordinate system

    Args:
        x, y, z (_type_): RGB image coordinates of the mesh
        depth_frame (_type_):  the depth image onto which the mesh will be projected
        depth_params (_type_): intrinsics of the depth camera
        color_params (_type_): intrinsics of the RGB camera
        R (_type_): Rotation matrix to go from RGB to depth
        T (_type_): Translation vector to go from RGB to depth

    Returns:
        _type_: Depth image coordinates of the mesh
    """

    ### Step 1 - Transform the RGB image coordinates to the RGB camera coordinates
    x, y, z_original = pcloud.T
    z = np.ones(z_original.shape)

    # Undistort and project the points
    rgb_camera_coordinates = np.zeros((len(x), 3))
    x, y = undistort_and_project_points(color_params, x, y)

    rgb_camera_coordinates[:, 0] = x
    rgb_camera_coordinates[:, 1] = y
    rgb_camera_coordinates[:, 2] = z

    ### Step 2 - Transform the RGB camera coordinates to the depth camera coordinates using the extrinsic stereo parameters
    depth_camera_coordinates = np.dot(rgb_camera_coordinates, R) + (T / 1000.0)

    ### Step 3 - Transform the depth camera coordinates to the depth image coordinates
    distortion = get_distortion_matrix(depth_params)
    depth_intrinsics = np.array(depth_params["IntrinsicMatrix"]).T

    if need_image_coordinates:
        # Distorts and projects the points into image coordinates
        points, _ = cv2.projectPoints(
            depth_camera_coordinates,
            rvec=(0, 0, 0),
            tvec=(0, 0, 0),
            cameraMatrix=depth_intrinsics,
            distCoeffs=distortion,
        )
        depthX, depthY = np.squeeze(np.array(points), axis=1).T
        depthZ = z_original
    else:
        depthX, depthY, depthZ = depth_camera_coordinates.T

    return depthX, depthY, depthZ


def read_pickles(pickle_dir):
    # Function to read frankmocap pickle files
    pickle_files = os.listdir(pickle_dir)
    pickle_files = [os.path.join(pickle_dir, f) for f in pickle_files if f.endswith(".pkl")]
    return pickle_files


def get_SMPL_vertices_in_img_coords(pickle_files):
    img_coord_dict = {}
    for i in pickle_files:
        obj = pd.read_pickle(i)
        if obj["pred_output_list"][0] != None:
            img_coord_dict[os.path.basename(obj["image_path"])] = obj[
                "pred_output_list"
            ][0]["pred_vertices_img"]
    return img_coord_dict


def get_pointcloud(image, pickle_dir):
    # Get the point cloud in RGB image coordinates for a given image
    # Uses caching to speed up the process
    if image in mesh_img_coord_dict:
        pcloud = mesh_img_coord_dict[image]
    else:
        pickle_files = read_pickles(pickle_dir)
        mesh_img_coord_dict = get_SMPL_vertices_in_img_coords(pickle_files)
        pcloud = mesh_img_coord_dict[image]
    return pcloud


def get_depth_image(image, depth_dir):
    image = image.replace(".jpg", ".png")
    file = os.path.join(depth_dir, image)
    depth_img = np.array(256 * np.array(Image.open(file)) / 0x0FFF, dtype=np.uint8)
    return depth_img


def make_bordered_img(depth_img, rgb_img):
    r, c, _ = rgb_img.shape
    depth_bigger = np.zeros((r, c, 3), dtype=np.uint8)
    nr, nc = depth_img.shape
    depthimg = cv2.merge((depth_img, depth_img, depth_img))
    depth_bigger[
        int(r / 2 - nr / 2) : int(r / 2 + nr / 2),
        int(c / 2 - nc / 2) : int(c / 2 + nc / 2),
    ] = depthimg
    return depth_bigger


def plot_mesh_on_img_2D(img, x, y):
    implot = plt.imshow(img)
    plt.scatter(x=x, y=y, c="cornflowerblue", s=0.5)
    plt.show()


def plot_mesh_3D(x, y, z, dst_filepath):
    fig = px.scatter_3d(x=x, y=y, z=z)
    fig.update_traces(marker={"size": 1})
    # fig.update_layout(
    #     autosize=False,
    #     width=500,
    #     height=500)
    fig.write_html(dst_filepath)


def get_z_randomly(depth_img_mesh, camera_distance_map):
    camera_distance = random.sample(camera_distance_map.keys(), 1)[0]
    pointX, pointY = camera_distance_map[camera_distance]
    ix = np.where(depth_img_mesh[0] == pointX)[0][
        0
    ]  # get the index of the first point in the mesh that has the same x coordinate as the pointX
    if depth_img_mesh[1, ix] == pointY:
        z = depth_img_mesh[2, ix]
    return z, camera_distance


def scale_depth(depth_img_mesh, camera_distance_map):
    scales = []
    for i in range(1000):
        z1, cam_dist1 = get_z_randomly(depth_img_mesh, camera_distance_map)
        z2, cam_dist2 = get_z_randomly(depth_img_mesh, camera_distance_map)
        if cam_dist1 != cam_dist2:
            scale = (z1 - z2) / (cam_dist1 - cam_dist2)
            scales.append(scale)

    return statistics.mode(scales)


def get_camera_distance_map(depth_img, depthX, depthY):
    # Get a dictionary of camera distance of the mesh and the corresponding pixel coordinates
    camera_distance_map = {}
    for i, j in zip(depthX, depthY):
        if 0 <= i < 512 and 0 <= j < 424:
            camera_distance_map[depth_img[int(j), int(i)]] = (i, j)
    return camera_distance_map


def shift_mesh_to_original_depth_img_size(rgb_img, depth_img, depthX, depthY):
    # Doing this because the depth image used to calibrate the camera is
    # made larger using borders, so the projected mesh is also to be shifted
    r, c, _ = rgb_img.shape
    nr, nc = depth_img.shape
    xmin, xmax = int(c / 2 - nc / 2), int(c / 2 + nc / 2)
    ymin, ymax = int(r / 2 - nr / 2), int(r / 2 + nr / 2)
    depthX -= xmin
    depthY -= ymin
    return depthX, depthY


def get_mesh_in_depth_coordinates(config, pickle_file, need_image_coordinates_flag = False):
    # TODO:
    # 1. Make this run on multiple images with the framekeeper class
        # a. Need to make the config just accept one base directory and not have depth and rgb images
    # 2. Complete correctly scaling the Z-axis and the camera distances
    
    # image = "1665057394063.jpg"  # selecting one random image from the pickle_dir
    # To run on multiple images # TODO: incomplete
    # mocap_images = [''.join((i.split('_')[0], '.jpg')) for i in os.listdir(pickle_dir)]
    # for image in mocap_images:
    directory, filename = os.path.split(pickle_file)
    
    image = [''.join((filename.split('_')[0], '.jpg'))]
    rgb_img = cv2.imread(os.path.join(rgb_images_dir, image))
    depth_img = get_depth_image(image, depth_images_dir)
    pcloud = get_pointcloud(image, pickle_dir)

    with open(params_file, "r") as fh:
        params = json.load(fh)

    depthX, depthY, depthZ = transform_RGBimgcoords_to_depthcoords(
        pcloud,
        depth_frame=depth_img,
        depth_params=params["CameraParameters2"],
        color_params=params["CameraParameters1"],
        R=np.array(params["RotationOfCamera2"]),
        T=np.array(params["TranslationOfCamera2"]).reshape(1, 3),
        need_image_coordinates=need_image_coordinates_flag,
    )
    if need_image_coordinates_flag:
        # depth_bigger = make_bordered_img(depth_img, rgb_img)
        # plot_mesh_on_img_2D(depth_bigger, depthX, depthY)
        # plot_mesh_3D(depthX, depthY, depthZ, dst_filepath="/home/sid/mesh.html")

        # Shifting the mesh to a depth image of original size
        depthX, depthY = shift_mesh_to_original_depth_img_size(
            rgb_img, depth_img, depthX, depthY
        )
        # shifting the depth values to start from 0; not sure if required
        depthZ_shifted = depthZ - min(depthZ) # shifting the depth values to start from 0
        depth_img_mesh = np.vstack((depthX, depthY, depthZ_shifted))

    # Getting the camera distance of the mesh
    # camera_distance_map = get_camera_distance_map(depth_img, depthX, depthY)
    # scale = scale_depth(depth_img_mesh, camera_distance_map)
    # print(f"Scale: {scale}")

    return depthX, depthY, depthZ


def main():
    config = load_config()
    depthX, depthY, depthZ = get_mesh_in_depth_coordinates(config)


if __name__ == "__main__":
    main()
