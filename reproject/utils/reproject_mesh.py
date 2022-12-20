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

config = load_config()
base_dir = config["base_dir"]
stereo_params_file = config["stereo_params_k0"]

mesh_img_coord_dict  = {}


def get_distortion_matrix(params):
    # Dk = np.asarray(params["RadialDistortion"])
    # Dp = np.asarray(params["TangentialDistortion"])
    # if len(Dk) == 3:
    #     distortion = np.asarray([Dk[0], Dk[1], Dp[0], Dp[1], Dk[2]])
    # else:
    #     distortion = np.asarray([Dk[0], Dk[1], Dp[0], Dp[1]])
    
    return params['distortion']


def undistort_and_project_points(params, x, y):
    # Projects from image coordinates to camera coordinates
    distortion = params['distortion']
    intrinsics = np.array(params["intrinsics"])
    temp = cv2.undistortPoints(
        src=np.vstack((x, y)).astype(np.float64),
        cameraMatrix=intrinsics,
        distCoeffs=distortion,
    )
    x, y = np.squeeze(np.array(temp), axis=1).T
    return x, y


def transform_RGBimgcoords_to_depthcoords(
    pcloud, depth_frame, depth_params, color_params, R, t, need_image_coordinates=True
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
        t (_type_): Translation vector to go from RGB to depth

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
    depth_camera_coordinates = np.dot(rgb_camera_coordinates, R).T + (t / 1000.0)

    ### Step 3 - Transform the depth camera coordinates to the depth image coordinates
    distortion = get_distortion_matrix(depth_params)
    # depth_intrinsics = np.array(depth_params["IntrinsicMatrix"]).T
    depth_intrinsics = np.array(depth_params["intrinsics"])

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

    return depthX, depthY, depthZ, z_original


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
    global mesh_img_coord_dict  # TODO - remove global variable and cache the dictionary
    if image in mesh_img_coord_dict:
        pcloud = mesh_img_coord_dict[image]
    else:
        pickle_files = read_pickles(pickle_dir)
        mesh_img_coord_dict = get_SMPL_vertices_in_img_coords(pickle_files)
        pcloud = mesh_img_coord_dict[image]
    return pcloud


def get_depth_image(image, depth_dir):
    image = image.replace(".jpg", ".png")
    filepath = os.path.join(depth_dir, image)
    # imgarray = np.array(Image.open(filepath))
    # depth_img = np.array(256 * imgarray / 0x0FFF, dtype=np.uint8)
    depth_img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    return depth_img


def make_padded_img(depth_img, rgb_img):
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
    # Get the index of the first point in the mesh that has the same x coordinate as the pointX
    ix = np.where(depth_img_mesh[0] == pointX)[0][0] 
    if depth_img_mesh[1, ix] == pointY:
        z = depth_img_mesh[2, ix]
    return z, camera_distance


def shift_and_scale_pelvicZ(depth_img_mesh, camera_distance_map):
    """
    Get the camera distances for the mesh pelvic coordinates so they can be correctly placed in the depth image
    Also need to scale the mesh Z axis to that of the depth image
    Ideally the camera distance is to be calculated on the intersection of the mesh and segmentation mask, so it filters out the part lying behind occlusions
    Now just calculating the average of the least 1000 points
    """
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
    camera_distance_map = []
    for i, j in zip(depthX, depthY):
        if 0 <= i < 512 and 0 <= j < 424:
            camera_distance_map.append((depth_img[int(j), int(i)],(i, j)))
    camera_distance_map_sorted = sorted(camera_distance_map) # sorts by first element of tuple by default
    return camera_distance_map_sorted


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


def get_directory_structure(k_idx):
    rgb_images_dir = f'{base_dir}/capture{k_idx}/rgb'
    depth_images_dir = f'{base_dir}/capture{k_idx}/depth'
    pickle_dir = f'{base_dir}/capture{k_idx}/mocap_output/mocap'
    return rgb_images_dir, depth_images_dir, pickle_dir


def get_mesh_in_depth_coordinates(config, pickle_file, k_ix, need_image_coordinates_flag=True):
    # TODO:
    # 1. Make this run on multiple images with the framekeeper class
        # a. Need to make the config just accept one base directory and not have depth and rgb images
    # 2. Complete correctly scaling the Z-axis and the camera distances

    rgb_images_dir, depth_images_dir, pickle_dir = get_directory_structure(k_ix)
    directory, filename = os.path.split(pickle_file)
    image = [''.join((filename.split('_')[0], '.jpg'))][0]
    rgb_img = cv2.imread(os.path.join(rgb_images_dir, image))
    depth_img = get_depth_image(image, depth_images_dir)
    pcloud = get_pointcloud(image, pickle_dir)
    rgb_params_file = config[f'rgb_params_k{k_ix}']
    depth_params_file = config[f'depth_params_k{k_ix}']
    omni_params_file = config['omni_params']

    with open(depth_params_file, "rb") as fh:
        depth_params = pd.read_pickle(fh)

    with open(stereo_params_file, "rb") as fh:
        stereo_params = pd.read_pickle(fh)

    with open(rgb_params_file, "rb") as fh:
        rgb_params = pd.read_pickle(fh)

    R, t = stereo_params['R'], stereo_params['t']
    depthX, depthY, depthZ, pelvicZ = transform_RGBimgcoords_to_depthcoords(
        pcloud,
        depth_frame=depth_img,
        depth_params=depth_params,#params["CameraParameters2"],
        color_params=rgb_params, #params["CameraParameters1"],
        R=R,#np.array(params["RotationOfCamera2"]),
        t=t,#np.array(params["TranslationOfCamera2"]).reshape(1, 3),
        need_image_coordinates=need_image_coordinates_flag,
    )
    
    # depth_bigger = make_padded_img(depth_img, rgb_img)
    # plot_mesh_on_img_2D(depth_bigger, depthX, depthY)

    # Getting the camera distance of the mesh    
    camera_distance_map = get_camera_distance_map(depth_img, depthX, depthY)
    if len(camera_distance_map):
        distances = [i[0] for i in camera_distance_map]
        avgdist = statistics.median(distances)
        depthZ = depthZ + avgdist

    # Transform mesh from depth image coordinates to depth camera coordinates
    depthX, depthY = undistort_and_project_points(depth_params, depthX, depthY)
    depthZ = pelvicZ
    # depthZ = depthZ - min(depthZ) # shifting the depth values to start from 0
    
    # plot_mesh_3D(depthX, depthY, pelvicZ, dst_filepath="/home/sid/mesh.html")
    # print(blah)

    # scale = scale_depth(depth_img_mesh, camera_distance_map)
    # print(f"Scale: {scale}")
    return depthX, depthY, depthZ

# def main():
#     depthX, depthY, depthZ = get_mesh_in_depth_coordinates(config)


# if __name__ == "__main__":
#     main()
