import numpy as np
import cv2
import json
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from config.load_config import load_config


def get_distortion_matrix(params):
    Dk = np.asarray(params["RadialDistortion"])
    Dp = np.asarray(params["TangentialDistortion"])
    if len(Dk) == 3:
        distortion = np.asarray([Dk[0], Dk[1], Dp[0], Dp[1], Dk[2]])
    else:
        distortion = np.asarray([Dk[0], Dk[1], Dp[0], Dp[1]])
    return distortion


def undistort_and_project_points(params, x, y):
    distortion = get_distortion_matrix(params)
    intrinsics = np.array(params["IntrinsicMatrix"]).T
    temp = cv2.undistortPoints(
        src=np.vstack((x, y)).astype(np.float64),
        cameraMatrix=intrinsics,
        distCoeffs=distortion,
    ) 
    x, y = np.squeeze(np.array(temp), axis=1).T
    return x, y


def transform_RGBimgcoords_to_depthimgcoords(
    pcloud, depth_frame, depth_params, color_params, R, T
):
    """_summary_: Function to transform the image coordinate system to the depth coordinate system

    Args:
        x, y, z (_type_): RGB image coordinates of the mesh
        depth_frame (_type_):  the depth image onto which the mesh points will be projected
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

    return depthX, depthY, depthZ


def read_pickles(pickle_dir):
    # Function to read frankmocap pickle files
    pickle_files = os.listdir(pickle_dir)
    pickle_files = [
        os.path.join(pickle_dir, f) for f in pickle_files if f.endswith(".pkl")
    ]
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
    pickle_files = read_pickles(pickle_dir)
    img_coord_dict = get_SMPL_vertices_in_img_coords(pickle_files)
    return img_coord_dict[image]


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


def main():
    config = load_config()
    params_file = config["stereo_params_k0"]
    pickle_dir = config["pickle_dir"]
    depth_images_dir = config["depth_images_dir"]
    rgb_images_dir = config["rgb_images_dir"]
    random_img = "1665057394063.jpg"  # selecting one random image from the pickle_dir

    rgb_img = cv2.imread(os.path.join(rgb_images_dir, random_img))
    depth_img = get_depth_image(random_img, depth_images_dir)
    depth_bigger = make_bordered_img(depth_img, rgb_img)
    pcloud = get_pointcloud(random_img, pickle_dir)

    with open(params_file, "r") as fh:
        params = json.load(fh)

    rot = np.array(params["RotationOfCamera2"])
    trans = np.array(params["TranslationOfCamera2"]).reshape(1, 3)

    depthX, depthY, depthZ = transform_RGBimgcoords_to_depthimgcoords(
        pcloud,
        depth_frame=depth_img,
        depth_params=params["CameraParameters2"],
        color_params=params["CameraParameters1"],
        R=rot,
        T=trans,
    )

    implot = plt.imshow(depth_bigger)
    plt.scatter(x=depthX, y=depthY, c="cornflowerblue", s=0.5)
    plt.show()


if __name__ == "__main__":
    main()
