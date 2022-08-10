import numpy as np
import cv2

ir_image_size = (424, 512)
rgb_image_size = (720, 1280)
# rgb_image_size = (1080, 1920)


def depth_to_world_pts(y, x, depth_frame, depth_params):
    depth_Fx, depth_Fy, depth_Cx, depth_Cy = depth_params
    world_coordinates = np.zeros((len(x), 4))
    z = depth_frame[y, x] / 1000.
    
    world_coordinates[:, 0] = z * (x - depth_Cx) / depth_Fx
    world_coordinates[:, 1] = z * (y - depth_Cy) / depth_Fy
    world_coordinates[:, 2] = z
    world_coordinates[:, 3] = 1

    valid_coordinates = np.zeros((len(x), 2), dtype=np.int32)
    valid_coordinates[:, 0] = x
    valid_coordinates[:, 1] = y

    return world_coordinates, valid_coordinates


def depth_to_world(mask, depth_frame, depth_params):
    y, x = mask.nonzero()
    return depth_to_world_pts(y, x, depth_frame, depth_params)


def get_depth_pixel_colors(world_coordinates, valid_depth_coordinates, color_frame, color_params, R, T):
    rgb_Fx, rgb_Fy, rgb_Cx, rgb_Cy = color_params
    colorised_depth = np.zeros((ir_image_size[0], ir_image_size[1], 3))
    color_coordinates = np.dot(world_coordinates[:, :3], R.T) + (T / 1000.)
    rgbX = color_coordinates[:, 0] * rgb_Fx / color_coordinates[:, 2] + rgb_Cx
    rgbY = color_coordinates[:, 1] * rgb_Fy / color_coordinates[:, 2] + rgb_Cy
    rgb_h, rgb_w = color_frame.shape[:2]
    rgbY[rgbY < 0] = 0
    rgbX[rgbX < 0] = 0
    rgbY[np.round(rgbY) >= rgb_h] = rgb_h-1
    rgbX[np.round(rgbX) >= rgb_w] = rgb_w-1
    colors = color_frame[np.int32(np.round(rgbY)), np.int32(np.round(rgbX))]
    colorised_depth[valid_depth_coordinates[:, 1], valid_depth_coordinates[:, 0]] = colors

    return colorised_depth, colors


def get_color_depth_values(world_coordinates, valid_depth_coordinates, color_frame, s, color_params, depth_frame, R, T):
    rgb_Fx, rgb_Fy, rgb_Cx, rgb_Cy = color_params

    color_coordinates = np.dot(world_coordinates[:, :3], R.T) + (T / 1000.)
    rgbX = color_coordinates[:, 0] * rgb_Fx / color_coordinates[:, 2] + rgb_Cx
    rgbY = color_coordinates[:, 1] * rgb_Fy / color_coordinates[:, 2] + rgb_Cy
    rgb_h, rgb_w = color_frame.shape[:2]
    rgb_h, rgb_w = np.int32(np.round(rgb_h*s)), np.int32(np.round(rgb_w*s))
    depth_on_color = np.zeros((rgb_h, rgb_w), dtype=np.float32)
    map_to_depth = np.ones((rgb_h, rgb_w, 2), dtype=np.int32) * -1
    rgbX *= s
    rgbY *= s
    rgbY[rgbY < 0] = 0
    rgbX[rgbX < 0] = 0
    rgbY[np.round(rgbY) >= rgb_h] = rgb_h - 1
    rgbX[np.round(rgbX) >= rgb_w] = rgb_w - 1
    rgbX = np.int32(np.round(rgbX))
    rgbY = np.int32(np.round(rgbY))
    for i in range(valid_depth_coordinates.shape[0]):
        depthX, depthY = valid_depth_coordinates[i, :]
        ry, rx = rgbY[i], rgbX[i]
        z = depth_frame[depthY, depthX]
        rz = np.float32(z/4500.*255)
        depth_on_color[ry, rx] = rz
        map_to_depth[ry, rx, 0] = depthX
        map_to_depth[ry, rx, 1] = depthY

    return depth_on_color, map_to_depth
