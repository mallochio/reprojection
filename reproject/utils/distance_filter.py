from utils import load_matlab_calibration as lomat
import matplotlib.pyplot as plt
from glob import glob
import cv2
import numpy as np
from math import sqrt


def euclidean(pt1, pt2):
    assert(len(pt1) == len(pt2))
    d = 0.
    for i in range(len(pt1)):
        d += (pt1[i] - pt2[i])**2
    return sqrt(d)


def proximity_filter(depth, k_params, **kwargs):
    depth_Fx, depth_Fy, depth_Cx, depth_Cy = k_params
    h, w = depth.shape[:2]
    z = depth / 1000.
    x = np.ones((h, w))
    x[0:h, :] *= np.arange(w)
    y = np.ones((h, w))
    y[:, 0:w] *= np.arange(h).reshape(h, 1)

    # pc stands for point cloud.
    pc = np.zeros((h, w, 3))
    pc[:, :, 0] = z * (x - depth_Cx) / depth_Fx
    pc[:, :, 1] = z * (y - depth_Cy) / depth_Fy
    pc[:, :, 2] = z

    mask = np.zeros((h, w))
    mask[pc[:, :, 0].nonzero()] = 255
    Is, Js = mask.nonzero()

    valid = np.zeros((h, w))
    for c in range(len(Is)):
        i, j = Is[c], Js[c]
        pt = pc[i, j, 0], pc[i, j, 1], pc[i, j, 2]
        ds = np.zeros(9)
        idx = 0
        for k in range(i - 1, i + 1):
            for l in range(j - 1, j + 1):
                pt2 = pc[k, l, 0], pc[k, l, 1], pc[k, l, 2]
                ds[idx] = euclidean(pt, pt2)
                idx += 1
        if np.mean(ds) <= 0.02:
            valid[i, j] = 255

    if 'get_pointcloud' in kwargs and kwargs['get_pointcloud'] is True:
        return valid, pc, len(Is)
    else:
        return valid


def self_test():
    files = sorted(glob('/media/pau/Data/Pictures/discon_experiment/*'))
    filename = files[len(files)//2]
    depth16 = cv2.imread(filename, cv2.CV_16UC1)
    depth = np.float32(depth16)

    plt.imshow(np.uint8(depth/4500.*255.))

    mats = lomat.get_mono_calibration_matrices('/media/pau/Data/Pictures/data_22jul/k1Params.json')

    valid, pc, num_pts = proximity_filter(depth, mats['k_params'], get_pointcloud=True)
    vi, vj = valid.nonzero()

    print('All: %d' % num_pts)
    print('Valid: %d' % len(vi))

    plt.figure()
    plt.imshow(valid)
    plt.title('valid')

    plt.figure()
    plt.scatter(pc[:,:,2], -pc[:,:,1], c=pc[:,:,0])
    plt.scatter(pc[vi,vj,2], -pc[vi,vj,1], marker=',', c = 'red')
    plt.title('ZY')

    plt.figure()
    plt.scatter(pc[:,:,0], pc[:,:,2], c=pc[:,:,1])
    plt.scatter(pc[vi,vj,0], pc[vi,vj,2], c= 'red')
    plt.title('XZ')
    plt.show()


if __name__ == '__main__':
    self_test()
