import numpy as np
import cv2

min_area = 1000
min_diff_mm = 50

k3 = np.asarray([[0, 1, 0],
                 [1, 1, 1],
                 [0, 1, 0]], dtype=np.uint8)

k5 = np.asarray([[0, 0, 1, 0, 0],
                 [0, 0, 1, 0, 0],
                 [1, 1, 1, 1, 1],
                 [0, 0, 1, 0, 0],
                 [0, 0, 1, 0, 0]], dtype=np.uint8)

k7 = np.asarray([[0, 0, 1, 1, 1, 0, 0],
                 [0, 1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1],
                 [0, 1, 1, 1, 1, 1, 0],
                 [0, 0, 1, 1, 1, 0, 0]], dtype=np.uint8)

k7b = np.ones((7, 7), np.uint8)
k5b = np.ones((5, 5), np.uint8)
k3b = np.ones((3, 3), np.uint8)


def clean_up_mask_omni(mask):
    mask = cv2.erode(mask, kernel=k3)
    mask = cv2.erode(mask, kernel=k3)
    mask = cv2.dilate(mask, kernel=k3)
    mask = cv2.dilate(mask, kernel=k3)

    return mask

def paco(mask):
    mask = cv2.dilate(mask, kernel=k3b)
    mask = cv2.erode(mask, kernel=k3b)

    return mask

def connected_components(mask_src):
    ret, labels = cv2.connectedComponents(mask_src)
    for label in range(1, ret):
        mask = np.array(labels, dtype=np.uint8)
        mask[labels == label] = 255
        cv2.imshow('component', mask)
        cv2.waitKey(0)

        # contours = cv2.findContours(mask_src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        # for cnt in contours:
        #    (x, y, w, h) = cv2.boundingRect(cnt)


def clean_up_mask_kinect(mask, fatter, cc):
    diff = np.uint8(mask * 255)
    diff = cv2.erode(diff, kernel=k3)
    diff = cv2.erode(diff, kernel=k3)
    diff = cv2.dilate(diff, kernel=k3)
    diff = cv2.dilate(diff, kernel=k3)

    if fatter:
        diff = cv2.dilate(diff, kernel=k3)
        diff = cv2.dilate(diff, kernel=k3)
        diff = cv2.dilate(diff, kernel=k3)

    if cc:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(diff, 4, cv2.CV_32S)
        final = np.zeros(diff.shape, dtype=np.uint8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > min_area:
                final[labels == i] = 255
    else:
        final = diff

    return final


def generate_mask(depth_now, depth_before, fatter=False, cc=False):
    """
    Generates a mask from two depth maps using a distance difference in millimetres.
    :param depth_now: the depth map of the most recent frame
    :param depth_before: the depth map from the previous or reference frame
    :param distance_mm: threshold distance in millimetres
    :return:
    """

    # cond1 = np.abs(depth_now - depth_before) > distance_mm
    # bg = np.maximum(depth_now, depth_before)
    cond1 = np.abs(depth_before - depth_now) > min_diff_mm  # distance_mm
    mask = cond1 # np.logical_and(cond1, depth_before != 0)  # mm
    mask = np.uint8(mask * 255.)

    mask = clean_up_mask_kinect(mask, fatter, cc)

    return mask
