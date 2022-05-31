import cv2
from glob import glob
import numpy as np

images = glob('capture/depth/*.png')

for image in images:
    img = cv2.imread(image, cv2.CV_16UC1)
    depth_vis = np.uint8(img/4500. * 255.)
    cv2.imshow('depth', depth_vis)
    cv2.waitKey(5)