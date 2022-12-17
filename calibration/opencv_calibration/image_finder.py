#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

import argparse
import cv2 as cv

"""
Simple (stolen) tool to click on an image and get img coordinates.
"""

# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):

    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        print(x, " ", y)

        # displaying the coordinates
        # on the image window
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img, str(x) + "," + str(y), (x, y), font, 1, (255, 0, 0), 2)
        cv.imshow("image", img)

    # checking for right mouse clicks
    if event == cv.EVENT_RBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        print(x, " ", y)

        # displaying the coordinates
        # on the image window
        font = cv.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv.putText(
            img, str(b) + "," + str(g) + "," + str(r), (x, y), font, 1, (255, 255, 0), 2
        )
        cv.imshow("image", img)


# driver function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str)
    args = parser.parse_args()
    img = cv.imread(args.image, 1)
    cv.imshow("image", img)
    cv.setMouseCallback("image", click_event)
    key = ''
    while key != ord('q'):
        key = cv.waitKey(1000)
    cv.destroyAllWindows()
