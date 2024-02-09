#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   calibrate_interactively.py
@Time    :   2024/01/22 18:29:08
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Code to interactively calibrate the camera using a checkerboard (Incomplete)
'''

# TODO - This code is incomplete

import cv2
import numpy as np

# Define the checkerboard dimensions
checkerboard_size = (6, 9)  # for 6x9 checkerboard

# Termination criteria for corner sub-pixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all images
objpoints = []  # 3d points in real-world space
imgpoints = []  # 2d points in image plane

# Coverage dictionary to keep track of different checkerboard orientations
orientation_coverage = {
    'left': 0, 'right': 0, 'top': 0, 'bottom': 0,
    'tilt_forward': 0, 'tilt_backward': 0,
    'rotate_cw': 0, 'rotate_ccw': 0  # clockwise, counter-clockwise
}

# Function to analyze the orientation of the checkerboard
def analyze_orientation(corners, image_shape):
    width, height = image_shape[1], image_shape[0]
    center = np.array([width//2, height//2])
    
    # Calculate the center of the checkerboard
    checkerboard_center = np.mean(corners, axis=0)
    
    # Calculate the direction vectors from the image center to the checkerboard center
    direction_vector = checkerboard_center - center
    
    # Calculate the average distances to the checkerboard edges
    left_dist = np.mean(corners[:,0,0])
    right_dist = width - np.mean(corners[:,0,0])
    top_dist = np.mean(corners[:,0,1])
    bottom_dist = height - np.mean(corners[:,0,1])
    
    # Update the orientation coverage counts
    if direction_vector[0] < 0:  # checkerboard is to the left
        orientation_coverage['left'] += 1
    if direction_vector[0] > 0:  # checkerboard is to the right
        orientation_coverage['right'] += 1
    if direction_vector[1] < 0:  # checkerboard is to the top
        orientation_coverage['top'] += 1
    if direction_vector[1] > 0:  # checkerboard is to the bottom
        orientation_coverage['bottom'] += 1
    
    # Check for tilts and rotations
    if left_dist > right_dist:
        orientation_coverage['tilt_forward'] += 1
    if left_dist < right_dist:
        orientation_coverage['tilt_backward'] += 1
    if top_dist > bottom_dist:
        orientation_coverage['rotate_cw'] += 1
    if top_dist < bottom_dist:
        orientation_coverage['rotate_ccw'] += 1

if __name__ == '__main__':
    # Capture images until 'q' is pressed
    # Open the camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Find the checkerboard corners
            found, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

            # If found, add object points, image points (after refining them)
            if found:
                # Refine corners
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Analyze orientation
                analyze_orientation(corners2.reshape(-1, 2), frame.shape)

                # Provide dynamic feedback based on the orientation coverage
                feedback = ""
                min_captures_per_orientation = 2  # minimum number of captures for each orientation
                for orientation, count in orientation_coverage.items():
                    if count < min_captures_per_orientation:
                        # Note: Customize these messages to be more user-friendly in your application
                        feedback += f"Please adjust the checkerboard: {orientation.replace('_', ' ')}. "

                if feedback:
                    print(feedback)
                else:
                    print("Good variety of orientations captured.")

                # Draw and display the corners
                frame = cv2.drawChessboardCorners(frame, checkerboard_size, corners2, found)

            # Display the resulting frame
            cv2.imshow('Camera Calibration', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    # Here you would use the objpoints and imgpoints to calibrate the camera
    # using cv2.calibrateCamera function and save the calibration matrix.

    if len(objpoints) > 0 and len(imgpoints) > 0:
        # Calibrate the camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        # Save the calibration matrix
        np.savez('calibration_matrix.npz', mtx=mtx, dist=dist)
        print("Calibration matrix saved to 'calibration_matrix.npz'.")
    else:
        print("Calibration failed. No checkerboard images were captured.")
