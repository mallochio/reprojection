#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   Untitled-1
@Time    :   2024/01/22 19:33:21
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Script to intelligently select a subset of images for calibration (Incomplete)
'''

# TODO - This code is incomplete

import cv2
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
from sklearn.cluster import KMeans

def variance_of_laplacian(image):
    """Apply the Laplacian operator and return the focus measure, which is the
    variance of the Laplacian -- the sharper the image, the higher the variance."""
    return cv2.Laplacian(image, cv2.CV_64F).var()

def select_sharp_images(directory_path, threshold=200.0):
    """Selects images that are above the given sharpness threshold."""
    sharp_images = []

    # List all image files in the provided directory path
    image_files = [f for f in Path(directory_path).glob('**/*') if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']]

    for file_path in image_files:
        # Read the image from disk
        image = cv2.imread(str(file_path))
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute the variance of the Laplacian
        focus_measure = variance_of_laplacian(gray)
        
        if focus_measure > threshold:
            sharp_images.append(file_path)
    return sharp_images

def extract_features(corners):
    """Extract features from the corners for clustering."""
    # Flatten the list of corners and use as features
    return corners.reshape(-1)

def detect_corners(image, board_size):
    """Detect checkerboard corners in the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)
    if ret:
        return corners
    else:
        return None


def select_variety_images(sharp_images, board_size, num_clusters=5):
    """Select a variety of images based on checkerboard positions and orientations."""
    features = []
    print("Obtaining Features...")
    print("Number of sharp_images: ", len(sharp_images))
    for img_path in tqdm(sharp_images):
        image = cv2.imread(str(img_path))
        corners = detect_corners(image, board_size)
        if corners is not None:
            feature_vector = extract_features(corners)
            features.append(feature_vector)

    features = np.array(features)

    # Use KMeans clustering to cluster the feature vectors
    if len(features) > num_clusters:
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(features)
        labels = kmeans.labels_

        selected_images = []
        print("Selecting images...")
        for i in tqdm(range(num_clusters)):
            # Select the first image from each cluster
            selected_images.append(sharp_images[np.where(labels == i)[0][0]])
    else:
        # Not enough images for the number of clusters desired
        selected_images = sharp_images

    return selected_images

# Example usage:
if __name__ == "__main__":
    images_path = "/home/sid/Projects/OmniScience/dataset/session-recordings/2024-01-12/at-unis/lab/calib/intrinsics/capture0/rgb/"  # Replace with your images directory path
    board_size = (9, 6)  # Define the size of your checkerboard (number of inner corners per chessboard row and column)

    # First, select sharp images
    sharp_images = select_sharp_images(images_path)

    # Then, select a variety of images from the sharp ones
    varied_images = select_variety_images(sharp_images, board_size, num_clusters=10)
    
    for img_path in varied_images:
        print(f"Selected varied image: {img_path}")
