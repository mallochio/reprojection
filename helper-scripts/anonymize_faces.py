#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   anonymize_faces.py
@Time    :   2023/01/17 11:47:37
@Author  :   Siddharth Ravi
@Version :   1.0
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Function to anonymize faces in images using mediapipe (Incomplete)
'''

import mediapipe as mp
import cv2
import os
import argparse
import numpy as np
from mediapipe.python.solutions import face_mesh


def anonymize_face_mediapipe(image: np.ndarray, results: face_mesh.FaceMesh, factor: float = 3.0) -> np.ndarray:
    '''
    Function to blur faces in an image using mediapipe
    '''
    image_hight, image_width, _ = image.shape
    w = int(image_width / factor)
    h = int(image_hight / factor)
    print(results.multi_face_landmarks)
    for face_landmarks in results.multi_face_landmarks:
        for facial_feature in face_landmarks.landmark:
            # Get the coordinates of the facial feature
            x = int(facial_feature.x * image_width)
            y = int(facial_feature.y * image_hight)
            # Get the box around the facial feature
            x1 = x - w // 2
            y1 = y - h // 2
            x2 = x + w // 2
            y2 = y + h // 2
            # Check if the box is within the image
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            if x2 > image_width:
                x2 = image_width
            if y2 > image_hight:
                y2 = image_hight
            # Get the ROI and blur it
            roi = image[y1:y2, x1:x2]
            roi = cv2.GaussianBlur(roi, (23, 23), 30)
            # Place the blurred ROI in the image
            image[y1:y2, x1:x2] = roi
    return image

def process_image(args: argparse.Namespace, img: np.ndarray, out_dir: str = None) -> None:
    """
    Processes an image and anonymizes all faces in it.
    """
    # Initialize mediapipe
    mp_face_detection = mp.solutions.face_detection
    # Anonymize image
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        print(results.detection)
        img = anonymize_face_mediapipe(img, results, factor=args.factor)
    if args.show:
        cv2.imshow('Anonymized', img)
        cv2.waitKey(0)
    cv2.imwrite(out_dir, img)


def main(args: argparse.Namespace) -> None:
    # Check if output directory exists
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    # Check if input is a directory or a file
    if os.path.isdir(args.input):
        # Loop through images in input directory
        for image in os.listdir(args.input):
            img = cv2.imread(os.path.join(args.input, image))
            out_dir = os.path.join(args.output, image)
            if img is not None:
                process_image(args, img, out_dir)   
    else:
        img = cv2.imread(args.input)
        out_dir = os.path.join(args.output, os.path.basename(args.input))
        out_dir = args.output
        if img is not None:
            process_image(args, img, out_dir)
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anonymize faces in images')

    parser.add_argument('--input', type=str, default='/home/sid/test/input', help='Path to input image or directory of images')
    parser.add_argument('--output', type=str, default='/home/sid/test/output', help='Path to output directory to dump anonymized images')
    parser.add_argument('--factor', type=float, default=3.0, help='Factor to use for pixelation. Higher values lead to more pixelation')
    parser.add_argument('--show', action='store_true', default=False, help='Show anonymized image')

    args = parser.parse_args()
    main(args)
     