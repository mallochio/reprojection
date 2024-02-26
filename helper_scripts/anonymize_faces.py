#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   anonymize_faces.py
@Time    :   2023/01/17 11:47:37
@Author  :   Siddharth Ravi
@Version :   1.1
@Contact :   siddharth.ravi@ua.es
@License :   (C)Copyright 2022-2023, Siddharth Ravi, Distributed under terms of the MIT license
@Desc    :   Function to anonymize faces in images using mediapipe
"""

import os
import cv2
import argparse
import mediapipe as mp
from alive_progress import alive_bar


def get_bounding_box(detection, image_width, image_height):
    """
    Function to get the bounding box of the face
    """
    bboxc = detection.location_data.relative_bounding_box
    bbox = (
        int(bboxc.xmin * image_width),
        int(bboxc.ymin * image_height),
        int(bboxc.width * image_width),
        int(bboxc.height * image_height)
    )
    bbox = tuple(max(0, i) for i in bbox)
    return None if len(bbox) != 4 else bbox


def detect_and_anonymize_faces(
    image_path,
    output_path,
    blur_factor=5.0,
    pixelate_factor=0.1,
    pixelate=True,
    blur=False,
    draw_bbox=False,
    bar=None,
):
    """
    Function to anonymize faces in images using mediapipe
    """
    # Ensure that either blur or pixelate is enabled
    if not blur and not pixelate:
        raise ValueError("Either blur or pixelate must be enabled")

    mp_face_detection = mp.solutions.face_detection
    image = cv2.imread(image_path)

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.detections:
        bar.text = f"Found {len(results.detections)} face(s) in the image {os.path.basename(image_path)}."
        image_height, image_width, _ = image.shape
        for detection in results.detections:
            bbox = get_bounding_box(detection, image_width, image_height)
            if not bbox:
                continue

            x, y, w, h = bbox
            x1, y1 = x + w, y + h
            face_image = image[y:y1, x:x1]

            if draw_bbox:
                cv2.rectangle(image, (x, y), (x1, y1), (0, 0, 255), 2)

            if blur:
                face_image = cv2.GaussianBlur(face_image, (0, 0), blur_factor)

            if pixelate:
                face_image = cv2.resize(face_image, (0, 0), fx=0.1, fy=0.1)
                face_image = cv2.resize(face_image, (w, h), interpolation=cv2.INTER_NEAREST)
            try:
                image[y:y1, x:x1] = face_image
            except ValueError as e:
                print(image_path, list(bbox), face_image.shape, image.shape)
                raise ValueError("Error in anonymizing the image") from e

        cv2.imwrite(output_path, image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_folder",
        "-i",
        help="Path to the folder containing images",
        default="/home/sid/Projects/OmniScience/dataset/session-recordings/2022-10-07/at-paus/bedroom/sid/capture0/rgb/",
    )
    parser.add_argument(
        "--output_folder",
        "-o",
        help="Path to the folder to save anonymized images",
        default="/home/sid/Projects/OmniScience/dataset/session-recordings/2022-10-07/at-paus/bedroom/sid/capture0/anonymized_images_pixelate/",
    )
    parser.add_argument(
        "--blur",
        help="Flag to enable Gaussian blur",
        action="store_true",
    )
    parser.add_argument(
        "--pixelate",
        help="Flag to enable pixelate",
        action="store_true",
    )
    parser.add_argument(
        "--blur_factor",
        help="Blur factor for Gaussian blur",
        default=5.0,
    )
    parser.add_argument(
        "--pixelate_factor",
        help="Pixelate factor for pixelate",
        default=0.1,
    )
    parser.add_argument(
        "--draw_bbox",
        help="Flag to draw bounding box around the face",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        
    with alive_bar(len(os.listdir(args.image_folder))) as bar:
        for image in os.listdir(args.image_folder):
            image_path = os.path.join(args.image_folder, image)
            output_path = os.path.join(args.output_folder, image)
            detect_and_anonymize_faces(
                image_path,
                output_path,
                args.blur_factor,
                args.pixelate_factor,
                args.pixelate,
                args.blur,
                args.draw_bbox,
                bar,
            )
            bar()


if __name__ == "__main__":
    main()
