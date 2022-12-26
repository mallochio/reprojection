import pandas as pd
import numpy as np
import argparse
import os
import cv2
from tqdm import tqdm


def get_mesh_dict(checkpath):
    # Get the pickle files from the mocap output directory
    pickle_dict = {}
    for i in os.listdir(checkpath):
        if i.endswith(".pkl"):
            df = pd.read_pickle(os.path.join(checkpath, i))
            # An if condition to check if the pickle has a person in it (this was seen during tests)
            if df["pred_output_list"][0] is not None:
                pickle_dict[i] = df
    return pickle_dict


def check_for_person(obj, threshold, image_width=1280, image_height=720):
    # Check if a person is in the image by using frankmocap meshes by checking
    # if the image coordinates of the person are within the image boundaries by above a threshold
    person_image_coordinates = obj["pred_output_list"][0]["pred_vertices_img"]
    
    # Convert list of coordinates to a NumPy array for easier indexing and boolean indexing
    coordinates = np.array(person_image_coordinates)
    
    # Check if the mesh vertices are within the image boundaries
    x_in_bounds = np.logical_and(coordinates[:, 0] >= 0, coordinates[:, 0] < image_width)
    y_in_bounds = np.logical_and(coordinates[:, 1] >= 0, coordinates[:, 1] < image_height)
    
    # Check if the percent of vertices within the image boundaries is above a threshold
    return np.mean(np.logical_and(x_in_bounds, y_in_bounds)) > threshold


def main(path, threshold, output_path):
    # Check if the person is within the image boundaries by above a threshold
    mesh_dict = get_mesh_dict(path)
    person_in_image = []
    print(f"[*] Checking for person in {len(mesh_dict)} images")
    for key, value in tqdm(mesh_dict.items()):
        if check_for_person(value, 0.97, image_width=1280, image_height=720):
            person_in_image.append(key)
    # (Over)Write the list of images with a person in it to a file
    print("=================================================================")
    if not output_path.endswith(".txt"):
        output_path = os.path.join(output_path, "person_detected.txt")
    print(f"[*] Writing list to {output_path}")
    with open(output_path, "w") as f:
        for item in tqdm(person_in_image):
            image_name = item.split("_")[0] + ".jpg"
            image_name = f'/home/sid/Projects/OmniScience/dataset/2022-10-06/bedroom/sid/round1/capture1/rgb/{image_name}'
            f.write(f"{image_name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpath",
        help="Path to the frankmocap output directory",
        type=str,
        default="/home/sid/Projects/OmniScience/dataset/2022-10-06/bedroom/sid/round1/capture1/mocap_output/mocap",
    )
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument(
        "--output_path",
        help="Path to write the list of images with a person in it",
        type=str,
        default="/home/sid/Projects/OmniScience/dataset/2022-10-06/bedroom/sid/round1/capture1/mocap_output",
    )
    args = parser.parse_args()
    main(args.checkpath, args.threshold, args.output_path)
