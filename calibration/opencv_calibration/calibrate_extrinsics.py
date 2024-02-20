from os.path import abspath
import sys

sys.path.append(abspath("../.."))
sys.path.append(abspath(".."))

from pathlib import Path
from tqdm import tqdm

import subprocess
# from config import load_config as conf
# config = conf.load_config()

script_path = "/home/sid/Projects/OmniScience/code/reprojection/calibration/opencv_calibration/get_cam_pose.py"
intrinsics_path = "/home/sid/Projects/OmniScience/code/reprojection/calibration/intrinsics"
base_dir = Path("/home/sid/Projects/OmniScience/mount-NAS/kinect-omni-ego/2024-01-12/at-unis/lab/calib/extrinsics")

# base_dir = Path("/openpose/data/dataset/session-recordings/a10/calib")
# script_path = "/openpose/data/code/reprojection/calibration/opencv_calibration/get_cam_pose.py"

if __name__ == "__main__":
    
    subdir = list(base_dir.glob('**'))
    # Get all subdirectories named "rgb", "ir", and "omni" from the base directory
    calib_rgb_dirs = [x for x in subdir if (x.name == "rgb" and "/calib/" in str(x))]
    calib_ir_dirs = [x for x in subdir if x.name == "ir" and "/calib/" in str(x)]
    calib_omni_dirs = [x for x in subdir if x.name == "omni" and "/calib/" in str(x)]
    # calib_dirs = calib_ir_dirs + calib_rgb_dirs + calib_omni_dirs

    # Let's just do the rgb and omni for now
    calib_dirs = calib_rgb_dirs + calib_omni_dirs

    print(calib_dirs)
    for path in tqdm(calib_dirs):
        prefixes = ["k0", "k1", "k2"]

        ix = [ix for ix in prefixes if ix in str(path)][0]
        dirtype = path.name
        if not ix or dirtype not in ["rgb", "ir", "omni"]:
            print(path)
            raise ValueError("No prefix found in path")

        else:
            # run get_cam_pose on the first image in each directory using the corresponding camera matrix
            img_path = path.glob("*.jpg").__next__()
            if dirtype == "omni":
                camera_matrix_file = "omni_calib.pkl"
            else:
                camera_matrix_file = f"{ix}_rgb_calib.pkl"

            camera_matrix_path = Path(intrinsics_path) / camera_matrix_file      
            command = f"python {script_path} {img_path} {camera_matrix_path} --dst {path.parent} --prefix {ix}_{dirtype}"
            if dirtype == "omni":
                command += " --fisheye"
            print(command)
            
            subprocess.run(command.split(" "))  
