from os.path import abspath
import sys

sys.path.append(abspath("../.."))
sys.path.append(abspath(".."))

from pathlib import Path
from tqdm import tqdm

import subprocess
from config import load_config as conf
config = conf.load_config()

# script_path = "/home/sid/Projects/OmniScience/code/reprojection/calibration/opencv_calibration/get_cam_pose.py"
script_path = "/openpose/data/code/reprojection/calibration/opencv_calibration/get_cam_pose.py"

if __name__ == "__main__":
    base_dir = Path("/openpose/data/dataset/session-recordings/a10/calib")
    # base_dir = Path("/home/sid/Projects/OmniScience/dataset/session-recordings/test/")
    subdir = list(base_dir.glob('**'))
    # Get all subdirectories named "rgb", "ir", and "omni" from the base directory
    calib_rgb_dirs = [x for x in subdir if (x.name == "rgb" and "/calib/" in str(x))]
    calib_ir_dirs = [x for x in subdir if x.name == "ir" and "/calib/" in str(x)]
    calib_omni_dirs = [x for x in subdir if x.name == "omni" and "/calib/" in str(x)]
    calib_dirs = calib_ir_dirs + calib_rgb_dirs + calib_omni_dirs

    for path in tqdm(calib_dirs):
        prefixes = ["k0", "k1", "k2"]
        ix = [ix for ix in prefixes if ix in str(path)][0]
        dirtype = path.name
        if not ix or dirtype not in ["rgb", "ir", "omni"]:
            print(path)
            raise ValueError("No prefix found in path")
        elif ix == "k2":
            # skip k2 for now, we haven't calibrated it yet
            continue
        else:
            # run get_cam_pose on the first image in each directory using the corresponding camera matrix
            img_path = path.glob("*.jpg").__next__()
            camera_matrix_path = config[f"{dirtype}_params_{ix}"] if dirtype != "omni" else config[f"{dirtype}_params"]            
            command = f"python {script_path} {img_path} {camera_matrix_path} --dst {path.parent} --prefix {ix}_{dirtype}"
            if dirtype == "omni":
                command += " --fisheye"
            
            subprocess.run(command.split(" "))  
