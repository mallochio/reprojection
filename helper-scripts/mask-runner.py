import os
import shutil
from python_on_whales import docker
from PIL import Image

# root = "/home/sid/Projects/OmniScience/dataset"
root = '/home/sid/Projects/OmniScience/dataset/2022-10-06/bedroom/sid/round1/capture2'

def run_docker_commands(rgb_dir):
    docker.run(
        image="detectron2:v0",
        command=["/bin/bash", "-c", "python run_on_file.py"],
        volumes=[(f"{rgb_dir}", "/data")],
        gpus="all",
        tty=True,
    )

def check_for_associated_files(brokenfiles_directory):
    # Patch function to check withing brokenfiles if there are its counterparts still in use
    # Hope we don't have to use this more than once
    all_dirs = ["depth", "ir", "rgb"]# "mocap_output/mocap"]
    for dirs in os.listdir(brokenfiles_directory):
        for i in os.listdir(os.path.join(brokenfiles_directory, dirs)):
            filepaths = []
            for d in all_dirs:
                checkdir = f'{root}/{d}/{i}'
                if 'rgb' in checkdir or 'ir' in checkdir:
                    file, ext = os.path.splitext(checkdir)
                    file_corrected = f'{file}.jpg'
                    filepaths.append(os.path.join(brokenfiles_directory, file_corrected))
                elif 'depth' in checkdir:
                    file, ext = os.path.splitext(checkdir)
                    file_corrected = f'{file}.png'
                    filepaths.append(os.path.join(brokenfiles_directory, file_corrected))
                else:
                    raise ValueError("Unknown directory")
            for checkfile in filepaths:
                if os.path.exists(checkfile):
                    filetype = checkfile.split('/')[-2]
                    dst = f'{brokenfiles_directory}/{filetype}/{os.path.basename(checkfile)}'
                    shutil.move(checkfile, dst)
    
    checkfiles_mocap = os.listdir(f'{brokenfiles_directory}/rgb')
    filenames = [i.split('.')[0] for i in checkfiles_mocap]
    filenames = [f'{i}_prediction_result.pkl' for i in filenames]

    mocap_directory = f'{root}/mocap_output/mocap'
    for i in os.listdir(mocap_directory):
        if i in filenames:
            dst = f'{brokenfiles_directory}/mocap/{i}'
            shutil.move(os.path.join(mocap_directory, i), dst)
        
            

def clean_up_broken_files(directory):
    for i in os.listdir(directory):
        try:
            img_path = os.path.join(directory, i)
            Image.open(img_path).load()
        except OSError as e:
            dst = os.path.join(os.path.dirname(directory), "brokenfiles", os.path.basename(directory))
            if not os.path.exists(dst):
                os.mkdir(dst)
            shutil.move(img_path, dst)


def main():
    directories = [x[0] for x in os.walk(root)]
    for i in directories:
        if (os.path.basename(i) == "depth" or os.path.basename(i) == "ir") and "/calib/" not in i:
            # if os.path.exists(
            #     os.path.join(os.path.dirname(i), "rgb_dp2_iuv")
            # ) and os.path.exists(os.path.join(os.path.dirname(i), "rgb_dp2_mask")):
            #     continue
            # else:
            print(f"Cleaning up {i}")
            clean_up_broken_files(directory=i)
            # run_docker_commands(rgb_dir=os.path.dirname(i))


if __name__ == "__main__":
    check_for_associated_files(brokenfiles_directory='/home/sid/Projects/OmniScience/dataset/2022-10-06/bedroom/sid/round1/capture2/brokenfiles')
    # main()
