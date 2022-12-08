import os
import shutil
from python_on_whales import docker
from PIL import Image

root = "/home/sid/Projects/OmniScience/dataset"


def run_docker_commands(rgb_dir):
    docker.run(
        image="detectron2:v0",
        command=["/bin/bash", "-c", "python run_on_file.py"],
        volumes=[(f"{rgb_dir}", "/data")],
        gpus="all",
        tty=True,
    )


def clean_up(directory):
    for i in os.listdir(directory):
        try:
            img_path = os.path.join(directory, i)
            Image.open(img_path).load()
        except OSError as e:
            dst = os.path.join(os.path.dirname(directory), "brokenfiles")
            if not os.path.exists(dst):
                os.mkdir(dst)
            shutil.move(img_path, dst)


def main():
    directories = [x[0] for x in os.walk(root)]
    for i in directories:
        if os.path.basename(i) == "rgb" and "/calib/" not in i:
            if os.path.exists(
                os.path.join(os.path.dirname(i), "rgb_dp2_iuv")
            ) and os.path.exists(os.path.join(os.path.dirname(i), "rgb_dp2_mask")):
                continue
            else:
                print(f"Cleaning up {i}")
                clean_up(directory=i)
                run_docker_commands(rgb_dir=os.path.dirname(i))


if __name__ == "__main__":
    main()
