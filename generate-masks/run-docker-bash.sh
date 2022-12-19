#!/bin/bash
# docker run --gpus all -it detectron2:v0 --mount src=/home/pau,target=/data,type=bind /bin/bash
echo "Mounting $1 on /data ..."
docker run --gpus all -v $1:/data detectron2:v0 /bin/bash -c "python run_on_file.py"
