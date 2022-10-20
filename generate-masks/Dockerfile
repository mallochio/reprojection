# FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
# use an older system (18.04) to avoid opencv incompatibility (issue#3524)

ENV DEBIAN_FRONTEND noninteractive

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo ninja-build
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/pip/3.6/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --user cmake onnx   # cmake from apt-get is too old
# RUN pip install --user torch==1.10 torchvision==0.11.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN pip install --user -e detectron2_repo
RUN pip install --user opencv-python av scipy

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
WORKDIR /home/appuser/detectron2_repo

COPY run_on_file.py /home/appuser/detectron2_repo/run_on_file.py
# COPY densepose_rcnn_R_50_FPN_s1x.pkl  /home/appuser/detectron2_repo/densepose_rcnn_R_50_FPN_s1x.pkl
COPY model_final_b1e525.pkl /home/appuser/detectron2_repo/model/model_final_b1e525.pkl

COPY densepose_rcnn_R_50_FPN_DL_WC1_s1x.yaml /home/appuser/detectron2_repo/model/densepose_rcnn_R_50_FPN_DL_WC1_s1x.yaml
COPY Base-DensePose-RCNN-FPN.yaml /home/appuser/detectron2_repo/model/Base-DensePose-RCNN-FPN.yaml
COPY Base-RCNN-FPN.yaml /home/appuser/detectron2_repo/model/Base-RCNN-FPN.yaml

# COPY densepose_rcnn_R_50_FPN_s1x.yaml /home/appuser/detectron2_repo/densepose_rcnn_R_50_FPN_s1x.yaml

# run detectron2 under user "appuser":
# wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
# python3 demo/demo.py  \
	#--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
	#--input input.jpg --output outputs/ \
	#--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
