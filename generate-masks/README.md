# Dockerfile for mask generation via Densepose

## Instructions

Clone the git repository for detectron2, create a `docker` directory in it, then copy the contents
of this folder there.

Download `model_final_b1e525.pkl` and place it in the same location as the rest of the scripts
(i.e. the `docker` folder).

## Build the Docker container

Simply run `./build-docker.sh`

## Run the container bash

Simply run `./run-docker-bash.sh <rgb_data_dir>`

## Obtain masks

From the docker machine bash, run `run_on_file.py`.
