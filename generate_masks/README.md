# Dockerfile for mask generation via Densepose

## Instructions

Download `model_final_b1e525.pkl` and place it in the same location as the rest of the scripts
(i.e. the `generate-masks` folder).

## Build the Docker container

Simply run `./build-docker.sh`

## Run the container bash

Simply run `./run-docker-bash.sh <rgb_data_dir>`. The directory passed as the first (only) argument
will be mapped into the docker machine in `/data`, so the script inside the machine will always find
the files it needs to process in that folder.

## Obtain masks

From the docker machine bash, go to the `detectron2_repo` folder and run `run_on_file.py`.
