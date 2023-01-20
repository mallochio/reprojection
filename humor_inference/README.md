# HuMoR inference and top-view reprojection

We're running the [HuMoR](https://github.com/davrempe/humor) method for test-time optimization of a SMPL model onto each of the RGB
views of a scene, given a 30 FPS video and the RGB intrinsics. The resulting file contains a
sequence of SMPL parameters that are in *camera frame*: no need to use the depth values, just
project it to the top!


## HuMoR setup

HuMoR requires *openpose* to function. The simplest way to set it up is probably to use a docker
image, as explained in [this
readme](https://github.com/gormonn/openpose-docker/blob/master/README.md).

Install this docker image and follow these steps:

1. Run the docker image while binding your home directory with the humorproject: `docker run
   --user=<user> --bindhome <humor_dir>` (might be different as I use udocker).
2. Install all HuMoR dependencies.
3. Follow HuMoR's
   [instructions](https://github.com/davrempe/humor#fitting-to-rgb-videos-test-time-optimization).
