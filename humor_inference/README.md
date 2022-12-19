# HuMoR inference and top-view reprojection

We're running the [HuMoR](https://github.com/davrempe/humor) method for test-time optimization of a SMPL model onto each of the RGB
views of a scene, given a 30 FPS video and the RGB intrinsics. The resulting file contains a
sequence of SMPL parameters that are in *camera frame*: no need to use the depth values, just
project it to the top!
