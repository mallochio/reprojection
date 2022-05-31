# Introduction

This project is aimed at reprojecting kinect depth point clouds into the image space of a ceiling-mounted omnidirectional (fisheye) camera.

### Steps to successful reprojection:

1) Setup your cameras and computers (see [capture](capture.md)).
2) Create new [calibration](calibration.md) images, with your cameras installed.
   * Mix your "old" and new images to calibrate each camera separately (mono).
     * One image should be "new" and "common" among a pair of cameras.
   * Kinect color-to-infrared calibrations, **are provided**.
   * Use MATLAB to obtain a calibration files.
3) [Capture](capture.md) your recordings.
4) [Synchronize](sync.md) your videos using the provided script.
5) [Reproject](reprojection.md) the lateral (Kinect) point clouds to the top view, using the reprojection script.