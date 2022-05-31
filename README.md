# Top-view reprojection project

This project aims at reprojecting lateral view camera (Kinect) pointclouds into the image space
of a top-view camera with a fisheye lens.

## Components

* ```calibration/``` scripts to learn the camera rotations and translations w.r.t. others
* ```capture/``` binaries to be able to capture Kinect and fisheye camera streams at good rates.
* ```sync/``` scripts for synchronization of all different video streams from each camera.
* ```reprojection/``` is used after all previous steps, to show reprojected point clouds.

## Usage

See the [Documentation](docs/README.md) for further information on how to use these.
