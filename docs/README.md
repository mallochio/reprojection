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

## OLD: Available code and scripts

To capture images for calibration and for recording sequences of video with reprojection, several scripts (in Python), as well as software packages (C++) are available. These scripts and packages are described next. Some obsolete scripts are also shown here, for the sake of completeness.

- **calib-charuco (obsolete):** this project contains multiple scripts that were useful to capture images using different chessboards and variations (google “charuco” for more information) . This was the first project that was created, and contains many obsolete scripts that used “ChArUco” boards for calibration (these combine chessboard and AR markers). It also includes scripts using Kinect v1 and v2 libraries to obtain depth, IR, and color images.
  - **Useful scripts:** there remain, however, some useful 	scripts in this project: the “utils” 	and “scratch” 	subfolders contain the following scripts:
    - read_all_streams.py: script that loads the 		“live” video from a Kinect and the omnidirectional camera, and 		that, upon pressing [Spacebar], saves images for calibration.
    - kinect-ir-adapt.py: already mentioned.
- **Ceilcam-demo:** this is the “celing cam” reprojection “demonstration” project. It takes kinect point clouds and reprojects them onto the omnidirectional camera, using the calibration of the scene that has been previously computed in Matlab from several images. It also contains scripts to prepare images so that they can be used in Matlab for calibration.
  - Scripts for synchronisation of videos 	(for demo) or for calibration:
    - **sync_videos.py:** this script shows a series of sequences captured with the “kinect-capture” C++ software and allows to 		syncronise them temporally (i.e. frame-to-frame). 		It generates a “shots0.txt” file, 		containing data about the times that the [Spacebar] has been 		presed, meaining that the user understands that the videos are 		synced in these positions.
    - **prepare_for_matlab.py:** 		prepares directories of images for the 		Matlab calibration app to load them more easily.
  - Video output generation (i.e. for the demo):
    - **generate_demo.py** 		(obsolete): 		this was used when the calibration 		relied on RGB images from the Kinect, instead of the IR images.
    - **generate_demo_norgb.py (****i.e. 		new version):** 		generates a 		demo (video) 		in which the reprojection of kinect 		point clouds is performed using images calibrated using pairs of 		Kinect IR images and images from the omnidirectional camera on the 		ceiling. Furthermore, it uses the Scaramuzza calibration for the 		omnidirectional camera.
- **kinect-capture:** A project in C++ to capture videos without delay (it uses a pool of separate threads for incoming frames). Once compiled, it consists of two executable files (apps): “**Kinect-capture**” and “**Omnidir-capture**”. The installation instructions follow.