# The “Calibration” shared directory

This shared directory (on Dropbox), contains the following subfolders:

- **Capturas:** (i.e. Captures) contains sequences of images for calibration (taken beforehand).
- **Código:** (i.e. Code) contains scripts and code that will be described later.
  - **calibration_suite**
    - **kinect-capture:** this is a threaded C++ program that captures images without delay (as compared to a Python only version).
    - **ceilcam-demo:** this utility is used to synchronise the video sources and create the demonstration videos.
    - **kinect-ir-adapt.py**: this script “adapts to Matlab” images taken from the IR sensor of a kinect (that is, it resizes them to the size expected by Matlab, which is the same as the omnidirecional camera images).
    - **calib-charuco:** obsolete code of previous experiments (i.e. calibration before we decided to use Matlab’s own).
- **Imágenes pregrabadas Matlab:** (i.e. pre-recorded images for Matlab): images used for intrinsic camera calibration so that only a “common” image with the chessboard on the ground is needed to determine the rotation of each camera with respect to the observed object (the chessboard).
- **Miscelánea:** (i.e. Miscelaneous) Other scripts and images previously used (can mostly be ignored).
- **Papers:** Papers and other publications relevant to calibration, reprojection, human joint localisation (skeletons), etc.
- **Vídeos generados:** (generated videos) generated demo videos showing reprojection of point clouds over the omnidirectional camera footage (this is the output of the ceilcam-demo/generate-demo-norgb.py script).